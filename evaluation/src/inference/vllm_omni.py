"""Shared vLLM omni judge executor for adapter-compatible models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.common import JUDGE_CATEGORIES_BY_MODALITY
from src.evaluation_protocol import (
    DEFAULT_JOINT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_joint_messages,
)
from src.inference.common import (
    normalize_condition_json,
    normalize_metadata,
    normalize_speech_metadata,
    serialize_prompt_value,
)
from src.inference.media import load_audio_payload, load_rgb_image
from src.judge_parsing import build_judge_json_schema, parse_judge_response
from src.model_adapters.base import ModelAdapter
from src.model_adapters.registry import get_adapter_by_model_name
from src.inference.vllm_runtime import (
    apply_model_chat_controls,
    get_chat_template_kwargs,
    get_or_create_llm,
    resolve_backend,
)


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["joint"]
_PROCESSOR_CACHE: Dict[str, Any] = {}


def _build_sampling_kwargs(
    adapter: Optional[ModelAdapter],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
    attempt_index: int,
    base_seed: Any,
) -> tuple[Dict[str, Any], Optional[int], str]:
    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    attempt_seed = base_seed if isinstance(base_seed, int) else None
    seed_scope = "engine"
    if attempt_index > 0:
        if adapter is not None:
            sampling_kwargs.update(adapter.retry_sampling_overrides)
        if isinstance(base_seed, int):
            attempt_seed = base_seed + attempt_index
            sampling_kwargs["seed"] = attempt_seed
            seed_scope = "retry_request"
    return sampling_kwargs, attempt_seed, seed_scope


def _get_processor(model_name: str) -> Any:
    from transformers import AutoProcessor

    cached = _PROCESSOR_CACHE.get(model_name)
    if cached is not None:
        return cached
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    _PROCESSOR_CACHE[model_name] = processor
    return processor


def _build_vllm_prompt(
    model_name: str,
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    required_template_kwargs = chat_template_kwargs or {}
    controlled_messages = apply_model_chat_controls(model_name, messages)
    try:
        return processor.apply_chat_template(
            controlled_messages,
            tokenize=False,
            add_generation_prompt=True,
            **required_template_kwargs,
        )
    except Exception as exc:
        if required_template_kwargs:
            raise RuntimeError(
                "Failed to apply required model chat template kwargs: "
                f"{required_template_kwargs}"
            ) from exc
        text_parts: List[str] = []
        for message in controlled_messages:
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") == "image":
                    text_parts.append("<image>")
                elif isinstance(item, dict) and item.get("type") == "audio":
                    text_parts.append("<audio>")
        return "\n\n".join(part for part in text_parts if part).strip()


def evaluate_omni_batch(
    model_name: str,
    requests: Sequence[Dict[str, Any]],
    *,
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_JOINT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> List[Dict[str, Any]]:
    if not requests:
        return []

    resolve_backend("vllm")
    from vllm import SamplingParams

    processor = _get_processor(model_name)
    chat_template_kwargs = get_chat_template_kwargs(model_name)
    adapter = get_adapter_by_model_name(model_name)
    prompts = []
    for request in requests:
        messages = build_joint_messages(
            metadata=serialize_prompt_value(
                normalize_metadata(request["metadata"])
            ),
            condition_json=serialize_prompt_value(
                normalize_condition_json(request["condition_json"])
            ),
            current_text=request["current_text"],
            ground_truth_text=request["ground_truth_text"],
            candidate_text=request["candidate_text"],
            ground_truth_speech_metadata=serialize_prompt_value(
                normalize_speech_metadata(
                    request["ground_truth_speech_metadata"]
                )
            ),
            current_image=request["current_image"],
            ground_truth_image=request["ground_truth_image"],
            candidate_image=request["candidate_image"],
            candidate_audio=request["candidate_audio"],
        )
        if adapter is not None and adapter.omni_prompt_adapter == "gemma4":
            from src.model_adapters.gemma4 import adapt_joint_messages

            messages = adapt_joint_messages(messages)
        elif adapter is not None and adapter.omni_prompt_adapter is not None:
            raise ValueError(
                "Unsupported omni prompt adapter: "
                f"{adapter.omni_prompt_adapter}"
            )
        prompts.append(
            {
                "prompt": _build_vllm_prompt(
                    model_name,
                    processor,
                    messages,
                    chat_template_kwargs=chat_template_kwargs,
                ),
                "multi_modal_data": {
                    "image": [
                        load_rgb_image(request["current_image"]),
                        load_rgb_image(request["ground_truth_image"]),
                        load_rgb_image(request["candidate_image"]),
                    ],
                    "audio": [
                        load_audio_payload(request["candidate_audio"])
                    ],
                },
            }
        )

    structured_schema = None
    structured_output_kwargs: Dict[str, Any] = {}
    repetition_detection_kwargs: Dict[str, Any] = {}
    if adapter is not None and adapter.enforce_json_schema:
        structured_schema = build_judge_json_schema(REQUIRED_SCORE_KEYS)
        structured_output_kwargs = dict(adapter.structured_output_kwargs)
    if adapter is not None:
        repetition_detection_kwargs = dict(
            adapter.repetition_detection_kwargs
        )

    llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
    max_generation_attempts = max(
        adapter.max_generation_attempts if adapter is not None else 1,
        1,
    )
    base_seed = (vllm_kwargs or {}).get("seed")
    pending_indices = list(range(len(prompts)))
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    best_ranks = [(-1, -1, -1, 0)] * len(prompts)
    generation_histories: List[List[Dict[str, Any]]] = [
        [] for _ in prompts
    ]

    for attempt_index in range(max_generation_attempts):
        if not pending_indices:
            break

        attempt_kwargs, attempt_seed, seed_scope = (
            _build_sampling_kwargs(
                adapter,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                attempt_index=attempt_index,
                base_seed=base_seed,
            )
        )

        if structured_schema is not None:
            from vllm.sampling_params import StructuredOutputsParams

            attempt_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=structured_schema,
                **structured_output_kwargs,
            )
        if repetition_detection_kwargs:
            from vllm.sampling_params import RepetitionDetectionParams

            attempt_kwargs["repetition_detection"] = RepetitionDetectionParams(
                **repetition_detection_kwargs
            )

        sampling_params = SamplingParams(**attempt_kwargs)
        current_indices = list(pending_indices)
        outputs = llm.generate(
            [prompts[index] for index in current_indices],
            sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(current_indices):
            raise RuntimeError(
                "Joint judge returned fewer outputs than prompts"
            )

        pending_indices = []
        for prompt_index, output in zip(current_indices, outputs):
            completion = output.outputs[0] if output.outputs else None
            response = (
                completion.text.strip() if completion is not None else ""
            )
            result = parse_judge_response(response, REQUIRED_SCORE_KEYS)
            result["omni_eval"] = result["judge_eval"]

            stop_reason = (
                completion.stop_reason if completion is not None else None
            )
            if not isinstance(
                stop_reason,
                (str, int, float, bool, type(None)),
            ):
                stop_reason = str(stop_reason)
            diagnostic = {
                "attempt": attempt_index + 1,
                "seed": attempt_seed,
                "seed_scope": seed_scope,
                "sampling": {
                    "temperature": attempt_kwargs["temperature"],
                    "top_p": attempt_kwargs["top_p"],
                    "top_k": attempt_kwargs.get("top_k", 0),
                    "repetition_penalty": attempt_kwargs[
                        "repetition_penalty"
                    ],
                },
                "finish_reason": (
                    completion.finish_reason
                    if completion is not None
                    else None
                ),
                "stop_reason": stop_reason,
                "generated_token_count": (
                    len(completion.token_ids)
                    if completion is not None
                    else 0
                ),
                "status": result.get("status"),
                "parse_status": result.get("parse_status"),
            }
            generation_histories[prompt_index].append(diagnostic)

            rank = (
                sum(
                    1
                    for category in REQUIRED_SCORE_KEYS
                    if category in response
                ),
                response.count('"score"'),
                response.count('"rationale"'),
                -len(response),
            )
            if result.get("status") == "ok":
                results[prompt_index] = result
                continue
            if rank > best_ranks[prompt_index]:
                best_ranks[prompt_index] = rank
                results[prompt_index] = result
            if attempt_index + 1 < max_generation_attempts:
                pending_indices.append(prompt_index)

    final_results: List[Dict[str, Any]] = []
    for index, result in enumerate(results):
        if result is None:
            result = parse_judge_response("", REQUIRED_SCORE_KEYS)
            result["omni_eval"] = result["judge_eval"]
        result["generation_attempt_count"] = len(
            generation_histories[index]
        )
        result["generation_attempts"] = generation_histories[index]
        final_results.append(result)
    return final_results
