"""Shared vLLM audio judge executor for MOSS, Kimi, and compatible models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoProcessor

from src.common import JUDGE_CATEGORIES_BY_MODALITY
from src.evaluation_protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_speech_prompt_text,
)
from src.inference.common import (
    normalize_condition_json,
    normalize_metadata,
    normalize_speech_metadata,
    serialize_prompt_value,
)
from src.inference.media import load_audio_payload
from src.judge_parsing import parse_judge_response
from src.inference.vllm_runtime import (
    apply_model_chat_controls,
    build_kimi_audio_prompt,
    build_moss_audio_prompt,
    get_chat_template_kwargs,
    get_stop_token_ids,
    get_or_create_llm,
    is_kimi_audio_model,
    is_moss_audio_model,
    resolve_backend,
)


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["speech"]
_PROCESSOR_CACHE: Dict[str, Any] = {}


def _build_prompt_text(
    *,
    metadata: str,
    condition_json: str,
    ground_truth_speech_metadata: str,
) -> str:
    return build_speech_prompt_text(
        metadata=metadata,
        condition_json=condition_json,
        ground_truth_speech_metadata=ground_truth_speech_metadata,
    )


def _build_messages(
    *,
    metadata: Any,
    condition_json: Any,
    ground_truth_speech_metadata: Any,
    candidate_audio_path: str | Path,
) -> List[Dict[str, Any]]:
    prompt = _build_prompt_text(
        metadata=serialize_prompt_value(normalize_metadata(metadata)),
        condition_json=serialize_prompt_value(
            normalize_condition_json(condition_json)
        ),
        ground_truth_speech_metadata=serialize_prompt_value(
            normalize_speech_metadata(ground_truth_speech_metadata)
        ),
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(candidate_audio_path)},
            ],
        }
    ]


def _get_processor(model_name: str) -> Any:
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
) -> str:
    controlled_messages = apply_model_chat_controls(model_name, messages)
    try:
        return processor.apply_chat_template(
            controlled_messages,
            tokenize=False,
            add_generation_prompt=True,
            **get_chat_template_kwargs(model_name),
        )
    except Exception:
        text_parts: List[str] = []
        for message in controlled_messages:
            content = message.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") == "audio":
                    text_parts.append("<audio>")
        return "\n\n".join(part for part in text_parts if part).strip()


def evaluate_audio_batch(
    model_name: str,
    requests: Sequence[Dict[str, Any]],
    *,
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> List[Dict[str, Any]]:
    if not requests:
        return []

    resolve_backend("vllm")
    from vllm import SamplingParams

    kimi_audio = is_kimi_audio_model(model_name)
    moss_audio = is_moss_audio_model(model_name)
    processor = None if kimi_audio or moss_audio else _get_processor(model_name)
    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    stop_token_ids = get_stop_token_ids(model_name)
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids
    sampling_params = SamplingParams(**sampling_kwargs)

    prompts = []
    for request in requests:
        audio_path = request["candidate_audio"]
        prompt_text = _build_prompt_text(
            metadata=serialize_prompt_value(
                normalize_metadata(request["metadata"])
            ),
            condition_json=serialize_prompt_value(
                normalize_condition_json(request["condition_json"])
            ),
            ground_truth_speech_metadata=serialize_prompt_value(
                normalize_speech_metadata(
                    request["ground_truth_speech_metadata"]
                )
            ),
        )
        if kimi_audio:
            prompt = build_kimi_audio_prompt(prompt_text)
            audio_payload: Any = load_audio_payload(audio_path)
        elif moss_audio:
            prompt = build_moss_audio_prompt(prompt_text)
            audio_payload = [load_audio_payload(audio_path)]
        else:
            messages = _build_messages(
                metadata=request["metadata"],
                condition_json=request["condition_json"],
                ground_truth_speech_metadata=request[
                    "ground_truth_speech_metadata"
                ],
                candidate_audio_path=audio_path,
            )
            prompt = _build_vllm_prompt(model_name, processor, messages)
            audio_payload = [load_audio_payload(audio_path)]
        prompts.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": audio_payload},
            }
        )

    llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    results: List[Dict[str, Any]] = []
    for output in outputs:
        completion = output.outputs[0] if output.outputs else None
        response = (getattr(completion, "text", "") or "").strip()
        token_ids = list(getattr(completion, "token_ids", ()) or ())
        result = parse_judge_response(response, REQUIRED_SCORE_KEYS)
        result["generation_output"] = {
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "token_count": len(token_ids),
        }
        if not response:
            result["generation_output"]["token_ids"] = token_ids
        result["speech_eval"] = result["judge_eval"]
        results.append(result)
    return results
