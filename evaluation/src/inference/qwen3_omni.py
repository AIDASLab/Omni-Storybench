"""Qwen3-Omni vLLM executor using qwen_omni_utils preprocessing."""

from __future__ import annotations

from pathlib import Path
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
from src.judge_parsing import parse_judge_response
from src.inference.vllm_runtime import (
    apply_model_chat_controls,
    get_chat_template_kwargs,
    get_or_create_llm,
    resolve_backend,
)


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["joint"]
_PROCESSOR_CACHE: Dict[str, Any] = {}


def _validate_messages(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("each chat message must be a dict")
        if "role" not in message or "content" not in message:
            raise ValueError("each chat message must contain role and content")
    return value


def _build_request(
    *,
    metadata: Any,
    condition_json: Any,
    current_text: str,
    ground_truth_text: str,
    candidate_text: str,
    ground_truth_speech_metadata: Any,
    current_image: str | Path,
    ground_truth_image: str | Path,
    candidate_image: str | Path,
    candidate_audio: str | Path,
) -> Dict[str, Any]:
    messages = build_joint_messages(
        metadata=serialize_prompt_value(normalize_metadata(metadata)),
        condition_json=serialize_prompt_value(
            normalize_condition_json(condition_json)
        ),
        current_text=current_text,
        ground_truth_text=ground_truth_text,
        candidate_text=candidate_text,
        ground_truth_speech_metadata=serialize_prompt_value(
            normalize_speech_metadata(ground_truth_speech_metadata)
        ),
        current_image=Path(current_image),
        ground_truth_image=Path(ground_truth_image),
        candidate_image=Path(candidate_image),
        candidate_audio=Path(candidate_audio),
    )
    return {"messages": _validate_messages(messages)}


def _get_processor(model_name: str) -> Any:
    cached = _PROCESSOR_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        from transformers import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
    except ImportError:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    _PROCESSOR_CACHE[model_name] = processor
    return processor


def _build_vllm_input(
    model_name: str,
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    use_audio_in_video: bool,
) -> Dict[str, Any]:
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise RuntimeError(
            "qwen-omni-utils is required for Qwen3-Omni multimodal "
            "preprocessing in the active Conda environment."
        ) from exc

    controlled_messages = apply_model_chat_controls(model_name, messages)
    prompt = processor.apply_chat_template(
        controlled_messages,
        tokenize=False,
        add_generation_prompt=True,
        **get_chat_template_kwargs(model_name),
    )
    audios, images, videos = process_mm_info(
        controlled_messages,
        use_audio_in_video=use_audio_in_video,
    )

    inputs: Dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {},
        "mm_processor_kwargs": {
            "use_audio_in_video": use_audio_in_video,
        },
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
    return inputs


def _generate_responses(
    model_name: str,
    requests: Sequence[Dict[str, Any]],
    *,
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_JOINT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    use_audio_in_video: bool = False,
) -> List[str]:
    if not requests:
        return []

    resolve_backend("vllm")
    from vllm import SamplingParams

    processor = _get_processor(model_name)
    inputs = [
        _build_vllm_input(
            model_name,
            processor,
            request["messages"],
            use_audio_in_video=use_audio_in_video,
        )
        for request in requests
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
    outputs = llm.generate(
        inputs,
        sampling_params,
        use_tqdm=False,
    )
    return [
        output.outputs[0].text.strip() if output.outputs else ""
        for output in outputs
    ]


def evaluate_qwen3_omni_batch(
    model_name: str,
    requests: Sequence[Dict[str, Any]],
    *,
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_JOINT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    use_audio_in_video: bool = False,
) -> List[Dict[str, Any]]:
    prepared_requests = [
        _build_request(
            metadata=request["metadata"],
            condition_json=request["condition_json"],
            current_text=request["current_text"],
            ground_truth_text=request["ground_truth_text"],
            candidate_text=request["candidate_text"],
            ground_truth_speech_metadata=request[
                "ground_truth_speech_metadata"
            ],
            current_image=request["current_image"],
            ground_truth_image=request["ground_truth_image"],
            candidate_image=request["candidate_image"],
            candidate_audio=request["candidate_audio"],
        )
        for request in requests
    ]
    responses = _generate_responses(
        model_name,
        prepared_requests,
        vllm_kwargs=vllm_kwargs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_audio_in_video=use_audio_in_video,
    )

    results: List[Dict[str, Any]] = []
    for response in responses:
        result = parse_judge_response(response, REQUIRED_SCORE_KEYS)
        result["qwen3_omni_eval"] = result["judge_eval"]
        results.append(result)
    return results
