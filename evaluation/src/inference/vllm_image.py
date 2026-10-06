"""Shared vLLM image judge executor for every evaluation stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoProcessor

from src.common import JUDGE_CATEGORIES_BY_MODALITY
from src.evaluation_protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_image_messages,
)
from src.inference.common import (
    normalize_condition_json,
    normalize_metadata,
    serialize_prompt_value,
)
from src.inference.media import load_rgb_image
from src.judge_parsing import parse_judge_response
from src.inference.vllm_runtime import (
    build_image_prompt,
    get_or_create_llm,
    resolve_backend,
)


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["image"]
_PROCESSOR_CACHE: Dict[str, Any] = {}


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


def evaluate_image_batch(
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

    processor = _get_processor(model_name)
    prompts: List[Dict[str, Any]] = []
    for request in requests:
        messages = build_image_messages(
            metadata=serialize_prompt_value(
                normalize_metadata(request["metadata"])
            ),
            condition_json=serialize_prompt_value(
                normalize_condition_json(request["condition_json"])
            ),
            current_scene_image=request["current_image"],
            ground_truth_image=request["ground_truth_image"],
            candidate_image=request["candidate_image"],
        )
        prompts.append(
            {
                "prompt": build_image_prompt(
                    model_name,
                    processor,
                    messages,
                    expected_image_count=3,
                ),
                "multi_modal_data": {
                    "image": [
                        load_rgb_image(request["current_image"]),
                        load_rgb_image(request["ground_truth_image"]),
                        load_rgb_image(request["candidate_image"]),
                    ]
                },
            }
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    return [
        parse_judge_response(
            output.outputs[0].text.strip() if output.outputs else "",
            REQUIRED_SCORE_KEYS,
        )
        for output in outputs
    ]
