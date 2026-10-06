"""Shared vLLM text judge executor for every evaluation stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.common import JUDGE_CATEGORIES_BY_MODALITY
from src.evaluation_protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_text_messages,
)
from src.inference.common import (
    normalize_condition_json,
    normalize_metadata,
    serialize_prompt_value,
)
from src.judge_parsing import parse_judge_response
from src.inference.vllm_runtime import (
    apply_model_chat_controls,
    get_chat_template_kwargs,
    get_or_create_llm,
    resolve_backend,
)


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["text"]


def evaluate_text_batch(
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

    messages_batch = []
    for request in requests:
        messages = build_text_messages(
            metadata=serialize_prompt_value(
                normalize_metadata(request["metadata"])
            ),
            condition_json=serialize_prompt_value(
                normalize_condition_json(request["condition_json"])
            ),
            prev_text=request["current_text"],
            next_text=request["ground_truth_text"],
            prediction=request["candidate_text"],
        )
        messages_batch.append(
            apply_model_chat_controls(model_name, messages)
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
    outputs = llm.chat(
        messages_batch,
        sampling_params,
        use_tqdm=False,
        chat_template_kwargs=get_chat_template_kwargs(model_name) or None,
    )
    return [
        parse_judge_response(
            output.outputs[0].text.strip() if output.outputs else "",
            REQUIRED_SCORE_KEYS,
        )
        for output in outputs
    ]
