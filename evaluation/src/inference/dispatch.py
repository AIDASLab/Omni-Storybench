"""Adapter-driven dispatch for all judge inference executors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.evaluation_protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from src.model_adapters.base import ModelAdapter


def initialize_inference(adapter: ModelAdapter, seed: int | None) -> None:
    if adapter.inference_family == "audio_flamingo_transformers":
        from src.inference.audio_flamingo import seed_audio_flamingo

        seed_audio_flamingo(seed)


def run_batch_inference(
    adapter: ModelAdapter,
    requests: Sequence[Dict[str, Any]],
    *,
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    use_audio_in_video: bool = False,
) -> List[Dict[str, Any]]:
    common_kwargs = {
        "vllm_kwargs": vllm_kwargs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    family = adapter.inference_family

    if family == "vllm_text":
        from src.inference.vllm_text import evaluate_text_batch

        return evaluate_text_batch(
            adapter.model_name,
            requests,
            **common_kwargs,
        )

    if family == "vllm_image":
        from src.inference.vllm_image import evaluate_image_batch

        return evaluate_image_batch(
            adapter.model_name,
            requests,
            **common_kwargs,
        )

    if family == "vllm_audio":
        from src.inference.vllm_audio import evaluate_audio_batch

        return evaluate_audio_batch(
            adapter.model_name,
            requests,
            **common_kwargs,
        )

    if family == "audio_flamingo_transformers":
        from src.inference.audio_flamingo import (
            evaluate_audio_flamingo_batch,
        )

        transformer_kwargs = dict(common_kwargs)
        transformer_kwargs.pop("vllm_kwargs")
        return evaluate_audio_flamingo_batch(
            adapter.model_name,
            requests,
            **transformer_kwargs,
        )

    if family == "qwen3_omni_vllm":
        from src.inference.qwen3_omni import evaluate_qwen3_omni_batch

        return evaluate_qwen3_omni_batch(
            adapter.model_name,
            requests,
            use_audio_in_video=use_audio_in_video,
            **common_kwargs,
        )

    if family == "vllm_omni":
        from src.inference.vllm_omni import (
            evaluate_omni_batch,
        )

        return evaluate_omni_batch(
            adapter.model_name,
            requests,
            **common_kwargs,
        )

    raise ValueError(
        f"Unsupported inference family {family!r} for adapter {adapter.key!r}"
    )
