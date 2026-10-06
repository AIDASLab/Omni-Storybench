"""Dedicated generation backend for stage-end JSON repair."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation_protocol import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from src.inference.vllm_runtime import get_or_create_llm, resolve_backend


_REPAIR_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _get_transformers_model(model_name: str) -> Tuple[Any, Any]:
    cached = _REPAIR_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    _REPAIR_MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def generate_json_repair_responses(
    model_name: str,
    messages_batch: Sequence[List[Dict[str, Any]]],
    *,
    json_schemas: Sequence[Dict[str, Any]],
    backend: str = "vllm",
    vllm_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> List[str]:
    if len(json_schemas) != len(messages_batch):
        raise ValueError("json_schemas must match messages_batch length")
    if not messages_batch:
        return []

    resolved_backend = resolve_backend(backend)
    if resolved_backend == "vllm":
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        sampling_params = [
            SamplingParams(
                **sampling_kwargs,
                structured_outputs=StructuredOutputsParams(json=schema),
            )
            for schema in json_schemas
        ]
        llm = get_or_create_llm(model_name, **(vllm_kwargs or {}))
        outputs = llm.chat(
            messages_batch,
            sampling_params,
            use_tqdm=False,
        )
        return [
            output.outputs[0].text.strip() if output.outputs else ""
            for output in outputs
        ]

    tokenizer, model = _get_transformers_model(model_name)
    responses: List[str] = []
    do_sample = temperature > 0.0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty

    for messages in messages_batch:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        responses.append(
            tokenizer.decode(generated, skip_special_tokens=True).strip()
        )
    return responses
