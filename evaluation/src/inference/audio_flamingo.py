"""Transformers executor for NVIDIA Audio Flamingo Next speech judging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import transformers
from transformers import AutoModel, AutoProcessor

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
from src.judge_parsing import parse_judge_response


REQUIRED_SCORE_KEYS = JUDGE_CATEGORIES_BY_MODALITY["speech"]
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_MODEL_ERROR_CACHE: Dict[str, RuntimeError] = {}


class AudioFlamingoNextSupportError(RuntimeError):
    pass


def seed_audio_flamingo(seed: int | None) -> None:
    if seed is not None:
        transformers.set_seed(seed)


def _build_messages(
    metadata: str,
    condition_json: str,
    ground_truth_speech_metadata: str,
    candidate_audio_path: Path,
) -> List[List[Dict[str, Any]]]:
    prompt = build_speech_prompt_text(
        metadata=metadata,
        condition_json=condition_json,
        ground_truth_speech_metadata=ground_truth_speech_metadata,
    )
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(candidate_audio_path)},
                ],
            }
        ]
    ]


def _validate_messages(value: Any) -> List[List[Dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list of conversations")
    for conversation in value:
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("each conversation must contain at least one message")
        for message in conversation:
            if not isinstance(message, dict):
                raise TypeError("each chat message must be a dict")
            if "role" not in message or "content" not in message:
                raise ValueError("each chat message must contain role and content")
    return value


def _install_hint() -> str:
    version = getattr(transformers, "__version__", "unknown")
    return (
        "Audio Flamingo Next is not supported by the currently imported "
        f"Transformers build (detected version: {version}). Install a newer "
        "Transformers/Accelerate stack or the branch recommended by the model card, "
        "then restart the Python process before rerunning evaluation."
    )


def _ensure_model_supported(model_name: str) -> None:
    cached_error = _MODEL_ERROR_CACHE.get(model_name)
    if cached_error is not None:
        raise cached_error

    missing = [
        symbol
        for symbol in (
            "AudioFlamingoNextProcessor",
            "AudioFlamingoNextForConditionalGeneration",
        )
        if not hasattr(transformers, symbol)
    ]
    if missing:
        error = AudioFlamingoNextSupportError(
            f"{model_name!r} requires Audio Flamingo Next support, but the current "
            f"Transformers build is missing: {', '.join(missing)}. {_install_hint()}"
        )
        _MODEL_ERROR_CACHE[model_name] = error
        raise error

    if importlib.util.find_spec("librosa") is None:
        error = AudioFlamingoNextSupportError(
            f"{model_name!r} requires librosa when local audio paths are passed "
            "to processor.apply_chat_template()."
        )
        _MODEL_ERROR_CACHE[model_name] = error
        raise error


def _get_model_and_processor(model_name: str) -> Tuple[Any, Any]:
    _ensure_model_supported(model_name)
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model.eval()
        if device != "cuda":
            model = model.to(device)
    except Exception as exc:
        error = AudioFlamingoNextSupportError(
            f"Failed to initialize {model_name!r}. {_install_hint()} "
            f"Original error: {type(exc).__name__}: {exc}"
        )
        _MODEL_ERROR_CACHE[model_name] = error
        raise error from exc

    _MODEL_CACHE[model_name] = (processor, model)
    return processor, model


def _evaluate_request(
    model_name: str,
    metadata: Any,
    condition_json: Any,
    ground_truth_speech_metadata: Any,
    candidate_audio_path: str | Path,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> Dict[str, Any]:
    audio_path = Path(candidate_audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Candidate audio file not found: {audio_path}")

    messages = _build_messages(
        metadata=serialize_prompt_value(normalize_metadata(metadata)),
        condition_json=serialize_prompt_value(
            normalize_condition_json(condition_json)
        ),
        ground_truth_speech_metadata=serialize_prompt_value(
            normalize_speech_metadata(ground_truth_speech_metadata)
        ),
        candidate_audio_path=audio_path,
    )
    processor, model = _get_model_and_processor(model_name)
    inputs = processor.apply_chat_template(
        _validate_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_features = inputs.get("input_features")
    if isinstance(input_features, torch.Tensor) and input_features.is_floating_point():
        inputs["input_features"] = input_features.to(model.dtype)

    prompt_length = inputs["input_ids"].shape[1]
    do_sample = temperature > 0.0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    completion = generated[:, prompt_length:]
    response = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return parse_judge_response(response, REQUIRED_SCORE_KEYS)


def evaluate_audio_flamingo_batch(
    model_name: str,
    requests: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> List[Dict[str, Any]]:
    return [
        _evaluate_request(
            model_name,
            request["metadata"],
            request["condition_json"],
            request["ground_truth_speech_metadata"],
            request["candidate_audio"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        for request in requests
    ]
