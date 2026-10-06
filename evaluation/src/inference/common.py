"""Small, dependency-light helpers shared by judge inference backends."""

from __future__ import annotations

import json
from typing import Any


def unwrap_mapping(value: Any, wrapper_key: str) -> Any:
    if isinstance(value, dict) and wrapper_key in value:
        return value[wrapper_key]
    return value


def normalize_metadata(value: Any) -> Any:
    return unwrap_mapping(value, "metadata")


def normalize_condition_json(value: Any) -> Any:
    return unwrap_mapping(value, "condition_json")


def normalize_speech_metadata(value: Any) -> Any:
    return unwrap_mapping(value, "parsed")


def serialize_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)
