from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class PromptAdapterError(ValueError):
    """Raised when canonical messages cannot be rendered for a model."""


@dataclass(frozen=True)
class ModelAdapter:
    key: str
    stage: str
    modality: str
    model_name: str
    inference_family: str
    backend: str = "vllm"
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    system_prompt_directive: str | None = None
    image_prompt_adapter: str | None = None
    audio_prompt_adapter: str | None = None
    omni_prompt_adapter: str | None = None
    stop_token_ids: tuple[int, ...] = ()
    engine_kwargs: Mapping[str, Any] = field(default_factory=dict)
    enforce_json_schema: bool = False
    retry_sampling_overrides: Mapping[str, Any] = field(default_factory=dict)
    structured_output_kwargs: Mapping[str, Any] = field(default_factory=dict)
    repetition_detection_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_generation_attempts: int = 1

    def reasoning_metadata(self) -> dict[str, Any]:
        return {
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "system_prompt_directive": self.system_prompt_directive,
        }

    def control_metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.key,
            "reasoning": self.reasoning_metadata(),
            "image_prompt_adapter": self.image_prompt_adapter,
            "audio_prompt_adapter": self.audio_prompt_adapter,
            "omni_prompt_adapter": self.omni_prompt_adapter,
            "stop_token_ids": list(self.stop_token_ids) or None,
            "engine_kwargs": dict(self.engine_kwargs),
            "enforce_json_schema": self.enforce_json_schema,
            "retry_sampling_overrides": dict(self.retry_sampling_overrides),
            "structured_output_kwargs": dict(self.structured_output_kwargs),
            "repetition_detection_kwargs": dict(
                self.repetition_detection_kwargs
            ),
            "max_generation_attempts": self.max_generation_attempts,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "modality": self.modality,
            "model_name": self.model_name,
            "inference_family": self.inference_family,
            "backend": self.backend,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "system_prompt_directive": self.system_prompt_directive,
            "image_prompt_adapter": self.image_prompt_adapter,
            "audio_prompt_adapter": self.audio_prompt_adapter,
            "omni_prompt_adapter": self.omni_prompt_adapter,
            "stop_token_ids": list(self.stop_token_ids) or None,
            "engine_kwargs": dict(self.engine_kwargs),
            "enforce_json_schema": self.enforce_json_schema,
            "retry_sampling_overrides": dict(self.retry_sampling_overrides),
            "structured_output_kwargs": dict(self.structured_output_kwargs),
            "repetition_detection_kwargs": dict(
                self.repetition_detection_kwargs
            ),
            "max_generation_attempts": self.max_generation_attempts,
        }
