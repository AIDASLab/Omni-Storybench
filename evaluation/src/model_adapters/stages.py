"""Canonical stage-to-adapter assignments for the evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.model_adapters.base import ModelAdapter
from src.model_adapters.registry import get_adapter, validate_adapter


Modality = Literal["text", "image", "speech", "joint"]


@dataclass(frozen=True)
class StageConfig:
    stage: str
    text_adapter: str
    image_adapter: str
    speech_adapter: str
    joint_adapter: str

    def adapter_key(self, modality: Modality) -> str:
        return {
            "text": self.text_adapter,
            "image": self.image_adapter,
            "speech": self.speech_adapter,
            "joint": self.joint_adapter,
        }[modality]

    def adapter(self, modality: Modality) -> ModelAdapter:
        return get_adapter(self.adapter_key(modality))

    @property
    def text_model(self) -> str:
        return self.adapter("text").model_name

    @property
    def image_model(self) -> str:
        return self.adapter("image").model_name

    @property
    def speech_model(self) -> str:
        return self.adapter("speech").model_name

    @property
    def joint_model(self) -> str:
        return self.adapter("joint").model_name

    @property
    def text_backend(self) -> str:
        return self.adapter("text").backend

    @property
    def image_backend(self) -> str:
        return self.adapter("image").backend

    @property
    def speech_backend(self) -> str:
        return self.adapter("speech").backend

    @property
    def joint_backend(self) -> str:
        return self.adapter("joint").backend


STAGE_CONFIGS: dict[str, StageConfig] = {
    "1": StageConfig(
        stage="1",
        text_adapter="stage1_qwen3_text",
        image_adapter="stage1_qwen3_image",
        speech_adapter="stage1_audio_flamingo_next",
        joint_adapter="stage1_qwen3_omni",
    ),
    "2": StageConfig(
        stage="2",
        text_adapter="stage2_seed_oss_text",
        image_adapter="stage2_internvl35_image",
        speech_adapter="stage2_moss_audio",
        joint_adapter="stage2_nemotron3_nano_omni",
    ),
    "3": StageConfig(
        stage="3",
        text_adapter="stage3_llama_nemotron_text",
        image_adapter="stage3_exaone45_image",
        speech_adapter="stage3_kimi_audio",
        joint_adapter="stage3_gemma4_omni",
    ),
}


def get_stage_config(stage: str) -> StageConfig:
    try:
        return STAGE_CONFIGS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation stage: {stage}") from exc


def get_stage_adapter(stage: str, modality: Modality) -> ModelAdapter:
    adapter = get_stage_config(stage).adapter(modality)
    validate_adapter(adapter, stage=stage, modality=modality)
    return adapter
