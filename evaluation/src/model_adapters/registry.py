from __future__ import annotations

from src.model_adapters.audio_flamingo_next import ADAPTER as AUDIO_FLAMINGO_NEXT
from src.model_adapters.base import ModelAdapter
from src.model_adapters.exaone45 import ADAPTER as EXAONE45
from src.model_adapters.gemma4 import ADAPTER as GEMMA4
from src.model_adapters.internvl35 import ADAPTER as INTERNVL35
from src.model_adapters.kimi_audio import ADAPTER as KIMI_AUDIO
from src.model_adapters.llama_nemotron import ADAPTER as LLAMA_NEMOTRON
from src.model_adapters.moss_audio import ADAPTER as MOSS_AUDIO
from src.model_adapters.nemotron3_nano_omni import ADAPTER as NEMOTRON3_NANO_OMNI
from src.model_adapters.qwen3_omni import ADAPTER as QWEN3_OMNI
from src.model_adapters.qwen3_text import ADAPTER as QWEN3_TEXT
from src.model_adapters.qwen3_vl import ADAPTER as QWEN3_VL
from src.model_adapters.seed_oss import ADAPTER as SEED_OSS


ALL_ADAPTERS = (
    QWEN3_TEXT,
    QWEN3_VL,
    AUDIO_FLAMINGO_NEXT,
    QWEN3_OMNI,
    SEED_OSS,
    INTERNVL35,
    MOSS_AUDIO,
    NEMOTRON3_NANO_OMNI,
    LLAMA_NEMOTRON,
    EXAONE45,
    KIMI_AUDIO,
    GEMMA4,
)


def _build_unique_index(
    attribute: str,
    *,
    casefold: bool = False,
) -> dict[str, ModelAdapter]:
    index: dict[str, ModelAdapter] = {}
    for adapter in ALL_ADAPTERS:
        value = str(getattr(adapter, attribute))
        lookup_key = value.casefold() if casefold else value
        if lookup_key in index:
            other = index[lookup_key]
            raise ValueError(
                f"Duplicate adapter {attribute} {value!r}: "
                f"{other.key!r} and {adapter.key!r}"
            )
        index[lookup_key] = adapter
    return index


_BY_KEY = _build_unique_index("key")
_BY_MODEL_NAME = _build_unique_index("model_name", casefold=True)


def adapter_keys(modality: str | None = None) -> tuple[str, ...]:
    return tuple(
        adapter.key
        for adapter in ALL_ADAPTERS
        if modality is None or adapter.modality == modality
    )


def get_adapter(key: str) -> ModelAdapter:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model adapter: {key}") from exc


def get_adapter_by_model_name(model_name: str) -> ModelAdapter | None:
    return _BY_MODEL_NAME.get(model_name.lower())


def validate_adapter(
    adapter: ModelAdapter,
    *,
    stage: str,
    modality: str,
    model_name: str | None = None,
    backend: str | None = None,
) -> None:
    mismatches = []
    if adapter.stage != stage:
        mismatches.append(f"stage={stage!r}, expected {adapter.stage!r}")
    if adapter.modality != modality:
        mismatches.append(f"modality={modality!r}, expected {adapter.modality!r}")
    if model_name is not None and model_name.casefold() != adapter.model_name.casefold():
        mismatches.append(
            f"model_name={model_name!r}, expected {adapter.model_name!r}"
        )
    if backend not in {None, "auto", adapter.backend}:
        mismatches.append(f"backend={backend!r}, expected {adapter.backend!r}")
    if mismatches:
        raise ValueError(
            f"Adapter {adapter.key!r} does not match this evaluation job: "
            + "; ".join(mismatches)
        )
