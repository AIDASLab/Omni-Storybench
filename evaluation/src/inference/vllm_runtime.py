"""Shared vLLM engine lifecycle and model-control helpers."""

import gc
import importlib.util
import json
from copy import deepcopy
from typing import Any, Dict, Iterable

from src.model_adapters.internvl35 import build_image_prompt as build_internvl_image_prompt
from src.model_adapters.kimi_audio import build_audio_prompt as build_kimi_audio_prompt
from src.model_adapters.moss_audio import build_audio_prompt as build_moss_audio_prompt
from src.model_adapters.registry import get_adapter_by_model_name


_VLLM_LLM_CACHE: Dict[str, Any] = {}


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def resolve_backend(requested_backend: str) -> str:
    backend = requested_backend.lower()
    if backend not in {"auto", "transformers", "vllm"}:
        raise ValueError(
            f"Unsupported backend: {requested_backend}. Expected one of auto/transformers/vllm."
        )

    if backend == "auto":
        return "vllm" if is_vllm_available() else "transformers"

    if backend == "vllm" and not is_vllm_available():
        raise RuntimeError(
            "The vLLM backend was requested, but the `vllm` package is not installed "
            "in the current Python environment."
        )

    return backend


def should_enable_expert_parallel(model_name: str) -> bool:
    adapter = get_adapter_by_model_name(model_name)
    if adapter is not None:
        return adapter.engine_kwargs.get("enable_expert_parallel") is True
    normalized = model_name.lower()
    return any(token in normalized for token in ("a3b", "moe"))


def get_chat_template_kwargs(model_name: str) -> Dict[str, Any]:
    adapter = get_adapter_by_model_name(model_name)
    return dict(adapter.chat_template_kwargs) if adapter is not None else {}


def apply_model_chat_controls(
    model_name: str,
    messages: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Apply model-specific non-reasoning controls to canonical messages."""
    controlled = deepcopy(list(messages))
    adapter = get_adapter_by_model_name(model_name)
    directive = adapter.system_prompt_directive if adapter is not None else None
    if not directive:
        return controlled

    for message in controlled:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            if directive not in content:
                message["content"] = f"{directive}\n{content}".strip()
            return controlled
        if isinstance(content, list):
            if not any(
                isinstance(item, dict) and directive in str(item.get("text", ""))
                for item in content
            ):
                content.insert(0, {"type": "text", "text": directive})
            return controlled

    controlled.insert(0, {"role": "system", "content": directive})
    return controlled


def is_kimi_audio_model(model_name: str) -> bool:
    adapter = get_adapter_by_model_name(model_name)
    return adapter is not None and adapter.audio_prompt_adapter == "kimi_audio"


def is_moss_audio_model(model_name: str) -> bool:
    adapter = get_adapter_by_model_name(model_name)
    return adapter is not None and adapter.audio_prompt_adapter == "moss_audio"


def build_image_prompt(
    model_name: str,
    processor: Any,
    messages: Iterable[Dict[str, Any]],
    *,
    expected_image_count: int,
) -> str:
    controlled_messages = apply_model_chat_controls(model_name, messages)
    adapter = get_adapter_by_model_name(model_name)
    if adapter is not None and adapter.image_prompt_adapter == "internvl35":
        return build_internvl_image_prompt(
            processor,
            controlled_messages,
            chat_template_kwargs=get_chat_template_kwargs(model_name),
            expected_image_count=expected_image_count,
        )
    return processor.apply_chat_template(
        controlled_messages,
        tokenize=False,
        add_generation_prompt=True,
        **get_chat_template_kwargs(model_name),
    )


def get_stop_token_ids(model_name: str) -> list[int] | None:
    adapter = get_adapter_by_model_name(model_name)
    if adapter is None:
        return None
    return list(adapter.stop_token_ids) or None


def _make_cacheable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _make_cacheable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_make_cacheable(item) for item in value]
    return value


def _build_cache_key(model_name: str, engine_kwargs: Dict[str, Any]) -> str:
    payload = {
        "model_name": model_name,
        "engine_kwargs": _make_cacheable(engine_kwargs),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def get_or_create_llm(model_name: str, **engine_kwargs: Any) -> Any:
    resolve_backend("vllm")
    filtered_kwargs = {
        key: value
        for key, value in engine_kwargs.items()
        if value is not None
    }
    cache_key = _build_cache_key(model_name, filtered_kwargs)

    cached = _VLLM_LLM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from vllm import LLM

    llm = LLM(model=model_name, **filtered_kwargs)
    _VLLM_LLM_CACHE[cache_key] = llm
    return llm


def is_model_loaded(model_name: str, **engine_kwargs: Any) -> bool:
    filtered_kwargs = {
        key: value for key, value in engine_kwargs.items() if value is not None
    }
    return _build_cache_key(model_name, filtered_kwargs) in _VLLM_LLM_CACHE


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_fatal_error(exc: BaseException) -> bool:
    fatal_class_names = {
        "EngineDeadError",
        "EngineGenerateError",
        "ImportError",
        "ModuleNotFoundError",
        "OutOfMemoryError",
        "PromptAdapterError",
    }
    fatal_message_fragments = (
        "cannot re-initialize cuda in forked subprocess",
        "engine core initialization failed",
        "enginecore failed to start",
        "engine core failed to start",
        "engine is dead",
        "engine has died",
        "workerproc failed",
        "cuda out of memory",
        "failed to apply required model chat template kwargs",
        "failed to apply prompt replacement",
    )
    for chained in _exception_chain(exc):
        if type(chained).__name__ in fatal_class_names:
            return True
        message = str(chained).lower()
        if any(fragment in message for fragment in fatal_message_fragments):
            return True
    return False


def should_retry_batch_error(
    exc: BaseException,
    model_name: str,
    **engine_kwargs: Any,
) -> bool:
    return is_model_loaded(
        model_name, **engine_kwargs
    ) and not is_fatal_error(exc)


def _iter_matching_cache_keys(model_name: str | None = None) -> Iterable[str]:
    for cache_key in list(_VLLM_LLM_CACHE):
        if model_name is None:
            yield cache_key
            continue
        payload = json.loads(cache_key)
        if payload.get("model_name") == model_name:
            yield cache_key


def force_memory_cleanup() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _shutdown_vllm_llm(llm: Any) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is None:
        return

    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()

    renderer = getattr(llm_engine, "renderer", None)
    if renderer is not None and hasattr(renderer, "shutdown"):
        renderer.shutdown()

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        pass


def release_models(model_name: str | None = None) -> None:
    released_any = False
    for cache_key in list(_iter_matching_cache_keys(model_name=model_name)):
        llm = _VLLM_LLM_CACHE.pop(cache_key, None)
        if llm is None:
            continue
        released_any = True
        try:
            _shutdown_vllm_llm(llm)
        except Exception:
            pass
        del llm

    if released_any:
        force_memory_cleanup()
