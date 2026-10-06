from copy import deepcopy
from typing import Any, Dict, List, Sequence

from src.model_adapters.base import ModelAdapter, PromptAdapterError


_IMAGE_LABELS = {
    "The next image is the current scene image.": (
        "Image 1 above is the current scene image."
    ),
    "The next image is the ground-truth next-scene image.": (
        "Image 2 above is the ground-truth next-scene image."
    ),
    "The next image is the generated candidate next-scene image.": (
        "Image 3 above is the generated candidate next-scene image."
    ),
}


def adapt_joint_messages(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Place Gemma 4 images before text and audio after all text."""

    adapted = deepcopy(list(messages))
    for message in adapted:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise PromptAdapterError("Gemma 4 joint user content must be a list")

        images = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        audios = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "audio"
        ]
        text_items = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(images) != 3 or len(audios) != 1:
            raise PromptAdapterError(
                "Gemma 4 joint prompt requires exactly three images and one audio"
            )

        body_texts: List[Dict[str, Any]] = []
        final_instruction = None
        audio_label = None
        for item in text_items:
            text = str(item.get("text", ""))
            if "Score all required criteria. Return JSON only." in text:
                final_instruction = item
                continue
            if "<generated_candidate_speech_audio>" in text:
                audio_label = item
                continue
            updated = dict(item)
            for source, target in _IMAGE_LABELS.items():
                text = text.replace(source, target)
            updated["text"] = text
            body_texts.append(updated)

        if final_instruction is None or audio_label is None:
            raise PromptAdapterError(
                "Gemma 4 joint prompt is missing its final instruction or audio label"
            )

        message["content"] = [
            *images,
            *body_texts,
            final_instruction,
            audio_label,
            *audios,
        ]
        return adapted

    raise PromptAdapterError("Gemma 4 joint prompt is missing a user message")


ADAPTER = ModelAdapter(
    key="stage3_gemma4_omni",
    stage="3",
    modality="joint",
    model_name="google/gemma-4-12B-it",
    inference_family="vllm_omni",
    chat_template_kwargs={"enable_thinking": False},
    omni_prompt_adapter="gemma4",
    enforce_json_schema=True,
    # Keep the benchmark's greedy first attempt. Only malformed generations
    # retry with the model release's recommended sampling configuration.
    retry_sampling_overrides={
        "temperature": 1.0, "top_p": 0.95, "top_k": 64
    },
    structured_output_kwargs={
        "disable_any_whitespace": True,
        "disable_additional_properties": True,
    },
    repetition_detection_kwargs={
        "max_pattern_size": 20,
        "min_pattern_size": 3,
        "min_count": 4,
    },
    max_generation_attempts=3,
    # Transformers 5 removed the fft_length attribute used by vLLM's
    # Gemma 4 dummy-audio profiler. Runtime audio processing is unaffected.
    engine_kwargs={
        "skip_mm_profiling": True,
        "mm_processor_kwargs": {"max_soft_tokens": 1120},
    },
)
