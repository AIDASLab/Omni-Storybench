from typing import Any, Mapping, Sequence

from src.model_adapters.base import ModelAdapter, PromptAdapterError


IMAGE_PLACEHOLDER = "<image>"


def _render_content(content: Any) -> tuple[str, int]:
    if isinstance(content, str):
        return content, 0
    if not isinstance(content, list):
        raise PromptAdapterError(
            f"InternVL message content must be a string or list, got {type(content).__name__}"
        )

    parts: list[str] = []
    image_count = 0
    for item in content:
        if not isinstance(item, Mapping):
            raise PromptAdapterError(
                "InternVL multimodal message items must be mappings"
            )
        item_type = item.get("type")
        if item_type == "text":
            parts.append(str(item.get("text", "")))
        elif item_type in {"image", "image_url"}:
            parts.append(IMAGE_PLACEHOLDER)
            image_count += 1
        else:
            raise PromptAdapterError(
                f"Unsupported InternVL message item type: {item_type!r}"
            )
    return "".join(parts), image_count


def build_image_prompt(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    chat_template_kwargs: Mapping[str, Any],
    expected_image_count: int,
) -> str:
    """Render OpenAI-style multimodal messages for InternVL offline inference."""
    flattened_messages = []
    image_count = 0
    for message in messages:
        content, content_image_count = _render_content(message.get("content", ""))
        flattened_messages.append(
            {"role": str(message.get("role", "user")), "content": content}
        )
        image_count += content_image_count

    if image_count != expected_image_count:
        raise PromptAdapterError(
            "InternVL prompt/image count mismatch: "
            f"rendered {image_count} placeholders for {expected_image_count} images"
        )

    try:
        return processor.apply_chat_template(
            flattened_messages,
            tokenize=False,
            add_generation_prompt=True,
            **dict(chat_template_kwargs),
        )
    except Exception as exc:
        raise PromptAdapterError(
            "InternVL chat template could not render string-valued messages"
        ) from exc


ADAPTER = ModelAdapter(
    key="stage2_internvl35_image",
    stage="2",
    modality="image",
    model_name="OpenGVLab/InternVL3_5-38B-Instruct",
    inference_family="vllm_image",
    image_prompt_adapter="internvl35",
)
