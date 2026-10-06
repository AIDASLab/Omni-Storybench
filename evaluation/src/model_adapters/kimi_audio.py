from src.model_adapters.base import ModelAdapter


AUDIO_PLACEHOLDER = "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"


def build_audio_prompt(request_prompt: str) -> str:
    return (
        "<|im_kimia_user_msg_start|>"
        f"{request_prompt}\n{AUDIO_PLACEHOLDER}"
        "<|im_msg_end|><|im_kimia_assistant_msg_start|>"
    )


ADAPTER = ModelAdapter(
    key="stage3_kimi_audio",
    stage="3",
    modality="speech",
    model_name="moonshotai/Kimi-Audio-7B-Instruct",
    inference_family="vllm_audio",
    audio_prompt_adapter="kimi_audio",
    stop_token_ids=(151644,),
)
