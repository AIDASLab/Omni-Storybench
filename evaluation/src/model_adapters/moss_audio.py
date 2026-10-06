from src.model_adapters.base import ModelAdapter


AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
SYSTEM_PROMPT = "You are a helpful assistant."


def build_audio_prompt(request_prompt: str) -> str:
    # Match the default conversation framing used by the official MOSS processor.
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{AUDIO_PLACEHOLDER}\n"
        f"{request_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


ADAPTER = ModelAdapter(
    key="stage2_moss_audio",
    stage="2",
    modality="speech",
    model_name="OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    inference_family="vllm_audio",
    audio_prompt_adapter="moss_audio",
    engine_kwargs={"mm_processor_kwargs": {"enable_time_marker": True}},
)
