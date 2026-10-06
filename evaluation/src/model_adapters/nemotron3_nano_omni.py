from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage2_nemotron3_nano_omni",
    stage="2",
    modality="joint",
    model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    inference_family="vllm_omni",
    chat_template_kwargs={"enable_thinking": False},
    engine_kwargs={"media_io_kwargs": {"video": {"fps": 2, "num_frames": 256}}},
)
