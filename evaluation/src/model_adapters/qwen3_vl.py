from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage1_qwen3_image",
    stage="1",
    modality="image",
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    inference_family="vllm_image",
)
