from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage1_qwen3_omni",
    stage="1",
    modality="joint",
    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    inference_family="qwen3_omni_vllm",
)
