from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage1_qwen3_text",
    stage="1",
    modality="text",
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    inference_family="vllm_text",
    engine_kwargs={"enable_expert_parallel": True},
)
