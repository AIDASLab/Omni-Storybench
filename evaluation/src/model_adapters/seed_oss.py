from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage2_seed_oss_text",
    stage="2",
    modality="text",
    model_name="ByteDance-Seed/Seed-OSS-36B-Instruct",
    inference_family="vllm_text",
    chat_template_kwargs={"thinking_budget": 0},
)
