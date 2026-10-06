from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage3_exaone45_image",
    stage="3",
    modality="image",
    model_name="LGAI-EXAONE/EXAONE-4.5-33B",
    inference_family="vllm_image",
    chat_template_kwargs={"enable_thinking": False},
)
