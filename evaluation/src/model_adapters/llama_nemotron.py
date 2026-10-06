from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage3_llama_nemotron_text",
    stage="3",
    modality="text",
    model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    inference_family="vllm_text",
    system_prompt_directive="/no_think",
)
