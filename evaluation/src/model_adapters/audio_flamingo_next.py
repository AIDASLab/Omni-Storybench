from src.model_adapters.base import ModelAdapter


ADAPTER = ModelAdapter(
    key="stage1_audio_flamingo_next",
    stage="1",
    modality="speech",
    model_name="nvidia/audio-flamingo-next-hf",
    inference_family="audio_flamingo_transformers",
    backend="transformers",
)
