"""Media loading helpers shared by multimodal inference executors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_audio_payload(path: str | Path) -> Any:
    audio_path = str(path)
    try:
        import librosa

        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        return (audio, sample_rate)
    except Exception:
        pass

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path)
        return (audio, sample_rate)
    except Exception as exc:
        raise RuntimeError(
            "Could not load audio for vLLM multimodal input. Install librosa "
            "or soundfile in the active environment."
        ) from exc
