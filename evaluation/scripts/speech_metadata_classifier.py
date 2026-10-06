#!/usr/bin/env python3
"""Standalone speech metadata classifier used by the evaluation harness."""

from __future__ import annotations

import json
import math
import os
import shutil
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent

CANONICAL_EMOTIONS = ("neutral", "happy", "sad", "angry")
EMOTION2VEC_LABELS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
    8: "unknown",
}


def require_package(module_name: str, package_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Required package '{package_name}' is not installed in this Python environment. "
            "Install the missing package before running the emotion ensemble."
        )


@dataclass
class ThresholdConfig:
    """Thresholds used for speed and pitch binning."""

    speed_slow_normal: float = 2.6
    speed_normal_fast: float = 4.7

    pitch_unknown_low_normal: float = 145.0
    pitch_unknown_normal_high: float = 220.0

    pitch_male_low_normal: float = 110.0
    pitch_male_normal_high: float = 170.0

    pitch_female_low_normal: float = 165.0
    pitch_female_normal_high: float = 255.0

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def pitch_pair(self, apparent_gender: Optional[str]) -> Tuple[float, float]:
        gender = canonicalize_gender(apparent_gender) if apparent_gender else None
        if gender == "male":
            return self.pitch_male_low_normal, self.pitch_male_normal_high
        if gender == "female":
            return self.pitch_female_low_normal, self.pitch_female_normal_high
        return self.pitch_unknown_low_normal, self.pitch_unknown_normal_high


@dataclass
class WhisperResult:
    text: str
    chunks: List[Dict[str, Any]]


@dataclass
class AcousticMeasurements:
    duration_s: float
    speech_duration_s: float
    units_per_sec: Optional[float]
    median_f0_hz: Optional[float]
    mean_f0_hz: Optional[float]
    voiced_ratio: float


@dataclass
class MetadataPrediction:
    line: str
    emotion: str
    speed: str
    pitch: str
    gender: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def default_cache_base() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser()
    return Path.home().expanduser() / ".cache"


def default_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HOME).expanduser()
    except Exception:
        return default_cache_base() / "huggingface"


def default_hf_hub_cache() -> Path:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE).expanduser()
    except Exception:
        return default_hf_home() / "hub"


def ensure_hf_cache_env() -> Path:
    hf_home = default_hf_home()
    hf_hub_cache = default_hf_hub_cache()

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache))

    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    return hf_hub_cache


def speechbrain_device_string(device: str) -> str:
    return "cuda:0" if device == "cuda" else device


def prepare_speechbrain_source(repo_id: str, wav2vec2_cache_dir: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    hf_hub_cache = ensure_hf_cache_env()
    snapshot_dir = Path(snapshot_download(repo_id=repo_id, cache_dir=str(hf_hub_cache)))
    patched_dir = default_hf_home() / "speech_metadata_classifier" / "patched_repos" / repo_id.replace("/", "--")
    patched_dir.mkdir(parents=True, exist_ok=True)

    for item in snapshot_dir.iterdir():
        if item.name == "hyperparams.yaml":
            continue
        dest = patched_dir / item.name
        if dest.exists():
            continue
        try:
            dest.symlink_to(item, target_is_directory=item.is_dir())
        except OSError:
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    source_yaml = snapshot_dir / "hyperparams.yaml"
    patched_yaml = patched_dir / "hyperparams.yaml"
    cache_path = Path(wav2vec2_cache_dir).expanduser()
    original_text = source_yaml.read_text(encoding="utf-8")
    patched_text = original_text.replace(
        "    save_path: wav2vec2_checkpoints",
        f"    save_path: '{cache_path.as_posix()}'",
    )
    patched_yaml.write_text(patched_text, encoding="utf-8")
    return patched_dir


def canonicalize_emotion(label: str | None) -> Optional[str]:
    if label is None:
        return None
    x = label.strip().lower()
    mapping = {
        "neu": "neutral",
        "neutral": "neutral",
        "ang": "angry",
        "anger": "angry",
        "angry": "angry",
        "hap": "happy",
        "happy": "happy",
        "joy": "happy",
        "sad": "sad",
        "sadness": "sad",
    }
    return mapping.get(x)


def canonicalize_gender(label: str | None) -> Optional[str]:
    if label is None:
        return None
    x = label.strip().lower()
    mapping = {
        "m": "male",
        "male": "male",
        "man": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
    }
    return mapping.get(x)


def choose_device(requested: Optional[str] = None) -> Tuple[str, int, Any]:
    import torch

    if requested:
        req = requested.lower()
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if req not in {"cpu", "cuda"}:
            raise ValueError("--device must be 'cpu' or 'cuda'.")
        device_str = req
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline_device = 0 if device_str == "cuda" else -1
    torch_dtype = torch.float16 if device_str == "cuda" else torch.float32
    return device_str, pipeline_device, torch_dtype


def load_audio(path: str | Path, sr: Optional[int] = None) -> Tuple[Any, int]:
    import librosa

    audio, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    return audio, sample_rate


def resample_audio(audio: Any, orig_sr: int, target_sr: int) -> Any:
    import librosa

    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(audio: Any, sr: int, top_db: int = 30) -> Any:
    import librosa

    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size > 0 else audio


def speech_duration_from_energy(audio: Any, sr: int, top_db: int = 30) -> float:
    import librosa

    if audio.size == 0:
        return 0.0
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return float(len(audio) / sr)
    total = sum((end - start) / sr for start, end in intervals)
    return float(max(total, 1e-6))


def safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def argmax_dict(d: Dict[str, float]) -> str:
    if not d:
        raise ValueError("Cannot argmax an empty dictionary.")
    return max(d.items(), key=lambda kv: kv[1])[0]


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in probs.values()))
    if total <= 0.0:
        uniform = 1.0 / len(probs)
        return {k: uniform for k in probs}
    return {k: max(v, 0.0) / total for k, v in probs.items()}


def average_prob_dicts(dicts: Sequence[Dict[str, float]], weights: Optional[Sequence[float]] = None) -> Dict[str, float]:
    if not dicts:
        raise ValueError("No probability dictionaries were provided.")
    if weights is None:
        weights = [1.0] * len(dicts)
    if len(weights) != len(dicts):
        raise ValueError("weights and dicts must have the same length.")

    acc = {label: 0.0 for label in CANONICAL_EMOTIONS}
    weight_sum = 0.0
    for probs, weight in zip(dicts, weights):
        weight_sum += float(weight)
        for label in CANONICAL_EMOTIONS:
            acc[label] += float(weight) * float(probs.get(label, 0.0))
    if weight_sum <= 0.0:
        return normalize_probs(acc)
    return normalize_probs({k: v / weight_sum for k, v in acc.items()})


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device, self.pipeline_device, self.torch_dtype = choose_device(device)
        self.chunk_length_s = int(chunk_length_s)
        self.batch_size = int(batch_size)
        self._pipe = None

    def _lazy_load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        cache_dir = str(ensure_hf_cache_env())
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        if self.device == "cuda":
            model = model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            torch_dtype=self.torch_dtype,
            device=self.pipeline_device,
        )

    def transcribe(self, audio_path: str | Path, language: Optional[str] = None) -> WhisperResult:
        self._lazy_load()
        assert self._pipe is not None

        gen_kwargs: Dict[str, Any] = {"task": "transcribe"}
        if language:
            gen_kwargs["language"] = language

        result = self._pipe(
            str(audio_path),
            return_timestamps="word",
            generate_kwargs=gen_kwargs,
        )
        text = str(result.get("text", "")).strip()
        chunks = result.get("chunks", []) or []
        return WhisperResult(text=text, chunks=chunks)


class SpeechBrainEmotionClassifier:
    LABEL_ORDER = ["neu", "ang", "hap", "sad"]

    def __init__(self, device: Optional[str] = None) -> None:
        self.device, _, _ = choose_device(device)
        self._clf = None

    def _lazy_load(self) -> None:
        if self._clf is not None:
            return
        from speechbrain.inference.interfaces import foreign_class

        cache_dir = ensure_hf_cache_env()
        source_dir = prepare_speechbrain_source(
            repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            wav2vec2_cache_dir=cache_dir,
        )
        run_opts = {"device": speechbrain_device_string(self.device)}
        self._clf = foreign_class(
            source=str(source_dir),
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts=run_opts,
        )

    def predict_proba(self, audio_16k: Any) -> Dict[str, float]:
        self._lazy_load()
        import torch

        assert self._clf is not None
        waveform = torch.tensor(audio_16k, dtype=torch.float32)
        out_prob, _score, _index, _text_lab = self._clf.classify_batch(waveform)
        probs = out_prob.detach().cpu().numpy().reshape(-1).tolist()

        canonical = {label: 0.0 for label in CANONICAL_EMOTIONS}
        for raw_label, prob in zip(self.LABEL_ORDER, probs):
            canon = canonicalize_emotion(raw_label)
            if canon is not None:
                canonical[canon] += float(prob)
        return normalize_probs(canonical)


class HubertEmotionClassifier:
    def __init__(self, model_name: str = "superb/hubert-large-superb-er", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device, _, _ = choose_device(device)
        self._feature_extractor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None and self._feature_extractor is not None:
            return
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        cache_dir = str(ensure_hf_cache_env())
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model = AutoModelForAudioClassification.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model.eval()
        if self.device == "cuda":
            self._model = self._model.to(self.device)

    def predict_proba(self, audio_16k: Any) -> Dict[str, float]:
        self._lazy_load()
        import torch

        assert self._feature_extractor is not None
        assert self._model is not None

        inputs = self._feature_extractor(audio_16k, sampling_rate=16000, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()

        canonical = {label: 0.0 for label in CANONICAL_EMOTIONS}
        for idx, prob in enumerate(probs):
            raw_label = str(self._model.config.id2label[idx])
            canon = canonicalize_emotion(raw_label)
            if canon is not None:
                canonical[canon] += float(prob)
        return normalize_probs(canonical)


class Emotion2VecClassifier:
    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_large",
        hub: str = "hf",
        output_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.hub = hub
        self.output_dir = Path(output_dir) if output_dir else None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        require_package("funasr", "funasr")
        from funasr import AutoModel

        kwargs: Dict[str, Any] = {"model": self.model_name}
        if self.hub:
            kwargs["hub"] = self.hub
        self._model = AutoModel(**kwargs)

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _flatten_singleton(value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value

    @staticmethod
    def _label_name(label: Any) -> Optional[str]:
        if isinstance(label, int):
            return EMOTION2VEC_LABELS.get(label)
        if isinstance(label, float) and label.is_integer():
            return EMOTION2VEC_LABELS.get(int(label))
        text = str(label).strip().lower()
        if not text:
            return None
        text = text.split("/")[-1]
        if text in {"<unk>", "unk"}:
            return "unknown"
        if text.isdigit():
            return EMOTION2VEC_LABELS.get(int(text))
        return text

    def predict_proba(self, audio_16k: Any, key: str = "audio") -> Dict[str, float]:
        self._lazy_load()
        assert self._model is not None

        generate_kwargs: Dict[str, Any] = {
            "key": key,
            "granularity": "utterance",
            "extract_embedding": False,
            "fs": 16000,
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            generate_kwargs["output_dir"] = str(self.output_dir)

        result = self._model.generate(audio_16k, **generate_kwargs)
        item = result[0] if isinstance(result, list) and result else result
        if not isinstance(item, dict):
            raise RuntimeError(f"Unexpected emotion2vec output: {result!r}")

        labels = self._as_list(self._flatten_singleton(item.get("labels")))
        scores = self._as_list(self._flatten_singleton(item.get("scores")))
        if len(scores) == 1 and isinstance(scores[0], (list, tuple)):
            scores = list(scores[0])

        raw_probs: Dict[str, float] = {}
        if len(scores) == len(EMOTION2VEC_LABELS) and (not labels or len(labels) != len(scores)):
            raw_probs = {EMOTION2VEC_LABELS[idx]: float(score) for idx, score in enumerate(scores)}
        elif labels and len(labels) == len(scores):
            for label, score in zip(labels, scores):
                name = self._label_name(label)
                if name is not None:
                    raw_probs[name] = float(score)
        else:
            raise RuntimeError(f"Could not parse emotion2vec labels/scores: {item!r}")

        canonical = {label: 0.0 for label in CANONICAL_EMOTIONS}
        for raw_label, prob in raw_probs.items():
            canon = canonicalize_emotion(raw_label)
            if canon is not None:
                canonical[canon] += float(prob)
        return normalize_probs(canonical)


class EmotionPredictor:
    def __init__(self, backend: str = "ensemble", device: Optional[str] = None) -> None:
        backend = backend.lower().strip()
        if backend not in {"ensemble", "speechbrain", "hubert", "emotion2vec"}:
            raise ValueError("emotion backend must be one of: ensemble, speechbrain, hubert, emotion2vec")
        self.backend = backend
        if backend in {"ensemble", "speechbrain"}:
            require_package("speechbrain", "speechbrain")
        if backend in {"ensemble", "emotion2vec"}:
            require_package("funasr", "funasr")
            require_package("modelscope", "modelscope")
        self.sb = SpeechBrainEmotionClassifier(device=device) if backend in {"ensemble", "speechbrain"} else None
        self.hb = HubertEmotionClassifier(device=device) if backend in {"ensemble", "hubert"} else None
        self.e2v = Emotion2VecClassifier() if backend in {"ensemble", "emotion2vec"} else None

    def predict(self, audio_16k: Any, audio_path: str | Path) -> Tuple[str, Dict[str, float]]:
        if self.backend == "speechbrain":
            assert self.sb is not None
            probs = self.sb.predict_proba(audio_16k)
            return argmax_dict(probs), probs

        if self.backend == "hubert":
            assert self.hb is not None
            probs = self.hb.predict_proba(audio_16k)
            return argmax_dict(probs), probs

        if self.backend == "emotion2vec":
            assert self.e2v is not None
            probs = self.e2v.predict_proba(audio_16k, key=Path(audio_path).stem)
            return argmax_dict(probs), probs

        prob_dicts: List[Dict[str, float]] = []
        weights: List[float] = []

        assert self.backend == "ensemble"
        assert self.sb is not None
        assert self.hb is not None
        assert self.e2v is not None

        if self.sb is not None:
            prob_dicts.append(self.sb.predict_proba(audio_16k))
            weights.append(1.0)

        if self.hb is not None:
            prob_dicts.append(self.hb.predict_proba(audio_16k))
            weights.append(1.0)

        if self.e2v is not None:
            prob_dicts.append(self.e2v.predict_proba(audio_16k, key=Path(audio_path).stem))
            weights.append(1.0)

        if not prob_dicts:
            raise RuntimeError("No emotion model could be loaded.")

        probs = average_prob_dicts(prob_dicts, weights)
        return argmax_dict(probs), probs


class ApparentGenderClassifier:
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-24-ft-age-gender", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device, _, _ = choose_device(device)
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class ModelHead(nn.Module):
            def __init__(self, config: Any, num_labels: int) -> None:
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.post_init()

            def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits_age = self.age(hidden_states)
                logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
                return hidden_states, logits_age, logits_gender

        cache_dir = str(ensure_hf_cache_env())
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model = AgeGenderModel.from_pretrained(self.model_name, cache_dir=cache_dir)
        self._model.eval()
        if self.device == "cuda":
            self._model = self._model.to(self.device)

    @staticmethod
    def _window_audio(audio_16k: Any, sr: int = 16000, window_s: float = 3.0, hop_s: float = 1.5) -> List[Any]:
        if audio_16k.size == 0:
            return [audio_16k]
        win = max(int(window_s * sr), 1)
        hop = max(int(hop_s * sr), 1)
        if len(audio_16k) <= win:
            return [audio_16k]

        segments = []
        for start in range(0, len(audio_16k) - win + 1, hop):
            segments.append(audio_16k[start:start + win])
        if not segments:
            segments = [audio_16k]
        return segments

    def predict(self, audio_16k: Any) -> Tuple[str, Dict[str, float]]:
        self._lazy_load()
        import numpy as np
        import torch

        assert self._processor is not None
        assert self._model is not None

        female_scores: List[float] = []
        male_scores: List[float] = []
        child_scores: List[float] = []

        for seg in self._window_audio(audio_16k):
            if seg.size == 0:
                continue
            processed = self._processor(seg, sampling_rate=16000)
            values = processed["input_values"][0]
            tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, -1)
            if self.device == "cuda":
                tensor = tensor.to(self.device)

            with torch.no_grad():
                _emb, _age, gender_logits = self._model(tensor)
            probs = gender_logits.detach().cpu().numpy().reshape(-1).tolist()
            if len(probs) != 3:
                raise RuntimeError(f"Unexpected gender output dimension: {len(probs)}")
            female_scores.append(float(probs[0]))
            male_scores.append(float(probs[1]))
            child_scores.append(float(probs[2]))

        female = safe_mean(female_scores) or 0.0
        male = safe_mean(male_scores) or 0.0
        child = safe_mean(child_scores) or 0.0
        label = "male" if male >= female else "female"
        return label, normalize_probs({"female": female, "male": male, "child": child})


class ProsodyAnalyzer:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds

    def measure_pitch(self, audio: Any, sr: int) -> Tuple[Optional[float], Optional[float], float]:
        import numpy as np
        import librosa

        if audio.size == 0:
            return None, None, 0.0

        fmin = librosa.note_to_hz("C2")
        fmax = librosa.note_to_hz("C7")

        try:
            f0, _voiced_flag, _voiced_prob = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax)
            voiced = np.asarray(f0)[np.isfinite(f0)]
            total_frames = 0 if f0 is None else len(f0)
            voiced_ratio = float(len(voiced) / total_frames) if total_frames > 0 else 0.0
            if voiced.size >= 5:
                return float(np.nanmedian(voiced)), float(np.nanmean(voiced)), voiced_ratio
        except Exception:
            pass

        try:
            f0_yin = librosa.yin(audio, sr=sr, fmin=fmin, fmax=fmax)
            voiced = np.asarray(f0_yin)[np.isfinite(f0_yin) & (np.asarray(f0_yin) > 0)]
            total_frames = len(f0_yin)
            voiced_ratio = float(len(voiced) / total_frames) if total_frames > 0 else 0.0
            if voiced.size >= 5:
                return float(np.nanmedian(voiced)), float(np.nanmean(voiced)), voiced_ratio
        except Exception:
            pass

        return None, None, 0.0

    def measure_speed(self, whisper_result: WhisperResult, audio: Any, sr: int) -> Tuple[Optional[float], float]:
        timed_chunks: List[Tuple[float, float, str]] = []
        for chunk in whisper_result.chunks or []:
            timestamp = chunk.get("timestamp")
            text = str(chunk.get("text", "")).strip()
            if not text or not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            if start is None or end is None:
                continue
            if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
                continue
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            timed_chunks.append((float(start), float(end), text))

        if timed_chunks:
            start = min(x[0] for x in timed_chunks)
            end = max(x[1] for x in timed_chunks)
            span = max(end - start, 1e-6)
            units = len(timed_chunks)
            return float(units / span), float(span)

        text = whisper_result.text.strip()
        token_like_count = len(text.split()) if text else 0
        if token_like_count == 0 and text:
            token_like_count = len(text)
        speech_dur = speech_duration_from_energy(audio, sr)
        if token_like_count == 0 or speech_dur <= 0.0:
            return None, speech_dur
        return float(token_like_count / speech_dur), speech_dur

    def classify_speed(self, units_per_sec: Optional[float]) -> str:
        if units_per_sec is None or not math.isfinite(units_per_sec):
            return "normal"
        if units_per_sec < self.thresholds.speed_slow_normal:
            return "slow"
        if units_per_sec > self.thresholds.speed_normal_fast:
            return "fast"
        return "normal"

    def classify_pitch(self, median_f0_hz: Optional[float], apparent_gender: Optional[str]) -> str:
        if median_f0_hz is None or not math.isfinite(median_f0_hz):
            return "normal"
        low_normal, normal_high = self.thresholds.pitch_pair(apparent_gender)
        if median_f0_hz < low_normal:
            return "low"
        if median_f0_hz > normal_high:
            return "high"
        return "normal"

    def analyze(self, audio: Any, sr: int, whisper_result: WhisperResult, apparent_gender: Optional[str]) -> Tuple[AcousticMeasurements, str, str]:
        median_f0, mean_f0, voiced_ratio = self.measure_pitch(audio, sr)
        units_per_sec, speech_duration_s = self.measure_speed(whisper_result, audio, sr)
        duration_s = float(len(audio) / sr) if sr > 0 else 0.0

        measurements = AcousticMeasurements(
            duration_s=duration_s,
            speech_duration_s=speech_duration_s,
            units_per_sec=units_per_sec,
            median_f0_hz=median_f0,
            mean_f0_hz=mean_f0,
            voiced_ratio=voiced_ratio,
        )
        speed_label = self.classify_speed(units_per_sec)
        pitch_label = self.classify_pitch(median_f0, apparent_gender)
        return measurements, speed_label, pitch_label


class SpeechMetadataClassifier:
    def __init__(
        self,
        device: Optional[str] = None,
        emotion_backend: str = "ensemble",
        thresholds: Optional[ThresholdConfig] = None,
        whisper_model: str = "openai/whisper-large-v3",
    ) -> None:
        self.thresholds = thresholds or ThresholdConfig()
        self.transcriber = WhisperTranscriber(model_name=whisper_model, device=device)
        self.emotion = EmotionPredictor(backend=emotion_backend, device=device)
        self.gender = ApparentGenderClassifier(device=device)
        self.prosody = ProsodyAnalyzer(self.thresholds)

    def predict(self, audio_path: str | Path, language: Optional[str] = None, return_debug: bool = False) -> Tuple[MetadataPrediction, Optional[Dict[str, Any]]]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_orig, sr_orig = load_audio(audio_path, sr=None)
        audio_16k = resample_audio(audio_orig, orig_sr=sr_orig, target_sr=16000)
        audio_16k = trim_silence(audio_16k, sr=16000, top_db=30)

        whisper_result = self.transcriber.transcribe(audio_path, language=language)
        emotion_label, emotion_probs = self.emotion.predict(audio_16k, audio_path=audio_path)
        gender_label, gender_probs = self.gender.predict(audio_16k)
        measurements, speed_label, pitch_label = self.prosody.analyze(
            audio=audio_16k,
            sr=16000,
            whisper_result=whisper_result,
            apparent_gender=gender_label,
        )

        prediction = MetadataPrediction(
            line=whisper_result.text,
            emotion=emotion_label,
            speed=speed_label,
            pitch=pitch_label,
            gender=gender_label,
        )

        if not return_debug:
            return prediction, None

        debug = {
            "emotion_probs": emotion_probs,
            "gender_probs": gender_probs,
            "measurements": asdict(measurements),
            "thresholds": asdict(self.thresholds),
            "whisper_chunk_count": len(whisper_result.chunks),
        }
        return prediction, debug
