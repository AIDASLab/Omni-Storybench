from src.models import SpeechMetadata
import hashlib
import importlib
import os
from typing import Any
import numpy as np
import torch


class T2SGenerator:
    @staticmethod
    def _impl_for(model_name: str) -> str:
        if "parler" in model_name:
            return "parler"
        if "VoxCPM2" in model_name:
            return "voxcpm"
        raise ValueError(f"Unknown T2S model: {model_name}")

    @classmethod
    def signature(cls, model_name: str) -> dict:
        """Everything that determines this generator's output, for cache keying.

        Resolvable without loading the model, so a fully cached run never
        constructs it.
        """
        impl = cls._impl_for(model_name)
        common = {"model": model_name, "impl": impl}

        if impl == "parler":
            return {
                **common,
                "seed": ParlerImpl.SEED,
                "male_voices": ParlerImpl._MALE_VOICES,
                "female_voices": ParlerImpl._FEMALE_VOICES,
            }
        return {
            **common,
            "cfg_value": VoxCPMImpl.CFG_VALUE,
            "inference_timesteps": VoxCPMImpl.INFERENCE_TIMESTEPS,
        }

    def __init__(self, model_name: str, device: str = "cuda"):
        impl = self._impl_for(model_name)
        if impl == "parler":
            self.impl = ParlerImpl(model_name, device)
        else:
            self.impl = VoxCPMImpl(model_name, device)

    def forward(self, speech_metadata: SpeechMetadata, output_path: str):
        self.impl.forward(speech_metadata, output_path)



class ParlerImpl:
    """
    Parler-TTS implementation.

    Uses the Parler-TTS `description` prompt for controllable speech attributes and
    writes a wav file to `output_path`.
    """

    _MODEL_CACHE: dict[tuple[str, str], Any] = {}
    _TOKENIZER_CACHE: dict[str, Any] = {}

    SEED = 0

    _MALE_VOICES = ["Jon", "Gary", "Mike"]
    _FEMALE_VOICES = ["Lea", "Jenna", "Laura"]

    def __init__(self, model_name: str, device: str = "cuda"):
        # Dynamic imports so the module remains importable even if dependencies are installed later.
        parler_mod = importlib.import_module("parler_tts")
        transformers_mod = importlib.import_module("transformers")
        self._install_config_compat_shims()

        ParlerTTSForConditionalGeneration = getattr(parler_mod, "ParlerTTSForConditionalGeneration")
        AutoTokenizer = getattr(transformers_mod, "AutoTokenizer")
        GenerationConfig = getattr(transformers_mod, "GenerationConfig")

        self.model_id = model_name
        self.device = device

        cache_key = (self.model_id, self.device)
        model = self._MODEL_CACHE.get(cache_key)
        if model is None:
            model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_id, revision="refs/pr/9")
            if getattr(model, "generation_config", None) is None:
                try:
                    model.generation_config = GenerationConfig.from_pretrained(
                        self.model_id,
                        revision="refs/pr/9",
                    )
                except Exception:
                    model.generation_config = GenerationConfig.from_model_config(model.config)
            model.to(self.device)
            model.eval()
            self._MODEL_CACHE[cache_key] = model
        self._install_generation_compat_shims(model)
        self.model = model

        tokenizer = self._TOKENIZER_CACHE.get(self.model_id)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._TOKENIZER_CACHE[self.model_id] = tokenizer
        self.tokenizer = tokenizer

    @staticmethod
    def _install_config_compat_shims():
        config_mod = importlib.import_module("parler_tts.configuration_parler_tts")
        ParlerTTSConfig = getattr(config_mod, "ParlerTTSConfig")

        # parler_tts 0.2.2 declares a composite config but does not mark that it
        # cannot be instantiated with empty defaults. transformers >= 4.52 now
        # calls `config.__repr__()` during `from_pretrained()`, which in turn
        # tries `ParlerTTSConfig()` unless this flag is set.
        if not getattr(ParlerTTSConfig, "has_no_defaults_at_init", False):
            ParlerTTSConfig.has_no_defaults_at_init = True

    def forward(self, speech_metadata: SpeechMetadata, output_path: str):
        """Generate speech audio and save it as a wav file."""
        spk = speech_metadata.speaker
        prompt = spk.line
        description = self._build_description(speech_metadata)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Make generation reproducible across runs.
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)

        with torch.inference_mode():
            description_tokens = self.tokenizer(description, return_tensors="pt")
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
            generation = self.model.generate(
                input_ids=description_tokens.input_ids.to(self.device),
                attention_mask=description_tokens.attention_mask.to(self.device),
                prompt_input_ids=prompt_tokens.input_ids.to(self.device),
            )

        audio_arr = generation.cpu().numpy().squeeze()
        self._write_wav(output_path, audio_arr, int(self.model.config.sampling_rate))

    @classmethod
    def _install_generation_compat_shims(cls, model: Any):
        model_cls = type(model)
        if getattr(model_cls, "_omnibench_generation_compat_shims_installed", False):
            return

        generation_utils_mod = importlib.import_module("transformers.generation.utils")
        GenerationMixin = getattr(generation_utils_mod, "GenerationMixin")

        # parler_tts 0.2.2 assumes `PreTrainedModel` still inherits
        # `GenerationMixin`, but transformers 4.52 removed that. Reattach the
        # mixin helpers the custom Parler `generate()` implementation depends on.
        for name, attr in GenerationMixin.__dict__.items():
            if name.startswith("__") or name == "generate" or hasattr(model_cls, name):
                continue
            if callable(attr):
                setattr(model_cls, name, attr)

        original_get_cache = getattr(model_cls, "_get_cache", None)
        original_get_initial_cache_position = getattr(model_cls, "_get_initial_cache_position", None)
        mixin_get_cache = getattr(GenerationMixin, "_get_cache")
        mixin_get_initial_cache_position = getattr(
            GenerationMixin,
            "_get_initial_cache_position",
        )
        original_prepare_attention_mask = getattr(
            GenerationMixin,
            "_prepare_attention_mask_for_generation",
        )

        def compat_prepare_attention_mask_for_generation(
            _self,
            inputs_tensor: torch.Tensor,
            generation_config_or_pad_token_id,
            model_kwargs_or_eos_token_id,
        ):
            # parler_tts 0.2.2 still calls the pre-4.47 transformers helper with
            # `(inputs_tensor, pad_token_id, eos_token_id)`. Newer transformers use
            # `(inputs_tensor, generation_config, model_kwargs)`.
            if hasattr(generation_config_or_pad_token_id, "_pad_token_tensor"):
                return original_prepare_attention_mask(
                    inputs_tensor,
                    generation_config_or_pad_token_id,
                    model_kwargs_or_eos_token_id,
                )

            return cls._prepare_legacy_attention_mask_for_generation(
                inputs_tensor,
                generation_config_or_pad_token_id,
                model_kwargs_or_eos_token_id,
            )

        def compat_get_cache(
            _self,
            cache_implementation: str,
            batch_size_or_max_batch_size: int,
            max_cache_len: int,
            device_or_model_kwargs,
            model_kwargs=None,
        ):
            # parler_tts 0.2.2 still calls `_get_cache` as
            # `(cache_implementation, batch_size, max_cache_len, model_kwargs)`.
            if model_kwargs is None and original_get_cache is not None:
                return original_get_cache(
                    _self,
                    cache_implementation,
                    batch_size_or_max_batch_size,
                    max_cache_len,
                    device_or_model_kwargs,
                )

            return mixin_get_cache(
                _self,
                cache_implementation,
                batch_size_or_max_batch_size,
                max_cache_len,
                device_or_model_kwargs,
                model_kwargs,
            )

        def compat_get_initial_cache_position(
            _self,
            seq_length_or_input_ids,
            device_or_model_kwargs,
            model_kwargs=None,
        ):
            # parler_tts 0.2.2 still defines `_get_initial_cache_position` as
            # `(input_ids, model_kwargs)`. transformers 4.52 now calls it as
            # `(seq_length, device, model_kwargs)` from `GenerationMixin._sample`.
            if model_kwargs is None and original_get_initial_cache_position is not None:
                return original_get_initial_cache_position(
                    _self,
                    seq_length_or_input_ids,
                    device_or_model_kwargs,
                )

            return mixin_get_initial_cache_position(
                _self,
                seq_length_or_input_ids,
                device_or_model_kwargs,
                model_kwargs,
            )

        setattr(
            model_cls,
            "_prepare_attention_mask_for_generation",
            compat_prepare_attention_mask_for_generation,
        )
        setattr(model_cls, "_get_cache", compat_get_cache)
        setattr(
            model_cls,
            "_get_initial_cache_position",
            compat_get_initial_cache_position,
        )
        model_cls._omnibench_generation_compat_shims_installed = True

    @staticmethod
    def _prepare_legacy_attention_mask_for_generation(
        inputs_tensor: torch.Tensor,
        pad_token_id,
        eos_token_id,
    ) -> torch.LongTensor:
        default_attention_mask = torch.ones(
            inputs_tensor.shape[:2],
            dtype=torch.long,
            device=inputs_tensor.device,
        )
        if pad_token_id is None:
            return default_attention_mask
        if inputs_tensor.ndim != 2 or inputs_tensor.is_floating_point():
            return default_attention_mask

        pad_token_tensor = torch.as_tensor(
            pad_token_id,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        ).reshape(-1)
        eos_token_tensor = None
        if eos_token_id is not None:
            eos_token_tensor = torch.as_tensor(
                eos_token_id,
                device=inputs_tensor.device,
                dtype=inputs_tensor.dtype,
            ).reshape(-1)

        is_pad_token_in_inputs = torch.isin(inputs_tensor, pad_token_tensor).any()
        is_pad_token_not_equal_to_eos_token_id = eos_token_tensor is None or not torch.isin(
            eos_token_tensor,
            pad_token_tensor,
        ).any()
        if not bool(is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id):
            return default_attention_mask

        return (~torch.isin(inputs_tensor, pad_token_tensor)).long()

    def _stable_int(self, s: str) -> int:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return int(h, 16)

    def _pick_voice(self, speech_metadata: SpeechMetadata) -> str:
        spk = speech_metadata.speaker
        pool = self._MALE_VOICES if spk.gender == "male" else self._FEMALE_VOICES

        idx = self._stable_int(spk.name or "default") % len(pool)
        return pool[idx]

    def _build_description(self, speech_metadata: SpeechMetadata) -> str:
        spk = speech_metadata.speaker
        voice = self._pick_voice(speech_metadata)

        emotion_map = {
            "angry": "angry and tense",
            "happy": "cheerful and upbeat",
            "neutral": "calm and neutral",
            "sad": "sad and soft",
        }
        speed_map = {
            "slow": "slow",
            "normal": "moderate",
            "fast": "slightly fast",
        }
        pitch_map = {
            "low": "low",
            "normal": "moderate",
            "high": "slightly high",
        }

        emotion_desc = emotion_map.get(spk.emotion, "calm and neutral")
        speed_desc = speed_map.get(spk.speed, "moderate")
        pitch_desc = pitch_map.get(spk.pitch, "moderate")

        # "very clear audio" is recommended by the Parler-TTS model card for best quality.
        return (
            voice
            + "'s voice is "
            + emotion_desc
            + " with a "
            + speed_desc
            + " speaking rate and "
            + pitch_desc
            + " pitch. The recording is very clear audio, close-mic, and has almost no background noise."
        )

    def _write_wav(self, output_path: str, audio_arr, sample_rate: int):
        import soundfile as sf

        sf.write(output_path, audio_arr, sample_rate)



class VoxCPMImpl:
    """
    VoxCPM2-based TTS wrapper.

    Uses a prompt format like:
        "(voice description)actual text"

    This implementation uses voice design only.
    It does not use reference audio or cloning.
    """

    CFG_VALUE = 2.0
    INFERENCE_TIMESTEPS = 10

    _MODEL_CACHE: dict[tuple[str, str], Any] = {}

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

        cache_key = (self.model_name, self.device)
        model = self._MODEL_CACHE.get(cache_key)
        if model is None:
            voxcpm_mod = importlib.import_module("voxcpm")
            VoxCPM = getattr(voxcpm_mod, "VoxCPM")
            model = VoxCPM.from_pretrained(self.model_name, load_denoiser=False)
            self._MODEL_CACHE[cache_key] = model

        self.model = model

    def forward(self, speech_metadata: SpeechMetadata, output_path: str):
        spk = speech_metadata.speaker
        text = getattr(spk, "line", None)

        if not isinstance(text, str) or not text.strip():
            raise ValueError("speaker.line is empty")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        prompt_text = self._build_controlled_text(speech_metadata)

        wav = self.model.generate(
            text=prompt_text,
            cfg_value=self.CFG_VALUE,
            inference_timesteps=self.INFERENCE_TIMESTEPS,
        )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        sample_rate = int(getattr(self.model.tts_model, "sample_rate", 24000))
        self._write_wav(output_path, wav, sample_rate)

    def _build_controlled_text(self, speech_metadata: SpeechMetadata) -> str:
        spk = speech_metadata.speaker
        line = spk.line.strip()
        desc = self._build_voice_description(speech_metadata)

        if not desc:
            return line
        return f"({desc}){line}"

    def _build_voice_description(self, speech_metadata: SpeechMetadata) -> str:
        spk = speech_metadata.speaker

        gender = self._normalize_gender(getattr(spk, "gender", None))
        emotion = self._normalize_emotion(getattr(spk, "emotion", None))
        speed = self._normalize_speed(getattr(spk, "speed", None))
        pitch = self._normalize_pitch(getattr(spk, "pitch", None))

        parts: list[str] = []

        if gender == "male":
            parts.append("a male voice")
        elif gender == "female":
            parts.append("a female voice")

        if emotion == "happy":
            parts.append("cheerful and upbeat tone")
        elif emotion == "sad":
            parts.append("soft and slightly sad tone")
        elif emotion == "angry":
            parts.append("tense and forceful tone")
        elif emotion == "neutral":
            parts.append("calm and neutral tone")

        if speed == "slow":
            parts.append("slow pace")
        elif speed == "normal":
            parts.append("moderate pace")
        elif speed == "fast":
            parts.append("slightly faster pace")

        if pitch == "low":
            parts.append("slightly low-pitched")
        elif pitch == "normal":
            parts.append("moderately pitched")
        elif pitch == "high":
            parts.append("slightly high-pitched")

        parts.append("clear voice")
        return ", ".join(parts)

    @staticmethod
    def _normalize_gender(v) -> str | None:
        if not isinstance(v, str):
            return None
        x = v.strip().lower()
        if x == "male":
            return "male"
        if x == "female":
            return "female"
        return None

    @staticmethod
    def _normalize_emotion(v) -> str | None:
        if not isinstance(v, str):
            return None
        x = v.strip().lower()
        if x in {"happy", "neutral", "sad", "angry"}:
            return x
        return None

    @staticmethod
    def _normalize_speed(v) -> str | None:
        if not isinstance(v, str):
            return None
        x = v.strip().lower()
        if x in {"slow", "normal", "fast"}:
            return x
        return None

    @staticmethod
    def _normalize_pitch(v) -> str | None:
        if not isinstance(v, str):
            return None
        x = v.strip().lower()
        if x in {"low", "normal", "high"}:
            return x
        return None

    @staticmethod
    def _write_wav(output_path: str, audio_arr, sample_rate: int):
        import soundfile as sf
        sf.write(output_path, audio_arr, sample_rate)