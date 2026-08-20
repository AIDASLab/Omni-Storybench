from abc import ABC, abstractmethod
import json
import sys
import os
import re
import yaml
import torch
import numpy as np
from PIL import Image, ImageOps

from pathlib import Path
import gc
import inspect
from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoModel, AutoProcessor, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
from src.models import SpeechMetadata, SpeakerMetadata
from src.models import NextPageOutput

from src.cache import ArtifactCache, DEFAULT_CACHE_ROOT
from src.errors import BackboneOutputError, set_modalities
from src.t2i_generator import T2IGenerator
from src.t2s_generator import T2SGenerator
from src.third_party import prepend_import_path, require_checkout, repository_revision

# ---- Helpers ----

# Portable project locations. Environment variables and CLI flags may override them.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "OMNI_STORYBENCH_OUTPUT",
    str(PROJECT_ROOT / "outputs"),
)

QUOTE_PATTERN = re.compile(
    r"""
    (?:
        "(?P<double>(?:\\.|[^"\\])+)"
      | “(?P<curly_double>[^”]+)”
      | (?<!\w)'(?P<single>(?:\\.|[^'\\]|(?<=\w)'(?=\w))+?)'(?!\w)
      | ‘(?P<curly_single>(?:\\.|[^’\\]|(?<=\w)’(?=\w))+?)’(?!\w)
      | 「(?P<corner>[^」]+)」
      | 『(?P<double_corner>[^』]+)』
    )
    """,
    re.VERBOSE,
)

# ----- Abstract Base Class. Only for their role description and method signatures -----
class Orchestrator(ABC):
    def __init__(self, config_name: str):
        self.config_name = config_name
        if Path(config_name).name != config_name:
            raise ValueError("Config name must be a filename stem, not a path")
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if not config_path.is_file():
            available = ", ".join(path.stem for path in sorted(CONFIG_DIR.glob("*.yaml")))
            raise FileNotFoundError(
                f"Unknown config {config_name!r}. Available configs: {available}"
            )
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        if not isinstance(self.config, dict):
            raise ValueError(f"Config {config_path} must contain a YAML mapping")

        self.set_output_root(self.config.get("output_root") or DEFAULT_OUTPUT_ROOT)
        self.set_cache(ArtifactCache(
            self.config.get("cache_root") or DEFAULT_CACHE_ROOT,
            owner=self.config["output_path"],
        ))

        self._t2i_generator = None
        self._t2s_generator = None

    def set_output_root(self, output_root: str):
        """Point this run at a results root (see DEFAULT_OUTPUT_ROOT)."""
        self.output_root = output_root
        self.output_dir_base = f"{output_root}/{self.config['output_path']}"

    def set_cache(self, cache: ArtifactCache):
        self.cache = cache

    # ----- Generators -----
    # Built on first use so a run whose artifacts are all cached never loads them.

    @property
    def t2i_generator(self) -> T2IGenerator:
        if self._t2i_generator is None:
            self._t2i_generator = T2IGenerator(
                self.config["t2i_model"],
                device=self.config.get("t2i_device", "cuda"),
            )
        return self._t2i_generator

    @property
    def t2s_generator(self) -> T2SGenerator:
        if self._t2s_generator is None:
            self._t2s_generator = T2SGenerator(
                self.config["t2s_model"],
                device=self.config.get("t2s_device", "cuda"),
            )
        return self._t2s_generator

    # ----- Artifact caching -----

    @abstractmethod
    def raw_cache_payload(self) -> dict[str, Any]:
        """Everything determining this sample's backbone output: model and prompt."""

    def load_cached_raw(self) -> Optional[dict[str, Any]]:
        """Restore this sample's raw JSON and text from cache.

        Returns:
            The raw payload on a hit, so the caller can skip the backbone pass;
            None on a miss.
        """
        self._raw_cache_key = self.cache.key({
            "stage": "raw",
            "sample": self.sample_id,
            "orchestrator": type(self).__name__,
            **self.raw_cache_payload(),
        })
        restored = self.cache.restore("raw", self._raw_cache_key, self._raw_cache_files())
        if not restored:
            return None

        with open(self.output_raw_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_raw(self, raw_obj: Any, text: str) -> None:
        """Write this sample's raw JSON and text, and cache them for later runs."""
        raw_json = raw_obj.model_dump_json(indent=4) if hasattr(raw_obj, "model_dump_json") else json.dumps(
            raw_obj, ensure_ascii=False, indent=4
        )
        with open(self.output_raw_path, "w", encoding="utf-8") as f:
            f.write(raw_json)
        with open(self.output_text_path, "w", encoding="utf-8") as f:
            f.write(text)

        # adopt=False: this run's image and speech were just built from raw_obj, so
        # its raw/text on disk must stay that sampling even if a concurrent run won
        # the cache with a different one. Future runs still read the winner.
        self.cache.store("raw", self._raw_cache_key, self._raw_cache_files(), {
            "stage": "raw",
            "sample": self.sample_id,
            "orchestrator": type(self).__name__,
            **self.raw_cache_payload(),
        }, adopt=False)

    def _raw_cache_files(self) -> dict[str, str]:
        return {"raw.json": self.output_raw_path, "text.txt": self.output_text_path}

    def cached_artifact(self, stage: str, payload: dict[str, Any], dest_path: str, produce) -> bool:
        """Restore `stage` from cache, else run `produce(dest_path)` and cache the result.

        Args:
            stage: "image" or "speech".
            payload: Everything determining the artifact, beyond the sample itself.
            dest_path: Where this baseline wants the artifact.
            produce: Callable writing the artifact to the path it is given.

        Returns:
            True if the artifact came from cache.
        """
        full_payload = {"stage": stage, "sample": self.sample_id, **payload}
        key = self.cache.key(full_payload)
        files = {f"artifact{os.path.splitext(dest_path)[1]}": dest_path}

        if self.cache.restore(stage, key, files):
            return True

        # Artifacts are hardlinked from the cache, so a leftover destination must
        # be unlinked rather than written through -- that would edit the entry.
        if os.path.lexists(dest_path):
            os.remove(dest_path)
        try:
            produce(dest_path)
        except Exception as exc:
            # Only this modality is lost. Record it and let the remaining stages
            # run, so one failed image does not also cost us the speech.
            set_modalities(exc, (stage,))
            self.stage_errors.append(exc)
            # A crash mid-write can leave a truncated artifact, which would read
            # as a finished sample to --repair and to the harness.
            if os.path.lexists(dest_path):
                os.remove(dest_path)
            return False

        self.cache.store(stage, key, files, full_payload)
        return False

    def generate_image(self, t2i_prompt: str, reference_image_path: Optional[str] = None) -> bool:
        """Cache-aware image generation through the configured external T2I model."""
        payload = {"producer": T2IGenerator.signature(self.config["t2i_model"]), "prompt": t2i_prompt}
        if T2IGenerator._mode_for(self.config["t2i_model"]) == "kontext":
            # Only Kontext conditions on the previous page image.
            payload["reference_image"] = reference_image_path

        return self.cached_artifact(
            "image",
            payload,
            self.output_image_path,
            lambda path: self.t2i_generator.forward(t2i_prompt, path, reference_image_path),
        )

    def generate_speech(self, speech_metadata: Any, text: str = "") -> bool:
        """Cache-aware speech synthesis through the configured external T2S model.

        Args:
            speech_metadata: Speaker metadata as the backbone returned it.
            text: Page text, used only to report what the backbone gave us.
        """
        try:
            speech_metadata = self._speech_metadata_with_defaults(speech_metadata, text)
        except Exception as exc:
            set_modalities(exc, ("speech",))
            self.stage_errors.append(exc)
            return False

        payload = {
            "producer": T2SGenerator.signature(self.config["t2s_model"]),
            "speech_metadata": speech_metadata.model_dump(),
        }
        return self.cached_artifact(
            "speech",
            payload,
            self.output_speech_path,
            lambda path: self.t2s_generator.forward(speech_metadata, path),
        )

    @staticmethod
    def _prepare_sample_dir(path: str) -> str:
        """Create an artifact directory and clear it.

        The evaluation harness resolves a sample by globbing its index directory,
        so each one must end up holding exactly one artifact. Dropping stale files
        first keeps that true when a sample is regenerated under a different name.
        """
        os.makedirs(path, exist_ok=True)
        for stale_name in os.listdir(path):
            stale_path = os.path.join(path, stale_name)
            if os.path.isfile(stale_path):
                os.remove(stale_path)
        return path

    @abstractmethod
    def format_input(self, entry: dict[str, Any]):
        """
        Save model input as instance variable(self.input)
        In base class, it creates the output directories and the output paths.
        Args:
            entry: Single sample from src.dataset.load_dataset
        """
        # Reset first: run.py reads this even when the rest of setup fails.
        self.stage_errors: list[BaseException] = []

        index = entry["index"]
        self.sample_id = entry["id"]

        self.output_text_dir = self._prepare_sample_dir(f"{self.output_dir_base}/text/{index}")
        self.output_image_dir = self._prepare_sample_dir(f"{self.output_dir_base}/image/{index}")
        self.output_speech_dir = self._prepare_sample_dir(f"{self.output_dir_base}/speech/{index}")
        # Not read by the harness; kept alongside for debugging and speech re-synthesis.
        self.output_raw_dir = self._prepare_sample_dir(f"{self.output_dir_base}/raw/{index}")

        self.output_text_path = f"{self.output_text_dir}/generated.txt"
        self.output_image_path = f"{self.output_image_dir}/generated.png"
        self.output_speech_path = f"{self.output_speech_dir}/generated.wav"
        self.output_raw_path = f"{self.output_raw_dir}/generated_raw.json"

    @staticmethod
    def _strip_wrapping_quotes(value: Any) -> str:
        text = str(value or "").strip()
        quote_pairs = [
            ('"', '"'),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
            ("「", "」"),
            ("『", "』"),
        ]

        changed = True
        while changed and len(text) >= 2:
            changed = False
            for left, right in quote_pairs:
                if text.startswith(left) and text.endswith(right):
                    text = text[len(left):-len(right)].strip()
                    changed = True
                    break
        return text

    @staticmethod
    def _non_empty_string(value: Any, default: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

    def _extract_speech_line_from_text(self, text: str) -> str:
        """Return the quote inside `text`, or "" when it holds none."""
        match = QUOTE_PATTERN.search(text or "")
        if match is not None:
            for group in match.groups():
                line = self._strip_wrapping_quotes(group)
                if line:
                    return line
        return ""

    @staticmethod
    def _raw_text_values(raw_text: Any) -> list[str]:
        if raw_text is None:
            return []
        if isinstance(raw_text, str):
            return [raw_text]
        if isinstance(raw_text, list):
            values = []
            for item in raw_text:
                values.extend(Orchestrator._raw_text_values(item))
            return values
        if isinstance(raw_text, dict):
            values = []
            for key in ("text", "content", "reasoning", "reasoning_content"):
                if key in raw_text:
                    values.extend(Orchestrator._raw_text_values(raw_text[key]))
            if any(str(value or "").strip() for value in values):
                return values
            try:
                return [json.dumps(raw_text, ensure_ascii=False)]
            except TypeError:
                return [str(raw_text)]
        if hasattr(raw_text, "model_dump"):
            try:
                return Orchestrator._raw_text_values(raw_text.model_dump())
            except Exception:
                pass
        return [str(raw_text)]

    @staticmethod
    def _balanced_json_candidates(raw: str) -> list[str]:
        candidates = []
        start = None
        depth = 0
        in_string = False
        escape = False

        for idx, char in enumerate(raw):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start:idx + 1].strip())
                    start = None

        return candidates

    @staticmethod
    def _json_candidates(raw_text: Any) -> list[str]:
        candidates = []
        for raw_value in Orchestrator._raw_text_values(raw_text):
            raw = str(raw_value or "").strip()
            if not raw:
                continue

            candidates.append(raw)
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
            if fence:
                candidates.append(fence.group(1).strip())

            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                candidates.append(raw[start:end + 1].strip())

            candidates.extend(Orchestrator._balanced_json_candidates(raw))

        deduped = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    def _coerce_next_page_output(self, parsed: Any, raw_text: Any, source: str) -> NextPageOutput:
        if isinstance(parsed, NextPageOutput):
            return parsed

        parsed_error = None
        if parsed is not None:
            try:
                return NextPageOutput.model_validate(parsed)
            except Exception as exc:
                parsed_error = exc

        raw_error = None
        for candidate in self._json_candidates(raw_text):
            try:
                return NextPageOutput.model_validate_json(candidate)
            except Exception as exc:
                raw_error = exc
                try:
                    return NextPageOutput.model_validate(json.loads(candidate))
                except Exception as nested_exc:
                    raw_error = nested_exc

        preview_values = self._raw_text_values(raw_text)
        preview_source = "\n".join(preview_values) if preview_values else str(raw_text or "")
        preview = preview_source.replace("\n", "\\n")[:1000]
        detail = parsed_error or raw_error
        suffix = f" Last parser error: {detail}" if detail is not None else ""
        raise ValueError(
            f"{source} returned no parseable NextPageOutput. "
            f"Raw response preview: {preview!r}.{suffix}"
        )

    def _speech_metadata_with_defaults(self, speech_metadata: Any, text: str) -> SpeechMetadata:
        if hasattr(speech_metadata, "model_dump"):
            obj = speech_metadata.model_dump()
        elif hasattr(speech_metadata, "dict"):
            obj = speech_metadata.dict()
        elif isinstance(speech_metadata, dict):
            obj = speech_metadata
        else:
            obj = {}

        spk = obj.get("speaker", {}) if isinstance(obj, dict) else {}
        if not isinstance(spk, dict):
            spk = {}

        default_speaker = {
            "name": self.config.get("default_speaker_name", "Narrator"),
            "emotion": self.config.get("default_speaker_emotion", "neutral"),
            "speed": self.config.get("default_speaker_speed", "normal"),
            "pitch": self.config.get("default_speaker_pitch", "normal"),
            "gender": self.config.get("default_speaker_gender", "female"),
        }

        line = self._strip_wrapping_quotes(spk.get("line"))
        if not line:
            raise set_modalities(
                BackboneOutputError(
                    "Backbone output has no speech line to synthesize.",
                    raw_output=text,
                ),
                ("speech",),
            )

        return SpeechMetadata(
            speaker=SpeakerMetadata(
                name=self._non_empty_string(spk.get("name"), default_speaker["name"]),
                line=line,
                emotion=self._non_empty_string(spk.get("emotion"), default_speaker["emotion"]),
                speed=self._non_empty_string(spk.get("speed"), default_speaker["speed"]),
                pitch=self._non_empty_string(spk.get("pitch"), default_speaker["pitch"]),
                gender=self._non_empty_string(spk.get("gender"), default_speaker["gender"]),
            )
        )
        
    @abstractmethod
    def forward(self):
        # Run model (need to consider batching)
        
        # Save output to path
        pass


# ----- T2T Orchestators ------
class OpenAIOrchestrator(Orchestrator):
    SYSTEM_PROMPT = """
    You are an AI storybook next-page generator. Given the previous page text and image, and the next-page condition below, generate the next page text, image, and speech.
    Requirements:
    - Next page text should be coherent and consistent with the story, while keeping concise.
    - Next page image should match the described scene and characters.
    - Text must contain exactly one dialogue quote.
    - The provided condition may include scene, characters, actions, emotions, and ambient sound.
    Answer should be in the following format:
    - "text" corresponds to the text of the next page.
    - "t2i" corresponds to the prompt to the image generation model which will be used to generate the image of the next page.
    - "speech_metadata" corresponds to the speaker's dialogue line and its metadata (name, line, emotion, speed, pitch, gender).
       - "emotion" should be one of the following: 'angry', 'happy', 'neutral', 'sad'
       - "speed" should be one of the following: 'normal', 'slow', 'fast'
       - "pitch" should be one of the following: 'normal', 'high', 'low'
       - "gender" should be one of the following: 'male', 'female'
    Only return the JSON, nothing else.
    """

    PROMPT = """
    Story metadata:
    - {metadata}

    Previous page text:
    - {current_page_text}

    Next page condition:
    - {next_page_condition}
    """

    def __init__(self, config_name: str):
        super().__init__(config_name)
        from openai import OpenAI
        self.client = OpenAI(max_retries=int(self.config.get("api_max_retries", 0)))
        self.model = self.config["openai_model"]


    def _create_file(self, file_path: str) -> str:
        with open(file_path, "rb") as file_content:
            result = self.client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)
        
        metadata, current_page, next_page_condition = entry["book_metadata"], entry["current_page"], entry["next_page_condition"]

        current_page_text = current_page["text"]
        self.image_path = current_page["image_path"]
        self.prompt = self.PROMPT.format(metadata=metadata, current_page_text=current_page_text, next_page_condition=next_page_condition)

    def raw_cache_payload(self) -> dict[str, Any]:
        return {"model": self.model, "system": self.SYSTEM_PROMPT, "prompt": self.prompt}

    def forward(self):
        from src.models import NextPageOutput

        cached = self.load_cached_raw()
        if cached is not None:
            output = NextPageOutput.model_validate(cached)
        else:
            # OpenAI API call
            image_file_id = self._create_file(self.image_path)
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "input_text", "text": self.prompt},
                            {"type": "input_image", "file_id": image_file_id},
                        ]},
                    ],
                    text_format=NextPageOutput,
                )
            finally:
                try:
                    self.client.files.delete(image_file_id)
                except Exception as exc:
                    print(
                        f"[WARN] Failed to delete temporary OpenAI file {image_file_id}: {exc}",
                        file=sys.stderr,
                    )
            output = self._coerce_next_page_output(
                getattr(response, "output_parsed", None),
                getattr(response, "output_text", None),
                "OpenAI",
            )
            self.save_raw(output, output.text)

        # T2I generation
        self.generate_image(output.t2i, self.image_path)

        # T2S generation
        self.generate_speech(output.speech_metadata, output.text)


class ClaudeOrchestrator(Orchestrator):
    SYSTEM_PROMPT = OpenAIOrchestrator.SYSTEM_PROMPT
    PROMPT = OpenAIOrchestrator.PROMPT

    def __init__(self, config_name: str):
        super().__init__(config_name)
        from anthropic import Anthropic
        self.client = Anthropic(max_retries=int(self.config.get("api_max_retries", 0)))
        self.model = self.config.get("claude_model", "claude-opus-4-7")
        self.max_tokens = self.config.get("claude_max_tokens", 2048)
        self.image_max_bytes = int(self.config.get("claude_image_max_bytes", 5 * 1024 * 1024))
        self.image_target_bytes = max(
            1,
            min(
                self.image_max_bytes,
                int(self.config.get("claude_image_target_bytes", self.image_max_bytes - 128 * 1024)),
            ),
        )
        self.image_min_side = int(self.config.get("claude_image_min_side", 512))


    @staticmethod
    def _serialize_image(image: Image.Image, format_name: str, **save_kwargs: Any) -> bytes:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format=format_name, **save_kwargs)
        return buffer.getvalue()

    @staticmethod
    def _base64_size(num_bytes: int) -> int:
        return 4 * ((num_bytes + 2) // 3)

    @staticmethod
    def _prepare_image_for_jpeg(image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image

        if image.mode in {"RGBA", "LA"} or "A" in image.getbands() or image.info.get("transparency") is not None:
            rgba_image = image.convert("RGBA")
            background = Image.new("RGB", rgba_image.size, (255, 255, 255))
            background.paste(rgba_image, mask=rgba_image.getchannel("A"))
            return background

        return image.convert("RGB")

    def _encode_image(self, path: str) -> tuple[str, str]:
        """Return (media_type, base64_data) for the image at path."""
        import base64
        import math
        import mimetypes

        media_type, _ = mimetypes.guess_type(path)
        if media_type is None:
            media_type = "image/png"

        with open(path, "rb") as f:
            raw_bytes = f.read()

        if self._base64_size(len(raw_bytes)) <= self.image_max_bytes:
            return media_type, base64.standard_b64encode(raw_bytes).decode("ascii")

        with Image.open(path) as source_image:
            image = ImageOps.exif_transpose(source_image)
            image.load()

        if media_type == "image/png":
            optimized_png = self._serialize_image(image, "PNG", optimize=True)
            if self._base64_size(len(optimized_png)) <= self.image_max_bytes:
                print(
                    f"[Claude] Re-encoded {os.path.basename(path)}: "
                    f"{len(raw_bytes)} raw bytes / {self._base64_size(len(raw_bytes))} base64 bytes -> "
                    f"{len(optimized_png)} raw bytes / {self._base64_size(len(optimized_png))} base64 bytes",
                    file=sys.stderr,
                )
                return "image/png", base64.standard_b64encode(optimized_png).decode("ascii")

        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        candidate_image = image
        best_jpeg = b""
        jpeg_qualities = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40)

        while True:
            jpeg_image = self._prepare_image_for_jpeg(candidate_image)
            for quality in jpeg_qualities:
                candidate_bytes = self._serialize_image(
                    jpeg_image,
                    "JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
                if not best_jpeg or len(candidate_bytes) < len(best_jpeg):
                    best_jpeg = candidate_bytes
                if self._base64_size(len(candidate_bytes)) <= self.image_max_bytes:
                    print(
                        f"[Claude] Re-encoded {os.path.basename(path)}: "
                        f"{len(raw_bytes)} raw bytes / {self._base64_size(len(raw_bytes))} base64 bytes -> "
                        f"{len(candidate_bytes)} raw bytes / {self._base64_size(len(candidate_bytes))} base64 bytes",
                        file=sys.stderr,
                    )
                    return "image/jpeg", base64.standard_b64encode(candidate_bytes).decode("ascii")

            if max(candidate_image.size) <= self.image_min_side:
                break

            shrink_ratio = math.sqrt(
                self.image_target_bytes / max(self._base64_size(len(best_jpeg)), 1)
            )
            shrink_ratio = min(0.95, shrink_ratio)
            if shrink_ratio >= 1.0:
                shrink_ratio = 0.95

            new_size = (
                max(1, int(candidate_image.width * shrink_ratio)),
                max(1, int(candidate_image.height * shrink_ratio)),
            )
            if new_size == candidate_image.size:
                break

            candidate_image = candidate_image.resize(new_size, resample=resample)

        raise ValueError(
            f"Unable to shrink {path} below Claude's image limit of {self.image_max_bytes} bytes "
            f"(best effort: {len(best_jpeg) or len(raw_bytes)} raw bytes / "
            f"{self._base64_size(len(best_jpeg) or len(raw_bytes))} base64 bytes)."
        )

    @staticmethod
    def _enforce_strict_schema(schema: dict) -> dict:
        """Recursively set additionalProperties=False on every object node.

        Anthropic's json_schema structured outputs require every object
        (including those in $defs) to explicitly forbid extra properties.
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema["additionalProperties"] = False
            for v in schema.values():
                if isinstance(v, dict):
                    ClaudeOrchestrator._enforce_strict_schema(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            ClaudeOrchestrator._enforce_strict_schema(item)
        return schema

    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)

        metadata, current_page, next_page_condition = entry["book_metadata"], entry["current_page"], entry["next_page_condition"]

        current_page_text = current_page["text"]
        self.image_path = current_page["image_path"]
        self.prompt = self.PROMPT.format(metadata=metadata, current_page_text=current_page_text, next_page_condition=next_page_condition)

    def raw_cache_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": self.SYSTEM_PROMPT,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
        }

    def forward(self):
        cached = self.load_cached_raw()
        if cached is not None:
            output = NextPageOutput.model_validate(cached)
        else:
            media_type, image_data = self._encode_image(self.image_path)
            schema = self._enforce_strict_schema(NextPageOutput.model_json_schema())
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                        {"type": "text", "text": self.prompt},
                    ]},
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    },
                },
            )
            raw_json = response.content[0].text
            output = self._coerce_next_page_output(None, raw_json, "Claude")
            self.save_raw(output, output.text)

        # T2I generation
        self.generate_image(output.t2i, self.image_path)

        # T2S generation
        self.generate_speech(output.speech_metadata, output.text)


class VLLMOrchestrator(Orchestrator):
    """Orchestrator that talks to a local vLLM server via its OpenAI-compatible HTTP API.

    Uses vLLM's grammar-constrained structured output and validates the returned
    JSON locally instead of relying on the OpenAI SDK's parse helper.
    """
    SYSTEM_PROMPT = OpenAIOrchestrator.SYSTEM_PROMPT
    PROMPT = OpenAIOrchestrator.PROMPT

    def __init__(self, config_name: str):
        super().__init__(config_name)
        from openai import OpenAI
        self.client = OpenAI(
            base_url=self.config["vllm_base_url"],
            api_key=self.config.get("vllm_api_key", "EMPTY"),
            max_retries=int(self.config.get("api_max_retries", 0)),
        )
        self.model = self.config["vllm_model"]
        self.max_tokens = self.config.get("vllm_max_tokens", 2048)
        self.parse_retries = int(self.config.get("vllm_parse_retries", 0))
        self.fallback_without_response_format = self.config.get("vllm_fallback_without_response_format", False) is True


    def _encode_image(self, path: str) -> tuple[str, str]:
        """Return (media_type, base64_data) for the image at path."""
        import base64
        import mimetypes
        media_type, _ = mimetypes.guess_type(path)
        if media_type is None:
            media_type = "image/png"
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("ascii")
        return media_type, data

    @staticmethod
    def _message_extra_value(message: Any, key: str) -> Any:
        if hasattr(message, key):
            value = getattr(message, key)
            if value is not None:
                return value

        value = getattr(message, key, None)
        if value is not None:
            return value

        for extra_attr in ("model_extra", "__pydantic_extra__"):
            extra = getattr(message, extra_attr, None)
            if isinstance(extra, dict) and key in extra:
                return extra[key]

        if hasattr(message, "model_dump"):
            try:
                dumped = message.model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict) and key in dumped:
                return dumped[key]

        return None

    def _vllm_extra_body(self) -> dict[str, Any]:
        configured = self.config.get("vllm_extra_body") or {}
        if not isinstance(configured, dict):
            raise TypeError("vllm_extra_body must be a mapping if provided.")

        extra_body = dict(configured)

        chat_template_kwargs = self.config.get("vllm_chat_template_kwargs")
        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, dict):
                raise TypeError("vllm_chat_template_kwargs must be a mapping if provided.")
            merged_chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            merged_chat_template_kwargs.update(chat_template_kwargs)
            extra_body["chat_template_kwargs"] = merged_chat_template_kwargs

        return extra_body

    def _vllm_message_text(self, message: Any) -> list[Any]:
        values = []
        for key in ("content", "reasoning", "reasoning_content", "refusal", "tool_calls"):
            value = self._message_extra_value(message, key)
            if value is not None:
                values.append(value)

        if any(str(value or "").strip() for value in self._raw_text_values(values)):
            return values

        if hasattr(message, "model_dump"):
            try:
                return [message.model_dump()]
            except Exception:
                pass
        return [str(message)]

    @staticmethod
    def _vllm_debug_value(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                pass
        return value

    @classmethod
    def _vllm_debug_preview(cls, value: Any, limit: int = 2000) -> str:
        value = cls._vllm_debug_value(value)
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
        return text.replace("\n", "\\n")[:limit]

    def _vllm_response_debug(self, response: Any, message: Any, finish_reason: Any, attempt: str) -> str:
        payload = {
            "attempt": attempt,
            "finish_reason": finish_reason,
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": self._vllm_debug_value(getattr(response, "usage", None)),
            "message": self._vllm_debug_value(message),
        }
        return self._vllm_debug_preview(payload)

    def _length_stop_error(self, raw_text: Any, debug: str, parse_error: Exception) -> BackboneOutputError:
        preview_values = self._raw_text_values(raw_text)
        preview_source = "\n".join(preview_values)
        preview = preview_source.replace("\n", "\\n")[:4000]

        return BackboneOutputError(
            f"vLLM reached max_tokens ({self.max_tokens}) before producing a complete "
            f"NextPageOutput JSON object. Parse error: {parse_error}",
            raw_output=f"{preview}\n\nresponse metadata: {debug}",
        )

    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)

        metadata, current_page, next_page_condition = entry["book_metadata"], entry["current_page"], entry["next_page_condition"]

        current_page_text = current_page["text"]
        self.image_path = current_page["image_path"]
        self.prompt = self.PROMPT.format(metadata=metadata, current_page_text=current_page_text, next_page_condition=next_page_condition)

    def raw_cache_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": self.SYSTEM_PROMPT,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "extra_body": self._vllm_extra_body(),
        }

    def forward(self):
        cached = self.load_cached_raw()
        if cached is not None:
            output = NextPageOutput.model_validate(cached)
            self.generate_image(output.t2i, self.image_path)
            self.generate_speech(output.speech_metadata, output.text)
            return

        media_type, image_data = self._encode_image(self.image_path)
        data_url = f"data:{media_type};base64,{image_data}"

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "NextPageOutput",
                "schema": NextPageOutput.model_json_schema(),
                "strict": True,
            },
        }

        request_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            "extra_body": self._vllm_extra_body(),
        }

        attempts: list[tuple[str, Optional[dict[str, Any]]]] = [
            (f"structured #{idx + 1}", response_format)
            for idx in range(self.parse_retries + 1)
        ]
        if self.fallback_without_response_format:
            attempts.append(("unconstrained JSON fallback", None))

        output = None
        last_error = None
        last_debug = ""
        for attempt_idx, (attempt_name, attempt_response_format) in enumerate(attempts, start=1):
            call_kwargs = dict(request_kwargs)
            if attempt_response_format is not None:
                call_kwargs["response_format"] = attempt_response_format

            response = self.client.chat.completions.create(**call_kwargs)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError(
                    "vLLM returned no choices. "
                    f"Response metadata: {self._vllm_debug_preview(response)}."
                )

            choice = choices[0]
            message = getattr(choice, "message", None)
            finish_reason = getattr(choice, "finish_reason", None)
            raw_text = self._vllm_message_text(message)
            last_debug = self._vllm_response_debug(response, message, finish_reason, attempt_name)

            try:
                output = self._coerce_next_page_output(None, raw_text, "vLLM")
                break
            except ValueError as exc:
                last_error = exc
                if finish_reason == "length":
                    raise self._length_stop_error(raw_text, last_debug, exc) from exc
                if attempt_idx < len(attempts):
                    next_attempt = attempts[attempt_idx][0]
                    print(
                        "[WARNING] "
                        f"vLLM {attempt_name} returned no parseable NextPageOutput; "
                        f"retrying with {next_attempt}. {str(exc)[:500]}",
                        file=sys.stderr,
                    )
                    continue
                raise ValueError(f"{exc} Response metadata: {last_debug}.") from exc

        if output is None:
            raise ValueError(f"vLLM returned no parseable NextPageOutput. Last error: {last_error}")

        self.save_raw(output, output.text)

        # T2I generation
        self.generate_image(output.t2i, self.image_path)

        # T2S generation
        self.generate_speech(output.speech_metadata, output.text)


class EmovaOrchestrator(Orchestrator):
    PASS1_PROMPT = """
    You are an AI storybook next-page generator. Given the previous page text and image, and the next-page condition below, generate the next page text, image, and dialogue quote.
    Requirements:
    - Next page text should be coherent and consistent with the story, while keeping concise.
    - Next page image should match the described scene and characters.
    - Quote should be a concise dialogue quote suitable for the next page, consistent with next page text and next page condition.
    - The provided condition may include scene, characters, actions, emotions, and ambient sound.
    Answer should be in the following format:
    - "text" corresponds to the text of the next page.
    - "quote" corresponds to the dialogue quote in the next page text.
    - "t2i" corresponds to the prompt to the image generation model which will be used to generate the image of the next page.
    Only return the JSON, nothing else.

    Story metadata:
    - {metadata}

    Previous page text:
    - {current_page_text}

    Next page condition:
    - {next_page_condition}
    """

    MODEL_ID = "Emova-ollm/emova-qwen-2-5-7b-hf"
    SPEECH_TOKENIZER_ID = "Emova-ollm/emova_speech_tokenizer_hf"

    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.orch_device = torch.device(self.config["orch_device"])
        self.emova_cfg = AutoConfig.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            config=self.emova_cfg,
            torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(self.orch_device)
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        self.speeck_tokenizer = AutoModel.from_pretrained(self.SPEECH_TOKENIZER_ID, torch_dtype=torch.float32, trust_remote_code=True).eval().to(self.orch_device)
        self.processor.set_speech_tokenizer(self.speeck_tokenizer)
        self._emova_t2s_allowed_token_ids = self._build_emova_t2s_allowed_token_ids()


    def _build_emova_t2s_allowed_token_ids(self) -> list[int]:
        token_ids = []
        for unit_id in range(4096):
            token_id = self.processor.tokenizer.convert_tokens_to_ids(f"<|speech_{unit_id}|>")
            if isinstance(token_id, int):
                token_ids.append(token_id)

        eos_token_id = self.processor.tokenizer.eos_token_id
        if eos_token_id is not None:
            token_ids.append(eos_token_id)

        if len(token_ids) <= 1:
            raise ValueError("EMOVA tokenizer does not expose speech unit tokens")
        return token_ids

    def _emova_t2s_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor) -> list[int]:
        return self._emova_t2s_allowed_token_ids

    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)

        metadata, current_page, next_page_condition = entry["book_metadata"], entry["current_page"], entry["next_page_condition"]

        current_page_text = current_page["text"]
        self.resolved_img_path = current_page["image_path"]
        self.pass1_prompt = self.PASS1_PROMPT.format(metadata=metadata, current_page_text=current_page_text, next_page_condition=next_page_condition)

    def _parse_json_best_effort(self, s: str) -> dict[str, Any]:
        """Best-effort JSON extraction from model output.

        Raises:
            BackboneOutputError: Nothing parseable in the output.
        """
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r > l:
            try:
                return json.loads(s[l : r + 1])
            except json.JSONDecodeError:
                pass
        raise BackboneOutputError("EMOVA returned no parseable JSON object.", raw_output=s)

    def raw_cache_payload(self) -> dict[str, Any]:
        return {"model": self.MODEL_ID, "prompt": self.pass1_prompt}

    def _run_pass1(self) -> dict[str, Any]:
        pass1_inputs = dict(
            text=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.pass1_prompt}]},
            ],
            images=Image.open(self.resolved_img_path),
        )
        pass1_inputs = self.processor(**pass1_inputs, return_tensors="pt")
        pass1_inputs = pass1_inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**pass1_inputs, max_new_tokens=1024, do_sample=False, no_repeat_ngram_size=6)
            outputs = outputs[:, pass1_inputs["input_ids"].shape[1]:]
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        parsed = self._parse_json_best_effort(decoded)
        missing = [field for field in ("text", "t2i", "quote") if not parsed.get(field)]
        if missing:
            raise BackboneOutputError(
                f"EMOVA output is missing required field(s): {', '.join(missing)}.",
                raw_output=decoded,
            )

        return {
            "text": parsed["text"],
            "t2i": parsed["t2i"],
            "quote": parsed["quote"],
            "raw_decoded": decoded,
        }

    def _run_pass2(self, quote: str, out_path: str):
        """EMOVA speaks the quote itself, so speech is a backbone artifact here."""
        t2s_prompt = f"Please synthesize the speech corresponding to the follwing text.\n{quote}"
        t2s_inputs = dict(
            text=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": t2s_prompt}]},
            ]
        )
        t2s_inputs = self.processor(**t2s_inputs, return_tensors="pt")
        t2s_inputs = t2s_inputs.to(self.model.device)

        wav_prefix = out_path[:-4]
        with torch.no_grad():
            t2s_out = self.model.generate(
                **t2s_inputs,
                max_new_tokens=4096,
                do_sample=False,
                prefix_allowed_tokens_fn=self._emova_t2s_allowed_tokens,
            )
            t2s_out = t2s_out[:, t2s_inputs["input_ids"].shape[1]:]
            self.processor.batch_decode(
                t2s_out, skip_special_tokens=True,
                speaker="female", output_wav_prefix=wav_prefix,
            )

        generated_wav = f"{wav_prefix}_0.wav"
        if os.path.exists(generated_wav) and generated_wav != out_path:
            os.rename(generated_wav, out_path)

    def forward(self):
        # 1) emova pass 1: Generate next page text and t2i prompt
        raw = self.load_cached_raw()
        if raw is None:
            raw = self._run_pass1()
            self.save_raw(raw, raw["text"])

        t2i, quote = raw["t2i"], raw["quote"]

        # 2) emova pass 2: Generate speech in T2S style from parsing the quote in next page text
        self.cached_artifact(
            "speech",
            {"producer": {"model": self.MODEL_ID, "speaker": "female"}, "quote": quote},
            self.output_speech_path,
            lambda path: self._run_pass2(quote, path),
        )

        # 3) T2I generation
        self.generate_image(t2i, self.resolved_img_path)


# ----- T2TS Orchestrators -----
class Qwen2_5OmniOrchestrator(Orchestrator):
    SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    PASS1_PROMPT = """
    You are generating only a spoken quote for the next storybook page.
    Requirements:
    - Generate exactly one concise dialogue quote suitable for the next page.
    - Do not generate narration.
    - The quote must match the story context and next page condition.
    - Output only one line containing only the quote text (no JSON, no speaker tags, no extra explanation).
    - Do not output patterns like: "Rabbit said ...", "The rabbit shouted ...", or scene descriptions.
    Good examples:
    - Help! My ears are growing too big!
    - I wish I could join the party too.

    Story metadata:
    - {metadata}

    Previous page text:
    - {current_page_text}

    Next page condition:
    - {next_page_condition}
    """

    PASS2_PROMPT = """
    You are an AI storybook next-page generator. Given the previous page text and image, and the next-page condition below, generate the next page text and image.
    Requirements:
    - Next page text should be coherent and consistent with the story, while keeping concise.
    - Next page image should match the described scene and characters.
    - The provided condition may include scene, characters, actions, emotions, and ambient sound.
    - A quote for the next page is provided as text and audio.
    - The generated next page text and t2i must be consistent with this provided quote.
    - Next page text must contain exactly one dialogue quote, and it must be the provided quote.
    Answer should be in the following format:
    - "text" corresponds to the text of the next page.
    - "t2i" corresponds to the prompt to the image generation model which will be used to generate the image of the next page.
    Only return the JSON, nothing else

    Story metadata:
    - {metadata}

    Previous page text:
    - {current_page_text}

    Next page condition:
    - {next_page_condition}

    Pass 1 quote:
    - {pass1_quote}
    """
    
    MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

    def __init__(self, config_name: str):
        super().__init__(config_name)
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map=self.config["orch_device"],
            attn_implementation="flash_attention_2",
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.MODEL_ID)


    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)
        
        metadata, current_page, next_page_condition = entry["book_metadata"], entry["current_page"], entry["next_page_condition"]

        current_page_text = current_page["text"]
        self.metadata = metadata
        self.current_page_text = current_page_text
        self.next_page_condition = next_page_condition
        self.resolved_img_path = current_page["image_path"]
        self.pass1_prompt = self.PASS1_PROMPT.format(
            metadata=self.metadata,
            current_page_text=self.current_page_text,
            next_page_condition=self.next_page_condition,
        )

    def _build_inputs(self, messages: list[dict[str, Any]]):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        from qwen_omni_utils import process_mm_info
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        return inputs.to(self.model.device).to(self.model.dtype)

    def _parse_assistant_json(self, asst_text: str) -> Dict[str, Any] | None:
        def _decode_json_string(s: str) -> str:
            try:
                return json.loads(f"\"{s}\"")
            except json.JSONDecodeError:
                return s

        def _extract_text_t2i_from_malformed(s: str) -> Dict[str, Any] | None:
            text_key_match = re.search(
                r'["\']?text["\']?\s*:\s*"((?:\\.|[^"\\])*)"',
                s,
                flags=re.DOTALL,
            )
            t2i_key_match = re.search(
                r'["\']?t2i["\']?\s*:\s*"((?:\\.|[^"\\])*)"',
                s,
                flags=re.DOTALL,
            )
            if text_key_match and t2i_key_match:
                return {
                    "text": _decode_json_string(text_key_match.group(1)).strip(),
                    "t2i": _decode_json_string(t2i_key_match.group(1)).strip(),
                }

            if t2i_key_match:
                t2i = _decode_json_string(t2i_key_match.group(1)).strip()
                prefix = s[:t2i_key_match.start()]

                # Handles malformed outputs like: "<text>"."t2i": "<prompt>"
                quoted_prefix = re.search(r'"((?:\\.|[^"\\])*)"', prefix, flags=re.DOTALL)
                if quoted_prefix:
                    text = _decode_json_string(quoted_prefix.group(1)).strip()
                    return {"text": text, "t2i": t2i}
            return None

        asst_text = asst_text.strip()

        try:
            return json.loads(asst_text)
        except json.JSONDecodeError:
            pass

        # Strip fenced wrappers, if any.
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", asst_text, flags=re.IGNORECASE)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                asst_text = fence_match.group(1).strip()

        # Try parsing a detected JSON object substring.
        match = re.search(r"\{[\s\S]*\}", asst_text)
        if match is not None:
            candidate = match.group(0).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                recovered = _extract_text_t2i_from_malformed(candidate)
                if recovered is not None:
                    return recovered

        # Try recovery directly on full assistant output.
        recovered = _extract_text_t2i_from_malformed(asst_text)
        if recovered is not None:
            return recovered

        # Try wrapping bare key-value JSON fragments (missing outer braces).
        wrapped = asst_text.strip().strip(",")
        if wrapped and not wrapped.startswith("{") and ("\"text\"" in wrapped or "\"t2i\"" in wrapped):
            try:
                return json.loads("{" + wrapped + "}")
            except json.JSONDecodeError:
                pass

        return None

    def _decode_assistant_text(self, text_ids: torch.Tensor) -> str:
        texts = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(texts) == 1, "Batch size should be 1"
        return texts[0].split("assistant\n")[-1].strip()

    def _normalize_quote_text(self, raw_text: str) -> str:
        quote = raw_text.strip()
        if "\n" in quote:
            lines = [line.strip() for line in quote.splitlines() if line.strip()]
            if lines:
                quote = lines[0]
        quote = quote.strip().strip('"').strip("'").strip()
        return quote

    def _synthesize_audio_from_text(self, transcript: str) -> torch.Tensor:
        tts_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": transcript}
                ]
            }
        ]

        tts_text = self.processor.apply_chat_template(tts_messages, add_generation_prompt=True, tokenize=False)
        tts_inputs = self.processor(text=tts_text, return_tensors="pt", padding=True)
        tts_inputs = tts_inputs.to(self.model.device).to(self.model.dtype)

        prompt_len = tts_inputs["input_ids"].shape[1]
        target_ids = self.processor.tokenizer.encode(transcript, add_special_tokens=False)
        eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        from transformers import LogitsProcessorList
        from src.model_utils.qwen2_5omni import ForceTokensProcessor
        force_proc = ForceTokensProcessor(target_ids, eos_id, prompt_len)

        tts_output = self.model.generate(
            **tts_inputs,
            use_audio_in_video=False,
            thinker_logits_processor=LogitsProcessorList([force_proc]),
            thinker_max_new_tokens=len(target_ids) + 1,
            temperature=0.4,
        )
        _, tts_audio = tts_output
        return tts_audio


    def raw_cache_payload(self) -> dict[str, Any]:
        # Pass 2 is conditioned on pass 1's quote, so the quote belongs in the key.
        return {
            "model": self.MODEL_ID,
            "system": self.SYSTEM_PROMPT,
            "prompt": self.pass2_prompt,
            "pass1_quote": self.pass1_quote,
        }

    def _pass1_cache_key(self) -> str:
        return self.cache.key({
            "stage": "speech",
            "sample": self.sample_id,
            "producer": {"model": self.MODEL_ID, "pass": 1},
            "system": self.SYSTEM_PROMPT,
            "prompt": self.pass1_prompt,
            "image": self.resolved_img_path,
        })

    def _run_pass1(self) -> str:
        """Generate the spoken quote and its audio; returns the quote."""
        pass1_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.pass1_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.resolved_img_path}
                ],
            },
        ]
        pass1_inputs = self._build_inputs(pass1_messages)
        pass1_output = self.model.generate(**pass1_inputs, use_audio_in_video=False)

        pass1_audio = None
        if isinstance(pass1_output, tuple):
            pass1_text_ids, pass1_audio = pass1_output
        else:
            pass1_text_ids = pass1_output

        pass1_asst_text = self._decode_assistant_text(pass1_text_ids)
        pass1_quote = self._normalize_quote_text(pass1_asst_text)

        if not pass1_quote:
            match = QUOTE_PATTERN.search(pass1_asst_text)
            try:
                pass1_quote = next(group for group in match.groups() if group is not None).strip()
            except (AttributeError, StopIteration):
                raise BackboneOutputError(
                    "Qwen2.5-Omni pass 1 produced no usable quote.",
                    raw_output=pass1_asst_text,
                ) from None

        if pass1_audio is None:
            print(f"[WARNING] No audio generated from pass 1 output:\n{pass1_asst_text}\n\nFalling back to forced audio.")
            pass1_audio = self._synthesize_audio_from_text(pass1_quote)

        import soundfile as sf
        sf.write(self.output_speech_path, pass1_audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        return pass1_quote

    def forward(self):
        # 1) Qwen2.5-omni pass 1: Generate quote text and audio.
        # The quote is cached alongside the wav because pass 2 needs it.
        pass1_key = self._pass1_cache_key()
        quote_path = f"{self.output_raw_dir}/pass1_quote.txt"
        pass1_files = {"artifact.wav": self.output_speech_path, "quote.txt": quote_path}

        if self.cache.restore("speech", pass1_key, pass1_files):
            with open(quote_path, "r", encoding="utf-8") as f:
                self.pass1_quote = f.read()
        else:
            self.pass1_quote = self._run_pass1()
            with open(quote_path, "w", encoding="utf-8") as f:
                f.write(self.pass1_quote)
            self.cache.store("speech", pass1_key, pass1_files, {
                "stage": "speech",
                "sample": self.sample_id,
                "producer": {"model": self.MODEL_ID, "pass": 1},
                "prompt": self.pass1_prompt,
            })

        pass1_quote = self.pass1_quote

        # 2) Qwen2.5-omni pass 2: Generate next-page text and t2i with pass1 quote+audio as extra inputs
        self.pass2_prompt = pass2_prompt = self.PASS2_PROMPT.format(
            metadata=self.metadata,
            current_page_text=self.current_page_text,
            next_page_condition=self.next_page_condition,
            pass1_quote=pass1_quote,
        )
        raw = self.load_cached_raw()
        if raw is None:
            try:
                raw = self._run_pass2(pass2_prompt, pass1_quote)
            except Exception as exc:
                # Pass 1 wrote the speech already; only text and image are lost.
                set_modalities(exc, ("text", "image"))
                raise
            self.save_raw(raw, raw["text"])

        # 3) T2I generation
        self.generate_image(raw["t2i"], self.resolved_img_path)

    def _run_pass2(self, pass2_prompt: str, pass1_quote: str) -> dict[str, Any]:
        pass2_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": pass2_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Pass 1 quote text: {pass1_quote}"},
                    {"type": "audio", "audio": self.output_speech_path},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.resolved_img_path}
                ],
            },
        ]
        pass2_inputs = self._build_inputs(pass2_messages)
        pass2_text_ids = self.model.generate(**pass2_inputs, use_audio_in_video=False, return_audio=False)
        pass2_asst_text = self._decode_assistant_text(pass2_text_ids)

        asst_json = self._parse_assistant_json(pass2_asst_text)
        if not isinstance(asst_json, dict):
            raise BackboneOutputError(
                "Qwen2.5-Omni pass 2 returned no parseable JSON object.",
                raw_output=pass2_asst_text,
            )

        missing = [field for field in ("text", "t2i") if not asst_json.get(field)]
        if missing:
            raise BackboneOutputError(
                f"Qwen2.5-Omni pass 2 output is missing required field(s): {', '.join(missing)}.",
                raw_output=pass2_asst_text,
            )

        return {
            "quote": pass1_quote,
            "text": asst_json["text"],
            "t2i": asst_json["t2i"],
        }





class Emu3Orchestrator(Orchestrator):
    """
    Emu3 orchestrator.

    Pass 1) Emu3-Chat:
        previous page image + context -> JSON {text, t2i, quote}
    Pass 2) Emu3-Gen:
        t2i -> image
    Pass 3) External T2SGenerator:
        quote -> SpeechMetadata -> wav
    """

    SYSTEM_PROMPT = """
Write the next page of a children's storybook.

Return ONLY valid JSON with exactly this format:
{"text":"...","t2i":"...","quote":"..."}

Rules:
- text is the next page text
- quote MUST be spoken by a character in the scene
- t2i MUST be a detailed visual prompt for the illustration, describing the characters, scene, composition, and mood
- output nothing except the JSON object
- use the given context, but do not copy or continue the prompt itself
- Do not repeat or continue given prompt
""".strip()

    USER_PROMPT = """
Story title: {title}
Story style: {genre_style}
Main characters: {characters}

Previous page text:
{current_page_text}

Next page requirements:
{next_page_condition_text}

Return only JSON with keys text, t2i, quote.
""".strip()

    POSITIVE_PROMPT_SUFFIX = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
        "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry."
    )

    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.emu3_source_revision = repository_revision("emu3")
        emu3_root = require_checkout("emu3")
        prepend_import_path(emu3_root)
        from emu3.mllm.processing_emu3 import Emu3Processor

        self.chat_model_id = self.config.get("emu3_chat_model", "BAAI/Emu3-Chat")
        self.gen_model_id = self.config.get("emu3_gen_model", "BAAI/Emu3-Gen")
        self.vq_model_id = self.config.get("emu3_vq_model", "BAAI/Emu3-VisionTokenizer")

        self.device_str = self.config.get("orch_device", "cuda:0")
        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        self.attn_impl = self.config.get("attn_implementation", "sdpa")
        self.chat_max_new_tokens = int(self.config.get("chat_max_new_tokens", 1024))
        self.image_max_new_tokens = int(self.config.get("image_max_new_tokens", 40960))
        self.image_do_sample = bool(self.config.get("image_do_sample", True))
        self.image_top_k = int(self.config.get("image_top_k", 2048))
        self.classifier_free_guidance = float(self.config.get("classifier_free_guidance", 3.0))
        self.image_ratio = self.config.get("image_ratio", "1:1")

        # -----------------------------
        # Chat model + processor
        # -----------------------------
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.chat_model_id,
            device_map=self.device_str if self.device.type == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=self.attn_impl if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        self.chat_model.eval()

        self.chat_tokenizer = AutoTokenizer.from_pretrained(
            self.chat_model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        self.chat_image_processor = AutoImageProcessor.from_pretrained(
            self.vq_model_id,
            trust_remote_code=True,
        )
        self.chat_image_tokenizer = AutoModel.from_pretrained(
            self.vq_model_id,
            device_map=self.device_str if self.device.type == "cuda" else None,
            trust_remote_code=True,
        ).eval()

        self.chat_processor = Emu3Processor(
            self.chat_image_processor,
            self.chat_image_tokenizer,
            self.chat_tokenizer,
        )

        # -----------------------------
        # Gen model + processor
        # -----------------------------
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            self.gen_model_id,
            device_map=self.device_str if self.device.type == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=self.attn_impl if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        self.gen_model.eval()

        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            self.gen_model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        self.gen_image_processor = AutoImageProcessor.from_pretrained(
            self.vq_model_id,
            trust_remote_code=True,
        )
        self.gen_image_tokenizer = AutoModel.from_pretrained(
            self.vq_model_id,
            device_map=self.device_str if self.device.type == "cuda" else None,
            trust_remote_code=True,
        ).eval()

        self.gen_processor = Emu3Processor(
            self.gen_image_processor,
            self.gen_image_tokenizer,
            self.gen_tokenizer,
        )

        # -----------------------------
        # T2S
        # -----------------------------

        self.default_speaker = {
            "name": self.config.get("default_speaker_name", "Narrator"),
            "emotion": self.config.get("default_emotion", "neutral"),
            "speed": self.config.get("default_speed", "normal"),
            "pitch": self.config.get("default_pitch", "normal"),
            "gender": self.config.get("default_gender", "female"),
        }

    def _metadata_to_prompt_text(self, metadata: Any) -> tuple[str, str, str]:
        if not isinstance(metadata, dict):
            return ("Unknown title", "children's story", "Unknown characters")

        title = metadata.get("title") or metadata.get("topic") or "Unknown title"
        genre_style = (
            metadata.get("genre")
            or metadata.get("style")
            or metadata.get("theme")
            or "children's story"
        )

        chars = metadata.get("characters", [])
        char_names = []
        if isinstance(chars, list):
            for c in chars[:5]:
                if isinstance(c, dict):
                    name = c.get("name")
                    if name:
                        char_names.append(str(name))
                elif isinstance(c, str):
                    char_names.append(c)

        characters = ", ".join(char_names) if char_names else "Unknown characters"
        return str(title), str(genre_style), characters

    def _condition_to_prompt_text(self, cond: dict[str, Any]) -> str:
        pieces = []

        scene = cond["scene"]
        if scene:
            pieces.append(f"Scene: {scene}")

        text_instruction = cond["text_instruction"]
        if text_instruction:
            pieces.append(f"Narrative goal: {text_instruction}")

        ambient = cond["ambient_sound"]
        if ambient:
            pieces.append(f"Ambient mood: {ambient}")

        chars = cond["characters"]
        if chars:
            char_bits = []
            for c in chars[:3]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name", "Unknown")
                emotion = c.get("emotion")
                action = c.get("action")

                bit = str(name)
                if emotion:
                    bit += f" feeling {emotion}"
                if action:
                    bit += f", {action}"
                char_bits.append(bit)

            if char_bits:
                pieces.append("Character states: " + "; ".join(char_bits))

        if not pieces:
            return "Continue the story naturally."

        return "\n".join(pieces)

    def format_input(self, entry: dict[str, Any]):
        super().format_input(entry)

        metadata = entry["book_metadata"]
        current_page = entry["current_page"]
        next_page_condition = entry["next_page_condition"]

        self.current_page_text = current_page["text"]
        self.prev_image_path = current_page["image_path"]

        title, genre_style, characters = self._metadata_to_prompt_text(metadata)
        next_page_condition_text = self._condition_to_prompt_text(next_page_condition)

        self.chat_prompt = self.USER_PROMPT.format(
            title=title,
            genre_style=genre_style,
            characters=characters,
            current_page_text=self.current_page_text,
            next_page_condition_text=next_page_condition_text,
        )

    def _strip_wrapping_quotes(self, s: str) -> str:
        s = (s or "").strip()
        if len(s) >= 2:
            pairs = [('“', '”'), ('"', '"'), ("'", "'"), ('‘', '’')]
            for lq, rq in pairs:
                if s.startswith(lq) and s.endswith(rq):
                    return s[1:-1].strip()
        return s

    def _extract_transcript(self, text: str) -> str:
        """Return the quote inside `text`, or "" when it holds none."""
        match = QUOTE_PATTERN.search(text or "")
        try:
            return next(g for g in match.groups() if g is not None).strip()
        except Exception:
            return ""

    def _parse_json_best_effort(self, s: str) -> Dict[str, Any]:
        s = s.strip()

        try:
            return json.loads(s)
        except Exception:
            pass

        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence:
            inner = fence.group(1).strip()
            try:
                return json.loads(inner)
            except Exception:
                s = inner

        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            candidate = s[l:r + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        def _grab(key: str):
            m = re.search(
                rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
                s,
                flags=re.DOTALL,
            )
            if not m:
                return None
            try:
                return json.loads(f'"{m.group(1)}"')
            except Exception:
                return m.group(1)

        recovered = {
            "text": _grab("text"),
            "t2i": _grab("t2i"),
            "quote": _grab("quote"),
        }
        if any(v is not None for v in recovered.values()):
            return {k: v for k, v in recovered.items() if v is not None}

        raise BackboneOutputError("Emu3-Chat returned no parseable JSON object.", raw_output=s)

    def _ensure_quote_in_text(self, text: str, quote: str) -> str:
        """Guarantee the page text carries the spoken quote exactly once.

        Both callers validate text and quote before getting here.
        """
        text = (text or "").strip()
        quote = self._strip_wrapping_quotes(quote)

        if quote in text:
            return text

        if QUOTE_PATTERN.search(text):
            def repl(_):
                return f'"{quote}"'
            return QUOTE_PATTERN.sub(repl, text, count=1)

        if text.endswith((".", "!", "?")):
            return f'{text} "{quote}"'
        return f'{text}. "{quote}"'

    def _build_speech_metadata_from_quote(self, quote: str):
        

        d = self.default_speaker
        quote = self._strip_wrapping_quotes(quote)

        return SpeechMetadata(
            speaker=SpeakerMetadata(
                name=d["name"],
                line=quote,
                emotion=d["emotion"],
                speed=d["speed"],
                pitch=d["pitch"],
                gender=d["gender"],
            )
        )

    @torch.no_grad()
    def _run_chat_pass(self) -> Dict[str, Any]:
        """
        Emu3-Chat:
        previous page image + story context -> JSON {text, t2i, quote}
        """
        image = Image.open(self.prev_image_path).convert("RGB")

        inputs = self.chat_processor(
            text=[self.SYSTEM_PROMPT + "\n\n" + self.chat_prompt],
            image=[image],
            mode="U",
            padding_image=True,
            padding="longest",
            return_tensors="pt",
        )

        dev = self.device_str if self.device.type == "cuda" else self.device
        input_ids = inputs.input_ids.to(dev)
        attention_mask = inputs.attention_mask.to(dev)

        generation_config = GenerationConfig(
            pad_token_id=self.chat_tokenizer.pad_token_id,
            bos_token_id=self.chat_tokenizer.bos_token_id,
            eos_token_id=self.chat_tokenizer.eos_token_id,
        )

        outputs = self.chat_model.generate(
            input_ids,
            generation_config,
            max_new_tokens=self.chat_max_new_tokens,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        decoded = self.chat_processor.batch_decode(outputs, skip_special_tokens=True)[0]

        parsed = self._parse_json_best_effort(decoded)
        parsed["_raw_decoded"] = decoded
        return parsed

    @torch.no_grad()
    def _run_gen_pass(self, prompt: str, out_path: str):
        """
        Emu3-Gen:
        t2i prompt -> image
        """
        prompt = (prompt or "").strip()
        if not prompt:
            prompt = "storybook illustration of the next scene."

        prompt = prompt + self.POSITIVE_PROMPT_SUFFIX

        kwargs = dict(
            mode="G",
            ratio=[self.image_ratio],
            image_area=self.gen_model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )

        pos_inputs = self.gen_processor(text=[prompt], **kwargs)
        neg_inputs = self.gen_processor(text=[self.NEGATIVE_PROMPT], **kwargs)

        generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.gen_model.config.eos_token_id,
            pad_token_id=self.gen_model.config.pad_token_id,
            max_new_tokens=self.image_max_new_tokens,
            do_sample=self.image_do_sample,
            top_k=self.image_top_k,
        )

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.gen_processor.build_prefix_constrained_fn(h, w)

        dev = self.device_str if self.device.type == "cuda" else self.device

        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                self.classifier_free_guidance,
                self.gen_model,
                unconditional_ids=neg_inputs.input_ids.to(dev),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])

        outputs = self.gen_model.generate(
            pos_inputs.input_ids.to(dev),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(dev),
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        saved = False
        for out in outputs:
            mm_list = self.gen_processor.decode(out)
            for im in mm_list:
                if isinstance(im, Image.Image):
                    im.save(out_path)
                    saved = True
                    break
            if saved:
                break

        if not saved:
            raise RuntimeError("Emu3-Gen failed to decode any image output.")

    def raw_cache_payload(self) -> dict[str, Any]:
        return {
            "model": self.chat_model_id,
            "source_revision": self.emu3_source_revision,
            "system": self.SYSTEM_PROMPT,
            "prompt": self.chat_prompt,
            "image": self.prev_image_path,
            "max_new_tokens": self.chat_max_new_tokens,
        }

    def _image_cache_payload(self, t2i_prompt: str) -> dict[str, Any]:
        return {
            "producer": {
                "model": self.gen_model_id,
                "source_revision": self.emu3_source_revision,
                "vq_model": self.vq_model_id,
                "ratio": self.image_ratio,
                "max_new_tokens": self.image_max_new_tokens,
                "do_sample": self.image_do_sample,
                "top_k": self.image_top_k,
                "classifier_free_guidance": self.classifier_free_guidance,
                "positive_suffix": self.POSITIVE_PROMPT_SUFFIX,
                "negative_prompt": self.NEGATIVE_PROMPT,
            },
            "prompt": t2i_prompt,
        }

    def forward(self):
        raw = self.load_cached_raw()
        if raw is None:
            chat = self._run_chat_pass()

            text = chat.get("text")
            t2i = chat.get("t2i")
            quote = self._strip_wrapping_quotes(chat.get("quote"))

            missing = [name for name, value in (("text", text), ("t2i", t2i), ("quote", quote)) if not value]
            if missing:
                raise BackboneOutputError(
                    f"Emu3-Chat output is missing required field(s): {', '.join(missing)}.",
                    raw_output=chat.get("_raw_decoded", ""),
                )

            text = self._ensure_quote_in_text(text, quote)

            out_obj = NextPageOutput(
                text=text,
                t2i=t2i,
                speech_metadata=self._build_speech_metadata_from_quote(quote),
            )
            raw = {
                "text": out_obj.text,
                "t2i": out_obj.t2i,
                "quote": quote,
                "speech_metadata": out_obj.speech_metadata.model_dump(),
                "raw_decoded": chat.get("_raw_decoded", ""),
            }
            self.save_raw(raw, out_obj.text)

        self.cached_artifact(
            "image",
            self._image_cache_payload(raw["t2i"]),
            self.output_image_path,
            lambda path: self._run_gen_pass(raw["t2i"], path),
        )
        self.generate_speech(raw["speech_metadata"], raw["text"])










class MMaDAOrchestrator(Orchestrator):
    """
    Pass 1) MMaDA MMU:
        previous page image + previous page text + next-page condition
        -> JSON {
            text,
            quote,
            t2i,
            speech_metadata: {
                speaker: {name, line, emotion, speed, pitch, gender}
            }
        }

    Pass 2) MMaDA T2I:
        validated single-line t2i prompt -> image

    Pass 3) T2S:
        speech_metadata -> wav
    """

    SYSTEM_PROMPT = """
You are an AI assistant that writes the next page of a children's storybook.

Return ONLY valid JSON.

Required JSON keys:
- text
- quote
- t2i
- speech_metadata

Example output format:
{
  "text": "Mina tiptoed into the glowing garden and smiled at the fireflies dancing above the flowers.",
  "quote": "Look how bright they are tonight!",
  "t2i": "A children's storybook illustration of a little girl in a glowing night garden with fireflies above blooming flowers, soft moonlight, warm magical atmosphere, detailed background, storybook style",
  "speech_metadata": {
    "speaker": {
      "name": "Mina",
      "line": "Look how bright they are tonight!",
      "emotion": "happy",
      "speed": "normal",
      "pitch": "high",
      "gender": "female"
    }
  }
}

Rules:
- Output JSON only.
- Use the given story context.
- t2i must describe the visible scene, not instructions or schema text.
- Do not copy the example literally.
""".strip()

    USER_PROMPT = """
Story title: {title}
Story style: {genre_style}
Main characters: {characters}

Previous page text:
{current_page_text}

Next page requirements:
{next_page_condition_text}

Return only valid JSON.
""".strip()

    POSITIVE_PROMPT_SUFFIX = ", masterpiece, children's storybook illustration, best quality"

    def __init__(self, config_name: str):
        super().__init__(config_name)

        self.mmada_source_revision = repository_revision("mmada")
        mmada_root = require_checkout("mmada")
        prepend_import_path(mmada_root)
        from models import MAGVITv2, get_mask_schedule, MMadaModelLM
        from training.prompting_utils import UniversalPrompting
        from training.utils import image_transform, image_transform_squash
        from omegaconf import OmegaConf

        self._MAGVITv2 = MAGVITv2
        self._MMadaModelLM = MMadaModelLM
        self._UniversalPrompting = UniversalPrompting
        self._get_mask_schedule = get_mask_schedule
        self._image_transform = image_transform
        self._image_transform_squash = image_transform_squash

        self.t2i_demo_config_path = self.config.get(
            "mmada_t2i_config_path",
            str(mmada_root / "configs" / "mmada_demo.yaml"),
        )
        self.t2i_demo_config = OmegaConf.load(self.t2i_demo_config_path)

        self.device_str = self.config.get("orch_device", "cuda:0")
        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        self.mmada_model_id = self.config.get("mmada_model", "Gen-Verse/MMaDA-8B-MixCoT")
        self.mmada_vq_model_id = self.config.get("mmada_vq_model", "showlab/magvitv2")

        self.max_text_len = int(self.config.get("mmada_max_text_len", 1024))
        self.image_resolution = int(self.config.get("mmada_resolution", 512))

        self.mmu_max_new_tokens = int(self.config.get("mmu_max_new_tokens", self.max_text_len))
        self.mmu_steps = int(self.config.get("mmu_steps", max(1, self.mmu_max_new_tokens // 2)))
        self.mmu_block_length = int(self.config.get("mmu_block_length", max(1, self.mmu_max_new_tokens // 4)))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mmada_model_id,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=self.max_text_len,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100,
            cond_dropout_prob=float(self.config.get("cond_dropout_prob", 0.0)),
            use_reserved_token=True,
        )

        self.vq_model = MAGVITv2.from_pretrained(self.mmada_vq_model_id).to(self.device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        self.model = MMadaModelLM.from_pretrained(
            self.mmada_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.mask_token_id = self.model.config.mask_token_id
        self.num_vq_tokens = self.model.config.num_vq_tokens
        self.codebook_size = self.model.config.codebook_size


        self.default_speaker = {
            "name": self.config.get("default_speaker_name", "Narrator"),
            "emotion": self.config.get("default_emotion", "neutral"),
            "speed": self.config.get("default_speed", "normal"),
            "pitch": self.config.get("default_pitch", "normal"),
            "gender": self.config.get("default_gender", "female"),
        }

        self.next_page_condition_text = ""
        self.current_page_text = ""
        self.prev_image_path = ""

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _metadata_to_prompt_text(self, metadata):
        if not isinstance(metadata, dict):
            return ("Unknown title", "children's story", "Unknown characters")

        title = metadata.get("title") or metadata.get("topic") or "Unknown title"
        genre_style = (
            metadata.get("genre")
            or metadata.get("style")
            or metadata.get("theme")
            or "children's story"
        )

        chars = metadata.get("characters", [])
        names = []
        if isinstance(chars, list):
            for c in chars[:5]:
                if isinstance(c, dict) and c.get("name"):
                    names.append(str(c["name"]))
                elif isinstance(c, str):
                    names.append(str(c))

        characters = ", ".join(names) if names else "Unknown characters"
        return str(title), str(genre_style), characters

    def _condition_to_prompt_text(self, cond):
        pieces = []

        scene = cond["scene"]
        if scene:
            pieces.append(f"Scene: {scene}")

        text_instruction = cond["text_instruction"]
        if text_instruction:
            pieces.append(f"Narrative goal: {text_instruction}")

        ambient = cond["ambient_sound"]
        if ambient:
            pieces.append(f"Ambient mood: {ambient}")

        chars = cond["characters"]
        if chars:
            states = []
            for c in chars[:3]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name", "Unknown")
                emotion = c.get("emotion")
                action = c.get("action")

                bit = str(name)
                if emotion:
                    bit += f" feeling {emotion}"
                if action:
                    bit += f", {action}"
                states.append(bit)

            if states:
                pieces.append("Character states: " + "; ".join(states))

        return "\n".join(pieces) if pieces else "Continue the story naturally."

    def format_input(self, entry: dict):
        super().format_input(entry)

        metadata = entry["book_metadata"]
        current_page = entry["current_page"]
        next_page_condition = entry["next_page_condition"]

        self.current_page_text = current_page["text"]
        self.prev_image_path = current_page["image_path"]

        title, genre_style, characters = self._metadata_to_prompt_text(metadata)
        self.next_page_condition_text = self._condition_to_prompt_text(next_page_condition)

        self.mmu_prompt = self.USER_PROMPT.format(
            title=title,
            genre_style=genre_style,
            characters=characters,
            current_page_text=self.current_page_text,
            next_page_condition_text=self.next_page_condition_text,
        )

    def _strip_wrapping_quotes(self, s: str) -> str:
        s = (s or "").strip()
        if len(s) >= 2:
            pairs = [('“', '”'), ('"', '"'), ("'", "'"), ('‘', '’')]
            for lq, rq in pairs:
                if s.startswith(lq) and s.endswith(rq):
                    return s[1:-1].strip()
        return s

    def _extract_transcript(self, text: str) -> str:
        """Return the quote inside `text`, or "" when it holds none."""
        match = QUOTE_PATTERN.search(text or "")
        try:
            return next(g for g in match.groups() if g is not None).strip()
        except Exception:
            return ""

    def _safe_json_loads(self, s: str) -> Optional[dict]:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            return None

    def _parse_json_best_effort(self, s: str):
        s = (s or "").strip()

        obj = self._safe_json_loads(s)
        if obj is not None:
            return obj

        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if fence:
            inner = fence.group(1).strip()
            obj = self._safe_json_loads(inner)
            if obj is not None:
                return obj
            s = inner

        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            candidate = s[l:r + 1]
            obj = self._safe_json_loads(candidate)
            if obj is not None:
                return obj

        def _grab_str(key: str):
            m = re.search(
                rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
                s,
                flags=re.DOTALL,
            )
            if not m:
                return None
            try:
                return json.loads(f'"{m.group(1)}"')
            except Exception:
                return m.group(1)

        recovered = {
            "text": _grab_str("text"),
            "quote": _grab_str("quote"),
            "t2i": _grab_str("t2i"),
        }

        if '"speech_metadata"' in s:
            speech_match = re.search(
                r'"speech_metadata"\s*:\s*(\{[\s\S]*\})',
                s,
                flags=re.DOTALL,
            )
            if speech_match:
                speech_candidate = speech_match.group(1).strip()
                speech_obj = self._safe_json_loads(speech_candidate)
                if speech_obj is not None:
                    recovered["speech_metadata"] = speech_obj

        if any(v is not None for v in recovered.values()):
            return {k: v for k, v in recovered.items() if v is not None}

        raise BackboneOutputError("MMaDA returned no parseable JSON object.", raw_output=s)

    def _ensure_quote_in_text(self, text: str, quote: str) -> str:
        """Guarantee the page text carries the spoken quote exactly once.

        Both callers validate text and quote before getting here.
        """
        text = (text or "").strip()
        quote = self._strip_wrapping_quotes(quote)

        if quote in text:
            return text

        if QUOTE_PATTERN.search(text):
            def repl(_):
                return f'"{quote}"'
            return QUOTE_PATTERN.sub(repl, text, count=1)

        if text.endswith((".", "!", "?")):
            return f'{text} "{quote}"'
        return f'{text}. "{quote}"'

    def _build_default_speech_metadata_dict(self, quote: str):
        d = self.default_speaker
        quote = self._strip_wrapping_quotes(quote)

        return {
            "speaker": {
                "name": d["name"],
                "line": quote,
                "emotion": d["emotion"],
                "speed": d["speed"],
                "pitch": d["pitch"],
                "gender": d["gender"],
            }
        }

    def _normalize_speech_metadata(self, speech_metadata, quote: str):
        d = self.default_speaker
        quote = self._strip_wrapping_quotes(quote)

        if hasattr(speech_metadata, "model_dump"):
            obj = speech_metadata.model_dump()
        elif isinstance(speech_metadata, dict):
            obj = speech_metadata
        elif isinstance(speech_metadata, str):
            obj = self._safe_json_loads(speech_metadata) or {}
        else:
            obj = {}

        spk = obj.get("speaker", {}) if isinstance(obj, dict) else {}
        if not isinstance(spk, dict):
            spk = {}

        name = spk.get("name") or d["name"]
        line = spk.get("line") or quote
        emotion = spk.get("emotion") or d["emotion"]
        speed = spk.get("speed") or d["speed"]
        pitch = spk.get("pitch") or d["pitch"]
        gender = spk.get("gender") or d["gender"]

        if emotion not in {"angry", "happy", "neutral", "sad"}:
            emotion = d["emotion"]
        if speed not in {"normal", "slow", "fast"}:
            speed = d["speed"]
        if pitch not in {"normal", "high", "low"}:
            pitch = d["pitch"]
        if gender not in {"male", "female"}:
            gender = d["gender"]

        # quote를 source of truth로 둠
        line = self._strip_wrapping_quotes(line)
        if not line:
            line = quote

        return SpeechMetadata(
            speaker=SpeakerMetadata(
                name=name,
                line=quote if quote else line,
                emotion=emotion,
                speed=speed,
                pitch=pitch,
                gender=gender,
            )
        )

    # ------------------------------------------------------------------
    # t2i prompt repair
    # ------------------------------------------------------------------

    def _flatten_t2i_candidate(self, raw_t2i: Any) -> str:
        if raw_t2i is None:
            return ""

        if isinstance(raw_t2i, str):
            return " ".join(raw_t2i.split()).strip()

        if isinstance(raw_t2i, list):
            parts = [str(x).strip() for x in raw_t2i if str(x).strip()]
            return " ".join(parts).strip()

        if isinstance(raw_t2i, dict):
            if "image" in raw_t2i and isinstance(raw_t2i["image"], dict):
                d = raw_t2i["image"]
                parts = []
                for key in ["name", "type", "color", "size", "action", "emotion", "scene", "background"]:
                    val = d.get(key)
                    if val:
                        parts.append(str(val))
                return ", ".join(parts).strip()

            parts = []
            for k, v in raw_t2i.items():
                if isinstance(v, (str, int, float, bool)):
                    parts.append(f"{k}: {v}")
                elif isinstance(v, dict):
                    inner = ", ".join(
                        f"{ik}: {iv}"
                        for ik, iv in v.items()
                        if isinstance(iv, (str, int, float, bool))
                    )
                    if inner:
                        parts.append(f"{k}: {inner}")
            return "; ".join(parts).strip()

        return str(raw_t2i).strip()

    def _is_abnormal_t2i_prompt(self, prompt: str, quote: str = "") -> bool:
        p = (prompt or "").strip()
        q = (quote or "").strip().lower()

        if not p:
            return True

        low = p.lower()

        if low in {"...", "null", "none", "<t2i>", "dummy", "a", "an"}:
            return True

        if len(p.split()) < 6:
            return True

        if q and low == q:
            return True

        if p.startswith(("\"", "“", "'", "‘")):
            return True

        bad_prefixes = [
            "oh",
            "i'm",
            "i am",
            "he said",
            "she said",
            "rabbit said",
            "the rabbit said",
        ]
        if any(low.startswith(x) for x in bad_prefixes):
            return True

        if any(ch in p for ch in "{}[]"):
            return True

        if " + " in p or '": "' in p:
            return True

        visual_keywords = [
            "illustration", "scene", "background", "lighting", "storybook", "style",
            "character", "indoor", "outdoor", "warm", "colorful",
            "expression", "pose", "detailed", "rabbit", "animal", "forest",
            "room", "garden", "moonlight", "composition"
        ]
        if not any(k in low for k in visual_keywords):
            return True

        return False

    def _validated_t2i_prompt(self, raw_t2i: Any, quote: str) -> str:
        """Return the model's own t2i prompt.

        Raises:
            BackboneOutputError: The prompt is empty or degenerate. Rebuilding one
                from the page text would fabricate an input the model never gave.
        """
        candidate = self._flatten_t2i_candidate(raw_t2i)

        if self._is_abnormal_t2i_prompt(candidate, quote=quote):
            raise BackboneOutputError("MMaDA produced no usable t2i prompt.", raw_output=raw_t2i)
        return candidate

    # ------------------------------------------------------------------
    # MMU
    # ------------------------------------------------------------------

    def _build_mmu_messages(self):
        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self.mmu_prompt,
            },
        ]

    def _prepare_image_tensor(self, image_path: str):
        image_ori = Image.open(image_path).convert("RGB")
        file_name = os.path.basename(image_path).lower()

        if any(tag in file_name for tag in ["ai2d", "clevr", "docvqa", "geo", "llava"]):
            image = self._image_transform_squash(
                image_ori,
                resolution=self.image_resolution,
            ).to(self.device)
        else:
            image = self._image_transform(
                image_ori,
                resolution=self.image_resolution,
            ).to(self.device)

        return image.unsqueeze(0)

    @torch.no_grad()
    def _run_mmu_pass(self):
        image = self._prepare_image_tensor(self.prev_image_path)

        image_tokens = self.vq_model.get_code(image)
        image_tokens = image_tokens.to(self.device)
        image_tokens = image_tokens + len(self.uni_prompting.text_tokenizer)

        messages = self._build_mmu_messages()
        text_token_ids = self.uni_prompting.text_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        batch_size = image_tokens.shape[0]

        mmu_id = self.uni_prompting.sptids_dict["<|mmu|>"]
        soi_id = self.uni_prompting.sptids_dict["<|soi|>"]
        eoi_id = self.uni_prompting.sptids_dict["<|eoi|>"]

        if torch.is_tensor(mmu_id):
            mmu_id = mmu_id.to(self.device)
        if torch.is_tensor(soi_id):
            soi_id = soi_id.to(self.device)
        if torch.is_tensor(eoi_id):
            eoi_id = eoi_id.to(self.device)

        input_ids = torch.cat([
            (torch.ones(batch_size, 1, device=self.device) * mmu_id),
            (torch.ones(batch_size, 1, device=self.device) * soi_id),
            image_tokens,
            (torch.ones(batch_size, 1, device=self.device) * eoi_id),
            text_token_ids,
        ], dim=1).long()

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        mmu_kwargs = dict(
            max_new_tokens=self.mmu_max_new_tokens,
            steps=self.mmu_steps,
            block_length=self.mmu_block_length,
        )

        sig = inspect.signature(self.model.mmu_generate)
        if "attention_mask" in sig.parameters:
            mmu_kwargs["attention_mask"] = attention_mask

        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.model.mmu_generate(
                    input_ids,
                    **mmu_kwargs,
                )
        else:
            output_ids = self.model.mmu_generate(
                input_ids,
                **mmu_kwargs,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        decoded = self.uni_prompting.text_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = self._parse_json_best_effort(decoded)
        parsed["_raw_decoded"] = decoded
        return parsed

    # ------------------------------------------------------------------
    # T2I
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_t2i_pass(self, prompt: str, out_path: str):
        prompt = (prompt or "").strip()
        if not prompt:
            prompt = "A children's storybook illustration of the next scene"

        prompt = " ".join(prompt.split()).strip()
        if not prompt.endswith("."):
            prompt += "."
        prompt += self.POSITIVE_PROMPT_SUFFIX

        cfg = self.t2i_demo_config

        guidance_scale = float(getattr(cfg.training, "guidance_scale", 3.0))
        generation_timesteps = int(getattr(cfg.training, "generation_timesteps", 12))
        generation_temperature = float(getattr(cfg.training, "generation_temperature", 1.0))
        noise_type = getattr(cfg.training, "noise_type", "mask")

        if hasattr(cfg, "mask_schedule") and cfg.mask_schedule is not None:
            schedule = cfg.mask_schedule.schedule
            args_sched = cfg.mask_schedule.get("params", {})
            mask_schedule = self._get_mask_schedule(schedule, **args_sched)
        else:
            mask_schedule = self._get_mask_schedule(
                getattr(cfg.training, "mask_schedule", "cosine")
            )

        num_vq_tokens = int(getattr(cfg.model.mmada, "num_vq_tokens", self.num_vq_tokens))
        codebook_size = int(getattr(cfg.model.mmada, "codebook_size", self.codebook_size))

        image_tokens = torch.ones(
            (1, num_vq_tokens),
            dtype=torch.long,
            device=self.device,
        ) * self.mask_token_id

        input_ids, attention_mask = self.uni_prompting(([prompt], image_tokens), "t2i_gen")
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = self.uni_prompting(([""], image_tokens), "t2i_gen")
            uncond_input_ids = uncond_input_ids.to(self.device)
            uncond_attention_mask = uncond_attention_mask.to(self.device)
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        gen_token_ids = self.model.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=guidance_scale,
            temperature=generation_temperature,
            timesteps=generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=noise_type,
            seq_len=num_vq_tokens,
            uni_prompting=self.uni_prompting,
            config=cfg,
        )

        gen_token_ids = torch.clamp(
            gen_token_ids,
            max=codebook_size - 1,
            min=0,
        )

        images = self.vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images = images * 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(images[0]).save(out_path)

        del image_tokens
        del input_ids
        del attention_mask
        if uncond_input_ids is not None:
            del uncond_input_ids
        if uncond_attention_mask is not None:
            del uncond_attention_mask
        del gen_token_ids
        del images

    # ------------------------------------------------------------------
    # output normalization
    # ------------------------------------------------------------------

    def _normalize_raw_generation(self, raw: Dict[str, Any]):
        text = raw.get("text")
        quote = raw.get("quote")
        raw_t2i = raw.get("t2i")
        speech_metadata = raw.get("speech_metadata")

        if not text:
            raise BackboneOutputError(
                "MMaDA output is missing required field(s): text.",
                raw_output=raw.get("_raw_decoded", ""),
            )

        if not quote:
            if isinstance(speech_metadata, dict):
                speaker = speech_metadata.get("speaker", {})
                if isinstance(speaker, dict):
                    quote = speaker.get("line")
            elif isinstance(speech_metadata, str):
                speech_obj = self._safe_json_loads(speech_metadata)
                if isinstance(speech_obj, dict):
                    speaker = speech_obj.get("speaker", {})
                    if isinstance(speaker, dict):
                        quote = speaker.get("line")

        quote = self._strip_wrapping_quotes(quote)
        if not quote:
            raise BackboneOutputError(
                "MMaDA output is missing required field(s): quote.",
                raw_output=raw.get("_raw_decoded", ""),
            )

        text = self._ensure_quote_in_text(text, quote)

        final_t2i = self._validated_t2i_prompt(raw_t2i, quote=quote)

        if speech_metadata:
            speech_metadata_obj = self._normalize_speech_metadata(speech_metadata, quote)
        else:
            speech_metadata_obj = self._normalize_speech_metadata(
                self._build_default_speech_metadata_dict(quote),
                quote,
            )

        return text, quote, raw_t2i, final_t2i, speech_metadata_obj

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def raw_cache_payload(self) -> dict[str, Any]:
        return {
            "model": self.mmada_model_id,
            "source_revision": self.mmada_source_revision,
            "system": self.SYSTEM_PROMPT,
            "prompt": self.mmu_prompt,
            "image": self.prev_image_path,
            "resolution": self.image_resolution,
            "max_new_tokens": self.mmu_max_new_tokens,
            "steps": self.mmu_steps,
            "block_length": self.mmu_block_length,
        }

    def _image_cache_payload(self, t2i_prompt: str) -> dict[str, Any]:
        return {
            "producer": {
                "model": self.mmada_model_id,
                "source_revision": self.mmada_source_revision,
                "vq_model": self.mmada_vq_model_id,
                "resolution": self.image_resolution,
                "guidance_scale": float(self.config.get("t2i_guidance_scale", 3.5)),
                "generation_timesteps": int(self.config.get("t2i_generation_timesteps", 15)),
                "generation_temperature": float(self.config.get("t2i_generation_temperature", 1.0)),
                "noise_type": self.config.get("t2i_noise_type", "mask"),
                "mask_schedule": self.config.get("mask_schedule", "cosine"),
                "config_path": self.t2i_demo_config_path,
                "positive_suffix": self.POSITIVE_PROMPT_SUFFIX,
            },
            "prompt": t2i_prompt,
        }

    def forward(self):
        raw = self.load_cached_raw()
        if raw is None:
            mmu = self._run_mmu_pass()

            text, quote, raw_t2i, final_t2i, speech_metadata_obj = self._normalize_raw_generation(mmu)

            out_obj = NextPageOutput(
                text=text,
                t2i=final_t2i,
                speech_metadata=speech_metadata_obj,
            )
            raw = {
                "text": out_obj.text,
                "quote": quote,
                "raw_t2i": raw_t2i,
                "final_t2i": out_obj.t2i,
                "speech_metadata": out_obj.speech_metadata.model_dump(),
                "raw_decoded": mmu.get("_raw_decoded", ""),
            }
            self.save_raw(raw, out_obj.text)

        self.cached_artifact(
            "image",
            self._image_cache_payload(raw["final_t2i"]),
            self.output_image_path,
            lambda path: self._run_t2i_pass(raw["final_t2i"], path),
        )
        self.generate_speech(raw["speech_metadata"], raw["text"])

        self._cleanup_cuda()





