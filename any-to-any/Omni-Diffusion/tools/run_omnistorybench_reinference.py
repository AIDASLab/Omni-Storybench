#!/usr/bin/env python3
"""Re-run Omni-Diffusion on the downloaded Omni-StoryBench snapshot.

The Hugging Face Parquet row order is the only mapping to output indices
0 through 899. Runtime records contain only the current page, book metadata,
and next-page condition. Reference next-page and speech fields are never read
into runtime records or prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
import importlib.metadata
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT.parent.parent / "evaluation" / "downloaded_dataset"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT.parent.parent
    / "evaluation"
    / "baselines"
    / "Omni-Diffusion"
)
EXPECTED_SAMPLE_COUNT = 900
EXPECTED_SOURCE_COUNTS = {
    "africanstorybook": 54,
    "digitallibrary": 293,
    "storybookscanda": 7,
    "storyweaver": 546,
}
EXPECTED_SENTINEL_IDS = {
    0: "africanstorybook/asb10179/page_006__page_007",
    724: "africanstorybook/asb10474/page_006__page_007",
    757: "africanstorybook/asb21992/page_003__page_004",
    758: "africanstorybook/asb31833/page_009__page_010",
    899: "storyweaver/515152-rosie-the-caterpillar/page_007__page_008",
}

MODEL_ROOT = REPO_ROOT / "models" / "Omni-Diffusion"
AUDIO_TOKENIZER_ROOT = REPO_ROOT / "models" / "THUDM" / "glm-4-voice-tokenizer"
FLOW_ROOT = REPO_ROOT / "models" / "THUDM" / "glm-4-voice-decoder"
IMAGE_TOKENIZER_ROOT = REPO_ROOT / "models" / "showlab" / "magvitv2"
GLM4VOICE_ROOT = REPO_ROOT / "third_party" / "GLM-4-Voice"

SENSEVOICE_CACHE_REPO = (
    REPO_ROOT
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--FunAudioLLM--SenseVoiceSmall"
)
EXPECTED_SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
SENSEVOICE_REQUIRED_FILES = (
    "config.yaml",
    "configuration.json",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "am.mvn",
    "model.pt",
)

PROMPT_COLUMNS = (
    "id",
    "source",
    "book",
    "from_key",
    "to_key",
    "current_image",
    "current_text",
    "genre",
    "topic",
    "style",
    "narrative_tense",
    "narrative_perspective",
    "scene",
    "text_instruction",
    "ambient_sound",
    "condition_characters",
    "book_characters",
)
INPUT_PATH_COLUMNS = (
    "current_image",
    "current_text_path",
    "sample_path",
    "metadata_path",
)
ARTIFACT_NAMES = {
    "text": "generated.txt",
    "image": "generated.png",
    "speech": "generated.wav",
}
PLAN_FIELDS = ("narration", "speaker", "speech", "image_prompt")


class ValidationError(RuntimeError):
    """Raised when the dataset, model, or output contract is violated."""


class GenerationError(RuntimeError):
    """Raised when a model call cannot produce a required artifact."""


def require_nonempty_file(path: Path, label: str) -> int:
    try:
        if path.is_file():
            size = path.stat().st_size
            if size > 0:
                return size
    except OSError:
        pass
    raise ValidationError(f"{label} is missing or empty: {path}")


def canonical_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ValidationError(f"{label} is not a directory: {resolved}")
    return resolved


def resolve_runtime_paths(dataset_root: Path, output_root: Path) -> Tuple[Path, Path]:
    """Resolve user-selected paths without changing the dataset/output formats."""
    dataset_root = canonical_existing_directory(dataset_root, "dataset root")
    output_root = Path(os.path.abspath(output_root.expanduser()))
    resolved_output = output_root.resolve(strict=False)
    if resolved_output != output_root:
        raise ValidationError(
            "output path must not contain symlink components: "
            f"{output_root} resolves to {resolved_output}"
        )
    return dataset_root, output_root


def resolve_dataset_file(
    dataset_root: Path, value: Any, column: str, index: int
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"row {index}: {column} must be a non-empty repo-relative path"
        )
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError(f"row {index}: unsafe {column} path: {value!r}")
    try:
        candidate = (dataset_root / Path(*relative.parts)).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(
            f"row {index}: missing {column} path: {value}"
        ) from exc
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ValidationError(
            f"row {index}: {column} resolves outside dataset root: {value}"
        ) from exc
    require_nonempty_file(candidate, f"row {index} {column}")
    return candidate


def parse_json_object_list(value: Any, field: str, index: int) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"row {index}: malformed JSON in {field}: {exc}"
            ) from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(
            f"row {index}: {field} must be a JSON list of objects"
        )
    return value


def normalize_condition_characters(
    sample_id: str,
    characters: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    missing_key_character = {
        "digitallibrary/the-green-apple/page_006__page_007": "Strawberry",
        "storyweaver/213576-jolly-swing/page_007__page_008": "Mamuni",
    }.get(sample_id)
    normalized = [dict(character) for character in characters]
    if missing_key_character is None:
        return normalized

    removed = 0
    for character in normalized:
        if (
            character.get("name") == missing_key_character
            and character.get("speech_intent") == ""
        ):
            del character["speech_intent"]
            removed += 1
    if removed != 1:
        raise ValidationError(
            f"{sample_id}: expected one Parquet-materialized empty "
            f"speech_intent for {missing_key_character!r}, found {removed}"
        )
    return normalized


def load_and_validate_snapshot(
    dataset_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError(
            "pyarrow is required; use the existing omni-diff Conda environment"
        ) from exc

    parquet_path = dataset_root / "data" / "omni_storybench.parquet"
    require_nonempty_file(parquet_path, "canonical Parquet index")
    parquet_file = pq.ParquetFile(str(parquet_path))
    available_columns = set(parquet_file.schema_arrow.names)
    required_columns = set(PROMPT_COLUMNS) | set(INPUT_PATH_COLUMNS)
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValidationError(f"Parquet is missing required columns: {missing_columns}")

    read_columns = list(dict.fromkeys((*PROMPT_COLUMNS, *INPUT_PATH_COLUMNS)))
    rows = parquet_file.read(columns=read_columns).to_pylist()
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValidationError(
            f"expected exactly {EXPECTED_SAMPLE_COUNT} rows, found {len(rows)}"
        )

    ids = [row.get("id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValidationError("every Parquet row must have a non-empty string id")
    if len(set(ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValidationError(
            f"expected {EXPECTED_SAMPLE_COUNT} unique ids, found {len(set(ids))}"
        )
    for index, expected_id in EXPECTED_SENTINEL_IDS.items():
        if ids[index] != expected_id:
            raise ValidationError(
                f"canonical row order mismatch at index {index}: "
                f"expected {expected_id!r}, found {ids[index]!r}"
            )

    source_counts = Counter(str(row.get("source")) for row in rows)
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValidationError(
            f"unexpected source distribution: {dict(source_counts)}"
        )

    records: List[Dict[str, Any]] = []
    checked_paths = 0
    for index, row in enumerate(rows):
        expected_id = (
            f"{row.get('source')}/{row.get('book')}/"
            f"{row.get('from_key')}__{row.get('to_key')}"
        )
        if row["id"] != expected_id:
            raise ValidationError(
                f"row {index}: id {row['id']!r} does not match {expected_id!r}"
            )
        current_text = row.get("current_text")
        if not isinstance(current_text, str) or not current_text.strip():
            raise ValidationError(f"row {index}: current_text is empty")

        resolved_paths = {}
        for column in INPUT_PATH_COLUMNS:
            resolved_paths[column] = resolve_dataset_file(
                dataset_root, row.get(column), column, index
            )
            checked_paths += 1

        record = {column: row.get(column) for column in PROMPT_COLUMNS}
        record["current_text"] = current_text.strip()
        record["condition_characters"] = normalize_condition_characters(
            str(record["id"]),
            parse_json_object_list(
                record["condition_characters"], "condition_characters", index
            )
        )
        record["book_characters"] = parse_json_object_list(
            record["book_characters"], "book_characters", index
        )
        record["current_image_path"] = str(resolved_paths["current_image"])
        records.append(record)

    summary = {
        "dataset_root": str(dataset_root),
        "parquet": str(parquet_path),
        "rows": len(records),
        "unique_ids": len(set(ids)),
        "checked_input_paths": checked_paths,
        "source_counts": dict(source_counts),
        "prompt_columns": list(PROMPT_COLUMNS),
        "ground_truth_columns_read": [],
        "index_contract": "Parquet row i maps to output directory i",
        "wholesale_replacement_indices": [724, 757, 758],
        "old_artifact_reuse": False,
    }
    return records, summary


def ensure_repo_asset(path: Path, label: str, directory: bool = False) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"{label} does not exist: {path}") from exc
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValidationError(
            f"{label} must stay inside the Omni-Diffusion copy: {resolved}"
        ) from exc
    if directory:
        if not resolved.is_dir():
            raise ValidationError(f"{label} is not a directory: {resolved}")
    else:
        require_nonempty_file(resolved, label)
    return resolved


def resolve_sensevoice_snapshot() -> Path:
    candidate = (
        SENSEVOICE_CACHE_REPO
        / "snapshots"
        / EXPECTED_SENSEVOICE_REVISION
    )
    try:
        snapshot = ensure_repo_asset(
            candidate, "SenseVoiceSmall snapshot", directory=True
        )
    except ValidationError as exc:
        raise ValidationError(
            "missing pinned repo-local SenseVoiceSmall snapshot; see "
            "OMNISTORYBENCH_REINFERENCE.md for the download command"
        ) from exc
    for filename in SENSEVOICE_REQUIRED_FILES:
        try:
            resolved = (snapshot / filename).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValidationError(
                f"SenseVoiceSmall snapshot is missing {filename}"
            ) from exc
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValidationError(
                f"SenseVoiceSmall file escapes the copied repository: {filename}"
            ) from exc
        require_nonempty_file(resolved, f"SenseVoiceSmall {filename}")
    return snapshot


def validate_runtime_assets() -> Dict[str, Any]:
    model_root = ensure_repo_asset(MODEL_ROOT, "model root", directory=True)
    audio_root = ensure_repo_asset(
        AUDIO_TOKENIZER_ROOT, "audio tokenizer root", directory=True
    )
    flow_root = ensure_repo_asset(FLOW_ROOT, "audio decoder root", directory=True)
    image_root = ensure_repo_asset(
        IMAGE_TOKENIZER_ROOT, "image tokenizer root", directory=True
    )
    glm_root = ensure_repo_asset(
        GLM4VOICE_ROOT, "GLM-4-Voice root", directory=True
    )
    sensevoice_snapshot = resolve_sensevoice_snapshot()

    required_files = (
        model_root / "config.json",
        model_root / "generation_config.json",
        model_root / "tokenizer_config.json",
        model_root / "model.safetensors.index.json",
        audio_root / "config.json",
        audio_root / "model.safetensors",
        flow_root / "config.yaml",
        flow_root / "flow.pt",
        flow_root / "hift.pt",
        image_root / "config.json",
        image_root / "pytorch_model.safetensors",
        glm_root / "flow_inference.py",
    )
    sensevoice_files = tuple(
        sensevoice_snapshot / filename
        for filename in SENSEVOICE_REQUIRED_FILES
    )
    checked_bytes = sum(
        require_nonempty_file(path, "runtime asset")
        for path in required_files + sensevoice_files
    )

    index_path = model_root / "model.safetensors.index.json"
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            weight_index = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid model shard index: {index_path}") from exc
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValidationError(f"model shard index has no weight_map: {index_path}")

    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ValidationError(f"unsafe model shard name: {shard_name!r}")
        checked_bytes += require_nonempty_file(
            model_root / shard_name, "model shard"
        )

    return {
        "model_root": str(model_root),
        "audio_tokenizer_root": str(audio_root),
        "audio_decoder_root": str(flow_root),
        "image_tokenizer_root": str(image_root),
        "sensevoice_snapshot": str(sensevoice_snapshot),
        "sensevoice_revision": EXPECTED_SENSEVOICE_REVISION,
        "model_shards": len(shard_names),
        "checked_bytes": checked_bytes,
    }


def clean_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().strip('"').strip("'").split())


def extract_field(text: str, field_name: str) -> str:
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.*?)(?=^\s*[A-Z_]+\s*:|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text or "")
    return clean_value(match.group(1)) if match else ""


def parse_plan(raw_text: str) -> Dict[str, str]:
    return {
        "narration": extract_field(raw_text, "NARRATION"),
        "speaker": extract_field(raw_text, "SPEAKER"),
        "speech": extract_field(raw_text, "SPEECH"),
        "image_prompt": extract_field(raw_text, "IMAGE_PROMPT"),
    }


def legacy_story_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {
            "genre": record["genre"],
            "topic": record["topic"],
            "style": record["style"],
            "narrative_tense": record["narrative_tense"],
            "narrative_perspective": record["narrative_perspective"],
            "characters": record["book_characters"],
        }
    }


def legacy_condition(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "characters": record["condition_characters"],
        "scene": record["scene"],
        "text content": record["text_instruction"],
        "ambient_sound": record["ambient_sound"],
    }


def build_planning_prompt(
    record: Mapping[str, Any], existing_narration: str = ""
) -> str:
    existing = ""
    if existing_narration:
        existing = (
            "\n\nA narration artifact was already generated for this index. "
            "Keep it exactly unchanged:\n"
            f"NARRATION: {existing_narration}"
        )
    metadata = legacy_story_metadata(record)
    condition = legacy_condition(record)
    return f"""You are writing the next page of a children's picture book.

Use the provided current page image, current page text, story metadata, and next page condition.
Keep the next page consistent with the current story.
Use the same language as the current page text.

Current page text:
{record['current_text']}

Story metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Next page condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Return exactly in this format:
NARRATION: <one or two short sentences for the next page narration>
SPEAKER: <the character who speaks, or narrator if no named character fits>
SPEECH: <one short spoken line for the next page>
IMAGE_PROMPT: <one detailed sentence describing the next-page illustration>{existing}

Current page image:"""


def build_speech_recovery_prompt(
    record: Mapping[str, Any], narration: str
) -> str:
    metadata = legacy_story_metadata(record)
    condition = legacy_condition(record)
    return f"""You are generating one spoken line for the next page of a children's picture book.

Current page text:
{record['current_text']}

Planned next page narration:
{narration}

Story metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Next page condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Return exactly in this format:
SPEAKER: <character name>
SPEECH: <one short spoken line>"""


def character_prompt_bits(record: Mapping[str, Any]) -> List[str]:
    pieces = []
    for character in record["condition_characters"]:
        values = [
            clean_value(character.get(key))
            for key in ("name", "emotion", "action")
        ]
        value = ", ".join(item for item in values if item)
        if value:
            pieces.append(value)
    return pieces


def build_image_prompt(
    record: Mapping[str, Any], plan: Mapping[str, str]
) -> str:
    characters = character_prompt_bits(record)
    pieces = [
        "A children's picture book illustration.",
        plan.get("image_prompt", ""),
        f"Narration: {plan['narration']}",
        f"Scene: {record['scene']}",
        f"Characters: {'; '.join(characters)}" if characters else "",
        f"Style: {record['style']}",
        f"Topic: {record['topic']}",
        "No text overlay, no caption, no speech bubble.",
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def artifact_path(output_root: Path, modality: str, index: int) -> Path:
    return output_root / modality / str(index) / ARTIFACT_NAMES[modality]


def artifact_validation_error(modality: str, path: Path) -> Optional[str]:
    try:
        if path.is_symlink():
            return "symlinks are not allowed"
        if not path.is_file() or path.stat().st_size <= 0:
            return "missing or empty"
        if modality == "text":
            if not path.read_text(encoding="utf-8").strip():
                return "blank UTF-8 text"
        elif modality == "image":
            from PIL import Image

            with Image.open(path) as image:
                if image.format != "PNG" or image.width < 1 or image.height < 1:
                    return (
                        f"expected non-empty PNG, found format={image.format!r} "
                        f"size={image.size!r}"
                    )
                image.verify()
        elif modality == "speech":
            import soundfile as sf

            info = sf.info(str(path))
            if (
                info.format != "WAV"
                or info.frames < 1
                or info.samplerate < 1
                or info.channels < 1
            ):
                return "WAV has no decodable audio frames"
        else:
            return f"unknown modality {modality!r}"
    except Exception as exc:
        return f"invalid: {exc}"
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_run_fingerprint(
    args: argparse.Namespace, dataset_root: Path
) -> str:
    sensevoice_snapshot = resolve_sensevoice_snapshot()
    digest_paths = [
        dataset_root / "data" / "omni_storybench.parquet",
        Path(__file__).resolve(),
        REPO_ROOT / "tools" / "inference.py",
        MODEL_ROOT / "config.json",
        MODEL_ROOT / "generation_config.json",
        MODEL_ROOT / "model.safetensors.index.json",
        MODEL_ROOT / "added_tokens.json",
        MODEL_ROOT / "merges.txt",
        MODEL_ROOT / "special_tokens_map.json",
        MODEL_ROOT / "tokenizer_config.json",
        MODEL_ROOT / "vocab.json",
        AUDIO_TOKENIZER_ROOT / "config.json",
        AUDIO_TOKENIZER_ROOT / "preprocessor_config.json",
        IMAGE_TOKENIZER_ROOT / "config.json",
        FLOW_ROOT / "config.yaml",
    ]
    digest_paths.extend(
        sensevoice_snapshot / filename
        for filename in SENSEVOICE_REQUIRED_FILES
        if filename != "model.pt"
    )
    for source_root in (
        REPO_ROOT / "omni_diffusion",
        MODEL_ROOT,
        GLM4VOICE_ROOT,
    ):
        digest_paths.extend(
            path
            for path in source_root.rglob("*.py")
            if path.is_file() and ".git" not in path.parts
        )
    digest_paths = sorted(
        {path.resolve() for path in digest_paths if path.is_file()},
        key=str,
    )

    weight_paths = tuple(sorted(MODEL_ROOT.glob("model-*.safetensors"))) + (
        AUDIO_TOKENIZER_ROOT / "model.safetensors",
        IMAGE_TOKENIZER_ROOT / "pytorch_model.safetensors",
        FLOW_ROOT / "flow.pt",
        FLOW_ROOT / "hift.pt",
        sensevoice_snapshot / "model.pt",
    )
    payload = {
        "schema": 1,
        "protocol": "legacy-separate-single-pass",
        "digests": {
            str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path):
            sha256_file(path)
            for path in digest_paths
        },
        "weight_identity": {
            str(path.relative_to(REPO_ROOT)): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in weight_paths
        },
        "generation": {
            name: getattr(args, name)
            for name in (
                "seed",
                "dtype",
                "text_max_tokens",
                "text_steps",
                "image_max_tokens",
                "image_steps",
                "speech_max_tokens",
                "speech_steps",
            )
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "diffusers": package_version("diffusers"),
            "numpy": package_version("numpy"),
            "pillow": package_version("Pillow"),
            "soundfile": package_version("soundfile"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_safe_output_destination(output_root: Path, index: int) -> None:
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise ValidationError(
            f"output root must be a real directory, not a file or symlink: {output_root}"
        )
    if output_root.is_dir():
        unexpected_roots = sorted(
            entry.name
            for entry in output_root.iterdir()
            if entry.name not in ARTIFACT_NAMES
        )
        if unexpected_roots:
            raise ValidationError(
                f"output root contains unexpected entries: {unexpected_roots}"
            )

    for modality, filename in ARTIFACT_NAMES.items():
        modality_root = output_root / modality
        if modality_root.is_symlink() or (
            modality_root.exists() and not modality_root.is_dir()
        ):
            raise ValidationError(f"{modality_root} must be a real directory")
        index_dir = modality_root / str(index)
        if index_dir.is_symlink() or (
            index_dir.exists() and not index_dir.is_dir()
        ):
            raise ValidationError(f"{index_dir} must be a real directory")
        if not index_dir.exists():
            continue
        unexpected = sorted(
            entry.name
            for entry in index_dir.iterdir()
            if entry.name != filename
        )
        if unexpected:
            raise ValidationError(
                f"{index_dir} contains unexpected entries: {unexpected}"
            )
        artifact = index_dir / filename
        if artifact.is_symlink():
            raise ValidationError(f"artifact may not be a symlink: {artifact}")
        if artifact.exists() and not artifact.is_file():
            raise ValidationError(
                f"artifact path must be a regular file: {artifact}"
            )


def ensure_safe_output_tree(output_root: Path) -> None:
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise ValidationError(
            f"output root must be a real directory: {output_root}"
        )
    if not output_root.exists():
        return

    unexpected_roots = sorted(
        entry.name
        for entry in output_root.iterdir()
        if entry.name not in ARTIFACT_NAMES
    )
    if unexpected_roots:
        raise ValidationError(
            f"output root contains unexpected entries: {unexpected_roots}"
        )

    expected_indices = {
        str(index) for index in range(EXPECTED_SAMPLE_COUNT)
    }
    for modality in ARTIFACT_NAMES:
        modality_root = output_root / modality
        if not modality_root.exists() and not modality_root.is_symlink():
            continue
        if modality_root.is_symlink() or not modality_root.is_dir():
            raise ValidationError(
                f"{modality_root} must be a real directory"
            )
        entries = list(modality_root.iterdir())
        unexpected_indices = sorted(
            entry.name
            for entry in entries
            if entry.name not in expected_indices
        )
        if unexpected_indices:
            raise ValidationError(
                f"{modality_root} contains unexpected indices: "
                f"{unexpected_indices}"
            )
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                raise ValidationError(
                    f"{entry} must be a real index directory"
                )
            ensure_safe_output_destination(output_root, int(entry.name))


def inspect_selected(output_root: Path, indices: Sequence[int]) -> Dict[str, Any]:
    valid_counts = {modality: 0 for modality in ARTIFACT_NAMES}
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        return {
            "selected_count": len(indices),
            "valid_artifact_counts": valid_counts,
            "complete_triplets": 0,
            "incomplete_count": len(indices),
            "incomplete_indices": list(indices),
            "extra_entry_count": 1,
            "extra_entries": [f"unsafe output root: {output_root}"],
        }

    incomplete = []
    extra_entries = []
    for index in indices:
        complete = True
        for modality, filename in ARTIFACT_NAMES.items():
            modality_root = output_root / modality
            index_dir = modality_root / str(index)
            if modality_root.is_symlink() or (
                modality_root.exists() and not modality_root.is_dir()
            ):
                extra_entries.append(f"unsafe modality root: {modality_root}")
                complete = False
                continue
            if index_dir.is_symlink() or (
                index_dir.exists() and not index_dir.is_dir()
            ):
                extra_entries.append(f"unsafe index directory: {index_dir}")
                complete = False
                continue

            artifact = index_dir / filename
            error = artifact_validation_error(modality, artifact)
            if error is None:
                valid_counts[modality] += 1
            else:
                complete = False
            if index_dir.is_dir():
                for entry in index_dir.iterdir():
                    if entry.name != filename:
                        extra_entries.append(str(entry))
                        complete = False
        if not complete:
            incomplete.append(index)
    return {
        "selected_count": len(indices),
        "valid_artifact_counts": valid_counts,
        "complete_triplets": len(indices) - len(incomplete),
        "incomplete_count": len(incomplete),
        "incomplete_indices": incomplete,
        "extra_entry_count": len(extra_entries),
        "extra_entries": extra_entries,
    }


def verify_full_result_tree(output_root: Path) -> List[str]:
    errors: List[str] = []
    if output_root.is_symlink():
        return [f"result root may not be a symlink: {output_root}"]
    if not output_root.is_dir():
        return [f"result root does not exist or is not a directory: {output_root}"]

    expected_modalities = set(ARTIFACT_NAMES)
    actual_root_entries = {entry.name for entry in output_root.iterdir()}
    for name in sorted(expected_modalities - actual_root_entries):
        errors.append(f"missing modality directory: {name}")
    for name in sorted(actual_root_entries - expected_modalities):
        errors.append(f"unexpected result-root entry: {name}")

    expected_indices = {str(index) for index in range(EXPECTED_SAMPLE_COUNT)}
    for modality, filename in ARTIFACT_NAMES.items():
        modality_root = output_root / modality
        if modality_root.is_symlink():
            errors.append(f"{modality}: modality root may not be a symlink")
            continue
        if not modality_root.is_dir():
            if modality_root.exists():
                errors.append(f"{modality}: expected a directory")
            continue
        actual_indices = {entry.name for entry in modality_root.iterdir()}
        missing = sorted(expected_indices - actual_indices, key=int)
        extras = sorted(actual_indices - expected_indices)
        if missing:
            errors.append(
                f"{modality}: missing {len(missing)} index entries; first={missing[:10]}"
            )
        if extras:
            errors.append(f"{modality}: unexpected index entries: {extras[:10]}")

        for index in range(EXPECTED_SAMPLE_COUNT):
            index_dir = modality_root / str(index)
            if index_dir.is_symlink():
                errors.append(f"{modality}/{index}: directory may not be a symlink")
                continue
            if not index_dir.is_dir():
                if index_dir.exists():
                    errors.append(f"{modality}/{index}: expected a directory")
                continue
            entries = sorted(entry.name for entry in index_dir.iterdir())
            if entries != [filename]:
                errors.append(
                    f"{modality}/{index}: expected only {filename!r}, found {entries!r}"
                )
                continue
            error = artifact_validation_error(modality, index_dir / filename)
            if error is not None:
                errors.append(f"{modality}/{index}/{filename}: {error}")
    return errors




def configure_runtime_environment() -> Path:
    cache_root = REPO_ROOT / ".cache"
    if cache_root.is_symlink() or (
        cache_root.exists() and not cache_root.is_dir()
    ):
        raise ValidationError(
            f"cache root must be a real repository directory: {cache_root}"
        )
    cache_root.mkdir(exist_ok=True)
    if not cache_root.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValidationError(
            f"cache root resolves outside the repository: {cache_root}"
        )

    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_MODULES_CACHE": cache_root / "huggingface" / "modules_pinned",
        "XDG_CACHE_HOME": cache_root,
        "TORCH_HOME": cache_root / "torch",
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TMPDIR": cache_root / "tmp",
    }
    for key, path in paths.items():
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValidationError(
                f"cache path must be a real directory: {path}"
            )
        path.mkdir(exist_ok=True)
        if not path.resolve().is_relative_to(REPO_ROOT.resolve()):
            raise ValidationError(
                f"cache path resolves outside the repository: {path}"
            )
        os.environ[key] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PIP_NO_INDEX"] = "1"
    tempfile.tempdir = None
    return cache_root


def load_inference_backend(args: argparse.Namespace):
    configure_runtime_environment()
    tools_dir = str(REPO_ROOT / "tools")
    glm_dir = str(GLM4VOICE_ROOT)
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry not in (tools_dir, glm_dir)
    ]
    sys.path.insert(0, glm_dir)
    sys.path.insert(0, tools_dir)

    import torch
    import inference as inference_module

    if not torch.cuda.is_available():
        raise GenerationError("Omni-Diffusion inference requires a CUDA-visible GPU")
    inference_module.device_map = "cuda:0"
    inference_module.audio_tokenizer_rank = 0
    inference_module.torch_dtype = getattr(torch, args.dtype)
    inference_module.set_seed(args.seed)
    model = inference_module.S2SInference(
        str(MODEL_ROOT),
        str(AUDIO_TOKENIZER_ROOT),
        "sensevoice_glm4voice",
        str(IMAGE_TOKENIZER_ROOT),
        flow_path=str(FLOW_ROOT),
    )
    return inference_module, model, torch


def generate_plan(
    record: Mapping[str, Any],
    index: int,
    args: argparse.Namespace,
    inference_module,
    model,
) -> Dict[str, str]:
    planning_prompt = build_planning_prompt(record)
    raw, _, _ = model.run_infer(
        image_path=record["current_image_path"],
        message=planning_prompt,
        max_tokens=args.text_max_tokens,
        steps=args.text_steps,
        task="",
        alg="entropy",
        repeat_penalty=1.05,
    )
    plan = parse_plan(raw or "")

    if not plan["narration"]:
        plan["narration"] = (
            clean_value(record["text_instruction"])
            or clean_value(raw)
        )
    if not plan["narration"]:
        raise GenerationError(f"row {index}: model produced no usable narration")

    if not plan["speech"]:
        recovery_prompt = build_speech_recovery_prompt(
            record, plan["narration"]
        )
        recovery_raw, _, _ = model.run_infer(
            image_path=record["current_image_path"],
            message=recovery_prompt,
            max_tokens=80,
            steps=80,
            task="",
            alg="entropy",
            repeat_penalty=1.05,
        )
        recovered = parse_plan(recovery_raw or "")
        plan["speaker"] = plan["speaker"] or recovered["speaker"]
        plan["speech"] = plan["speech"] or recovered["speech"]

    plan["speaker"] = plan["speaker"] or "character"
    plan["speech"] = plan["speech"] or "Let's keep going."
    if not plan["image_prompt"]:
        plan["image_prompt"] = build_image_prompt(record, plan)
    return plan


def save_text_artifact(path: Path, narration: str) -> None:
    narration = clean_value(narration)
    if not narration:
        raise GenerationError("refusing to save an empty narration")
    path.write_text(narration + "\n", encoding="utf-8")
    error = artifact_validation_error("text", path)
    if error is not None:
        raise GenerationError(f"staged text artifact {error}")


def generate_image_artifact(
    path: Path,
    prompt: str,
    index: int,
    args: argparse.Namespace,
    inference_module,
    model,
) -> bool:
    from PIL import Image

    _, _, generated_image = model.run_infer(
        message=(
            "Generate an image based on the provided text description.\n"
            + prompt
        ),
        task="T2I",
        max_tokens=args.image_max_tokens,
        steps=args.image_steps,
        alg="entropy-penalty",
        repeat_penalty=1.2,
        max_position_penalty=2.0,
    )
    if generated_image is None:
        return False
    if getattr(generated_image, "ndim", 0) != 3:
        raise GenerationError(
            f"row {index}: unexpected image shape "
            f"{getattr(generated_image, 'shape', None)}"
        )
    Image.fromarray(generated_image[:, :, ::-1]).save(path, format="PNG")
    error = artifact_validation_error("image", path)
    if error is not None:
        raise GenerationError(f"row {index}: staged PNG {error}")
    return True


def generate_speech_artifact(
    path: Path,
    speech: str,
    index: int,
    args: argparse.Namespace,
    inference_module,
    model,
) -> bool:
    import numpy as np
    import soundfile as sf

    _, generated_speech, _ = model.run_infer(
        message="Convert the text to speech.\n" + speech,
        task="TTS",
        max_tokens=args.speech_max_tokens,
        steps=args.speech_steps,
        alg="entropy",
        repeat_penalty=1.0,
    )
    if generated_speech is None:
        return False
    audio = generated_speech.detach().cpu().float().numpy().squeeze()
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise GenerationError(
            f"row {index}: invalid generated waveform shape={audio.shape}"
        )
    sf.write(
        str(path),
        audio,
        22050,
        format="WAV",
        subtype="PCM_16",
    )
    error = artifact_validation_error("speech", path)
    if error is not None:
        raise GenerationError(f"row {index}: staged WAV {error}")
    return True


def publish_artifact(
    stage_dir: Path,
    output_root: Path,
    index: int,
    modality: str,
) -> None:
    if modality not in ARTIFACT_NAMES:
        raise GenerationError(f"unknown modality: {modality}")
    ensure_safe_output_destination(output_root, index)
    source = stage_dir / ARTIFACT_NAMES[modality]
    error = artifact_validation_error(modality, source)
    if error is not None:
        raise GenerationError(
            f"refusing to publish invalid {modality} artifact: {error}"
        )
    destination = artifact_path(output_root, modality, index)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_safe_output_destination(output_root, index)
    os.replace(str(source), str(destination))
    error = artifact_validation_error(modality, destination)
    if error is not None:
        raise GenerationError(
            f"published {modality} artifact failed validation: {error}"
        )


def process_sample(
    record: Mapping[str, Any],
    index: int,
    args: argparse.Namespace,
    inference_module,
    model,
    stage_parent: Path,
) -> str:
    ensure_safe_output_destination(args.output_root, index)
    for modality in ARTIFACT_NAMES:
        index_dir = args.output_root / modality / str(index)
        index_dir.mkdir(parents=True, exist_ok=True)
        ensure_safe_output_destination(args.output_root, index)

    with tempfile.TemporaryDirectory(
        prefix=f"{index:04d}_", dir=str(stage_parent)
    ) as temporary:
        stage_dir = Path(temporary)
        plan = generate_plan(record, index, args, inference_module, model)

        image_generated = generate_image_artifact(
            stage_dir / ARTIFACT_NAMES["image"],
            build_image_prompt(record, plan),
            index,
            args,
            inference_module,
            model,
        )
        speech_generated = generate_speech_artifact(
            stage_dir / ARTIFACT_NAMES["speech"],
            plan["speech"],
            index,
            args,
            inference_module,
            model,
        )

        save_text_artifact(
            stage_dir / ARTIFACT_NAMES["text"], plan["narration"]
        )
        publish_artifact(stage_dir, args.output_root, index, "text")
        if image_generated:
            publish_artifact(
                stage_dir, args.output_root, index, "image"
            )
        if speech_generated:
            publish_artifact(
                stage_dir, args.output_root, index, "speech"
            )

        missing_modalities = [
            modality
            for modality, generated in (
                ("image", image_generated),
                ("speech", speech_generated),
            )
            if not generated
        ]
        if missing_modalities:
            print(
                json.dumps(
                    {
                        "index": index,
                        "missing_modalities": missing_modalities,
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
    return "generated_partial" if missing_modalities else "generated"


def select_indices(start: int, end: int, sample_count: int) -> List[int]:
    if start < 0 or end < start or end > sample_count:
        raise ValidationError(
            f"invalid half-open range [{start}, {end}); "
            f"valid dataset range is [0, {sample_count})"
        )
    return list(range(start, end))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 900 Omni-Diffusion Omni-StoryBench "
            "image/text/speech triplets"
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Downloaded Omni-StoryBench directory (default: evaluation/downloaded_dataset)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Generated artifact directory (default: evaluation/baselines/Omni-Diffusion)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-results-only", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--text-max-tokens", type=int, default=192)
    parser.add_argument("--text-steps", type=int, default=192)
    parser.add_argument("--image-max-tokens", type=int, default=260)
    parser.add_argument("--image-steps", type=int, default=260)
    parser.add_argument("--speech-max-tokens", type=int, default=64)
    parser.add_argument("--speech-steps", type=int, default=32)
    return parser


def validate_generation_args(args: argparse.Namespace) -> None:
    for name in (
        "text_max_tokens",
        "text_steps",
        "image_max_tokens",
        "image_steps",
        "speech_max_tokens",
        "speech_steps",
    ):
        if getattr(args, name) < 1:
            raise ValidationError(
                f"--{name.replace('_', '-')} must be at least 1"
            )


def print_errors(errors: Sequence[str], label: str) -> None:
    print(f"{label}: {len(errors)} issue(s)", file=sys.stderr)
    for error in errors[:100]:
        print(f"  - {error}", file=sys.stderr)
    if len(errors) > 100:
        print(f"  ... {len(errors) - 100} more", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_generation_args(args)
    args.dataset_root, args.output_root = resolve_runtime_paths(
        args.dataset_root, args.output_root
    )
    os.chdir(REPO_ROOT)
    records, dataset_summary = load_and_validate_snapshot(args.dataset_root)
    asset_summary = validate_runtime_assets()
    indices = select_indices(args.start, args.end, len(records))
    args.run_fingerprint = build_run_fingerprint(args, args.dataset_root)

    preflight = {
        "mode": (
            "validate-only"
            if args.validate_only
            else "dry-run"
            if args.dry_run
            else "verify-results-only"
            if args.verify_results_only
            else "inference"
        ),
        "dataset": dataset_summary,
        "runtime_assets": asset_summary,
        "run_fingerprint": args.run_fingerprint,
        "selection": {
            "start": args.start,
            "end_exclusive": args.end,
            "count": len(indices),
        },
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

    if args.validate_only:
        print("validation-only passed; no model was loaded and no result was written")
        return 0

    if args.verify_results_only:
        if args.start != 0 or args.end != EXPECTED_SAMPLE_COUNT:
            raise ValidationError(
                "--verify-results-only requires the full default range 0..900"
            )
        errors = verify_full_result_tree(args.output_root)
        if errors:
            print_errors(errors, "result verification failed")
            return 1
        print("result verification passed: 900 exact, decodable triplets")
        return 0

    ensure_safe_output_tree(args.output_root)
    for index in indices:
        ensure_safe_output_destination(args.output_root, index)
    before = inspect_selected(args.output_root, indices)
    pending_indices = list(indices)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output_status": before,
                    "pending": len(pending_indices),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    generated = 0
    failures = []
    inference_module = model = stage_parent = None
    if pending_indices:
        cache_root = configure_runtime_environment()
        stage_parent = cache_root / "omnistorybench_omnidiff_tmp"
        stage_parent.mkdir(parents=True, exist_ok=True)
        inference_module, model, _ = load_inference_backend(args)

    for ordinal, index in enumerate(pending_indices, start=1):
        sample_id = records[index]["id"]
        print(
            f"[{ordinal}/{len(pending_indices)}] index={index} id={sample_id}",
            flush=True,
        )
        try:
            status = process_sample(
                records[index],
                index,
                args,
                inference_module,
                model,
                stage_parent,
            )
            generated += 1
            print(
                json.dumps(
                    {"index": index, "id": sample_id, "status": status},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "index": index,
                "id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                json.dumps({"failure": failure}, ensure_ascii=False),
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

    after = inspect_selected(args.output_root, indices)
    summary = {
        "selected": len(indices),
        "generated": generated,
        "caught_failures": failures,
        "output_status": after,
    }
    print(
        json.dumps({"run_summary": summary}, ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "interrupted", file=sys.stderr,
        )
        raise SystemExit(130)
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        raise SystemExit(2)
