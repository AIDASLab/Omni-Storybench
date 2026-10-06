#!/usr/bin/env python3
"""Re-run AnyGPT on the locally downloaded Omni-StoryBench test split.

The canonical sample index is the row order in ``data/omni_storybench.parquet``.
Only prompt-safe columns from the downloaded Parquet snapshot are read.
Ground-truth columns and per-sample JSON contents are deliberately never loaded
by this runner.

For sample ``i``, the only published files are:

* ``<results>/text/i/generated.txt``
* ``<results>/image/i/generated.png``
* ``<results>/speech/i/generated.wav``

Each modality is written immediately after its single legacy generation pass.
If a later pass fails, already generated artifacts remain in place and the run
continues with the next sample, matching the copied baseline behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO_ROOT.parents[1]
DEFAULT_DATASET_ROOT = BENCHMARK_ROOT / "evaluation" / "downloaded_dataset"
DEFAULT_RESULTS_ROOT = BENCHMARK_ROOT / "evaluation" / "baselines" / "AnyGPT"
EXPECTED_SAMPLE_COUNT = 900

ARTIFACT_NAMES = {
    "text": "generated.txt",
    "image": "generated.png",
    "speech": "generated.wav",
}

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

MMGPT_RESPONSE_PATTERN = re.compile(
    r"\[MMGPT\]\s*:\s*(.*?)(?:<eos>|<eom>)\s*$", flags=re.DOTALL
)


class ValidationError(RuntimeError):
    """Raised when an input, checkpoint, or result invariant is violated."""


@dataclass(frozen=True)
class BenchmarkInput:
    """Prompt-safe view of one benchmark row (contains no ground truth)."""

    index: int
    sample_id: str
    current_image: Path
    current_text: str
    book_metadata: Mapping[str, Any]
    next_page_condition: Mapping[str, Any]


@dataclass(frozen=True)
class StoryFields:
    current_page_text: str
    metadata_text: str
    condition_text: str


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_dataset_root(candidate: str) -> Path:
    resolved = Path(candidate).expanduser().resolve()
    if not resolved.is_dir():
        raise ValidationError(f"dataset root is not a directory: {resolved}")
    return resolved


def _resolve_results_root(candidate: str) -> Path:
    requested = Path(os.path.abspath(os.path.expanduser(candidate)))
    current = Path(requested.anchor)
    for component in requested.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValidationError(
                f"results root path may not contain symlinks: {current}"
            )
    return requested


def _resolve_repo_path(candidate: str, label: str, *, directory: bool = False) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.expanduser().resolve()
    exists = resolved.is_dir() if directory else _nonempty_file(resolved)
    expected_kind = "directory" if directory else "non-empty file"
    if not exists:
        raise ValidationError(f"{label} is not a {expected_kind}: {resolved}")
    return resolved


def _resolve_dataset_file(dataset_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValidationError(f"{label} must be a non-empty repo-relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValidationError(f"{label} must be repo-relative, not absolute: {relative}")

    root = dataset_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationError(f"{label} escapes the dataset root: {relative}") from exc
    if not _nonempty_file(resolved):
        raise ValidationError(f"{label} is missing or empty: {resolved}")
    return resolved


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty text")
    return value.strip()


def _parse_json_object_list(value: Any, label: str) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{label} contains malformed JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(f"{label} must be a JSON list of objects")
    return value


def _normalize_condition_characters(
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


def _read_canonical_rows(dataset_root: Path) -> List[Dict[str, Any]]:
    parquet_path = dataset_root / "data" / "omni_storybench.parquet"
    if not _nonempty_file(parquet_path):
        raise ValidationError(f"canonical Parquet file is missing or empty: {parquet_path}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ValidationError(
            "pyarrow is required to preserve the canonical Parquet row order; "
            "run with the existing anygpt Conda environment"
        ) from exc

    required_columns = list(dict.fromkeys((*PROMPT_COLUMNS, *INPUT_PATH_COLUMNS)))
    parquet_file = parquet.ParquetFile(str(parquet_path))
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(required_columns) - available)
    if missing:
        raise ValidationError(f"Parquet is missing required columns: {missing}")
    return parquet_file.read(columns=required_columns).to_pylist()


def validate_and_load_inputs(dataset_root: Path) -> List[BenchmarkInput]:
    """Validate the complete release and return prompt-safe inputs in row order."""

    rows = _read_canonical_rows(dataset_root)
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
        raise ValidationError(f"unexpected source distribution: {dict(source_counts)}")

    safe_inputs: List[BenchmarkInput] = []
    for index, row in enumerate(rows):
        sample_id = row["id"]
        expected_id = (
            f"{row.get('source')}/{row.get('book')}/"
            f"{row.get('from_key')}__{row.get('to_key')}"
        )
        if sample_id != expected_id:
            raise ValidationError(
                f"row {index} id {sample_id!r} does not match {expected_id!r}"
            )

        current_text = _require_text(row.get("current_text"), f"row {index} current_text")
        resolved_paths = {
            column: _resolve_dataset_file(
                dataset_root, row.get(column), f"row {index} {column}"
            )
            for column in INPUT_PATH_COLUMNS
        }
        book_characters = _parse_json_object_list(
            row.get("book_characters"), f"row {index} book_characters"
        )
        condition_characters = _normalize_condition_characters(
            sample_id,
            _parse_json_object_list(
                row.get("condition_characters"), f"row {index} condition_characters"
            )
        )
        book_metadata = {
            "genre": row.get("genre"),
            "topic": row.get("topic"),
            "style": row.get("style"),
            "narrative_tense": row.get("narrative_tense"),
            "narrative_perspective": row.get("narrative_perspective"),
            "characters": book_characters,
        }
        condition = {
            "characters": condition_characters,
            "scene": row.get("scene"),
            "text content": row.get("text_instruction"),
            "ambient_sound": row.get("ambient_sound"),
        }

        safe_inputs.append(
            BenchmarkInput(
                index=index,
                sample_id=sample_id,
                current_image=resolved_paths["current_image"],
                current_text=current_text,
                book_metadata=book_metadata,
                next_page_condition=condition,
            )
        )

    return safe_inputs


def validate_runtime_assets(args: argparse.Namespace) -> None:
    model_dir = _resolve_repo_path(args.model_name_or_path, "AnyGPT model", directory=True)
    for filename in (
        "config.json",
        "pytorch_model.bin",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        if not _nonempty_file(model_dir / filename):
            raise ValidationError(f"AnyGPT model is missing {filename}: {model_dir}")

    _resolve_repo_path(args.image_tokenizer_path, "image tokenizer")
    _resolve_repo_path(args.speech_tokenizer_path, "speech tokenizer")
    _resolve_repo_path(args.speech_tokenizer_config, "speech tokenizer config")
    _resolve_repo_path(args.soundstorm_path, "SoundStorm checkpoint")
    _resolve_repo_path(args.diffusion_model_path, "diffusion model", directory=True)
    for label, path in (
        ("text generation config", args.text_generation_config),
        ("image generation config", args.image_generation_config),
        ("speech generation config", args.speech_generation_config),
    ):
        config_path = _resolve_repo_path(path, label)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid JSON in {label}: {config_path}") from exc
        if not isinstance(config, dict):
            raise ValidationError(f"{label} must contain a JSON object: {config_path}")


def artifact_path(results_root: Path, modality: str, index: int) -> Path:
    return results_root / modality / str(index) / ARTIFACT_NAMES[modality]


def _artifact_validation_error(modality: str, path: Path) -> Optional[str]:
    if path.is_symlink():
        return "may not be a symlink"
    if not _nonempty_file(path):
        return "is missing or empty"
    try:
        if modality == "text":
            if not path.read_text(encoding="utf-8").strip():
                raise ValueError("text is blank")
        elif modality == "image":
            from PIL import Image

            with Image.open(path) as image:
                if image.format != "PNG" or image.width < 1 or image.height < 1:
                    raise ValueError(
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
                raise ValueError("WAV has no decodable audio frames")
        else:
            return f"has unknown modality {modality!r}"
    except Exception as exc:
        return f"is invalid: {exc}"
    return None


def ensure_safe_output_destination(results_root: Path, index: int) -> None:
    _resolve_results_root(str(results_root))
    if results_root.is_symlink() or (
        results_root.exists() and not results_root.is_dir()
    ):
        raise ValidationError(
            f"results root must be a real directory: {results_root}"
        )
    for modality, filename in ARTIFACT_NAMES.items():
        modality_root = results_root / modality
        if modality_root.is_symlink() or (
            modality_root.exists() and not modality_root.is_dir()
        ):
            raise ValidationError(f"{modality_root} must be a real directory")
        index_dir = modality_root / str(index)
        if not index_dir.exists() and not index_dir.is_symlink():
            continue
        if index_dir.is_symlink() or not index_dir.is_dir():
            raise ValidationError(f"{index_dir} must be a real directory")
        unexpected = sorted(
            entry.name for entry in index_dir.iterdir() if entry.name != filename
        )
        if unexpected:
            raise ValidationError(
                f"{index_dir} contains unexpected entries: {unexpected}"
            )
        artifact = index_dir / filename
        if artifact.is_symlink():
            raise ValidationError(f"artifact may not be a symlink: {artifact}")
        if artifact.exists() and not artifact.is_file():
            raise ValidationError(f"artifact must be a regular file: {artifact}")


def ensure_safe_output_tree(results_root: Path, sample_count: int) -> None:
    _resolve_results_root(str(results_root))
    if results_root.is_symlink() or (
        results_root.exists() and not results_root.is_dir()
    ):
        raise ValidationError(
            f"results root must be a real directory: {results_root}"
        )
    if not results_root.exists():
        return

    unexpected_root = sorted(
        entry.name for entry in results_root.iterdir()
        if entry.name not in ARTIFACT_NAMES
    )
    if unexpected_root:
        raise ValidationError(
            f"results root contains unexpected entries: {unexpected_root}"
        )

    expected_indices = {str(index) for index in range(sample_count)}
    for modality in ARTIFACT_NAMES:
        modality_root = results_root / modality
        if not modality_root.exists() and not modality_root.is_symlink():
            continue
        if modality_root.is_symlink() or not modality_root.is_dir():
            raise ValidationError(f"{modality_root} must be a real directory")
        unexpected_indices = sorted(
            entry.name for entry in modality_root.iterdir()
            if entry.name not in expected_indices
        )
        if unexpected_indices:
            raise ValidationError(
                f"{modality_root} contains unexpected indices: {unexpected_indices}"
            )
        for entry in modality_root.iterdir():
            ensure_safe_output_destination(results_root, int(entry.name))


def verify_result_tree(results_root: Path, sample_count: int) -> List[str]:
    errors: List[str] = []
    if results_root.is_symlink():
        return [f"result root may not be a symlink: {results_root}"]
    if not results_root.is_dir():
        return [f"result root does not exist: {results_root}"]

    expected_modalities = set(ARTIFACT_NAMES)
    actual_modalities = {entry.name for entry in results_root.iterdir()}
    missing_modalities = sorted(expected_modalities - actual_modalities)
    extra_modalities = sorted(actual_modalities - expected_modalities)
    if missing_modalities:
        errors.append(f"missing modality directories: {missing_modalities}")
    if extra_modalities:
        errors.append(f"unexpected modality directories: {extra_modalities}")

    for modality, filename in ARTIFACT_NAMES.items():
        modality_root = results_root / modality
        if modality_root.is_symlink():
            errors.append(f"{modality}: directory may not be a symlink")
            continue
        if not modality_root.is_dir():
            if modality_root.exists():
                errors.append(f"{modality}: expected a directory")
            continue
        expected_indices = {str(i) for i in range(sample_count)}
        actual_indices = {entry.name for entry in modality_root.iterdir()}
        missing_indices = sorted(expected_indices - actual_indices, key=int)
        extra_indices = sorted(actual_indices - expected_indices)
        if missing_indices:
            errors.append(
                f"{modality}: missing {len(missing_indices)} index directories; "
                f"first={missing_indices[:10]}"
            )
        if extra_indices:
            errors.append(f"{modality}: unexpected index directories: {extra_indices[:10]}")

        for index in range(sample_count):
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
            artifact_error = _artifact_validation_error(modality, index_dir / filename)
            if artifact_error is not None:
                errors.append(
                    f"{modality}/{index}/{filename} {artifact_error}"
                )
    return errors


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_wrapping_quotes(text: str) -> str:
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def strip_meta_lead_sentence(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    if len(parts) != 2:
        return text
    first = parts[0].lower()
    meta_prefixes = (
        "write a sentence",
        "write one sentence",
        "describe the next page",
        "describe the scene",
        "generate the next page",
        "here is the narration",
    )
    return parts[1].strip() if first.startswith(meta_prefixes) else text


def extract_assistant_payload(decoded: str) -> str:
    matches = list(MMGPT_RESPONSE_PATTERN.finditer(decoded))
    if matches:
        return matches[-1].group(1).strip()
    if "[MMGPT]" in decoded:
        decoded = decoded.rsplit("[MMGPT]", 1)[-1]
    return re.sub(r"^\s*:\s*", "", decoded, count=1).strip()


def extract_modality_segments(
    response: str, modality: str, special_strings: Mapping[str, Mapping[str, str]]
) -> List[str]:
    special = special_strings[modality]
    pattern = re.compile(
        re.escape(special["sos"]) + r"(.*?)" + re.escape(special["eos"]),
        flags=re.DOTALL,
    )
    return [match.group(1) for match in pattern.finditer(response)]


def remove_modality_segments(
    text: str, special_strings: Mapping[str, Mapping[str, str]]
) -> str:
    cleaned = text
    for special in special_strings.values():
        cleaned = re.sub(
            re.escape(special["sos"]) + r".*?" + re.escape(special["eos"]),
            " ",
            cleaned,
            flags=re.DOTALL,
        )
    cleaned = re.sub(r"^\s*:\s*", "", cleaned, count=1)
    return normalize_whitespace(cleaned)


def parse_text_response(
    assistant_response: str, special_strings: Mapping[str, Mapping[str, str]]
) -> Tuple[str, str]:
    clean_text = remove_modality_segments(assistant_response, special_strings)
    narration = ""
    speech_text = ""

    json_match = re.search(r"\{.*\}", clean_text, flags=re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            narration = normalize_whitespace(parsed_json.get("narration", ""))
            speech_text = normalize_whitespace(
                parsed_json.get("speech", parsed_json.get("speech_text", ""))
            )
        except json.JSONDecodeError:
            pass

    narration_match = re.search(
        r"(?is)(?:^|\n)\s*narration\s*:\s*(.+?)(?=(?:\n\s*speech\s*:)|\Z)",
        clean_text,
    )
    speech_match = re.search(
        r"(?is)(?:^|\n)\s*speech\s*:\s*(.+?)\s*$",
        clean_text,
    )
    if narration_match:
        narration = normalize_whitespace(narration_match.group(1))
    if speech_match:
        speech_text = normalize_whitespace(speech_match.group(1))

    quoted_spans = re.findall(
        r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'',
        clean_text,
    )
    quoted_spans = [
        normalize_whitespace("".join(span))
        for span in quoted_spans
        if "".join(span).strip()
    ]
    if not speech_text and quoted_spans:
        speech_text = quoted_spans[-1]

    if not narration:
        lines = []
        for line in clean_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(
                r"^\s*(narration|speech)\s*:\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
            stripped = normalize_whitespace(stripped)
            if stripped:
                lines.append(stripped)
        if lines:
            narration = lines[0]
        if len(lines) > 1 and not speech_text:
            speech_text = lines[1]

    narration = strip_meta_lead_sentence(strip_wrapping_quotes(narration))
    speech_text = strip_wrapping_quotes(speech_text)
    if speech_text and narration:
        narration = normalize_whitespace(
            narration.replace(f'"{speech_text}"', "").replace(f"“{speech_text}”", "")
        )

    return narration or clean_text, speech_text
def story_fields(sample: BenchmarkInput) -> StoryFields:
    return StoryFields(
        current_page_text=sample.current_text,
        metadata_text=json.dumps(sample.book_metadata, ensure_ascii=False),
        condition_text=json.dumps(sample.next_page_condition, ensure_ascii=False),
    )


def build_text_instruction(fields: StoryFields) -> str:
    return (
        "You are writing the next page of a children's picture book.\n"
        "The current page image is provided before this instruction.\n"
        "Use the current page image, current page text, book metadata, and the next page condition.\n"
        "Output exactly two lines and nothing else.\n"
        "Line 1 must start with 'Narration:' followed by the actual next-page narration in story style.\n"
        "Line 2 must start with 'Speech:' followed by one short spoken line for an appropriate character.\n"
        "The narration must read like a finished book sentence, not like an instruction, explanation, or scene label.\n"
        "Keep Narration to 1-2 short sentences. Keep Speech to 3-12 words.\n"
        "Put only the spoken words after 'Speech:'. Do not add speaker names, bullets, JSON, image tokens, or speech tokens.\n"
        "Do not use meta phrases such as 'Write a sentence', 'Describe the scene', 'The next scene depicts', or 'Here is'.\n\n"
        "Example output:\n"
        "Narration: The little fox tiptoes into the moonlit garden and smiles at the silver flowers.\n"
        "Speech: They're glowing tonight!\n\n"
        f"Current page text:\n{fields.current_page_text}\n\n"
        f"Book metadata:\n{fields.metadata_text}\n\n"
        f"Next page condition:\n{fields.condition_text}"
    )


def build_image_instruction(fields: StoryFields, narration: str, speech_text: str) -> str:
    parts = [
        "Generate the next-page illustration for the same children's picture book.",
        "The current page image is provided before this instruction as a style and continuity reference.",
        "Return only the image.",
        "",
        f"Current page text:\n{fields.current_page_text}",
        "",
        f"Generated next page narration:\n{narration or fields.condition_text}",
    ]
    if speech_text:
        parts.extend(["", f"Suggested spoken line in the scene:\n{speech_text}"])
    parts.extend(
        [
            "",
            f"Book metadata:\n{fields.metadata_text}",
            "",
            f"Next page condition:\n{fields.condition_text}",
        ]
    )
    return "\n".join(parts)


def build_speech_instruction(fields: StoryFields, narration: str, speech_text: str) -> str:
    parts = [
        "Generate speech audio for one short character line from the next scene of a children's picture book.",
        "Return only speech audio. Do not return text.",
        "",
        f"Current page text:\n{fields.current_page_text}",
        "",
        f"Next page narration:\n{narration or fields.condition_text}",
    ]
    if speech_text:
        parts.extend(["", f"Read this line aloud exactly:\n{speech_text}"])
    else:
        parts.extend(
            [
                "",
                "Infer one short spoken line for an appropriate character from the context below and speak it naturally.",
            ]
        )
    parts.extend(
        [
            "",
            f"Book metadata:\n{fields.metadata_text}",
            "",
            f"Next page condition:\n{fields.condition_text}",
        ]
    )
    return "\n".join(parts)


class InferenceRuntime:
    """Lazy imports keep validation-only mode free of model initialization."""

    def __init__(self, args: argparse.Namespace, output_dir: Path) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "anygpt" / "src"))

        import torch
        import torchaudio
        from infer.cli_infer_chat_model import AnyGPTChatInference
        from m_utils.anything2token import modal_special_str, modality_tokens_to_string
        from m_utils.conversation import get_conv_template
        from transformers import GenerationConfig

        self.torch = torch
        self.torchaudio = torchaudio
        self.modal_special_str = modal_special_str
        self.modality_tokens_to_string = modality_tokens_to_string
        self.get_conv_template = get_conv_template

        if not torch.cuda.is_available():
            raise RuntimeError("AnyGPT inference requires a CUDA-visible GPU")

        os.environ["ANYGPT_DIFFUSION_MODEL_PATH"] = str(
            _resolve_repo_path(args.diffusion_model_path, "diffusion model", directory=True)
        )
        self.inferencer = AnyGPTChatInference(
            model_name_or_path=str(_resolve_repo_path(args.model_name_or_path, "AnyGPT model", directory=True)),
            image_tokenizer_path=str(_resolve_repo_path(args.image_tokenizer_path, "image tokenizer")),
            output_dir=str(output_dir),
            speech_tokenizer_path=str(_resolve_repo_path(args.speech_tokenizer_path, "speech tokenizer")),
            speech_tokenizer_config=str(
                _resolve_repo_path(args.speech_tokenizer_config, "speech tokenizer config")
            ),
            soundstorm_path=str(_resolve_repo_path(args.soundstorm_path, "SoundStorm checkpoint")),
        )

        self.generation_configs: Dict[str, Any] = {}
        for modality, path in (
            ("text", args.text_generation_config),
            ("image", args.image_generation_config),
            ("speech", args.speech_generation_config),
        ):
            config_path = _resolve_repo_path(path, f"{modality} generation config")
            with config_path.open("r", encoding="utf-8") as handle:
                self.generation_configs[modality] = GenerationConfig(**json.load(handle))

    def encode_image(self, image_path: Path) -> str:
        image_tokens = self.inferencer.encode_image(image_path=str(image_path))[0]
        return self.modality_tokens_to_string(tokens=image_tokens, modality="image")

    def build_prompt(
        self,
        content: str,
        *,
        force_modality: Optional[str] = None,
        response_prefix: Optional[str] = None,
    ) -> str:
        conversation = self.get_conv_template("MMGPT")
        conversation.append_message(conversation.roles[0], content.strip())
        if force_modality == "image":
            return conversation.get_prompt(force_image_generation=True)
        if force_modality == "speech":
            return conversation.get_prompt(force_speech_generation=True)
        return conversation.get_prompt(force_res_prefix=response_prefix)

    def generate(self, prompt: str, modality: str) -> str:
        input_ids = self.inferencer.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).input_ids.to(self.inferencer.device)
        with self.torch.no_grad():
            generated_ids = self.inferencer.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_configs[modality],
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = generated_ids.sequences
        decoded = self.inferencer.tokenizer.batch_decode(
            sequences.cpu(), skip_special_tokens=True
        )[0]
        return extract_assistant_payload(decoded)


def generate_text(
    runtime: InferenceRuntime,
    image_string: str,
    fields: StoryFields,
) -> Tuple[str, str]:
    prompt = runtime.build_prompt(
        f"{image_string}\n{build_text_instruction(fields)}",
        response_prefix="Narration:",
    )
    response = runtime.generate(prompt, "text")
    return parse_text_response(response, runtime.modal_special_str)


def generate_image(
    runtime: InferenceRuntime,
    image_string: str,
    fields: StoryFields,
    narration: str,
    speech_text: str,
    destination: Path,
    *,
    sample_index: int,
) -> int:
    prompt = runtime.build_prompt(
        f"{image_string}\n{build_image_instruction(fields, narration, speech_text)}",
        force_modality="image",
    )
    response = runtime.generate(prompt, "image")
    segments = extract_modality_segments(
        response, "image", runtime.modal_special_str
    )
    if not segments:
        return 0

    for segment_index, segment in enumerate(segments):
        image = runtime.inferencer.decode_image(segment)
        if segment_index == 0:
            image.save(destination, format="PNG")
    if len(segments) > 1:
        print(
            f"[warn] sample {sample_index}: model returned {len(segments)} images; "
            "saved only the first canonical artifact",
            flush=True,
        )
    return len(segments)


def generate_speech(
    runtime: InferenceRuntime,
    fields: StoryFields,
    narration: str,
    speech_text: str,
    destination: Path,
    *,
    sample_index: int,
) -> int:
    prompt = runtime.build_prompt(
        build_speech_instruction(fields, narration, speech_text),
        force_modality="speech",
    )
    response = runtime.generate(prompt, "speech")
    segments = extract_modality_segments(
        response, "speech", runtime.modal_special_str
    )
    if not segments:
        return 0

    for segment_index, segment in enumerate(segments):
        waveform = runtime.inferencer.decode_speech(segment, prompt_path=None)
        if segment_index == 0:
            runtime.torchaudio.save(
                str(destination),
                waveform,
                runtime.inferencer.speech_tokenizer.sample_rate,
            )
    if len(segments) > 1:
        print(
            f"[warn] sample {sample_index}: model returned {len(segments)} speech segments; "
            "saved only the first canonical artifact",
            flush=True,
        )
    return len(segments)


def run_inference(
    args: argparse.Namespace,
    samples: Sequence[BenchmarkInput],
    results_root: Path,
) -> int:
    start_index = args.start_index
    end_index = len(samples) - 1 if args.end_index is None else args.end_index
    if start_index < 0 or end_index < start_index or end_index >= len(samples):
        raise ValidationError(
            f"invalid inclusive index range {start_index}..{end_index}; "
            f"valid range is 0..{len(samples) - 1}"
        )

    selected = samples[start_index : end_index + 1]
    ensure_safe_output_tree(results_root, len(samples))
    runtime = InferenceRuntime(args, results_root)

    processed = 0
    failed: List[int] = []
    missing_image: List[int] = []
    missing_speech: List[int] = []
    for sample in selected:
        print(f"[run  {sample.index:03d}] {sample.sample_id}", flush=True)
        ensure_safe_output_destination(results_root, sample.index)
        for modality in ARTIFACT_NAMES:
            artifact_path(results_root, modality, sample.index).parent.mkdir(
                parents=True, exist_ok=True
            )

        try:
            fields = story_fields(sample)
            image_string = runtime.encode_image(sample.current_image)

            narration, speech_text = generate_text(runtime, image_string, fields)
            artifact_path(results_root, "text", sample.index).write_text(
                narration, encoding="utf-8"
            )

            image_count = generate_image(
                runtime,
                image_string,
                fields,
                narration,
                speech_text,
                artifact_path(results_root, "image", sample.index),
                sample_index=sample.index,
            )
            if image_count == 0:
                missing_image.append(sample.index)

            speech_count = generate_speech(
                runtime,
                fields,
                narration,
                speech_text,
                artifact_path(results_root, "speech", sample.index),
                sample_index=sample.index,
            )
            if speech_count == 0:
                missing_speech.append(sample.index)

            processed += 1
            print(
                f"[done {sample.index:03d}] "
                f"text=1 image={image_count} speech={speech_count}",
                flush=True,
            )
        except Exception:
            failed.append(sample.index)
            print(
                f"[fail {sample.index:03d}] {sample.sample_id}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

    print(
        f"processed_samples={processed} failed_samples={len(failed)} "
        f"missing_image={len(missing_image)} missing_speech={len(missing_speech)}",
        flush=True,
    )
    if failed:
        print(f"failed_indices={failed}", file=sys.stderr, flush=True)
    if missing_image:
        print(f"missing_image_indices={missing_image}", file=sys.stderr, flush=True)
    if missing_speech:
        print(f"missing_speech_indices={missing_speech}", file=sys.stderr, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 900 AnyGPT Omni-StoryBench image/text/speech triplets"
    )
    parser.add_argument(
        "--dataset-root", default=str(DEFAULT_DATASET_ROOT),
        help="Path to the downloaded Omni-StoryBench dataset (default: evaluation/downloaded_dataset)",
    )
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT),
        help="Output directory (default: evaluation/baselines/AnyGPT)",
    )
    parser.add_argument("--model-name-or-path", default="models/anygpt/chat")
    parser.add_argument(
        "--image-tokenizer-path", default="models/seed-tokenizer-2/seed_quantizer.pt"
    )
    parser.add_argument("--speech-tokenizer-path", default="models/speechtokenizer/ckpt.dev")
    parser.add_argument(
        "--speech-tokenizer-config", default="models/speechtokenizer/config.json"
    )
    parser.add_argument(
        "--soundstorm-path", default="models/soundstorm/speechtokenizer_soundstorm_mls.pt"
    )
    parser.add_argument(
        "--diffusion-model-path", default="models/stable-diffusion-2-1-unclip"
    )
    parser.add_argument(
        "--text-generation-config", default="config/text_generate_config.json"
    )
    parser.add_argument(
        "--image-generation-config", default="config/image_generate_config.json"
    )
    parser.add_argument(
        "--speech-generation-config", default="config/speech_generate_config.json"
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=899)
    parser.add_argument(
        "--validate-only",
        "--dry-run",
        dest="validate_only",
        action="store_true",
        help="validate all 900 inputs and local assets without loading models or creating output",
    )
    parser.add_argument(
        "--verify-results-only",
        action="store_true",
        help="validate dataset and exact 900-sample artifact tree without loading models",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = _resolve_dataset_root(args.dataset_root)
    results_root = _resolve_results_root(args.results_root)
    os.chdir(REPO_ROOT)
    validate_runtime_assets(args)
    samples = validate_and_load_inputs(dataset_root)
    print(
        f"validated samples={len(samples)} unique_ids={len({sample.sample_id for sample in samples})} "
        f"canonical_index=0..{len(samples) - 1} "
        "ground_truth_columns_read=[] sample_json_contents_read=false",
        flush=True,
    )

    if args.verify_results_only:
        errors = verify_result_tree(results_root, len(samples))
        if errors:
            print(f"result verification failed with {len(errors)} issue(s):", file=sys.stderr)
            for error in errors[:100]:
                print(f"  - {error}", file=sys.stderr)
            if len(errors) > 100:
                print(f"  ... {len(errors) - 100} more", file=sys.stderr)
            return 1
        print("result verification passed: 900 complete artifact-only triplets")
        return 0

    if args.validate_only:
        print("validation-only passed; models were not loaded and results were not modified")
        return 0

    return run_inference(args, samples, results_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        raise SystemExit(2)
