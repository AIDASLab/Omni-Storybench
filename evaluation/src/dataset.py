"""Load the local Hugging Face release into the evaluator's canonical schema."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.common import HARNESS_ROOT, ensure_inside_harness


DATASET_REPO_ID = "snu-aidas/Omni-StoryBench"
DATASET_SCHEMA_VERSION = 1
EXPECTED_RECORD_COUNT = 900
PARQUET_RELATIVE_PATH = Path("data/omni_storybench.parquet")
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "OMNISTORYBENCH_DATASET_ROOT",
        str(HARNESS_ROOT / "downloaded_dataset"),
    )
)

REQUIRED_PARQUET_COLUMNS = {
    "id",
    "source",
    "book",
    "from_key",
    "to_key",
    "current_image",
    "current_text",
    "current_text_path",
    "next_image",
    "next_text",
    "next_text_path",
    "genre",
    "topic",
    "style",
    "narrative_tense",
    "narrative_perspective",
    "scene",
    "text_instruction",
    "ambient_sound",
    "speech_utterance",
    "speech_speaker",
    "speech_emotion",
    "speech_speed",
    "speech_pitch",
    "speech_gender",
    "condition_characters",
    "book_characters",
    "sample_path",
    "metadata_path",
}


@dataclass(frozen=True)
class DatasetBundle:
    dataset_root: Path
    records: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    manifest_path: Optional[Path] = None


def _string(row: Mapping[str, Any], key: str, row_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"Dataset row {row_number} has an invalid {key!r} value"
        )
    return value


def _required_string(row: Mapping[str, Any], key: str, row_number: int) -> str:
    value = _string(row, key, row_number)
    if not value.strip():
        raise ValueError(
            f"Dataset row {row_number} has an empty {key!r} value"
        )
    return value


def _json_list(value: Any, label: str, row_number: int) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Dataset row {row_number} has invalid JSON in {label}: {exc}"
            ) from exc
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(
            f"Dataset row {row_number} field {label} must be a JSON list of objects"
        )
    return [dict(item) for item in value]


def _repo_file(
    dataset_root: Path,
    relative_path: str,
    label: str,
    row_number: int,
    *,
    validate_files: bool,
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(
            f"Dataset row {row_number} field {label} must be repo-relative: "
            f"{relative_path}"
        )
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Dataset row {row_number} field {label} escapes the dataset root: "
            f"{relative_path}"
        ) from exc
    if validate_files and not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset row {row_number} field {label} does not exist: {resolved}"
        )
    return resolved


def _canonical_sample_view(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the public sample shape represented by one canonical record."""

    metadata = record["metadata"]["metadata"]
    condition = record["next_page_condition"]["condition_json"]
    speech = record["speech_metadata_next_page"]["parsed"]["speaker"]
    current_page = record["current_page"]
    next_page = record["next_page"]

    return {
        "id": record["id"],
        "source": record["source"],
        "book": record["book"],
        "transition": {
            "from": record["prev_key"],
            "to": record["next_key"],
        },
        "book_metadata_path": record["_repo_paths"]["metadata"],
        "input": {
            "current_page": {
                "image": record["_repo_paths"]["current_image"],
                "text": current_page["text"],
                "text_path": record["_repo_paths"]["current_text"],
            },
            "book_metadata": metadata,
        },
        "next_page_condition": {
            "characters": condition["characters"],
            "scene": condition["scene"],
            "text_instruction": condition["text content"],
            "ambient_sound": condition["ambient_sound"],
        },
        "ground_truth": {
            "next_page": {
                "image": record["_repo_paths"]["next_image"],
                "text": next_page["text"],
                "text_path": record["_repo_paths"]["next_text"],
            },
            "speech": {
                "utterance": speech["line"],
                "speaker": speech["name"],
                "emotion": speech["emotion"],
                "speed": speech["speed"],
                "pitch": speech["pitch"],
                "gender": speech["gender"],
            },
        },
    }


def canonical_record_from_parquet_row(
    row: Mapping[str, Any],
    dataset_root: Path,
    row_number: int,
    *,
    validate_files: bool = True,
    validate_sample_json: bool = True,
) -> Dict[str, Any]:
    """Convert one public parquet row to the schema consumed by all judges."""

    dataset_root = dataset_root.expanduser().resolve()
    source = _required_string(row, "source", row_number)
    book = _required_string(row, "book", row_number)
    from_key = _required_string(row, "from_key", row_number)
    to_key = _required_string(row, "to_key", row_number)
    record_id = _required_string(row, "id", row_number)
    expected_id = f"{source}/{book}/{from_key}__{to_key}"
    if record_id != expected_id:
        raise ValueError(
            f"Dataset row {row_number} id mismatch: {record_id!r} != "
            f"{expected_id!r}"
        )

    repo_paths = {
        "current_image": _required_string(row, "current_image", row_number),
        "current_text": _required_string(row, "current_text_path", row_number),
        "next_image": _required_string(row, "next_image", row_number),
        "next_text": _required_string(row, "next_text_path", row_number),
        "sample": _required_string(row, "sample_path", row_number),
        "metadata": _required_string(row, "metadata_path", row_number),
    }
    resolved_paths = {
        key: _repo_file(
            dataset_root,
            value,
            key,
            row_number,
            validate_files=validate_files,
        )
        for key, value in repo_paths.items()
    }

    condition = {
        "characters": _json_list(
            row.get("condition_characters"),
            "condition_characters",
            row_number,
        ),
        "scene": _required_string(row, "scene", row_number),
        "text content": _required_string(row, "text_instruction", row_number),
        "ambient_sound": _required_string(row, "ambient_sound", row_number),
    }
    metadata = {
        "genre": _string(row, "genre", row_number),
        "topic": _string(row, "topic", row_number),
        "style": _string(row, "style", row_number),
        "narrative_tense": _string(
            row,
            "narrative_tense",
            row_number,
        ),
        "narrative_perspective": _string(
            row,
            "narrative_perspective",
            row_number,
        ),
        "characters": _json_list(
            row.get("book_characters"),
            "book_characters",
            row_number,
        ),
    }
    speaker = {
        "name": _required_string(row, "speech_speaker", row_number),
        "line": _required_string(row, "speech_utterance", row_number),
        "emotion": _required_string(row, "speech_emotion", row_number),
        "speed": _required_string(row, "speech_speed", row_number),
        "pitch": _required_string(row, "speech_pitch", row_number),
        "gender": _required_string(row, "speech_gender", row_number),
    }

    record: Dict[str, Any] = {
        "id": record_id,
        "source": source,
        "book": book,
        "prev_key": from_key,
        "next_key": to_key,
        "metadata": {"metadata": metadata},
        "current_page": {
            "image_path": str(resolved_paths["current_image"]),
            "text": _required_string(row, "current_text", row_number),
            "text_path": str(resolved_paths["current_text"]),
        },
        "next_page_condition": {
            "condition_json": condition,
            "condition_text": json.dumps(condition, ensure_ascii=False),
        },
        "next_page": {
            "image_path": str(resolved_paths["next_image"]),
            "text": _required_string(row, "next_text", row_number),
            "text_path": str(resolved_paths["next_text"]),
        },
        "speech_metadata_next_page": {
            "raw": json.dumps(
                {"speaker": speaker},
                ensure_ascii=False,
                indent=2,
            ),
            "parsed": {"speaker": speaker},
        },
        "_repo_paths": repo_paths,
    }

    if validate_sample_json:
        try:
            sample = json.loads(
                resolved_paths["sample"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read sample JSON for dataset row {row_number}: {exc}"
            ) from exc
        expected_sample = _canonical_sample_view(record)
        if sample != expected_sample:
            raise ValueError(
                f"Dataset row {row_number} does not match {repo_paths['sample']}"
            )

    record.pop("_repo_paths")
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_record_sequence(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Omni-StoryBench must contain {EXPECTED_RECORD_COUNT} records; "
            f"found {len(records)}"
        )
    ids = [record.get("id") for record in records]
    if not all(isinstance(record_id, str) and record_id for record_id in ids):
        raise ValueError("Every dataset record must have a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset record ids must be unique")


def load_downloaded_dataset(dataset_root: Path) -> DatasetBundle:
    """Load and strictly validate a local Hugging Face dataset snapshot."""

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    parquet_path = (dataset_root / PARQUET_RELATIVE_PATH).resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Canonical dataset parquet does not exist: {parquet_path}"
        )

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Loading the downloaded Omni-StoryBench snapshot requires pyarrow. "
            "Run manifest preparation in the configured dataset Conda environment."
        ) from exc

    table = parquet.read_table(parquet_path)
    missing_columns = REQUIRED_PARQUET_COLUMNS - set(table.column_names)
    if missing_columns:
        raise ValueError(
            "Dataset parquet is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    rows = table.to_pylist()
    records = [
        canonical_record_from_parquet_row(
            row,
            dataset_root,
            row_number,
        )
        for row_number, row in enumerate(rows, start=1)
    ]
    _validate_record_sequence(records)
    metadata = {
        "repo_id": DATASET_REPO_ID,
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256(parquet_path),
        "record_count": len(records),
        "index_base": 0,
        "first_id": records[0]["id"],
        "last_id": records[-1]["id"],
    }
    return DatasetBundle(
        dataset_root=dataset_root,
        records=records,
        metadata=metadata,
    )


def write_dataset_manifest(bundle: DatasetBundle, output_path: Path) -> Path:
    output_path = ensure_inside_harness(output_path, "dataset manifest path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset": bundle.metadata,
        "records": bundle.records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_dataset_manifest(
    manifest_path: Path,
    *,
    expected_dataset_root: Optional[Path] = None,
) -> DatasetBundle:
    manifest_path = ensure_inside_harness(
        manifest_path,
        "dataset manifest path",
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read dataset manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest schema in {manifest_path}: "
            f"{payload.get('schema_version')!r}"
        )

    metadata = payload.get("dataset")
    records = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise ValueError(
            f"Dataset manifest must contain dataset metadata and records: "
            f"{manifest_path}"
        )
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(
            f"Dataset manifest records must be JSON objects: {manifest_path}"
        )
    _validate_record_sequence(records)
    if metadata.get("record_count") != len(records):
        raise ValueError(
            f"Dataset manifest record_count does not match its records: "
            f"{manifest_path}"
        )

    root_value = metadata.get("dataset_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"Dataset manifest has no dataset_root: {manifest_path}")
    dataset_root = Path(root_value).expanduser().resolve()
    if expected_dataset_root is not None:
        expected = expected_dataset_root.expanduser().resolve()
        if dataset_root != expected:
            raise ValueError(
                f"Dataset manifest root mismatch: {dataset_root} != {expected}"
            )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root recorded by manifest does not exist: {dataset_root}"
        )

    return DatasetBundle(
        dataset_root=dataset_root,
        records=[dict(record) for record in records],
        metadata=dict(metadata),
        manifest_path=manifest_path,
    )


def load_evaluation_dataset(
    dataset_root: Path,
    dataset_manifest_path: Optional[Path] = None,
) -> DatasetBundle:
    if dataset_manifest_path is not None:
        return load_dataset_manifest(
            dataset_manifest_path,
            expected_dataset_root=dataset_root,
        )
    return load_downloaded_dataset(dataset_root)
