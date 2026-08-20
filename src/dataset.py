"""Loader for the Omni-StoryBench release (HF dataset repo layout).

The release ships a parquet index (``data/omni_storybench.parquet``) whose media
columns hold repo-relative paths, plus ``images/``, ``texts/``, ``samples/`` and
``metadata/`` asset directories. :func:`load_dataset` reads that index and
normalizes each row into the entry dict the orchestrators consume, resolving all
paths against the dataset root so they are absolute and CWD-independent.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = os.environ.get(
    "OMNI_STORYBENCH_DATASET",
    str(PROJECT_ROOT / "data"),
)

PARQUET_NAME = "omni_storybench.parquet"

BOOK_METADATA_FIELDS = ("genre", "topic", "style", "narrative_tense", "narrative_perspective")
SPEECH_FIELDS = ("utterance", "speaker", "emotion", "speed", "pitch", "gender")


def _resolve_index(dataset_path: str) -> tuple[str, str]:
    """Return (parquet_path, dataset_root) for a dataset dir or parquet file."""
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

    if os.path.isdir(dataset_path):
        parquet_path = os.path.join(dataset_path, "data", PARQUET_NAME)
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"No data/{PARQUET_NAME} under dataset directory {dataset_path}")
    elif os.path.isfile(dataset_path):
        parquet_path = dataset_path
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Assets live next to data/, i.e. <root>/{data,images,texts,samples,metadata}.
    index_dir = os.path.dirname(parquet_path)
    root = os.path.dirname(index_dir) if os.path.basename(index_dir) == "data" else index_dir
    if not os.path.isdir(os.path.join(root, "images")):
        raise FileNotFoundError(
            f"Resolved dataset root {root} has no images/ directory; "
            f"expected the Omni-StoryBench repo layout around {parquet_path}."
        )
    return parquet_path, root


def _normalize(row: Dict[str, Any], index: int, root: str) -> Dict[str, Any]:
    """Turn one parquet row into an orchestrator entry with absolute paths."""
    def abspath(rel: str) -> str:
        return os.path.join(root, rel) if rel else rel

    return {
        # Parquet row order is the evaluation harness' canonical 0..899 sample index.
        "index": index,
        "id": row["id"],
        "source": row["source"],
        "book": row["book"],
        "from_key": row["from_key"],
        "to_key": row["to_key"],
        "current_page": {
            "image_path": abspath(row["current_image"]),
            "text": row["current_text"],
            "text_path": abspath(row["current_text_path"]),
        },
        "book_metadata": {
            **{field: row[field] for field in BOOK_METADATA_FIELDS},
            "characters": json.loads(row["book_characters"]),
        },
        "next_page_condition": {
            "characters": json.loads(row["condition_characters"]),
            "scene": row["scene"],
            "text_instruction": row["text_instruction"],
            "ambient_sound": row["ambient_sound"],
        },
        # Ground truth below: unused for generation, kept for evaluation.
        "next_page": {
            "image_path": abspath(row["next_image"]),
            "text": row["next_text"],
            "text_path": abspath(row["next_text_path"]),
        },
        "speech": {field: row[f"speech_{field}"] for field in SPEECH_FIELDS},
        "sample_path": abspath(row["sample_path"]),
        "metadata_path": abspath(row["metadata_path"]),
    }


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load the benchmark as a list of entries, in parquet row order.

    Args:
        dataset_path: The dataset directory or its parquet index.
    """
    import pyarrow.parquet as pq

    parquet_path, root = _resolve_index(dataset_path)
    rows = pq.read_table(parquet_path).to_pylist()
    return [_normalize(row, index, root) for index, row in enumerate(rows)]
