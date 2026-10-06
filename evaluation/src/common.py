from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


HARNESS_ROOT = Path(__file__).resolve().parents[1]

TEXT_JUDGE_CATEGORIES = [
    "alignment_with_metadata",
    "natural_flow_from_current_narration",
    "satisfaction_of_generation_conditions",
    "semantic_consistency_with_ground_truth",
]

IMAGE_JUDGE_CATEGORIES = [
    "alignment_with_metadata",
    "visual_narrative_continuity_from_current_scene",
    "satisfaction_of_generation_conditions",
    "semantic_consistency_with_ground_truth",
]

SPEECH_JUDGE_CATEGORIES = [
    "alignment_with_metadata",
    "naturalness_and_conversational_relevance",
    "satisfaction_of_generation_conditions",
    "semantic_consistency_and_persona_match_with_ground_truth",
]

OMNI_JUDGE_CATEGORIES = [
    "alignment_with_metadata",
    "natural_multimodal_continuity_from_current_page",
    "satisfaction_of_generation_conditions",
    "multimodal_semantic_consistency_with_ground_truth",
]

JUDGE_CATEGORIES_BY_MODALITY = {
    "text": TEXT_JUDGE_CATEGORIES,
    "image": IMAGE_JUDGE_CATEGORIES,
    "speech": SPEECH_JUDGE_CATEGORIES,
    "joint": OMNI_JUDGE_CATEGORIES,
}


def ensure_inside_harness(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(HARNESS_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {HARNESS_ROOT}; got {resolved}") from exc
    return resolved


def write_json(path: Path, payload: Any) -> None:
    output_path = ensure_inside_harness(path, "output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def natural_sort_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_missing_reason(sample_dir: Path) -> Optional[str]:
    missing_path = sample_dir / "missing.txt"
    if missing_path.is_file():
        return read_text_file(missing_path) or f"Missing candidate artifact in {sample_dir}"
    return None


def build_missing_candidate(
    modality: str,
    sample_dir: Path,
    default_reason: str,
) -> Dict[str, Any]:
    return {
        "status": "missing",
        "sample_dir": str(sample_dir),
        "path": None,
        "missing_reason": read_missing_reason(sample_dir) or default_reason,
        "modality": modality,
    }


def list_candidate_files(
    sample_dir: Path,
    *,
    preferred_names: Sequence[str],
    patterns: Sequence[str],
    excluded_names: Sequence[str] = (),
    excluded_name_tokens: Sequence[str] = (),
) -> List[Path]:
    excluded = set(excluded_names)
    lowered_tokens = [token.lower() for token in excluded_name_tokens]
    candidates: List[Path] = []
    seen = set()

    def add(path: Path) -> None:
        lowered_name = path.name.lower()
        if not path.is_file() or path.name in excluded or path in seen:
            return
        if any(token in lowered_name for token in lowered_tokens):
            return
        candidates.append(path)
        seen.add(path)

    for name in preferred_names:
        add(sample_dir / name)
    for pattern in patterns:
        for path in sorted(sample_dir.glob(pattern), key=natural_sort_key):
            add(path)

    return candidates


def resolve_text_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "text" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "text",
            sample_dir,
            f"Text sample directory does not exist: {sample_dir}",
        )
    candidates = list_candidate_files(
        sample_dir,
        preferred_names=[
            "generated.txt",
            "generated_text.txt",
            "text.txt",
            "candidate.txt",
            "output.txt",
        ],
        patterns=[
            "page_*.txt",
            "generated_text_*.txt",
            "generated_[0-9]*.txt",
            "text_[0-9]*.txt",
            "candidate*.txt",
            "output*.txt",
            "*.txt",
        ],
        excluded_names=[
            "missing.txt",
            "generated_speech_text.txt",
            "raw_text_response.txt",
            "text_prompt.txt",
            "prompt.txt",
        ],
        excluded_name_tokens=["prompt", "raw", "response", "speech"],
    )
    if not candidates:
        return build_missing_candidate(
            "text",
            sample_dir,
            f"Text candidate file not found in {sample_dir}",
        )

    generated_path = candidates[0]
    text = read_text_file(generated_path)
    if not text:
        return {
            "status": "empty",
            "sample_dir": str(sample_dir),
            "path": str(generated_path),
            "all_candidate_paths": [str(path) for path in candidates],
            "text": "",
            "missing_reason": "Text candidate file is empty.",
            "modality": "text",
        }
    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(generated_path),
        "all_candidate_paths": [str(path) for path in candidates],
        "text": text,
        "modality": "text",
    }


def resolve_image_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "image" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "image",
            sample_dir,
            f"Image sample directory does not exist: {sample_dir}",
        )
    candidates = list_candidate_files(
        sample_dir,
        preferred_names=["generated.png"],
        patterns=[
            "page_*.png",
            "page_*.jpg",
            "page_*.jpeg",
            "generated_*.png",
            "generated_*.jpg",
            "generated_*.jpeg",
            "generated_*.webp",
            "image_*.png",
            "image_*.jpg",
            "image_*.jpeg",
            "image_*.webp",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.webp",
        ],
    )
    if not candidates:
        return build_missing_candidate(
            "image",
            sample_dir,
            f"Image candidate file not found in {sample_dir}",
        )
    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(candidates[0]),
        "all_candidate_paths": [str(path) for path in candidates],
        "modality": "image",
    }


def resolve_speech_candidate(results_root: Path, dataset_index: int) -> Dict[str, Any]:
    sample_dir = results_root / "speech" / str(dataset_index)
    if not sample_dir.is_dir():
        return build_missing_candidate(
            "speech",
            sample_dir,
            f"Speech sample directory does not exist: {sample_dir}",
        )
    candidates = list_candidate_files(
        sample_dir,
        preferred_names=["generated.wav"],
        patterns=[
            "page_*.wav",
            "generated_*.wav",
            "generated_*.flac",
            "generated_*.mp3",
            "generated_*.ogg",
            "generated_*.m4a",
            "speech_*.wav",
            "speech_*.flac",
            "speech_*.mp3",
            "speech_*.ogg",
            "speech_*.m4a",
            "*.wav",
            "*.flac",
            "*.mp3",
            "*.ogg",
            "*.m4a",
        ],
    )
    if not candidates:
        return build_missing_candidate(
            "speech",
            sample_dir,
            f"Speech candidate audio file not found in {sample_dir}",
        )
    return {
        "status": "ok",
        "sample_dir": str(sample_dir),
        "path": str(candidates[0]),
        "all_candidate_paths": [str(path) for path in candidates],
        "all_audio_paths": [str(path) for path in candidates],
        "modality": "speech",
    }


def resolve_all_candidates(results_root: Path, dataset_index: int) -> Dict[str, Dict[str, Any]]:
    return {
        "text": resolve_text_candidate(results_root, dataset_index),
        "image": resolve_image_candidate(results_root, dataset_index),
        "speech": resolve_speech_candidate(results_root, dataset_index),
    }


def extract_ground_truth_speech_text(record: Dict[str, Any]) -> str:
    speech_metadata = record.get("speech_metadata_next_page", {})
    parsed_metadata = (
        speech_metadata.get("parsed", speech_metadata)
        if isinstance(speech_metadata, dict)
        else speech_metadata
    )
    if isinstance(parsed_metadata, dict):
        speaker = parsed_metadata.get("speaker")
        if isinstance(speaker, dict):
            line = speaker.get("line")
            if isinstance(line, str) and line.strip():
                return line.strip()

        lines: List[str] = []
        for key, value in parsed_metadata.items():
            if not key.startswith("speaker_") or not isinstance(value, dict):
                continue
            line = value.get("line")
            if isinstance(line, str) and line.strip():
                lines.append(line.strip())
        if lines:
            return " ".join(lines)

        line = parsed_metadata.get("line")
        if isinstance(line, str) and line.strip():
            return line.strip()

    next_page = record.get("next_page", {})
    for key in ("text_processed", "text_speech_sep", "text"):
        value = next_page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_record_metadata(dataset_index: int, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_index": dataset_index,
        "source": record.get("source", "unknown"),
        "book": record.get("book", "unknown"),
        "prev_key": record.get("prev_key", "unknown"),
        "next_key": record.get("next_key", "unknown"),
    }


def selected_indices(total: int, start_index: int, end_index: Optional[int], limit: int) -> List[int]:
    start = max(start_index, 0)
    end = total if end_index is None else min(end_index, total)
    if start >= end:
        raise ValueError(f"Invalid range: start_index={start}, end_index={end}")
    values = list(range(start, end))
    if limit > 0:
        values = values[:limit]
    return values


def chunked(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def mean_zero_filled(values: Iterable[Any]) -> Dict[str, Any]:
    numeric: List[float] = []
    zero_filled_count = 0
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric.append(float(value))
        else:
            numeric.append(0.0)
            zero_filled_count += 1
    return {
        "average": sum(numeric) / len(numeric) if numeric else 0.0,
        "zero_filled_count": zero_filled_count,
    }


def score_from_judge(judge: Any, category: str) -> Optional[float]:
    if not isinstance(judge, dict):
        return None
    item = judge.get(category)
    if not isinstance(item, dict):
        return None
    score = item.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    return None


def average_judge_scores(records: Sequence[Dict[str, Any]], field_name: str, categories: Sequence[str]) -> Dict[str, Any]:
    sample_means: List[float] = []
    category_values: Dict[str, List[Optional[float]]] = {category: [] for category in categories}
    zero_filled_record_count = 0

    for record in records:
        judge = record.get(field_name)
        sample_scores: List[float] = []
        record_zero_filled = False
        for category in categories:
            score = score_from_judge(judge, category)
            category_values[category].append(score)
            if score is None:
                sample_scores.append(0.0)
                record_zero_filled = True
            else:
                sample_scores.append(score)
        sample_means.append(sum(sample_scores) / len(sample_scores) if sample_scores else 0.0)
        if record_zero_filled:
            zero_filled_record_count += 1

    return {
        "average": sum(sample_means) / len(sample_means) if sample_means else 0.0,
        "category_averages": {
            category: mean_zero_filled(values)["average"]
            for category, values in category_values.items()
        },
        "zero_filled_record_count": zero_filled_record_count,
        "zero_filled_category_count": {
            category: sum(1 for value in values if value is None)
            for category, values in category_values.items()
        },
    }


def status_counts(records: Sequence[Dict[str, Any]], key: str = "status") -> Dict[str, int]:
    return dict(Counter(str(record.get(key, "unknown")) for record in records))
