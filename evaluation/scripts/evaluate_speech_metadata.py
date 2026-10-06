#!/usr/bin/env python3
"""Evaluate generated speech metadata against Omni StoryBench ground truth."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
HARNESS_ROOT = THIS_DIR.parent
sys.path = [path for path in sys.path if "/.local/lib/python" not in path]
sys.path.insert(0, str(HARNESS_ROOT))

from src.dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_evaluation_dataset,
)

ALLOWED_WRITE_ROOT = Path(
    os.environ.get("OMNISTORYBENCH_ALLOWED_WRITE_ROOT", str(HARNESS_ROOT))
).expanduser().resolve()
ALLOWED_CACHE_ROOTS = [
    (Path.home() / ".cache").resolve(),
    Path(os.environ.get("TMPDIR", "/tmp")).expanduser().resolve(),
    HARNESS_ROOT.resolve(),
]
DEFAULT_SPEECH_ROOT = Path(
    os.environ.get(
        "OMNISTORYBENCH_SPEECH_ROOT",
        str(HARNESS_ROOT / "baselines" / "speech"),
    )
)
DEFAULT_CLASSIFIER_PY = THIS_DIR / "speech_metadata_classifier.py"
SCORED_FIELDS = ("emotion", "speed", "pitch", "gender")


def ensure_under_allowed_write_root(path: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else ALLOWED_WRITE_ROOT / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ALLOWED_WRITE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay inside {ALLOWED_WRITE_ROOT}; got {resolved}"
        ) from exc
    return resolved


def ensure_under_allowed_cache_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    for root in ALLOWED_CACHE_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in ALLOWED_CACHE_ROOTS)
    raise ValueError(f"{label} must stay inside one of: {allowed}; got {resolved}")


def load_classifier_module(classifier_py: Path) -> Any:
    if not classifier_py.exists():
        raise FileNotFoundError(f"Classifier source not found: {classifier_py}")

    module_name = "standalone_speech_metadata_classifier"
    spec = importlib.util.spec_from_file_location(module_name, classifier_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load classifier module from {classifier_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_allowed_cache(cache_dir: Path) -> None:
    cache_dir = ensure_under_allowed_cache_root(cache_dir, "--cache-dir")
    hf_home = cache_dir / "huggingface"
    hf_hub_cache = hf_home / "hub"

    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)


def parse_raw_speech_metadata(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def first_speaker(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def speaker_candidates(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    speaker = first_speaker(parsed.get("speaker"))
    if speaker:
        candidates.append(speaker)

    numbered_keys = sorted(
        (
            key
            for key in parsed
            if key.startswith("speaker_") and key[len("speaker_"):].isdigit()
        ),
        key=lambda key: int(key[len("speaker_"):]),
    )
    for key in numbered_keys:
        speaker = first_speaker(parsed.get(key))
        if speaker:
            candidates.append(speaker)

    return candidates


def extract_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("speech_metadata_next_page")
    if not isinstance(metadata, dict):
        return {}

    parsed = metadata.get("parsed")
    if not isinstance(parsed, dict):
        parsed = parse_raw_speech_metadata(metadata.get("raw"))

    if not isinstance(parsed, dict):
        return {}

    speakers = speaker_candidates(parsed)
    speaker = next(
        (
            item
            for item in speakers
            if isinstance(item.get("line"), str) and item["line"].strip()
        ),
        speakers[0] if speakers else {},
    )
    return {key: speaker.get(key) for key in ("name", "line", *SCORED_FIELDS) if key in speaker}


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def compare_metadata(candidate: Optional[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, bool]:
    if candidate is None:
        return {field: False for field in SCORED_FIELDS}
    return {
        field: normalize_label(candidate.get(field)) == normalize_label(ground_truth.get(field))
        for field in SCORED_FIELDS
    }


def find_candidate_wav(item_dir: Path, wav_glob: str) -> Optional[Path]:
    wavs = sorted(path for path in item_dir.glob(wav_glob) if path.is_file())
    if wavs:
        return wavs[0]
    return None


def count_label(counter: Counter[str], value: Any) -> None:
    label = normalize_label(value)
    if label:
        counter[label] += 1


def init_frequency_block() -> Dict[str, Dict[str, Counter[str]]]:
    return {
        field: {
            "candidate": Counter(),
            "ground_truth": Counter(),
            "correct_candidate": Counter(),
        }
        for field in SCORED_FIELDS
    }


def counter_to_plain(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items()))


def build_summary(
    entries: Sequence[Dict[str, Any]],
    frequencies: Dict[str, Dict[str, Counter[str]]],
    total_expected: int,
) -> Dict[str, Any]:
    missing_indices = [entry["index"] for entry in entries if entry["status"] == "missing"]
    error_indices = [entry["index"] for entry in entries if entry["status"] == "error"]
    predicted_indices = [entry["index"] for entry in entries if entry["status"] == "predicted"]

    per_category_correct = {
        field: sum(1 for entry in entries if entry["matches"].get(field) is True)
        for field in SCORED_FIELDS
    }
    per_category_accuracy = {
        field: per_category_correct[field] / total_expected if total_expected else 0.0
        for field in SCORED_FIELDS
    }

    item_accuracy_sum = sum(float(entry.get("metadata_accuracy", 0.0)) for entry in entries)
    exact_match_count = sum(1 for entry in entries if entry.get("exact_match") is True)

    frequency_json: Dict[str, Any] = {}
    for field, blocks in frequencies.items():
        frequency_json[field] = {
            "candidate": counter_to_plain(blocks["candidate"]),
            "ground_truth": counter_to_plain(blocks["ground_truth"]),
            "correct_candidate": counter_to_plain(blocks["correct_candidate"]),
        }

    return {
        "total_expected": total_expected,
        "total_records_written": len(entries),
        "predicted_count": len(predicted_indices),
        "missing_count": len(missing_indices),
        "error_count": len(error_indices),
        "missing_indices": missing_indices,
        "error_indices": error_indices,
        "scored_fields": list(SCORED_FIELDS),
        "accuracy_denominator": total_expected,
        "missing_and_error_policy": "missing/error entries receive 0.0 metadata_accuracy and count as incorrect for every scored field",
        "average_metadata_accuracy": item_accuracy_sum / total_expected if total_expected else 0.0,
        "exact_match_accuracy": exact_match_count / total_expected if total_expected else 0.0,
        "exact_match_count": exact_match_count,
        "per_category_correct": per_category_correct,
        "per_category_accuracy": per_category_accuracy,
        "class_frequencies": frequency_json,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_indices(start: int, end: int, limit: int) -> Iterable[int]:
    count = 0
    for idx in range(start, end + 1):
        if limit > 0 and count >= limit:
            break
        yield idx
        count += 1


def evaluate(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output_dir = ensure_under_allowed_write_root(Path(args.output_dir), "--output-dir")
    if args.cache_dir:
        setup_allowed_cache(Path(args.cache_dir))

    bundle = load_evaluation_dataset(
        Path(args.dataset_root),
        args.dataset_manifest_path,
    )
    records = bundle.records
    if len(records) < args.end_index + 1:
        raise ValueError(
            f"Ground-truth dataset has {len(records)} records, but end index "
            f"is {args.end_index}."
        )

    classifier_module = load_classifier_module(Path(args.classifier_py))
    thresholds = (
        classifier_module.ThresholdConfig.load(args.calibration)
        if args.calibration
        else classifier_module.ThresholdConfig()
    )
    classifier = classifier_module.SpeechMetadataClassifier(
        device=args.device,
        emotion_backend=args.emotion_backend,
        thresholds=thresholds,
        whisper_model=args.whisper_model,
    )

    frequencies = init_frequency_block()
    entries: List[Dict[str, Any]] = []
    speech_root = Path(args.speech_root)

    indices = list(iter_indices(args.start_index, args.end_index, args.limit))
    for position, idx in enumerate(indices, start=1):
        record = records[idx]
        gt = extract_ground_truth(record)
        item_dir = speech_root / str(idx)
        wav_path = find_candidate_wav(item_dir, args.wav_glob) if item_dir.exists() else None
        missing_txt = item_dir / "missing.txt"

        for field in SCORED_FIELDS:
            count_label(frequencies[field]["ground_truth"], gt.get(field))

        base_entry: Dict[str, Any] = {
            "index": idx,
            "speech_dir": str(item_dir),
            "audio_path": str(wav_path) if wav_path else None,
            "missing_txt_path": str(missing_txt) if missing_txt.exists() else None,
            "ground_truth": gt,
        }

        if wav_path is None:
            matches = {field: False for field in SCORED_FIELDS}
            entry = {
                **base_entry,
                "status": "missing",
                "candidate_metadata": None,
                "matches": matches,
                "metadata_accuracy": 0.0,
                "exact_match": False,
            }
            entries.append(entry)
            print(f"[{position}/{len(indices)}] index={idx} missing")
            continue

        try:
            prediction, debug = classifier.predict(
                wav_path,
                language=args.language,
                return_debug=args.debug,
            )
            candidate = asdict(prediction)
            matches = compare_metadata(candidate, gt)
            correct_count = sum(1 for value in matches.values() if value)
            metadata_accuracy = correct_count / len(SCORED_FIELDS)
            exact_match = correct_count == len(SCORED_FIELDS)

            for field in SCORED_FIELDS:
                count_label(frequencies[field]["candidate"], candidate.get(field))
                if matches[field]:
                    count_label(frequencies[field]["correct_candidate"], candidate.get(field))

            entry = {
                **base_entry,
                "status": "predicted",
                "candidate_metadata": candidate,
                "matches": matches,
                "metadata_accuracy": metadata_accuracy,
                "exact_match": exact_match,
            }
            if args.debug and debug is not None:
                entry["debug"] = debug
            entries.append(entry)
            print(
                f"[{position}/{len(indices)}] index={idx} "
                f"metadata_accuracy={metadata_accuracy:.2f} exact_match={exact_match}"
            )
        except Exception as exc:
            matches = {field: False for field in SCORED_FIELDS}
            entry = {
                **base_entry,
                "status": "error",
                "candidate_metadata": None,
                "matches": matches,
                "metadata_accuracy": 0.0,
                "exact_match": False,
                "error": str(exc),
            }
            entries.append(entry)
            print(f"[{position}/{len(indices)}] index={idx} error={exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    total_expected = len(indices)
    summary = build_summary(entries, frequencies, total_expected=total_expected)
    summary["speech_root"] = str(speech_root)
    summary["dataset_root"] = str(args.dataset_root)
    summary["dataset_manifest_path"] = str(args.dataset_manifest_path)
    summary["classifier_py"] = str(args.classifier_py)
    summary["output_dir"] = str(output_dir)

    return entries, summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract generated speech metadata and compare it with Omni StoryBench ground truth."
    )
    parser.add_argument("--speech-root", default=str(DEFAULT_SPEECH_ROOT), help="Folder containing 0..899 speech subfolders.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--classifier-py", default=str(DEFAULT_CLASSIFIER_PY), help="Standalone classifier Python file.")
    parser.add_argument("--output-dir", default=str(THIS_DIR), help="Directory for JSON outputs.")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional override for model cache. Defaults to the environment/Hugging Face default cache.",
    )
    parser.add_argument("--detail-output", default="metadata_comparison.json", help="Detailed JSON filename.")
    parser.add_argument("--summary-output", default="class_frequency_summary.json", help="Summary/frequency JSON filename.")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset index to evaluate.")
    parser.add_argument("--end-index", type=int, default=899, help="Last dataset index to evaluate.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of indices to process from start-index.")
    parser.add_argument("--wav-glob", default="*.wav", help="Glob used inside each index folder to locate candidate wav.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device. Defaults to classifier auto-select.")
    parser.add_argument("--language", default=None, help="Optional Whisper language hint.")
    parser.add_argument("--emotion-backend", default="ensemble", choices=["ensemble", "speechbrain", "hubert", "emotion2vec"])
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3")
    parser.add_argument("--calibration", default="", help="Optional calibrated speed/pitch threshold JSON.")
    parser.add_argument("--debug", action="store_true", help="Include classifier debug scores and measurements.")
    parser.add_argument("--dry-run", action="store_true", help="Write empty dry-run outputs without loading classifier models.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going and score errored items as 0.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    output_dir = ensure_under_allowed_write_root(Path(args.output_dir), "--output-dir")
    detail_output = ensure_under_allowed_write_root(output_dir / args.detail_output, "--detail-output")
    summary_output = ensure_under_allowed_write_root(output_dir / args.summary_output, "--summary-output")

    if args.dry_run:
        summary = {
            "dry_run": True,
            "total_expected": 0,
            "total_records_written": 0,
            "predicted_count": 0,
            "missing_count": 0,
            "error_count": 0,
            "scored_fields": list(SCORED_FIELDS),
            "average_metadata_accuracy": 0.0,
            "exact_match_accuracy": 0.0,
            "speech_root": str(args.speech_root),
            "dataset_root": str(args.dataset_root),
            "dataset_manifest_path": str(args.dataset_manifest_path),
            "classifier_py": str(args.classifier_py),
            "output_dir": str(output_dir),
        }
        write_json(detail_output, {"summary": summary, "entries": []})
        write_json(summary_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    entries, summary = evaluate(args)
    detail_payload = {
        "summary": summary,
        "entries": entries,
    }
    write_json(detail_output, detail_payload)
    write_json(summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
