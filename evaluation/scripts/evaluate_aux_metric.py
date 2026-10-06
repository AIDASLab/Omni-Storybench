#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence


HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if "/.local/lib/python" not in p]
sys.path.insert(0, str(HARNESS_ROOT))

from src.common import (  # noqa: E402
    build_record_metadata,
    chunked,
    resolve_all_candidates,
    selected_indices,
    status_counts,
    write_json,
)
from src.dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_evaluation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stage-1 auxiliary metrics.")
    parser.add_argument("--metric", required=True, choices=["bertscore", "clip"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    bundle = load_evaluation_dataset(
        args.dataset_root,
        args.dataset_manifest_path,
    )
    records = bundle.records
    indices = selected_indices(
        total=len(records),
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    prepared: List[Dict[str, Any]] = []
    for position, dataset_index in enumerate(indices, start=1):
        record = records[dataset_index]
        candidates = resolve_all_candidates(args.results_root, dataset_index)
        modality = "text" if args.metric == "bertscore" else "image"
        prepared.append(
            {
                **build_record_metadata(dataset_index, record),
                "status": "pending",
                "candidate": candidates[modality],
                "_record": record,
            }
        )
        print(f"[prepare {position}/{len(indices)}] metric={args.metric} dataset_index={dataset_index}")
    return prepared


def candidate_is_ok(item: Dict[str, Any]) -> bool:
    return (item.get("candidate") or {}).get("status") == "ok"


def skipped_record(item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
    cleaned["status"] = "skipped"
    cleaned["reason"] = (item.get("candidate") or {}).get("missing_reason", "Candidate artifact is unavailable.")
    cleaned["metric_value"] = None
    return cleaned


def error_record(item: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
    cleaned["status"] = "error"
    cleaned["reason"] = str(exc)
    cleaned["error_type"] = type(exc).__name__
    cleaned["traceback"] = traceback.format_exc()
    cleaned["metric_value"] = None
    return cleaned


def dry_run_records(prepared: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in prepared:
        cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
        cleaned["status"] = "dry_run_ok" if candidate_is_ok(item) else "dry_run_missing"
        cleaned["metric_value"] = None
        output.append(cleaned)
    return output


def evaluate_bertscore(prepared: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    from src.bertscore import clear_bert_score_cache, get_bert_scores

    output: List[Dict[str, Any]] = []
    valid = [item for item in prepared if candidate_is_ok(item)]
    skipped = {item["dataset_index"]: skipped_record(item) for item in prepared if not candidate_is_ok(item)}
    results_by_index: Dict[int, Dict[str, Any]] = {}

    for batch in chunked(valid, args.batch_size):
        print(f"[bertscore batch] {batch[0]['dataset_index']}..{batch[-1]['dataset_index']}")
        predictions = [item["candidate"]["text"] for item in batch]
        references = [item["_record"].get("next_page", {}).get("text", "") for item in batch]
        try:
            _precision, _recall, f1 = get_bert_scores(predictions, references)
            for item, score in zip(batch, f1):
                cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
                cleaned["status"] = "ok"
                cleaned["metric_value"] = float(score)
                results_by_index[item["dataset_index"]] = cleaned
        except Exception as exc:
            for item in batch:
                results_by_index[item["dataset_index"]] = error_record(item, exc)

    for item in prepared:
        dataset_index = item["dataset_index"]
        output.append(results_by_index[dataset_index] if dataset_index in results_by_index else skipped[dataset_index])
    clear_bert_score_cache()
    return output


def evaluate_clip(prepared: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    from src.clip_similarity import clear_clip_model_cache, clip_similarity_batch


    output: List[Dict[str, Any]] = []
    valid = [item for item in prepared if candidate_is_ok(item)]
    skipped = {item["dataset_index"]: skipped_record(item) for item in prepared if not candidate_is_ok(item)}
    results_by_index: Dict[int, Dict[str, Any]] = {}

    for batch in chunked(valid, args.batch_size):
        print(f"[clip batch] {batch[0]['dataset_index']}..{batch[-1]['dataset_index']}")
        pairs = [
            (
                item["_record"].get("next_page", {}).get("image_path", ""),
                item["candidate"]["path"],
            )
            for item in batch
        ]
        try:
            scores = clip_similarity_batch(pairs)
            for item, score in zip(batch, scores):
                cleaned = {key: value for key, value in item.items() if not key.startswith("_")}
                cleaned["status"] = "ok"
                cleaned["metric_value"] = float(score)
                results_by_index[item["dataset_index"]] = cleaned
        except Exception as exc:
            for item in batch:
                results_by_index[item["dataset_index"]] = error_record(item, exc)

    for item in prepared:
        dataset_index = item["dataset_index"]
        output.append(results_by_index[dataset_index] if dataset_index in results_by_index else skipped[dataset_index])
    clear_clip_model_cache()
    return output


def build_summary(args: argparse.Namespace, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = [
        float(record["metric_value"])
        for record in records
        if isinstance(record.get("metric_value"), (int, float)) and not isinstance(record.get("metric_value"), bool)
    ]
    return {
        "stage": "1",
        "metric": args.metric,
        "dataset_root": str(args.dataset_root),
        "dataset_manifest_path": str(args.dataset_manifest_path),
        "results_root": str(args.results_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "record_count": len(records),
        "status_counts": status_counts(records),
        "average": sum(values) / len(records) if records else 0.0,
        "valid_metric_count": len(values),
        "zero_filled_count": len(records) - len(values),
        "dry_run": args.dry_run,
    }


def main() -> int:
    args = parse_args()
    prepared = prepare_records(args)
    if args.dry_run:
        records = dry_run_records(prepared)
    elif args.metric == "bertscore":
        records = evaluate_bertscore(prepared, args)
    else:
        records = evaluate_clip(prepared, args)
    summary = build_summary(args, records)
    write_json(args.output_json, {"summary": summary, "records": records})
    write_json(args.summary_json, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
