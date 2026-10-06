#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence


HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if "/.local/lib/python" not in path]
sys.path.insert(0, str(HARNESS_ROOT))

from src.common import (  # noqa: E402
    JUDGE_CATEGORIES_BY_MODALITY,
    build_record_metadata,
    chunked,
    extract_ground_truth_speech_text,
    resolve_all_candidates,
    selected_indices,
    status_counts,
    write_json,
)
from src.dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_evaluation_dataset,
)
from src.evaluation_protocol import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generation_metadata,
)
from src.inference import (  # noqa: E402
    initialize_inference,
    run_batch_inference,
)
from src.judge_parsing import runtime_error_fields  # noqa: E402
from src.model_adapters import (  # noqa: E402
    adapter_keys,
    get_adapter,
    get_stage_adapter,
    validate_adapter,
)
from src.model_adapters.base import ModelAdapter  # noqa: E402
from src.inference.vllm_runtime import (  # noqa: E402
    release_models,
    should_retry_batch_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one Omni StoryBench modality."
    )
    parser.add_argument("--stage", required=True, choices=["1", "2", "3"])
    parser.add_argument("--modality", required=True, choices=["text", "image", "speech"])
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--model-name", help=argparse.SUPPRESS)
    parser.add_argument(
        "--adapter",
        choices=adapter_keys(),
        help="Override the adapter assigned to this stage and modality.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers", "auto"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-swap-space", type=float, default=8.0)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_adapter(args: argparse.Namespace) -> ModelAdapter:
    adapter = (
        get_adapter(args.adapter)
        if args.adapter
        else get_stage_adapter(args.stage, args.modality)
    )
    validate_adapter(
        adapter,
        stage=args.stage,
        modality=args.modality,
        model_name=getattr(args, "model_name", None),
        backend=getattr(args, "backend", None),
    )
    return adapter


def build_vllm_kwargs(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "tensor_parallel_size": max(args.vllm_tensor_parallel_size, 1),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "max_num_seqs": max(args.vllm_max_num_seqs, args.batch_size),
        "swap_space": args.vllm_swap_space,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": args.seed,
    }
    if args.modality == "image":
        kwargs["limit_mm_per_prompt"] = {"image": 3, "video": 0}
    elif args.modality == "speech":
        kwargs["limit_mm_per_prompt"] = {
            "audio": 1,
            "video": 0,
            "image": 0,
        }
    kwargs.update(adapter.engine_kwargs)
    return kwargs


def build_base_record(
    dataset_index: int,
    record: Dict[str, Any],
    candidates: Dict[str, Dict[str, Any]],
    modality: str,
) -> Dict[str, Any]:
    payload = {
        **build_record_metadata(dataset_index, record),
        "status": "pending",
        "candidate": candidates.get(modality),
        "metadata": record.get("metadata", {}),
        "next_page_condition": record.get("next_page_condition", {}),
    }
    if modality == "speech":
        payload["ground_truth_speech_text"] = extract_ground_truth_speech_text(
            record
        )
        payload["ground_truth_speech_metadata"] = record.get(
            "speech_metadata_next_page",
            {},
        )
    return payload


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
        item = build_base_record(
            dataset_index,
            record,
            candidates,
            args.modality,
        )
        item["_record"] = record
        prepared.append(item)
        print(
            f"[prepare {position}/{len(indices)}] stage={args.stage} "
            f"modality={args.modality} dataset_index={dataset_index}"
        )
    return prepared


def candidate_is_ok(item: Dict[str, Any]) -> bool:
    candidate = item.get("candidate") or {}
    return candidate.get("status") == "ok"


def _clean_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def skipped_record(item: Dict[str, Any], reason: str | None = None) -> Dict[str, Any]:
    candidate = item.get("candidate") or {}
    cleaned = _clean_record(item)
    cleaned.update(
        {
            "status": "skipped",
            "inference_status": "skipped",
            "parse_status": "not_attempted",
            "repair_status": "not_eligible",
            "parse_method": None,
            "reason": reason
            or candidate.get(
                "missing_reason",
                "Candidate artifact is unavailable.",
            ),
            "judge_eval": None,
            "raw_response": None,
            "initial_parse_error": None,
            "repair_raw_response": None,
            "final_parse_error": None,
            "parse_error": None,
        }
    )
    return cleaned


def runtime_result(exc: BaseException) -> Dict[str, Any]:
    error = str(exc)
    return {
        **runtime_error_fields(error),
        "reason": error,
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }


def inference_result_record(
    item: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    cleaned = _clean_record(item)
    normalized = dict(result) if isinstance(result, dict) else {}
    if "inference_status" not in normalized and normalized.get("error"):
        normalized = {
            **runtime_error_fields(str(normalized["error"])),
            "error_type": normalized.get("error_type", "JudgeError"),
        }
    if "status" not in normalized:
        normalized = runtime_error_fields("Judge returned an invalid result object")
    cleaned.update(normalized)
    cleaned["parse_error"] = cleaned.get("final_parse_error")
    if cleaned.get("status") == "error":
        cleaned.setdefault(
            "reason",
            cleaned.get("error") or cleaned.get("final_parse_error"),
        )
    elif cleaned.get("status") == "parse_error":
        cleaned.setdefault("reason", cleaned.get("final_parse_error"))
    return cleaned


def dry_run_records(prepared: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in prepared:
        cleaned = _clean_record(item)
        cleaned["status"] = (
            "dry_run_ok" if candidate_is_ok(item) else "dry_run_missing"
        )
        cleaned["inference_status"] = "dry_run"
        cleaned["parse_status"] = "not_attempted"
        cleaned["repair_status"] = "not_eligible"
        cleaned["parse_method"] = None
        cleaned["judge_eval"] = None
        cleaned["raw_response"] = None
        cleaned["initial_parse_error"] = None
        cleaned["repair_raw_response"] = None
        cleaned["final_parse_error"] = None
        cleaned["parse_error"] = None
        output.append(cleaned)
    return output


def build_inference_request(
    item: Dict[str, Any],
    modality: str,
) -> Dict[str, Any]:
    record = item["_record"]
    request: Dict[str, Any] = {
        "metadata": record.get("metadata", {}),
        "condition_json": record.get("next_page_condition", {}),
    }
    if modality == "text":
        request.update(
            {
                "current_text": record.get("current_page", {}).get("text", ""),
                "ground_truth_text": record.get("next_page", {}).get("text", ""),
                "candidate_text": item["candidate"]["text"],
            }
        )
    elif modality == "image":
        request.update(
            {
                "current_image": record.get("current_page", {}).get(
                    "image_path",
                    "",
                ),
                "ground_truth_image": record.get("next_page", {}).get(
                    "image_path",
                    "",
                ),
                "candidate_image": item["candidate"]["path"],
            }
        )
    elif modality == "speech":
        request.update(
            {
                "ground_truth_speech_metadata": record.get(
                    "speech_metadata_next_page",
                    {},
                ),
                "candidate_audio": item["candidate"]["path"],
            }
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return request


def _run_batch(
    adapter: ModelAdapter,
    requests: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    vllm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return run_batch_inference(
        adapter,
        requests,
        vllm_kwargs=vllm_kwargs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def _run_transformers_items(
    adapter: ModelAdapter,
    requests: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    vllm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for request in requests:
        try:
            response = _run_batch(adapter, [request], args, vllm_kwargs)
            if not response:
                raise RuntimeError("Judge backend returned no result")
            results.append(response[0])
        except Exception as exc:
            results.append(runtime_result(exc))
    return results


def _run_vllm_batch_with_fallback(
    adapter: ModelAdapter,
    requests: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    vllm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        return _run_batch(adapter, requests, args, vllm_kwargs)
    except Exception as exc:
        if not should_retry_batch_error(
            exc,
            adapter.model_name,
            **vllm_kwargs,
        ):
            raise

    results: List[Dict[str, Any]] = []
    for request in requests:
        try:
            response = _run_batch(adapter, [request], args, vllm_kwargs)
            if not response:
                raise RuntimeError("Judge backend returned no result")
            results.append(response[0])
        except Exception as exc:
            if not should_retry_batch_error(
                exc,
                adapter.model_name,
                **vllm_kwargs,
            ):
                raise
            results.append(runtime_result(exc))
    return results


def evaluate_records(
    prepared: Sequence[Dict[str, Any]],
    adapter: ModelAdapter,
    args: argparse.Namespace,
    vllm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    initialize_inference(adapter, args.seed)
    valid = [item for item in prepared if candidate_is_ok(item)]
    results_by_index: Dict[int, Dict[str, Any]] = {}

    for batch in chunked(valid, args.batch_size):
        first = batch[0]["dataset_index"]
        last = batch[-1]["dataset_index"]
        print(f"[{args.modality} batch] {first}..{last}")
        requests = [
            build_inference_request(item, args.modality)
            for item in batch
        ]
        if adapter.backend == "transformers":
            judge_results = _run_transformers_items(
                adapter,
                requests,
                args,
                vllm_kwargs,
            )
        else:
            judge_results = _run_vllm_batch_with_fallback(
                adapter,
                requests,
                args,
                vllm_kwargs,
            )

        for result_index, item in enumerate(batch):
            if result_index < len(judge_results):
                result = judge_results[result_index]
            else:
                result = runtime_error_fields(
                    "Judge backend returned fewer results than requests"
                )
            results_by_index[item["dataset_index"]] = inference_result_record(
                item,
                result,
            )

    return [
        results_by_index[item["dataset_index"]]
        if item["dataset_index"] in results_by_index
        else skipped_record(item)
        for item in prepared
    ]


def build_summary(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    model_controls = adapter.control_metadata()
    return {
        "stage": args.stage,
        "modality": args.modality,
        "model_name": adapter.model_name,
        "adapter": adapter.metadata(),
        "backend": adapter.backend,
        "chat_template_kwargs": dict(adapter.chat_template_kwargs),
        "reasoning_control": model_controls["reasoning"],
        "model_controls": model_controls,
        "dataset_root": str(args.dataset_root),
        "dataset_manifest_path": str(args.dataset_manifest_path),
        "results_root": str(args.results_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_config": {
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "swap_space": args.vllm_swap_space,
            "max_model_len": args.vllm_max_model_len,
            "max_num_seqs": args.vllm_max_num_seqs,
            "worker_multiproc_method": os.environ.get(
                "VLLM_WORKER_MULTIPROC_METHOD"
            ),
        },
        "record_count": len(records),
        "status_counts": status_counts(records),
        "judge_categories": JUDGE_CATEGORIES_BY_MODALITY[args.modality],
        "generation": generation_metadata(
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ),
        "parse_status_counts": status_counts(
            [
                {"status": record.get("parse_status", "unknown")}
                for record in records
            ]
        ),
        "repair_status_counts": status_counts(
            [
                {"status": record.get("repair_status", "unknown")}
                for record in records
            ]
        ),
        "dry_run": args.dry_run,
    }


def main() -> int:
    args = parse_args()
    adapter = resolve_adapter(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if adapter.backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    prepared = prepare_records(args)
    vllm_kwargs = build_vllm_kwargs(args, adapter)
    records = (
        dry_run_records(prepared)
        if args.dry_run
        else evaluate_records(prepared, adapter, args, vllm_kwargs)
    )
    summary = build_summary(args, adapter, records)
    write_json(args.output_json, {"summary": summary, "records": records})
    write_json(args.summary_json, summary)

    if adapter.backend == "vllm":
        release_models(adapter.model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
