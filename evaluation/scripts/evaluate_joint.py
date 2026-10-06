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
    OMNI_JUDGE_CATEGORIES,
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
    DEFAULT_JOINT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
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
        description="Evaluate one Omni StoryBench joint multimodal stage."
    )
    parser.add_argument("--stage", required=True, choices=["1", "2", "3"])
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--model-name", help=argparse.SUPPRESS)
    parser.add_argument(
        "--adapter",
        choices=adapter_keys("joint"),
        help="Override the joint adapter assigned to this stage.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_JOINT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--vllm-swap-space", type=float, default=8.0)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_adapter(args: argparse.Namespace) -> ModelAdapter:
    adapter = (
        get_adapter(args.adapter)
        if args.adapter
        else get_stage_adapter(args.stage, "joint")
    )
    validate_adapter(
        adapter,
        stage=args.stage,
        modality="joint",
        model_name=getattr(args, "model_name", None),
        backend="vllm",
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
        "limit_mm_per_prompt": {"image": 3, "audio": 1, "video": 0},
        "seed": args.seed,
    }
    kwargs.update(adapter.engine_kwargs)
    return kwargs


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
        prepared.append(
            {
                **build_record_metadata(dataset_index, record),
                "status": "pending",
                "candidate": candidates,
                "metadata": record.get("metadata", {}),
                "next_page_condition": record.get("next_page_condition", {}),
                "ground_truth_speech_text": extract_ground_truth_speech_text(
                    record
                ),
                "ground_truth_speech_metadata": record.get(
                    "speech_metadata_next_page",
                    {},
                ),
                "_record": record,
            }
        )
        print(
            f"[prepare {position}/{len(indices)}] stage={args.stage} "
            f"joint dataset_index={dataset_index}"
        )
    return prepared


def candidate_is_ok(item: Dict[str, Any]) -> bool:
    candidates = item.get("candidate") or {}
    return all(
        (candidates.get(modality) or {}).get("status") == "ok"
        for modality in ("text", "image", "speech")
    )


def _clean_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def skipped_record(item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = _clean_record(item)
    missing = [
        modality
        for modality in ("text", "image", "speech")
        if (cleaned["candidate"].get(modality) or {}).get("status") != "ok"
    ]
    cleaned.update(
        {
            "status": "skipped",
            "inference_status": "skipped",
            "parse_status": "not_attempted",
            "repair_status": "not_eligible",
            "parse_method": None,
            "reason": "At least one required candidate modality is unavailable.",
            "missing_modalities": missing,
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
    if "status" not in normalized:
        normalized = runtime_error_fields(
            "Joint judge returned an invalid result object"
        )
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


def build_request(item: Dict[str, Any]) -> Dict[str, Any]:
    record = item["_record"]
    candidates = item["candidate"]
    return {
        "metadata": record.get("metadata", {}),
        "condition_json": record.get("next_page_condition", {}),
        "current_text": record.get("current_page", {}).get("text", ""),
        "ground_truth_text": record.get("next_page", {}).get("text", ""),
        "candidate_text": candidates["text"]["text"],
        "ground_truth_speech_metadata": record.get(
            "speech_metadata_next_page",
            {},
        ),
        "current_image": record.get("current_page", {}).get("image_path", ""),
        "ground_truth_image": record.get("next_page", {}).get("image_path", ""),
        "candidate_image": candidates["image"]["path"],
        "candidate_audio": candidates["speech"]["path"],
    }


def run_batch(
    adapter: ModelAdapter,
    args: argparse.Namespace,
    requests: Sequence[Dict[str, Any]],
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
        use_audio_in_video=args.use_audio_in_video,
    )


def evaluate_joint(
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
        print(f"[joint batch] {first}..{last}")
        requests = [build_request(item) for item in batch]
        try:
            batch_results = run_batch(
                adapter,
                args,
                requests,
                vllm_kwargs,
            )
        except Exception as exc:
            if not should_retry_batch_error(
                exc,
                adapter.model_name,
                **vllm_kwargs,
            ):
                raise
            batch_results = []
            for request in requests:
                try:
                    response = run_batch(
                        adapter,
                        args,
                        [request],
                        vllm_kwargs,
                    )
                    if not response:
                        raise RuntimeError("Joint judge returned no result")
                    batch_results.append(response[0])
                except Exception as item_exc:
                    if not should_retry_batch_error(
                        item_exc,
                        adapter.model_name,
                        **vllm_kwargs,
                    ):
                        raise
                    batch_results.append(runtime_result(item_exc))

        for result_index, item in enumerate(batch):
            result = (
                batch_results[result_index]
                if result_index < len(batch_results)
                else runtime_error_fields(
                    "Joint judge returned fewer results than requests"
                )
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
        "modality": "joint",
        "model_name": adapter.model_name,
        "chat_template_kwargs": dict(adapter.chat_template_kwargs),
        "reasoning_control": model_controls["reasoning"],
        "model_controls": model_controls,
        "adapter": adapter.metadata(),
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
        "judge_categories": OMNI_JUDGE_CATEGORIES,
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
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    prepared = prepare_records(args)
    vllm_kwargs = build_vllm_kwargs(args, adapter)
    records = (
        dry_run_records(prepared)
        if args.dry_run
        else evaluate_joint(prepared, adapter, args, vllm_kwargs)
    )
    summary = build_summary(args, adapter, records)
    write_json(args.output_json, {"summary": summary, "records": records})
    write_json(args.summary_json, summary)

    if not args.dry_run:
        release_models(adapter.model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
