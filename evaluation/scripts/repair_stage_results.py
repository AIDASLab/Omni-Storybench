#!/usr/bin/env python3
"""Normalize and repair judge outputs after all jobs in a stage have stopped."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if "/.local/lib/python" not in path]
sys.path.insert(0, str(HARNESS_ROOT))

from src.common import (  # noqa: E402
    JUDGE_CATEGORIES_BY_MODALITY,
    read_json,
    status_counts,
    write_json,
)
from src.evaluation_protocol import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generation_metadata,
)
from src.judge_parsing import (  # noqa: E402
    build_repair_messages,
    build_repair_json_schema,
    is_unrecoverable_response,
    parse_judge_response,
)
from src.inference.vllm_runtime import (  # noqa: E402
    release_models,
    should_enable_expert_parallel,
)


MODALITIES = ("text", "image", "speech", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair one completed evaluation stage.")
    parser.add_argument("--stage", required=True, choices=["1", "2", "3"])
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument(
        "--repair-model-name",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument("--repair-backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-swap-space", type=float, default=8.0)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--max-systemic-error-rate", type=float, default=0.5)
    parser.add_argument(
        "--require-zero-unresolved",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to blocking the next stage when any result remains unresolved.",
    )
    return parser.parse_args()


def chunks(values: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    width = max(size, 1)
    return [values[index : index + width] for index in range(0, len(values), width)]


def _preserve_inference_status(record: Dict[str, Any]) -> str:
    current = record.get("inference_status")
    if isinstance(current, str):
        return current
    status = record.get("status")
    if status in {"error", "skipped", "dry_run_ok", "dry_run_missing"}:
        return "skipped" if status == "skipped" else status
    return "ok"


def normalize_record(
    record: Dict[str, Any],
    categories: Sequence[str],
) -> Tuple[Dict[str, Any], bool]:
    normalized = dict(record)
    inference_status = _preserve_inference_status(normalized)
    normalized["inference_status"] = inference_status

    if inference_status != "ok":
        normalized.setdefault("parse_status", "not_attempted")
        normalized.setdefault("repair_status", "not_eligible")
        normalized.setdefault("parse_method", None)
        normalized.setdefault("initial_parse_error", None)
        normalized.setdefault("repair_raw_response", None)
        normalized.setdefault("final_parse_error", None)
        normalized["parse_error"] = normalized.get("final_parse_error")
        return normalized, False

    raw_response = normalized.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        existing = normalized.get("judge_eval")
        if existing is not None:
            outcome = parse_judge_response(_json_text(existing), categories)
            outcome["raw_response"] = raw_response
            if outcome["status"] == "ok":
                outcome["parse_method"] = "normalized_existing_result"
                outcome["parse_status"] = "normalized"
        else:
            outcome = parse_judge_response(raw_response, categories)
        normalized.update(outcome)
        normalized["repair_status"] = "not_eligible"
        normalized["parse_error"] = normalized.get("final_parse_error")
        if normalized["status"] == "ok":
            normalized.pop("reason", None)
            normalized.pop("error_type", None)
        else:
            normalized["reason"] = normalized.get("final_parse_error")
        return normalized, False

    outcome = parse_judge_response(raw_response, categories)
    normalized.update(outcome)
    normalized["parse_error"] = normalized.get("final_parse_error")
    if normalized["status"] == "ok":
        normalized.pop("reason", None)
        normalized.pop("error_type", None)
    else:
        normalized["reason"] = normalized.get("final_parse_error")
    return normalized, outcome["status"] == "parse_error"


def _json_text(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def apply_repair_response(
    record: Dict[str, Any],
    repair_response: str,
    categories: Sequence[str],
) -> None:
    record["repair_raw_response"] = repair_response
    if is_unrecoverable_response(repair_response):
        error = "Repair model marked the response as unrecoverable"
        record.update(
            {
                "status": "parse_error",
                "parse_status": "unresolved",
                "repair_status": "unrecoverable",
                "parse_method": "unrecoverable",
                "judge_eval": None,
                "final_parse_error": error,
                "parse_error": error,
                "reason": error,
            }
        )
        return

    repaired = parse_judge_response(repair_response, categories)
    if repaired["status"] == "ok":
        record.update(
            {
                "status": "ok",
                "parse_status": "repaired",
                "repair_status": "succeeded",
                "parse_method": "qwen_repair",
                "judge_eval": repaired["judge_eval"],
                "final_parse_error": None,
                "parse_error": None,
            }
        )
        record.pop("reason", None)
        record.pop("error_type", None)
        return

    error = repaired.get("final_parse_error") or "Repair response could not be parsed"
    record.update(
        {
            "status": "parse_error",
            "parse_status": "unresolved",
            "repair_status": "failed",
            "parse_method": "unrecoverable",
            "judge_eval": None,
            "final_parse_error": error,
            "parse_error": error,
            "reason": error,
        }
    )


def mark_repair_runtime_error(record: Dict[str, Any], error: str) -> None:
    record.update(
        {
            "status": "parse_error",
            "parse_status": "unresolved",
            "repair_status": "failed",
            "parse_method": "unrecoverable",
            "repair_raw_response": None,
            "final_parse_error": f"Repair inference failed: {error}",
            "parse_error": f"Repair inference failed: {error}",
            "reason": f"Repair inference failed: {error}",
        }
    )


def update_summary(
    summary: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    repair_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(summary)
    updated["record_count"] = len(records)
    updated["status_counts"] = status_counts(records)
    updated["parse_status_counts"] = status_counts(
        [{"status": record.get("parse_status", "unknown")} for record in records]
    )
    updated["repair_status_counts"] = status_counts(
        [{"status": record.get("repair_status", "unknown")} for record in records]
    )
    updated["stage_end_repair"] = repair_metadata
    return updated


def assess_modality_quality(
    records: Sequence[Dict[str, Any]],
    max_systemic_error_rate: float,
    require_zero_unresolved: bool = False,
) -> Dict[str, Any]:
    eligible = [
        record
        for record in records
        if record.get("inference_status")
        not in {"skipped", "dry_run_ok", "dry_run_missing"}
    ]
    runtime_errors = [
        record for record in eligible if record.get("inference_status") == "error"
    ]
    unresolved_parse_errors = [
        record
        for record in eligible
        if record.get("inference_status") == "ok"
        and record.get("status") == "parse_error"
    ]
    runtime_error_rate = len(runtime_errors) / len(eligible) if eligible else 0.0
    parse_error_rate = (
        len(unresolved_parse_errors) / len(eligible) if eligible else 0.0
    )
    unresolved_error_rate = (
        (len(runtime_errors) + len(unresolved_parse_errors)) / len(eligible)
        if eligible
        else 0.0
    )
    systemic_rate_gate_passed = (
        not eligible or runtime_error_rate < max_systemic_error_rate
    )
    zero_unresolved_gate_passed = (
        not runtime_errors and not unresolved_parse_errors
    )
    quality_gate_passed = (
        zero_unresolved_gate_passed
        if require_zero_unresolved
        else systemic_rate_gate_passed
    )
    return {
        "runtime_error_count": len(runtime_errors),
        "runtime_error_rate": runtime_error_rate,
        "unresolved_parse_error_count": len(unresolved_parse_errors),
        "unresolved_parse_error_rate": parse_error_rate,
        "unresolved_error_rate": unresolved_error_rate,
        "systemic_rate_gate_passed": systemic_rate_gate_passed,
        "zero_unresolved_gate_passed": zero_unresolved_gate_passed,
        "quality_gate_passed": quality_gate_passed,
        "eligible_record_count": len(eligible),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.repair_backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    payloads: Dict[str, Dict[str, Any]] = {}
    repair_queue: List[Tuple[str, int, Dict[str, Any]]] = []
    modality_manifest: Dict[str, Any] = {}

    for modality in MODALITIES:
        result_path = args.stage_dir / modality / "all_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing completed modality result: {result_path}")
        payload = read_json(result_path)
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"Invalid records payload: {result_path}")

        categories = JUDGE_CATEGORIES_BY_MODALITY[modality]
        normalized_records: List[Dict[str, Any]] = []
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at {result_path}:{record_index}")
            normalized, needs_repair = normalize_record(record, categories)
            normalized_records.append(normalized)
            if needs_repair:
                repair_queue.append((modality, record_index, normalized))

        payload["records"] = normalized_records
        payloads[modality] = payload
        modality_manifest[modality] = {
            "record_count": len(normalized_records),
            "repair_candidate_count": sum(
                1 for queued_modality, _, _ in repair_queue if queued_modality == modality
            ),
        }

    repair_generation = generation_metadata(
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    repair_metadata = {
        "model_name": args.repair_model_name,
        "backend": args.repair_backend,
        "structured_output_enabled": args.repair_backend == "vllm",
        "require_zero_unresolved": args.require_zero_unresolved,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "generation": repair_generation,
        "vllm_config": {
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "swap_space": args.vllm_swap_space,
            "max_model_len": args.vllm_max_model_len,
            "max_num_seqs": max(args.batch_size, 1),
            "worker_multiproc_method": os.environ.get(
                "VLLM_WORKER_MULTIPROC_METHOD"
            ),
        },
        "candidate_count": len(repair_queue),
    }

    if repair_queue:
        print(
            f"[repair-model] loading={args.repair_model_name} "
            f"candidates={len(repair_queue)}"
        )
        from src.inference.json_repair import (
            generate_json_repair_responses,
        )

        vllm_kwargs = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.vllm_max_model_len,
            "max_num_seqs": max(args.batch_size, 1),
            "swap_space": args.vllm_swap_space,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": args.seed,
            "enable_expert_parallel": (
                True if should_enable_expert_parallel(args.repair_model_name) else None
            ),
        }
        repair_batches = chunks(repair_queue, args.batch_size)
        for batch_index, batch in enumerate(repair_batches, start=1):
            print(
                f"[repair-batch] stage={args.stage} batch={batch_index}/"
                f"{len(repair_batches)} records={len(batch)}"
            )
            messages = [
                build_repair_messages(
                    str(record["raw_response"]),
                    str(record.get("initial_parse_error") or "Unknown parse error"),
                    JUDGE_CATEGORIES_BY_MODALITY[modality],
                )
                for modality, _, record in batch
            ]
            json_schemas = [
                build_repair_json_schema(JUDGE_CATEGORIES_BY_MODALITY[modality])
                for modality, _, _ in batch
            ]
            try:
                responses = generate_json_repair_responses(
                    args.repair_model_name,
                    messages,
                    json_schemas=json_schemas,
                    backend=args.repair_backend,
                    vllm_kwargs=vllm_kwargs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
            except Exception as exc:
                for _, _, record in batch:
                    mark_repair_runtime_error(record, f"{type(exc).__name__}: {exc}")
                continue

            for response_index, (modality, _, record) in enumerate(batch):
                if response_index >= len(responses):
                    mark_repair_runtime_error(
                        record,
                        "Repair backend returned fewer responses than requests",
                    )
                    continue
                apply_repair_response(
                    record,
                    responses[response_index],
                    JUDGE_CATEGORIES_BY_MODALITY[modality],
                )

        release_models(args.repair_model_name)

    gate_failures: List[Dict[str, Any]] = []
    for modality, payload in payloads.items():
        records = payload["records"]
        quality = assess_modality_quality(
            records,
            args.max_systemic_error_rate,
            args.require_zero_unresolved,
        )
        modality_manifest[modality].update(
            {
                "status_counts": status_counts(records),
                "parse_status_counts": status_counts(
                    [{"status": record.get("parse_status", "unknown")} for record in records]
                ),
                "repair_status_counts": status_counts(
                    [{"status": record.get("repair_status", "unknown")} for record in records]
                ),
                **quality,
            }
        )
        if not quality["quality_gate_passed"]:
            gate_failures.append(
                {
                    "failure_policy": (
                        "zero_unresolved"
                        if args.require_zero_unresolved
                        else "systemic_runtime_error_rate"
                    ),
                    "modality": modality,
                    **{
                        key: quality[key]
                        for key in (
                            "runtime_error_count",
                            "runtime_error_rate",
                            "unresolved_parse_error_count",
                            "unresolved_parse_error_rate",
                            "unresolved_error_rate",
                            "eligible_record_count",
                        )
                    },
                }
            )

        summary = update_summary(payload.get("summary", {}), records, repair_metadata)
        payload["summary"] = summary
        write_json(args.stage_dir / modality / "all_results.json", payload)
        write_json(args.stage_dir / modality / "summary.json", summary)

    manifest = {
        "stage": args.stage,
        "repair": repair_metadata,
        "modalities": modality_manifest,
        "quality_gate": {
            "max_systemic_error_rate": args.max_systemic_error_rate,
            "require_zero_unresolved": args.require_zero_unresolved,
            "passed": not gate_failures,
            "failures": gate_failures,
        },
    }
    write_json(args.stage_dir / "repair_manifest.json", manifest)

    if gate_failures:
        print(f"[quality-gate] failed: {gate_failures}", file=sys.stderr)
        return 1
    unresolved_count = sum(
        item["unresolved_parse_error_count"]
        for item in modality_manifest.values()
    )
    print(
        f"[stage-repair] stage={args.stage} repair_candidates={len(repair_queue)} "
        f"unresolved_parse_errors={unresolved_count} quality_gate=passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
