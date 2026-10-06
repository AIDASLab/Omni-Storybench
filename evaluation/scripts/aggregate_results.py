#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from src.common import (  # noqa: E402
    IMAGE_JUDGE_CATEGORIES,
    OMNI_JUDGE_CATEGORIES,
    SPEECH_JUDGE_CATEGORIES,
    TEXT_JUDGE_CATEGORIES,
    average_judge_scores,
    ensure_inside_harness,
    mean_zero_filled,
    read_json,
    write_json,
)
from src.model_adapters import STAGE_CONFIGS  # noqa: E402


TABLE_HEADERS = [
    "run_name",
    "stage",
    "text_model",
    "image_model",
    "speech_model",
    "joint_model",
    "llm_judge_text",
    "llm_judge_image",
    "llm_judge_speech",
    "llm_judge_joint",
    "bert_score_text",
    "clip_similarity_image",
    "speech_classifier_metadata_accuracy",
    "speech_classifier_exact_match_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Omni StoryBench harness outputs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--decimals", type=int, default=6)
    return parser.parse_args()


def parse_stages(value: str) -> List[str]:
    stages: List[str] = []
    for raw in value.split(","):
        stage = raw.strip()
        if not stage:
            continue
        if stage not in STAGE_CONFIGS:
            raise ValueError(f"Unsupported stage: {stage}")
        if stage not in stages:
            stages.append(stage)
    if not stages:
        raise ValueError("--stages must include at least one stage")
    return stages


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    payload = read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    return records if isinstance(records, list) else []


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def judge_average(run_dir: Path, stage: str, modality: str, categories: Sequence[str]) -> Dict[str, Any]:
    records = load_records(run_dir / f"stage_{stage}" / modality / "all_results.json")
    return {
        "record_count": len(records),
        **average_judge_scores(records, "judge_eval", categories),
    }


def aux_average(run_dir: Path, metric: str) -> Optional[float]:
    summary = load_summary(run_dir / "stage_1" / metric / "summary.json")
    value = summary.get("average")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def speech_classifier_summary(run_dir: Path) -> Dict[str, Any]:
    return load_summary(run_dir / "stage_1" / "speech_classifier" / "class_frequency_summary.json")


def build_stage_row(run_name: str, run_dir: Path, stage: str) -> Dict[str, Any]:
    config = STAGE_CONFIGS[stage]
    text = judge_average(run_dir, stage, "text", TEXT_JUDGE_CATEGORIES)
    image = judge_average(run_dir, stage, "image", IMAGE_JUDGE_CATEGORIES)
    speech = judge_average(run_dir, stage, "speech", SPEECH_JUDGE_CATEGORIES)
    joint = judge_average(run_dir, stage, "joint", OMNI_JUDGE_CATEGORIES)
    classifier = speech_classifier_summary(run_dir) if stage == "1" else {}

    return {
        "run_name": run_name,
        "stage": stage,
        "text_model": config.text_model,
        "image_model": config.image_model,
        "speech_model": config.speech_model,
        "joint_model": config.joint_model,
        "llm_judge_text": text["average"],
        "llm_judge_image": image["average"],
        "llm_judge_speech": speech["average"],
        "llm_judge_joint": joint["average"],
        "bert_score_text": aux_average(run_dir, "bertscore") if stage == "1" else None,
        "clip_similarity_image": aux_average(run_dir, "clip") if stage == "1" else None,
        "speech_classifier_metadata_accuracy": classifier.get("average_metadata_accuracy"),
        "speech_classifier_exact_match_accuracy": classifier.get("exact_match_accuracy"),
        "details": {
            "text": text,
            "image": image,
            "speech": speech,
            "joint": joint,
            "speech_classifier": classifier,
        },
    }


def format_value(value: Any, decimals: int) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{decimals}f}"
    return ""


def write_csv_table(path: Path, rows: Sequence[Dict[str, Any]], decimals: int) -> None:
    output_path = ensure_inside_harness(path, "output csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    header: (
                        format_value(row.get(header), decimals)
                        if header not in {"run_name", "stage", "text_model", "image_model", "speech_model", "joint_model"}
                        else str(row.get(header) or "")
                    )
                    for header in TABLE_HEADERS
                }
            )


def write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], decimals: int) -> None:
    output_path = ensure_inside_harness(path, "output markdown")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(TABLE_HEADERS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |",
    ]
    for row in rows:
        values = []
        for header in TABLE_HEADERS:
            if header in {"run_name", "stage", "text_model", "image_model", "speech_model", "joint_model"}:
                values.append(str(row.get(header) or ""))
            else:
                values.append(format_value(row.get(header), decimals))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = ensure_inside_harness(args.run_dir, "run dir")
    stages = parse_stages(args.stages)
    run_name = run_dir.name
    rows = [build_stage_row(run_name, run_dir, stage) for stage in stages]
    payload = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "stages": stages,
        "rows": rows,
        "zero_fill_policy": "Missing, skipped, error, and parse-error judge scores are averaged as 0.0.",
    }
    write_json(args.output_json, payload)
    write_csv_table(args.output_csv, rows, args.decimals)
    write_markdown_table(args.output_md, rows, args.decimals)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
