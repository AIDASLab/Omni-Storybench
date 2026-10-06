#!/usr/bin/env python3
"""Import-only smoke checks for the stage-specific Conda environments."""

from __future__ import annotations

import importlib
import sys


MODULES_BY_PROFILE = {
    "dataset": (
        "pyarrow.parquet",
        "src.dataset",
        "scripts.prepare_dataset_manifest",
    ),
    "stage1": (
        "src.inference",
        "src.inference.vllm_text",
        "src.inference.vllm_image",
        "src.inference.audio_flamingo",
        "scripts.evaluate_modality",
    ),
    "vllm": (
        "src.inference",
        "src.inference.vllm_text",
        "src.inference.vllm_image",
        "src.inference.vllm_audio",
        "src.inference.vllm_omni",
        "scripts.evaluate_modality",
        "scripts.evaluate_joint",
    ),
    "omni": (
        "src.inference",
        "src.inference.qwen3_omni",
        "scripts.evaluate_joint",
    ),
    "speech": (
        "src.inference",
        "src.inference.audio_flamingo",
        "scripts.evaluate_modality",
    ),
    "repair": (
        "src.inference.json_repair",
        "scripts.repair_stage_results",
    ),
    "classifier": (
        "scripts.evaluate_speech_metadata",
        "scripts.speech_metadata_classifier",
    ),
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MODULES_BY_PROFILE:
        profiles = ", ".join(sorted(MODULES_BY_PROFILE))
        raise SystemExit(f"usage: {sys.argv[0]} <{profiles}>")

    profile = sys.argv[1]
    for module_name in MODULES_BY_PROFILE[profile]:
        importlib.import_module(module_name)
    print(f"{profile} imports: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
