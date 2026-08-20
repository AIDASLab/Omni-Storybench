#!/usr/bin/env python3
"""Import-check the unified environment without downloading model weights."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "numba"))


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        print(
            f"ERROR Python 3.10 is required; found {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return 1

    critical_versions = {
        "torch": "2.10.0",
        "transformers": "4.57.6",
        "vllm": "0.19.1",
        "numpy": "2.2.6",
        "protobuf": "6.33.4",
        "librosa": "0.11.0",
    }
    failures: list[str] = []
    for package, expected in critical_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{package}: not installed")
            continue
        if actual.split("+")[0] != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")
        else:
            print(f"OK version {package}=={actual}")

    # Each backend gets a fresh interpreter because EMOVA and MMaDA both use
    # the top-level module name `models`. They are separate runner processes in
    # normal use and should not contaminate one another during this check.
    checks = [
        ("runner", "import src.orchestrator"),
        ("OpenAI", "import openai"),
        ("Claude", "import anthropic"),
        ("vLLM", "import vllm"),
        ("Diffusers", "import diffusers"),
        (
            "Qwen2.5-Omni",
            "from transformers import "
            "Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor",
        ),
        ("EMOVA speech tokenizer", "import emova_speech_tokenizer.speech_utils"),
        (
            "Emu3",
            "from src.third_party import prepend_import_path, require_checkout; "
            "prepend_import_path(require_checkout('emu3')); "
            "from emu3.mllm.modeling_emu3 import Emu3ForCausalLM; "
            "from emu3.mllm.processing_emu3 import Emu3Processor",
        ),
        (
            "MMaDA",
            "from src.third_party import prepend_import_path, require_checkout; "
            "prepend_import_path(require_checkout('mmada')); "
            "from models import MAGVITv2, MMadaModelLM, get_mask_schedule; "
            "from training.prompting_utils import UniversalPrompting; "
            "from training.utils import image_transform, image_transform_squash",
        ),
        ("Parler-TTS", "import parler_tts"),
        ("VoxCPM", "import voxcpm"),
        ("FlashAttention", "import flash_attn"),
    ]
    child_environment = os.environ.copy()
    for label, code in checks:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"OK import {label}")
            continue
        detail = (result.stderr or result.stdout).strip()
        failures.append(f"{label}: {detail[-2000:]}")

    if failures:
        print("\nEnvironment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nUnified environment import check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
