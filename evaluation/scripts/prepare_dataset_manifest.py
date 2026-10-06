#!/usr/bin/env python3
"""Validate the local HF snapshot and freeze its ordered evaluation records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path = [path for path in sys.path if "/.local/lib/python" not in path]
sys.path.insert(0, str(HARNESS_ROOT))

from src.dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_downloaded_dataset,
    write_dataset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a downloaded Omni-StoryBench snapshot and build "
            "a run-local manifest."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local root downloaded from snu-aidas/Omni-StoryBench.",
    )
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_downloaded_dataset(args.dataset_root)
    manifest_path = write_dataset_manifest(bundle, args.output_manifest)
    print(
        json.dumps(
            {
                **bundle.metadata,
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
