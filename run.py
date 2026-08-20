import glob
import os
import argparse
import sys
from typing import Dict, Any


def _maybe_reexec_with_venv_torch_lib() -> None:
    """Prefer this venv's libtorch over any globally injected torch libraries."""
    reexec_marker = "_OMNIBENCH_TORCH_LIB_REEXEC"
    if os.environ.get(reexec_marker) == "1":
        return

    venv_root = os.environ.get("VIRTUAL_ENV") or sys.prefix
    torch_lib = os.path.join(
        venv_root,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "torch",
        "lib",
    )
    if not os.path.isdir(torch_lib):
        return

    ld_library_path_entries = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if ld_library_path_entries and ld_library_path_entries[0] == torch_lib:
        return

    new_env = os.environ.copy()
    new_env[reexec_marker] = "1"
    new_env["LD_LIBRARY_PATH"] = ":".join([torch_lib] + [p for p in ld_library_path_entries if p != torch_lib])
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


_maybe_reexec_with_venv_torch_lib()


def is_sample_generated(output_dir_base: str, entry: Dict[str, Any]) -> bool:
    """True when all three modalities hold an artifact, as the harness resolves them."""
    index = entry["index"]
    expected = [("text", "txt"), ("image", "png"), ("speech", "wav")]
    return all(
        glob.glob(f"{output_dir_base}/{modality}/{index}/*.{ext}")
        for modality, ext in expected
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="openai_flux_voxcpm", help="Config name (loads configs/<config>.yaml)")
    parser.add_argument("--dataset", default=None, help="Dataset dir or parquet index. Overrides the config's dataset_path")
    parser.add_argument("--output-root", default=None, help="Results root holding one directory per baseline. Overrides the config's output_root")
    parser.add_argument("--cache-root", default=None, help="Artifact cache root shared by baselines. Overrides the config's cache_root")
    parser.add_argument("--no-cache", action="store_true", help="Generate every artifact from scratch and store nothing")
    parser.add_argument("--refresh", action="store_true", help="Regenerate every artifact and overwrite its cache entry")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--until", type=int, default=None, help="End index (exclusive). Default: process until the end")
    parser.add_argument("--repair", action="store_true", help="Skip samples already generated in the output dir")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.until is not None and args.until < args.start:
        parser.error("--until must be >= --start")

    # Select orchestrator based on config name
    if "openai" in args.config:
        from src.orchestrator import OpenAIOrchestrator
        orchestrator = OpenAIOrchestrator(args.config)
    elif "claude" in args.config:
        from src.orchestrator import ClaudeOrchestrator
        orchestrator = ClaudeOrchestrator(args.config)
    elif "vllm" in args.config:
        from src.orchestrator import VLLMOrchestrator
        orchestrator = VLLMOrchestrator(args.config)
    elif "qwen2_5omni" in args.config:
        from src.orchestrator import Qwen2_5OmniOrchestrator
        orchestrator = Qwen2_5OmniOrchestrator(args.config)
    elif "emova" in args.config:
        from src.orchestrator import EmovaOrchestrator
        orchestrator = EmovaOrchestrator(args.config)
    elif "MMaDA" in args.config:
        from src.orchestrator import MMaDAOrchestrator
        orchestrator = MMaDAOrchestrator(args.config)
    elif "Emu3" in args.config:
        from src.orchestrator import Emu3Orchestrator
        orchestrator = Emu3Orchestrator(args.config)
    else:
        raise ValueError(f"Unknown orchestrator: {args.config}")

    if args.output_root:
        orchestrator.set_output_root(args.output_root)

    from src.cache import DEFAULT_CACHE_ROOT, ArtifactCache
    orchestrator.set_cache(ArtifactCache(
        args.cache_root or orchestrator.config.get("cache_root") or DEFAULT_CACHE_ROOT,
        owner=orchestrator.config["output_path"],
        enabled=not args.no_cache,
        refresh=args.refresh,
    ))

    from src.dataset import DEFAULT_DATASET_PATH, load_dataset
    dataset_path = args.dataset or orchestrator.config.get("dataset_path") or DEFAULT_DATASET_PATH
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    print(f"Writing results to {orchestrator.output_dir_base}")

    from src.errors import affected_modalities, write_error_log
    from tqdm import tqdm

    processed = 0
    failed = 0
    for idx, item in enumerate(tqdm(dataset[args.start:args.until]), start=args.start):
        if args.repair and is_sample_generated(orchestrator.output_dir_base, item):
            print("Skipping already-generated sample:", item["book"], item["to_key"])
            continue

        print(f"[{idx}] Book: {item['book']} Predicting page: {item['to_key']}")

        # One bad sample must not end the run: record what it cost and move on.
        # A stage that fails alone is recorded by the orchestrator and leaves the
        # other modalities intact; anything raised took the whole sample down.
        errors = []
        try:
            orchestrator.format_input(item)
            orchestrator.forward()
        except Exception as exc:
            errors.append(exc)
        errors = getattr(orchestrator, "stage_errors", []) + errors

        for exc in errors:
            log_paths = write_error_log(orchestrator.output_dir_base, item, exc, args.config)
            print(f"[ERROR] [{idx}] {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"        lost {', '.join(affected_modalities(exc))}; recorded at {', '.join(log_paths)}", file=sys.stderr)

        if errors:
            failed += 1
        else:
            processed += 1

    print(f"Done. Processed {processed} items, {failed} with failures.")
    if failed:
        print(f"See {orchestrator.output_dir_base}/errors.jsonl")
    print(orchestrator.cache.summary())
    if failed:
        raise SystemExit(1)
