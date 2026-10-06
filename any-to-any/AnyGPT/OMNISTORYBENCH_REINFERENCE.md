# AnyGPT generation protocol for Omni-StoryBench

See [README.md](README.md) for environment setup, checkpoint sources, dataset download, configurable paths, and launch commands.

The benchmark adapter `anygpt/src/infer/run_omnistorybench_hf.py` uses the row order in `data/omni_storybench.parquet` as canonical sample indices `0..899`. It reads prompt-safe columns and validates referenced files without loading ground-truth columns or sample JSON contents.

For each sample it performs one current-image encode, one text generation, one image generation, and one speech generation. The three prompt paths, generation configurations, and model generation arguments follow the benchmark's AnyGPT protocol. There is no explicit seed, per-sample reseeding, retry, or automatic recovery pass. Run the full index range in one process to preserve the continuous random-state progression.

The default result directory is `evaluation/baselines/AnyGPT/` from the repository root. Set `--results-root` to use another location. Each modality is saved immediately as:

```text
<results-root>/
├── text/<index>/generated.txt
├── image/<index>/generated.png
└── speech/<index>/generated.wav
```

If several image or speech segments are returned, all are decoded in order and only the first is saved. Missing segments produce no file. If a later pass raises an exception, earlier artifacts remain and the next sample is attempted. Re-running an index range can overwrite existing files; use a fresh result directory for a separate experiment.

`--validate-only` checks dataset and runtime assets without initializing the model. `--verify-results-only` additionally checks for exactly 900 complete text/image/speech triplets. The strict audit can report missing artifacts after a completed single-pass run; it does not trigger generation or retry.
