# Omni-StoryBench

Baseline generation code for Omni-StoryBench, a context-aware benchmark for
generating the next storybook page as text, image, and speech.

## Repository status

This is a cleaned release staging tree. It contains the baseline runner,
minimal configuration presets, and one locked environment for every backend.
The final license and public dataset format still need project-owner decisions.
See [docs/release-decisions.md](docs/release-decisions.md).

The large upstream Emu3 and MMaDA source trees are intentionally not vendored.
They are fetched reproducibly from their official repositories at immutable
commits recorded in `third_party/manifest.json`.

## Supported baselines

The release will support all seven research backends used by the project:

- OpenAI
- Anthropic Claude
- vLLM, with InternVL, GLM, and Qwen model presets
- Qwen2.5-Omni
- EMOVA
- Emu3
- MMaDA

OpenAI and Claude use hosted APIs; vLLM uses an OpenAI-compatible local server;
the remaining backends run their models locally.

## Installation

The backends share one environment. Install the locked Python 3.10 environment
with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run python scripts/bootstrap_third_party.py
uv run python scripts/check_environment.py
```

The environment converges on PyTorch 2.10.0 and Transformers 4.57.6. This is
intentional: vLLM needs Transformers 4.56 or newer, while Transformers 5 removes
interfaces still used by several research backends. Four legacy transitive
pins are overridden explicitly and all resolved packages are frozen in
`uv.lock`. See [docs/environment.md](docs/environment.md) for the compatibility
evidence, system prerequisites, and validation limits.

## Pinned third-party source

Fetch the exact Emu3 and MMaDA source revisions required by their adapters:

```bash
python3 scripts/bootstrap_third_party.py
python3 scripts/bootstrap_third_party.py --check
```

The checkouts are ignored by Git and are validated against full commit hashes
and required file paths. The runner resolves them only when the corresponding
backend is selected. Alternative existing checkouts can be supplied through
`OMNI_STORYBENCH_EMU3_ROOT` and `OMNI_STORYBENCH_MMADA_ROOT`.

Pinned revisions:

- Emu3: `dbbf9858194d70b8c58293e219ecffe22df0f9c7`
- MMaDA: `3cdb8709635beb29f66e4e23cd66060872ac3996`

No upstream patch is currently required. The required files at both commits
match the research snapshot; local import-path validation and cache integration
live in `src/third_party.py` and `src/orchestrator.py`. The bootstrap command
fetches source only; Python and CUDA packages come from the unified lockfile.

## Configuration

Credentials must be supplied through the environment. Start from
`.env.example`; never put tokens in source files.

The runner accepts portable path overrides:

```bash
python3 run.py \
  --config openai_flux_voxcpm \
  --dataset /path/to/omni-storybench \
  --output-root ./outputs \
  --cache-root ./.cache/artifacts \
  --start 0 \
  --until 1
```

Without flags, paths can be set through `OMNI_STORYBENCH_DATASET`,
`OMNI_STORYBENCH_OUTPUT`, and `OMNI_STORYBENCH_CACHE`.

The `configs/` directory contains one canonical preset for each distinct
backbone, image-generator, and speech-generator pipeline used in the research
code. Legacy aliases, superseded tuning copies, and files that differed only by
GPU/server placement were removed, reducing the directory from 44 to 28
presets. The explicitly named preset now defines the canonical generation
settings for that pipeline. CUDA device fields and local vLLM URLs are reference
placements and may need adjustment on another machine.

The dataset loader currently expects `data/omni_storybench.parquet` and sibling
asset directories. This contract must be reconciled with the public dataset
before the first release.

## Outputs

Each configured baseline writes one artifact per sample and modality:

```text
outputs/<baseline>/
  text/<index>/generated.txt
  image/<index>/generated.png
  speech/<index>/generated.wav
  raw/<index>/generated_raw.json
```

Runs continue across sample failures, record errors, and exit nonzero if any
sample failed.

## Development checks

The lightweight checks do not download models or require a GPU:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q run.py src scripts tests
```
