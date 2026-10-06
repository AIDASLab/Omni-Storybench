#!/usr/bin/env bash
set -euo pipefail

# Legacy JSONL adapter. Use run_omnistorybench_hf.sh for the public dataset.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${ANYGPT_PYTHON:-python}"
CACHE_ROOT="${ANYGPT_CACHE_ROOT:-$REPO_DIR/.cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${ANYGPT_GPU:-0}}"
export ANYGPT_DIFFUSION_MODEL_PATH="${ANYGPT_DIFFUSION_MODEL_PATH:-$REPO_DIR/models/stable-diffusion-2-1-unclip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PIP_CONSTRAINT PYTHONPATH

exec "$PYTHON_BIN" -B anygpt/src/infer/run_omnibench_story_batch.py \
  --model-name-or-path models/anygpt/chat \
  --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
  --speech-tokenizer-path models/speechtokenizer/ckpt.dev \
  --speech-tokenizer-config models/speechtokenizer/config.json \
  --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
  --results-root "$REPO_DIR/../../evaluation/baselines/AnyGPT_legacy" \
  --start-index 0 \
  --end-index 899 \
  "$@"
