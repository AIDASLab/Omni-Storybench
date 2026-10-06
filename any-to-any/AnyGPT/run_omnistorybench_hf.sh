#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${ANYGPT_PYTHON:-python}"
DATASET_ROOT="${ANYGPT_DATASET_ROOT:-$REPO_DIR/../../evaluation/downloaded_dataset}"
RESULTS_ROOT="${ANYGPT_RESULTS_ROOT:-$REPO_DIR/../../evaluation/baselines/AnyGPT}"
CACHE_ROOT="${ANYGPT_CACHE_ROOT:-$REPO_DIR/.cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${ANYGPT_GPU:-0}}"
export ANYGPT_DIFFUSION_MODEL_PATH="$REPO_DIR/models/stable-diffusion-2-1-unclip"
export HF_HOME="$CACHE_ROOT/huggingface"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TMPDIR="$CACHE_ROOT/tmp"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
unset PIP_CONSTRAINT

ensure_cache_dir() {
  local cache_dir="$1"
  case "$cache_dir" in
    "$CACHE_ROOT"|"$CACHE_ROOT"/*) ;;
    *)
      echo "refusing cache path outside ANYGPT_CACHE_ROOT: $cache_dir" >&2
      exit 2
      ;;
  esac
  if [[ -L "$cache_dir" || ( -e "$cache_dir" && ! -d "$cache_dir" ) ]]; then
    echo "cache path must be a real directory: $cache_dir" >&2
    exit 2
  fi
  mkdir -p -- "$cache_dir"
}

for cache_dir in "$CACHE_ROOT" "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" "$TMPDIR"; do
  ensure_cache_dir "$cache_dir"
done

exec "$PYTHON_BIN" -B anygpt/src/infer/run_omnistorybench_hf.py \
  --dataset-root "$DATASET_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --model-name-or-path models/anygpt/chat \
  --image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
  --speech-tokenizer-path models/speechtokenizer/ckpt.dev \
  --speech-tokenizer-config models/speechtokenizer/config.json \
  --soundstorm-path models/soundstorm/speechtokenizer_soundstorm_mls.pt \
  --diffusion-model-path models/stable-diffusion-2-1-unclip \
  "$@"
