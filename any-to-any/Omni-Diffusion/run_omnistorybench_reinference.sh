#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${OMNIDIFF_PYTHON:-python}"
DATASET_ROOT="${OMNIDIFF_DATASET_ROOT:-$SCRIPT_DIR/../../evaluation/downloaded_dataset}"
OUTPUT_ROOT="${OMNIDIFF_OUTPUT_ROOT:-$SCRIPT_DIR/../../evaluation/baselines/Omni-Diffusion}"
CACHE_ROOT="$SCRIPT_DIR/.cache"
if ! PYTHON_BIN="$(command -v -- "$PYTHON_BIN")" || [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Set OMNIDIFF_PYTHON to a Python executable or activate your inference environment." >&2
  exit 2
fi
PYTHON_BIN_DIR="${PYTHON_BIN%/*}"
export PATH="$PYTHON_BIN_DIR:$PATH"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${OMNIDIFF_GPU:-0}}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/third_party/GLM-4-Voice"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules_pinned"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TORCH_HOME="$CACHE_ROOT/torch"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TMPDIR="$CACHE_ROOT/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_INDEX=1
unset PIP_CONSTRAINT

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec "$PYTHON_BIN" -B -s "$SCRIPT_DIR/tools/run_omnistorybench_reinference.py" --help
  fi
done

ensure_cache_dir() {
  local cache_dir="$1"
  case "$cache_dir" in
    "$CACHE_ROOT"|"$CACHE_ROOT"/*) ;;
    *)
      echo "refusing cache path outside copied repository: $cache_dir" >&2
      exit 2
      ;;
  esac
  if [[ -L "$cache_dir" || ( -e "$cache_dir" && ! -d "$cache_dir" ) ]]; then
    echo "cache path must be a real directory: $cache_dir" >&2
    exit 2
  fi
  mkdir -p -- "$cache_dir"
}

for cache_dir in "$CACHE_ROOT" "$HF_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" "$TMPDIR"; do
  ensure_cache_dir "$cache_dir"
done

exec "$PYTHON_BIN" -B -s "$SCRIPT_DIR/tools/run_omnistorybench_reinference.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  "$@"
