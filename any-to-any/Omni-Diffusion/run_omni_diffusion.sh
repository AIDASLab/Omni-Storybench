#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${OMNIDIFF_PYTHON:-python}"

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    cat <<'USAGE'
Usage: bash run_omni_diffusion.sh --jsonl /path/to/story_inputs.jsonl [options]

This launcher accepts the legacy nested story JSONL format. For the public
Omni-StoryBench Parquet dataset, use run_omnistorybench_reinference.sh instead.

Options are forwarded to tools/run_omnibench_story_infer.py, including
--output_root, --start, --end, --seed, --dtype and --overwrite.
Select your Python with OMNIDIFF_PYTHON and your GPU with CUDA_VISIBLE_DEVICES.
Model asset options can override the repository-local defaults below.
USAGE
    exit 0
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${OMNIDIFF_GPU:-0}}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/third_party/GLM-4-Voice"

exec "$PYTHON_BIN" -B -s "$SCRIPT_DIR/tools/run_omnibench_story_infer.py" \
  --model_name_or_path "$SCRIPT_DIR/models/Omni-Diffusion" \
  --audio_tokenizer_path "$SCRIPT_DIR/models/THUDM/glm-4-voice-tokenizer" \
  --image_tokenizer_path "$SCRIPT_DIR/models/showlab/magvitv2" \
  --flow_path "$SCRIPT_DIR/models/THUDM/glm-4-voice-decoder" \
  --output_root "${OMNIDIFF_OUTPUT_ROOT:-$SCRIPT_DIR/results}" \
  --device cuda:0 \
  --start 0 \
  --end 900 \
  "$@"
