#!/usr/bin/env bash
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_ROOT=""
GPUS=""
# Stages 1, 2 and 3 correspond to paper judge panels A, B and C.
STAGES="1"
RUN_NAME=""
DATASET_ROOT="${OMNISTORYBENCH_DATASET_ROOT:-$HARNESS_DIR/downloaded_dataset}"
DATASET_CONDA_ENV="${OMNISTORYBENCH_DATASET_CONDA_ENV:-omnistory-judge-a}"
START_INDEX=0
END_INDEX=""
LIMIT=0
DRY_RUN=0

TEXT_VLLM_MAX_MODEL_LEN=8192
IMAGE_VLLM_MAX_MODEL_LEN=20000
SPEECH_VLLM_MAX_MODEL_LEN=32768
KIMI_VLLM_MAX_MODEL_LEN=8192
JOINT_VLLM_MAX_MODEL_LEN=32768
VLLM_TENSOR_PARALLEL_SIZE=1
TEXT_VLLM_GPU_MEMORY_UTILIZATION=0.90
IMAGE_VLLM_GPU_MEMORY_UTILIZATION=0.90
SPEECH_VLLM_GPU_MEMORY_UTILIZATION=0.90
JOINT_VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_SWAP_SPACE=8

TEXT_BATCH_SIZE=16
IMAGE_BATCH_SIZE=4
SPEECH_BATCH_SIZE=1
JOINT_BATCH_SIZE=4
BERTSCORE_BATCH_SIZE=16
CLIP_BATCH_SIZE=4

TEXT_SEED=0
IMAGE_SEED=0
SPEECH_SEED=0
JOINT_SEED=1234
JSON_REPAIR_SEED=0
EVAL_MAX_NEW_TOKENS=1024
EVAL_JOINT_MAX_NEW_TOKENS=2048
EVAL_TEMPERATURE=0.0
EVAL_TOP_P=1.0
EVAL_REPETITION_PENALTY=1.0
JSON_REPAIR_CONDA_ENV="omnistory-judge-a"
JSON_REPAIR_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
JSON_REPAIR_BATCH_SIZE=16
MAX_SYSTEMIC_ERROR_RATE=0.5

TEXT_CONDA_ENV_STAGE1="omnistory-judge-a"
TEXT_CONDA_ENV_OTHER="omnistory-judge-bc"
IMAGE_CONDA_ENV_STAGE1="omnistory-judge-a"
IMAGE_CONDA_ENV_OTHER="omnistory-judge-bc"
SPEECH_CONDA_ENV_STAGE1="omnistory-judge-a"
SPEECH_CONDA_ENV_OTHER="omnistory-judge-bc"
JOINT_CONDA_ENV_STAGE1="omnistory-judge-a-omni"
JOINT_CONDA_ENV_OTHER="omnistory-judge-bc"
AUX_CONDA_ENV="omnistory-judge-a"
SPEECH_CLASSIFIER_CONDA_ENV="omnistory-speech"

TMPDIR_DEFAULT="${OMNISTORYBENCH_TMPDIR:-$HARNESS_DIR/.tmp}"

usage() {
  cat <<'EOF'
Usage:
  bash run_evaluation.sh --results-root <baseline_output_dir> --gpus 0,1,2,3 --stages 1,2,3 [options]

Required:
  --results-root PATH      Directory containing text/, image/, speech/ subdirectories.
  --gpus LIST             Four comma-separated GPU ids. Example: 0,1,2,3

Options:
  --stages LIST           Judge panels A/B/C as stages 1/2/3. Default: 1
  --run-name NAME         Output directory name under results/. Defaults to basename of --results-root.
  --dataset-root PATH     Local snu-aidas/Omni-StoryBench snapshot. Default: ./downloaded_dataset
  --dataset-conda-env NAME Environment used to validate parquet. Default: omnistory-judge-a
  --start-index N         Inclusive start index. Default: 0
  --end-index N           Exclusive end index. Default: dataset script default/full range
  --limit N               Optional number of records from start-index. Default: 0 means no limit
  --dry-run               Validate scheduling/input/output paths without loading models.

  --text-vllm-max-model-len N       Default: 8192
  --image-vllm-max-model-len N      Default: 20000
  --speech-vllm-max-model-len N     Stage 1/2 default: 32768
  --kimi-vllm-max-model-len N       Stage 3 Kimi-Audio default: 8192
  --joint-vllm-max-model-len N      Default: 32768
  --vllm-tensor-parallel-size N     Fixed default: 1 (one GPU per modality)
  --vllm-gpu-memory-utilization X   Override every modality (defaults: text/image/speech 0.90, joint 0.95)
  --vllm-swap-space X               Default: 8

  --seed N                          Override every judge/engine seed
  --max-new-tokens N                Text/image/speech generation limit. Default: 1024
  --joint-max-new-tokens N          Joint generation limit. Default: 2048
  --temperature X                   Shared sampling temperature. Default: 0.0
  --top-p X                         Shared nucleus sampling value. Default: 1.0
  --repetition-penalty X            Shared repetition penalty. Default: 1.0
  --json-repair-conda-env NAME      Stage-end repair environment. Default: omnistory-judge-a
  --json-repair-model NAME          Stage-end repair model.
  --json-repair-batch-size N        Default: 16
  --max-systemic-error-rate X       Block on modality runtime errors at or above this rate. Default: 0.5

  --text-conda-env NAME             Override text environment for every stage
  --text-conda-env-stage1 NAME      Default: omnistory-judge-a
  --text-conda-env-other NAME       Stage 2/3 default: omnistory-judge-bc
  --image-conda-env NAME            Override image environment for every stage
  --image-conda-env-stage1 NAME     Default: omnistory-judge-a
  --image-conda-env-other NAME      Stage 2/3 default: omnistory-judge-bc
  --speech-conda-env NAME           Override speech environment for every stage
  --speech-conda-env-stage1 NAME    Default: omnistory-judge-a
  --speech-conda-env-other NAME     Stage 2/3 default: omnistory-judge-bc
  --joint-conda-env-stage1 NAME     Default: omnistory-judge-a-omni; use "current" to skip conda run
  --joint-conda-env-other NAME      Default: omnistory-judge-bc; use "current" to skip conda run
  --aux-conda-env NAME              Default: omnistory-judge-a; use "current" to skip conda run
  --speech-classifier-conda-env NAME Default: omnistory-speech; use "current" to skip conda run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --dataset-conda-env) DATASET_CONDA_ENV="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --end-index) END_INDEX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --text-vllm-max-model-len) TEXT_VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --image-vllm-max-model-len) IMAGE_VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --speech-vllm-max-model-len) SPEECH_VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --kimi-vllm-max-model-len) KIMI_VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --joint-vllm-max-model-len) JOINT_VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --vllm-tensor-parallel-size) VLLM_TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --vllm-gpu-memory-utilization)
      TEXT_VLLM_GPU_MEMORY_UTILIZATION="$2"
      IMAGE_VLLM_GPU_MEMORY_UTILIZATION="$2"
      SPEECH_VLLM_GPU_MEMORY_UTILIZATION="$2"
      JOINT_VLLM_GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --vllm-swap-space) VLLM_SWAP_SPACE="$2"; shift 2 ;;
    --seed)
      TEXT_SEED="$2"
      IMAGE_SEED="$2"
      SPEECH_SEED="$2"
      JOINT_SEED="$2"
      JSON_REPAIR_SEED="$2"
      shift 2
      ;;
    --max-new-tokens) EVAL_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --joint-max-new-tokens) EVAL_JOINT_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) EVAL_TEMPERATURE="$2"; shift 2 ;;
    --top-p) EVAL_TOP_P="$2"; shift 2 ;;
    --repetition-penalty) EVAL_REPETITION_PENALTY="$2"; shift 2 ;;
    --json-repair-conda-env) JSON_REPAIR_CONDA_ENV="$2"; shift 2 ;;
    --json-repair-model) JSON_REPAIR_MODEL="$2"; shift 2 ;;
    --json-repair-batch-size) JSON_REPAIR_BATCH_SIZE="$2"; shift 2 ;;
    --max-systemic-error-rate) MAX_SYSTEMIC_ERROR_RATE="$2"; shift 2 ;;
    --text-conda-env)
      TEXT_CONDA_ENV_STAGE1="$2"
      TEXT_CONDA_ENV_OTHER="$2"
      shift 2
      ;;
    --text-conda-env-stage1) TEXT_CONDA_ENV_STAGE1="$2"; shift 2 ;;
    --text-conda-env-other) TEXT_CONDA_ENV_OTHER="$2"; shift 2 ;;
    --image-conda-env)
      IMAGE_CONDA_ENV_STAGE1="$2"
      IMAGE_CONDA_ENV_OTHER="$2"
      shift 2
      ;;
    --image-conda-env-stage1) IMAGE_CONDA_ENV_STAGE1="$2"; shift 2 ;;
    --image-conda-env-other) IMAGE_CONDA_ENV_OTHER="$2"; shift 2 ;;
    --speech-conda-env)
      SPEECH_CONDA_ENV_STAGE1="$2"
      SPEECH_CONDA_ENV_OTHER="$2"
      shift 2
      ;;
    --speech-conda-env-stage1) SPEECH_CONDA_ENV_STAGE1="$2"; shift 2 ;;
    --speech-conda-env-other) SPEECH_CONDA_ENV_OTHER="$2"; shift 2 ;;
    --joint-conda-env-stage1) JOINT_CONDA_ENV_STAGE1="$2"; shift 2 ;;
    --joint-conda-env-other) JOINT_CONDA_ENV_OTHER="$2"; shift 2 ;;
    --aux-conda-env) AUX_CONDA_ENV="$2"; shift 2 ;;
    --speech-classifier-conda-env) SPEECH_CLASSIFIER_CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RESULTS_ROOT" || -z "$GPUS" ]]; then
  usage
  exit 2
fi

RESULTS_ROOT="$(realpath "$RESULTS_ROOT")"
if [[ ! -d "$RESULTS_ROOT/text" || ! -d "$RESULTS_ROOT/image" || ! -d "$RESULTS_ROOT/speech" ]]; then
  echo "Input results root must contain text/, image/, and speech/: $RESULTS_ROOT" >&2
  exit 1
fi

DATASET_ROOT="$(realpath "$DATASET_ROOT")"
if [[ ! -f "$DATASET_ROOT/data/omni_storybench.parquet" ]]; then
  echo "Dataset root must contain data/omni_storybench.parquet: $DATASET_ROOT" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -ne 4 ]]; then
  echo "--gpus must contain exactly four comma-separated GPU ids; got: $GPUS" >&2
  exit 1
fi

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
for stage in "${STAGE_LIST[@]}"; do
  if [[ "$stage" != "1" && "$stage" != "2" && "$stage" != "3" ]]; then
    echo "Unsupported stage: $stage" >&2
    exit 1
  fi
done

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "$RESULTS_ROOT")"
fi

RUN_DIR="$HARNESS_DIR/results/$RUN_NAME"
DATASET_MANIFEST="$RUN_DIR/dataset_manifest.json"
mkdir -p "$RUN_DIR/logs"

export TMPDIR="$TMPDIR_DEFAULT"
export OMNISTORYBENCH_ALLOWED_WRITE_ROOT="$HARNESS_DIR"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$TMPDIR"

END_ARGS=()
if [[ -n "$END_INDEX" ]]; then
  END_ARGS=(--end-index "$END_INDEX")
fi
DRY_ARGS=()
if [[ "$DRY_RUN" -eq 1 ]]; then
  DRY_ARGS=(--dry-run)
fi
GENERATION_ARGS=(
  --temperature "$EVAL_TEMPERATURE"
  --top-p "$EVAL_TOP_P"
  --repetition-penalty "$EVAL_REPETITION_PENALTY"
)
SPEECH_SEED_ARGS=()
if [[ -n "$SPEECH_SEED" ]]; then
  SPEECH_SEED_ARGS=(--seed "$SPEECH_SEED")
fi

run_python() {
  local env_name="$1"
  shift
  if [[ "$env_name" == "current" || -z "$env_name" ]]; then
    python "$@"
  else
    conda run --no-capture-output -n "$env_name" python "$@"
  fi
}

start_job() {
  local name="$1"
  local gpu="$2"
  local env_name="$3"
  local log_file="$4"
  shift 4

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[job-start] $name gpu=$gpu env=$env_name"
    run_python "$env_name" "$@"
    echo "[job-done] $name"
  ) >"$log_file" 2>&1 &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_NAME["$pid"]="$name"
  RUNNING_PIDS+=("$pid")
  echo "[launch] pid=$pid name=$name gpu=$gpu log=$log_file"
}

remove_pid() {
  local done_pid="$1"
  local next=()
  for pid in "${RUNNING_PIDS[@]}"; do
    if [[ "$pid" != "$done_pid" ]]; then
      next+=("$pid")
    fi
  done
  RUNNING_PIDS=("${next[@]}")
}

start_stage_modality_jobs() {
  local stage="$1"
  local stage_dir="$RUN_DIR/stage_$stage"
  mkdir -p "$stage_dir/text" "$stage_dir/image" "$stage_dir/speech" "$stage_dir/joint"

  local text_env="$TEXT_CONDA_ENV_OTHER"
  local image_env="$IMAGE_CONDA_ENV_OTHER"
  local speech_env="$SPEECH_CONDA_ENV_OTHER"
  local joint_env="$JOINT_CONDA_ENV_OTHER"
  local speech_max_model_len="$SPEECH_VLLM_MAX_MODEL_LEN"
  if [[ "$stage" == "3" ]]; then
    speech_max_model_len="$KIMI_VLLM_MAX_MODEL_LEN"
  fi
  if [[ "$stage" == "1" ]]; then
    text_env="$TEXT_CONDA_ENV_STAGE1"
    image_env="$IMAGE_CONDA_ENV_STAGE1"
    speech_env="$SPEECH_CONDA_ENV_STAGE1"
    joint_env="$JOINT_CONDA_ENV_STAGE1"
  fi

  start_job "stage_${stage}_text" "${GPU_LIST[0]}" "$text_env" "$stage_dir/text/log.txt" \
    "$HARNESS_DIR/scripts/evaluate_modality.py" \
    --stage "$stage" --modality text \
    --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
    --results-root "$RESULTS_ROOT" \
    --output-json "$stage_dir/text/all_results.json" --summary-json "$stage_dir/text/summary.json" \
    --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
    --batch-size "$TEXT_BATCH_SIZE" --vllm-max-model-len "$TEXT_VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$TEXT_BATCH_SIZE" --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm-gpu-memory-utilization "$TEXT_VLLM_GPU_MEMORY_UTILIZATION" --vllm-swap-space "$VLLM_SWAP_SPACE" \
    --seed "$TEXT_SEED" --max-new-tokens "$EVAL_MAX_NEW_TOKENS" "${GENERATION_ARGS[@]}" "${DRY_ARGS[@]}"

  start_job "stage_${stage}_image" "${GPU_LIST[1]}" "$image_env" "$stage_dir/image/log.txt" \
    "$HARNESS_DIR/scripts/evaluate_modality.py" \
    --stage "$stage" --modality image \
    --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
    --results-root "$RESULTS_ROOT" \
    --output-json "$stage_dir/image/all_results.json" --summary-json "$stage_dir/image/summary.json" \
    --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
    --batch-size "$IMAGE_BATCH_SIZE" --vllm-max-model-len "$IMAGE_VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$IMAGE_BATCH_SIZE" --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm-gpu-memory-utilization "$IMAGE_VLLM_GPU_MEMORY_UTILIZATION" --vllm-swap-space "$VLLM_SWAP_SPACE" \
    --seed "$IMAGE_SEED" --max-new-tokens "$EVAL_MAX_NEW_TOKENS" "${GENERATION_ARGS[@]}" "${DRY_ARGS[@]}"

  start_job "stage_${stage}_speech" "${GPU_LIST[2]}" "$speech_env" "$stage_dir/speech/log.txt" \
    "$HARNESS_DIR/scripts/evaluate_modality.py" \
    --stage "$stage" --modality speech \
    --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
    --results-root "$RESULTS_ROOT" \
    --output-json "$stage_dir/speech/all_results.json" --summary-json "$stage_dir/speech/summary.json" \
    --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
    --batch-size "$SPEECH_BATCH_SIZE" --vllm-max-model-len "$speech_max_model_len" \
    --vllm-max-num-seqs "$SPEECH_BATCH_SIZE" --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm-gpu-memory-utilization "$SPEECH_VLLM_GPU_MEMORY_UTILIZATION" --vllm-swap-space "$VLLM_SWAP_SPACE" \
    "${SPEECH_SEED_ARGS[@]}" --max-new-tokens "$EVAL_MAX_NEW_TOKENS" "${GENERATION_ARGS[@]}" "${DRY_ARGS[@]}"

  start_job "stage_${stage}_joint" "${GPU_LIST[3]}" "$joint_env" "$stage_dir/joint/log.txt" \
    "$HARNESS_DIR/scripts/evaluate_joint.py" \
    --stage "$stage" \
    --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
    --results-root "$RESULTS_ROOT" \
    --output-json "$stage_dir/joint/all_results.json" --summary-json "$stage_dir/joint/summary.json" \
    --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
    --batch-size "$JOINT_BATCH_SIZE" --vllm-max-model-len "$JOINT_VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$JOINT_BATCH_SIZE" --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm-gpu-memory-utilization "$JOINT_VLLM_GPU_MEMORY_UTILIZATION" --vllm-swap-space "$VLLM_SWAP_SPACE" \
    --seed "$JOINT_SEED" --max-new-tokens "$EVAL_JOINT_MAX_NEW_TOKENS" "${GENERATION_ARGS[@]}" "${DRY_ARGS[@]}"
}

start_aux_job() {
  local aux_name="$1"
  local gpu="$2"
  local stage_dir="$RUN_DIR/stage_1"

  case "$aux_name" in
    speech_classifier)
      mkdir -p "$stage_dir/speech_classifier"
      local classifier_end=899
      if [[ -n "$END_INDEX" ]]; then
        classifier_end=$((END_INDEX - 1))
      fi
      start_job "stage_1_speech_classifier" "$gpu" "$SPEECH_CLASSIFIER_CONDA_ENV" "$stage_dir/speech_classifier/log.txt" \
        "$HARNESS_DIR/scripts/evaluate_speech_metadata.py" \
        --speech-root "$RESULTS_ROOT/speech" \
        --dataset-root "$DATASET_ROOT" \
        --dataset-manifest-path "$DATASET_MANIFEST" \
        --classifier-py "$HARNESS_DIR/scripts/speech_metadata_classifier.py" \
        --output-dir "$stage_dir/speech_classifier" \
        --start-index "$START_INDEX" --end-index "$classifier_end" --limit "$LIMIT" \
        "${DRY_ARGS[@]}"
      ;;
    bertscore)
      mkdir -p "$stage_dir/bertscore"
      start_job "stage_1_bertscore" "$gpu" "$AUX_CONDA_ENV" "$stage_dir/bertscore/log.txt" \
        "$HARNESS_DIR/scripts/evaluate_aux_metric.py" \
        --metric bertscore \
        --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
        --results-root "$RESULTS_ROOT" \
        --output-json "$stage_dir/bertscore/all_results.json" --summary-json "$stage_dir/bertscore/summary.json" \
        --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
        --batch-size "$BERTSCORE_BATCH_SIZE" "${DRY_ARGS[@]}"
      ;;
    clip)
      mkdir -p "$stage_dir/clip"
      start_job "stage_1_clip" "$gpu" "$AUX_CONDA_ENV" "$stage_dir/clip/log.txt" \
        "$HARNESS_DIR/scripts/evaluate_aux_metric.py" \
        --metric clip \
        --dataset-root "$DATASET_ROOT" --dataset-manifest-path "$DATASET_MANIFEST" \
        --results-root "$RESULTS_ROOT" \
        --output-json "$stage_dir/clip/all_results.json" --summary-json "$stage_dir/clip/summary.json" \
        --start-index "$START_INDEX" "${END_ARGS[@]}" --limit "$LIMIT" \
        --batch-size "$CLIP_BATCH_SIZE" "${DRY_ARGS[@]}"
      ;;
    *) echo "Unknown aux job: $aux_name" >&2; exit 1 ;;
  esac
}

run_stage() {
  local stage="$1"
  echo "[stage] starting stage $stage"
  declare -g -A PID_TO_GPU=()
  declare -g -A PID_TO_NAME=()
  declare -g -a RUNNING_PIDS=()
  local aux_queue=()
  if [[ "$stage" == "1" ]]; then
    aux_queue=(speech_classifier bertscore clip)
  fi

  start_stage_modality_jobs "$stage"

  while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    local done_pid=""
    local rc=0
    if wait -n -p done_pid "${RUNNING_PIDS[@]}"; then
      rc=0
    else
      rc=$?
    fi

    local done_name="${PID_TO_NAME[$done_pid]:-unknown}"
    local freed_gpu="${PID_TO_GPU[$done_pid]:-}"
    remove_pid "$done_pid"

    if [[ "$rc" -ne 0 ]]; then
      echo "[failed] $done_name pid=$done_pid rc=$rc" >&2
      echo "See log under $RUN_DIR/stage_$stage/" >&2
      exit "$rc"
    fi

    echo "[complete] $done_name freed_gpu=$freed_gpu"
    if [[ "${#aux_queue[@]}" -gt 0 && -n "$freed_gpu" ]]; then
      local next_aux="${aux_queue[0]}"
      aux_queue=("${aux_queue[@]:1}")
      start_aux_job "$next_aux" "$freed_gpu"
    fi
  done

  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "[stage-repair] starting stage $stage on gpu=${GPU_LIST[0]} env=$JSON_REPAIR_CONDA_ENV"
    (
      export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}"
      run_python "$JSON_REPAIR_CONDA_ENV" \
        "$HARNESS_DIR/scripts/repair_stage_results.py" \
        --stage "$stage" \
        --stage-dir "$RUN_DIR/stage_$stage" \
        --repair-model-name "$JSON_REPAIR_MODEL" \
        --batch-size "$JSON_REPAIR_BATCH_SIZE" \
        --seed "$JSON_REPAIR_SEED" \
        --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
        --temperature "$EVAL_TEMPERATURE" \
        --top-p "$EVAL_TOP_P" \
        --repetition-penalty "$EVAL_REPETITION_PENALTY" \
        --vllm-gpu-memory-utilization "$TEXT_VLLM_GPU_MEMORY_UTILIZATION" \
        --vllm-swap-space "$VLLM_SWAP_SPACE" \
        --vllm-max-model-len "$TEXT_VLLM_MAX_MODEL_LEN" \
        --max-systemic-error-rate "$MAX_SYSTEMIC_ERROR_RATE"
    )
  fi

  echo "[stage] finished stage $stage"
}

cd "$HARNESS_DIR"
echo "[config] run_name=$RUN_NAME"
echo "[config] run_dir=$RUN_DIR"
echo "[config] results_root=$RESULTS_ROOT"
echo "[config] dataset_root=$DATASET_ROOT manifest=$DATASET_MANIFEST env=$DATASET_CONDA_ENV"
echo "[config] stages=$STAGES gpus=$GPUS dry_run=$DRY_RUN"
echo "[config] stage1_envs=text:$TEXT_CONDA_ENV_STAGE1 image:$IMAGE_CONDA_ENV_STAGE1 speech:$SPEECH_CONDA_ENV_STAGE1 joint:$JOINT_CONDA_ENV_STAGE1 aux:$AUX_CONDA_ENV classifier:$SPEECH_CLASSIFIER_CONDA_ENV"
echo "[config] stage_other_envs=text:$TEXT_CONDA_ENV_OTHER image:$IMAGE_CONDA_ENV_OTHER speech:$SPEECH_CONDA_ENV_OTHER joint:$JOINT_CONDA_ENV_OTHER"
echo "[config] adapter_registry=src/model_adapters/stages.py"
echo "[config] generation=text_seed:$TEXT_SEED image_seed:$IMAGE_SEED speech_seed:${SPEECH_SEED:-unset} joint_seed:$JOINT_SEED repair_seed:$JSON_REPAIR_SEED modality_max_new_tokens:$EVAL_MAX_NEW_TOKENS joint_max_new_tokens:$EVAL_JOINT_MAX_NEW_TOKENS temperature:$EVAL_TEMPERATURE top_p:$EVAL_TOP_P repetition_penalty:$EVAL_REPETITION_PENALTY"
echo "[config] max_model_len=text:$TEXT_VLLM_MAX_MODEL_LEN image:$IMAGE_VLLM_MAX_MODEL_LEN speech:$SPEECH_VLLM_MAX_MODEL_LEN kimi:$KIMI_VLLM_MAX_MODEL_LEN joint:$JOINT_VLLM_MAX_MODEL_LEN"
echo "[config] vllm=tp:$VLLM_TENSOR_PARALLEL_SIZE text_mem:$TEXT_VLLM_GPU_MEMORY_UTILIZATION image_mem:$IMAGE_VLLM_GPU_MEMORY_UTILIZATION speech_mem:$SPEECH_VLLM_GPU_MEMORY_UTILIZATION joint_mem:$JOINT_VLLM_GPU_MEMORY_UTILIZATION swap:$VLLM_SWAP_SPACE"
echo "[config] stage_repair=env:$JSON_REPAIR_CONDA_ENV model:$JSON_REPAIR_MODEL batch_size:$JSON_REPAIR_BATCH_SIZE max_systemic_error_rate:$MAX_SYSTEMIC_ERROR_RATE require_zero_unresolved:0"
echo "[config] TMPDIR=$TMPDIR"
echo "[config] VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
echo "[dataset] validating downloaded snapshot and preparing manifest"
run_python "$DATASET_CONDA_ENV" \
  "$HARNESS_DIR/scripts/prepare_dataset_manifest.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-manifest "$DATASET_MANIFEST"
echo "[dataset] manifest ready: $DATASET_MANIFEST"

for stage in "${STAGE_LIST[@]}"; do
  run_stage "$stage"
done

python "$HARNESS_DIR/scripts/aggregate_results.py" \
  --run-dir "$RUN_DIR" \
  --stages "$STAGES" \
  --output-json "$RUN_DIR/final_summary.json" \
  --output-csv "$RUN_DIR/final_table.csv" \
  --output-md "$RUN_DIR/final_table.md"

echo "[done] final summary: $RUN_DIR/final_summary.json"
echo "[done] final csv: $RUN_DIR/final_table.csv"
echo "[done] final markdown: $RUN_DIR/final_table.md"
