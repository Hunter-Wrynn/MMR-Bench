#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${REPO_ROOT}/third_party/VLMEvalKit"
BASE_RUNNER="${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh"
DEFAULT_CONFIG_FILE="${REPO_ROOT}/configs/qwen25vl_72b_vllm_mmrbench.env"
DEFAULT_MMR_BENCHMARKS="MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only"

CONFIG_FILE="${MMR_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
CLI_BENCHMARKS=""
CLI_JUDGE_IP=""
CLI_JUDGE_PORT=""
CLI_JUDGE_BASE_URL=""
CLI_JUDGE_MODEL=""
CLI_JUDGE_KEY=""
CLI_MODE=""
CLI_RUN_NAME=""
CLI_WORK_DIR=""
CLI_LOG_DIR=""
CLI_CSV_PATH=""
CLI_MODEL=""
CLI_MODEL_PATH=""
CLI_MODEL_CLASS=""
CLI_MODEL_ARGS_JSON=""
CLI_GPUS=""
CLI_NPROC=""
CLI_MASTER_PORT=""
CLI_USE_VLLM=""
CLI_USE_COT=""
CONTINUE_ON_ERROR=0
MERGE_REQUESTED=""
DO_BACKUP=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mmrbench_vlmeval_one_by_one.sh [options]

Runs MMR-Bench benchmarks sequentially. Each benchmark gets its own VLMEvalKit
launch and log file, while all outputs are written under the same work-dir so
the final merge can collect the standard 7 benchmarks.

Default target:
  Qwen2.5-VL-72B-Instruct, vLLM TP=8, judge http://33.41.18.26:8000/v1.

Examples:
  # Full MMR-Bench, one benchmark at a time
  scripts/run_mmrbench_vlmeval_one_by_one.sh

  # Run only two benchmarks
  scripts/run_mmrbench_vlmeval_one_by_one.sh --benchmarks MMStar,MathVista_MINI

  # Switch judge IP
  scripts/run_mmrbench_vlmeval_one_by_one.sh --judge-ip 33.3.183.91

  # Use a different model config
  scripts/run_mmrbench_vlmeval_one_by_one.sh --config-file configs/qwen25vl_32b_mmrbench_controlled.env --no-use-vllm

Options:
  --config-file PATH      Runtime config file.
  --benchmarks LIST       Comma-separated or space-separated benchmark list.
  --judge-ip IP           Judge host IP. Default: 33.41.18.26.
  --judge-port PORT       Judge port. Default: 8000.
  --judge-base-url URL    Full judge base URL.
  --judge-model NAME      Judge model name.
  --judge-key KEY         Judge API key.
  --mode all|infer|eval   VLMEvalKit mode. Default comes from config, usually all.
  --run-name NAME         Shared run name prefix.
  --work-dir DIR          Shared VLMEvalKit work-dir root.
  --log-dir DIR           Log directory.
  --csv PATH              MMR-Bench CSV path.
  --model NAME            Override model key.
  --model-path PATH       Override model path.
  --model-class CLASS     Override VLMEvalKit model class.
  --model-args-json JSON  Override model args JSON.
  --gpus LIST             CUDA_VISIBLE_DEVICES list.
  --nproc N               torchrun nproc-per-node.
  --master-port PORT      Base master port. Each benchmark increments this by index.
  --use-vllm              Enable VLMEvalKit --use-vllm.
  --no-use-vllm           Disable VLMEvalKit --use-vllm.
  --use-cot 0|1           Set USE_COT.
  --merge                 Force final merge. Only valid for the standard 7 benchmarks.
  --no-merge              Skip final merge.
  --no-backup             Do not create data/MMR-Bench.before_<model>.csv.
  --continue-on-error     Keep running later benchmarks if one benchmark fails.
  --dry-run               Print generated commands without running them.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --benchmarks) CLI_BENCHMARKS="$2"; shift 2 ;;
    --judge-ip) CLI_JUDGE_IP="$2"; shift 2 ;;
    --judge-port) CLI_JUDGE_PORT="$2"; shift 2 ;;
    --judge-base-url) CLI_JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-model) CLI_JUDGE_MODEL="$2"; shift 2 ;;
    --judge-key) CLI_JUDGE_KEY="$2"; shift 2 ;;
    --mode) CLI_MODE="$2"; shift 2 ;;
    --run-name) CLI_RUN_NAME="$2"; shift 2 ;;
    --work-dir) CLI_WORK_DIR="$2"; shift 2 ;;
    --log-dir) CLI_LOG_DIR="$2"; shift 2 ;;
    --csv) CLI_CSV_PATH="$2"; shift 2 ;;
    --model) CLI_MODEL="$2"; shift 2 ;;
    --model-path) CLI_MODEL_PATH="$2"; shift 2 ;;
    --model-class) CLI_MODEL_CLASS="$2"; shift 2 ;;
    --model-args-json) CLI_MODEL_ARGS_JSON="$2"; shift 2 ;;
    --gpus) CLI_GPUS="$2"; shift 2 ;;
    --nproc) CLI_NPROC="$2"; shift 2 ;;
    --master-port) CLI_MASTER_PORT="$2"; shift 2 ;;
    --use-vllm) CLI_USE_VLLM=1; shift ;;
    --no-use-vllm) CLI_USE_VLLM=0; shift ;;
    --use-cot) CLI_USE_COT="$2"; shift 2 ;;
    --merge) MERGE_REQUESTED=1; shift ;;
    --no-merge) MERGE_REQUESTED=0; shift ;;
    --no-backup) DO_BACKUP=0; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing runtime config: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -x "${BASE_RUNNER}" ]]; then
  echo "Missing base runner: ${BASE_RUNNER}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

CONDA_ENV="${VLMEVAL_CONDA_ENV:-${CONDA_ENV:-/root/storage/miniconda3/envs/vlmevalkit}}"
MODEL="${CLI_MODEL:-${MMR_MODEL:-${MODEL:-}}}"
MODEL_PATH="${CLI_MODEL_PATH:-${MMR_MODEL_PATH:-${MODEL_PATH:-}}}"
MODEL_CLASS="${CLI_MODEL_CLASS:-${MMR_MODEL_CLASS:-${MODEL_CLASS:-}}}"
DEFAULT_MODEL_ARGS_JSON='{}'
MODEL_ARGS_JSON="${MODEL_ARGS_JSON:-${DEFAULT_MODEL_ARGS_JSON}}"
MODEL_ARGS_JSON="${MMR_MODEL_ARGS_JSON:-${MODEL_ARGS_JSON}}"
MODEL_ARGS_JSON="${CLI_MODEL_ARGS_JSON:-${MODEL_ARGS_JSON}}"
CSV_PATH="${CLI_CSV_PATH:-${MMR_CSV_PATH:-${CSV_PATH:-${REPO_ROOT}/data/MMR-Bench.csv}}}"
BENCHMARKS="${CLI_BENCHMARKS:-${MMR_BENCHMARKS:-${BENCHMARKS:-${DEFAULT_MMR_BENCHMARKS}}}}"

JUDGE_MODEL="${CLI_JUDGE_MODEL:-${MMR_JUDGE_MODEL:-${JUDGE_MODEL:-Qwen3.5-122B-A10B}}}"
JUDGE_IP="${CLI_JUDGE_IP:-${MMR_JUDGE_IP:-33.41.18.26}}"
JUDGE_PORT="${CLI_JUDGE_PORT:-${MMR_JUDGE_PORT:-${JUDGE_PORT:-8000}}}"
JUDGE_BASE_URL="${CLI_JUDGE_BASE_URL:-${MMR_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-}}}"
JUDGE_KEY="${CLI_JUDGE_KEY:-${MMR_JUDGE_KEY:-${JUDGE_KEY:-EMPTY}}}"

GPUS="${CLI_GPUS:-${MMR_GPUS:-${GPUS:-0,1,2,3,4,5,6,7}}}"
NPROC="${CLI_NPROC:-${MMR_NPROC:-${NPROC:-}}}"
MASTER_PORT="${CLI_MASTER_PORT:-${MMR_MASTER_PORT:-${MASTER_PORT:-29574}}}"
MODE="${CLI_MODE:-${MMR_MODE:-${MODE:-all}}}"
USE_VLLM="${CLI_USE_VLLM:-${MMR_USE_VLLM:-${USE_VLLM:-0}}}"
USE_COT="${CLI_USE_COT:-${MMR_USE_COT:-${USE_COT:-1}}}"
LOG_DIR="${CLI_LOG_DIR:-${MMR_LOG_DIR:-${LOG_DIR:-${REPO_ROOT}/logs}}}"

if [[ -z "${MODEL}" || -z "${MODEL_PATH}" || -z "${MODEL_CLASS}" ]]; then
  echo "MODEL, MODEL_PATH, and MODEL_CLASS must be set by config or CLI." >&2
  exit 2
fi
if [[ -z "${JUDGE_BASE_URL}" ]]; then
  JUDGE_BASE_URL="http://${JUDGE_IP}:${JUDGE_PORT}/v1"
fi
if [[ "${MODE}" != "all" && "${MODE}" != "infer" && "${MODE}" != "eval" ]]; then
  echo "Invalid --mode: ${MODE}. Expected all, infer, or eval." >&2
  exit 2
fi
if [[ "${USE_VLLM}" != "0" && "${USE_VLLM}" != "1" ]]; then
  echo "Invalid USE_VLLM value: ${USE_VLLM}. Expected 0 or 1." >&2
  exit 2
fi
if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  echo "Missing python in conda env: ${CONDA_ENV}" >&2
  exit 1
fi

BENCHMARKS_CANON="$("${CONDA_ENV}/bin/python" - "$BENCHMARKS" <<'PY'
import re
import sys
items = [x for x in re.split(r'[\s,]+', sys.argv[1].strip()) if x]
print(','.join(items))
PY
)"
if [[ -z "${BENCHMARKS_CANON}" ]]; then
  echo "No benchmarks selected." >&2
  exit 2
fi

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
MODEL_SAFE="$("${CONDA_ENV}/bin/python" - "$MODEL" <<'PY'
import re
import sys
print(re.sub(r'[^A-Za-z0-9_.-]+', '_', sys.argv[1]).strip('_') or 'model')
PY
)"
RUN_NAME="${CLI_RUN_NAME:-${MMR_RUN_NAME:-mmrbench_one_by_one_${MODEL_SAFE}_${TIMESTAMP}}}"
WORK_DIR="${CLI_WORK_DIR:-${MMR_WORK_DIR:-${VLMEVAL_DIR}/outputs/${RUN_NAME}}}"
mkdir -p "${LOG_DIR}" "${WORK_DIR}"

MERGE_ALLOWED="$("${CONDA_ENV}/bin/python" - "$BENCHMARKS_CANON" "$DEFAULT_MMR_BENCHMARKS" <<'PY'
import sys
selected = set(x for x in sys.argv[1].split(',') if x)
default = set(x for x in sys.argv[2].split(',') if x)
print("1" if selected == default else "0")
PY
)"
if [[ "${MERGE_REQUESTED}" == "1" && "${MERGE_ALLOWED}" != "1" ]]; then
  echo "--merge is only supported when running the standard 7 MMR-Bench benchmarks." >&2
  exit 2
fi
if [[ "${MERGE_REQUESTED}" == "0" || "${MODE}" == "infer" ]]; then
  DO_MERGE=0
elif [[ "${MERGE_REQUESTED}" == "1" ]]; then
  DO_MERGE=1
else
  DO_MERGE="${MERGE_ALLOWED}"
fi

IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS_CANON}"
FAILED=()

echo "run_name=${RUN_NAME}"
echo "config_file=${CONFIG_FILE}"
echo "model=${MODEL}"
echo "benchmarks=${BENCHMARKS_CANON}"
echo "judge_base_url=${JUDGE_BASE_URL}"
echo "mode=${MODE}"
echo "use_vllm=${USE_VLLM}"
echo "gpus=${GPUS}"
echo "nproc=${NPROC:-auto}"
echo "work_dir=${WORK_DIR}"
echo "log_dir=${LOG_DIR}"
echo "final_merge_to_mmr_csv=${DO_MERGE}"

for idx in "${!BENCH_ARRAY[@]}"; do
  bench="${BENCH_ARRAY[$idx]}"
  bench_safe="$("${CONDA_ENV}/bin/python" - "$bench" <<'PY'
import re
import sys
print(re.sub(r'[^A-Za-z0-9_.-]+', '_', sys.argv[1]).strip('_') or 'benchmark')
PY
)"
  bench_port="$((MASTER_PORT + idx))"
  bench_run_name="${RUN_NAME}_${bench_safe}"

  cmd=(
    "${BASE_RUNNER}"
    --config-file "${CONFIG_FILE}"
    --model "${MODEL}"
    --model-path "${MODEL_PATH}"
    --model-class "${MODEL_CLASS}"
    --model-args-json "${MODEL_ARGS_JSON}"
    --benchmarks "${bench}"
    --judge-base-url "${JUDGE_BASE_URL}"
    --judge-model "${JUDGE_MODEL}"
    --judge-key "${JUDGE_KEY}"
    --gpus "${GPUS}"
    --master-port "${bench_port}"
    --work-dir "${WORK_DIR}"
    --csv "${CSV_PATH}"
    --run-name "${bench_run_name}"
    --mode "${MODE}"
    --use-cot "${USE_COT}"
    --no-merge
  )
  if [[ -n "${NPROC}" ]]; then
    cmd+=(--nproc "${NPROC}")
  fi
  if [[ "${USE_VLLM}" -eq 1 ]]; then
    cmd+=(--use-vllm)
  else
    cmd+=(--no-use-vllm)
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi

  echo
  echo "===== [$((idx + 1))/${#BENCH_ARRAY[@]}] ${bench} ====="
  printf 'command='
  printf ' %q' MMR_LOG_DIR="${LOG_DIR}" "${cmd[@]}"
  printf '\n'

  if ! MMR_LOG_DIR="${LOG_DIR}" "${cmd[@]}"; then
    FAILED+=("${bench}")
    if [[ "${CONTINUE_ON_ERROR}" -ne 1 ]]; then
      echo "Stopping after failed benchmark: ${bench}" >&2
      exit 1
    fi
  fi
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed benchmark(s): ${FAILED[*]}" >&2
  exit 1
fi

if [[ "${DO_MERGE}" -eq 1 ]]; then
  merge_cmd=(
    "${REPO_ROOT}/scripts/merge_vlmeval_results_to_mmr.py"
    --csv "${CSV_PATH}"
    --run-root "${WORK_DIR}"
    --model "${MODEL}"
    --model-path "${MODEL_PATH}"
    --judge "${JUDGE_MODEL}"
    --fix-mistral-regex
  )
  if [[ "${DO_BACKUP}" -eq 1 ]]; then
    merge_cmd+=(--backup)
  fi

  echo
  echo "===== final merge ====="
  printf 'command='
  printf ' %q' PYTHONPATH="${VLMEVAL_DIR}:${PYTHONPATH:-}" "${CONDA_ENV}/bin/python" "${merge_cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" -ne 1 ]]; then
    PYTHONPATH="${VLMEVAL_DIR}:${PYTHONPATH:-}" "${CONDA_ENV}/bin/python" "${merge_cmd[@]}"
  fi
else
  echo
  echo "Skipping final merge. It is enabled only for the standard 7 benchmarks in mode=all/eval unless --merge is passed."
fi

echo "done"
