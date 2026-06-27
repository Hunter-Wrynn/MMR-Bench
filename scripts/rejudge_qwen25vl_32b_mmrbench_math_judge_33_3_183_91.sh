#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${REPO_ROOT}/third_party/VLMEvalKit"

CONDA_ENV="${VLMEVAL_CONDA_ENV:-/root/storage/miniconda3/envs/vlmevalkit}"
MODEL="${MMR_MODEL:-Qwen2.5-VL-32B-Instruct}"
MODEL_PATH="${MMR_MODEL_PATH:-/root/storage/mahaoxuan.mhx/model/Qwen2.5-VL-32B-Instruct}"
RUN_ROOT="${MMR_RUN_ROOT:-${VLMEVAL_DIR}/outputs/mmrbench_controlled_Qwen2.5-VL-32B-Instruct_20260612_081510}"
CSV_PATH="${MMR_CSV_PATH:-${REPO_ROOT}/data/MMR-Bench.csv}"
BENCHMARKS="${MMR_BENCHMARKS:-MathVision,MathVerse_MINI_Vision_Only}"

JUDGE_MODEL="${MMR_JUDGE_MODEL:-Qwen3.5-122B-A10B}"
JUDGE_IP="${MMR_JUDGE_IP:-33.3.183.91}"
JUDGE_PORT="${MMR_JUDGE_PORT:-8000}"
JUDGE_BASE_URL="${MMR_JUDGE_BASE_URL:-}"
JUDGE_KEY="${MMR_JUDGE_KEY:-EMPTY}"
JUDGE_NPROC="${MMR_JUDGE_NPROC:-8}"
JUDGE_RETRY="${MMR_JUDGE_RETRY:-2}"
JUDGE_TIMEOUT="${MMR_JUDGE_TIMEOUT:-900}"
JUDGE_ARGS="${MMR_JUDGE_ARGS:-{\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}}"

LOG_DIR="${MMR_LOG_DIR:-${REPO_ROOT}/logs}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${MMR_LOG_FILE:-${LOG_DIR}/rejudge_qwen25vl32b_math_${TIMESTAMP}.log}"
DO_MERGE=1
DO_BACKUP=1
HEALTH_CHECK=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/rejudge_qwen25vl_32b_mmrbench_math_judge_33_3_183_91.sh [options]

Re-evaluate the existing Qwen2.5-VL-32B MMR-Bench MathVision and MathVerse
prediction files with an OpenAI-compatible judge. This does not rerun 32B
inference and does not load the 32B model.

Options:
  --judge-ip IP           Judge host IP. Default: 33.3.183.91.
  --judge-port PORT       Judge port. Default: 8000.
  --judge-base-url URL    Full judge base URL, e.g. http://33.3.183.91:8000/v1.
  --judge-model NAME      Judge model name. Default: Qwen3.5-122B-A10B.
  --judge-key KEY         Judge API key. Default: EMPTY.
  --benchmarks LIST       Benchmarks to rejudge. Default: MathVision,MathVerse_MINI_Vision_Only.
  --run-root DIR          Existing VLMEvalKit work-dir root.
  --csv PATH              MMR-Bench CSV to merge into.
  --log-file PATH         Log path.
  --no-merge              Only generate score files; do not update data/MMR-Bench.csv.
  --no-backup             Do not create data/MMR-Bench.before_<model>.csv during merge.
  --skip-health-check     Skip the preflight judge API request.
  --dry-run               Print commands without running them.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-ip) JUDGE_IP="$2"; shift 2 ;;
    --judge-port) JUDGE_PORT="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-key) JUDGE_KEY="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --csv) CSV_PATH="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --no-merge) DO_MERGE=0; shift ;;
    --no-backup) DO_BACKUP=0; shift ;;
    --skip-health-check) HEALTH_CHECK=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${JUDGE_BASE_URL}" ]]; then
  JUDGE_BASE_URL="http://${JUDGE_IP}:${JUDGE_PORT}/v1"
fi

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  echo "Missing python in conda env: ${CONDA_ENV}" >&2
  exit 1
fi
if [[ ! -d "${RUN_ROOT}/${MODEL}" ]]; then
  echo "Missing existing VLMEvalKit model output root: ${RUN_ROOT}/${MODEL}" >&2
  exit 1
fi
if [[ "${DO_MERGE}" -eq 1 && ! -f "${CSV_PATH}" ]]; then
  echo "Missing CSV for merge: ${CSV_PATH}" >&2
  exit 1
fi

BENCHMARKS_CANON="$("${CONDA_ENV}/bin/python" - "$BENCHMARKS" <<'PY'
import re
import sys
items = [x for x in re.split(r'[\s,]+', sys.argv[1].strip()) if x]
print(','.join(items))
PY
)"
IFS=',' read -r -a BENCH_ARRAY <<< "${BENCHMARKS_CANON}"
if [[ "${#BENCH_ARRAY[@]}" -eq 0 ]]; then
  echo "No benchmarks selected." >&2
  exit 2
fi

JUDGE_HOST="${JUDGE_BASE_URL#http://}"
JUDGE_HOST="${JUDGE_HOST#https://}"
JUDGE_HOST="${JUDGE_HOST%%/*}"
JUDGE_HOST="${JUDGE_HOST%%:*}"
NO_PROXY_VALUE="${NO_PROXY:-${no_proxy:-}}"
for host in localhost 127.0.0.1 ::1 "${JUDGE_HOST}"; do
  if [[ ",${NO_PROXY_VALUE}," != *",${host},"* ]]; then
    NO_PROXY_VALUE="${NO_PROXY_VALUE:+${NO_PROXY_VALUE},}${host}"
  fi
done

mkdir -p "${LOG_DIR}" "$(dirname "${LOG_FILE}")"

RUN_CMD=(
  "${CONDA_ENV}/bin/python"
  run.py
  --model "${MODEL}"
  --data "${BENCH_ARRAY[@]}"
  --judge "${JUDGE_MODEL}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-key "${JUDGE_KEY}"
  --judge-api-nproc "${JUDGE_NPROC}"
  --judge-retry "${JUDGE_RETRY}"
  --judge-timeout "${JUDGE_TIMEOUT}"
  --judge-args "${JUDGE_ARGS}"
  --api-nproc 1
  --work-dir "${RUN_ROOT}"
  --mode eval
  --reuse
  --reuse-aux infer
)

MERGE_CMD=(
  "${REPO_ROOT}/scripts/merge_vlmeval_results_to_mmr.py"
  --csv "${CSV_PATH}"
  --run-root "${RUN_ROOT}"
  --model "${MODEL}"
  --model-path "${MODEL_PATH}"
  --judge "${JUDGE_MODEL}"
  --fix-mistral-regex
)
if [[ "${DO_BACKUP}" -eq 1 ]]; then
  MERGE_CMD+=(--backup)
fi

echo "model=${MODEL}"
echo "run_root=${RUN_ROOT}"
echo "benchmarks=${BENCHMARKS_CANON}"
echo "judge_base_url=${JUDGE_BASE_URL}"
echo "judge_model=${JUDGE_MODEL}"
echo "log_file=${LOG_FILE}"
echo "merge_to_mmr_csv=${DO_MERGE}"
printf 'rejudge_command='
printf ' %q' "${RUN_CMD[@]}"
printf '\n'
if [[ "${DO_MERGE}" -eq 1 ]]; then
  printf 'merge_command='
  printf ' %q' "${CONDA_ENV}/bin/python" "${MERGE_CMD[@]}"
  printf '\n'
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

if [[ "${HEALTH_CHECK}" -eq 1 ]]; then
  echo "Checking judge API..."
  if ! (
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    export NO_PROXY="${NO_PROXY_VALUE}"
    export no_proxy="${NO_PROXY_VALUE}"
    "${CONDA_ENV}/bin/python" - "$JUDGE_BASE_URL" "$JUDGE_MODEL" "$JUDGE_KEY" <<'PY'
import json
import sys
import urllib.request

base_url, model, key = sys.argv[1:4]
url = base_url.rstrip("/") + "/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Return exactly OK."}],
    "temperature": 0,
    "max_tokens": 16,
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    body = resp.read().decode("utf-8")
data = json.loads(body)
answer = data["choices"][0]["message"]["content"].strip()
if not answer:
    raise SystemExit("empty judge response")
print(answer[:200])
PY
  ); then
    echo "Judge API health check failed: ${JUDGE_BASE_URL}" >&2
    exit 1
  fi
fi

(
  cd "${VLMEVAL_DIR}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="${NO_PROXY_VALUE}"
  export no_proxy="${NO_PROXY_VALUE}"
  export PRED_FORMAT=xlsx
  "${RUN_CMD[@]}"
) >"${LOG_FILE}" 2>&1 || {
  status=$?
  echo "Rejudge failed with status ${status}. Tail of log:" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  exit "${status}"
}

echo "Rejudge finished."

if [[ "${DO_MERGE}" -eq 1 ]]; then
  PYTHONPATH="${VLMEVAL_DIR}:${PYTHONPATH:-}" "${CONDA_ENV}/bin/python" "${MERGE_CMD[@]}" | tee -a "${LOG_FILE}"
fi

echo "done"
