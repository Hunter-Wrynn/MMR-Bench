#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${REPO_ROOT}/third_party/VLMEvalKit"
CONDA_ENV="${VLMEVAL_CONDA_ENV:-/root/storage/miniconda3/envs/vlmevalkit}"

JUDGE_MODEL="${MMR_QWEN35_JUDGE_MODEL:-Qwen3.5-122B-A10B}"
JUDGE_BASE_URL="${MMR_QWEN35_JUDGE_BASE_URL:-http://33.3.175.120:8000/v1}"
JUDGE_KEY="${MMR_QWEN35_JUDGE_KEY:-EMPTY}"
JUDGE_NPROC="${MMR_QWEN35_JUDGE_NPROC:-4}"
JUDGE_RETRY="${MMR_QWEN35_JUDGE_RETRY:-2}"
JUDGE_TIMEOUT="${MMR_QWEN35_JUDGE_TIMEOUT:-600}"
JUDGE_ARGS="${MMR_QWEN35_JUDGE_ARGS:-{\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}}"

append_no_proxy() {
  local host="$1"
  local current="${NO_PROXY:-${no_proxy:-}}"
  if [[ ",${current}," != *",${host},"* ]]; then
    current="${current:+${current},}${host}"
  fi
  export NO_PROXY="${current}"
  export no_proxy="${current}"
}

append_no_proxy "33.3.175.120"

cd "${VLMEVAL_DIR}"

exec "${CONDA_ENV}/bin/python" run.py \
  --judge "${JUDGE_MODEL}" \
  --judge-base-url "${JUDGE_BASE_URL}" \
  --judge-key "${JUDGE_KEY}" \
  --judge-api-nproc "${JUDGE_NPROC}" \
  --judge-retry "${JUDGE_RETRY}" \
  --judge-timeout "${JUDGE_TIMEOUT}" \
  --judge-args "${JUDGE_ARGS}" \
  "$@"
