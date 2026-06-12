#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/root/storage/mahaoxuan.mhx/model}"
REPO_ID="${REPO_ID:-Qwen/Qwen2.5-VL-72B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-${MODEL_ROOT}/Qwen2.5-VL-72B-Instruct}"
MAX_WORKERS="${MAX_WORKERS:-8}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export NO_PROXY="*"

mkdir -p "${LOCAL_DIR}"

echo "repo_id=${REPO_ID}"
echo "local_dir=${LOCAL_DIR}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "max_workers=${MAX_WORKERS}"

exec hf download "${REPO_ID}" \
  --repo-type model \
  --local-dir "${LOCAL_DIR}" \
  --max-workers "${MAX_WORKERS}"
