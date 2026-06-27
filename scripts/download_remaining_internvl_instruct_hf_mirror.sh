#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/root/storage/mahaoxuan.mhx/model}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"
MAX_WORKERS="${MAX_WORKERS:-8}"
RETRIES="${RETRIES:-3}"
REVISION=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_remaining_internvl_instruct_hf_mirror.sh [options]

Download the remaining InternVL Instruct checkpoints needed under:
  /root/storage/mahaoxuan.mhx/model

This intentionally excludes the base InternVL3-* checkpoints and the already
completed InternVL3-1B-Instruct / InternVL3_5-4B-Instruct / InternVL3_5-8B-Instruct.

Options:
  --target-dir DIR      Download root. Default: /root/storage/mahaoxuan.mhx/model
  --max-workers N       Parallel file workers per model. Default: 8
  --retries N           Retry each repo up to N times. Default: 3
  --revision REV        Optional Hugging Face revision.
  --dry-run             Print planned downloads only.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir) TARGET_DIR="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
else
  echo "Missing hf CLI. Activate an environment with huggingface_hub installed." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${TARGET_DIR}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export NO_PROXY="*"
export no_proxy="*"

repos=(
  OpenGVLab/InternVL3-2B-Instruct
  OpenGVLab/InternVL3-8B-Instruct
  OpenGVLab/InternVL3-9B-Instruct
  OpenGVLab/InternVL3-14B-Instruct
  OpenGVLab/InternVL3-38B-Instruct
  OpenGVLab/InternVL3_5-1B-Instruct
  OpenGVLab/InternVL3_5-2B-Instruct
  OpenGVLab/InternVL3_5-14B-Instruct
  OpenGVLab/InternVL3_5-38B-Instruct
)

echo "hf_bin=${HF_BIN}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_hub_disable_xet=${HF_HUB_DISABLE_XET}"
echo "hf_hub_enable_hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "target_dir=${TARGET_DIR}"
echo "max_workers=${MAX_WORKERS}"
echo "retries=${RETRIES}"
echo "num_repos=${#repos[@]}"

for repo in "${repos[@]}"; do
  local_name="${repo#OpenGVLab/}"
  local_dir="${TARGET_DIR}/${local_name}"
  mkdir -p "${local_dir}"

  cmd=(
    "${HF_BIN}" download "${repo}"
    --repo-type model
    --local-dir "${local_dir}"
    --max-workers "${MAX_WORKERS}"
  )
  if [[ -n "${REVISION}" ]]; then
    cmd+=(--revision "${REVISION}")
  fi

  echo
  echo "repo=${repo}"
  echo "local_dir=${local_dir}"
  printf 'command='
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  attempt=1
  while true; do
    echo "attempt=${attempt}/${RETRIES}"
    set +e
    "${cmd[@]}"
    status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
      echo "repo_done=${repo}"
      break
    fi
    if [[ "${attempt}" -ge "${RETRIES}" ]]; then
      echo "repo_failed=${repo} status=${status}" >&2
      exit "${status}"
    fi
    attempt=$((attempt + 1))
    sleep 10
  done
done

echo
echo "done"
