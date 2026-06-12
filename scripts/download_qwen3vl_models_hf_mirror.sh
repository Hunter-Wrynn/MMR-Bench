#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/root/storage/mahaoxuan.mhx/model"
HF_ENDPOINT_VALUE="https://hf-mirror.com"
MAX_WORKERS="8"
REVISION=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_qwen3vl_models_hf_mirror.sh [options] [2B|4B|32B|30B-A3B ...]

Download native Qwen3-VL Instruct models through hf-mirror without proxies.
If no model size is specified, all default models are downloaded:
  2B, 4B, 32B, 30B-A3B

Options:
  --target-dir DIR      Download root. Default: /root/storage/mahaoxuan.mhx/model
  --max-workers N       Parallel download workers for hf download. Default: 8
  --revision REV        Optional Hugging Face revision.
  --dry-run             Print planned downloads only.
  -h, --help            Show this help.

Examples:
  scripts/download_qwen3vl_models_hf_mirror.sh
  scripts/download_qwen3vl_models_hf_mirror.sh 32B 30B-A3B
EOF
}

declare -A REPOS=(
  ["2B"]="Qwen/Qwen3-VL-2B-Instruct"
  ["4B"]="Qwen/Qwen3-VL-4B-Instruct"
  ["32B"]="Qwen/Qwen3-VL-32B-Instruct"
  ["30B-A3B"]="Qwen/Qwen3-VL-30B-A3B-Instruct"
)

declare -A LOCAL_NAMES=(
  ["2B"]="Qwen3-VL-2B-Instruct"
  ["4B"]="Qwen3-VL-4B-Instruct"
  ["32B"]="Qwen3-VL-32B-Instruct"
  ["30B-A3B"]="Qwen3-VL-30B-A3B-Instruct"
)

SELECTED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    2b|2B)
      SELECTED+=("2B")
      shift
      ;;
    4b|4B)
      SELECTED+=("4B")
      shift
      ;;
    32b|32B)
      SELECTED+=("32B")
      shift
      ;;
    30b-a3b|30B-A3B|30b-A3B|30B-a3b)
      SELECTED+=("30B-A3B")
      shift
      ;;
    *)
      echo "Unknown option or model size: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("2B" "4B" "32B" "30B-A3B")
fi

if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
else
  echo "Missing hf CLI. Install huggingface_hub or activate the vlmevalkit environment." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

# Force direct mirror access and prevent inherited proxy settings from being used.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${TARGET_DIR}/.hf_cache}"
export NO_PROXY="*"
export no_proxy="*"

echo "hf_bin=${HF_BIN}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "target_dir=${TARGET_DIR}"
echo "max_workers=${MAX_WORKERS}"
echo "models=${SELECTED[*]}"

for key in "${SELECTED[@]}"; do
  repo="${REPOS[$key]}"
  local_dir="${TARGET_DIR}/${LOCAL_NAMES[$key]}"

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

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
done

echo
echo "done"
