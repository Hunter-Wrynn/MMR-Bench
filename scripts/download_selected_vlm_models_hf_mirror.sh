#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/root/storage/mahaoxuan.mhx/model}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"
MAX_WORKERS="${MAX_WORKERS:-8}"
REVISION=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_selected_vlm_models_hf_mirror.sh [options] [Gemma4-12B|Gemma3-27B|InternVL35-38B ...]

Sequentially download selected VLM checkpoints through hf-mirror without proxies.
Default models:
  Gemma4-12B, InternVL35-38B

Options:
  --target-dir DIR      Download root. Default: /root/storage/mahaoxuan.mhx/model
  --max-workers N       Parallel file workers for one model download. Default: 8
  --revision REV        Optional Hugging Face revision.
  --dry-run             Print planned downloads only.
  -h, --help            Show this help.

Notes:
  google/gemma-3-27b-it is manual gated on Hugging Face. Downloading it requires
  a token/account that has accepted the Gemma license.
EOF
}

declare -A REPOS=(
  ["Gemma4-12B"]="google/gemma-4-12B-it"
  ["Gemma3-27B"]="google/gemma-3-27b-it"
  ["InternVL35-38B"]="OpenGVLab/InternVL3_5-38B-Instruct"
)

declare -A LOCAL_NAMES=(
  ["Gemma4-12B"]="Gemma-4-12B-it"
  ["Gemma3-27B"]="Gemma-3-27B-it"
  ["InternVL35-38B"]="InternVL3_5-38B-Instruct"
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
    Gemma4-12B|gemma4-12b|Gemma-4-12B|gemma-4-12b)
      SELECTED+=("Gemma4-12B")
      shift
      ;;
    Gemma3-27B|gemma3-27b|Gemma-3-27B|gemma-3-27b)
      SELECTED+=("Gemma3-27B")
      shift
      ;;
    InternVL35-38B|internvl35-38b|InternVL3_5-38B|internvl3_5-38b)
      SELECTED+=("InternVL35-38B")
      shift
      ;;
    *)
      echo "Unknown option or model key: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("Gemma4-12B" "InternVL35-38B")
fi

if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
else
  echo "Missing hf CLI. Activate an environment with huggingface_hub installed." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

# Force direct mirror access and prevent inherited proxy settings from being used.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${TARGET_DIR}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export NO_PROXY="*"
export no_proxy="*"

echo "hf_bin=${HF_BIN}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_hub_disable_xet=${HF_HUB_DISABLE_XET}"
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
    mkdir -p "${local_dir}"
    "${cmd[@]}"
  fi
done

echo
echo "done"
