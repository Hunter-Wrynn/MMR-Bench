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
  scripts/download_p0_vlm_models_hf_mirror.sh [options] [model_key ...]

Download P0 MLLM checkpoints through hf-mirror without proxies.

Default model keys:
  Ovis2.5-2B Ovis2.5-9B GLM-4.1V-9B-Thinking GLM-4.5V
  Kimi-VL-A3B-Instruct Kimi-VL-A3B-Thinking-2506 MiniCPM-V-4_5

Extra optional keys:
  Ovis2-34B MiniCPM-o-4_5

Options:
  --target-dir DIR      Download root. Default: /root/storage/mahaoxuan.mhx/model
  --max-workers N       Parallel file workers for one model download. Default: 8
  --revision REV        Optional Hugging Face revision.
  --dry-run             Print planned downloads only.
  -h, --help            Show this help.
EOF
}

declare -A REPOS=(
  ["Ovis2.5-2B"]="AIDC-AI/Ovis2.5-2B"
  ["Ovis2.5-9B"]="AIDC-AI/Ovis2.5-9B"
  ["Ovis2-34B"]="AIDC-AI/Ovis2-34B"
  ["GLM-4.1V-9B-Thinking"]="THUDM/GLM-4.1V-9B-Thinking"
  ["GLM-4.5V"]="THUDM/GLM-4.5V"
  ["Kimi-VL-A3B-Instruct"]="moonshotai/Kimi-VL-A3B-Instruct"
  ["Kimi-VL-A3B-Thinking-2506"]="moonshotai/Kimi-VL-A3B-Thinking-2506"
  ["MiniCPM-V-4_5"]="openbmb/MiniCPM-V-4_5"
  ["MiniCPM-o-4_5"]="openbmb/MiniCPM-o-4_5"
)

declare -A LOCAL_NAMES=(
  ["Ovis2.5-2B"]="Ovis2.5-2B"
  ["Ovis2.5-9B"]="Ovis2.5-9B"
  ["Ovis2-34B"]="Ovis2-34B"
  ["GLM-4.1V-9B-Thinking"]="GLM-4.1V-9B-Thinking"
  ["GLM-4.5V"]="GLM-4.5V"
  ["Kimi-VL-A3B-Instruct"]="Kimi-VL-A3B-Instruct"
  ["Kimi-VL-A3B-Thinking-2506"]="Kimi-VL-A3B-Thinking-2506"
  ["MiniCPM-V-4_5"]="MiniCPM-V-4_5"
  ["MiniCPM-o-4_5"]="MiniCPM-o-4_5"
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
    *)
      key="$1"
      if [[ -z "${REPOS[$key]+set}" ]]; then
        echo "Unknown model key: ${key}" >&2
        usage >&2
        exit 2
      fi
      SELECTED+=("${key}")
      shift
      ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=(
    "Ovis2.5-2B"
    "Ovis2.5-9B"
    "GLM-4.1V-9B-Thinking"
    "GLM-4.5V"
    "Kimi-VL-A3B-Instruct"
    "Kimi-VL-A3B-Thinking-2506"
    "MiniCPM-V-4_5"
  )
fi

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
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NO_PROXY="*"
export no_proxy="*"

echo "hf_bin=${HF_BIN}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_hub_disable_xet=${HF_HUB_DISABLE_XET}"
echo "hf_hub_enable_hf_transfer=${HF_HUB_ENABLE_HF_TRANSFER}"
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
