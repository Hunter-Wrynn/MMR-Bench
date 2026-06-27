#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/root/storage/mahaoxuan.mhx/model}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"
MAX_WORKERS="${MAX_WORKERS:-8}"
STALL_SECONDS="${STALL_SECONDS:-180}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
FINALIZE_SECONDS="${FINALIZE_SECONDS:-30}"
REVISION=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/download_internvl_under_40b_smart_hf_mirror.sh [options]

Download one evaluation-ready checkpoint per InternVL3 / InternVL3.5 size below
40B through hf-mirror. Each repo is first downloaded with hf_transfer for speed.
If the hf_transfer process stalls, the script kills it and reruns the same repo
with the regular Hugging Face downloader to finalize or continue.

Options:
  --target-dir DIR      Download root. Default: /root/storage/mahaoxuan.mhx/model
  --max-workers N       Parallel file workers per model. Default: 8
  --stall-seconds N     No-growth timeout before fallback. Default: 180
  --check-interval N    Monitor interval in seconds. Default: 30
  --finalize-seconds N  Max seconds for regular downloader finalize pass. Default: 30
  --revision REV        Optional Hugging Face revision.
  --dry-run             Print planned downloads only.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir) TARGET_DIR="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --stall-seconds) STALL_SECONDS="$2"; shift 2 ;;
    --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
    --finalize-seconds) FINALIZE_SECONDS="$2"; shift 2 ;;
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
export NO_PROXY="*"
export no_proxy="*"

repos=(
  OpenGVLab/InternVL3-1B-Instruct
  OpenGVLab/InternVL3-2B-Instruct
  OpenGVLab/InternVL3-8B-Instruct
  OpenGVLab/InternVL3-9B-Instruct
  OpenGVLab/InternVL3-14B-Instruct
  OpenGVLab/InternVL3-38B-Instruct
  OpenGVLab/InternVL3_5-1B-Instruct
  OpenGVLab/InternVL3_5-2B-Instruct
  OpenGVLab/InternVL3_5-4B-Instruct
  OpenGVLab/InternVL3_5-8B-Instruct
  OpenGVLab/InternVL3_5-14B-Instruct
  OpenGVLab/InternVL3_5-38B-Instruct
  OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview
  OpenGVLab/InternVL3_5-30B-A3B
)

build_cmd() {
  local repo="$1"
  local local_dir="$2"
  cmd=(
    "${HF_BIN}" download "${repo}"
    --repo-type model
    --local-dir "${local_dir}"
    --max-workers "${MAX_WORKERS}"
  )
  if [[ -n "${REVISION}" ]]; then
    cmd+=(--revision "${REVISION}")
  fi
}

incomplete_total_bytes() {
  local local_dir="$1"
  find "${local_dir}/.cache/huggingface/download" -name '*.incomplete' -printf '%s\n' 2>/dev/null \
    | awk '{s += $1} END {print s + 0}'
}

run_regular_finalize() {
  local repo="$1"
  local local_dir="$2"
  build_cmd "${repo}" "${local_dir}"
  echo "fallback=regular_hf_finalize"
  printf 'finalize_command='
  printf ' %q' "${cmd[@]}"
  printf '\n'
  set +e
  HF_HUB_ENABLE_HF_TRANSFER=0 timeout "${FINALIZE_SECONDS}s" "${cmd[@]}"
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    echo "finalize_status=complete"
    return 0
  fi
  if [[ "${status}" -eq 124 ]]; then
    echo "finalize_status=timeout_continue_fast_path"
    return 124
  fi
  echo "finalize_status=failed status=${status}"
  return "${status}"
}

echo "hf_bin=${HF_BIN}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_hub_disable_xet=${HF_HUB_DISABLE_XET}"
echo "target_dir=${TARGET_DIR}"
echo "max_workers=${MAX_WORKERS}"
echo "stall_seconds=${STALL_SECONDS}"
echo "check_interval=${CHECK_INTERVAL}"
echo "finalize_seconds=${FINALIZE_SECONDS}"
echo "num_repos=${#repos[@]}"

for repo in "${repos[@]}"; do
  local_name="${repo#OpenGVLab/}"
  local_dir="${TARGET_DIR}/${local_name}"
  mkdir -p "${local_dir}"
  build_cmd "${repo}" "${local_dir}"

  echo
  echo "repo=${repo}"
  echo "local_dir=${local_dir}"
  printf 'command='
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi

  while true; do
    HF_HUB_ENABLE_HF_TRANSFER=1 "${cmd[@]}" &
    child=$!
    last_total=-1
    unchanged=0
    stalled=0

    while kill -0 "${child}" 2>/dev/null; do
      sleep "${CHECK_INTERVAL}"
      current_total="$(incomplete_total_bytes "${local_dir}")"
      echo "monitor repo=${repo} pid=${child} incomplete_bytes=${current_total} unchanged_seconds=${unchanged}"
      if [[ "${current_total}" == "${last_total}" ]]; then
        unchanged=$((unchanged + CHECK_INTERVAL))
      else
        last_total="${current_total}"
        unchanged=0
      fi
      if [[ "${current_total}" -gt 0 && "${unchanged}" -ge "${STALL_SECONDS}" ]]; then
        echo "stall_detected repo=${repo} incomplete_bytes=${current_total}; finalize_then_resume_fast_path"
        kill "${child}" 2>/dev/null || true
        wait "${child}" 2>/dev/null || true
        stalled=1
        break
      fi
    done

    if [[ "${stalled}" -eq 1 ]]; then
      set +e
      run_regular_finalize "${repo}" "${local_dir}"
      finalize_status=$?
      set -e
      if [[ "${finalize_status}" -eq 0 ]]; then
        break
      fi
      if [[ "${finalize_status}" -eq 124 ]]; then
        continue
      fi
      exit "${finalize_status}"
    fi

    if wait "${child}"; then
      break
    fi

    echo "hf_transfer_failed repo=${repo}; finalize_then_resume_fast_path"
    set +e
    run_regular_finalize "${repo}" "${local_dir}"
    finalize_status=$?
    set -e
    if [[ "${finalize_status}" -eq 0 ]]; then
      break
    fi
    if [[ "${finalize_status}" -eq 124 ]]; then
      continue
    fi
    exit "${finalize_status}"
  done
done

echo
echo "done"
