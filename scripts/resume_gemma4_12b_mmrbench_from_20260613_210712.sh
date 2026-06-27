#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_IP="${MMR_JUDGE_IP:-33.41.7.100}"
BENCHMARKS="${MMR_BENCHMARKS:-MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only}"
RESUME_WORK_DIR="${MMR_GEMMA4_12B_RESUME_WORK_DIR:-${REPO_ROOT}/third_party/VLMEvalKit/outputs/mmrbench_controlled_Gemma4-12B-it_20260613_210712}"
RUN_NAME="${MMR_RUN_NAME:-mmrbench_resume_Gemma4-12B-it_$(date -u +%Y%m%d_%H%M%S)}"
ONLY_REMAINING=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_gemma4_12b_mmrbench_from_20260613_210712.sh [options]

Resume the interrupted Gemma4-12B-it MMR-Bench run from:
  third_party/VLMEvalKit/outputs/mmrbench_controlled_Gemma4-12B-it_20260613_210712

Defaults:
  judge IP:   33.41.7.100
  benchmarks: standard 7 MMR-Bench benchmarks
  resume:     --reuse --reuse-aux all
  backend:    torchrun data parallel over 8 shards, not vLLM

The default full 7-benchmark resume is intentional: completed benchmark outputs
are reused, interrupted MathVision shards are reused when possible, MathVerse is
run, and the final standard-7 merge can run after all scores exist.

Options handled by this wrapper:
  --judge-ip IP          Override judge IP.
  --work-dir DIR         Override the resume work-dir.
  --benchmarks LIST      Override benchmark list.
  --only-remaining       Resume only MathVision and MathVerse_MINI_Vision_Only.
  --only-math            Alias for --only-remaining.
  --run-name NAME        Override generated run name.
  -h, --help             Show this help.

All other options are forwarded to scripts/run_mmrbench_vlmeval_controlled_8gpu.sh,
for example: --dry-run, --gpus, --nproc, --no-merge, --mode infer, --master-port.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-ip)
      JUDGE_IP="$2"
      shift 2
      ;;
    --work-dir)
      RESUME_WORK_DIR="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --only-remaining|--only-math)
      ONLY_REMAINING=1
      shift
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${ONLY_REMAINING}" -eq 1 ]]; then
  BENCHMARKS="MathVision,MathVerse_MINI_Vision_Only"
  EXTRA_ARGS+=("--no-merge")
fi

if [[ ! -d "${RESUME_WORK_DIR}" ]]; then
  echo "Missing resume work-dir: ${RESUME_WORK_DIR}" >&2
  exit 1
fi

echo "=== Resuming Gemma4-12B-it from ${RESUME_WORK_DIR}; judge=${JUDGE_IP} ==="
"${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
  --config-file "${REPO_ROOT}/configs/gemma4_12b_mmrbench_controlled.env" \
  --judge-ip "${JUDGE_IP}" \
  --benchmarks "${BENCHMARKS}" \
  --work-dir "${RESUME_WORK_DIR}" \
  --run-name "${RUN_NAME}" \
  --reuse \
  --reuse-aux all \
  "${EXTRA_ARGS[@]}"
