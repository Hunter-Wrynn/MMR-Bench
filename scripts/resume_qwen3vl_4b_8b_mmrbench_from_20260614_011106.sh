#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_IP="${MMR_JUDGE_IP:-33.41.7.100}"
BENCHMARKS="${MMR_BENCHMARKS:-MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only}"
RESUME_4B_WORK_DIR="${MMR_QWEN3VL_4B_RESUME_WORK_DIR:-${REPO_ROOT}/third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-4B-Instruct_20260614_011106}"
RESUME_8B_WORK_DIR="${MMR_QWEN3VL_8B_RESUME_WORK_DIR:-${REPO_ROOT}/third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-8B-Instruct_20260613_115237}"
RUN_4B=1
RUN_8B=1
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_qwen3vl_4b_8b_mmrbench_from_20260614_011106.sh [options]

Resume Qwen3-VL-4B-Instruct from:
  third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-4B-Instruct_20260614_011106

Then run/resume Qwen3-VL-8B-Instruct from:
  third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-8B-Instruct_20260613_115237

Defaults:
  judge IP:   33.41.7.100
  benchmarks: standard 7 MMR-Bench benchmarks
  resume:     --reuse --reuse-aux all
  backend:    torchrun data parallel over 8 shards, not vLLM

Options handled by this wrapper:
  --judge-ip IP          Override judge IP.
  --4b-work-dir DIR      Override the 4B resume work-dir.
  --8b-work-dir DIR      Override the 8B resume work-dir.
  --benchmarks LIST      Override benchmark list.
  --only-4b              Resume only 4B.
  --only-8b              Run/resume only 8B.
  --skip-4b              Skip 4B.
  --skip-8b              Skip 8B.
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
    --4b-work-dir)
      RESUME_4B_WORK_DIR="$2"
      shift 2
      ;;
    --8b-work-dir)
      RESUME_8B_WORK_DIR="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --only-4b)
      RUN_4B=1
      RUN_8B=0
      shift
      ;;
    --only-8b)
      RUN_4B=0
      RUN_8B=1
      shift
      ;;
    --skip-4b)
      RUN_4B=0
      shift
      ;;
    --skip-8b)
      RUN_8B=0
      shift
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

if [[ "${RUN_4B}" -eq 0 && "${RUN_8B}" -eq 0 ]]; then
  echo "Nothing to run: both 4B and 8B are disabled." >&2
  exit 2
fi

if [[ "${RUN_4B}" -eq 1 ]]; then
  if [[ ! -d "${RESUME_4B_WORK_DIR}" ]]; then
    echo "Missing 4B resume work-dir: ${RESUME_4B_WORK_DIR}" >&2
    exit 1
  fi

  echo "=== Resuming Qwen3-VL-4B-Instruct from ${RESUME_4B_WORK_DIR}; judge=${JUDGE_IP} ==="
  "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
    --config-file "${REPO_ROOT}/configs/qwen3vl_4b_mmrbench_vllm_controlled.env" \
    --judge-ip "${JUDGE_IP}" \
    --benchmarks "${BENCHMARKS}" \
    --work-dir "${RESUME_4B_WORK_DIR}" \
    --run-name "mmrbench_resume_Qwen3-VL-4B-Instruct_$(date -u +%Y%m%d_%H%M%S)" \
    --reuse \
    --reuse-aux all \
    "${EXTRA_ARGS[@]}"
fi

if [[ "${RUN_8B}" -eq 1 ]]; then
  mkdir -p "${RESUME_8B_WORK_DIR}"

  echo "=== Running/resuming Qwen3-VL-8B-Instruct from ${RESUME_8B_WORK_DIR}; judge=${JUDGE_IP} ==="
  "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
    --config-file "${REPO_ROOT}/configs/qwen3vl_8b_mmrbench_vllm_controlled.env" \
    --judge-ip "${JUDGE_IP}" \
    --benchmarks "${BENCHMARKS}" \
    --work-dir "${RESUME_8B_WORK_DIR}" \
    --run-name "mmrbench_resume_Qwen3-VL-8B-Instruct_$(date -u +%Y%m%d_%H%M%S)" \
    --reuse \
    --reuse-aux all \
    "${EXTRA_ARGS[@]}"
fi

echo "done"
