#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_IP="${MMR_JUDGE_IP:-33.41.7.100}"
BENCHMARKS="${MMR_BENCHMARKS:-MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only}"
RESUME_2B_WORK_DIR="${MMR_QWEN3VL_2B_RESUME_WORK_DIR:-${REPO_ROOT}/third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-2B-Instruct_20260613_121944}"
MODELS_AFTER="${MMR_QWEN3VL_RESUME_AFTER_MODELS:-4B,8B}"
RUN_2B=1
RUN_AFTER=1
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/resume_qwen3vl_2b_4b_8b_mmrbench_from_20260613_121944.sh [options]

Resume the interrupted Qwen3-VL-2B MMR-Bench run from:
  third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-2B-Instruct_20260613_121944

Then continue Qwen3-VL-4B and Qwen3-VL-8B on the same benchmark set.

Defaults:
  judge IP:   33.41.7.100
  2B resume:  reuse existing predictions and rank shards with --reuse --reuse-aux all
  after 2B:   run 4B,8B from fresh output dirs
  backend:    torchrun data parallel, same as Qwen2.5-VL-7B

Options handled by this wrapper:
  --judge-ip IP          Override judge IP.
  --2b-work-dir DIR      Override the interrupted 2B work-dir.
  --benchmarks LIST      Override benchmark list.
  --after-models LIST    Comma/space separated subset after 2B, e.g. 4B or 8B.
  --only-2b              Resume only 2B, do not run 4B/8B.
  --skip-2b              Skip 2B, run only --after-models.
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
    --2b-work-dir)
      RESUME_2B_WORK_DIR="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --after-models)
      MODELS_AFTER="$2"
      shift 2
      ;;
    --only-2b)
      RUN_AFTER=0
      shift
      ;;
    --skip-2b)
      RUN_2B=0
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

if [[ "${RUN_2B}" -eq 1 ]]; then
  if [[ ! -d "${RESUME_2B_WORK_DIR}" ]]; then
    echo "Missing 2B resume work-dir: ${RESUME_2B_WORK_DIR}" >&2
    exit 1
  fi

  echo "=== Resuming Qwen3-VL-2B-Instruct from ${RESUME_2B_WORK_DIR}; judge=${JUDGE_IP} ==="
  "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
    --config-file "${REPO_ROOT}/configs/qwen3vl_2b_mmrbench_vllm_controlled.env" \
    --judge-ip "${JUDGE_IP}" \
    --benchmarks "${BENCHMARKS}" \
    --work-dir "${RESUME_2B_WORK_DIR}" \
    --run-name "mmrbench_resume_Qwen3-VL-2B-Instruct_$(date -u +%Y%m%d_%H%M%S)" \
    --reuse \
    --reuse-aux all \
    "${EXTRA_ARGS[@]}"
fi

if [[ "${RUN_AFTER}" -eq 1 ]]; then
  MODEL_ITEMS="${MODELS_AFTER//,/ }"
  read -r -a MODELS <<< "${MODEL_ITEMS}"
  for size in "${MODELS[@]}"; do
    case "${size}" in
      4B|4b)
        CONFIG_FILE="${REPO_ROOT}/configs/qwen3vl_4b_mmrbench_vllm_controlled.env"
        ;;
      8B|8b)
        CONFIG_FILE="${REPO_ROOT}/configs/qwen3vl_8b_mmrbench_vllm_controlled.env"
        ;;
      "")
        continue
        ;;
      *)
        echo "Unknown after-model size: ${size}. Expected one of: 4B, 8B." >&2
        exit 2
        ;;
    esac

    echo "=== Running Qwen3-VL-${size}-Instruct on MMR-Bench; judge=${JUDGE_IP} ==="
    "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
      --config-file "${CONFIG_FILE}" \
      --judge-ip "${JUDGE_IP}" \
      --benchmarks "${BENCHMARKS}" \
      "${EXTRA_ARGS[@]}"
  done
fi

echo "done"
