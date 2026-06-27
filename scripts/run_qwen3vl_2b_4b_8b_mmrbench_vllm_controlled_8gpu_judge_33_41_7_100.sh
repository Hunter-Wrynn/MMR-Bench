#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_IP="${MMR_JUDGE_IP:-33.41.7.100}"
MODELS_LIST="${MMR_QWEN3VL_MODELS:-2B,4B,8B}"
BENCHMARKS="${MMR_BENCHMARKS:-MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_qwen3vl_2b_4b_8b_mmrbench_vllm_controlled_8gpu_judge_33_41_7_100.sh [options]

Runs Qwen3-VL-2B/4B/8B-Instruct on the selected MMR-Bench benchmarks with
inference + evaluation, then merges standard 7-benchmark results into data/MMR-Bench.csv.

Defaults:
  judge IP:   33.41.7.100
  models:     2B,4B,8B
  benchmarks: MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only
  GPUs:       0,1,2,3,4,5,6,7
  backend:    torchrun data parallel, same as Qwen2.5-VL-7B

Options handled by this wrapper:
  --judge-ip IP        Override judge IP.
  --models LIST        Comma/space separated subset, e.g. 2B,8B.
  --benchmarks LIST    Override benchmark list.
  -h, --help           Show this help.

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
    --models)
      MODELS_LIST="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="$2"
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

MODEL_ITEMS="${MODELS_LIST//,/ }"
read -r -a MODELS <<< "${MODEL_ITEMS}"

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models selected. Use --models 2B,4B,8B or set MMR_QWEN3VL_MODELS." >&2
  exit 2
fi

for size in "${MODELS[@]}"; do
  case "${size}" in
    2B|2b)
      CONFIG_FILE="${REPO_ROOT}/configs/qwen3vl_2b_mmrbench_vllm_controlled.env"
      ;;
    4B|4b)
      CONFIG_FILE="${REPO_ROOT}/configs/qwen3vl_4b_mmrbench_vllm_controlled.env"
      ;;
    8B|8b)
      CONFIG_FILE="${REPO_ROOT}/configs/qwen3vl_8b_mmrbench_vllm_controlled.env"
      ;;
    *)
      echo "Unknown Qwen3-VL size: ${size}. Expected one of: 2B, 4B, 8B." >&2
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

echo "done"
