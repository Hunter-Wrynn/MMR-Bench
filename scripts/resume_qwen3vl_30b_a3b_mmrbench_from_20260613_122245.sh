#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGE_IP="${MMR_JUDGE_IP:-33.41.7.100}"
WORK_DIR="${MMR_QWEN3VL_30B_A3B_RESUME_WORK_DIR:-${REPO_ROOT}/third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-30B-A3B-Instruct_20260613_122245}"
BENCHMARKS="${MMR_BENCHMARKS:-MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only}"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
  --config-file "${REPO_ROOT}/configs/qwen3vl_30b_a3b_mmrbench_controlled.env" \
  --judge-ip "${JUDGE_IP}" \
  --benchmarks "${BENCHMARKS}" \
  --work-dir "${WORK_DIR}" \
  --run-name "mmrbench_resume_Qwen3-VL-30B-A3B-Instruct_$(date -u +%Y%m%d_%H%M%S)" \
  --reuse \
  --reuse-aux all \
  "$@"
