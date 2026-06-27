#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
  --config-file "${REPO_ROOT}/configs/qwen3vl_32b_mmrbench_controlled.env" \
  --judge-ip 33.41.7.100 \
  --benchmarks MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only \
  "$@"
