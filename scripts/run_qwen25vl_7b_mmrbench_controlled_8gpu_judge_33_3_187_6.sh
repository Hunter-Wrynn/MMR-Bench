#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_qwen25vl_7b_mmrbench_controlled_8gpu.sh" \
  --judge-ip 33.3.187.6 \
  --benchmarks MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only \
  "$@"
