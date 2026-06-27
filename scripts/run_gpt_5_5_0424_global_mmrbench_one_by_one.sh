#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_one_by_one.sh" \
  --config-file "${REPO_ROOT}/configs/gpt_5_5_0424_global_mmrbench_api.env" \
  --benchmarks MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only \
  --no-use-vllm \
  --nproc 1 \
  --sleep-between 0 \
  --no-merge \
  "$@"
