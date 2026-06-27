#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_one_by_one.sh" \
  --config-file "${REPO_ROOT}/configs/gemma4_12b_mmrbench_vllm.env" \
  --judge-ip 33.41.7.100 \
  "$@"
