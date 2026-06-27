#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
  --config-file "${REPO_ROOT}/configs/internvl3_8b_mmrbench_controlled.env" \
  --judge-ip 33.41.7.100 \
  --gpus 0,1,2,3,4,5,6,7 \
  --nproc 8 \
  --master-port 29621 \
  "$@"
