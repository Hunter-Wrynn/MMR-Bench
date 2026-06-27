#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${REPO_ROOT}/scripts/run_mmrbench_vlmeval_controlled_8gpu.sh" \
  --config-file "${REPO_ROOT}/configs/internvl35_30b_a3b_mmrbench_controlled.env" \
  "$@"
