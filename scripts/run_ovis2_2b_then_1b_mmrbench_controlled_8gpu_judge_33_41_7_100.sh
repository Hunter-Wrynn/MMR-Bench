#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-mmrbench_controlled_ovis2_small_sequence_${TIMESTAMP}}"

echo "sequence_run_prefix=${RUN_PREFIX}"
echo "first=Ovis2-2B"
echo "second=Ovis2-1B"

"${REPO_ROOT}/scripts/run_ovis2_2b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-2B" \
  "$@"

"${REPO_ROOT}/scripts/run_ovis2_1b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-1B" \
  "$@"
