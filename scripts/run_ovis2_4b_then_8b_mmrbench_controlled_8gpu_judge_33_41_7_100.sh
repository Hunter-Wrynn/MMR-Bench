#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-mmrbench_controlled_ovis2_sequence_${TIMESTAMP}}"

echo "sequence_run_prefix=${RUN_PREFIX}"
echo "first=Ovis2-4B"
echo "second=Ovis2-8B"

"${REPO_ROOT}/scripts/run_ovis2_4b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-4B" \
  "$@"

"${REPO_ROOT}/scripts/run_ovis2_8b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-8B" \
  "$@"
