#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-mmrbench_controlled_ovis25_sequence_${TIMESTAMP}}"

echo "sequence_run_prefix=${RUN_PREFIX}"
echo "first=Ovis2.5-9B"
echo "second=Ovis2.5-2B"

"${REPO_ROOT}/scripts/run_ovis25_9b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2.5-9B" \
  "$@"

"${REPO_ROOT}/scripts/run_ovis25_2b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2.5-2B" \
  "$@"
