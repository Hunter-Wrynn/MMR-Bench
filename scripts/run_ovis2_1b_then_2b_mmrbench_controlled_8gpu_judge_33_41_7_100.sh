#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_PREFIX="${RUN_PREFIX:-mmrbench_controlled_ovis2_1b_then_2b_${TIMESTAMP}}"

cd "${REPO_ROOT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "sequence_run_prefix=${RUN_PREFIX}"
echo "first=Ovis2-1B"
echo "second=Ovis2-2B"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash "${REPO_ROOT}/scripts/run_ovis2_1b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-1B" \
  "$@"

python "${REPO_ROOT}/scripts/update_summary.py" || true

bash "${REPO_ROOT}/scripts/run_ovis2_2b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
  --run-name "${RUN_PREFIX}_Ovis2-2B" \
  "$@"

python "${REPO_ROOT}/scripts/update_summary.py" || true

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
