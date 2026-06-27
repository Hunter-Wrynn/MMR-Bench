#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs
ts="$(date -u +%Y%m%d_%H%M%S)"

run_1b="mmrbench_controlled_InternVL3-1B-Instruct_parallel8gpu_${ts}"
run_2b="mmrbench_controlled_InternVL3-2B-Instruct_parallel8gpu_${ts}"
log_1b="${REPO_ROOT}/logs/${run_1b}.launcher.log"
log_2b="${REPO_ROOT}/logs/${run_2b}.launcher.log"
session_1b="mmr_internvl3_1b_8gpu_${ts}"
session_2b="mmr_internvl3_2b_8gpu_${ts}"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${session_1b}" \
    "cd '${REPO_ROOT}' && '${REPO_ROOT}/scripts/run_internvl3_1b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh' --run-name '${run_1b}' > '${log_1b}' 2>&1"
  tmux new-session -d -s "${session_2b}" \
    "cd '${REPO_ROOT}' && '${REPO_ROOT}/scripts/run_internvl3_2b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh' --run-name '${run_2b}' > '${log_2b}' 2>&1"
  pid_1b="$(tmux list-panes -t "${session_1b}" -F '#{pane_pid}' | head -n 1)"
  pid_2b="$(tmux list-panes -t "${session_2b}" -F '#{pane_pid}' | head -n 1)"
  launcher="tmux"
else
  setsid "${REPO_ROOT}/scripts/run_internvl3_1b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
    --run-name "${run_1b}" \
    > "${log_1b}" 2>&1 < /dev/null &
  pid_1b=$!
  setsid "${REPO_ROOT}/scripts/run_internvl3_2b_mmrbench_controlled_8gpu_judge_33_41_7_100.sh" \
    --run-name "${run_2b}" \
    > "${log_2b}" 2>&1 < /dev/null &
  pid_2b=$!
  launcher="setsid"
fi

cat <<EOF
launcher=${launcher}
launched_1b_pid=${pid_1b}
launched_1b_session=${session_1b}
launched_1b_run_name=${run_1b}
launched_1b_gpus=0,1,2,3,4,5,6,7
launched_1b_launcher_log=${log_1b}

launched_2b_pid=${pid_2b}
launched_2b_session=${session_2b}
launched_2b_run_name=${run_2b}
launched_2b_gpus=0,1,2,3,4,5,6,7
launched_2b_launcher_log=${log_2b}
EOF
