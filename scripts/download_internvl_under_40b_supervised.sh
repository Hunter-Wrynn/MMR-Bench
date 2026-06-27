#!/usr/bin/env bash
set -euo pipefail

MAX_RESTARTS="${MAX_RESTARTS:-200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"

args=("$@")
restart=0

while true; do
  echo
  echo "supervisor_start restart=${restart} time=$(date -Is)"
  set +e
  bash scripts/download_internvl_under_40b_smart_hf_mirror.sh "${args[@]}"
  status=$?
  set -e
  echo "supervisor_child_exit status=${status} time=$(date -Is)"

  if [[ "${status}" -eq 0 ]]; then
    echo "supervisor_done"
    exit 0
  fi

  restart=$((restart + 1))
  if [[ "${restart}" -gt "${MAX_RESTARTS}" ]]; then
    echo "supervisor_failed max_restarts=${MAX_RESTARTS}" >&2
    exit "${status}"
  fi
  sleep "${SLEEP_SECONDS}"
done
