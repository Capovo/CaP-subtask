#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/activate_local.sh"
export CAP_SUBTASK_PLANNER_CONFIG="${CAP_SUBTASK_PLANNER_CONFIG:-$ROOT/configs/subtask/planner.yaml}"
exec python "$ROOT/scripts/eval_robotwin_release.py" "${EXECUTION_MODE:-wam}" \
  --config cap_subtask --gpus 0 --vis-gpu "${VIS_GPU:-1}" "$@" --launch-args --extra paths=local
