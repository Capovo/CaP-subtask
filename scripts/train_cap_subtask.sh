#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/activate_local.sh"
export NPROC="${NPROC:-2}"
exec bash "$ROOT/scripts/train_robotwin.sh" cap_subtask_robotwin paths=local "$@"
