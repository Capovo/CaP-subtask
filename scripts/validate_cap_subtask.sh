#!/usr/bin/env bash
# CPU tests and command planning only; no model downloads, GPU jobs or robot motion.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/activate_local.sh"
python -m unittest discover -s tests -v
python -m compileall -q src scripts robotwin
python scripts/launch_train.py --task cap_subtask_robotwin --nproc 2 --dry-run paths=local
bash scripts/eval_cap_subtask.sh \
  --ckpt /home/yilianli/project/models/agentic-wam-robotwin-release/hybrid.pt \
  --tasks click_alarmclock --plan smoke --protocols clean --episodes 1 --shards 1 --dry-run
