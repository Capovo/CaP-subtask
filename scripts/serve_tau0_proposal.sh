#!/usr/bin/env bash
# Run in a SEPARATE Transformers 5 environment; do not alter AgenticWAM's env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROPOSAL_PYTHON="${PROPOSAL_PYTHON:-$ROOT/.venv-proposal/bin/python}"
if [[ ! -x "$PROPOSAL_PYTHON" ]]; then
  echo "Missing Proposal environment: $PROPOSAL_PYTHON. See README.txt." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROPOSAL_PYTHON" -m tau0_proposal.serve \
  --model "${TAU0_PROPOSAL_WEIGHTS:?Set TAU0_PROPOSAL_WEIGHTS to the full Proposal checkpoint}" \
  --device "${PROPOSAL_DEVICE:-cuda:1}" --host 127.0.0.1 --port "${PROPOSAL_PORT:-10089}" "$@"
