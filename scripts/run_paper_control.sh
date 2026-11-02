#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
  while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" scripts/train_retrieval.py --data-version 1 --attention transformer --steps 6000 --easy-curriculum --output ../control_run --eval-samples 128
