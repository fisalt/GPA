#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
while tmux has-session -t gpa_fullctrl_20261102 2>/dev/null; do sleep 10; done
"$python_bin" -m pytest -q tests/test_paper_dense.py
"$python_bin" scripts/train_retrieval.py --data-version 1 --memory-update paper_dense --paper-transport --full-controls --slot-decay --curriculum-span 1000 --steps 6000 --easy-curriculum --cuda-graph --output ../paper_slot --eval-samples 128
