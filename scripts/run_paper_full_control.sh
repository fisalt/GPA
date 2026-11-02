#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
while tmux has-session -t gpa_transport_20261102 2>/dev/null; do sleep 10; done
"$python_bin" scripts/train_retrieval.py --data-version 1 --memory-update paper_dense --paper-transport --full-controls --curriculum-span 1000 --steps 6000 --easy-curriculum --cuda-graph --output ../paper_fullcontrol --eval-samples 128
"$python_bin" scripts/train_retrieval.py --data-version 1 --attention transformer --curriculum-span 1000 --steps 6000 --easy-curriculum --output ../control_slow --eval-samples 128
