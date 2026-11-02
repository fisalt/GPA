#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" scripts/train_retrieval.py --memory-update paper_dense --paper-transport --full-controls --slot-decay --strong-relations --data-version 2 --sequence-balanced --curriculum-span 1000 --steps 6000 --easy-curriculum --cuda-graph --output ../paper_final --eval-samples 128
"$python_bin" scripts/train_retrieval.py --attention transformer --data-version 2 --sequence-balanced --curriculum-span 1000 --steps 6000 --easy-curriculum --output ../control_final --eval-samples 128
"$python_bin" scripts/train_retrieval.py --memory-update paper_dense --paper-transport --full-controls --slot-decay --strong-relations --data-version 2 --curriculum-span 1000 --eval-only --resume ../paper_final/checkpoint-6000.pt --output ../heldout_paper --eval-seed 910000 --eval-samples 256
"$python_bin" scripts/train_retrieval.py --attention transformer --data-version 2 --curriculum-span 1000 --eval-only --resume ../control_final/checkpoint-6000.pt --output ../heldout_control --eval-seed 910000 --eval-samples 256
