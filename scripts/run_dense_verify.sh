#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q tests
"$python_bin" scripts/select_dense.py
"$python_bin" scripts/compare_efficiency.py --levels 2 --slots 32 --rsa-update --paper-product-read --balanced-bridge --relation-marginal --lengths 256,1024 --iterations 10 --cuda-graph --output docs/results/v3/efficiency_marginal.json
"$python_bin" scripts/report_dense.py --root ..
