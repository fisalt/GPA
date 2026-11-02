#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q -s tests/test_staged.py tests/test_packed.py --tb=short
"$python_bin" scripts/benchmark_lm.py ../paired_temporal_v3/checkpoint-20000.pt --control ../control_temporal_v3b/checkpoint-20000.pt --lengths 256,1024 --iterations 10 --output docs/results/v3/efficiency_lm_staged.json
