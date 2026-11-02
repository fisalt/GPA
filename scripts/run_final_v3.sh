#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
"$python_bin" -m pytest -q tests --tb=short
"$python_bin" scripts/record_runtime.py --output docs/results/v3/runtime_final.json
# Frozen phase1 winner. This is a backend regression, NOT selection on test data.
"$python_bin" scripts/eval_retrieval.py ../paired_temporal_v3/checkpoint-20000.pt --execution staged --output ../heldout_fast_v3 --seed 920000 --samples 256
"$python_bin" scripts/analyze_retrieval.py ../paired_temporal_v3/checkpoint-20000.pt --task uuid4 --plot --ablate-relation-read --output docs/results/v3/diagnostics_final.json
"$python_bin" scripts/benchmark_lm.py ../paired_temporal_v3/checkpoint-20000.pt --control ../control_temporal_v3b/checkpoint-20000.pt --lengths 4096 --variants triton_graph,staged_graph --iterations 10 --output docs/results/v3/efficiency_lm_4096.json
"$python_bin" scripts/report_dense.py --root ..
