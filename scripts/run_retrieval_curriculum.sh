#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
result_dir="${GPA_RUN_DIR:?Set GPA_RUN_DIR}"
base_results="${GPA_BASE_RESULTS:?Set GPA_BASE_RESULTS to the initial 400-step run}"
mkdir -p "$result_dir"
"$python_bin" -m pytest -q tests/test_training_graph.py
"$python_bin" scripts/compare_efficiency.py --cuda-graph --output "$result_dir/efficiency_graph.json"
"$python_bin" scripts/analyze_retrieval.py "$base_results/gpa/checkpoint-400.pt" --output "$result_dir/diagnostics_before.json" --plot
"$python_bin" scripts/train_retrieval.py --data-version 1 --resume "$base_results/gpa/checkpoint-400.pt" --steps 2000 --easy-curriculum --cuda-graph --output "$result_dir/gpa" --eval-samples 64
"$python_bin" scripts/train_retrieval.py --data-version 1 --attention transformer --resume "$base_results/transformer/checkpoint-400.pt" --steps 2000 --easy-curriculum --output "$result_dir/transformer" --eval-samples 64
"$python_bin" scripts/analyze_retrieval.py "$result_dir/gpa/checkpoint-2000.pt" --output "$result_dir/diagnostics_after.json" --plot
