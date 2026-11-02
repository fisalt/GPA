#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
python_bin="${GPA_PYTHON:-python}"
result_dir="${GPA_RUN_DIR:-retrieval_results}"
mkdir -p "$result_dir"
"$python_bin" -m pytest -q tests/test_triton.py::test_fp32_master_parameters_with_bf16_autocast
"$python_bin" scripts/train_retrieval.py --data-version 1 --steps 400 --output "$result_dir/gpa" --eval-samples 32
"$python_bin" scripts/train_retrieval.py --data-version 1 --attention transformer --steps 400 --output "$result_dir/transformer" --eval-samples 32
