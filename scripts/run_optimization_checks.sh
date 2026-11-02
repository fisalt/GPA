#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
python_bin="${GPA_PYTHON:-python}"
"$python_bin" -m pytest -q -s tests/test_triton.py tests/test_packed.py
"$python_bin" scripts/compare_efficiency.py --output "${GPA_RESULTS:-efficiency.json}"
