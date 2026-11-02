#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
python_bin="${GPA_PYTHON:-python}"
"$python_bin" -m pytest -q -s tests/test_cuda.py
"$python_bin" examples/smoke.py --device cuda --backend fla --chunk-size 64 --steps 10
"$python_bin" scripts/benchmark.py --device cuda --backend fla --chunk-size 64 --length 256 --backward
