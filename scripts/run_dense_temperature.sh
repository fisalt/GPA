#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
"$python_bin" -m pytest -q tests/test_positive_cuda.py -k temperature
common=(--data-version 3 --sequence-balanced --steps 20000 --easy-curriculum --curriculum-span 1000 --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128)
"$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --balanced-bridge --context-controls --quadratic-features --temporal-qk-init --learn-read-scale --cuda-graph "${common[@]}" --output ../paired_temperature_v3b
"$python_bin" scripts/train_retrieval.py --attention transformer --transformer-rope --temporal-qk-init "${common[@]}" --output ../control_temporal_v3b
