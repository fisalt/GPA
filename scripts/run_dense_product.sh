#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q tests/test_packed.py tests/test_triton.py
common=(--data-version 3 --sequence-balanced --curriculum-span 1000 --steps 20000 --easy-curriculum --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128)
"$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --cuda-graph "${common[@]}" --output ../paired_product_v3
"$python_bin" scripts/train_retrieval.py --attention transformer --transformer-rope "${common[@]}" --output ../control_rope_v3
