#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q -s tests/test_staged.py -k cuda --tb=short
common=(--data-version 3 --sequence-balanced --steps 20000 --easy-curriculum --curriculum-span 1000 --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128)
# Explicit capacity study: 2x64 slots, Dk64, Dv32; NOT an equal-state claim.
"$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 64,64 --key-dim 64 --paper-product-read --balanced-bridge --context-controls --quadratic-features --temporal-qk-init --execution staged --cuda-graph "${common[@]}" --output ../paired_memory64_v3b
# Similar parameter-count Transformer; the old width128 control is retained too.
"$python_bin" scripts/train_retrieval.py --attention transformer --transformer-rope --width 256 --layers 2 --temporal-qk-init "${common[@]}" --output ../control_param_v3b
