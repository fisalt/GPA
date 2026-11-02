#!/usr/bin/env bash
# Fresh pure-CE reproduction; no retrieval-trained initialization or oracle mask.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_bin="${GPA_PYTHON:-python}"
output="${GPA_OUTPUT:-runs/selected_v3}"
# triton reproduces the reported training backend. staged is an opt-in faster
# schedule of the same lagged model, with a different BF16 reduction order.
execution="${GPA_EXECUTION:-triton}"
"$python_bin" scripts/train_retrieval.py --execution "$execution" \
  --rsa-update --aligned-query --orthogonal-centers --slots 32,32 \
  --paper-product-read --balanced-bridge --context-controls \
  --quadratic-features --temporal-qk-init --cuda-graph \
  --data-version 3 --sequence-balanced --steps 20000 \
  --easy-curriculum --curriculum-span 1000 --cosine \
  --warmup-steps 100 --min-lr .0002 --eval-samples 128 --output "$output"
# This seed was already used in the published bundle; new architecture selection
# requires a fresh test seed rather than repeated tuning on this test stream.
"$python_bin" scripts/eval_retrieval.py "$output/checkpoint-20000.pt" \
  --output "${output}_test" --seed 920000 --samples 256
