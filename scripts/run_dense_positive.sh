#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q tests/test_positive_cuda.py tests/test_packed.py
# Try a further structural change only if the first V3 candidate misses
# the validation target. This decision never reads the final held-out seed.
if "$python_bin" - <<'PY'
import json,sys
from pathlib import Path
a=json.loads(Path('../paired_product_v3/results.json').read_text())['evaluations']['256']['tasks']
b=json.loads(Path('../control_rope_v3/results.json').read_text())['evaluations']['256']['tasks']
score=lambda x,k:x[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
bad=any(score(b,k)-score(a,k)>.05 for k in a)
print('Validation gap requires nonlinear feature trial:',bad,flush=True)
sys.exit(0 if bad else 1)
PY
then
 "$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --full-controls --slot-feature softmax --cuda-graph --data-version 3 --sequence-balanced --curriculum-span 1000 --steps 20000 --easy-curriculum --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128 --output ../paired_positive_v3
fi
