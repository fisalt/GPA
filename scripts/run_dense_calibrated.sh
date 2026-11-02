#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
if "$python_bin" - <<'PY'
import json,sys
from pathlib import Path
paths=['../paired_product_v3/results.json','../paired_positive_v3/results.json']
b=json.loads(Path('../control_rope_v3/results.json').read_text())['evaluations']['256']['tasks']
score=lambda x,k:x[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
good=False
for p in paths:
 if Path(p).is_file():
  a=json.loads(Path(p).read_text())['evaluations']['256']['tasks']
  good |= all(score(b,k)-score(a,k)<=.05 for k in a)
print('Validation gap requires calibrated GDN output:',not good,flush=True)
sys.exit(1 if good else 0)
PY
then
 "$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --full-controls --slot-feature tanh --gated-output --cuda-graph --data-version 3 --sequence-balanced --curriculum-span 1000 --steps 20000 --easy-curriculum --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128 --output ../paired_calibrated_v3
fi
