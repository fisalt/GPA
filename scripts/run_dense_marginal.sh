#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${GPA_FLA_ROOT:+:$GPA_FLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python_bin="${GPA_PYTHON:-python}"
if [[ -n "${GPA_WAIT_SESSION:-}" ]]; then
 while tmux has-session -t "$GPA_WAIT_SESSION" 2>/dev/null; do sleep 10; done
fi
"$python_bin" -m pytest -q tests/test_triton.py tests/test_packed.py
if "$python_bin" - <<'PY'
import json,sys
from pathlib import Path
p=Path('../paired_context_v3/results.json')
if p.is_file():
 a=json.loads(p.read_text())['evaluations']['256']['tasks']
 score=lambda k:a[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
 if all(score(k)>=.95 for k in a):sys.exit(1)
sys.exit(0)
PY
then
 "$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --balanced-bridge --relation-marginal --cuda-graph --data-version 3 --sequence-balanced --steps 20000 --easy-curriculum --curriculum-span 1000 --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128 --output ../paired_marginal_v3
fi
"$python_bin" scripts/report_dense.py --root ..
