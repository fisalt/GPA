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
p=Path('../paired_balanced_v3/results.json')
if not p.is_file():sys.exit(1)
a=json.loads(p.read_text())['evaluations']['256']['tasks']
b=json.loads(Path('../control_rope_v3/results.json').read_text())['evaluations']['256']['tasks']
score=lambda x,k:x[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
sys.exit(0 if any(score(a,k)<max(.95,score(b,k)-.05) for k in a) else 1)
PY
then
 "$python_bin" scripts/analyze_retrieval.py ../paired_balanced_v3/checkpoint-20000.pt --task uuid4 --data-version 3 --ablate-relation-read --plot --output docs/results/v3/diagnostics_balanced.json
 "$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --balanced-bridge --context-controls --cuda-graph --data-version 3 --sequence-balanced --steps 20000 --easy-curriculum --curriculum-span 1000 --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128 --output ../paired_context_v3
fi
"$python_bin" scripts/report_dense.py --root ..
