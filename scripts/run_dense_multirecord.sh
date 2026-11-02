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
a=json.loads(Path('../paired_balanced_v3/results.json').read_text())['evaluations']['256']['tasks']
b=json.loads(Path('../control_rope_v3/results.json').read_text())['evaluations']['256']['tasks']
score=lambda x,k:x[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
sys.exit(0 if any(score(b,k)-score(a,k)>.05 for k in a) else 1)
PY
then
 common=(--data-version 3 --sequence-balanced --steps 20000 --cosine --warmup-steps 100 --min-lr .0002 --eval-samples 128)
 # No single-record phase: first100 steps T128/4, then T256/8. Both fresh.
 "$python_bin" scripts/train_retrieval.py --rsa-update --aligned-query --orthogonal-centers --slots 32,32 --paper-product-read --balanced-bridge --cuda-graph "${common[@]}" --output ../paired_multirecord_v3
 "$python_bin" scripts/train_retrieval.py --attention transformer --transformer-rope "${common[@]}" --output ../control_multirecord_v3
fi
"$python_bin" scripts/report_dense.py --root ..
