"""Same-capacity generic convolution initialization, selected flags, fresh model."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    candidates={}
    for name in ('paired_quadratic_v3','paired_readgate_v3','paired_context_v3'):
        p=Path('..')/name/'results.json'
        if p.is_file():candidates[name]=json.loads(p.read_text())
    def score(run):
        return min(v['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
                   for k,v in run['evaluations']['256']['tasks'].items())
    scaled=Path('../paired_scaled_v3/results.json')
    if (scaled.is_file() and score(json.loads(scaled.read_text()))>=.95) or any(score(v)>=.95 for v in candidates.values()):
        print('Validation target reached; skip initialization trial',flush=True);return
    chosen=max(candidates,key=lambda k:score(candidates[k]))
    cfg=candidates[chosen]['config']
    features=['--'+key.replace('_','-') for key in ('context_controls','quadratic_features','contextual_output') if cfg.get(key)]
    print(json.dumps({'architecture_from_validation':chosen,'fresh_initialization':True,'features':features}),flush=True)
    subprocess.run([sys.executable,'scripts/train_retrieval.py','--rsa-update','--aligned-query','--orthogonal-centers',
        '--slots','32,32','--paper-product-read','--balanced-bridge',*features,'--temporal-qk-init','--cuda-graph',
        '--data-version','3','--sequence-balanced','--steps','20000','--easy-curriculum','--curriculum-span','1000',
        '--cosine','--warmup-steps','100','--min-lr','.0002','--eval-samples','128','--output','../paired_temporal_v3'],check=True)


if __name__=='__main__':main()
