"""Four complementary codebooks, same 64 primary slots; full R cost reported."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    runs={}
    for name in ('paired_quadratic_v3','paired_context_v3','paired_readgate_v3','paired_temporal_v3','paired_widekey_v3'):
        p=Path('..')/name/'results.json'
        if p.is_file():runs[name]=json.loads(p.read_text())
    def score(run):
        return min(v['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
                   for k,v in run['evaluations']['256']['tasks'].items())
    if any(score(v)>=.95 for v in runs.values()):
        print('Validation target reached; skip extra geometry trial',flush=True);return
    name=max(runs,key=lambda k:score(runs[k]));cfg=runs[name]['config']
    features=['--'+key.replace('_','-') for key in ('context_controls','quadratic_features','contextual_output','temporal_qk_init') if cfg.get(key)]
    parent={'architecture_from_validation':name,'key_dim':cfg.get('key_dim',0),'slots':[16]*4,
            'primary_slots_per_head':64,'relation_state_per_head':12*16*16,
            'note':'Same primary slots; total cache increases because every ordered R is retained'}
    Path('../geometry_selection.json').write_text(json.dumps(parent,indent=2)+'\n')
    print(json.dumps(parent),flush=True)
    subprocess.run([sys.executable,'scripts/train_retrieval.py','--rsa-update','--aligned-query','--orthogonal-centers',
        '--slots','16,16,16,16','--key-dim',str(cfg.get('key_dim',0)),
        '--paper-product-read','--balanced-bridge',*features,'--cuda-graph','--data-version','3',
        '--sequence-balanced','--steps','20000','--easy-curriculum','--curriculum-span','1000',
        '--cosine','--warmup-steps','100','--min-lr','.0002','--eval-samples','128','--output','../paired_dense4_v3'],check=True)


if __name__=='__main__':main()
