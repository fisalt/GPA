"""Freeze validation selection before any final held-out evaluation."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    pairs={"paired_product_v3":"control_rope_v3","paired_balanced_v3":"control_rope_v3",
           "paired_context_v3":"control_rope_v3","paired_marginal_v3":"control_rope_v3",
           "paired_quadratic_v3":"control_rope_v3","paired_scaled_v3":"control_scaled_v3",
           "paired_readgate_v3":"control_rope_v3","paired_temporal_v3":"control_rope_v3",
           "paired_widekey_v3":"control_widekey_v3"}
    geometry=Path('../paired_dense4_v3/results.json')
    if geometry.is_file():
        cfg=json.loads(geometry.read_text())['config']
        pairs['paired_dense4_v3']='control_widekey_v3' if cfg.get('key_dim') else 'control_rope_v3'
    candidates={}
    for name,control in pairs.items():
        path=Path('..')/name/'results.json';cp=Path('..')/control/'results.json'
        if not path.is_file() or not cp.is_file():continue
        run=json.loads(path.read_text());baseline=json.loads(cp.read_text())
        t=run['evaluations']['256']['tasks'];bt=baseline['evaluations']['256']['tasks']
        score=lambda v,k:v[k]['free_generation_exact_match' if k=='uuid4' else 'query_exact_match_teacher_forced']
        scores={k:score(t,k) for k in t}
        gap=min(score(t,k)-score(bt,k) for k in t)
        candidates[name]={'control':control,'worst_task':min(scores.values()),'worst_margin':gap,'scores':scores}
    best=max(candidates,key=lambda k:(min(candidates[k]['worst_task']-.95,candidates[k]['worst_margin']+.05),candidates[k]['worst_task']))
    chosen=candidates[best]
    accepted=chosen['worst_task']>=.95 and chosen['worst_margin']>=-.05
    record={'selected':best,**chosen,'validation_seed':900000,'heldout_seed':920000,
            'validation_target_met':accepted,'candidates':candidates,
            'note':'Validation point-estimate target only, not a statistical equivalence claim'}
    Path('../selection_v3.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    # Even a failed validation candidate is independently evaluated, but its
    # failure flag is retained. Do not use this held-out set for later tuning.
    for name,output in ((best,'heldout_best_v3'),(chosen['control'],'heldout_best_control_v3')):
        subprocess.run([sys.executable,'scripts/eval_retrieval.py',f'../{name}/checkpoint-20000.pt',
                        '--output',f'../{output}','--seed','920000','--samples','256'],check=True)
    Path('docs/results/v3').mkdir(parents=True,exist_ok=True)
    Path('docs/results/v3/selection.json').write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run([sys.executable,'scripts/analyze_retrieval.py',f'../{best}/checkpoint-20000.pt',
                    '--task','uuid4','--plot','--ablate-relation-read','--output','docs/results/v3/diagnostics_selected.json'],check=True)


if __name__=='__main__':main()
