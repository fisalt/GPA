"""Offline end-to-end smoke: random tiny Qwen -> GPA -> CPT -> resume -> merge.

Local text is a fixture, NOT evidence of FineWeb convergence or retrieval scores.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def run(script, *args):
    previous = sys.argv
    try:
        sys.argv = [script, *map(str,args)]
        runpy.run_path(str(Path(__file__).with_name(script)),run_name="__main__")
    finally: sys.argv = previous


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    source,converted,training,merged=(root/n for n in ('native','converted','train','merged'))
    torch.manual_seed(7);torch.set_num_threads(2)
    cfg=AutoConfig.for_model('qwen3',vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        tie_word_embeddings=True,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    native=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager')
    native.save_pretrained(source)
    vocab={'<pad>':0,'<s>':1,'</s>':2,'<unk>':3,**{f'token{i}':i+4 for i in range(124)}}
    raw=Tokenizer(WordLevel(vocab,unk_token='<unk>'));raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',bos_token='<s>',
                               eos_token='</s>',unk_token='<unk>')
    tok.save_pretrained(source)
    data=root/'fixture.jsonl'
    data.write_text('\n'.join(json.dumps({'text':' '.join(f'token{(i+j)%100}' for j in range(30))})
                              for i in range(80))+'\n',encoding='utf-8')
    run('convert_qwen_to_gpa.py','--model',source,'--output',converted,'--backend','torch',
        '--execution','packed','--chunk-size','8','--local-files-only')
    common=['--model',converted,'--output',training,'--data-files',data,'--length','32',
            '--accumulation','1','--warmup-steps','0','--save-steps','1','--loss-chunk-size','16',
            '--lora-rank','2','--precision','fp32','--backend','torch','--shuffle-buffer','0',
            '--device','cpu','--local-files-only']
    # Same planned schedule on both arms: resume an intermediate checkpoint,
    # not a completed cosine schedule whose LR has already reached zero.
    run('train_gpa_fineweb.py',*common,'--max-steps','3')
    cp=training/'checkpoint-2'
    assert (cp/'trainer_state.json').is_file()
    baseline=AutoModelForCausalLM.from_pretrained(training/'final',trust_remote_code=True,local_files_only=True)
    expected={n:v.detach().clone() for n,v in baseline.state_dict().items()}
    resumed=root/'resumed'
    continuation=list(common);continuation[continuation.index('--output')+1]=resumed
    run('train_gpa_fineweb.py',*continuation,'--resume',cp,'--max-steps','3')
    assert json.loads((resumed/'checkpoint-3/trainer_state.json').read_text())['global_step']==3
    restored=AutoModelForCausalLM.from_pretrained(resumed/'final',trust_remote_code=True,local_files_only=True)
    for n,v in restored.state_dict().items():
        torch.testing.assert_close(v,expected[n],atol=0,rtol=0,msg=n)
    run('convert_qwen_to_gpa.py','--model',resumed/'final','--output',merged,'--merge-lora',
        '--local-files-only')
    model=AutoModelForCausalLM.from_pretrained(merged,trust_remote_code=True,local_files_only=True).eval()
    assert model.config.lora['rank']==0
    ids=tok('token1 token2 token3',return_tensors='pt')['input_ids']
    with torch.no_grad():
        generated=model.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True)
    record=dict(status='PASS',source='random tiny Qwen3',training_data='local text fixture',
        conversion=True,full_lora_checkpoint=True,resume_global_step=3,
        resume_matches_uninterrupted_exactly=True,
        merged_auto_map_reload=True,use_cache_generation=True,generated_length=generated.shape[1],
        real_fineweb_quality_claim=False)
    (root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
