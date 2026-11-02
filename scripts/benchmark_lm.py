"""Same-checkpoint LM timing: forward CE/backward/clip/fused AdamW, no weight changes.

lr=0 preserves the checkpoint while exercising the real optimizer kernels.
Compile/warmup, data generation, logs and checkpoint IO are excluded.
"""
import argparse
from collections import Counter
from dataclasses import replace,asdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from gpa.retrieval import RetrievalConfig,RetrievalLM,make_batch
from gpa.training_graph import GradientGraph


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--control")
    p.add_argument("--output",required=True)
    p.add_argument("--lengths",default="256,1024")
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--iterations",type=int,default=10)
    p.add_argument("--variants",default="triton,triton_graph,staged,staged_graph")
    args=p.parse_args();torch.set_num_threads(4)
    source=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    control=torch.load(args.control,map_location="cpu",weights_only=False) if args.control else None
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "scope":"forward answer CE + backward + clip1 + fused AdamW(lr0); excludes data/logs/IO/compile/warmup",
        "records":[]}
    for length in map(int,args.lengths.split(',')):
        ids,labels,meta=make_batch(901222,args.batch,length,8,device="cuda",data_version=3)
        variants=[(name,source) for name in args.variants.split(',')]
        if control:variants.extend((name,control) for name in ('transformer','transformer_graph'))
        baseline={}
        for name,data in variants:
            is_control=name.startswith('transformer');graph=name.endswith('_graph')
            execution=name.removesuffix('_graph') if not is_control else 'triton'
            cfg=replace(RetrievalConfig(**data['config']),execution=execution)
            model=RetrievalLM(cfg).cuda().train();model.load_state_dict(data['model'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.01,fused=True)
            def loss_fn(x,y):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(x)
                    loss=F.cross_entropy(logits.float().transpose(1,2),y,reduction='none')
                    loss=(loss.sum(1)/(y!=-100).sum(1).clamp_min(1)).mean()
                return loss,logits
            captured=GradientGraph(model,(ids,labels),loss_fn) if graph else None
            def forward_backward():
                if captured:return captured.replay(ids,labels)
                model.zero_grad(set_to_none=True)
                loss,logits=loss_fn(ids,labels);loss.backward()
                return loss,logits
            loss,logits=forward_backward()
            grads=torch.cat([x.grad.detach().float().flatten() for x in model.parameters() if x.grad is not None])
            key='control' if is_control else 'gpa'
            if key not in baseline:baseline[key]=(logits.detach().float().clone(),grads.clone())
            ref,refgrad=baseline[key]
            item={"variant":name,"length":length,"batch":args.batch,"config":asdict(cfg),
                "parameters":sum(x.numel() for x in model.parameters()),
                "task_counts":dict(Counter(x['task'] for x in meta)),"loss":loss.item(),
                "logits_relative_l2":((logits.float()-ref).norm()/ref.norm().clamp_min(1e-12)).item(),
                "parameter_gradient_relative_l2":((grads-refgrad).norm()/refgrad.norm().clamp_min(1e-12)).item()}
            def step():
                result=forward_backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                return result
            for _ in range(4):step()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            for _ in range(args.iterations):step()
            torch.cuda.synchronize()
            item.update(seconds_per_step=(time.perf_counter()-begin)/args.iterations,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
            item['tokens_per_second']=args.batch*length/item['seconds_per_step']
            report['records'].append(item)
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(item),flush=True)
            del captured,optimizer,model
    print(output,flush=True)


if __name__=='__main__':main()
