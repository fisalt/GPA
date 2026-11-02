"""Evaluate a strict-config tiny-LM checkpoint, preserving its training protocol."""
import argparse
import json
from pathlib import Path
import torch
from dataclasses import replace,asdict
from gpa.retrieval import RetrievalConfig,RetrievalLM,evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=920000)
    p.add_argument("--samples",type=int,default=256)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--lengths",default="128,256,512")
    p.add_argument("--data-version",type=int,choices=[1,2,3],default=3)
    p.add_argument("--execution",choices=["reference","packed","triton","staged"])
    args=p.parse_args();torch.set_num_threads(4)
    data=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg=RetrievalConfig(**data["config"])
    if args.execution:cfg=replace(cfg,execution=args.execution)
    model=RetrievalLM(cfg).cuda().eval();model.load_state_dict(data["model"])
    meta=dict(data.get("training_metadata",{}))
    meta.update(test_seed=args.seed,data_version=args.data_version,evaluation_only=True,
                samples_per_task=args.samples)
    if args.execution:meta["execution_override"]=args.execution
    result={"config":asdict(cfg),"steps":data["step"],"training_metadata":meta,
            "parameters":sum(p.numel() for p in model.parameters()),"evaluations":{}}
    output=Path(args.output)/"results.json";output.parent.mkdir(parents=True,exist_ok=True)
    maximum=meta.get("max_records",8)
    for length in map(int,args.lengths.split(",")):
        records=min(4,maximum) if length<=128 else min(maximum+(length//256-1)*4,16)
        tasks=evaluate(model,length=length,records=records,samples=args.samples,batch=args.batch,
                       seed=args.seed,free_generate=True,data_version=args.data_version)
        result["evaluations"][str(length)]={"records":records,"tasks":tasks,
                                               "extrapolation":length>meta.get("max_length",256)}
        output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"length":length,"tasks":tasks}),flush=True)


if __name__=="__main__":main()
