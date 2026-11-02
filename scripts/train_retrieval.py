"""Single-GPU mixed synthetic retrieval training and independent evaluation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import math
import torch
import torch.nn.functional as F
from gpa.retrieval import RetrievalConfig, RetrievalLM, make_batch, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attention", choices=["gpa", "transformer"], default="gpa")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--records", type=int, default=8)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--key-dim",type=int,default=0)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--slots", default="16,16,16,16")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--execution",choices=["reference","packed","triton","staged"],default="triton")
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--read-scale", type=float, default=8.)
    p.add_argument("--seed", type=int, default=400)
    p.add_argument("--output", default="retrieval_run")
    p.add_argument("--resume")
    p.add_argument("--eval-samples", type=int, default=32)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-lengths", default="128,256,512")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--easy-curriculum", action="store_true")
    p.add_argument("--aligned-query", action="store_true")
    p.add_argument("--memory-update", choices=["legacy","paper_dense"], default="legacy")
    p.add_argument("--no-dense-relations", action="store_true")
    p.add_argument("--paper-raw", action="store_true")
    p.add_argument("--paper-transport", action="store_true")
    p.add_argument("--rsa-update", action="store_true")
    p.add_argument("--full-controls", action="store_true")
    p.add_argument("--slot-decay", action="store_true")
    p.add_argument("--orthogonal-centers", action="store_true")
    p.add_argument("--strong-relations", action="store_true")
    p.add_argument("--paper-product-read", action="store_true")
    p.add_argument("--transformer-rope", action="store_true")
    p.add_argument("--slot-feature",choices=["linear","softmax","tanh"],default="linear")
    p.add_argument("--gated-output",action="store_true")
    p.add_argument("--balanced-bridge",action="store_true")
    p.add_argument("--context-controls",action="store_true")
    p.add_argument("--relation-marginal",action="store_true")
    p.add_argument("--quadratic-features",action="store_true")
    p.add_argument("--contextual-output",action="store_true")
    p.add_argument("--temporal-qk-init",action="store_true")
    p.add_argument("--learn-read-scale",action="store_true")
    p.add_argument("--curriculum-span", type=int, default=200)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr", type=float, default=.0001)
    p.add_argument("--data-version", type=int, choices=[1,2,3], default=3)
    p.add_argument("--sequence-balanced", action="store_true")
    args = p.parse_args()
    if args.contextual_output:
        args.context_controls=True;args.gated_output=True
    if args.aligned_query and args.attention != "gpa":
        p.error("--aligned-query is a GPA-only parameterization")
    if args.aligned_query and args.memory_update != "legacy":
        p.error("aligned-query is only an old-model ablation")
    if args.memory_update=="paper_dense" and any((args.paper_product_read,args.balanced_bridge,args.relation_marginal,args.quadratic_features,args.learn_read_scale)):
        p.error("product/bridge/marginal/quadratic flags apply to the paired GDN model")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = RetrievalConfig(width=args.width, heads=args.heads, layers=args.layers,
                              slots=tuple(map(int, args.slots.split(","))), attention=args.attention,
                              chunk_size=args.chunk_size, read_scale=args.read_scale,
                              aligned_query=args.aligned_query,memory_update=args.memory_update,
                              dense_relations=not args.no_dense_relations,paper_raw=args.paper_raw,
                              paper_transport=args.paper_transport,rsa_update=args.rsa_update,
                              full_controls=args.full_controls,slot_decay=args.slot_decay,
                              orthogonal_centers=args.orthogonal_centers,strong_relations=args.strong_relations,
                              paper_product_read=args.paper_product_read,transformer_rope=args.transformer_rope,
                              slot_feature=args.slot_feature,gated_output=args.gated_output,
                              balanced_bridge=args.balanced_bridge,context_controls=args.context_controls,
                              relation_marginal=args.relation_marginal,quadratic_features=args.quadratic_features,
                              contextual_output=args.contextual_output,temporal_qk_init=args.temporal_qk_init,
                              key_dim=args.key_dim,learn_read_scale=args.learn_read_scale,execution=args.execution)
    model = RetrievalLM(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01, fused=True)
    start_step = 0
    if args.resume:
        data = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_config = asdict(RetrievalConfig(**data["config"]))
        desired = asdict(config)
        convert = old_config != desired
        allowed_conversion = (args.aligned_query and not old_config["aligned_query"] and
                              {k: v for k, v in old_config.items() if k != "aligned_query"} ==
                              {k: v for k, v in desired.items() if k != "aligned_query"})
        if convert and not allowed_conversion:
            raise ValueError("checkpoint config differs outside the explicit aligned-query conversion")
        model.load_state_dict(data["model"])
        if not args.eval_only:
            optimizer.load_state_dict(data["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        start_step = data["step"]
        if convert:
            for block in model.blocks:
                block.reset_aligned_queries()
                for parameter in (*block.attention.query_weight, *block.attention.route_weight):
                    optimizer.state.pop(parameter, None)
            print("EXPLICIT CONVERSION: W_qs=C^T/query_scale+delta, delta=0; route=.02*(I-1/M); reset corresponding optimizer moments", flush=True)
    params = sum(p.numel() for p in model.parameters())
    training_metadata = {"seed": args.seed, "test_seed": args.eval_seed, "batch": args.batch,
                         "max_length": args.length, "max_records": args.records,
                         "lr": args.lr, "start_step": start_step,
                         "target_step": args.steps, "easy_curriculum": args.easy_curriculum,
                         "cuda_graph": args.cuda_graph}
    training_metadata.update(cosine=args.cosine,warmup_steps=args.warmup_steps,min_lr=args.min_lr)
    training_metadata.update(curriculum_span=args.curriculum_span,loss_balance="mean_sequence_answer_CE" if args.sequence_balanced else "mean_over_answer_tokens",data_version=args.data_version)
    print(json.dumps({"config": asdict(config), "parameters": params, "train_seed": args.seed,
                      "test_seed": args.eval_seed, "mixture": {"single": .3, "uuid4": .3, "mqar": .2, "overwrite": .2},
                      "note": "synthetic structured retrieval, not official RULER"}), flush=True)
    if not args.eval_only:
        model.train()
        begin = time.perf_counter()
        captured, captured_length = None, None

        def loss_fn(ids, labels):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids)
                if args.sequence_balanced:
                    tokens=F.cross_entropy(logits.float().transpose(1,2),labels,reduction="none")
                    loss=(tokens.sum(1)/(labels!=-100).sum(1).clamp_min(1)).mean()
                else:
                    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
            return loss, logits

        for step in range(start_step, args.steps):
            if args.cosine:
                fraction=max(0.,(step-args.warmup_steps)/max(1,args.steps-args.warmup_steps-1))
                rate=args.min_lr+.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*fraction))
                if step<args.warmup_steps:rate=args.lr*(step+1)/max(1,args.warmup_steps)
                for group in optimizer.param_groups:group["lr"]=rate
            length = min(128, args.length) if step < 100 else args.length
            records = min(4, args.records) if step < 100 else args.records
            if args.easy_curriculum:
                phase = step
                span=args.curriculum_span
                length = min(args.length, 64 if phase < span else 128 if phase < 3*span else args.length)
                records = min(args.records, 1 if phase < span else 2 if phase < 2*span else 4 if phase < 3*span else args.records)
            ids, labels, _ = make_batch(args.seed*100000+step, args.batch, length, records, device="cuda",data_version=args.data_version)
            if args.cuda_graph:
                if captured_length != length:
                    from gpa.training_graph import GradientGraph
                    captured = GradientGraph(model, (ids, labels), loss_fn)
                    captured_length = length
                loss, logits = captured.replay(ids, labels)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, logits = loss_fn(ids, labels)
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError(f"nonfinite training at step {step}")
            optimizer.step()
            if (step+1) % 20 == 0:
                torch.cuda.synchronize()
                mask = labels != -100
                acc = ((logits.argmax(-1) == labels) & mask).sum()/mask.sum()
                print(json.dumps({"step": step+1, "loss": loss.item(), "grad_norm": norm.item(),
                                  "answer_token_acc": acc.item(), "length": length,
                                  "records": records,
                                  "lr":optimizer.param_groups[0]["lr"],
                                  "seconds_per_step_recent": (time.perf_counter()-begin)/20}), flush=True)
                begin = time.perf_counter()
            if (step+1) % 100 == 0 or step+1 == args.steps:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step+1, "config": asdict(config),
                            "training_metadata": training_metadata}, out/f"checkpoint-{step+1}.pt")
    results = {"config": asdict(config), "steps": max(start_step, args.steps) if not args.eval_only else start_step,
               "parameters": params, "training_metadata": training_metadata, "evaluations": {}}
    for length in map(int, args.eval_lengths.split(",")):
        records = min(4, args.records) if length <= 128 else min(args.records+(length//256-1)*4, 16)
        result = evaluate(model, length=length, records=records, samples=args.eval_samples,
                          batch=args.batch, seed=args.eval_seed, free_generate=True,data_version=args.data_version)
        results["evaluations"][str(length)] = {"records": records, "tasks": result,
                                                  "extrapolation": length > args.length}
        (out/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({"eval_length": length, "records": records, "results": result}), flush=True)


if __name__ == "__main__":
    main()
