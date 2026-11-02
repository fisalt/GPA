"""One-GPU, matched-weight timing/gradient comparisons; compilation excluded."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time
import torch
from gpa import DenseGPA, GPAConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", default="256,1024")
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--output", default="efficiency.json")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--rsa-update",action="store_true")
    p.add_argument("--paper-product-read",action="store_true")
    p.add_argument("--relation-marginal",action="store_true")
    p.add_argument("--balanced-bridge",action="store_true")
    p.add_argument("--quadratic-features",action="store_true")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    cfg = GPAConfig(heads=2, key_dim=32, value_dim=32, slots=(args.slots,)*args.levels,
                     chunk_size=args.chunk_size, backend="fla",rsa_update=args.rsa_update,
                     paper_product_read=args.paper_product_read,relation_marginal=args.relation_marginal,
                     balanced_bridge=args.balanced_bridge,quadratic_features=args.quadratic_features)
    master = DenseGPA(cfg).cuda().bfloat16()
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "records": []}
    for length in map(int, args.lengths.split(",")):
        inputs = [torch.randn(1, length, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        baseline_output = baseline_grad = None
        variants = ["reference", "packed", "triton", "single_level_equal_primary_state"]
        if args.cuda_graph:
            variants += ["triton_cuda_graph", "single_level_cuda_graph"]
        for variant in variants:
            if variant.startswith("single"):
                config = replace(cfg, slots=(args.slots*args.levels,), execution="packed")
            else:
                config = replace(cfg, execution="triton" if variant.endswith("_cuda_graph") else variant)
            model = DenseGPA(config).cuda().bfloat16()
            if not variant.startswith("single"):
                model.load_state_dict(master.state_dict())
            xs = [x.detach().clone().requires_grad_() for x in inputs]

            if variant.endswith("_cuda_graph"):
                from gpa.training_graph import GradientGraph

                def fn(*vals):
                    y, state = model(*vals)
                    return y.float().square().mean(), (y, state)

                captured = GradientGraph(model, tuple(xs), fn)
                gradient_inputs = captured.inputs

                def run():
                    _, output = captured.replay(*xs)
                    return output
            else:
                gradient_inputs = xs

                def run():
                    model.zero_grad(set_to_none=True)
                    for x in xs:
                        x.grad = None
                    y, state = model(*xs)
                    y.float().square().mean().backward()
                    return y, state

            for _ in range(2):
                y, state = run()
            torch.cuda.synchronize()
            if variant == "reference":
                baseline_output = y.detach().float()
                baseline_grad = [x.grad.detach().float().clone() for x in xs]
            record = {"variant": variant, "length": length, "config": config.__dict__,
                      "state_elements": state.numel()}
            if not variant.startswith("single"):
                record["output_relative_l2_vs_reference"] = ((y.float()-baseline_output).norm()/baseline_output.norm()).item()
                record["gradient_relative_l2_vs_reference"] = [
                    ((x.grad.float()-g).norm()/g.norm().clamp_min(1e-9)).item() for x, g in zip(gradient_inputs, baseline_grad)]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(args.iterations):
                y, state = run()
            torch.cuda.synchronize()
            record["fwd_bwd_seconds"] = (time.perf_counter()-begin)/args.iterations
            record["tokens_per_second"] = length/record["fwd_bwd_seconds"]
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if length == int(args.lengths.split(",")[0]) and variant in ("reference", "triton"):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run()
                    torch.cuda.synchronize()
                record["profile_top_cuda"] = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12)
                record["profile_top_cpu"] = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
            report["records"].append(record)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in record.items() if not k.startswith("profile_")}), flush=True)


if __name__ == "__main__":
    main()
