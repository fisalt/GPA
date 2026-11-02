"""Measure this implementation, not an unmeasured speed-equivalence claim."""
import argparse
import json
import platform
import time
import torch
from gpa import DenseGPA, GPAConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", choices=["torch", "fla"], default="torch")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--slots", default="16,16,16,16")
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--execution", choices=["reference", "packed", "triton"], default="reference")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cuda = args.device.startswith("cuda")
    dtype = torch.bfloat16 if cuda else torch.float32
    cfg = GPAConfig(heads=args.heads, key_dim=args.dim, value_dim=args.dim,
                    slots=tuple(map(int, args.slots.split(","))),
                    chunk_size=args.chunk_size, backend=args.backend, execution=args.execution)
    model = DenseGPA(cfg).to(args.device, dtype)
    x = [torch.randn(1, args.length, args.heads, args.dim, device=args.device, dtype=dtype,
                     requires_grad=args.backward) for _ in range(3)]

    def run():
        model.zero_grad(set_to_none=True)
        for value in x:
            value.grad = None
        with torch.set_grad_enabled(args.backward):
            out, state = model(*x)
            if args.backward:
                out.float().square().mean().backward()
        return state

    for _ in range(2):
        run()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.iterations):
        state = run()
    if cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)/args.iterations
    print(json.dumps({"config": cfg.__dict__, "length": args.length, "batch": 1,
                      "backward": args.backward, "seconds": elapsed,
                      "tokens_per_second": args.length/elapsed,
                      "state_elements": state.numel(), "torch": torch.__version__,
                      "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if cuda else None,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None}, indent=2))


if __name__ == "__main__":
    main()
