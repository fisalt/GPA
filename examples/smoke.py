"""Small optimizer smoke test; not a NIAH/RULER accuracy experiment."""
import argparse
import json
import torch
from gpa import DenseGPA, GPAConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["torch", "fla"], default="torch")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dtype = torch.bfloat16 if args.backend == "fla" else torch.float32
    model = DenseGPA(GPAConfig(heads=2, key_dim=8, value_dim=8, slots=(4, 4),
                               chunk_size=args.chunk_size, backend=args.backend)).to(args.device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    k = torch.randn(2, 12, 2, 8, device=args.device, dtype=dtype)
    v = torch.randn_like(k)
    # Fixed finite fixture only: learn same-token value reconstruction. This
    # exercises backward, not long-range/generalization or retrieval accuracy.
    history = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out, state = model(k, k, v)
        loss = (out.float()-v.float()).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("nonfinite optimization smoke test")
        optimizer.step()
        history.append(loss.item())
    print(json.dumps({"task": "fixed-fixture optimizer smoke, NOT retrieval evaluation",
                      "steps": args.steps, "initial_mse": history[0], "final_mse": history[-1],
                      "state_elements": state.numel(), "device": args.device,
                      "backend": args.backend}, indent=2))


if __name__ == "__main__":
    main()
