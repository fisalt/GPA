"""Same projected QKV stream: full prefill versus arbitrary cached calls."""
import torch
from gpa import DenseGPA, GPAConfig


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = DenseGPA(GPAConfig(heads=2, key_dim=8, value_dim=6, slots=(4, 8), chunk_size=4))
    q = torch.randn(1, 13, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn(1, 13, 2, 6)
    with torch.no_grad():
        full, _ = model(q, k, v)
        state, outputs = None, []
        for lo, hi in [(0, 5), (5, 6), (6, 10), (10, 13)]:
            out, state = model(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], state)
            outputs.append(out)
        error = (full-torch.cat(outputs, 1)).abs().max().item()
        assert error < 1e-5
        print(f"max_cache_partition_error={error:.3g}; state_elements={state.numel()}")


if __name__ == "__main__":
    main()
