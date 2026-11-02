"""Optional CUDA acceptance tests. Skipped, not declared passed, on CPU."""
from dataclasses import replace
import json
import pytest
import torch
from gpa import GPAConfig, DenseGPA


@pytest.mark.parametrize("position", ["gated", "moments"])
@pytest.mark.parametrize("dim,slots", [(32, (16, 32)), (8, (4, 7))])
def test_fla_forward_backward_and_streaming_parity(position, dim, slots):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("fla")
    torch.manual_seed(21)
    config = GPAConfig(heads=2, key_dim=dim, value_dim=dim, slots=slots, chunk_size=64,
                       positional_mode=position)
    reference = DenseGPA(config).cuda().bfloat16()
    fused = DenseGPA(replace(config, backend="fla")).cuda().bfloat16()
    fused.load_state_dict(reference.state_dict())
    inputs = [torch.randn(1, 128, 2, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
              for _ in range(3)]
    other = [x.detach().clone().requires_grad_() for x in inputs]
    a, sa = reference(*inputs)
    b, sb = fused(*other)
    torch.testing.assert_close(a.float(), b.float(), atol=.025, rtol=.06)
    forward_error = (a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8)
    assert forward_error < .06
    a.float().square().mean().backward()
    b.float().square().mean().backward()
    gradient_errors = []
    for x, y in zip(inputs, other):
        # A global normalized error avoids meaningless relative errors near 0.
        error = (x.grad.float()-y.grad.float()).norm()/x.grad.float().norm().clamp_min(1e-8)
        assert error < .15, error.item()
        gradient_errors.append(error.item())
    for ra, rb in zip(sa.relations, sb.relations):
        state_error = (ra.memory-rb.memory).norm()/ra.memory.norm().clamp_min(1e-8)
        assert state_error < .06
    assert fused.inner_prefix[0].grad is not None
    assert fused.inner_prefix[0].grad.abs().sum() > 0
    assert fused.relation_prefix[0].grad is not None
    assert torch.isfinite(fused.relation_prefix[0].grad).all()
    # Same mathematical block boundaries even during one-token decode.
    with torch.no_grad():
        state, outputs = None, []
        for lo, hi in [(0, 47), (47, 65), (65, 128)]:
            out, state = fused(*(x[:, lo:hi] for x in other), state)
            outputs.append(out)
        torch.testing.assert_close(torch.cat(outputs, 1).float(), b.float(), atol=.025, rtol=.06)
        cache_error = (torch.cat(outputs, 1).float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)
        assert cache_error < .06
    print(json.dumps({"mode": position, "dim": dim, "slots": slots,
                      "forward_relative_l2": forward_error.item(),
                      "input_gradient_relative_l2": gradient_errors,
                      "cache_partition_relative_l2": cache_error.item()}))
