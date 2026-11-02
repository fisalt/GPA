from dataclasses import replace
import pytest
import torch
from gpa import GPAConfig, DenseGPA
from gpa.packed import dense_route_reference


@pytest.mark.parametrize("levels,m", [(2, 7), (4, 16), (4, 64), (8, 128)])
@pytest.mark.parametrize("mode", ["legacy", "product", "marginal"])
def test_triton_dense_route_forward_backward(levels, m, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gpa.triton_route import dense_route
    product=mode!="legacy"
    torch.manual_seed(100)
    edges = [(i, j) for i in range(levels) for j in range(levels) if i != j]
    shapes = [(2, 11, levels, 2, m), (2, 11, len(edges), 2, m), (len(edges), 2), (levels, 2)]
    xs = [torch.randn(s, device="cuda", requires_grad=True) for s in shapes]
    if mode=="marginal":xs.append((torch.rand(shapes[0],device="cuda")*100).requires_grad_())
    ys = [x.detach().clone().requires_grad_() for x in xs]
    a = dense_route_reference(*xs[:4], edges, product=product,counts=xs[4] if mode=="marginal" else None)
    b = dense_route(*ys[:4], product=product,counts=ys[4] if mode=="marginal" else None)
    # Product logits can be O(M*exp(gain)). Different FP32 reduction trees
    # amplify a few ulps near tied maxima at M=128; check global error too.
    torch.testing.assert_close(a, b, atol=2e-5 if product else 2e-6,
                               rtol=2e-4 if product else 2e-5)
    assert (a-b).norm()/a.norm().clamp_min(1e-12)<1e-4
    grad = torch.randn_like(a)
    a.backward(grad)
    b.backward(grad)
    for x, y in zip(xs, ys):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-4 if product else 2e-5,
                                   rtol=3e-3 if product else 3e-4)
        assert (x.grad-y.grad).norm()/x.grad.norm().clamp_min(1e-12)<5e-4


def test_packed_fla_triton_model_gradients_and_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("fla")
    cfg = GPAConfig(heads=2, key_dim=32, value_dim=32, slots=(16,)*4,
                     chunk_size=64, backend="fla", execution="packed")
    torch.manual_seed(101)
    a = DenseGPA(cfg).cuda().bfloat16()
    b = DenseGPA(replace(cfg, execution="triton")).cuda().bfloat16()
    b.load_state_dict(a.state_dict())
    xs = [torch.randn(1, 128, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    ya, _ = a(*xs)
    yb, sb = b(*ys)
    torch.testing.assert_close(ya.float(), yb.float(), atol=.01, rtol=.03)
    ya.float().square().mean().backward()
    yb.float().square().mean().backward()
    for x, y in zip(xs, ys):
        error = (x.grad.float()-y.grad.float()).norm()/x.grad.float().norm().clamp_min(1e-9)
        assert error < .06
    with torch.no_grad():
        state, out = None, []
        for lo, hi in [(0, 31), (31, 79), (79, 128)]:
            z, state = b(*(x[:, lo:hi] for x in ys), state)
            out.append(z)
        torch.testing.assert_close(torch.cat(out, 1).float(), yb.float(), atol=.015, rtol=.06)


def test_fp32_master_parameters_with_bf16_autocast():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("fla")
    cfg = GPAConfig(heads=2, key_dim=32, value_dim=32, slots=(16,)*2,
                     chunk_size=64, backend="fla", execution="triton")
    model = DenseGPA(cfg).cuda()
    xs = [torch.randn(1, 64, 2, 32, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y, _ = model(*xs)
        loss = y.float().square().mean()
    loss.backward()
    assert y.dtype == torch.bfloat16
    assert model.center[0].grad.dtype == torch.float32
    assert torch.isfinite(model.center[0].grad).all()
