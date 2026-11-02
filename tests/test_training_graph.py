import pytest
import torch


def test_gradient_graph_overwrites_instead_of_accumulating():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from gpa.training_graph import GradientGraph
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).cuda()
    x, y = torch.randn(4, 16, device="cuda", requires_grad=True), torch.randn(4, 8, device="cuda")

    def fn(a, b):
        out = model(a)
        return (out-b).square().mean(), out

    captured = GradientGraph(model, (x, y), fn)
    captured.replay(x, y)
    once = [p.grad.clone() for p in model.parameters()]
    input_once = captured.inputs[0].grad.clone()
    captured.replay(x, y)
    for p, g in zip(model.parameters(), once):
        torch.testing.assert_close(p.grad, g)
    torch.testing.assert_close(captured.inputs[0].grad, input_once)
    # Changed data really affects gradients, rather than replaying fixed loss.
    captured.replay(x+1, y)
    assert any(not torch.allclose(p.grad, g) for p, g in zip(model.parameters(), once))


def test_full_gpa_graph_and_input_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("fla")
    from gpa import DenseGPA, GPAConfig
    from gpa.training_graph import GradientGraph
    model = DenseGPA(GPAConfig(heads=2, key_dim=32, value_dim=32, slots=(16, 16),
                              chunk_size=64, backend="fla", execution="triton")).cuda()
    xs = tuple(torch.randn(1, 64, 2, 32, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3))

    def fn(*inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output, _ = model(*inputs)
        return output.float().square().mean(), output

    captured = GradientGraph(model, xs, fn)
    loss, output = captured.replay(*xs)
    once = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    captured.replay(*xs)
    for p, g in zip((p for p in model.parameters() if p.grad is not None), once):
        torch.testing.assert_close(p.grad, g, atol=1e-5, rtol=1e-4)
    assert all(torch.isfinite(x.grad).all() for x in captured.inputs)
