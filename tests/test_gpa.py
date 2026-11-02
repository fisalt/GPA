from dataclasses import replace
import pytest
import torch
from gpa import DenseGPA, GPAConfig, TwoStageState, two_stage_gdn
from gpa.ops import gdn_scan, gdn_update, memory_read, unit
from gpa.state import tensors


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(t=7, batch=2, device="cpu", dtype=torch.double):
    return (torch.randn(batch, t, 2, 4, device=device, dtype=dtype),
            torch.randn(batch, t, 2, 4, device=device, dtype=dtype),
            torch.randn(batch, t, 2, 3, device=device, dtype=dtype))


def assert_states(a, b, atol=1e-10, rtol=1e-10):
    assert a.position == b.position and a.chunk_offset == b.chunk_offset
    for x, y in zip(tensors(a), tensors(b)):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)


def test_gdn_manual_equation_variable_dimensions_and_gradient():
    torch.manual_seed(10)
    q, k, v = inputs()
    g = -torch.rand(2, 7, 2, dtype=torch.double)*.1
    beta = torch.rand_like(g)*.8
    s0 = torch.randn(2, 2, 4, 3, dtype=torch.double, requires_grad=True)
    actual, final = gdn_scan(q, k, v, g, beta, s0)
    s, outputs = s0, []
    for t in range(7):
        kh = unit(k[:, t])
        decayed = g[:, t].exp()[..., None, None]*s
        prediction = (kh[..., None, :] @ decayed).squeeze(-2)
        s = decayed + beta[:, t, :, None, None]*kh[..., :, None]*(v[:, t]-prediction)[..., None, :]
        outputs.append((unit(q[:, t])[..., None, :] @ s).squeeze(-2))
    torch.testing.assert_close(actual, torch.stack(outputs, 1))
    torch.testing.assert_close(final, s)
    actual.square().mean().backward()
    assert s0.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        gdn_update(s0, k[:, 0], v[:, 0], g[:, 0, :, None], beta[:, 0])


def test_two_stage_is_two_gdn_scans_not_abc():
    q, k, v = inputs()
    slot = torch.randn(2, 7, 2, 5, dtype=torch.double)
    g = torch.full((2, 7, 2), -.02, dtype=torch.double)
    beta = torch.full_like(g, .5)
    state = TwoStageState(torch.randn(2, 2, 4, 5, dtype=torch.double),
                          torch.randn(2, 2, 5, 3, dtype=torch.double))
    out, final = two_stage_gdn(q, k, slot, v, g, beta, g, beta, state)
    route, a = gdn_scan(q, k, slot, g, beta, state.inner)
    ref, b = gdn_scan(route.softmax(-1), slot, v, g, beta, state.outer)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(final.inner, a)
    torch.testing.assert_close(final.outer, b)


@pytest.mark.parametrize("chunk", [1, 3, 4])
@pytest.mark.parametrize("position", ["gated", "moments", "none"])
def test_dense_cache_causality_and_arbitrary_external_partition(chunk, position):
    torch.manual_seed(11)
    model = DenseGPA(GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3, 2, 4),
                               positional_mode=position, chunk_size=chunk)).double()
    q, k, v = inputs(t=9)
    full, state = model(q, k, v)
    running, outputs = None, []
    for lo, hi in [(0, 2), (2, 3), (3, 7), (7, 9)]:
        part, running = model(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], running)
        outputs.append(part)
    torch.testing.assert_close(full, torch.cat(outputs, 1), atol=1e-10, rtol=1e-10)
    assert_states(state, running)
    for end in (1, 4, 8):
        prefix, _ = model(q[:, :end], k[:, :end], v[:, :end])
        torch.testing.assert_close(full[:, :end], prefix, atol=1e-10, rtol=1e-10)
    # Every later->earlier edge exists, not merely previous->next.
    assert len(state.relations) == 6
    assert (2, 0) in model.edges and (0, 2) in model.edges


@pytest.mark.parametrize("chunk", [1, 3])
def test_gradients_through_prefixes_relations_and_cached_calls(chunk):
    torch.manual_seed(12)
    model = DenseGPA(GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3, 2), chunk_size=chunk)).double()
    q, k, v = (x.requires_grad_() for x in inputs(t=7))
    _, state = model(q[:, :3], k[:, :3], v[:, :3])
    final, state = model(q[:, 3:], k[:, 3:], v[:, 3:], state)
    loss = final[:, -1].square().sum()
    loss.backward()
    for name, p in model.named_parameters():
        if "moment_gain" not in name:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
    assert k.grad[:, 0].abs().sum() > 1e-12  # no hidden truncation
    assert model.inner_prefix[0].grad.abs().sum() > 0
    assert model.relation_prefix[0].grad.abs().sum() > 0
    assert not any(t.requires_grad for t in tensors(state.detach()))


@pytest.mark.parametrize("chunk", [1, 3])
def test_memory_writes_do_not_depend_on_query(chunk):
    torch.manual_seed(14)
    model = DenseGPA(GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3, 2), chunk_size=chunk)).double()
    q, k, v = inputs(t=6)
    y1, a = model(q, k, v)
    y2, b = model(torch.randn_like(q)*3, k, v)
    assert_states(a, b)
    assert not torch.allclose(y1, y2)


def test_reverse_dense_edge_actually_changes_output():
    torch.manual_seed(15)
    model = DenseGPA(GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3, 2))).double()
    q, k, v = inputs()
    y, _ = model(q, k, v)
    with torch.no_grad():
        e = model.edges.index((1, 0))
        model.relation_prefix[e].add_(torch.randn_like(model.relation_prefix[e])*3)
    z, _ = model(q, k, v)
    assert (y-z).abs().max() > 1e-8


def test_fixed_cache_size_diagnostics_empty_and_positions():
    model = DenseGPA(GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3, 2))).double()
    q, k, v = inputs(t=10)
    _, small = model(q[:, :2], k[:, :2], v[:, :2])
    y, large, d = model(q, k, v, return_diagnostics=True)
    assert small.numel() == large.numel()
    assert d["routes"][0].shape == (2, 10, 2, 3)
    z, same = model(q[:, :0], k[:, :0], v[:, :0], large)
    assert z.shape == (2, 0, 2, 3) and same is large
    mm = DenseGPA(replace(model.config, positional_mode="moments")).double()
    ordinary, _ = mm(q, k, v)
    explicit, _ = mm(q, k, v, position_ids=torch.arange(10).expand(2, -1))
    torch.testing.assert_close(ordinary, explicit)


def test_gradcheck_recurrent_dense_model_inputs():
    torch.manual_seed(16)
    model = DenseGPA(GPAConfig(heads=1, key_dim=2, value_dim=2, slots=(2, 2))).double()
    x = tuple(torch.randn(1, 2, 1, 2, dtype=torch.double, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda q, k, v: model(q, k, v)[0], x, eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("slots", [(4,)*4, (4,)*8, (128, 256, 512)])
def test_four_eight_levels_and_large_unequal_slots(slots):
    torch.manual_seed(17)
    model = DenseGPA(GPAConfig(heads=1, key_dim=4, value_dim=3, slots=slots))
    q = torch.randn(1, 2, 1, 4)
    y, state = model(q, q, torch.randn(1, 2, 1, 3))
    assert y.shape == (1, 2, 1, 3)
    assert len(state.relations) == len(slots)*(len(slots)-1)
    y.square().sum().backward()
    assert torch.isfinite(model.center[0].grad).all()


def test_single_level_and_state_dict_roundtrip():
    config = GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(5,))
    original, loaded = DenseGPA(config).double(), DenseGPA(config).double()
    loaded.load_state_dict(original.state_dict(), strict=True)
    q, k, v = inputs(t=3)
    y, state = original(q, k, v)
    z, restored = loaded(q, k, v)
    assert state.relations == ()
    torch.testing.assert_close(y, z, atol=0, rtol=0)
    assert_states(state, restored, atol=0, rtol=0)
