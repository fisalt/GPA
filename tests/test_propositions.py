import math
import pytest
import torch
from gpa.ops import rope
from gpa.propositions import (
    ExactKVCache, DecoupledRoPESequence, PositionalMomentCache, exponential_taylor_bound,
    analyze_interlayer_similarity, propagate_capacity, ResidualCodebook,
    contraction_error_bound, CountedSlotState, counted_slot_step,
)


def test_p1_joint_bitwise_dedup_causal_exact():
    torch.manual_seed(1)
    keys = torch.randn(4, 6, dtype=torch.double)
    values = torch.randn(4, 3, dtype=torch.double)
    # Equal keys, different values must not be merged.
    keys[2] = keys[0]
    ids = [0, 1, 0, 2, 3, 2, 0, 1]
    k, v = keys[ids], values[ids]
    cache = ExactKVCache()
    for t in range(len(ids)):
        cache.append(k[t], v[t])
        q = torch.randn(6, dtype=torch.double)
        dense = (q @ k[:t+1].T / math.sqrt(6)).softmax(-1) @ v[:t+1]
        torch.testing.assert_close(cache.read(q), dense, atol=1e-12, rtol=1e-12)
    assert len(cache.keys) == 4
    assert sum(cache.counts) == len(ids)


def test_p1_signed_zero_and_overflow():
    c = ExactKVCache(capacity=2)
    c.append(torch.tensor([0.]), torch.tensor([1.]))
    c.append(torch.tensor([-0.]), torch.tensor([1.]))
    assert len(c.keys) == 2  # stored bits, not numeric equality
    with pytest.raises(OverflowError):
        c.append(torch.tensor([1.]), torch.tensor([1.]))
    with pytest.raises(ValueError):
        c.append(torch.tensor([float("nan")]), torch.tensor([1.]))


def test_p2_reconstruction_and_nonconstant_indices():
    torch.manual_seed(2)
    e = torch.randn(7, 8, dtype=torch.double)
    ids = torch.tensor([0, 1, 0, 6, 4])
    pos = torch.arange(len(ids))
    representation = DecoupledRoPESequence(e, ids, pos)
    torch.testing.assert_close(representation.reconstruct(), rope(e[ids], pos), atol=0, rtol=0)
    assert representation.metadata_elements() == 2*len(ids)


def test_p3_small_angle_approximation_and_remainder():
    # Only the lowest-frequency rotary pair is populated, a controlled regime.
    k = torch.tensor([[0., 0., 0., 0., 0., 0., .5, .2],
                      [0., 0., 0., 0., 0., 0., -.3, .8]], dtype=torch.double)
    v = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.double)
    q = k[0]
    ids, positions = [0, 1, 0, 1], [0, 1, 2, 3]
    c = PositionalMomentCache()
    for i, p in zip(ids, positions):
        c.append(k[i], v[i], p)
    kr = rope(k[ids], torch.tensor(positions))
    qr = rope(q, 4)
    dense = (qr @ kr.T / math.sqrt(8)).softmax(-1) @ v[ids]
    torch.testing.assert_close(c.read(q, 4), dense, atol=1e-6, rtol=1e-6)
    a = torch.linspace(-2, 2, 25, dtype=torch.double)
    d = torch.linspace(-.8, .8, 25, dtype=torch.double)
    err = ((a+d).exp() - a.exp()*(1+d)).abs()
    assert (err <= exponential_taylor_bound(a, d)+1e-15).all()


def test_p3_invalid_mass_is_not_hidden():
    c = PositionalMomentCache()
    k = torch.tensor([10., 0.])
    c.append(k, torch.tensor([1.]), 3)
    with pytest.raises(ArithmeticError):
        c.read(k, 0)
    assert torch.isfinite(c.read(k, 0, invalid="clamp")).all()


def test_p4_conditional_capacity_and_collision_audit():
    eye = torch.eye(3, dtype=torch.double)
    good = analyze_interlayer_similarity(eye, eye, threshold=.99)
    assert good.bound_applicable
    assert good.conditional_capacity_bound == 3
    collision = torch.tensor([[1., .001], [1., -.001]], dtype=torch.double)
    bad = analyze_interlayer_similarity(torch.tensor([[1., 0.]], dtype=torch.double), collision)
    assert not bad.matched_map_injective
    assert not bad.bound_applicable
    assert bad.conditional_capacity_bound is None
    assert bad.relative_shortfall_bound is None
    assert propagate_capacity(8, [.5, 0, .8], 100) == [8., 16., 100., 100.]


def test_p5_residual_telescopes_and_capacity_is_not_distinct_sums():
    c = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.double)
    model = ResidualCodebook([c, c])
    x = torch.tensor([[1.8, .2], [.3, -.1]], dtype=torch.double)
    reconstruction, ids, residuals = model(x)
    torch.testing.assert_close(x, reconstruction+residuals[-1])
    assert ids.shape == (2, 2)
    assert model.index_tuple_capacity == 4
    assert model.audit_distinct_sums() == 3
    reconstruction.square().sum().backward()
    assert all(c.grad is not None for c in model.codebooks)
    assert contraction_error_bound(torch.tensor(10.), [.5, .2]).item() == 1.
    with pytest.raises(ValueError):
        contraction_error_bound(torch.tensor(10.), [.5, 1.1])


@pytest.mark.parametrize("activation", ["sigmoid", "softmax"])
def test_count_normalized_update_and_gradients(activation):
    torch.manual_seed(5)
    memory = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    count = torch.rand(2, 3, dtype=torch.double)
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    residual, new, p = counted_slot_step(x, CountedSlotState(memory, count), activation=activation)
    expected = memory + (p/(count+p).clamp_min(1))[..., None]*(x[:, None]-memory)
    torch.testing.assert_close(new.memory, expected)
    torch.testing.assert_close(residual, x-torch.einsum("bm,bmd->bd", p, expected))
    residual.square().sum().backward()
    assert torch.isfinite(memory.grad).all() and torch.isfinite(x.grad).all()
