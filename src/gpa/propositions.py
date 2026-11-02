"""Small, auditable proposition references, NOT performance kernels.

Exact statements and engineering surrogates are kept separate. These routines
may enumerate codebooks/types or compare pairs; only DenseGPA is the fixed-
shape sequence-linear attention implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import torch
from torch import Tensor, nn
from .ops import rope


def _bits(row: Tensor) -> bytes:
    return row.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


class ExactKVCache:
    """P1: merge identical JOINT K,V bit patterns; preserve Python-int counts.

    Single stream/head educational reference. Not a differentiable grouping
    operator or an efficient GPU hash table. Capacity overflow raises; never
    silently evicts or merges different values with equal keys.
    """

    def __init__(self, capacity: int | None = None):
        self.capacity = capacity
        self.keys, self.values, self.counts, self._ids = [], [], [], {}
        self._signature = None

    def append(self, key: Tensor, value: Tensor):
        if key.ndim != 1 or value.ndim != 1 or not key.is_floating_point() or not value.is_floating_point():
            raise ValueError("expected finite floating vectors")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("nonfinite patterns are outside this reference's contract")
        signature = (key.dtype, key.device, key.shape, value.dtype, value.device, value.shape)
        if self._signature is not None and signature != self._signature:
            raise ValueError("stream dtype/device/dimensions must stay unchanged")
        pattern = (_bits(key), _bits(value))
        if pattern in self._ids:
            i = self._ids[pattern]
            self.counts[i] += 1
        else:
            if self.capacity is not None and len(self.keys) >= self.capacity:
                raise OverflowError("distinct joint KV types exceed explicit capacity")
            self._signature = signature
            self._ids[pattern] = len(self.keys)
            self.keys.append(key.clone())
            self.values.append(value.clone())
            self.counts.append(1)

    def read(self, query: Tensor, *, scale: float | None = None):
        if not self.keys:
            raise ValueError("cannot read an empty cache")
        k, v = torch.stack(self.keys), torch.stack(self.values)
        scale = query.shape[-1] ** -.5 if scale is None else scale
        # math.log handles arbitrarily large Python integer counts; converting
        # the counts to floating integers first would lose their exactness.
        log_counts = query.new_tensor([math.log(n) for n in self.counts])
        logits = query @ k.T * scale + log_counts
        return logits.softmax(-1) @ v


@dataclass
class DecoupledRoPESequence:
    """P2: exactly reconstruct shared embedding + O(T) indices + positions.

    The embedding table is not copied. This representation is NOT by itself
    a constant-memory recurrent cache.
    """
    embedding: Tensor
    indices: Tensor
    positions: Tensor
    base: float = 10000.

    def reconstruct(self) -> Tensor:
        if self.indices.ndim != 1 or self.positions.shape != self.indices.shape:
            raise ValueError("indices/positions must be equally sized vectors")
        return rope(self.embedding[self.indices], self.positions, self.base)

    def metadata_elements(self) -> int:
        return self.indices.numel() + self.positions.numel()


class PositionalMomentCache:
    """P3: grouped FIRST-ORDER RoPE expansion, including its failure mode.

    For a joint content type (k,v), store n and D=sum(R_pos k-k).
    Attention mass = exp(q_rot.k/sqrt(d)) [n + q_rot.D/sqrt(d)].
    Unrestricted positions do not satisfy a uniform small-angle guarantee.
    """

    def __init__(self, base: float = 10000.):
        self.base = base
        self.keys, self.values, self.counts, self.moments, self._ids = [], [], [], [], {}

    def append(self, key: Tensor, value: Tensor, position: int):
        if key.ndim != 1 or value.ndim != 1 or key.shape[-1] % 2:
            raise ValueError("expected vector K,V with even key dimension")
        pattern = (_bits(key), _bits(value))
        delta = rope(key, position, self.base) - key
        if pattern not in self._ids:
            self._ids[pattern] = len(self.keys)
            self.keys.append(key.clone())
            self.values.append(value.clone())
            self.counts.append(1)
            self.moments.append(delta)
        else:
            i = self._ids[pattern]
            self.counts[i] += 1
            self.moments[i] = self.moments[i] + delta

    def read(self, query: Tensor, position: int, *, invalid: str = "raise"):
        if invalid not in ("raise", "clamp"):
            raise ValueError("invalid must be raise or clamp")
        if not self.keys:
            raise ValueError("empty positional moment cache")
        q = rope(query, position, self.base)
        k, v, delta = torch.stack(self.keys), torch.stack(self.values), torch.stack(self.moments)
        scale = q.shape[-1] ** -.5
        base = q @ k.T * scale
        correction = q @ delta.T * scale
        factor = q.new_tensor(self.counts) + correction
        if (factor <= 0).any():
            if invalid == "raise":
                raise ArithmeticError("first-order masses are nonpositive; small-angle regime violated")
            # Explicit extra approximation, never a purported exact theorem.
            factor = factor.clamp_min(torch.finfo(factor.dtype).eps)
        mass = (base-base.max()).exp() * factor
        return (mass / mass.sum()) @ v


def exponential_taylor_bound(a: Tensor, delta: Tensor) -> Tensor:
    """|e^(a+delta)-e^a(1+delta)| <= this nonnegative remainder envelope."""
    return .5 * torch.exp(a + delta.abs()) * delta.square()


def _unique_rows(x: Tensor) -> Tensor:
    seen, rows = set(), []
    for row in x:
        pattern = _bits(row)
        if pattern not in seen:
            rows.append(row)
            seen.add(pattern)
    return torch.stack(rows) if rows else x


@dataclass(frozen=True)
class InterlayerReport:
    previous_types: int
    current_types: int
    matched_types: int
    match_ratio: float
    matched_map_injective: bool
    separation_satisfied: bool
    bound_applicable: bool
    conditional_capacity_bound: float | None
    relative_shortfall_bound: float | None


def analyze_interlayer_similarity(previous: Tensor, current: Tensor, *, threshold: float = .99,
                                  capacity_multiplier: float = 1.) -> InterlayerReport:
    """P4: audit the conditional bound; explicitly reject many-to-one matches.

    Diagnostic O(N_previous*N_current), NOT part of the sequence-linear core.
    A sufficient current-row separation is distance > 2 sqrt(2(1-tau))
    on the unit sphere, guaranteeing disjoint threshold balls.
    """
    if previous.ndim != 2 or current.ndim != 2 or previous.shape[1] != current.shape[1]:
        raise ValueError("expected [types,dimension] matrices with common dimension")
    if not 0 <= threshold <= 1 or capacity_multiplier < 0:
        raise ValueError("threshold in [0,1], capacity_multiplier >= 0 required")
    p, c = _unique_rows(previous), _unique_rows(current)
    if not len(p) or not len(c) or (p.norm(dim=-1) == 0).any() or (c.norm(dim=-1) == 0).any():
        raise ValueError("nonempty nonzero type vectors required")
    pn, cn = torch.nn.functional.normalize(p, dim=-1), torch.nn.functional.normalize(c, dim=-1)
    scores, indices = (cn @ pn.T).max(-1)
    matched = scores >= threshold
    n = int(matched.sum())
    ratio = n / len(c)
    injective = len(indices[matched].unique()) == n
    if len(c) < 2:
        separation = True
    else:
        distances = torch.cdist(cn, cn)
        distances.fill_diagonal_(float("inf"))
        separation = bool((distances > 2 * math.sqrt(2 * (1-threshold))).all())
    applicable = bool(n and injective and separation)
    # Actual injectivity already suffices to count matched rows; separation is
    # also required here to certify the manuscript's stated assumptions.
    bound = len(p) / ratio if applicable else None
    return InterlayerReport(len(p), len(c), n, ratio, injective, separation, applicable,
                            bound, max(0., 1-capacity_multiplier*ratio) if applicable else None)


def propagate_capacity(vocabulary: int, ratios: list[float], sequence_length: int) -> list[float]:
    """Conditional capacity recurrence; p=0 falls back to sequence length T."""
    if vocabulary <= 0 or sequence_length <= 0 or any(not 0 <= p <= 1 for p in ratios):
        raise ValueError("positive sizes and ratios in [0,1] required")
    bounds = [float(min(vocabulary, sequence_length))]
    for p in ratios:
        bounds.append(min(bounds[-1]/p, sequence_length) if p else float(sequence_length))
    return bounds


class ResidualCodebook(nn.Module):
    """P5: residual quantization with learnable codebooks; nearest assignment.

    Product(capacities) counts index tuples, not necessarily distinct sums.
    Nearest assignments are nondifferentiable; selected centroids get gradients.
    """

    def __init__(self, codebooks: list[Tensor]):
        super().__init__()
        if not codebooks or any(c.ndim != 2 or c.shape[1] != codebooks[0].shape[1] for c in codebooks):
            raise ValueError("nonempty [slots,common_dimension] codebooks required")
        self.codebooks = nn.ParameterList([nn.Parameter(c.clone()) for c in codebooks])

    def forward(self, x: Tensor):
        residual, reconstruction, indices, history = x, torch.zeros_like(x), [], [x]
        for codebook in self.codebooks:
            index = (residual[..., None, :] - codebook).square().sum(-1).argmin(-1)
            selected = codebook[index]
            reconstruction = reconstruction + selected
            residual = residual - selected
            indices.append(index)
            history.append(residual)
        return reconstruction, torch.stack(indices, -1), tuple(history)

    @property
    def index_tuple_capacity(self):
        return math.prod(c.shape[0] for c in self.codebooks)

    def audit_distinct_sums(self, max_tuples: int = 4096) -> int:
        if self.index_tuple_capacity > max_tuples:
            raise ValueError("enumeration limit exceeded")
        sums = torch.stack([sum(parts) for parts in itertools.product(*self.codebooks)])
        return len(_unique_rows(sums))


def contraction_error_bound(initial_norm: Tensor, rhos: list[float]) -> Tensor:
    if any(not 0 <= rho < 1 for rho in rhos):
        raise ValueError("the product bound requires every assumed contraction in [0,1)")
    return initial_norm * math.prod(rhos)


@dataclass(frozen=True)
class CountedSlotState:
    memory: Tensor  # [...,M,D], joint K/V or any stored feature
    count: Tensor   # [...,M]


def counted_slot_step(feature: Tensor, state: CountedSlotState, *, activation: str = "softmax"):
    """P1+P5: count-normalized soft assignment centroid update, no GDN claim.

    Identical initial slots remain symmetric: use asymmetric learnable prefixes
    or codebooks when training a real model.
    """
    logits = torch.einsum("...d,...md->...m", feature, state.memory)
    if activation == "softmax":
        assignment = logits.softmax(-1)
    elif activation == "sigmoid":
        assignment = logits.sigmoid()
    else:
        raise ValueError("activation must be softmax or sigmoid")
    count = state.count + assignment
    weight = assignment / count.clamp_min(1.)
    memory = state.memory + weight[..., None] * (feature[..., None, :] - state.memory)
    residual = feature - torch.einsum("...m,...md->...d", assignment, memory)
    return residual, CountedSlotState(memory, count), assignment
