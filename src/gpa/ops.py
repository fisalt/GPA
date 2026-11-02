"""Key-major GDN and additive-state primitives. No temporal attention matrix."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def state_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def unit(x: Tensor) -> Tensor:
    # Use the same explicit normalization in torch and FLA; never rely on a
    # backend's different epsilon or implicit scale.
    return F.normalize(x.to(state_dtype(x)), p=2, dim=-1, eps=1e-6).to(x.dtype)


def _pad_width(x: Tensor, extra: int) -> Tensor:
    # Avoid an otherwise unnecessary full clone for the normal >=16 case.
    return F.pad(x, (0, extra)) if extra else x


def _pad_state(x: Tensor, pk: int, pv: int) -> Tensor:
    return F.pad(x, (0, pv, 0, pk)) if pk or pv else x


def gdn_update(state: Tensor, key: Tensor, value: Tensor, log_gate: Tensor,
               beta: Tensor) -> Tensor:
    """S:[B,H,K,V], k:[B,H,K], v:[B,H,V]; scalar g,b:[B,H].

    S_t = alpha S_prev + beta khat (v - khat^T alpha S_prev)^T.
    A scalar gate cannot secretly select a K/V axis; vector gates are NOT
    accepted by this API. Inputs are normalized only on the key axis.
    """
    if log_gate.shape != key.shape[:-1] or beta.shape != key.shape[:-1]:
        raise ValueError("GDN scalar gates/beta must be [B,H], not vector gates")
    k = unit(key).to(state.dtype)
    decayed = state * log_gate.to(state.dtype).exp()[..., None, None]
    error = value.to(state.dtype) - torch.einsum("bhk,bhkv->bhv", k, decayed)
    return decayed + beta.to(state.dtype)[..., None, None] * k[..., :, None] * error[..., None, :]


def memory_read(query: Tensor, state: Tensor) -> Tensor:
    return torch.einsum("bhk,bhkv->bhv", unit(query).to(state.dtype), state).to(query.dtype)


def gdn_scan(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
             initial_state: Tensor, *, backend: str = "torch") -> tuple[Tensor, Tensor]:
    """[B,T,H,D] public interface, scale=1, normalized Q/K, arbitrary V."""
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k must share [B,T,H,K], v must share [B,T,H]")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must be scalar per token/head: [B,T,H]")
    if q.shape[1] == 0:
        return v.new_empty(*v.shape), initial_state
    if backend == "fla":
        if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FLA requires CUDA fp16/bf16 tensors; use backend='torch' otherwise")
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as exc:
            raise ImportError("Install flash-linear-attention in your CUDA environment") from exc
        # Tensor-core backward kernels in some FLA versions require reduction
        # widths >=16. Zero-padding is algebraically neutral, unlike changing
        # the slot count or falling back silently to another recurrence.
        kd, vd = q.shape[-1], v.shape[-1]
        pk, pv = max(0, 16-kd), max(0, 16-vd)
        out, final = chunk_gated_delta_rule(
            _pad_width(unit(q), pk).contiguous(), _pad_width(unit(k), pk).contiguous(),
            _pad_width(v, pv).contiguous(),
            g.contiguous().float(), beta.to(q.dtype).contiguous(),
            scale=1., initial_state=_pad_state(initial_state.float(), pk, pv).contiguous(),
            output_final_state=True, use_qk_l2norm_in_kernel=False,
        )
        return out[..., :vd], final[..., :kd, :vd]
    if backend != "torch":
        raise ValueError(f"unknown backend: {backend}")
    state = initial_state
    outputs = []
    for t in range(q.shape[1]):
        state = gdn_update(state, k[:, t], v[:, t], g[:, t], beta[:, t])
        outputs.append(memory_read(q[:, t], state))
    return torch.stack(outputs, dim=1), state


def linear_scan(q: Tensor, k: Tensor, v: Tensor, initial_state: Tensor,
                *, backend: str = "torch") -> tuple[Tensor, Tensor]:
    """Additive positional moments, S += k v^T; no normalization of inputs."""
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("expected matching [B,T,H,K] Q/K and [B,T,H,V] V")
    if q.shape[1] == 0:
        return v.new_empty(v.shape), initial_state
    if backend == "fla":
        if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FLA requires CUDA fp16/bf16 tensors")
        try:
            from fla.ops.linear_attn import chunk_linear_attn
        except ImportError as exc:
            raise ImportError("This FLA version does not expose chunk_linear_attn") from exc
        # The public FLA operator uses time-first by default in current FLA.
        kd, vd = q.shape[-1], v.shape[-1]
        pk, pv = max(0, 16-kd), max(0, 16-vd)
        out, final = chunk_linear_attn(
            _pad_width(q, pk).contiguous(), _pad_width(k, pk).contiguous(),
            _pad_width(v, pv).contiguous(), scale=1.,
            initial_state=_pad_state(initial_state.float(), pk, pv).contiguous(),
            output_final_state=True, normalize=False)
        return out[..., :vd], final[..., :kd, :vd]
    if backend != "torch":
        raise ValueError(f"unknown backend: {backend}")
    state = initial_state
    outputs = []
    for t in range(q.shape[1]):
        state = state + k[:, t].to(state.dtype)[..., :, None] * v[:, t].to(state.dtype)[..., None, :]
        outputs.append(torch.einsum("bhd,bhdm->bhm", q[:, t].to(state.dtype), state).to(q.dtype))
    return torch.stack(outputs, 1), state


def rope(x: Tensor, positions: Tensor | int | float, base: float = 10000.) -> Tensor:
    """Adjacent-pair RoPE; x is [...,D] or [B,T,H,D], positions [T]/[B,T]."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE needs an even head dimension")
    work = x.to(state_dtype(x))
    pos = torch.as_tensor(positions, device=x.device, dtype=work.dtype)
    if x.ndim == 4:
        if pos.ndim == 1:
            pos = pos[None, :, None]
        elif pos.ndim == 2:
            pos = pos[:, :, None]
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=work.dtype) / d)
    angle = pos[..., None] * freq
    even, odd = work[..., 0::2], work[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(),
                        even * angle.sin() + odd * angle.cos()), -1).flatten(-2).to(x.dtype)


@dataclass(frozen=True)
class TwoStageState:
    inner: Tensor  # [B,H,Dk,M], key-major
    outer: Tensor  # [B,H,M,Dv], key-major


def two_stage_gdn(q: Tensor, k: Tensor, slots: Tensor, v: Tensor,
                  gi: Tensor, bi: Tensor, go: Tensor, bo: Tensor,
                  state: TwoStageState, *, route_weight: Tensor | None = None,
                  query_weight: Tensor | None = None, query_scale: float = 1/16,
                  backend: str = "torch") -> tuple[Tensor, TwoStageState]:
    """GDN(softmax(GDN(q,k,slots)) W_rs + q W_qs/16, slots, v).

    This exact two-scan building block is also useful independently of the
    full multilevel GPA. No ABC or hidden Transformer attention is used.
    """
    inner, a = gdn_scan(q, k, slots, gi, bi, state.inner, backend=backend)
    route = inner.to(state_dtype(inner)).softmax(-1).to(q.dtype)
    if route_weight is not None:
        route = F.linear(route, route_weight.to(route.dtype))
    if query_weight is not None:
        route = route + query_scale * F.linear(q, query_weight.to(q.dtype))
    out, b = gdn_scan(route, slots, v, go, bo, state.outer, backend=backend)
    return out, TwoStageState(a, b)

