"""Fused dense all-edge aggregation -> norm/tanh -> route softmax, fwd/bwd.

Every ordered edge is retained. Computation is FP32; dtype casts at the API
boundary follow PyTorch autograd conventions. No Triton import on CPU paths.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _route_fwd(Base, Msg, Edge, Gain, Counts, Out, P,
               H: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
               E: tl.constexpr, BS: tl.constexpr, BM: tl.constexpr, PRODUCT: tl.constexpr, MARGINAL: tl.constexpr):
    row = tl.program_id(0)
    h = row % H
    j = (row // H) % L
    bt = row // (H*L)
    src = tl.arange(0, BS)
    col = tl.arange(0, BM)
    valid = (src < L) & (src != j)
    e = src*(L-1)+tl.where(j > src, j-1, j)
    wlog = tl.load(Edge+e*H+h, valid, other=-float("inf")).to(tl.float32)
    w = tl.exp(wlog-tl.max(wlog, 0))
    w = w/tl.sum(w, 0)
    msg = tl.load(Msg+((bt*E+e[:, None])*H+h)*M+col[None, :],
                  valid[:, None] & (col[None, :] < M), other=0).to(tl.float32)
    incoming = tl.sum(w[:, None]*msg, 0)
    g = tl.load(Gain+j*H+h).to(tl.float32)
    if PRODUCT:
        t = M*incoming
        if MARGINAL:
            n=tl.load(Counts+row*M+col,col<M,other=0).to(tl.float32)
            t=t-tl.log(1+n)
        g = tl.exp(tl.minimum(4.,tl.maximum(-4.,g)))
    else:
        mean = tl.sum(tl.where(col < M, incoming, 0), 0)/M
        centered = tl.where(col < M, incoming-mean, 0)
        iv = tl.rsqrt(tl.sum(centered*centered, 0)/M+1e-5)
        z = centered*iv
        t = 2/(1+tl.exp(-2*z))-1
    x = tl.load(Base+row*M+col, col < M, other=0).to(tl.float32)+g*t
    x = tl.where(col < M, x, -float("inf"))
    prob = tl.exp(x-tl.max(x, 0))
    prob = prob/tl.sum(prob, 0)
    tl.store(Out+row*M+col, prob, col < M)
    tl.store(P+row*M+col, prob, col < M)


@triton.jit
def _route_bwd(Msg, Edge, Gain, Counts, P, DO, DB, DM, DEpart, DGpart, DC,
               H: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
               E: tl.constexpr, BS: tl.constexpr, BM: tl.constexpr, PRODUCT: tl.constexpr, MARGINAL: tl.constexpr):
    row = tl.program_id(0)
    h = row % H
    j = (row // H) % L
    bt = row // (H*L)
    src = tl.arange(0, BS)
    col = tl.arange(0, BM)
    valid = (src < L) & (src != j)
    e = src*(L-1)+tl.where(j > src, j-1, j)
    wlog = tl.load(Edge+e*H+h, valid, other=-float("inf")).to(tl.float32)
    w = tl.exp(wlog-tl.max(wlog, 0))
    w = w/tl.sum(w, 0)
    offsets = ((bt*E+e[:, None])*H+h)*M+col[None, :]
    mask = valid[:, None] & (col[None, :] < M)
    msg = tl.load(Msg+offsets, mask, other=0).to(tl.float32)
    incoming = tl.sum(w[:, None]*msg, 0)
    raw_g = tl.load(Gain+j*H+h).to(tl.float32)
    g = raw_g
    if PRODUCT:
        t = M*incoming
        if MARGINAL:
            n=tl.load(Counts+row*M+col,col<M,other=0).to(tl.float32)
            t=t-tl.log(1+n)
        g = tl.exp(tl.minimum(4.,tl.maximum(-4.,g)))
    else:
        mean = tl.sum(tl.where(col < M, incoming, 0), 0)/M
        centered = tl.where(col < M, incoming-mean, 0)
        iv = tl.rsqrt(tl.sum(centered*centered, 0)/M+1e-5)
        z = centered*iv
        t = 2/(1+tl.exp(-2*z))-1
    p = tl.load(P+row*M+col, col < M, other=0)
    dy = tl.load(DO+row*M+col, col < M, other=0).to(tl.float32)
    dx = p*(dy-tl.sum(p*dy, 0))
    tl.store(DB+row*M+col, dx, col < M)
    if PRODUCT:
        tl.store(DGpart+row, tl.sum(dx*t, 0)*g*((raw_g>=-4.) & (raw_g<=4.)))
        di = dx*g*M
        if MARGINAL:
            tl.store(DC+row*M+col,-dx*g/(1+n),col<M)
    else:
        tl.store(DGpart+row, tl.sum(dx*t, 0))
        dz = tl.where(col < M, dx*g*(1-t*t), 0)
        di = iv*(dz-tl.sum(dz, 0)/M-z*tl.sum(dz*z, 0)/M)
    di = tl.where(col < M, di, 0)
    tl.store(DM+offsets, w[:, None]*di[None, :], mask)
    dw = tl.sum(msg*di[None, :], 1)
    de = w*(dw-tl.sum(w*dw, 0))
    tl.store(DEpart+(bt*E+e)*H+h, de, valid)


class _DenseRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, messages, edge_logits, gain, product,counts):
        b, t, levels, heads, m = base.shape
        edges = levels*(levels-1)
        if not base.is_cuda or levels < 2:
            raise ValueError("dense_route requires CUDA and >=2 levels")
        if base.dtype != torch.float32:
            raise ValueError("dense_route uses FP32 logits; use the torch reference for double checks")
        if messages.shape != (b, t, edges, heads, m):
            raise ValueError("messages must contain every ordered non-self edge")
        if edge_logits.shape != (edges, heads) or gain.shape != (levels, heads):
            raise ValueError("edge/gain dimensions do not match")
        marginal=counts is not None
        if marginal and (not product or counts.shape!=base.shape):
            raise ValueError("marginal correction needs product read and counts matching base")
        counts=counts.contiguous() if marginal else base.new_empty(0)
        base, messages = base.contiguous(), messages.contiguous()
        edge_logits, gain = edge_logits.contiguous(), gain.contiguous()
        out = torch.empty_like(base)
        p = torch.empty_like(base, dtype=torch.float32)
        _route_fwd[(b*t*levels*heads,)](
            base, messages, edge_logits, gain, counts, out, p, heads, levels, m, edges,
            triton.next_power_of_2(levels), triton.next_power_of_2(m), product,marginal, num_warps=4)
        ctx.save_for_backward(messages, edge_logits, gain, counts, p)
        ctx.shape = (b, t, levels, heads, m)
        ctx.product = product
        ctx.marginal=marginal
        return out

    @staticmethod
    def backward(ctx, dout):
        messages, edge, gain, counts, p = ctx.saved_tensors
        b, t, levels, heads, m = ctx.shape
        edges = levels*(levels-1)
        db = torch.empty_like(p)
        dm = torch.empty_like(messages)
        de = torch.empty(b*t, edges, heads, device=p.device, dtype=torch.float32)
        dg = torch.empty(b*t, levels, heads, device=p.device, dtype=torch.float32)
        dc=torch.empty_like(counts)
        _route_bwd[(b*t*levels*heads,)](
            messages, edge, gain, counts, p, dout.contiguous(), db, dm, de, dg, dc,
            heads, levels, m, edges, triton.next_power_of_2(levels),
            triton.next_power_of_2(m), ctx.product,ctx.marginal, num_warps=4)
        return db, dm, de.sum(0).to(edge.dtype), dg.sum(0).to(gain.dtype), None, dc if ctx.marginal else None


def dense_route(base, messages, edge_logits, gain, product=False,counts=None):
    return _DenseRoute.apply(base, messages, edge_logits, gain, product,counts)
