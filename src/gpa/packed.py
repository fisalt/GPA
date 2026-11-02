"""Pack independent level/edge scans into the head axis; same lagged blocks.

Three GDN launches per block (inner, all R edges, outer), rather than
2*L + L*(L-1). No level/edge is removed, and no block boundary changes.
"""
import torch
from .ops import gdn_scan, linear_scan, rope, unit
from .state import GPAState, LevelState, RelationState, ResidualSnapshot


def pack_sequence(xs):
    x=torch.stack(xs,2)
    return x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],*x.shape[4:])


def dense_route_reference(base, messages, edge_logits, gain, edges, product=False,counts=None):
    """[B,T,L,H,M], [B,T,E,H,M]. FP32 relation normalization/softmax."""
    dtype = torch.float64 if base.dtype == torch.float64 else torch.float32
    routes = []
    for j in range(base.shape[2]):
        ids = [e for e, (_, target) in enumerate(edges) if target == j]
        logits = base[:, :, j].to(dtype)
        if ids:
            weights = edge_logits[ids].to(dtype).softmax(0)
            incoming = (messages[:, :, ids].to(dtype)*weights[None, None, :, :, None]).sum(2)
            if product:
                logits=logits+base.shape[-1]*gain[j].to(dtype).clamp(-4,4).exp()[None,None,:,None]*incoming
                if counts is not None:
                    logits=logits-gain[j].to(dtype).clamp(-4,4).exp()[None,None,:,None]*counts[:,:,j].to(dtype).log1p()
            else:
                centered = incoming-incoming.mean(-1, keepdim=True)
                normalized = centered*(centered.square().mean(-1, keepdim=True)+1e-5).rsqrt()
                logits = logits + gain[j].to(dtype)[None, None, :, None]*normalized.tanh()
        routes.append(logits.softmax(-1))
    return torch.stack(routes, 2)


def block_segment(model, q, k, v, state, positions,control_input=None):
    cfg = model.config
    bsz, length, heads, kd = q.shape
    levels, m, vd = len(cfg.slots), cfg.slots[0], v.shape[-1]
    packed_h = levels*heads
    frozen = state.frozen or tuple(ResidualSnapshot(x.inner, x.outer) for x in state.levels)
    qr = rope(q, positions) if cfg.positional_mode == "moments" else q
    key, value = k, v
    keys, values, slots, writes, counts, gates = [], [], [], [], [], []
    deltas = []
    for j, (old, snap) in enumerate(zip(state.levels, frozen)):
        slot = model._slot(key, j)
        keys.append(key)
        values.append(value)
        slots.append(slot)
        gates.append(model._gates(key, j,control_input))
        write = model._assignment(slot)
        if write is None:write=torch.einsum("bthd,bhdm->bthm", unit(key).to(snap.inner.dtype), snap.inner).softmax(-1)
        write = write.to(q.dtype)
        writes.append(write)
        counts.append(old.count[:, None]+write.to(old.count.dtype).cumsum(1))
        if old.position is not None:
            kh = unit(key)
            deltas.append(rope(kh, positions)-kh)
        rk = torch.einsum("bhdm,bthm->bthd", snap.inner, write.to(snap.inner.dtype))
        rv = torch.einsum("bthm,bhmv->bthv", write.to(snap.outer.dtype), snap.outer)
        key, value = model._residual(key, value, rk, rv, j)

    def pack(xs):
        x = torch.stack(xs, 2)
        return x.reshape(bsz, length, packed_h, *x.shape[4:])

    def pack_state(xs):
        x = torch.stack(xs, 1)
        return x.reshape(bsz, packed_h, *x.shape[3:])

    qp = qr[:, :, None].expand(bsz, length, levels, heads, kd).reshape(bsz, length, packed_h, kd)
    kp, sp, vp, wp = pack(keys), pack(slots), pack(values), pack(writes)
    gi, bi, go, bo = [pack([g[n] for g in gates]) for n in range(4)]
    raw, afinal = gdn_scan(qp, kp, sp, gi, bi, pack_state([s.inner for s in state.levels]), backend=cfg.backend)
    raw = raw.reshape(bsz, length, levels, heads, m)
    afinal = afinal.reshape(bsz, levels, heads, kd, m)
    moments, corrections = None, None
    if deltas:
        corrections, moments = linear_scan(unit(qp), pack(deltas), wp,
                                           pack_state([s.position for s in state.levels]), backend=cfg.backend)
        corrections = corrections.reshape(bsz, length, levels, heads, m)
        moments = moments.reshape(bsz, levels, heads, kd, m)
    base = torch.stack([model._base_logits(raw[:, :, j], counts[j],
                        corrections[:, :, j] if corrections is not None else None, j)
                        for j in range(levels)], 2)
    return read_and_output(model,q,qr,base,wp.reshape(bsz,length,levels,heads,m),
                           sp,vp,go,bo,state,afinal,counts,moments,frozen)


def read_and_output(model,q,qr,base,write_tensor,sp,vp,go,bo,state,afinal,counts,
                    moments=None,frozen=None):
    """Common all-edge read/outer pass for both schedules; identical formulas."""
    cfg=model.config
    bsz,length,heads,kd=q.shape
    levels,m,vd=len(cfg.slots),cfg.slots[0],vp.shape[-1]
    packed_h=levels*heads
    def pack_state(xs):
        x=torch.stack(xs,1)
        return x.reshape(bsz,packed_h,*x.shape[3:])
    local = base.softmax(-1).to(q.dtype)
    if model.edges:
        source, target = model._edge_source, model._edge_target
        edges = len(model.edges)
        eh = edges*heads
        eq = local.index_select(2, source).reshape(bsz, length, eh, m)
        ek = write_tensor.index_select(2, source).reshape(bsz, length, eh, m)
        ev = write_tensor.index_select(2, target).reshape(bsz, length, eh, m)
        rg = -torch.nn.functional.softplus(model.relation_decay_raw)
        if cfg.positional_mode == "none":
            rg = rg*0
        rg = rg.reshape(1, 1, eh).expand(bsz, length, eh)
        rb = model.relation_beta_raw.sigmoid().reshape(1, 1, eh).expand(bsz, length, eh)
        if cfg.rsa_update:
            norm=(ek.to(torch.float64 if ek.dtype==torch.float64 else torch.float32).square().sum(-1)+1e-6).sqrt()
            ev=(ev/norm[...,None]).to(q.dtype)
            rb=(rb*norm.square()).clamp_max(.9)
        r0 = torch.stack([s.memory for s in state.relations], 1).reshape(bsz, eh, m, m)
        messages, rfinal = gdn_scan(eq, ek, ev, rg, rb, r0, backend=cfg.backend)
        messages = messages.reshape(bsz, length, edges, heads, m)
        if cfg.paper_product_read:
            messages=messages*local.index_select(2,source).norm(dim=-1,keepdim=True)
        rfinal = rfinal.reshape(bsz, edges, heads, m, m)
        relations = tuple(RelationState(i, j, rfinal[:, e]) for e, (i, j) in enumerate(model.edges))
    else:
        messages = q.new_empty(bsz, length, 0, heads, m)
        relations = ()
    if cfg.execution in ("triton","staged") and q.is_cuda and levels > 1:
        from .triton_route import dense_route
        route = dense_route(base, messages, model.edge_logits, model.relation_gain,product=cfg.paper_product_read,
                            counts=torch.stack(counts,2) if cfg.relation_marginal else None).to(q.dtype)
    else:
        route = dense_route_reference(base, messages, model.edge_logits, model.relation_gain, model.edges,product=cfg.paper_product_read,
                                      counts=torch.stack(counts,2) if cfg.relation_marginal else None).to(q.dtype)
    up, down = torch.stack(tuple(model.route_up)), torch.stack(tuple(model.route_down))
    hidden = torch.nn.functional.gelu(torch.einsum("btlhm,lhmr->btlhr", route, up.to(q.dtype)))
    route = route+model.route_mlp_gain[None, None, :, :, None]*torch.einsum(
        "btlhr,lhrm->btlhm", hidden, down.to(q.dtype))
    route = route.to(q.dtype)
    if cfg.rsa_update:
        stable=route.to(torch.float64 if route.dtype==torch.float64 else torch.float32)
        route=(stable/stable.abs().sum(-1,keepdim=True).clamp(.25,4.)).to(q.dtype)
    rw, qw = torch.stack(tuple(model.route_weight)), torch.stack(tuple(model.query_weight))
    if cfg.shared_query_center:
        qw = qw+torch.stack(tuple(model.center)).transpose(-1, -2)/cfg.query_scale
    qo = torch.einsum("btlhm,lhmn->btlhn", route, rw.to(q.dtype))
    direct=cfg.query_scale*torch.einsum("bthd,lhdm->btlhm", qr, qw.to(q.dtype))
    direct=model._slot_feature(direct)
    if cfg.balanced_bridge:
        mix=.1+.8*model.bridge_mix_raw.sigmoid()[None,None,:,:,None]
        qo=mix*unit(qo)+(1-mix)*unit(direct)
    else:qo=qo+direct
    out, bfinal = gdn_scan(qo.to(q.dtype).reshape(bsz, length, packed_h, m), sp, vp, go, bo,
                           pack_state([s.outer for s in state.levels]), backend=cfg.backend)
    out = out.reshape(bsz, length, levels, heads, vd)
    bfinal = bfinal.reshape(bsz, levels, heads, m, vd)
    out = (out*model.output_logits.softmax(0)[None, None, :, :, None]).sum(2)
    final_levels = tuple(LevelState(afinal[:, j], bfinal[:, j], counts[j][:, -1],
                                   moments[:, j] if moments is not None else None) for j in range(levels))
    offset = (state.chunk_offset+length) % cfg.chunk_size
    new = GPAState(final_levels, relations, state.position+length, cfg.chunk_size,
                   offset, frozen if offset else None)
    return out.to(q.dtype), new, list(route.unbind(2))
