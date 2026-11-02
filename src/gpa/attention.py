"""Dense, snapshot-based multilevel GPA; reference and optional FLA scans.

Residual storage is hierarchical; READS are all-to-all, not a triangular
previous-level gate. The recurrent definition and block-lagged approximation
are intentionally separate execution modes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .ops import gdn_update, gdn_scan, linear_scan, memory_read, rope, state_dtype, unit
from .state import GPAState, LevelState, RelationState, ResidualSnapshot


@dataclass(frozen=True)
class GPAConfig:
    heads: int = 4
    key_dim: int = 32
    value_dim: int = 32
    slots: tuple[int, ...] = (16, 16, 16, 16)
    positional_mode: str = "gated"  # gated | moments | none
    query_scale: float = 1 / 16
    beta_init: float = .1
    beta_max: float = .9
    decay_init: float = .001
    relation_beta_init: float = .6
    relation_decay_init: float = .0001
    chunk_size: int = 1
    backend: str = "torch"  # fla is the public CUDA chunk GDN backend
    execution: str = "reference"  # reference | packed | triton | staged
    read_scale: float = 1.  # content-logit inverse temperature, retrieval knob
    shared_query_center: bool = False  # W_qs = C^T/query_scale + learned delta
    rsa_update: bool = False  # paired assignment-normalized R, tiered controls
    paper_product_read: bool = False
    control_dim: int = 0  # optional full hidden input, separate from projected K
    slot_feature: str = "linear"  # linear | softmax | tanh; symmetric feature map
    slot_scale: float = 4.
    balanced_bridge: bool = False
    relation_marginal: bool = False
    quadratic_features: bool = False
    learn_read_scale: bool = False

    def __post_init__(self):
        if min(self.heads, self.key_dim, self.value_dim, self.chunk_size) <= 0:
            raise ValueError("dimensions and chunk_size must be positive")
        if self.control_dim<0:
            raise ValueError("control_dim must be nonnegative")
        if self.slot_feature not in ("linear","softmax","tanh") or self.slot_scale<=0:
            raise ValueError("slot_feature must be linear/softmax/tanh, slot_scale positive")
        if self.relation_marginal and not self.paper_product_read:
            raise ValueError("relation marginal correction needs product reading")
        if not self.slots or min(self.slots) <= 0:
            raise ValueError("slots must contain positive capacities")
        if self.positional_mode not in ("gated", "moments", "none"):
            raise ValueError("unknown positional_mode")
        if self.positional_mode == "moments" and self.key_dim % 2:
            raise ValueError("positional moments require even key_dim")
        if not 0 < self.beta_init < self.beta_max <= 1:
            raise ValueError("require 0 < beta_init < beta_max <= 1")
        if not 0 < self.relation_beta_init < 1:
            raise ValueError("relation_beta_init must be in (0,1)")
        if self.decay_init <= 0 or self.relation_decay_init <= 0:
            raise ValueError("initial decays must be positive")
        if self.backend not in ("torch", "fla"):
            raise ValueError("backend must be torch or fla")
        if self.execution not in ("reference", "packed", "triton","staged"):
            raise ValueError("execution must be reference, packed, triton, or staged")
        if self.read_scale <= 0:
            raise ValueError("read_scale must be positive")
        if self.shared_query_center and self.query_scale <= 0:
            raise ValueError("shared query-center parameterization needs query_scale > 0")
        if self.backend == "fla" and self.chunk_size == 1:
            raise ValueError("FLA is a chunk backend: select chunk_size > 1")


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class DenseGPA(nn.Module):
    """Projected Q/K/V -> output, cache. Inputs [B,T,H,D], output [B,T,H,Dv].

    This is an attention core, not an entire pretrained LM. Projection layers,
    padding/packing, and an LM generation adapter belong to the caller.
    All ordered cross-level pairs own a full, trainable-prefix GDN relation.
    No cache is detached, even between calls, unless state.detach() is called.
    """

    def __init__(self, config: GPAConfig):
        super().__init__()
        self.config = config
        h, d, dv = config.heads, config.key_dim, config.value_dim
        self.center = nn.ParameterList()
        self.inner_prefix = nn.ParameterList()
        self.outer_prefix = nn.ParameterList()
        self.route_weight = nn.ParameterList()
        self.query_weight = nn.ParameterList()
        self.route_up = nn.ParameterList()
        self.route_down = nn.ParameterList()
        self.gate_weight = nn.ParameterList()
        self.gate_bias = nn.ParameterList()
        for m in config.slots:
            self.center.append(nn.Parameter(torch.randn(h, m, d) / math.sqrt(d)))
            self.inner_prefix.append(nn.Parameter(torch.randn(h, d, m) * .002))
            self.outer_prefix.append(nn.Parameter(torch.randn(h, m, dv) * .002))
            self.route_weight.append(nn.Parameter(torch.eye(m).repeat(h, 1, 1)))
            self.query_weight.append(nn.Parameter(torch.randn(h, d, m) / math.sqrt(d)))
            self.route_up.append(nn.Parameter(torch.randn(h, m, 2*m) / math.sqrt(m)))
            self.route_down.append(nn.Parameter(torch.randn(h, 2*m, m) * .002))
            self.gate_weight.append(nn.Parameter(torch.zeros(h, config.control_dim or d, 4)))
            b = torch.tensor([_inverse_softplus(config.decay_init),
                              math.log(config.beta_init / (config.beta_max-config.beta_init))] * 2)
            self.gate_bias.append(nn.Parameter(b.repeat(h, 1)))
        levels = len(config.slots)
        self.residual_raw = nn.Parameter(torch.zeros(levels, h, 2))
        self.count_gain_raw = nn.Parameter(torch.full((levels, h), _inverse_softplus(.1)))
        self.moment_gain = nn.Parameter(torch.full((levels, h), .1))
        self.route_mlp_gain = nn.Parameter(torch.full((levels, h), .1))
        self.output_logits = nn.Parameter(torch.zeros(levels, h))
        self.edges = tuple((i, j) for i in range(levels) for j in range(levels) if i != j)
        # Construct constants outside forward: CPU-list -> CUDA copies are not
        # capture-safe. Nonpersistent buffers preserve old state_dict keys.
        self.register_buffer("_edge_source", torch.tensor([i for i, _ in self.edges], dtype=torch.long), persistent=False)
        self.register_buffer("_edge_target", torch.tensor([j for _, j in self.edges], dtype=torch.long), persistent=False)
        self.relation_prefix = nn.ParameterList([
            nn.Parameter(torch.zeros(h, config.slots[i], config.slots[j])) for i, j in self.edges
        ])
        self.edge_logits = nn.Parameter(torch.zeros(len(self.edges), h))
        self.relation_gain = nn.Parameter(torch.full((levels, h), .1))
        self.relation_decay_raw = nn.Parameter(torch.full((len(self.edges), h),
                                                          _inverse_softplus(config.relation_decay_init)))
        rb = math.log(config.relation_beta_init / (1-config.relation_beta_init))
        self.relation_beta_raw = nn.Parameter(torch.full((len(self.edges), h), rb))
        if config.balanced_bridge:
            self.bridge_mix_raw=nn.Parameter(torch.zeros(levels,h))
        if config.quadratic_features:
            from .features import QuadraticFeatures
            self.feature_map=QuadraticFeatures(h,d)
        if config.learn_read_scale:
            self.read_log_scale=nn.Parameter(torch.full((levels,h),math.log(config.read_scale)))
        probe_block=64 if config.backend=="fla" else config.chunk_size
        self.register_buffer("_snapshot_basis",F.pad(torch.eye(d),(0,0,0,(-d)%probe_block)),persistent=False)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> GPAState:
        ref = self.center[0]
        device = ref.device if device is None else device
        dtype = ref.dtype if dtype is None else dtype
        work = torch.float64 if dtype == torch.float64 else torch.float32
        expand = lambda p: p.to(device=device, dtype=work).unsqueeze(0).expand(batch, *p.shape)
        levels = tuple(LevelState(
            expand(a), expand(b), torch.zeros(batch, self.config.heads, m, device=device, dtype=work),
            torch.zeros(batch, self.config.heads, self.config.key_dim, m, device=device, dtype=work)
            if self.config.positional_mode == "moments" else None,
        ) for m, a, b in zip(self.config.slots, self.inner_prefix, self.outer_prefix))
        relations = tuple(RelationState(i, j, expand(p))
                          for (i, j), p in zip(self.edges, self.relation_prefix))
        return GPAState(levels, relations, chunk_size=self.config.chunk_size)

    def _gates(self, key: Tensor, level: int, control_input=None):
        if self.config.control_dim:
            raw=torch.einsum("...d,hdc->...hc",control_input,self.gate_weight[level].to(control_input.dtype))
        else:
            raw = torch.einsum("...hd,hdc->...hc", key if self.config.rsa_update else unit(key), self.gate_weight[level].to(key.dtype))
        raw = raw.to(state_dtype(key)) + self.gate_bias[level]
        gi, bi, go, bo = raw.unbind(-1)
        gi, go = -F.softplus(gi).clamp(max=10.), -F.softplus(go).clamp(max=10.)
        if self.config.rsa_update:
            tier=(1.,.1,.01,.001)[level] if len(self.config.slots)==4 else .1**level
            gi,go=gi*tier,go*tier
        if self.config.positional_mode == "none":
            gi, go = gi * 0, go * 0
        return gi, self.config.beta_max * bi.sigmoid(), go, self.config.beta_max * bo.sigmoid()

    def _slot(self, k: Tensor, level: int):
        return self._slot_feature(torch.einsum("...hd,hmd->...hm", k, self.center[level].to(k.dtype)))

    def _slot_feature(self, projected):
        if self.config.slot_feature=="softmax":
            return (self.config.slot_scale*projected.to(state_dtype(projected))).softmax(-1).to(projected.dtype)
        if self.config.slot_feature=="tanh":
            return (self.config.slot_scale*projected.to(state_dtype(projected))).tanh().to(projected.dtype)
        return projected

    def _assignment(self, slot):
        if self.config.slot_feature=="softmax":return slot
        return (slot/.25).softmax(-1) if self.config.rsa_update else None

    def _residual(self, key, value, reconstruction_k, reconstruction_v, level):
        # Bounded learned step sizes; storage never depends on Q.
        eta = 1 + .25 * self.residual_raw[level].tanh()
        return ((key - eta[:, 0, None] * reconstruction_k.to(key.dtype)).to(key.dtype),
                (value - eta[:, 1, None] * reconstruction_v.to(value.dtype)).to(value.dtype))

    def _base_logits(self, raw, count, moment, level):
        gain = F.softplus(self.count_gain_raw[level])[..., None]
        scale=self.config.read_scale
        if self.config.learn_read_scale:
            scale=self.read_log_scale[level].clamp(math.log(.25),math.log(64.)).exp()[...,None]
        logits = scale*raw.to(count.dtype) + gain * count.log1p()
        if moment is not None:
            logits = logits + self.moment_gain[level][..., None] * torch.tanh(moment / (1 + count))
        return logits

    def _relations(self, local, writes, state, backend):
        messages, new_states = [[] for _ in local], []
        for e, ((i, j), old) in enumerate(zip(self.edges, state)):
            shape = writes[i].shape[:-1]
            g = -F.softplus(self.relation_decay_raw[e]).expand(shape)
            if self.config.positional_mode == "none":
                g = g * 0
            beta = self.relation_beta_raw[e].sigmoid().expand(shape)
            value=writes[j]
            if self.config.rsa_update:
                norm=(writes[i].to(state_dtype(writes[i])).square().sum(-1)+1e-6).sqrt()
                value=(value/norm[...,None]).to(value.dtype)
                beta=(beta*norm.square()).clamp_max(.9)
            msg, memory = gdn_scan(local[i], writes[i], value, g, beta, old.memory,
                                   backend=backend)
            messages[j].append((e, msg))
            new_states.append(RelationState(i, j, memory))
        return messages, tuple(new_states)

    def _bridge(self, q, logits, messages, counts=None):
        routes, qouts = [], []
        for j, base in enumerate(logits):
            if messages[j]:
                ids = [e for e, _ in messages[j]]
                weights = self.edge_logits[ids].softmax(0)
                if self.config.paper_product_read:
                    incoming=sum(w[...,None]*msg*logits[self.edges[e][0]].softmax(-1).norm(dim=-1,keepdim=True)
                                 for w,(e,msg) in zip(weights,messages[j]))
                    base=base+base.shape[-1]*self.relation_gain[j].clamp(-4,4).exp()[...,None]*incoming
                    if self.config.relation_marginal:
                        base=base-self.relation_gain[j].clamp(-4,4).exp()[...,None]*counts[j].log1p()
                else:
                    incoming = sum(w[..., None] * msg for w, (_, msg) in zip(weights, messages[j]))
                    centered = incoming - incoming.mean(-1, keepdim=True)
                    normalized = centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
                    base = base + self.relation_gain[j][..., None] * normalized.tanh()
            route = base.softmax(-1).to(q.dtype)
            hidden = F.gelu(torch.einsum("...hm,hmr->...hr", route, self.route_up[j].to(q.dtype)))
            route = route + self.route_mlp_gain[j][..., None] * torch.einsum(
                "...hr,hrm->...hm", hidden, self.route_down[j].to(q.dtype))
            route = route.to(q.dtype)
            if self.config.rsa_update:
                stable=route.to(state_dtype(route))
                route=(stable/stable.abs().sum(-1,keepdim=True).clamp(.25,4.)).to(q.dtype)
            qo = torch.einsum("...hm,hmn->...hn", route, self.route_weight[j].to(q.dtype))
            qw = self.query_weight[j]
            if self.config.shared_query_center:
                qw = qw+self.center[j].transpose(-1, -2)/self.config.query_scale
            direct=self.config.query_scale*torch.einsum("...hd,hdm->...hm", q, qw.to(q.dtype))
            direct=self._slot_feature(direct)
            if self.config.balanced_bridge:
                mix=.1+.8*self.bridge_mix_raw[j].sigmoid()[...,None]
                qo=mix*unit(qo)+(1-mix)*unit(direct)
            else:qo=qo+direct
            routes.append(route)
            qouts.append(qo.to(q.dtype))
        return routes, qouts

    def _token(self, q, k, v, state, position, control_input=None):
        qr = rope(q, position) if self.config.positional_mode == "moments" else q
        key, value = k, v
        levels, writes, logits = [], [], []
        for j, old in enumerate(state.levels):
            slot = self._slot(key, j)
            gi, bi, go, bo = self._gates(key, j, control_input)
            a = gdn_update(old.inner, key, slot, gi, bi)
            b = gdn_update(old.outer, slot, value, go, bo)
            write = self._assignment(slot)
            if write is None:write=memory_read(key,a).softmax(-1)
            count = old.count + write.to(old.count.dtype)
            moment_state, correction = old.position, None
            if moment_state is not None:
                kh = unit(key)
                delta = (rope(kh, position) - kh).to(moment_state.dtype)
                moment_state = moment_state + delta[..., :, None] * write.to(delta.dtype)[..., None, :]
                correction = memory_read(qr, moment_state)
            raw = memory_read(qr, a)
            logits.append(self._base_logits(raw, count, correction, j))
            writes.append(write)
            levels.append(LevelState(a, b, count, moment_state))
            rk = torch.einsum("bhdm,bhm->bhd", a, write.to(a.dtype))
            rv = torch.einsum("bhm,bhmv->bhv", write.to(b.dtype), b)
            key, value = self._residual(key, value, rk, rv, j)
        local = [x.softmax(-1).to(q.dtype).unsqueeze(1) for x in logits]
        messages, relations = self._relations(local, [x.unsqueeze(1) for x in writes], state.relations, "torch")
        messages = [[(e, m[:, 0]) for e, m in group] for group in messages]
        routes, qouts = self._bridge(qr, logits, messages,[x.count for x in levels])
        weights = self.output_logits.softmax(0)
        out = sum(weights[j][..., None] * memory_read(qo, levels[j].outer)
                  for j, qo in enumerate(qouts))
        new = GPAState(tuple(levels), relations, state.position+1, self.config.chunk_size)
        return out.to(v.dtype), new, routes

    def _block_segment(self, q, k, v, state, positions, control_input=None):
        """FLA-compatible lagged residuals, with fixed block boundaries in cache."""
        if self.config.execution != "reference" and len(set(self.config.slots)) == 1:
            from .packed import block_segment
            return block_segment(self, q, k, v, state, positions, control_input)
        frozen = state.frozen or tuple(ResidualSnapshot(x.inner, x.outer) for x in state.levels)
        qr = rope(q, positions) if self.config.positional_mode == "moments" else q
        backend = self.config.backend
        key, value = k, v
        levels, writes, logits, outer_inputs, count_history = [], [], [], [], []
        for j, (old, snap) in enumerate(zip(state.levels, frozen)):
            slot = self._slot(key, j)
            gi, bi, go, bo = self._gates(key, j, control_input)
            raw, a = gdn_scan(qr, key, slot, gi, bi, old.inner, backend=backend)
            write = self._assignment(slot)
            if write is None:write=torch.einsum("bthd,bhdm->bthm", unit(key).to(snap.inner.dtype), snap.inner).softmax(-1)
            write = write.to(q.dtype)
            counts = old.count[:, None] + write.to(old.count.dtype).cumsum(1)
            count_history.append(counts)
            moment_state, correction = old.position, None
            if moment_state is not None:
                kh = unit(key)
                delta = rope(kh, positions) - kh
                correction, moment_state = linear_scan(unit(qr), delta, write, moment_state, backend=backend)
            logits.append(self._base_logits(raw, counts, correction, j))
            writes.append(write)
            levels.append((a, counts[:, -1], moment_state))
            outer_inputs.append((slot, value, go, bo))
            rk = torch.einsum("bhdm,bthm->bthd", snap.inner, write.to(snap.inner.dtype))
            rv = torch.einsum("bthm,bhmv->bthv", write.to(snap.outer.dtype), snap.outer)
            key, value = self._residual(key, value, rk, rv, j)
        local = [x.softmax(-1).to(q.dtype) for x in logits]
        messages, relations = self._relations(local, writes, state.relations, backend)
        routes, qouts = self._bridge(qr, logits, messages,count_history)
        outputs, final_levels = [], []
        for j, qo in enumerate(qouts):
            slot, vl, go, bo = outer_inputs[j]
            out, b = gdn_scan(qo, slot, vl, go, bo, state.levels[j].outer, backend=backend)
            a, count, moment_state = levels[j]
            final_levels.append(LevelState(a, b, count, moment_state))
            outputs.append(out)
        weights = self.output_logits.softmax(0)
        out = sum(weights[j][..., None] * x for j, x in enumerate(outputs))
        offset = (state.chunk_offset + q.shape[1]) % self.config.chunk_size
        new = GPAState(tuple(final_levels), relations, state.position+q.shape[1],
                       self.config.chunk_size, offset, frozen if offset else None)
        return out.to(v.dtype), new, routes

    def forward(self, q: Tensor, k: Tensor, v: Tensor, state: GPAState | None = None,
                *, position_ids: Tensor | None = None, return_diagnostics: bool = False,control_input=None):
        cfg = self.config
        if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
            raise ValueError("expected Q,K [B,T,H,Dk], V [B,T,H,Dv]")
        if q.shape[2:] != (cfg.heads, cfg.key_dim) or v.shape[-1] != cfg.value_dim:
            raise ValueError("input heads/dimensions do not match GPAConfig")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q,K,V must share device and floating dtype")
        if cfg.control_dim and (control_input is None or control_input.shape!=q.shape[:2]+(cfg.control_dim,)):
            raise ValueError("full hidden controls [B,T,control_dim] required")
        if cfg.quadratic_features:
            q,k=self.feature_map(torch.stack((q,k))).unbind(0)
        if state is None:
            state = self.initial_state(q.shape[0], device=q.device, dtype=q.dtype)
        if state.chunk_size != cfg.chunk_size:
            raise ValueError("cannot switch cache between recurrent and lagged chunk semantics")
        if len(state.levels) != len(cfg.slots) or tuple((s.source, s.target) for s in state.relations) != self.edges:
            raise ValueError("cache topology does not match this model")
        if position_ids is None:
            position_ids = torch.arange(state.position, state.position+q.shape[1], device=q.device)
        if position_ids.shape not in ((q.shape[1],), (q.shape[0], q.shape[1])):
            raise ValueError("position_ids must be [T] or [B,T]")
        staged_eligible=(cfg.execution=="staged" and cfg.chunk_size>1
            and (cfg.backend!="fla" or cfg.chunk_size%64==0)
            and cfg.positional_mode!="moments" and len(set(cfg.slots))==1
            and cfg.key_dim==cfg.slots[0])
        outputs, route_history = [], [[] for _ in cfg.slots]
        start = 0
        while start < q.shape[1]:
            if staged_eligible and state.chunk_offset==0 and q.shape[1]-start>=4*cfg.chunk_size:
                from .staged import staged_sequence
                end=start+(q.shape[1]-start)//cfg.chunk_size*cfg.chunk_size
                out,state,routes=staged_sequence(self,q[:,start:end],k[:,start:end],v[:,start:end],state,
                    control_input[:,start:end] if control_input is not None else None)
                outputs.append(out)
                if return_diagnostics:
                    for history,r in zip(route_history,routes):history.append(r)
                start=end
                continue
            if cfg.chunk_size == 1:
                pos = position_ids[start] if position_ids.ndim == 1 else position_ids[:, start, None]
                out, state, routes = self._token(q[:, start], k[:, start], v[:, start], state, pos,
                                               control_input[:,start] if control_input is not None else None)
                outputs.append(out.unsqueeze(1))
                if return_diagnostics:
                    for history, r in zip(route_history, routes):
                        history.append(r.unsqueeze(1))
                start += 1
            else:
                end = min(q.shape[1], start+cfg.chunk_size-state.chunk_offset)
                pos = position_ids[start:end] if position_ids.ndim == 1 else position_ids[:, start:end]
                out, state, routes = self._block_segment(q[:, start:end], k[:, start:end], v[:, start:end], state, pos,
                                                       control_input[:,start:end] if control_input is not None else None)
                outputs.append(out)
                if return_diagnostics:
                    for history, r in zip(route_history, routes):
                        history.append(r)
                start = end
        output = torch.cat(outputs, 1) if outputs else v.new_empty(v.shape)
        if return_diagnostics:
            # Do not keep time-length route tensors unless explicitly requested.
            routes = tuple(torch.cat(x, 1) if x else q.new_empty(q.shape[:3]+(m,))
                           for x, m in zip(route_history, cfg.slots))
            return output, state, {"routes": routes, "edges": self.edges,
                                   "state_elements": state.numel(), "position": state.position}
        return output, state
