"""Decayed counted KV moments + full paired relations + dense product read.

Not RSA's adjacent/rank32 reader. The normalized centroid update is the
paper writer with a decayed count. Chunk residuals are explicitly lagged;
all moment/relation recurrences themselves are causal and differentiable.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .ops import unit, state_dtype, gdn_scan, _pad_width, _pad_state
from .state import GPAState, LevelState, RelationState, ResidualSnapshot


def moment_scan(q, k, v, g, initial, backend, axis="key"):
    """S=exp(g)S+k v^T; y=q^T S. No hidden L2 normalization."""
    if backend == "fla":
        from fla.ops.simple_gla import chunk_simple_gla
        kd, vd = q.shape[-1], v.shape[-1]
        pk, pv = max(0, 16-kd), max(0, 16-vd)
        if g.ndim == 4:
            if axis=="value":
                from fla.ops.gla import fused_recurrent_gla
                out,final=fused_recurrent_gla(_pad_width(q,pk).contiguous(),_pad_width(k,pk).contiguous(),
                    _pad_width(v,pv).contiguous(),gv=_pad_width(g,pv).float().contiguous(),scale=1.,
                    initial_state=_pad_state(initial.float(),pk,pv).contiguous(),output_final_state=True)
            else:
                from fla.ops.gla import chunk_gla
                out,final=chunk_gla(_pad_width(q,pk).contiguous(),_pad_width(k,pk).contiguous(),
                    _pad_width(v,pv).contiguous(),_pad_width(g,pk).float().contiguous(),scale=1.,
                    initial_state=_pad_state(initial.float(),pk,pv).contiguous(),output_final_state=True)
            return out[...,:vd],final[...,:kd,:vd]
        out, final = chunk_simple_gla(
            _pad_width(q, pk).contiguous(), _pad_width(k, pk).contiguous(),
            _pad_width(v, pv).contiguous(), g=g.float().contiguous(), scale=1.,
            initial_state=_pad_state(initial.float(), pk, pv).contiguous(), output_final_state=True)
        return out[..., :vd], final[..., :kd, :vd]
    if backend != "torch":
        raise ValueError(backend)
    s, output = initial, []
    for t in range(q.shape[1]):
        decay=g[:,t].exp()
        decay=decay[...,None,None] if g.ndim==3 else decay[...,None,:] if axis=="value" else decay[..., :,None]
        s = decay*s+k[:, t].to(s.dtype)[..., :, None]*v[:, t].to(s.dtype)[..., None, :]
        output.append(torch.einsum("bhk,bhkv->bhv", q[:, t].to(s.dtype), s).to(q.dtype))
    return torch.stack(output, 1), s


def decayed_counts(p, g, initial):
    # Bounded fixed chunks avoid exp-overflow of whole-sequence inverse gates.
    if g.ndim==p.ndim:
        cumulative=g.cumsum(1)
        # Stable even for sharp slot gates: never form exp(-cumulative).
        added=torch.logcumsumexp(p.to(initial.dtype).clamp_min(1e-30).log()-cumulative,dim=1)
        return (cumulative+torch.logaddexp(initial.clamp_min(1e-30).log()[:,None],added)).exp()
    cumulative = g.cumsum(1)[..., None]
    return cumulative.exp()*(initial[:, None]+(p.to(initial.dtype)*(-cumulative).exp()).cumsum(1))


@dataclass(frozen=True)
class PaperConfig:
    heads: int = 4
    key_dim: int = 32
    value_dim: int = 32
    slots: tuple = (16, 16, 16, 16)
    chunk_size: int = 64
    backend: str = "torch"
    read_scale: float = 8.
    decay_init: float = .0001
    write_init: float = .6
    relation_beta: float = .6
    use_relations: bool = True
    decay_tiers: tuple | None = None
    normalize_features: bool = True  # legacy-stable ablation; raw dot follows paper
    dense_read: str = "local"  # local | transport (common-space leave-one-out)
    control_dim: int = 0
    slot_decay: bool = False
    strong_relations: bool = False


@dataclass(frozen=True)
class PaperState(GPAState):
    pass


class PaperDenseGPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        if len(set(c.slots)) != 1 or c.chunk_size < 1 or c.backend not in ("torch", "fla"):
            raise ValueError("uniform positive slots/chunk and torch/fla required")
        h, d, dv, m, l = c.heads, c.key_dim, c.value_dim, c.slots[0], len(c.slots)
        self.center = nn.Parameter(torch.randn(l, h, m, d)/math.sqrt(d))
        self.key_prefix = nn.Parameter(torch.randn(l, h, d, m)*.002)
        self.value_prefix = nn.Parameter(torch.randn(l, h, m, dv)*.002)
        self.count_raw = nn.Parameter(torch.full((l, h, m), math.log(math.expm1(.001))))
        self.control_weight = nn.Parameter(torch.zeros(l, h, c.control_dim or d, 2))
        self.control_bias = nn.Parameter(torch.tensor([math.log(math.expm1(c.decay_init)),
                                            math.log(c.write_init/(1-c.write_init))]).repeat(l, h, 1))
        self.decay_log_rate = nn.Parameter(torch.zeros(l,h))
        if c.slot_decay:self.forget_raw=nn.Parameter(torch.full((l,h),math.log(.9/.1)))
        self.feature_log_scale = nn.Parameter(torch.full((l, h), math.log(4.)))
        self.content_log_scale = nn.Parameter(torch.full((l, h), math.log(c.read_scale)))
        self.count_gain = nn.Parameter(torch.ones(l, h))
        self.residual_raw = nn.Parameter(torch.zeros(l, h, 2))
        self.mlp_up = nn.Parameter(torch.randn(l, h, m, 2*m)/math.sqrt(m))
        self.mlp_down = nn.Parameter(torch.zeros(l, h, 2*m, m))
        self.edges = tuple((i,j) for i in range(l) for j in range(l) if i != j)
        e = len(self.edges)
        self.relation_prefix = nn.Parameter(torch.zeros(e, h, m, m))
        self.relation_beta_raw = nn.Parameter(torch.full((e, h), math.log(c.relation_beta/(1-c.relation_beta))))
        self.relation_decay_raw = nn.Parameter(torch.full((e, h), math.log(math.expm1(.0001))))
        self.relation_mix_raw = nn.Parameter(torch.full((l, h), math.log(.1/.9)))
        if c.strong_relations:
            self.read_edge_logits=nn.Parameter(torch.zeros(e,h))
            self.read_local_raw=nn.Parameter(torch.zeros(l,h))
        self.level_logits = nn.Parameter(torch.zeros(l, h))
        self.register_buffer("source", torch.tensor([i for i,j in self.edges], dtype=torch.long), persistent=False)
        self.register_buffer("target", torch.tensor([j for i,j in self.edges], dtype=torch.long), persistent=False)
        self.register_buffer("incoming", torch.tensor([[e for e,(i,target) in enumerate(self.edges) if target==j]
                            for j in range(l)],dtype=torch.long),persistent=False)
        self.register_buffer("reverse_edge",torch.tensor([self.edges.index((j,i)) for i,j in self.edges],dtype=torch.long),persistent=False)
        tiers = c.decay_tiers or ((1.,.1,.01,.001) if l==4 else
                    (1.,.3,.1,.03,.01,.003,.001,.0003) if l==8 else tuple(.1**i for i in range(l)))
        if len(tiers)!=l or any(s<=0 for s in tiers):raise ValueError("one positive decay tier per level required")
        self.register_buffer("time_scales", torch.tensor(tiers).float(), persistent=False)

    def initial_state(self, batch, device, dtype):
        work = torch.float64 if dtype == torch.float64 else torch.float32
        n = F.softplus(self.count_raw).to(device, work)
        a = self.key_prefix.to(device, work)*n[..., None, :]
        b = self.value_prefix.to(device, work)*n[..., :, None]
        levels = tuple(LevelState(a[i][None].expand(batch,-1,-1,-1), b[i][None].expand(batch,-1,-1,-1),
                                  n[i][None].expand(batch,-1,-1)) for i in range(len(self.config.slots)))
        r = self.relation_prefix.to(device, work)
        relations = tuple(RelationState(i,j,r[e][None].expand(batch,-1,-1,-1)) for e,(i,j) in enumerate(self.edges))
        return PaperState(levels, relations, chunk_size=self.config.chunk_size)

    def _segment(self, q, k, v, state, control_input=None):
        c = self.config
        b,t,h,d = q.shape
        l,m,dv = len(c.slots), c.slots[0], c.value_dim
        work = state.levels[0].count.dtype
        feature = unit if c.normalize_features else (lambda x:x)
        frozen = state.frozen or tuple(ResidualSnapshot(s.inner/s.count.clamp_min(1e-5)[...,None,:],
                            s.outer/s.count.clamp_min(1e-5)[..., :,None]) for s in state.levels)
        keys, values, probs, writes, gates, counts = [],[],[],[],[],[]
        for i, snap in enumerate(frozen):
            keys.append(k); values.append(v)
            scale = self.feature_log_scale[i].clamp(math.log(.25), math.log(64.)).exp()
            p = (torch.einsum("bthd,hmd->bthm", feature(k), self.center[i].to(k.dtype)).to(work)*scale[None,None,:,None]).softmax(-1).to(k.dtype)
            ctrl = (torch.einsum("btc,hcf->bthf",control_input,self.control_weight[i].to(control_input.dtype)) if c.control_dim else
                    torch.einsum("bthd,hdc->bthc",feature(k),self.control_weight[i].to(k.dtype)))
            ctrl = ctrl.to(work)+self.control_bias[i]
            g = -(self.decay_log_rate[i].exp()*F.softplus(ctrl[...,0])).clamp(max=.02)*self.time_scales[i]
            w = ctrl[...,1].sigmoid()
            pwrite = (p*w[...,None]).to(q.dtype)
            if c.slot_decay:
                g=g[...,None]+torch.log1p(-(self.forget_raw[i].sigmoid()[None,None,:,None]*pwrite.to(work)).clamp_max(.95))
            probs.append(p); writes.append(pwrite.to(q.dtype)); gates.append(g)
            counts.append(decayed_counts(pwrite,g,state.levels[i].count))
            rk = torch.einsum("bthm,bhdm->bthd", p.to(snap.inner.dtype),snap.inner)
            rv = torch.einsum("bthm,bhmv->bthv", p.to(snap.outer.dtype),snap.outer)
            eta = 1+.25*self.residual_raw[i].tanh()
            k = (k-eta[:,0,None]*rk.to(k.dtype)).to(k.dtype)
            v = (v-eta[:,1,None]*rv.to(v.dtype)).to(v.dtype)
        pack = lambda xs: torch.stack(xs,2).reshape(b,t,l*h,*xs[0].shape[3:])
        pack_state = lambda xs: torch.stack(xs,1).reshape(b,l*h,*xs[0].shape[2:])
        qp = feature(q)[:,:,None].expand(b,t,l,h,d).reshape(b,t,l*h,d)
        raw, af = moment_scan(qp,pack(keys),pack(writes),pack(gates),pack_state([s.inner for s in state.levels]),c.backend,axis="value")
        raw = raw.reshape(b,t,l,h,m)
        n = torch.stack(counts,2).clamp_min(1e-5)
        feature_logits = torch.einsum("bthd,lhmd->btlhm", feature(q),self.center.to(q.dtype)).to(n.dtype)
        feature_logits = feature_logits*self.feature_log_scale.clamp(math.log(.25),math.log(64.)).exp()[None,None,:,:,None]
        logits = self.content_log_scale.clamp(0,math.log(64.)).exp()[None,None,:,:,None]*raw.to(n.dtype)/n+feature_logits
        if c.dense_read == "local":
            logits = logits+self.count_gain.clamp(0,1)[None,None,:,:,None]*n.log()
        local = logits.softmax(-1).to(q.dtype)
        relations = state.relations
        if self.edges and c.use_relations:
            ep = torch.stack(probs,2)
            source = ep.index_select(2,self.source)
            target = ep.index_select(2,self.target)
            norm = (source.to(work).square().sum(-1)+1e-6).sqrt()
            val = target/norm.to(q.dtype)[...,None]
            rb = (self.relation_beta_raw.sigmoid()[None,None]*norm.square()).clamp_max(.9)
            if c.strong_relations:
                activity=torch.stack(writes,2).to(work).sum(-1)
                rb=rb*(activity.index_select(2,self.source)*activity.index_select(2,self.target)).clamp_min(1e-12).sqrt()
            rg = -F.softplus(self.relation_decay_raw)[None,None].expand(b,t,-1,-1)
            e = len(self.edges)
            r0 = torch.stack([s.memory for s in state.relations],1).reshape(b,e*h,m,m)
            rq = local.index_select(2,self.source)
            msg, rf = gdn_scan(rq.reshape(b,t,e*h,m),source.reshape(b,t,e*h,m), val.to(q.dtype).reshape(b,t,e*h,m),
                rg.reshape(b,t,e*h),rb.reshape(b,t,e*h),r0,backend=c.backend)
            msg = msg.reshape(b,t,e,h,m)*rq.to(work).norm(dim=-1,keepdim=True)
            messages = (m*msg.to(work)).log_softmax(-1)
            relation_mix=self.relation_mix_raw.clamp(-6,math.log(16.)).exp() if c.strong_relations else self.relation_mix_raw.sigmoid()
            if c.dense_read == "transport":
                # Align all levels to EACH reference space. Exclude the target's
                # own incoming message before transporting consensus back.
                incoming_total = torch.stack([messages.index_select(2,self.incoming[j]).sum(2)
                                              +local[:,:,j].to(work).clamp_min(1e-8).log() for j in range(l)],2)
                complements = incoming_total.index_select(2,self.source)-messages.index_select(2,self.reverse_edge)
                consensus = complements.softmax(-1)
                root_local = local.index_select(2,self.source).to(work)
                strength = relation_mix[self.source][None,None,:,:,None]
                root_query = ((root_local+strength*consensus)/(1+strength)).to(q.dtype)
                # Same initial R and same writes: this is a second READ of the
                # SAME R_t, not a second state update. Persist rf from pass 1.
                transported,_ = gdn_scan(root_query.reshape(b,t,e*h,m),source.reshape(b,t,e*h,m),
                    val.to(q.dtype).reshape(b,t,e*h,m),rg.reshape(b,t,e*h),rb.reshape(b,t,e*h),r0,backend=c.backend)
                transported = transported.reshape(b,t,e,h,m)*root_query.to(work).norm(dim=-1,keepdim=True)
                transport_logits = (m*transported.to(work)).log_softmax(-1)
                local_mass = local.to(work).clamp_min(1e-8).log()+self.count_gain.clamp(0,1)[None,None,:,:,None]*n.log()
                local_route = local_mass.softmax(-1)
                cross_mass=local_mass
                if c.strong_relations:
                    cross_mass=self.read_local_raw.sigmoid()[None,None,:,:,None]*local.to(work).clamp_min(1e-8).log()+self.count_gain.clamp(0,1)[None,None,:,:,None]*n.log()
                cross = (transport_logits+cross_mass.index_select(2,self.target)).softmax(-1)
                routed=[]
                for j in range(l):
                    incoming=cross.index_select(2,self.incoming[j])
                    if c.strong_relations:
                        weights=self.read_edge_logits.index_select(0,self.incoming[j]).softmax(0)
                        proposal=(incoming*weights[None,None,:,:,None]).sum(2)
                    else:proposal=incoming.mean(2)
                    mix=relation_mix[j][None,None,:,None]
                    routed.append((local_route[:,:,j]+mix*proposal)/(1+mix))
                route=torch.stack(routed,2)
            else:
                routed = []
                for j in range(l):
                    product = messages.index_select(2,self.incoming[j]).sum(2).softmax(-1)
                    mix = relation_mix[j][None,None,:,None]
                    routed.append((local[:,:,j].to(work)+mix*product)/(1+mix))
                route = torch.stack(routed,2)
            rf = rf.reshape(b,e,h,m,m)
            relations = tuple(RelationState(i,j,rf[:,e]) for e,(i,j) in enumerate(self.edges))
        else:
            route = local.to(work) if c.dense_read=="local" else (local.to(work).clamp_min(1e-8).log()+self.count_gain.clamp(0,1)[None,None,:,:,None]*n.log()).softmax(-1)
        delta = torch.einsum("btlhr,lhrm->btlhm",F.gelu(torch.einsum("btlhm,lhmr->btlhr",route.to(q.dtype),self.mlp_up.to(q.dtype))),self.mlp_down.to(q.dtype))
        route = (route.clamp_min(1e-8).log()+.1*delta.to(work)).softmax(-1)
        # Bounded L1 calibration; unlike RSA this is a fully dense positive read.
        route = route/route.abs().sum(-1,keepdim=True).clamp(.25,4.)
        oq = (route/n).to(q.dtype).reshape(b,t,l*h,m)
        out,bf = moment_scan(oq,pack(writes),pack(values),pack(gates),pack_state([s.outer for s in state.levels]),c.backend)
        out = out.reshape(b,t,l,h,dv)
        output = (out*self.level_logits.softmax(0)[None,None,:,:,None]*l).sum(2).to(q.dtype)
        af,bf = af.reshape(b,l,h,d,m),bf.reshape(b,l,h,m,dv)
        states = tuple(LevelState(af[:,i],bf[:,i],counts[i][:,-1]) for i in range(l))
        offset = (state.chunk_offset+t)%c.chunk_size
        new = PaperState(states,relations,state.position+t,c.chunk_size,offset,frozen if offset else None)
        return output,new,route

    def forward(self,q,k,v,state=None,*,return_diagnostics=False,control_input=None):
        if q.ndim!=4 or q.shape!=k.shape or q.shape[:3]!=v.shape[:3] or q.shape[2:]!=(self.config.heads,self.config.key_dim) or v.shape[-1]!=self.config.value_dim:
            raise ValueError("Q/K [B,T,H,Dk], V [B,T,H,Dv] must match config")
        if state is None: state=self.initial_state(q.shape[0],q.device,q.dtype)
        if self.config.control_dim and (control_input is None or control_input.shape!=q.shape[:2]+(self.config.control_dim,)):
            raise ValueError("full hidden controls [B,T,control_dim] required")
        if not isinstance(state,PaperState) or state.chunk_size!=self.config.chunk_size:
            raise ValueError("PaperDenseGPA requires its own matching cache")
        outputs,routes=[],[]
        start=0
        while start<q.shape[1]:
            end=min(q.shape[1],start+self.config.chunk_size-state.chunk_offset)
            y,state,r=self._segment(q[:,start:end],k[:,start:end],v[:,start:end],state,
                                   control_input[:,start:end] if control_input is not None else None)
            outputs.append(y)
            if return_diagnostics: routes.append(r)
            start=end
        y=torch.cat(outputs,1) if outputs else v.new_empty(v.shape)
        if return_diagnostics:
            return y,state,{"routes":torch.cat(routes,1) if routes else q.new_empty(q.shape[:2]+(len(self.config.slots),self.config.heads,self.config.slots[0])),"state_elements":state.numel()}
        return y,state
