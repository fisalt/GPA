"""Small mixed synthetic retrieval benchmark; NOT official RULER/BABILong.

Train/test seeds specify independent random records and answers. No example
lookup table is shared between datasets. UUID-style means a random FOUR-token
identifier value here, not a tokenizer-equivalent natural-language UUID task.
"""
from dataclasses import dataclass
import random
import torch
from torch import nn
import torch.nn.functional as F
from .attention import DenseGPA, GPAConfig

TASKS = ("single", "uuid4", "mqar", "overwrite")
VOCAB = 256
PAIR, VAL, END, QUERY, ANSWER = 1, 2, 3, 4, 5


def make_example(rng, length, task, records, data_version=1):
    if task not in TASKS:
        raise ValueError(task)
    nvalue, nkey = (4, 2) if task == "uuid4" else (1, 1)
    keys = []
    while len(keys) < records:
        key = (rng.randrange(16, 32), rng.randrange(32, 48)) if nkey == 2 else (rng.randrange(16, 48),)
        if key not in keys:
            keys.append(key)
    values = {key: tuple(rng.randrange(64, 80) for _ in range(nvalue)) for key in keys}
    record_streams = [[PAIR, *key, VAL, *values[key], END] for key in keys]
    if task == "overwrite":
        updated_keys=rng.sample(keys,min(4,records)) if data_version>=3 else [rng.choice(keys)]
        for key in updated_keys:
            old_value=values[key]
            values[key] = (rng.randrange(64, 80),)
            if data_version>=2:
                while values[key]==old_value:values[key]=(rng.randrange(64,80),)
            record_streams.append([PAIR, *key, VAL, *values[key], END])
        overwritten_key=rng.choice(updated_keys) if data_version>=3 else updated_keys[0]
    query_keys = rng.sample(keys, min(3, records)) if task == "mqar" else [rng.choice(keys)]
    if task=="overwrite" and data_version>=2:query_keys=[overwritten_key]
    # Queries use only keys; the target values are never inserted before the
    # query except at their actual records. MQAR queries are non-repeating.
    query_size = sum(2+len(key)+nvalue for key in query_keys)
    available = length-query_size-sum(map(len, record_streams))
    if available < 0:
        raise ValueError("sequence too short for this record/query budget")
    boundaries = sorted(rng.randrange(available+1) for _ in record_streams)
    noise = [rng.randrange(128, VOCAB) for _ in range(available)]
    stream, last = [], 0
    for cut, record in zip(boundaries, record_streams):
        stream.extend(noise[last:cut])
        stream.extend(record)
        last = cut
    stream.extend(noise[last:])
    labels, groups = [-100]*length, []
    for key in query_keys:
        stream.extend([QUERY, *key, ANSWER])
        positions = []
        for value in values[key]:
            pos = len(stream)-1
            labels[pos] = value  # causal next-token target
            positions.append(pos)
            stream.append(value)
        groups.append(positions)
    assert len(stream) == length
    return stream, labels, groups


def make_batch(seed, batch, length, records, task=None, device="cpu",data_version=1):
    rng = random.Random(seed)
    examples, labels, metadata = [], [], []
    for _ in range(batch):
        kind = task or rng.choices(TASKS, weights=(.3, .3, .2, .2))[0]
        ids, target, groups = make_example(rng, length, kind, records,data_version)
        examples.append(ids)
        labels.append(target)
        metadata.append({"task": kind, "groups": groups})
    return (torch.tensor(examples, device=device), torch.tensor(labels, device=device), metadata)


@dataclass
class RetrievalConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    slots: tuple = (16, 16, 16, 16)
    chunk_size: int = 64
    convolution: int = 8
    attention: str = "gpa"
    execution: str = "triton"
    backend: str = "fla"
    read_scale: float = 8.
    beta_init: float = .6
    decay_init: float = .0001
    aligned_query: bool = False
    memory_update: str = "legacy"  # legacy | paper_dense
    dense_relations: bool = True
    paper_raw: bool = False
    paper_transport: bool = False
    rsa_update: bool = False
    full_controls: bool = False
    slot_decay: bool = False
    orthogonal_centers: bool = False
    strong_relations: bool = False
    paper_product_read: bool = False
    transformer_rope: bool = False
    slot_feature: str = "linear"
    gated_output: bool = False
    balanced_bridge: bool = False
    context_controls: bool = False
    relation_marginal: bool = False
    quadratic_features: bool = False
    contextual_output: bool = False
    temporal_qk_init: bool = False
    key_dim: int = 0  # zero keeps width/heads; V/output width stays unchanged
    learn_read_scale: bool = False


class RetrievalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.contextual_output and not (cfg.context_controls and cfg.gated_output):
            raise ValueError("contextual output needs the causal controller and gated output")
        width = cfg.width
        key_dim=cfg.key_dim or width//cfg.heads
        key_width=key_dim*cfg.heads
        self.norm = nn.LayerNorm(width)
        if cfg.context_controls:
            self.control_conv=nn.Conv1d(width,width,cfg.convolution,groups=width,bias=False)
            nn.init.constant_(self.control_conv.weight,1/cfg.convolution)
        self.qkv = nn.Linear(width, 2*key_width+width, bias=False)
        self.conv = nn.Conv1d(2*key_width+width, 2*key_width+width, cfg.convolution, groups=2*key_width+width, bias=False)
        with torch.no_grad():
            self.conv.weight.normal_(0, .03)
            self.conv.weight[:, 0, -1].add_(1.)
            if cfg.temporal_qk_init:
                # Generic causal context/order channels, no token IDs or roles.
                dim=key_dim
                for part in range(2):  # Q/K only: retain V's current-token path.
                    for head in range(cfg.heads):
                        start=part*key_width+head*dim
                        self.conv.weight[start:start+dim//2].fill_(cfg.convolution**-.5)
                        for channel in range(dim//2,dim):
                            row=self.conv.weight[start+channel,0]
                            row.zero_();row[-1-(channel-dim//2)%cfg.convolution]=1.
        self.out = nn.Linear(width, width, bias=False)
        if cfg.gated_output:
            self.head_norm_weight=nn.Parameter(torch.ones(cfg.heads,width//cfg.heads))
            self.output_gate=nn.Linear(width,width)
            nn.init.normal_(self.output_gate.weight,std=.02)
            nn.init.ones_(self.output_gate.bias)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
        if cfg.attention == "gpa":
            if cfg.memory_update == "paper_dense":
                from .paper_dense import PaperDenseGPA, PaperConfig
                self.attention = PaperDenseGPA(PaperConfig(heads=cfg.heads,key_dim=key_dim,
                    value_dim=width//cfg.heads,slots=cfg.slots,chunk_size=cfg.chunk_size,backend=cfg.backend,
                    read_scale=cfg.read_scale,decay_init=cfg.decay_init,write_init=cfg.beta_init,
                    use_relations=cfg.dense_relations,normalize_features=not cfg.paper_raw,
                    dense_read="transport" if cfg.paper_transport else "local",
                control_dim=width if cfg.full_controls or cfg.context_controls else 0,slot_decay=cfg.slot_decay,
                    strong_relations=cfg.strong_relations))
            elif cfg.memory_update == "legacy":
                self.attention = DenseGPA(GPAConfig(
                heads=cfg.heads, key_dim=key_dim, value_dim=width//cfg.heads,
                slots=cfg.slots, chunk_size=cfg.chunk_size, backend=cfg.backend,
                execution=cfg.execution, read_scale=cfg.read_scale,
                beta_init=cfg.beta_init, decay_init=cfg.decay_init,
                shared_query_center=cfg.aligned_query,rsa_update=cfg.rsa_update,paper_product_read=cfg.paper_product_read,
                control_dim=width if cfg.full_controls or cfg.context_controls else 0,slot_feature=cfg.slot_feature,
                balanced_bridge=cfg.balanced_bridge,relation_marginal=cfg.relation_marginal,
                quadratic_features=cfg.quadratic_features,learn_read_scale=cfg.learn_read_scale))
            else:
                raise ValueError(cfg.memory_update)
            if cfg.aligned_query:
                self.reset_aligned_queries()
            if cfg.orthogonal_centers:
                with torch.no_grad():
                    for level in self.attention.center:
                        for matrix in level:nn.init.orthogonal_(matrix)
        elif cfg.attention != "transformer":
            raise ValueError("attention must be gpa or transformer")

    @torch.no_grad()
    def reset_aligned_queries(self):
        for qw, rw in zip(self.attention.query_weight, self.attention.route_weight):
            qw.zero_()  # learned delta, NOT the shared C projection
            m = rw.shape[-1]
            if self.cfg.balanced_bridge:rw.copy_(torch.eye(m,device=rw.device,dtype=rw.dtype)[None])
            else:rw.copy_(.02*(torch.eye(m, device=rw.device, dtype=rw.dtype)-1/m)[None])

    def forward(self, x):
        b, t, width = x.shape
        control = self.norm(x)
        write_control=control
        if self.cfg.context_controls:
            write_control=F.silu(self.control_conv(F.pad(control.transpose(1,2),
                (self.cfg.convolution-1,0))).transpose(1,2))
        projected = self.qkv(control).transpose(1, 2)
        projected = self.conv(F.pad(projected, (self.cfg.convolution-1, 0))).transpose(1, 2)
        key_dim=self.cfg.key_dim or width//self.cfg.heads
        key_width=key_dim*self.cfg.heads
        qp,kp,vp=projected.split((key_width,key_width,width),-1)
        q,k=[part.reshape(b,t,self.cfg.heads,key_dim) for part in (qp,kp)]
        v=vp.reshape(b,t,self.cfg.heads,width//self.cfg.heads)
        q, k = F.silu(q), F.silu(k)
        if self.cfg.attention == "gpa":
            if self.cfg.full_controls or self.cfg.context_controls:
                o,_=self.attention(q,k,v,control_input=write_control)
            else:
                o, _ = self.attention(q, k, v)
        else:
            if self.cfg.transformer_rope:
                from .ops import rope
                position = torch.arange(t, device=q.device)
                q, k = rope(q, position), rope(k, position)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), is_causal=True).transpose(1, 2)
        if self.cfg.gated_output:
            normalized=o.float()*(o.float().square().mean(-1,keepdim=True)+1e-5).rsqrt()
            o=normalized.to(o.dtype)*self.head_norm_weight.to(o.dtype)
            o=o.reshape(b,t,width)*F.silu(self.output_gate(write_control if self.cfg.contextual_output else control))
        x = x+self.out(o.reshape(b, t, width))
        return x+self.ff(self.ffnorm(x))


class RetrievalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError("width must be divisible by heads")
        self.config = cfg
        self.embedding = nn.Embedding(VOCAB, cfg.width)
        self.blocks = nn.ModuleList(RetrievalBlock(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, VOCAB, bias=False)
        self.head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=.02)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


@torch.no_grad()
def evaluate(model, *, length, records, samples=32, batch=8, seed=900000, free_generate=False,data_version=1):
    model.eval()
    device = next(model.parameters()).device
    amp = device.type == "cuda"
    report = {}
    for task in TASKS:
        correct, total, exact, groups_total, first, first_total = 0, 0, 0, 0, 0, 0
        sequence_exact,sequence_total=0,0
        ordinal_correct,ordinal_total=[0]*4,[0]*4
        free_correct = free_total = 0
        for start in range(0, samples, batch):
            size = min(batch, samples-start)
            ids, labels, metadata = make_batch(seed+start, size, length, records, task, device,data_version)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(ids).argmax(-1)
            mask = labels != -100
            correct += int(((pred == labels) & mask).sum())
            total += int(mask.sum())
            for i, info in enumerate(metadata):
                positions=[pos for group in info["groups"] for pos in group]
                sequence_exact+=int((pred[i,positions]==labels[i,positions]).all())
                sequence_total+=1
                for group in info["groups"]:
                    groups_total += 1
                    exact += int((pred[i, group] == labels[i, group]).all())
                    first += int(pred[i, group[0]] == labels[i, group[0]])
                    first_total += 1
                    if task=="uuid4":
                        for ordinal,pos in enumerate(group):
                            ordinal_correct[ordinal]+=int(pred[i,pos]==labels[i,pos])
                            ordinal_total[ordinal]+=1
            if free_generate and task == "uuid4":
                # UUID has exactly one query per row. Recompute prefixes to
                # avoid claiming an unimplemented LM+shortconv cache adapter.
                position = metadata[0]["groups"][0][0]
                assert all(info["groups"][0][0] == position for info in metadata)
                prefix = ids[:, :position+1]
                generated = []
                for _ in range(4):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        token = model(prefix)[:, -1].argmax(-1)
                    generated.append(token)
                    prefix = torch.cat((prefix, token[:, None]), 1)
                answer = labels[:, position:position+4]
                free_correct += int((torch.stack(generated, 1) == answer).all(1).sum())
                free_total += size
        report[task] = {"token_accuracy": correct/total, "query_exact_match_teacher_forced": exact/groups_total,
                        "first_answer_accuracy": first/first_total, "queries": groups_total,
                        "sequence_exact_match_teacher_forced":sequence_exact/sequence_total,
                        "sequences":sequence_total}
        if task=="uuid4":report[task]["teacher_forced_accuracy_by_identifier_position"]=[c/n for c,n in zip(ordinal_correct,ordinal_total)]
        if free_total:
            report[task]["free_generation_exact_match"] = free_correct/free_total
    return report
