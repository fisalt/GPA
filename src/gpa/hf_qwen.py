"""Qwen2.5/Qwen3 -> all-GPA causal LM, optional native LoRA, HF checkpoints.

The decoder loop deliberately never calls a Transformer attention/mask builder.
Install this package's [hf] extra. Saved auto_map code still depends on gpa.
"""
from dataclasses import dataclass, fields, is_dataclass, asdict
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers import __version__ as transformers_version
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from gpa.attention import DenseGPA, GPAConfig


class GPAQwenConfig(PretrainedConfig):
    model_type = "gpa_qwen"

    def __init__(self, base_config=None, gpa=None, lora=None, keep_rope=False,
                 convolution=4, loss_chunk_size=512, gpa_last_slots=None, **kwargs):
        base_config = copy.deepcopy(base_config or {"model_type": "qwen3"})
        if base_config.get("model_type") not in ("qwen2", "qwen3"):
            raise ValueError("Only native Qwen2/Qwen2.5 and Qwen3 causal LMs are supported")
        raw = dict(base_config); kind = raw.pop("model_type")
        native = AutoConfig.for_model(kind, **raw)
        for name in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "max_position_embeddings", "initializer_range"):
            kwargs.setdefault(name, getattr(native, name))
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings"):
            kwargs.setdefault(name, getattr(native, name, None))
        kwargs.setdefault("is_encoder_decoder", False)
        kwargs.setdefault("use_cache", True)
        super().__init__(**kwargs)
        self.base_config = base_config
        self.gpa = dict(gpa or {})
        self.gpa_last_slots = list(gpa_last_slots) if gpa_last_slots is not None else None
        self.lora = dict(lora or {"rank": 0, "alpha": 64., "dropout": 0.})
        self.keep_rope = bool(keep_rope)
        self.convolution = int(convolution)
        self.loss_chunk_size = int(loss_chunk_size)
        if self.convolution < 1 or self.loss_chunk_size < 1:
            raise ValueError("convolution and loss_chunk_size must be positive")


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x. Full checkpoints include base AND adapters."""
    def __init__(self, base, rank, alpha, dropout=0.):
        super().__init__()
        if rank < 1: raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self): return self.base.weight

    def forward(self, x):
        a = F.linear(self.dropout(x), self.lora_A.to(x.dtype))
        return self.base(x) + self.scale * F.linear(a, self.lora_B.to(a.dtype))

    @torch.no_grad()
    def merge(self):
        update = self.lora_B.float() @ self.lora_A.float()
        self.base.weight.add_((self.scale * update).to(self.base.weight.dtype))
        return self.base


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor): return fn(value)
    if isinstance(value, tuple): return tuple(_map_tensors(v, fn) for v in value)
    if is_dataclass(value):
        return type(value)(**{f.name: _map_tensors(getattr(value, f.name), fn) for f in fields(value)})
    return value


@dataclass(frozen=True)
class AttentionMemory:
    state: object
    projected_tail: torch.Tensor
    control_tail: torch.Tensor


class GPAQwenCache:
    """No growing K/V tensors. Short-conv tails and all A/B/R states reorder."""
    is_compileable = False

    def __init__(self, layers):
        self.layers = [None] * layers
        self.seen_tokens = 0

    def get_seq_length(self, layer_idx=0): return self.seen_tokens
    def get_max_cache_shape(self): return None
    def get_max_length(self): return None

    def reorder_cache(self, beam_idx):
        self.layers = [_map_tensors(s, lambda x: x.index_select(0, beam_idx.to(x.device)))
                       for s in self.layers]

    def batch_repeat_interleave(self, repeats):
        self.layers = [_map_tensors(s, lambda x: x.repeat_interleave(repeats, dim=0))
                       for s in self.layers]

    def detach(self):
        result = GPAQwenCache(len(self.layers)); result.seen_tokens = self.seen_tokens
        result.layers = [_map_tensors(s, lambda x: x.detach()) for s in self.layers]
        return result


def _causal_conv(x, weight, tail):
    # Cast weights rather than parameters: FP32 master weights keep gradients.
    width = weight.shape[-1] - 1
    prefix = x.new_zeros(x.shape[0], x.shape[-1], width) if tail is None else tail.to(x.dtype)
    joined = torch.cat((prefix, x.transpose(1, 2)), -1)
    out = F.conv1d(joined, weight.to(x.dtype), groups=x.shape[-1]).transpose(1, 2)
    return out, joined[:, :, -width:] if width else joined[:, :, :0]


def _rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class GPAQwenAttention(nn.Module):
    def __init__(self, native, native_config, config):
        super().__init__()
        self.q_proj, self.k_proj = native.q_proj, native.k_proj
        self.v_proj, self.o_proj = native.v_proj, native.o_proj
        self.q_norm = getattr(native, "q_norm", nn.Identity())
        self.k_norm = getattr(native, "k_norm", nn.Identity())
        self.heads = native_config.num_attention_heads
        self.kv_heads = native_config.num_key_value_heads
        self.head_dim = self.q_proj.out_features // self.heads
        if self.heads % self.kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        options = dict(config.gpa)
        options.setdefault("slots", (self.head_dim, self.head_dim))
        options["slots"] = tuple(options["slots"])
        defaults = dict(heads=self.heads, key_dim=self.head_dim, value_dim=self.head_dim,
                        control_dim=native_config.hidden_size, chunk_size=64,
                        backend="fla", execution="staged", rsa_update=True,
                        paper_product_read=True, shared_query_center=True,
                        balanced_bridge=True, beta_init=.6, decay_init=.0001, read_scale=8.)
        for key, value in defaults.items(): options.setdefault(key, value)
        for key in ("heads", "key_dim", "value_dim", "control_dim"):
            if options[key] != defaults[key]:
                raise ValueError(f"{key} must match the inherited native Qwen projections")
        self.gpa = DenseGPA(GPAConfig(**options))
        # Aligned direct query; these are NEW parameters, never reset on load.
        with torch.no_grad():
            for q in self.gpa.query_weight: q.zero_()
        channels = self.q_proj.out_features + self.k_proj.out_features + self.v_proj.out_features
        self.projected_conv = nn.Parameter(torch.zeros(channels, 1, config.convolution))
        self.control_conv = nn.Parameter(torch.full((native_config.hidden_size, 1, config.convolution),
                                                    1 / config.convolution))
        with torch.no_grad(): self.projected_conv[:, :, -1] = 1.

    def forward(self, hidden, rotary=None, memory=None):
        b, t, _ = hidden.shape
        projected = torch.cat((self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)), -1)
        projected, ptail = _causal_conv(projected, self.projected_conv,
                                      None if memory is None else memory.projected_tail)
        control, ctail = _causal_conv(hidden, self.control_conv,
                                    None if memory is None else memory.control_tail)
        q, k, v = projected.split((self.q_proj.out_features, self.k_proj.out_features,
                                    self.v_proj.out_features), -1)
        q = self.q_norm(q.reshape(b, t, self.heads, self.head_dim))
        k = self.k_norm(k.reshape(b, t, self.kv_heads, self.head_dim))
        v = v.reshape(b, t, self.kv_heads, self.head_dim)
        if rotary is not None:
            cos, sin = (x[:, :, None].to(q.dtype) for x in rotary)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        k, v = (x.repeat_interleave(groups, dim=2) for x in (k, v))
        if self.gpa.config.backend == "fla":
            if not q.is_cuda: raise ValueError("FLA checkpoint needs CUDA; set_runtime_backend('torch') for CPU")
            q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
        output, state = self.gpa(q, k, v, None if memory is None else memory.state,
                                 control_input=F.silu(control))
        output = self.o_proj(output.reshape(b, t, -1).to(hidden.dtype))
        return output, AttentionMemory(state, ptail, ctail)


class GPAQwenDecoderLayer(nn.Module):
    def __init__(self, native, native_config, config):
        super().__init__()
        self.input_layernorm = native.input_layernorm
        self.self_attn = GPAQwenAttention(native.self_attn, native_config, config)
        self.post_attention_layernorm = native.post_attention_layernorm
        self.mlp = native.mlp

    def forward(self, hidden, rotary=None, memory=None):
        branch, memory = self.self_attn(self.input_layernorm(hidden), rotary, memory)
        hidden = hidden + branch
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), memory


class GPAQwenDecoder(nn.Module):
    def __init__(self, native, native_config, config):
        super().__init__()
        self.embed_tokens, self.norm = native.embed_tokens, native.norm
        self.rotary_emb = native.rotary_emb
        layers=[]
        for i,layer in enumerate(native.layers):
            current=copy.copy(config)
            current.gpa=dict(config.gpa)
            if i==len(native.layers)-1 and config.gpa_last_slots is not None:
                current.gpa['slots']=config.gpa_last_slots
            layers.append(GPAQwenDecoderLayer(layer,native_config,current))
        self.layers=nn.ModuleList(layers)


class GPAQwenForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GPAQwenConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _is_stateful = True
    _no_split_modules = ["GPAQwenDecoderLayer"]
    # dict also iterates as key patterns on Transformers 4.x.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config, native_model=None):
        super().__init__(config)
        raw = dict(config.base_config); kind = raw.pop("model_type")
        native_config = AutoConfig.for_model(kind, **raw)
        native_model = native_model or AutoModelForCausalLM.from_config(native_config,
                                                                       attn_implementation="eager")
        self.model = GPAQwenDecoder(native_model.model, native_config, config)
        self.config.gpa = asdict(self.model.layers[0].self_attn.gpa.config)
        self.lm_head = native_model.lm_head
        self.gradient_checkpointing = False
        self.checkpoint_options = {"use_reentrant": False}
        if int(config.lora.get("rank", 0)):
            self.enable_lora(**config.lora)
        self.post_init()

    def _init_weights(self, module):
        # Native Qwen and DenseGPA constructors already initialize their parts.
        # Never reinitialize inherited pretrained weights during conversion.
        pass

    @classmethod
    def from_transformer(cls, source, **options):
        if source.config.model_type not in ("qwen2", "qwen3"):
            raise ValueError("Expected native Qwen2.5 or Qwen3, not an RSA/custom checkpoint")
        config = GPAQwenConfig(base_config=source.config.to_dict(), **options)
        model = cls(config, native_model=source)
        model.generation_config = copy.deepcopy(source.generation_config)
        return model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Canonical checkpoints use FP32 master parameters; use BF16 autocast.
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if dtype not in (None, "auto", "float32", torch.float32):
            raise ValueError("Load GPA master weights in float32; use bf16 autocast for computation")
        kwargs.pop("dtype", None)
        kwargs.pop("torch_dtype", None)
        kwargs["dtype" if int(transformers_version.split('.')[0])>=5 else "torch_dtype"] = torch.float32
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        model._rebuild_runtime_buffers()
        return result

    def _rebuild_runtime_buffers(self):
        # HF meta loading can materialize nonpersistent buffers with to_empty.
        # Restore ONLY deterministic constants, never learned A/B/R/centers.
        for layer in self.model.layers:
            memory = layer.self_attn.gpa
            device = memory.center[0].device
            memory._edge_source = torch.tensor([i for i,j in memory.edges],dtype=torch.long,device=device)
            memory._edge_target = torch.tensor([j for i,j in memory.edges],dtype=torch.long,device=device)
            block = 64 if memory.config.backend == "fla" else memory.config.chunk_size
            d = memory.config.key_dim
            memory._snapshot_basis = F.pad(torch.eye(d,device=device),(0,0,0,(-d)%block))
        raw = dict(self.config.base_config); kind = raw.pop("model_type")
        native_config = AutoConfig.for_model(kind, **raw)
        self.model.rotary_emb = type(self.model.rotary_emb)(config=native_config).to(
            self.model.embed_tokens.weight.device)

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, value): self.lm_head = value

    @classmethod
    def _supports_default_dynamic_cache(cls): return False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        self.checkpoint_options = dict(gradient_checkpointing_kwargs or {"use_reentrant": False})
        if self.checkpoint_options.get("use_reentrant", False):
            raise ValueError("Use non-reentrant checkpointing for GPA")

    def gradient_checkpointing_disable(self): self.gradient_checkpointing = False

    def enable_lora(self, rank=32, alpha=64., dropout=0., targets=None):
        targets = targets or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if any(isinstance(m, LoRALinear) for m in self.modules()):
            if int(self.config.lora["rank"]) != rank: raise ValueError("LoRA already exists with another rank")
            return
        for parent in list(self.modules()):
            for name, child in list(parent.named_children()):
                if name in targets and isinstance(child, nn.Linear):
                    setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
        self.config.lora = dict(rank=rank, alpha=alpha, dropout=dropout, targets=list(targets))

    def merge_lora(self):
        for parent in list(self.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoRALinear): setattr(parent, name, child.merge())
        self.config.lora = dict(rank=0, alpha=64., dropout=0.)
        return self

    def set_runtime_backend(self, backend, execution=None):
        from dataclasses import replace
        self.config.gpa.update(backend=backend)
        if execution: self.config.gpa["execution"] = execution
        for layer in self.model.layers:
            layer.self_attn.gpa.config = replace(layer.self_attn.gpa.config, backend=backend,
                execution=execution or layer.self_attn.gpa.config.execution)
        return self

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None, **kwargs):
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if seen:
            input_ids = input_ids[:, seen:] if input_ids.shape[1] > seen else input_ids[:, -1:]
        output = dict(input_ids=input_ids.contiguous(), past_key_values=past_key_values,
                      attention_mask=attention_mask, use_cache=kwargs.get("use_cache", True), logits_to_keep=1)
        if inputs_embeds is not None and not seen:
            output.update(input_ids=None, inputs_embeds=inputs_embeds)
        return output

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_hidden_states=False, output_attentions=False, return_dict=None,
                logits_to_keep=0, num_logits_to_keep=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids or inputs_embeds")
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, t, _ = hidden.shape
        if t < 1: raise ValueError("Empty token input")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != b or attention_mask.shape[1] < t:
                raise ValueError("Expected a 2D all-valid attention_mask")
            if not bool((attention_mask == 1).all()):
                raise ValueError("Padding/packed-document masks are unsupported: use unpadded equal-length streams")
        if past_key_values is not None and not isinstance(past_key_values, GPAQwenCache):
            raise TypeError("GPA requires GPAQwenCache, not a Transformer DynamicCache/KV tuple")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.training and self.gradient_checkpointing:
            if past_key_values is not None: raise ValueError("Checkpointed training requires a fresh stream")
            use_cache = False
        cache = past_key_values
        if use_cache and cache is None: cache = GPAQwenCache(len(self.model.layers))
        seen = 0 if cache is None else cache.seen_tokens
        if position_ids is None:
            position_ids = torch.arange(seen, seen + t, device=hidden.device)[None].expand(b, -1)
        rotary = self.model.rotary_emb(hidden, position_ids) if self.config.keep_rope else None
        history = () if output_hidden_states else None
        for i, layer in enumerate(self.model.layers):
            if history is not None: history += (hidden,)
            old = None if cache is None else cache.layers[i]
            if self.training and self.gradient_checkpointing:
                def run(x, layer=layer): return layer(x, rotary, None)[0]
                hidden = checkpoint(run, hidden, **self.checkpoint_options)
            else:
                hidden, memory = layer(hidden, rotary, old)
                if use_cache: cache.layers[i] = memory
        hidden = self.model.norm(hidden)
        if history is not None: history += (hidden,)
        if use_cache: cache.seen_tokens += t
        loss, logits = None, None
        if labels is not None:
            if labels.shape != (b, t): raise ValueError("labels must match this input block")
            count = (labels[:, 1:] != -100).sum().clamp_min(1)
            loss = hidden.sum() * 0.
            for start in range(0, t - 1, self.config.loss_chunk_size):
                stop = min(t - 1, start + self.config.loss_chunk_size)
                def ce(x, y):
                    return F.cross_entropy(self.lm_head(x).float().reshape(-1, self.config.vocab_size),
                                           y.reshape(-1), reduction="sum", ignore_index=-100)
                x, y = hidden[:, start:stop], labels[:, start+1:stop+1]
                value = checkpoint(ce, x, y, use_reentrant=False) if self.training else ce(x, y)
                loss = loss + value / count
        else:
            keep = logits_to_keep if num_logits_to_keep is None else num_logits_to_keep
            logits = self.lm_head(hidden[:, -keep:] if isinstance(keep, int) and keep > 0 else hidden)
        result = CausalLMOutputWithPast(loss=loss, logits=logits,
            past_key_values=cache if use_cache else None, hidden_states=history, attentions=None)
        if return_dict is False: return result.to_tuple()
        return result


GPAQwenConfig.register_for_auto_class()
GPAQwenForCausalLM.register_for_auto_class("AutoModelForCausalLM")
AutoConfig.register(GPAQwenConfig.model_type, GPAQwenConfig, exist_ok=True)
AutoModelForCausalLM.register(GPAQwenConfig, GPAQwenForCausalLM, exist_ok=True)
