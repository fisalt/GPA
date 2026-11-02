"""Packed FineWeb streams and an explicit Qwen->GPA trainability audit."""
import random
import torch
from torch.utils.data import IterableDataset


class PackedTextDataset(IterableDataset):
    """Every rank reads the same stream; Accelerate shards BATCHES exactly once.

    No internal rank/worker sharding. Trainer must use dataloader_num_workers=0.
    Each fixed-size block starts with fresh model memory. Documents are EOS-separated.
    """
    def __init__(self, tokenizer, block_size, dataset="HuggingFaceFW/fineweb-edu",
                 dataset_config="sample-10BT", data_files=None, text_column="text",
                 seed=42, shuffle_buffer=10000, repeat=True, rows=None):
        if block_size < 2: raise ValueError("block_size must be at least 2")
        if tokenizer.eos_token_id is None: raise ValueError("Tokenizer needs an EOS token")
        self.tokenizer, self.block_size = tokenizer, block_size
        self.dataset, self.dataset_config, self.data_files = dataset, dataset_config, data_files
        self.text_column, self.seed, self.shuffle_buffer = text_column, seed, shuffle_buffer
        self.repeat, self.rows, self.epoch = repeat, rows, 0

    def set_epoch(self, epoch): self.epoch = int(epoch)

    def _rows(self, epoch):
        if self.rows is not None:
            values = list(self.rows)
            if self.shuffle_buffer: random.Random(self.seed + epoch).shuffle(values)
            return iter(values)
        from datasets import load_dataset
        if self.data_files:
            kind = "parquet" if all(str(f).endswith(".parquet") for f in self.data_files) else "json"
            stream = load_dataset(kind, data_files=list(self.data_files), split="train", streaming=True)
        else:
            stream = load_dataset(self.dataset, name=self.dataset_config, split="train", streaming=True)
        if self.shuffle_buffer:
            stream = stream.shuffle(seed=self.seed + epoch, buffer_size=self.shuffle_buffer)
        return iter(stream)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Use num_workers=0; Accelerate owns distributed stream sharding")
        pending, offset, epoch = [], 0, self.epoch
        while True:
            seen = False
            for row in self._rows(epoch):
                text = row[self.text_column]
                if not isinstance(text, str): raise TypeError("Text column must contain strings")
                if not text.strip(): continue
                seen = True
                pending.extend(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                pending.append(self.tokenizer.eos_token_id)
                while len(pending) - offset >= self.block_size:
                    yield {"input_ids": torch.tensor(pending[offset:offset+self.block_size], dtype=torch.long)}
                    offset += self.block_size
                if offset:
                    pending, offset = pending[offset:], 0
            if not self.repeat: return  # Explicitly drop the final short block.
            if not seen: raise ValueError("No nonempty text rows in the selected dataset")
            epoch += 1


def packed_collator(rows):
    ids = torch.stack([r["input_ids"] for r in rows])
    return dict(input_ids=ids, labels=ids.clone(), attention_mask=torch.ones_like(ids))


def configure_trainable(model, tuning="lora", rank=None, alpha=64.,
                        train_embedding=True, train_norm=True):
    from .hf_qwen import LoRALinear
    if tuning not in ("lora", "full", "gpa-only"): raise ValueError("Unknown tuning mode")
    if tuning == "lora":
        saved_rank = int(model.config.lora.get("rank", 0))
        if saved_rank and rank is not None and rank != saved_rank:
            raise ValueError("Cannot change LoRA rank while resuming; merge/export first")
        if not saved_rank: model.enable_lora(rank=rank or 32, alpha=alpha)
    elif int(model.config.lora.get("rank", 0)):
        raise ValueError("Existing LoRA checkpoint requires lora tuning or a merged export")
    model.requires_grad_(tuning == "full")
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.gpa.requires_grad_(True)
        attention.projected_conv.requires_grad_(True)
        attention.control_conv.requires_grad_(True)
        if attention.gpa.config.positional_mode != "moments":
            attention.gpa.moment_gain.requires_grad_(False)
        if not attention.gpa.edges: attention.gpa.relation_gain.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if tuning == "lora" and (".lora_A" in name or ".lora_B" in name): parameter.requires_grad_(True)
        if train_norm and "norm" in name: parameter.requires_grad_(True)
        if parameter.numel() == 0: parameter.requires_grad_(False)
    if train_embedding: model.get_input_embeddings().requires_grad_(True)
    # Untied LM heads follow full tuning only, preserving the pretrained readout
    # in LoRA/GPA-only mode. Tied heads train with the shared input embedding.
    audit = dict(tuning=tuning, train_embedding=train_embedding, train_norm=train_norm,
                 layers=len(model.model.layers), softmax_attention_layers=0,
                 lora_modules=sum(isinstance(m, LoRALinear) for m in model.modules()),
                 total_parameters=sum(p.numel() for p in model.parameters()),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trainable={n: {"shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
                            for n, p in model.named_parameters() if p.requires_grad})
    model.config.training_policy = {k: v for k, v in audit.items() if k != "trainable"}
    return audit
