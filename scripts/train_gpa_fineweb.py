"""Pure FineWeb causal-LM adaptation, full checkpoints including new GPA/LoRA."""
import argparse
import glob
import inspect
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
from gpa.hf_qwen import GPAQwenForCausalLM
from gpa.finetune import PackedTextDataset, packed_collator, configure_trainable


class GPATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, use_cache=False)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Converted GPA checkpoint directory")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Full Trainer checkpoint, including optimizer/scheduler")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--data-files", nargs="+", help="Local parquet/json/jsonl paths, directories or globs")
    p.add_argument("--text-column", default="text")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--tokens-per-update", type=int, default=524288)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--loss-chunk-size", type=int, default=512)
    p.add_argument("--tuning", choices=["lora", "full", "gpa-only"], default="lora")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--lora-alpha", type=float, default=64.)
    p.add_argument("--freeze-embedding", action="store_true")
    p.add_argument("--freeze-norm", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--deepspeed", help="Optional ds_configs/stage2.json or stage3.json")
    p.add_argument("--backend", choices=["auto", "fla", "torch"], default="auto")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    destination=Path(args.output)
    if not args.resume and destination.exists() and (
            any(destination.glob("checkpoint-*")) or (destination/"final").exists()):
        raise ValueError("Output already contains training weights; pass --resume or choose a new directory")
    set_seed(args.seed); torch.set_num_threads(4)
    world = int(os.environ.get("WORLD_SIZE", 1))
    quantum = world * args.microbatch * args.length
    accumulation = args.accumulation
    if accumulation is None:
        if args.tokens_per_update % quantum:
            raise ValueError("tokens-per-update must divide world*microbatch*length exactly; or set --accumulation")
        accumulation = args.tokens_per_update // quantum
    if accumulation < 1: raise ValueError("Accumulation must be positive")
    source = args.resume or args.model
    # Load BEFORE constructing TrainingArguments/ZeRO-3 global init hooks.
    model, info = GPAQwenForCausalLM.from_pretrained(source,
        local_files_only=args.local_files_only, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Non-exact checkpoint load: {info}")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda: raise ValueError("CUDA is unavailable")
    backend = args.backend if args.backend != "auto" else ("fla" if use_cuda else "torch")
    model.set_runtime_backend(backend)
    model.config.loss_chunk_size = args.loss_chunk_size
    model.config.use_cache = False
    audit = configure_trainable(model, args.tuning, args.lora_rank, args.lora_alpha,
                               not args.freeze_embedding, not args.freeze_norm)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    files = []
    for value in args.data_files or []:
        files.extend(str(p) for p in Path(value).rglob("*.parquet")) if Path(value).is_dir() else files.extend(glob.glob(value, recursive=True))
    files = sorted(set(files))
    if args.data_files and not files: raise ValueError("No data files matched")
    dataset = PackedTextDataset(tokenizer, args.length, args.dataset, args.dataset_config,
        files or None, args.text_column, args.seed, args.shuffle_buffer)
    bf16 = args.precision == "bf16" or (args.precision == "auto" and use_cuda)
    if bf16 and not use_cuda: raise ValueError("Use fp32 for this CPU training path")
    if backend == "fla" and not bf16: raise ValueError("FLA training requires --precision bf16/auto on CUDA")
    save_options={"save_safetensors":True} if "save_safetensors" in inspect.signature(TrainingArguments).parameters else {}
    training = TrainingArguments(output_dir=args.output, per_device_train_batch_size=args.microbatch,
        gradient_accumulation_steps=accumulation, max_steps=args.max_steps, learning_rate=args.lr,
        warmup_steps=args.warmup_steps, lr_scheduler_type="cosine", max_grad_norm=1.,
        bf16=bf16, use_cpu=not use_cuda, gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, deepspeed=args.deepspeed,
        save_steps=args.save_steps, save_total_limit=3,
        logging_steps=1, logging_nan_inf_filter=False, disable_tqdm=True, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_drop_last=True,
        accelerator_config={"dispatch_batches": False, "split_batches": False},
        ddp_find_unused_parameters=False, seed=args.seed, **save_options)
    trainer = GPATrainer(model=model, args=training, train_dataset=dataset, data_collator=packed_collator)
    trainer.model_accepts_loss_kwargs = False
    record = dict(training_data="local text files" if files else args.dataset,
                  objective="dense next-token CE", retrieval_data_added_by_script=False,
                  model=args.model, resumed_from=args.resume, backend=backend,
                  sequence_length=args.length, world_size=world, microbatch=args.microbatch,
                  accumulation=accumulation, input_tokens_per_update=quantum*accumulation,
                  data_seed=args.seed,
                  learning_rate=args.lr, max_grad_norm=1., checkpoint_format="full model including native LoRA",
                  stream_sharding="Accelerate only; no additional dataset rank sharding",
                  exact_saved_iterator_cursor=False, local_data_files=files,
                  dataset=None if files else args.dataset, dataset_config=args.dataset_config)
    if trainer.is_world_process_zero():
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out)
        (out/"trainable_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
        (out/"training_run.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"setup": record, "trainable_parameters": audit["trainable_parameters"]}), flush=True)
    from transformers import TrainerCallback
    class CheckpointTokenizer(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                folder=Path(args.output_dir)/f"checkpoint-{state.global_step}"
                tokenizer.save_pretrained(folder)
                (folder/'training_run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
                (folder/'trainable_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    trainer.add_callback(CheckpointTokenizer())
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(args.output)/"final"))
    if trainer.is_world_process_zero(): tokenizer.save_pretrained(Path(args.output)/"final")


if __name__ == "__main__": main()
