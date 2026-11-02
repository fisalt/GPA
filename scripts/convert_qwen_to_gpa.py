"""Convert native Qwen2.5/Qwen3 weights, or merge a full GPA+LoRA checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
from gpa.hf_qwen import GPAQwenForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--slots", help="Comma-separated per-level slots; default two levels of native head_dim")
    p.add_argument("--last-slots", help="Optional independent slot layout for the final decoder layer")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--convolution", type=int, default=4)
    p.add_argument("--backend", choices=["torch", "fla"], default="fla")
    p.add_argument("--execution", choices=["reference", "packed", "triton", "staged"], default="staged")
    p.add_argument("--keep-rope", action="store_true")
    p.add_argument("--quadratic-features", action="store_true")
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--skip-tokenizer", action="store_true", help="For architecture-only tests")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError("Output must be empty; never overwrite a checkpoint")
    if args.merge_lora:
        model = GPAQwenForCausalLM.from_pretrained(args.model, local_files_only=args.local_files_only)
        model.merge_lora()
        model.config.use_cache = True
        record = dict(operation="merge_lora", all_gpa=True, source=args.model)
    else:
        source_config=AutoConfig.from_pretrained(args.model,trust_remote_code=False,
                                                local_files_only=args.local_files_only)
        if source_config.model_type not in ('qwen2','qwen3') or getattr(source_config,'auto_map',None):
            raise ValueError('Conversion requires a native Transformer Qwen2.5/Qwen3 checkpoint, not custom RSA/GPA code')
        dtype_options={"dtype" if int(transformers_version.split('.')[0])>=5 else "torch_dtype":torch.float32}
        native,loading = AutoModelForCausalLM.from_pretrained(args.model, **dtype_options,
            attn_implementation="eager", local_files_only=args.local_files_only,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'Non-exact native Transformer load: {loading}')
        # Snapshot inherited tensors BEFORE constructing the GPA wrapper.
        original = {n: (p, p._version) for n, p in native.named_parameters()}
        options = dict(chunk_size=args.chunk_size, backend=args.backend, execution=args.execution,
                       quadratic_features=args.quadratic_features)
        if args.slots: options["slots"] = tuple(map(int, args.slots.split(",")))
        model = GPAQwenForCausalLM.from_transformer(native, gpa=options,
                    keep_rope=args.keep_rope, convolution=args.convolution,
                    gpa_last_slots=tuple(map(int,args.last_slots.split(','))) if args.last_slots else None)
        inherited = dict(model.named_parameters())
        for name, (parameter, version) in original.items():
            assert name in inherited and inherited[name] is parameter, name
            assert parameter._version == version, ("Unexpected pretrained-weight mutation", name)
        record = dict(operation="transformer_to_gpa", source=args.model,
                      source_model_type=native.config.model_type, inherited_parameter_tensors=len(original),
                      inherited_tensor_identity_and_version_verified=True, all_gpa=True,
                      softmax_attention_layers=0, layers=len(model.model.layers),
                      slots=list(model.model.layers[0].self_attn.gpa.config.slots),
                      last_slots=list(model.model.layers[-1].self_attn.gpa.config.slots),
                      attention_function_preserved=False)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        tokenizer.save_pretrained(out)
    (out / "conversion.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__": main()
