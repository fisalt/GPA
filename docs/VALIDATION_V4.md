# v0.4 Qwen conversion validation — 2026-11-03

This record covers engineering correctness of the Qwen2.5/Qwen3 adapter and
FineWeb training pipeline. It does not transfer the v0.3 tiny retrieval scores
to a converted pretrained Qwen model.

| Environment | Checks actually run | Result |
|---|---|---|
| PyTorch 2.9.0+cu128, Transformers 5.15.0 | 10 CPU integration/data cases | Passed |
| Same, A100-SXM4-40GB | BF16 FLA forward, backward and partitioned cache | 1 passed |
| PyTorch 2.9.0+cu128, Transformers 4.51.3 | Same 10 CPU integration/data cases | Passed |
| Same 4.51 environment | Additional local parquet streaming/packing case | 1 passed |
| Both HF versions | CLI conversion → training → intermediate-checkpoint resume → merged auto_map generation | Passed |
| Both HF versions | Resumed final weights vs uninterrupted final weights | **Every state_dict tensor equal, atol=rtol=0** |
| 4.51 environment | Additional resumed checkpoint with tokenizer, data seed/provenance and trainable audit | Passed |

The overlapping CPU cases above are not 20 distinct tests. They cover both
native Qwen families; GQA; Qwen3 geometry where heads*head_dim differs from
hidden_size; optional inherited RoPE; no token SDPA calls; chunked CE; initial
state/LoRA gradients; fixed-block cache splits; beam state/conv-tail reorder;
greedy and beam generation; exact full checkpoint reload; LoRA merge; and an
independent final-layer layout (first layers 2×16, final layer 4×8).

The pipeline fixture is random tiny Qwen3 (width32, two layers, H4, KV heads2,
head_dim16, vocabulary128) with locally generated text. It trains a planned
three-step cosine schedule, saves step2, resumes that checkpoint into another
output directory, and compares all final tensors to uninterrupted step3.
The resumed step has a nonzero LR; this avoids a vacuous test that resumes an
already completed cosine schedule at LR=0. Full LoRA, GPA states, Embedding/Norm
and optimizer/RNG restoration are exercised. This does not establish exact
sample resumption for changed data files, world size or software versions.

The original core-only Windows CPU regression also completed **86 passed**;
HF integration was correctly skipped there because that interpreter has
PyTorch 2.0.1. HF is optional and requires PyTorch>=2.6. CUDA skips are never
reported as successful GPU checks. The pre-existing v0.3 kernel suite and its
116-test record remain historical; the new adapter does not change those
recurrence/route kernels.
The final local packaging/packing/parquet subset completed 3 passed and one
optional Accelerate-sharding skip; that sharding case did pass on the HF hosts.

## Compatibility fixes checked

* HF meta loading can materialize nonpersistent buffers using empty storage.
  Successful state_dict loading alone did not ensure valid edge indexes.
  The adapter rebuilds only deterministic source/target indexes, identity probe
  constants and native RoPE frequencies after loading. **Learned A/B/R,
  centers and controls are never reset.** Tests exercise the loaded model.
* Canonical model loading keeps FP32 master weights; BF16 computation uses
  autocast. Native source tensors are verified by identity/version during
  conversion and compared by value in the small architecture tests.
* HF4/5 dtype keyword and save-argument differences are handled explicitly.
* A tied LM head is stored once. Trainer can warn about the omitted alias;
  exact resumed-vs-uninterrupted tensor equality includes the tied readout.
* Cache generation uses the GPA cache directly, not a native DynamicCache.

The 4.51 tests used an isolated dependency view without DeepSpeed/PEFT. The
shared host's installed DeepSpeed release circularly imports OPT while
Transformers 4.51 initializes; changing that host environment was unnecessary.
No shared conda dependencies or existing training jobs were modified. ZeRO-2/3
configuration files are supplied, but **multi-GPU/DeepSpeed training was not
executed in this validation**. The old-library dependency conflict must not be
presented as a passing DeepSpeed4 test.

## Records and reproduction

Sanitized machine-readable records, including current source SHA256 values:
[results/v4/summary.json](results/v4/summary.json). Full per-version results:
[4.51](results/v4/hf451.json), [5.15](results/v4/hf515.json),
[exact resume4](results/v4/exact_resume_451.json),
[exact resume5](results/v4/exact_resume_515.json),
[parquet/checkpoint metadata](results/v4/data_and_metadata.json).

```bash
python -m pytest -q tests/test_hf_qwen.py tests/test_finetune_data.py
python scripts/smoke_qwen_conversion.py --output runs/check_conversion
# Combined CPU, CLI and optional single-GPU check, sharing one process:
python scripts/check_hf_pipeline.py --output runs/check_pipeline --cuda
```

No full-size pretrained Qwen was trained on FineWeb here; no resulting
perplexity, retrieval, large-model throughput or long-context convergence is
claimed. The runnable conversion/training instructions are in
[QWEN_CONVERSION.md](QWEN_CONVERSION.md).
