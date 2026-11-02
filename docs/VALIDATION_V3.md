# GPA v0.3 validation record

This supersedes no historical measurements: the v0.1/v0.2 records remain in
`VALIDATION.md` and `OPTIMIZATION.md`. Results below concern the v0.3 source.

## Final suite

* Local Windows, Python/PyTorch CPU: **85 passed, 31 CUDA-only skipped**,
  25.27 seconds (`python -m pytest -q tests --tb=short`). CUDA skips are not
  counted as GPU successes.
* Single A100-SXM4-40GB, Python 3.11, PyTorch 2.9.0+cu128, Triton 3.5.0:
  **116 passed, 11 warnings**, 586.43 seconds, same command. No skipped tests.
* GPU suite includes the new staged schedule's square and unequal-value
  dimensions, prefix length509, and cached splits256/65/191 of T512.
  Runtime versions and tested source digests are in
  [runtime_final.json](results/v3/runtime_final.json).

FLA's 11 warnings are input-layout heuristics on tiny remainder calls where
T is less than packed head count. Inputs are explicitly `[B,T,H,D]`; tests
compare output/cache/gradients and pass. No warning suppression was introduced.

## Scope of the checks

CPU double comparisons cover output, every cache tensor, Q/K/V, control input,
parameters, and losses with nonzero final-state adjoints. They include initial
state gradients, one/two/four levels, both enhanced and baseline variants,
prefix causality, and arbitrary cached call partitions. Staging is compared
against the **same fixed-block lagged model**, not tokenwise post-update
residuals. Packed/reference, counted scalar/vector gate directions, learned
read scale, angular features, and contextual controllers have separate tests.

The dense route Triton suite covers forward/backward for original additive,
compatibility-product, and optional marginal-corrected reading, including
gain/count/edge gradients. Shapes include L2/M7, L4/M16, L4/M64 and L8/M128.
Large random product logits need both elementwise and relative-L2 bounds:
softmax's sensitivity makes bitwise equality an inappropriate expectation.

The CUDA Graph optimizer test runs real AdamW updates. A separate three-step
trajectory check found eager/graph gradient relative L2 zero at each step for
the checked counted model. Neither check proves all runtimes/optimizers safe.

The selected paired-GDN model completed 20,000 finite training steps under
answer-only CE. Its original backend is `triton` with CUDA Graph. `staged` is
an optional algebra-preserving schedule with different BF16 accumulation;
its accuracy regression uses identical frozen weights, not a new trained model.
The final staged rerun at T128/T256/T512 has **identical recorded task metrics**
to the original selected model: same seed920000, 256 examples/task, including
identifier free generation and whole-three-query MQAR EM. This is task-score
agreement, not a claim of bitwise-identical BF16 logits. The complete rerun is
stored in [staged_regression.json](results/v3/staged_regression.json).
See the [report](REPORT_V3.zh-CN.md) and [full data](results/v3/summary.json).

## Timing interpretation

`efficiency_lm_staged.json` times the complete selected tiny LM's forward CE,
backward, gradient clipping and fused AdamW(lr=0), B8/BF16, 10 iterations after
warmup/capture. It excludes compile, data, logging and IO. `lr=0` preserves
checkpoint parameters while executing optimizer kernels. Peak allocated bytes
are PyTorch allocator measurements, not total process/driver memory.

At T256, original/staged Graph step times are 21.02/10.56 ms; at T1024,
74.62/26.46 ms. The same-width Transformer is still faster. New/old logit
relative L2 is .0001903/.0005910 at these lengths, and parameter-gradient
relative L2 is .002874/.005038. The old raw-operator timing file is a different
shape/scope and must not replace these LM measurements.

The final T4096 supplement measured 302.36→105.02 ms (2.88×) for old/staged
Graph GPA. Logit/parameter-gradient relative L2 was .0005401/.005034.
Default-SDPA Transformer Graph measured 131.56 ms in this setup; no optimal
FlashAttention configuration was established. This is a speed-only input
length, not long-length training or retrieval evaluation. Raw records:
[efficiency_lm_4096.json](results/v3/efficiency_lm_4096.json).

## Reproduce

```bash
python -m pytest -q tests --tb=short
bash scripts/run_selected_v3.sh
# Once the model checkpoint exists:
python scripts/eval_retrieval.py runs/selected_v3/checkpoint-20000.pt \
  --execution staged --output runs/staged_regression --seed 920000 --samples 256
python scripts/benchmark_lm.py runs/selected_v3/checkpoint-20000.pt \
  --lengths 256,1024 --output efficiency.json
```

Tests do not imply an HF whole-LM incremental-generation adapter. Attention
state caching is tested; tiny-LM evaluation recomputes causal prefixes.
Checkpoints and dependencies are not included in the source archive.

The v0.3 wheel also builds offline without dependencies, installs into an
isolated target, imports from that target, and completes a double-precision
staged forward/backward with unequal key/value dimensions. The source ZIP
tests verify reproducible bytes and each manifest digest. The untouched v0.2
archive still hashes to
`2705020454c70d1d14e9d65335a63e61e9e1f4411985d32b80c096be2d6b5076`.

The updated diagnostics CLI also completed a CPU FP32 reference pass on the
selected checkpoint, including read-only R ablation and exported arrays.
Optional plots run in a separate matplotlib process; this avoids a duplicate
OpenMP runtime failure seen when importing plotting libraries into the local
Windows torch process. CPU arrays/plots are explicitly labelled, not presented
as the original GPU BF16 tensors. Core source hashes/tests were unchanged by
this diagnostics-only improvement.
