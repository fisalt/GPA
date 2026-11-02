# GPA: five proposition references and dense multilevel gated-delta attention

Version 0.3 adds **updated paired GDN storage and fully dense compatibility-product reading**,
plus an explicit counted-KV alternative (`PaperDenseGPA`). Storage is informed
by the latest RSA controls/relations, without its adjacent/rank32 sparse reader. All
v0.2 results remain available. See [v0.3 equations/experiments](docs/DENSE_V3.md).

The selected 0.494M-parameter pure-CE pilot reaches **93.36%** four-token-ID
free-generation EM versus **96.09%** for a same-width/depth Transformer on an
independent T256 test; single/MQAR/overwrite are 100% for both. This is synthetic
structured retrieval, not official RULER or a 100%-retrieval claim. Same-model
level staging improves GPA training-step time by 1.99× at T256 and 2.82× at T1024;
the Transformer remains faster. See the [complete report](docs/REPORT_V3.zh-CN.md)
and [raw results](docs/results/v3/summary.json), including unsuccessful variants.

[中文说明](README.zh-CN.md) · [Equations](docs/EQUATIONS.md) · [Validation](docs/VALIDATION.md) · [Release guide](docs/RELEASE.md)

Version 0.2 adds packed level/edge scans, a custom Triton dense-route forward
and backward, static-shape CUDA Graph support, and a mixed synthetic retrieval
LM experiment. See [optimization/experiment notes](docs/OPTIMIZATION.md).

The historical v0.2 single-A100 suite had **58 passed**. Its retrieval pilot is
**not yet competitive with its Transformer control**; speed/correctness
improvements must not be mistaken for validated model quality. The optional
aligned-query pilot is nondefault and has mixed results.

This release provides (1) auditable implementations of the five propositions
in the local 2026-10-24 GPA manuscript and (2) a trainable, sequence-linear
GPA attention core based on two Gated DeltaNet (GDN) states per level and
**full, dynamic, all-to-all cross-level relation matrices**.

It is not the previous adjacent/triangular RSA reader. Later levels can route
information back to earlier levels. Residual **storage** remains hierarchical;
**reading** uses a simultaneous snapshot of every updated level. It does not
call ABC, perform sequence-length softmax attention, or cache a growing KV list.

## Scope and honesty

The original proposition references use the author-confirmed 2026-10-24
`neurips_2026.tex`. Version 0.3 additionally inspected the 2026-11-02 source
and updated RSA implementation; see [provenance](docs/PROVENANCE.md).
The user's explicit dense-routing request takes precedence over adjacent
connections in that manuscript.

The proposition references distinguish exact identities, conditional bounds,
and approximations. The GDN integrated model is an **engineering approximation**,
not a proof that finite practical slots exactly reproduce arbitrary softmax
attention. No paper accuracy table, NIAH/RULER result, or baseline-equivalent
speed is claimed for this new implementation. No trained weights/data are bundled.
The optimized backend delegates public GDN scans to FLA/Triton; the entire
multilevel reader is **not** claimed to be one fully fused Triton kernel.

## Install and run

Python >= 3.10, PyTorch >= 2.0. The default backend also runs on CPU.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/smoke.py
python examples/cache_decode.py
python examples/propositions.py
```

Optional CUDA scans require a compatible CUDA/PyTorch/Triton installation and
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention).
FLA is an external dependency under its own license; no FLA code is vendored.
Widths below 16 are internally zero-padded for compatible tensor-core backward
kernels, then sliced back; the model's logical capacity is unchanged.

```bash
python -m pip install -e '.[test,cuda]'
python -m pytest -q tests/test_cuda.py
python examples/smoke.py --device cuda --backend fla --chunk-size 64
python scripts/benchmark.py --device cuda --backend fla --length 1024 --backward
```

For a local FLA checkout, add its root to `PYTHONPATH`. The CUDA checks report a
skip rather than a success if CUDA/FLA is absent. The measured environment and
actual test outcomes are recorded in `docs/VALIDATION.md`.

## Minimal API

```python
import torch
from gpa import DenseGPA, GPAConfig

model = DenseGPA(GPAConfig(
    heads=4, key_dim=32, value_dim=48,
    slots=(64, 64, 64, 128),  # per-level capacities; unequal is supported
    positional_mode="gated", chunk_size=1, backend="torch",
))
q = torch.randn(2, 16, 4, 32)
k = torch.randn_like(q)
v = torch.randn(2, 16, 4, 48)
out, state = model(q, k, v)  # out: [2,16,4,48]
out.square().mean().backward()  # includes learnable initial states and R
next_out, state = model(q[:, :1], k[:, :1], v[:, :1], state)
```

Inputs are already projected Q/K/V, time-first `[B,T,H,D]`. The caller supplies
an LM's projections, output projection, FFN, packing/padding, and generation
adapter. There is **no implicit padding mask**: padded tokens would update
memory. Use equal-length unpadded streams or separate caches. Start with `None`
for a new document, not for every chunk. To truncate BPTT explicitly use
`state = state.detach()`; this never happens automatically. Independent batch
items have independent states, including relation matrices.

`return_diagnostics=True` additionally returns per-level route tensors, edge
indices, and logical cache element count. Route histories cost O(T sum M_i)
only when requested. Checkpoint a model with `model.state_dict()` and retain
the same `GPAConfig`; caching does not alter parameter names or capacities.

## Five propositions → code

| Proposition | Reference in `src/gpa/propositions.py` | What is / is not guaranteed |
|---|---|---|
| 1. Redundant joint KV patterns | `ExactKVCache` | Bitwise-equal **joint K,V** types and exact integer counts; +log(count) reproduces softmax read. Not cosine clustering; K-only merging is wrong. |
| 2. Content/position separation | `DecoupledRoPESequence` | Exact reconstruction from shared embedding, token indices and positions; metadata is still O(T). |
| 3. Position compression | `PositionalMomentCache`, `exponential_taylor_bound` | First-order grouped RoPE moments and a remainder bound; unrestricted long positions are not guaranteed. Invalid negative masses raise by default. |
| 4. Interlayer conditional capacity | `analyze_interlayer_similarity`, `propagate_capacity` | Measures matching and audits injectivity/separation; rejects collisions instead of asserting an unconditional fixed bound. |
| 5. Residual state decomposition | `ResidualCodebook`, `contraction_error_bound` | Product counts index tuples, not automatically distinct reconstructions; contraction bounds require actual contraction assumptions. |

`counted_slot_step` additionally implements the manuscript's count-normalized
soft centroid update. It is **separate from** the GDN state update. References
are intentionally small/inspectable; pairwise audits and enumeration are not
part of the linear-time attention path. Type deduplication has nondifferentiable
selection; this is not claimed to preserve independent per-occurrence gradients.

## Integrated model

For level i, capacities M_i and head dimensions Dk,Dv:

* `A_i: [B,H,Dk,M_i]`, `B_i: [B,H,M_i,Dv]`: learnable-prefix inner/outer GDN states.
* `n_i: [B,H,M_i]`: soft occurrence count; optional position moment
  `P_i: [B,H,Dk,M_i]`.
* **Every** i != j owns `R_ij: [B,H,M_i,M_j]`: a full dynamic GDN matrix,
  including j < i. It is not a static projection or a low-rank factor.
* `slot_i = center_i @ k_i` is a raw linear projection. No old log-sigmoid /
  expm1 slot approximation is used.
* Gates and beta are scalars per token/head `[B,T,H]`. With scalar alpha,
  transposing a state does not create a different gate axis; vector K/V gates
  require a different operator and are deliberately not accepted here.

The precise write, residual, count, relation, and read formulas are in
[EQUATIONS.md](docs/EQUATIONS.md), with code identifiers. In brief:

```text
for each token:
    for each level:                         # hierarchical STORAGE only
        slot = center @ residual_key
        update A and B by GDN; update counts
        reconstruct K and V separately; subtract to form next residual
    compute local read probabilities from ALL updated A states
    update every full R_ij using write assignments; read using local probabilities
    aggregate all incoming messages simultaneously
    q_out_i = refined_route_i W_rs_i + (q/16) W_qs_i
    output = learned weighted sum of normalized(q_out_i) @ B_i
```

Relation writes, K/V writes, and storage residuals never depend on Q. The
read does not run a previous-level-guides-next-level feedback loop. Scalar
beta <= .9, normalized GDN keys, nonpositive log gates, bounded relation
corrections, and bounded residual step sizes give controlled updates, but
they are not a universal convergence theorem.

`positional_mode="gated"` uses learned decay, no explicit RoPE. `"moments"`
adds query RoPE and additive content/position moments as an engineering
surrogate. `"none"` disables all decay and moments. Counts are active in all
three. The integrated moment reader is not identical to the P3 reference.

## Recurrent versus chunk-prefill semantics

For uniform capacities in chunk mode, select `execution="packed"` to pack
independent levels/edges into the head axis, or `execution="triton"` to also
fuse dense message aggregation, normalization and route softmax. All edges and
block boundaries are retained. Nonuniform capacities or tokenwise mode keep
the reference path; do not claim a fused-path speedup in those cases.
`execution="staged"` additionally schedules eligible aligned prefixes by level,
using differentiable identity probes to expose block-boundary states. It retains
the same lagged model and falls back for unsupported shapes/partial blocks;
see [staging semantics](docs/DENSE_V3.md#6-same-model-level-staging-and-backward).

```python
cfg = GPAConfig(heads=4, key_dim=32, value_dim=32, slots=(16,)*4,
                chunk_size=64, backend="fla", execution="triton")
model = DenseGPA(cfg).cuda()  # FP32 master weights + bf16 autocast is supported
```

* `chunk_size=1, backend="torch"`: the defined post-update residual recurrence.
  Autograd preserves gradients through all tokens and initial states.
* `chunk_size=C>1, backend="torch"` or `"fla"`: **block-lagged residual
  approximation**. Each block uses its start-of-block A/B snapshots only for
  residual reconstruction/write assignment. GDN recurrences within the block
  remain causal; the relation read is still dense. This permits parallel public
  GDN scans without storing every token's entire state.

The latter is not asserted equal to `chunk_size=1`. Crucially, caches retain a
partial block's original snapshots and offset, so token decoding and arbitrarily
split external calls use the **same block boundaries**, not a new approximation
on every call. Do not change `chunk_size` mid-cache. FLA and torch tests compare
the **same** block-lagged model, not different residual definitions.

## Complexity

For fixed heads/dimensions/capacities and fixed block size C, runtime is O(T).
The tensor cache per batch/head (without optional block snapshots) has

```text
sum_i [Dk*M_i + M_i*Dv + M_i + optional(Dk*M_i)]
    + sum_{i != j} M_i*M_j
```

elements. Dense relations cost O(L² M²), not O(L M); increasing the number of
levels is not free. Prefix parameters have the same main/relation dimensions;
bridge/route MLP parameters add per-level quadratic terms. Full training
activations are not O(1) memory in T. FLA uses fixed-size local temporal tiles,
not a full T×T attention matrix. Counters still require growing numerical range
(exact arbitrary counts need O(log T) bits); fixed tensor shape is not an
unbounded-information guarantee.

## Release

### Reproduce optimization / synthetic retrieval experiments

```bash
python scripts/compare_efficiency.py --cuda-graph --output efficiency.json
python scripts/train_retrieval.py --steps 400 --output runs/gpa
python scripts/train_retrieval.py --attention transformer --steps 400 --output runs/transformer
python scripts/train_retrieval.py --resume runs/gpa/checkpoint-400.pt --steps 2000 \
    --easy-curriculum --cuda-graph --output runs/gpa_curriculum
python scripts/analyze_retrieval.py runs/gpa_curriculum/checkpoint-2000.pt --plot

# Explicit fresh reproduction of the selected V3 configuration:
bash scripts/run_selected_v3.sh
# Optional faster training schedule (different BF16 reduction order):
# GPA_EXECUTION=staged GPA_OUTPUT=runs/selected_staged bash scripts/run_selected_v3.sh
```

The LM is deliberately small (width 128, two blocks); it is not pretrained
Qwen or a reproduction of a paper-scale LM. Tasks mix structured single-value
lookup, random four-token identifier values, MQAR, and last-write-wins. The
UUID-style task is **not** official RULER NIAH3 or a natural-language UUID
benchmark. Evaluation seeds/records are independent of training; report both
answer-token accuracy and whole-query exact match. A Transformer with the same
width/depth/short convolution is a control, not an exact parameter match.
Training `read_scale=8, beta_init=.6, decay_init=.0001` differs explicitly from
the default attention-core initialization. First-use compilation and graph
capture must be excluded from steady-state timing and reported separately.

```bash
python scripts/build_release.py
```

Produces `dist/gpa-attention-0.3.0-source.zip` with a per-file SHA256 manifest
and archive checksum. It excludes checkpoints, data, the manuscript, machine
addresses, private logs, environments, and dependency source. See
[RELEASE.md](docs/RELEASE.md) for uploading to `fisalt/GPA`. A license must be
chosen by the rights holder before public redistribution; none is silently
assigned by this packaging work.
