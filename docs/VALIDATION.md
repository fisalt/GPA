# Initial v0.1 validation record — 2026-10-24

Current v0.3 results are recorded separately in
[VALIDATION_V3.md](VALIDATION_V3.md): 116 passed on the single-A100 suite.
The historical results below are preserved unchanged.

These are implementation checks, not pretrained-LM retrieval results. No
NIAH/RULER/MQAR/BABILong accuracy or paper-table reproduction is asserted.

The expanded v0.2 scheduling, fused Triton and retrieval experiment results
are recorded separately in [OPTIMIZATION.md](OPTIMIZATION.md). Counts below
are the initial reference implementation's validation, not the expanded suite.

## CPU

Environment: Windows, Python 3.11.3, PyTorch 2.0.1+cpu, pytest.

```bash
python -m pytest -q
```

Result: **32 passed, 4 skipped** (the four CUDA tests require a GPU).

Coverage includes exact prefix-wise joint-KV deduplication; bitwise signed-zero
distinction; explicit overflow; content/position reconstruction; controlled
first-order moment approximation and its invalid-mass failure; conditional
capacity collision audit; residual telescoping and duplicate sum codebooks;
count-normalized centroid gradients; direct GDN recurrence equations;
two-stage GDN equivalence; dense reverse edges affecting output; prefix
causality; arbitrary cached-call partitioning in both recurrent and lagged
chunk modes; gradients through cached calls, initial states and R; explicit
detach; fixed tensor-cache size; diagnostics; single-level operation; strict
state-dict roundtrip; four/eight levels; unequal 128/256/512 slot capacities;
double-precision input gradcheck; reproducible ZIP and per-file manifest.

CPU double gradcheck uses eps=1e-6, atol=1e-4, rtol=1e-3. Cache partition
checks in double use atol=rtol=1e-10. These are comparisons within the same
residual definition, not chunk-size=1 versus chunk-size>1 equivalence.

### Numerical proposition example

`python examples/propositions.py`, fixed seed 0:

| Check | Observed |
|---|---:|
| P1 compressed vs original softmax max absolute error | 1.1102230246251565e-16 |
| P1 joint types / tokens | 3 / 6 |
| P2 exact reconstruction error | 0 |
| P2 indices + positions stored | 12 elements for 6 tokens; not constant memory |
| P3 controlled low-frequency small-angle error | 5.709151033872484e-9 |
| P4 certified fixture capacity bound | 3 |
| P5 index tuples / distinct summed codewords | 4 / 3 |
| P5 telescoping residual error | 0 |

P3's number is for the controlled fixture, not an unrestricted-context bound.
P4's fixture deliberately satisfies the conditions; the tests also include a
collision counterexample that is rejected.

### Optimizer and cache smoke examples

`python examples/smoke.py`, fixed fixture, seed 42, 20 AdamW steps at lr=.01:
MSE **0.9177752 → 0.6971294**, all gradients finite. This is same-token fixture
reconstruction, not a long-range retrieval experiment or generalization claim.

`python examples/cache_decode.py`: max full-vs-partitioned error
**1.49e-8** in float32, including a partial lagged block.

## CUDA / FLA

CUDA tests are separately run on a verified idle NVIDIA A100-SXM4-40GB, with
PyTorch 2.9.0+cu128, Triton 3.5.0 and the existing local FLA installation. No running model
training is stopped or changed for these checks.

The checks use BF16 Q/K/V and FP32 state arithmetic, B=1, T=128, H=2,
C=64, both gated and moment variants. Dimensions `(Dk,Dv)=(32,32)` with
slots `(16,32)`, and a tiny-width `(8,8)` with slots `(4,7)` exercise the
zero-padding compatibility path. Backend comparisons hold **the same** lagged
residual semantics. They check forward outputs, Q/K/V gradients, learned
prefix gradients, final relation states, and cached partitions at 47/65/128.

Acceptance thresholds: relative L2 forward/cache/relation-state error < .06,
input-gradient relative L2 error < .15, plus forward/cache atol=.025,
rtol=.06 elementwise. Loose BF16 ceilings are not the measured errors; measured
results are listed below.

The initial CUDA run passed its two ordinary-width parity cases but exposed
an FLA tensor-core backward width constraint in the tiny optimizer example.
The release fixes this by padding reduction/value widths below 16 with zeros
and slicing outputs/states back to logical sizes. No dependency kernels were
edited and no logical slot capacity was changed.

### Final CUDA results

**4 passed**. Measured relative L2 errors (BF16):

| Mode / dimensions / slots | Forward | dQ | dK | dV | Cached partitions |
|---|---:|---:|---:|---:|---:|
| gated / 32 / 16,32 | .0057021 | .0081268 | .0073954 | .0055602 | .0054412 |
| moments / 32 / 16,32 | .0056990 | .0082263 | .0075604 | .0054978 | .0054743 |
| gated / 8 / 4,7 | .0084799 | .0133207 | .0093657 | .0076946 | .0076759 |
| moments / 8 / 4,7 | .0088325 | .0141059 | .0101347 | .0074875 | .0070997 |

The tiny-width case passes forward/backward after neutral zero padding.
The FLA optimizer smoke example also completed 10 finite steps: MSE
**0.8807613 → 0.7752058**. The larger CPU/CUDA numerical fixtures differ;
their losses are not a backend-equivalence comparison.

FLA emits its input-layout heuristic warning when an external call leaves a
one-token remainder (`T=1 < H=2`). The tensors are explicitly time-first and
the cache parity checks pass; no warning suppression was added.

### Small throughput smoke measurement

`scripts/benchmark.py --device cuda --backend fla --chunk-size 64 --length
256 --backward`, with its default H=2, Dk=Dv=32, four 16-slot levels, B=1:
mean forward+backward **0.272708 seconds** over three timed iterations after
two warmups; peak PyTorch allocated memory **25,525,760 bytes**. This excludes
first-use Triton compilation and excludes optimizer/LM projection/FFN costs.
It is only a small-shape execution/measurement check, not proof of speed
parity with an optimized GDN/RSA/Transformer baseline. Many Python-level
scan calls still incur dispatch overhead in this reference integration.

## Packaging

The wheel builds offline with `pip wheel --no-deps --no-build-isolation
--no-index --wheel-dir dist .`; installing it into an isolated target and
importing `gpa` succeeds. The source ZIP is the primary deliverable because it
also contains proposition tests, examples, equations, provenance and release
instructions. The wheel alone does not bundle those project directories.

`tests/test_distribution.py` builds twice and checks byte-for-byte ZIP
reproducibility, absence of weights/manuscripts/pycache, and every manifest
digest. The archive checksum is separate from its internal per-file manifest.
