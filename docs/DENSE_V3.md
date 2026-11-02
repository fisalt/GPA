# GPA v0.3: updated storage and theory-inspired DENSE reading

This is an experimental implementation, not a claim of achieved Transformer
parity. All results, including failures, are retained. The earlier v0.2
source archive remains unchanged. No live Qwen training code was modified.

## Sources inspected (2026-11-02)

The author-linked RSA discussion and its actual latest Qwen3/Qwen3.5 source
were checked, not just old ROOT modules. The relevant source fingerprints are:

* `qwen_rsa_fast_retrieval.py`: `b71aa43fdf3adeaaa03e3bbeeabfdf57046119501ae3814b0e2f3f87ef40e862`.
* `qwen_rsa_fast_relation_prepare.py`: `cb21040f5e94c5d2e67f1f2b2c38df308e080dd63d8e06c47956ba9603304663`.
* `modeling_qwen3_uuid.py`: `4d5f3849a2218f016485d238c8287026ce81decea2c6d850009858b4f29aad4c`.
* Author's current `neurips_2026.tex`: `d00aeb30ddae8529dddf193edc154f2ad95cef06c498bd82cae59f592a60be0e`.

The manuscript was read for its formulas, not to transfer any reported model
accuracy to this new implementation. No private RSA source is vendored here.

Latest RSA uses normalized paired relation writes, long-memory controls,
adaptive residuals and routes divided by a clamped L1 denominator. Its active R32 reader is adjacent
and low-rank. GPA below instead maintains **every ordered pair**, full M×M
matrices, and a simultaneous dense read. No top-k, adjacent-only guidance,
token Softmax Attention fallback, or answer-derived write mask is used.

## 1. Counted storage: explicit alternative to two-GDN approximation

`PaperDenseGPA` is intentionally a different storage operator from legacy
`DenseGPA`; their state classes cannot be silently interchanged. Let A_i be
key moments [Dk,M], B_i value moments [M,Dv], and n_i [M]. Positive assignments:

```text
p_i = softmax(s_i C_i phi(k_i)),      u_i = sigmoid(W_write x + bias)
w_i = u_i p_i
g_i = -tier_i * min(exp(A_log_i) softplus(W_decay x + bias), .02)
```

`phi(x)` is either L2 normalization (initial stability ablation) or identity
(`--paper-raw`). Full-control mode consumes the complete normalized hidden
input x, as in current RSA controls, not only a compressed per-head K.
No token roles/labels are supplied as controls. Four-level tiers are
1/.1/.01/.001; the eight-level schedule follows 1/.3/.1/.03/.01/.003/.001/.0003.
These are cold-start experiments, not abrupt rescalings of trained Qwen gates.

Baseline scalar forgetting has a_i=exp(g_i). The slot-decay candidate uses:

```text
a_i,m = exp(g_i) [1 - sigmoid(gamma_i) w_i,m]
n_t = a_t * n_prev + w_t
A_t = A_prev diag(a_t) + k_t w_t^T
B_t = diag(a_t) B_prev + w_t v_t^T
S_K = A_t^T / n_t[:,None];  S_V = B_t / n_t[:,None]
```

Thus each normalized slot is a convex update of its old centroid and current
feature with weights `(a*n_prev)/n_t` and `w/n_t`. A learnable nonzero count
prefix and asymmetric K/V prefixes supply initial memories. They are not
pretrained retrieval experts. Baseline scalar forgetting can average conflicting
values for the same key; slot-local decay explicitly tests write-selective recency.

**Gate directions matter:** A[Dk,M] is gated on its value/slot columns;
B[M,Dv] on its key/slot rows. The FLA implementations use `gv` for A and
`gk` for B, not a mistaken Dk-vector gate on both. Scalar mode uses chunk
simple-GLA; vector A uses fused-recurrent GLA and vector B chunk-GLA. Both
remain sequence-linear. Gradients are retained throughout; no automatic detach.

Within a fixed chunk, hierarchical residuals use start-of-chunk frozen
centroids, with independent bounded K/V step sizes:

```text
k_next = k_i - eta_K_i p_i S_K_frozen
v_next = v_i - eta_V_i p_i S_V_frozen
eta = 1 + .25 tanh(raw)
```

This is a stated lagged residual approximation, not equality with a
tokenwise post-update residual model. Block offset/frozen snapshots persist
across arbitrarily partitioned calls. Counts are computed with a stable
log-cumulative-sum expression for strong vector gates, never exp(-G) overflow.

## 2. Relation writes inherited and generalized from RSA

For **all i != j**, preserve full R_ij[M_i,M_j]. Let z=||p_i||_2^2+eps:

```text
k_R=p_i; v_R=p_j/sqrt(z)
beta_R=min(sigmoid(beta_raw)*z,.9)
log_alpha_R=-softplus(decay_raw)
R_t=GDN_update(R_prev,k_R,v_R,log_alpha_R,beta_R)
```

The GDN backend normalizes its key. Scaling beta and value is necessary;
using a constant .6 with normalized almost-uniform assignments over-writes
the relation. With eps→0 this is the unnormalized paired delta update
`alpha R + rho p_i (p_j - p_i^T alpha R)^T`. The implementation retains eps
as in the inspected RSA source; it is not advertised as an exact eps=0 identity.
Unlike RSA R32 there is no factor U E, no rank truncation, and no omitted
backward edge. Q-dependent reads never enter these state writes.

## 3. Dense reading, not RSA's sparse guide

Local evidence combines a query against the current K centroid and a direct
query-feature path, with a small trainable logit MLP:

```text
e_i = softmax(tau_i phi(q)^T A_i / n_i + s_i C_i phi(q))
```

The earlier `local` ablation sums log incoming messages in each destination
space and mixes that product with its own evidence. The stronger `transport`
reader first aligns all local evidences into **each possible reference space r**:

```text
a_j^r = softmax(M e_j R_jr),           a_r^r=e_r
b_(r,i) = normalize(e_r + lambda_r softmax(sum_{j != i} log a_j^r))
t_(r,i) = softmax(M b_(r,i) R_ri)
w_(r,i) = normalize(e_i * t_(r,i) * n_i^count_gain)
```

All roots and all non-self targets are computed, without an ordered level
feedback sweep. Mix the incoming w_(r,i) with the local counted evidence,
then read `route_i S_V_i` and aggregate levels. Positive routes satisfy the
bounded-L1 quality constraint by construction. Full relation matrices are
read twice for transport, but **updated only once**: pass 2 replays the same
initial R and same writes with a different query; only pass 1's final state
is persisted. Reverse associations are independently learned full R_ji,
not falsely assumed equal to R_ij^T.

This is a dimensionally explicit dense generalization inspired by the paper's
leave-one-out products and relation-transported values. It is not a literal
claim of equality with every shorthand in the manuscript or its approximation
theorem. No proof establishes Transformer-equivalent accuracy at these slot sizes.

## 4. Two-GDN updated control

`--rsa-update --aligned-query` retains legacy two-GDN state equations and
adds assignment-norm-corrected **full dense** relations, tiered dynamic gates,
slot assignment from Ck, and a clamped L1 bridge denominator. This clamp does
not guarantee a global bound on the resulting route norm. Its optional orthogonal
center initialization and 2×32 geometry test conditioning/rank at the same
64 total slots/head as 4×16. This control is separate from the counted
paper-transport reader; gains must not be misattributed between them.

`--full-controls` also works with the two-GDN model: four gate/beta logits
are projected directly from the complete hidden x, independently for each
head and level. Q/K/V projected width no longer limits write-control context.
Its default-off parameter shape is preserved for all older checkpoints.

### Positive compatibility product for the two-GDN reader

With `--paper-product-read`, keep the paired GDN storage and replace the old
center/variance/tanh message transform by the following dense product:

```text
e_i = softmax(base_i)
m_ij = e_i R_ij                         # query norm restored after GDN read
w_ij = softmax_over_all_sources(edge_logit_ij)
cross_j = M_j exp(clamp(gain_j,-4,4)) sum_(i != j) w_ij m_ij
route_j = softmax(base_j + cross_j)
        ∝ exp(base_j) product_(i != j) exp(m_ij)^(M_j exp(gain_j) w_ij)
```

This uses every incoming ordered edge simultaneously. Exponentiating each
compatibility makes the product positive even though GDN R can contain
negative entries. It is a theory-inspired product-of-experts parameterization,
not an equality claim to the counted-centroid manuscript model. The route
MLP/L1 calibration and query bypass then follow the documented two-GDN bridge.
The Triton route kernel implements both this mode and the older transform,
including gradients to base, every message, edge weights and gain.

`--relation-marginal` additionally calibrates the product against the empirical
target-slot marginal. Let pi_j=(1+n_j)/(M_j+sum n_j). Then:

```text
route_j ∝ exp(base_j) [ product_i exp(M_j m_ij)^w_ij / pi_j ]^G_j
G_j = exp(clamp(gain_j,-4,4))
logit_j = base_j + G_j (M_j sum_i w_ij m_ij - log(1+n_j))
```

The omitted log normalizer of pi is constant over the destination slots and
cancels in the final softmax. This tests a likelihood-ratio-style correction
for frequent noise slots, rather than reinforcing a common marginal once per
relation. GDN R is not a certified probability table; this interpretation is
an engineering potential model, not a proven mutual-information estimator.
The prefix counts follow the legacy assignment/count recurrence and are not
claimed equal to R's decayed empirical distribution.

The correction is fused into the same Triton forward/backward. With x the
softmax-input gradient, the additional derivatives include
`dcounts=-G*x/(1+counts)` and
`dgain_raw=G*sum(x*(M*incoming-log1p(counts)))` inside the unclamped range.
Base, message and edge gradients are retained. Counts and gain receive real
training gradients; no constants or detached diagnostic masks replace them.

### Branch calibration and contextual write controls

The old aligned-query initialization sets a centered route matrix to .02 I
while the shared direct Cq branch has unit scale. `--balanced-bridge` gives
the branches explicit, trainable, bounded norm calibration:

```text
u_j = route_after_MLP_j W_rs_j
d_j = phi((q/16) W_qs_effective_j)
z_j = .1 + .8 sigmoid(mix_raw_j)            # head/level, initially .5
qout_j = z_j unit(u_j) + (1-z_j) unit(d_j)
o_j = unit(qout_j)^T B_j
```

The route matrix starts at I in this mode to avoid normalizing an almost-zero
centered vector. The bounds ensure both nonzero branches remain represented;
they do not prove the model uses R effectively. `analyze_retrieval.py` therefore
compares identical inputs with all R read messages removed, and reports query
branch norms, route entropy, effective R ranks, and actual prediction changes.
The ablation keeps all storage writes unchanged.

The two nonlinear-feature ablations apply `softmax(4x)` or `tanh(4x)` symmetrically
to projected slots and the direct query feature. They retain the same state
dimensions and are opt-in, not retroactive changes to saved checkpoints.
The calibrated-output bundle additionally tests per-head RMS output scaling
and a learned SiLU gate before the LM output projection. These bundles must
not be called individually beneficial without a controlled comparison.

In a randomly initialized small LM, the raw hidden input of block1 has no
previous-token context. `--context-controls` separates a causal contextual
write controller from the retrieval Q/K features:

```text
c_t = SiLU(depthwise_causal_conv_8(LN(x))_t)
(g_inner,beta_inner,g_outer,beta_outer)_j = control_j(c_t)
```

The convolution initially averages its window. This can learn whether a value
occurs in storage or answer context; no parsed role/key/answer mask is supplied
to it, and its gradients come only from the same answer CE. Inference uses
exactly these observable tokens and learned parameters. Token roles are parsed
only by the separate diagnostics script for reporting. A CPU prefix-causality
and backward test covers the contextual controller, including nonzero conv
gradients after a nonzero gate projection.

`--quadratic-features` tests a shared nonlinear angular feature map for Q/K.
With u=unit(x), independent learned per-head matrices U,V and initially a=.25:

```text
a = 2 sigmoid(a_raw)
z = u + a sqrt(Dk) (u U) * (u V)
Phi(x) = sqrt(||x||²+1e-12) unit(z)
q <- Phi(q); k <- Phi(k)                  # same map, V tokens unchanged
```

This adds multiplicative key/position interactions before state writing,
useful when a two-token name and an identifier position must be bound together.
It is a learned compressed quadratic sketch, not an exact full tensor-product
kernel identity. State/slot dimensions stay fixed; projection work and extra
trainable Dk×Dk matrices are reported rather than treated as free capacity.
Norm and zero-input/gradient checks, double gradcheck, packed/reference parameter
gradients, and a CUDA recurrent-vs-FLA check cover this option. The optional
feature projections use PyTorch GPU/autograd under the same CUDA Graph; they
are not claimed to be a custom fully fused Triton feature kernel.

`--temporal-qk-init` changes only the initialization of the existing causal
Q/K convolution: half of each head's channels initially pool the window with
weights `1/sqrt(K)`, while the other half use staggered one-hot lag taps.
All taps remain trainable. The value convolution retains its current-token
initialization. This does not encode token IDs, labels or record positions,
add inference operations, or increase parameter count. The controlled
quadratic-to-temporal comparison improves validation UUID4 EM from 74.22%
to 90.625%. The final independent comparison is reported separately.

`key_dim` can exceed `width/heads` without changing V/output dimensions.
`--learn-read-scale` optionally learns per-head/per-level temperatures in
`[.25,64]`, initially 8. Neither a wider key, more slots, nor a learned
temperature improved on the selected compact model in these runs. These
options remain explicit and off in the selected configuration.

`--contextual-output` enables both the causal contextual controller and gated
per-head output normalization, and uses that controller as the read-gate input:

```text
h_t = sum_j softmax(level_logits)_j o_tj
y_t = W_O [ RMSNorm_per_head(h_t) * SiLU(W_G c_t + b_G) ]
```

This gate sits outside GDN's query L2 normalization: changing the magnitude
of a nonzero q alone cannot attenuate a head after q/||q||. The separate
contextual gate can learn identifier-order-dependent read contributions.
It receives ordinary causal hidden features, never an answer-position mask.
The diagnostics script records gate magnitudes at answer positions only
after inference, in addition to R and slot-route heatmaps.

**Legacy reconstruction caveat:** the lagged GDN residual uses frozen A p and
p^T B as learned feature reconstructions. A is a delta state, not the counted
centroid S_K. This must not be confused with the counted model's exact convex
mean interpretation. Changing to a tokenwise post-update residual would define
a different operator; current cache tests compare the documented fixed-block
approximation with itself.

## 5. Pure-CE comparison protocol

The author explicitly rejected extra key-alignment supervision: **no auxiliary
loss is used**. Mixture remains 30% single, 30% random four-token identifier,
20% MQAR, 20% overwrite by examples. Only answer tokens have next-token CE.
Early ablations use token-averaged answer loss, so their gradient weights
differ from example ratios. Later V2/V3 comparisons explicitly use the
**mean of each sequence's answer-token CE**, then batch mean, preserving
the requested example weights without adding any auxiliary objective.
No actual natural-language UUID, RULER or BABILong score is claimed.

All candidates start from random weights, not a trained retrieval checkpoint.
Standard group: two blocks, width128, H4, 6000 steps, B8, LR.002, clip1,
curriculum 200 steps T64/1 record, 200 T128/2, 200 T128/4, then T256/8.
The full-controls/slot-decay group uses 1000-step curriculum stages; its
Transformer receives that same schedule. The four-block/cosine group has
its own matched-depth and matched-optimizer Transformer. Do not compare
different curricula as equal token budgets merely because steps match.

Validation streams use seed900000, 128 examples/task/length (384 MQAR queries).
The early V2 alternative used held-out seed910000. The V3 candidate selection
was frozen before evaluating seed920000 (256 examples/task); seed930000 remains
unused. Later optional variants were compared on validation only, not selected
against seed920000. A rerun of the same frozen weights on the staged backend is
a numerical regression check, not a new model-selection result.
T512/12 records is extrapolation, not same-length training. Whole-query
and free identifier generation EM are reported separately from token accuracy.
New evaluations also report whole-sequence MQAR EM (all three queries), and
identifier accuracy at each of its four positions. Older stored JSON files
lack those fields; missing whole-sequence results must not be inferred by
raising per-query accuracy to a power.
The acceptance target is a gap of no more than five percentage points on
**each** T256 task, not a favorable average that hides failed identifiers.
For further trials require at least 95% on every task as well, so a poorly
trained Transformer cannot make the GPA acceptance target artificially easy.

The original and updated failures are retained in `docs/results/v3`. The
selected model's validation target flag remains **false**: UUID4 validation
EM is 90.625%, below the additional absolute 95% requirement. Independent
UUID4 EM is 93.36% versus 96.09% for the Transformer. This supports a small
observed gap, not 100% retrieval or statistical equivalence.

The evolving complete tables are generated into
[results/v3/summary.md](results/v3/summary.md), with configs/protocol in
[results/v3/summary.json](results/v3/summary.json). Check model width/depth,
curriculum, test seed and data version before attributing any difference to
one architectural change. A wider/deeper comparison has its own equally
wide/deep Transformer and is not a same-budget gain over the 128-wide pilot.

### Corrected overwrite data, versioned rather than silently replaced

* V1 rewrites one key but queries a random key, usually unchanged. Its
  reported overwrite scores are not evidence of reliable update retrieval.
* V2 queries the overwritten key and enforces a changed value, but that
  overwrite is the final record: a model can ignore the key and read the
  latest value. High V2 accuracy alone is also insufficient.
* V3 updates four distinct keys (or all keys when fewer than four), enforces
  changed values, and queries one random updated key. It must find that
  key's latest value; blindly taking the last record succeeds only by chance.

Unit tests verify the queried key, changed value, number of updates, and
that the target is not always the last record. V1/V2 remain selectable for
reproduction. The main V3 comparison trains both models from scratch for
20,000 steps, same seed/mixture/curriculum/sequence CE/cosine schedule.
GPA: 2×32 total64 slots/head, orthogonal centers, paired R, product read.
Transformer: same width128/H4/two blocks and short-conv frontend, plus standard
RoPE to represent update order. Parameter counts differ and are reported.
This experiment is distinct from every older 6000-step run.

## Reproduce

```bash
python -m pytest -q tests
python scripts/train_retrieval.py --memory-update paper_dense --paper-transport \
  --full-controls --slot-decay --data-version 1 --curriculum-span 1000 --steps 6000 \
  --easy-curriculum --cuda-graph --output runs/paper_slot --eval-samples 128
python scripts/train_retrieval.py --attention transformer --data-version 1 --curriculum-span 1000 \
  --steps 6000 --easy-curriculum --output runs/control --eval-samples 128

# Main corrected-data comparison, pure answer CE only:
bash scripts/run_dense_product.sh
python scripts/report_dense.py --root ..

# Explicit selected configuration, no prior ablation outputs needed:
bash scripts/run_selected_v3.sh
```

The GPU eager-vs-graph optimizer trajectory check compares gradients, losses,
and parameters across three real AdamW steps, not just one backward: its
measured gradient relative L2 was zero at all three steps. This ruled out a
large graph-replay training error in that check, not every possible runtime bug.

## 6. Same-model level staging and backward

`execution="staged"` changes scheduling, not the defined lagged residual.
For an aligned prefix of at least four chunks, equal per-level slot widths,
`Dk=M`, and a non-moment model:

```text
for level i:                              # over levels, not time chunks
    pair the A_i and storage-only B_i scans along the head axis
    append an identity-read probe block after each real block
    scan the whole augmented sequence by public GDN
    reconstruct next-level K/V from previous-block snapshots
scan all R edges once over the original sequence
compute dense route and balanced bridge
scan all actual B outputs once, packed over levels
```

Probe tokens have `key=value=0, beta=0, log_gate=0`, so their state transition
is the identity. Identity queries expose block-end states; they are not real
tokens and do not increment logical positions/counts or change the cache.
Their adjoints flow through the public GDN backward, without state detach.
Both main scans' initial states and all trainable prefixes receive gradients.
This costs extra probe work/activations but reduces public scan invocations
from `3*ceil(T/C)` to `L+2`. It is **not** a single fully fused kernel.

Unsupported shapes/modes, incoming partial blocks, and trailing partial blocks
retain the old packed schedule. `Dv` may differ from `Dk`; neutral temporary
padding is sliced back. Full aligned prefixes within longer calls still use
staging. Cache offset and frozen residual snapshots preserve original block
boundaries through arbitrary call partitions. CPU double tests compare outputs,
states, Q/K/V/control and parameter gradients; CUDA tests also cover prefix
causality, nonzero final-state adjoints and partial cached calls.

On the selected trained LM, B8/A100/BF16 with CUDA Graph, forward answer CE,
backward, clip1 and fused AdamW(lr=0) took 21.02→10.56 ms at T256 and
74.62→26.46 ms at T1024. This is 1.99×/2.82× faster than the prior GPA schedule.
The same-width Transformer remains faster (2.38/11.50 ms respectively).
Compilation, graph capture, data creation, logging and IO are excluded.
At T256, staged-vs-old logit relative L2 is .019% and parameter-gradient
relative L2 is .287%; BF16 accumulation order is different. See the full
[timing records](results/v3/efficiency_lm_staged.json) and
[Chinese report](REPORT_V3.zh-CN.md) for qualifications and final validation.
