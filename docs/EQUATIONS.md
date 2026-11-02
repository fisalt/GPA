# Exact implemented equations and shapes

All equations below are per stream and head; batch/head axes are omitted.
Vectors use column convention. Stored states in code are **key-major**.
`u(x)=x/max(||x||_2,1e-6)`. Standard floating-point roundoff still applies.

## 1. Common GDN primitive (`ops.gdn_update`)

For S ∈ R^(K×V), key k ∈ R^K, value v ∈ R^V, scalar log gate g <= 0,
scalar beta ∈ [0,1]:

```text
alpha = exp(g)
S_bar = alpha S_prev
e = v - u(k)^T S_bar
S_new = S_bar + beta u(k) e^T
y = u(q)^T S_new
```

Equivalently `S_new = alpha (I-beta u(k)u(k)^T) S_prev + beta u(k)v^T`.
This is a key-major convention, the transpose of a value-major equation.
The scalar gate has no distinguishable K versus V axis. A vector gate would;
this API rejects vector gates rather than pretending a transpose is harmless.

`gdn_scan` applies exactly this definition causally. Its torch implementation
is the reference; its optional FLA implementation explicitly normalizes Q/K,
sets `scale=1`, and requests the final state. State arithmetic is float32
(float64 in CPU double checks), even for bf16/fp16 input. No implicit detach.

## 2. Level storage (`DenseGPA._token`)

For level i, let M_i be its slot count, Dk and Dv the input dimensions:

```text
A_i ∈ R^(Dk×M_i), B_i ∈ R^(M_i×Dv), n_i ∈ R^M_i
C_i ∈ R^(M_i×Dk), W_rs_i ∈ R^(M_i×M_i), W_qs_i ∈ R^(Dk×M_i)
k_0 = k_t, v_0 = v_t
s_i = C_i k_i                            # raw linear slot projection
(g_i,beta_i,g_o,beta_o) = gates_i(u(k_i))
A_i,t = GDN_update(A_i,t-1, k_i, s_i, g_i, beta_i)
B_i,t = GDN_update(B_i,t-1, s_i, v_i, g_o, beta_o)
a_i,t = softmax(u(k_i)^T A_i,t)            # write assignment, length M_i
n_i,t = n_i,t-1 + a_i,t
k_reconstruction_i = A_i,t a_i,t
v_reconstruction_i = a_i,t^T B_i,t
k_i+1 = k_i - eta_k_i k_reconstruction_i
v_i+1 = v_i - eta_v_i v_reconstruction_i
eta_{k/v,i} = 1 + .25 tanh(residual_raw_{k/v,i})
```

`gates_i` is a per-head affine map Dk→4. Decays are
`g=-min(softplus(raw_g),10)` and `beta=.9*sigmoid(raw_beta)` by default.
Initial beta=.1 and alpha=exp(-.001). Gates are [B,T,H], not vectors.
The initial A/B are **learned asymmetric prefixes**, not `None` or identical
zero slots. C is learned independently at every level.

Counts are soft occurrence counts, not an exact equivalence-class count after
GDN writes. Key and value residuals are reconstructed separately; there is no
subtraction of a Q-dependent final attention output from K or V. All storage
updates are independent of Q, which is covered by a test.

## 3. Local read and optional positional moments

`q_star=q` for `gated`/`none`. For `moments`, `q_star=RoPE(q,t)`, and an
additional additive moment is kept:

```text
P_i,t = P_i,t-1 + [RoPE(u(k_i),t)-u(k_i)] a_i,t^T    # Dk×M_i
z_i = u(q_star)^T P_i,t
ell_i = u(q_star)^T A_i,t + softplus(c_i) log(1+n_i,t)
ell_i += moment_gain_i tanh(z_i/(1+n_i,t))          # moments mode only
p_i = softmax(ell_i)
```

This bounded moment correction is an engineering integration of P3's idea.
Version 0.2 exposes positive `read_scale` (default 1), multiplying only the
content term `u(q_star)^T A_i,t` before count/moment biases. The small retrieval
experiment uses 8; this is an explicit hyperparameter, not a change to the
default reference equations.
The separate `PositionalMomentCache` implements the actual first-order grouped
exponential formula, with nonpositive-mass detection. The two are not asserted
algebraically identical. `none` zeroes both main and relation decays; counts
and content-based updates remain active.

## 4. Dense dynamic relation matrices (`DenseGPA._relations`)

There is a full R_ij ∈ R^(M_i×M_j) for every ordered pair i != j:

```text
R_ij,t = GDN_update(R_ij,t-1, a_i,t, a_j,t, gR_ij, betaR_ij)
m_i→j = u(p_i)^T R_ij,t
gR_ij = -softplus(relation_decay_raw_ij)
betaR_ij = sigmoid(relation_beta_raw_ij)
```

R prefixes are learned, initially zero. Initial relation beta=.6 and decay
=.0001. Gates are learned per edge/head, constant across tokens in this
release; the matrix contents change at every token. No low-rank R factor,
static-only route projection, or triangular sparsity is used. Relations write
with **write assignments a**, but read with **query-dependent local routes p**.

After **all** local reads are available, for each target j:

```text
w_ij = softmax_over_sources(edge_logits_ij), i != j
m_j = sum_i w_ij m_i→j
c_j = m_j - mean_slots(m_j)
b_j = relation_gain_j tanh(c_j / sqrt(mean_slots(c_j^2)+1e-5))
r_j = softmax(ell_j + b_j)                # if no edges: softmax(ell_j)
r_j = r_j + mlp_gain_j [GELU(r_j U_j) D_j]
qout_j = r_j W_rs_j + (q_star/16) W_qs_j
y_j = u(qout_j)^T B_j,t
o_t = sum_j softmax_over_levels(output_logits)_j y_j
```

Each message uses the original p_i, not an already mixed route. This is one
simultaneous/Jacobi dense read, not a level-order-dependent feedback sweep.
The final residual route after its MLP need not be a probability vector; it
is normalized for the outer GDN read. For L=1 the relation step is empty.

## 5. Chunk approximation (`DenseGPA._block_segment`)

For a fixed block of C>1 tokens, freeze only residual snapshots
`A_frozen_i, B_frozen_i` at block start. Within that block replace:

```text
a_i,t = softmax(u(k_i,t)^T A_frozen_i)
k_i+1,t = k_i,t - eta_k_i A_frozen_i a_i,t
v_i+1,t = v_i,t - eta_v_i a_i,t^T B_frozen_i
```

All A/B/R recurrences, cumulative counts, and optional moments still include
each token causally. Q-dependent base logits use the updated inner state via
the GDN scan, not A_frozen. Two scans per level compute the inner logits and
outer outputs; extra GDN scans compute dense R messages. This is why the
public FLA chunk GDN can be used without materializing every full state at
every timestep. It is a **different residual approximation** from section 2.

At an incomplete block boundary, cache the original frozen snapshots and
offset. The next call resumes the same block. At a complete boundary discard
the snapshots and freeze the newly completed states. This makes external
call partitioning irrelevant to the defined approximation (modulo numeric
roundoff). Cached snapshots retain their autograd graph until explicit detach.

## 6. Standalone proposition references

P1 exact compressed read, for identical joint KV types c:

```text
o = sum_c softmax_c(q^T k_c/sqrt(Dk) + log n_c) v_c
```

P2 reconstructs `RoPE(E[index_t],position_t)`; the length-T indices are not
discarded. P3 stores `D_c=sum_occurrences(R_pos k_c-k_c)` and approximates:

```text
w_c = exp(q_rot^T k_c/sqrt(Dk)) * [n_c + q_rot^T D_c/sqrt(Dk)]
o_approx = sum_c w_c v_c / sum_c w_c
```

The first-order bracket can be nonpositive. Clamping it is a separately named
approximation, not the theorem. The Taylor envelope is
`.5 exp(a+abs(delta)) delta^2` for `exp(a+delta)-exp(a)(1+delta)`.

P4 conditionally propagates `M_l <= min(M_l-1/p_l,T)`; p=0 gives T.
`analyze_interlayer_similarity` reports when the injectivity/separation
preconditions cannot be certified instead of pretending the bound applies.

P5 reconstructs `x_hat=sum_i C_i[index_i]`, `r_i=r_i-1-C_i[index_i]`.
The number of index tuples is `product_i M_i`; the number of distinct sums may
be smaller. If each stage actually contracts by rho_i<1, its residual bound
is `||r_L|| <= product_i rho_i ||x||`. This is an assumption, not universal.

Finally `counted_slot_step` implements the manuscript-style soft centroid rule:

```text
f = activation(feature @ memory_prev^T)
count = count_prev + f
memory = memory_prev + [f/max(count,1)]^T * (feature-memory_prev)
residual = feature - f @ memory
```

It is tested separately and is not silently substituted for GDN.

## 7. v0.2 scan packing and route fusion

Uniform-slot lagged blocks can prepare storage residuals before any current
inner scan. Pack level into head (`L*H`) for inner/outer scans and edge into
head (`L*(L-1)*H`) for relation scans. The number of scan calls falls from
`2L+L(L-1)` to three (two for L=1). No edge, residual level or chunk boundary
is removed. Nonuniform slots and tokenwise post-update residuals retain the
reference execution path.

`triton_route.py` fuses incoming-edge weight softmax, weighted messages,
centered variance normalization, tanh correction and route softmax. Its
analytic backward computes all base, message, edge-logit and gain gradients.
Each edge has one target/writer; parameter-gradient partials are reduced over
batch/time with PyTorch sums. It is not a claim that the whole model is one
kernel. The route accumulation uses FP32 (FP64 in CPU references), avoiding
intermediate BF16 roundoff of the initial reader. Actual GPU discrepancies
are reported, not described as bitwise equality.

The optional `shared_query_center` parameterization uses the effective
`W_qs=C^T/query_scale+Delta_W`, adding an anchored `q C^T` term and the
original scaled learnable delta. It changes training parameterization, not
state sizes; it is experimental and disabled by default. The retrieval pilot
initializes its delta to zero and route matrix to `.02*(I-11^T/M)`. Results
are mixed, not a proven fix; see the optimization report.
