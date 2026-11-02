# Source and implementation provenance

Prepared 2026-10-24 for the GPA authors.

* User-referenced article: https://chatgpt.com/s/cx_6adc2951cba4819196b37ad8cd90857e
* The article body was not accessible to this build session (empty extraction,
  interactive browser timeout). No claim is made to have verified that body.
  The author subsequently explicitly confirmed that the local source below is
  the latest version corresponding to the shared article and authorized using it.
* Author-confirmed source: local `neurips_2026.tex`, modification time 2026-10-24
  11:36:58 local time, SHA256
  `1c5241e1af5bcddd606d17ba8a4f8cac4fbbf420d56aeddcc3baee1c0fea56db`.
  The five-proposition ordering and caveats follow this version, not an older
  September extracted text.
* The code URL in that manuscript is https://github.com/fisalt/GPA. It was
  publicly readable but empty when inspected for this packaging task.
* User-authorized architectural change: full, dynamic all-to-all relation
  matrices instead of adjacent or one-way hierarchical read guidance.
* Source in this package was written for this release; existing training code
  and dependencies were not copied into the distribution. Public FLA operators
  are optional external dependencies. This is not a redistribution of custom
  remote FLA kernels or a port of private checkpoints.

## Important manuscript/engineering boundaries

1. Joint bit-pattern KV deduplication is exact as a finite-prefix function;
   cosine merging and practical fixed-slot GDN are not the same theorem.
2. Content-position decoupling retains O(T) indices until further compression.
3. First-order rotary moments need a controlled regime and can yield invalid
   masses. The integrated gated/moment alternatives are surrogates.
4. The conditional type bound is not certified if matching collisions or the
   stated separation assumptions fail.
5. Residual codebook tuple count is not automatically an injective sum map;
   a contraction product is conditional, not a universal approximation rate.
6. The integrated core deliberately learns asymmetric initial memories. Zero
   identical soft-assignment slots otherwise remain symmetric.
7. `chunk_size>1` changes the residual definition to a documented lagged
   approximation. Backend parity tests do not turn it into an identity with
   the tokenwise model.

No paper benchmark numbers are transferred to this newly written model.
No license is inferred from access to the manuscript or from the licenses of
unrelated projects. The owner must choose the publication license.

## 2026-11-02 update

The latest local source was read again, SHA256
`d00aeb30ddae8529dddf193edc154f2ad95cef06c498bd82cae59f592a60be0e`.
The author-linked latest RSA discussion and actual implementation were also
inspected for control and paired-relation changes. Source fingerprints,
new engineering approximations, pure-CE protocol, and corrected overwrite
data are recorded in [DENSE_V3.md](DENSE_V3.md). Private RSA source and
pretrained retrieval weights are not included in this distribution.
