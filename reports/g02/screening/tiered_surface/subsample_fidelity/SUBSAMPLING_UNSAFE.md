# Video-subsample fidelity verdict

Decision: **SUBSAMPLING_UNSAFE**

No tested `(k, scheme)` satisfies the preregistered screening-fidelity rule. No
`SUBSAMPLE_CONTRACT` exists.

The analysis standardized APNCC, C1, H1-ZWRFEAT, SA-DELETE, SA-SWAP, and NODETOPO
on the already-opened common 36-video held-44b6 gate and the common CG0 reference.
All banked fragment hashes and all eight aggregate replay checks matched to absolute
tolerance `1e-12`. Each cell used 10,000 fixed-seed finite-population draws.

| k | scheme | median rho | sign agreement | max 95% width | usable |
|---:|---|---:|---:|---:|---|
| 3 | simple random | 0.485714 | 0.682850 | 0.062612 | no |
| 3 | residual top-10 | 0.485714 | 0.685950 | 0.058475 | no |
| 3 | family + residual | 0.485714 | 0.685783 | 0.057514 | no |
| 5 | simple random | 0.600000 | 0.736083 | 0.044305 | no |
| 5 | residual top-10 | 0.542857 | 0.717567 | 0.048025 | no |
| 5 | family + residual | 0.542857 | 0.714667 | 0.049040 | no |
| 8 | simple random | 0.771429 | 0.777283 | 0.031278 | no |
| 8 | residual top-10 | 0.771429 | 0.776333 | 0.031281 | no |
| 8 | family + residual | 0.771429 | 0.775533 | 0.031494 | no |
| 12 | simple random | 0.828571 | 0.820683 | 0.023057 | no |
| 12 | residual top-10 | 0.828571 | 0.813733 | 0.023017 | no |
| 12 | family + residual | 0.828571 | 0.814883 | 0.023308 | no |

The acceptance line was `|rho| >= 0.8`, sign agreement `>= 0.9`, and maximum
candidate 95% width `< 0.005`. At `k=12`, all schemes pass rank only. Sign agreement
misses by at least `0.079317`, and the narrowest maximum interval is `0.023017`, more
than 4.6 times the gate width. Residual stratification does not rescue fidelity.
Because the amended universe contains only 44b6, `family_residual` is a degenerate
duplicate of `residual_top10`.

An independent implementation exactly matched the common universe, source hashes,
aggregate replays, and all simple-random metrics. Its stratified sensitivity used raw
`FP+FN` rather than the preregistered adjusted-edge residual mass
`(TP+FP+FN) * (1-adj_edge_jaccard)`; root reconciliation retained the preregistered
definition. The alternate strata also produced `SUBSAMPLING_UNSAFE`.

This is a screening diagnostic only. It moves no tier, revisits no DEAD mechanism,
and changes no promotion rule. Every promotion still requires the full frozen
44b6-first graph gate.

Authoritative evidence:

- `lanes/data/fidelity_tables.json` — SHA-256
  `67cf33ed6cf6da351b8f2d1a4c42562c6b5637ef75354c149c12ef4eb2c9a3b9`
- `lanes/data/common36_decomposition.json` — SHA-256
  `21c25cac997511493810757bf17a8cbf73d73be8a991a9ad076b697905064777`
- `lanes/data/verdict.json` — SHA-256
  `651c601212b6471f7cf4b916a3765b0256d4c3f09f62299fa817472c3c7299a1`
- `lanes/validation/amended_validation_checks.json` — SHA-256
  `12dcdcb26747a14c97d66313e882597b64591f52eb5f1acc6c49ca5eb7dbbb6f`
- `lanes/literature/METHOD_CITATIONS.json` — SHA-256
  `a6ef08b0116c6666656a6263ca9ddef1dd258e8204da7f828ec439b07ee389e9`
