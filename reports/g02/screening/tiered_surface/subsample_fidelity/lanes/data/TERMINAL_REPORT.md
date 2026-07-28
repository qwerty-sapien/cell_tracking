# Video-subsample fidelity — data lane terminal report

**SUBSAMPLING_UNSAFE**. No cell passed all thresholds.

The scorer-only replay authenticated six banked candidate graphs against one common CG0 reference on the exact 36-video held-44b6 universe and reproduced every cited aggregate to absolute tolerance `1e-12` before any draw. APNCC's CV57 and SA-DELETE's five extra 44b6 rows were retained only as provenance diagnostics, never mixed into ranks.

Exactly 10,000 draws were run for each of 12 `(k, scheme)` cells. `family_residual` is reported as the preregistered one-family degeneracy and cannot beat `residual_top10` on a tie.

## Full common-36 deltas

| candidate | delta vs CG0 |
|---|---:|
| APNCC | +0.006949544 |
| C1 | -0.000653390 |
| H1-ZWRFEAT | -0.004198532 |
| SA-DELETE | -0.000281351 |
| SA-SWAP | +0.006639257 |
| NODETOPO | -0.000003436 |

## Fidelity results

| k | scheme | median rho | pooled sign agreement | maximum candidate 95% width | usable |
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

The preregistered rule requires rank `|rho| >= 0.8`, sign agreement `>= 0.9`, and maximum candidate 95% width `< 0.005`. At `k=12`, all three schemes clear only the rank threshold. The strongest observed sign agreement is `0.820683`, while the narrowest maximum width is `0.023017`, 4.60 times the gate width. C1 is the width-limiting candidate at `k=12` (about `0.023`); its small full-family delta changes sign frequently under video subsampling. Therefore no fixed video list or `SUBSAMPLE_CONTRACT` is emitted.

## Validation

- Common universe SHA-256: `855947ecd1720ce1be9a2803bc714a2133be29341971d8b9fddee892d6caaa61`.
- Eight aggregate-reproduction checks passed with zero observed absolute error at tolerance `1e-12`.
- Each cell records the exact uint64 seed, candidate order, allocation, draw-zero IDs and weights, complete draw-matrix SHA-256, and estimate-matrix SHA-256.
- Six estimator unit tests passed.
- A clean second execution reproduced all deterministic scientific artifacts byte for byte.
- The 45-second preflight, banked-fragment replay guard, and final draw guard all passed with zero swap growth. Peak process-tree RSS was `0.5824 GiB` for replay and `0.0398 GiB` for the final draws.

No model inference, graph construction, new family/split, GPU, Kaggle, RunPod, budget debit, submission, tier move, DEAD action, or root-owned state write occurred. This lane proposes state and ledger fragments only. Every promotion still requires the full frozen 44b6-first graph gate.
