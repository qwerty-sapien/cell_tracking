# Independent validation — amended video-subsample fidelity

Status: **TERMINAL / SUBSAMPLING_UNSAFE**

Date: 2026-07-29

## Verdict

No tested `(k, scheme)` satisfies the frozen acceptance rule. The amended
common-36 decomposition is complete and authenticated, all 12 cells were run with
10,000 draws, and the independent cross-check reproduced all 120,000 draw matrices.
No `SUBSAMPLE_CONTRACT` may be emitted.

The largest tested subsample, `k=12`, crosses the rank threshold but fails both
remaining gates:

| scheme | median rho | pooled sign agreement | maximum candidate 95% width | usable |
|---|---:|---:|---:|---|
| simple random | 0.828571 | 0.820683 | 0.0230573 | no |
| residual top-10 | 0.828571 | 0.817567 | 0.0230925 | no |
| family + residual | 0.828571 | 0.815383 | 0.0229620 | no |

The frozen requirements are `|rho| >= 0.8`, sign agreement `>= 0.9`, and strict
maximum width `< 0.005`. The best observed width is more than four times the limit,
and the best sign agreement is about eight percentage points short. Smaller
subsamples fail more strongly. Stratification does not rescue fidelity in this
six-candidate bank.

## Authenticated amended estimand

The root-authorized, pre-outcome amendment
`PREREGISTRATION_AMENDMENT_V1.md`
(`3c497948189b9518c624e3c008cbfa021bc0072a7067a9ebd93e08163b9673fc`)
standardizes all six candidates to the same common-36 `44b6` universe and CG0
reference. The full common-36 scores and deltas are:

| candidate | score | delta versus CG0 |
|---|---:|---:|
| APNCC | 0.8665345260361912 | +0.0069495443951133 |
| C1 | 0.8589315915343100 | -0.0006533901067679 |
| H1-ZWRFEAT | 0.8553864494551265 | -0.0041985321859513 |
| SA-DELETE | 0.8593036309767133 | -0.0002813506643646 |
| SA-SWAP | 0.8662242385086600 | +0.0066392568675822 |
| NODETOPO | 0.8595815454907098 | -0.0000034361503681 |

CG0 is `0.8595849816410779`. C1, H1-ZWRFEAT, APNCC, and SA-SWAP per-video
sufficient statistics were recovered by replaying the frozen official scorer over
their already-banked, hash-matched fragments. Every replay reproduced its banked
aggregate score with absolute error `0`. SA-DELETE and NODETOPO used their banked
per-video rows; the CG0 rows in those two sources were byte-canonically identical.
SA-DELETE's five out-of-universe rows were excluded exactly as frozen by the
amendment.

This replay did not fit a model, run inference, construct a graph, or open CV57,
6bba, audit, or test outcomes.

## Estimator and draw audit

For selected video `i`, the exact design-weighted score is:

`sum(d_i W_i A_i) / sum(d_i W_i) + 0.1 * sum(d_i T_i) /
sum(d_i (T_i + P_i + N_i))`,

where `W_i = edge_tp_i + edge_fp_i + edge_fn_i`,
`A_i = adj_edge_jaccard_i`, and `(T_i,P_i,N_i)` are division sufficient
statistics. The division term is omitted when its denominator is zero. Per-video
delta averaging was not used.

Candidate order is frozen as:
`APNCC, C1, H1-ZWRFEAT, SA-DELETE, SA-SWAP, NODETOPO`.

`amended_reproduction.json` records every derived uint64 seed plus the exact
draw-zero IDs, design weights, and allocation. `amended_validation_checks.json`
independently replays those seeds, recomputes the full scores, CG0 residual top ten,
all 120,000 draw deltas, all fidelity metrics, every draw-zero list/weight/allocation,
and records a deterministic little-endian float64 draw-matrix SHA-256 for each cell.
All checks pass to absolute tolerance `1e-12`.

`family_residual` is correctly marked degenerate with `residual_top10` because the
amended universe contains only `44b6`; its independent scheme-specific seed is still
reported, and it cannot win a tie.

## Evidence

- `amended_source_manifest.json`:
  `01f375c6dbc160de0c9091a0997087419c6510513bb200bacbf65d7cd158a836`
- `amended_decomposition_rows.json`:
  `e42e1a6caf9016fafed34598adcb06973048342cc4bbb710d002da72101ede60`
- `amended_reproduction.json`:
  `43fce5d9d88a5ad72902b0a9ec26a07a41465f2a6e021dac2287b5c62cdaae07`
- `amended_fidelity_table.csv`:
  `bd2713dfea953224cab90243d7068315bdce5a0910b28dab3d9ca0534dbc23d5`
- `amended_validation_checks.json`:
  `12dcdcb26747a14c97d66313e882597b64591f52eb5f1acc6c49ca5eb7dbbb6f`
- continuous guard:
  `8b0af9bb4bde3d05c34223e96daf508df32bc4ddb3207bf42f599d3df277138e`

The guarded run passed with zero swap growth, 0.578 GiB peak process-tree RSS, no
GPU action, no Kaggle/RunPod/submission action, and no budget debit.

## Governance caveat

The validation lane's original audit correctly found that the original
preregistration could not combine the historical 57/41/36-video presentations and
forbade rescoring. It therefore emitted a fail-closed
`UNSAFE_INCOMPLETE_EVIDENCE` result before any draw. Root subsequently froze and
authorized the common-36 decomposition amendment before any subsample outcome.
This terminal verdict is valid only under that amendment; it does not retroactively
reinterpret APNCC's historical CV57 delta or SA-DELETE's historical 41-video delta.

No tier move, DEAD move, promotion, or contract is proposed. Full frozen
44b6-first graph scoring remains mandatory for every promotion decision.
