# CGL D-14 learned-representation fast-fail

**DEAD**

The 12 GiB CPU retry completed all 105 fit-train feature pools, five held-video folds,
compact-head training, and full-pool evaluation. The learned tail-ranking head retained
only **1/766 = 0.1305%** of known-correct 44b6 edges at the exact FP budget. This is far
below the preregistered **15% GO** threshold and the **5% material-lift** floor, and below
the frozen CG1b reference of approximately 1.83%.

## Preregistered result

| family | known-correct kept | retention | background FP / budget | videos | verdict contribution |
|---|---:|---:|---:|---:|---|
| 44b6 | 1 / 766 | **0.1305%** | 742 / 742 | 36 | below 5% → **DEAD** |
| 6bba | 27 / 2,993 | **0.9021%** | 2,367 / 2,367 | 69 | below 10% non-collapse |

The secondary deployable hard-top-k lower bound retained 1/766 44b6 edges with 731 FP
and 7/2,993 6bba edges with 1,880 FP.

The primary metric intentionally scans each video's ranking until its exact background
budget is exhausted, so it measures the maximum known-correct retention achievable at
that learned ordering and FP allocation. Every per-video budget passed.

## Full-pool certification

| family | candidate pairs | exact background | sampled background | all candidates scored |
|---|---:|---:|---:|---|
| 44b6 | 3,617,372 | 3,606,536 | no | yes |
| 6bba | 3,651,162 | 3,594,107 | no | yes |

The evaluation used the complete fit-train105 r10/k8 deployment pool. No small
background sample, validation12, CV57, holdout, audit, test, or graph replay was used.

## Head and objective

| field | frozen value |
|---|---|
| core | released TemporalUNet3D V45 32-D node embeddings |
| core weight SHA-256 | `12f6881e…e771` |
| trainable head | `Linear(135,32) → GELU → Linear(32,1)` |
| trainable parameters | 4,385 |
| objective | per-video budget-tail pairwise softplus plus distributional ranking |
| hard negatives | top `max(1,024, 64 × video FP budget)`, refreshed once |
| generic BCE | **not used** |
| folds | five deterministic held-video folds, seed 250725 |
| head artifact | `cgl_d14_fastfail_v1_heads.pt`, SHA-256 `095b611a…2525` |

All five fold-head hashes reproduced across the initial packaging-failure attempt and
the corrected report-writing attempt. The first attempt completed training with zero swap
growth but exposed a relative-output-path serialization bug after the head artifact was
written. That artifact and guard report are preserved; the narrow path fix was then
rerun under the identical scientific contract.

## Runtime contract

The goal-owner-authorized 12 GiB manifest was recorded before launch under SHA-256
`eb3177543edc2910b0de19e9e545da1cd2da4ff92043cffd971b459ea0be30f7`.

| guard | ceiling | observed | result |
|---|---:|---:|---|
| process-tree RSS | ≤12 GiB | 2.971 GiB | pass |
| cumulative swap growth | ≤1.5 GiB | 0 MiB | pass |
| system free memory | ≥25% | 84% minimum | pass |
| wall time | ≤360 min | 168.77 s | pass |

The preregistration is
[`cgl_d14_runtime_guard_amendment_preregistered_v3.json`](cgl_d14_runtime_guard_amendment_preregistered_v3.json);
the final telemetry is
[`cgl_d14_fastfail_guard_amended_v3.json`](cgl_d14_fastfail_guard_amended_v3.json).

## Verdict and cascade

**DEAD**

- The compact tail-aligned head does not break the 44b6 ordering wall.
- T-03 is not analytically authorized and no GPU launch is requested.
- Route to
  [`CG_ESCALATE_goal_feasibility.md`](CG_ESCALATE_goal_feasibility.md)
  **Option D**: reopen D-12/AC-02 and renegotiate the 0.92 endpoint around the honest
  banked approximately 0.881 completion.
- No GPU, Kaggle, RunPod, competition submission, full T-03 training, or forbidden-split
  action occurred.

Machine-readable result:
[`cgl_d14_fastfail_v1.json`](cgl_d14_fastfail_v1.json).
