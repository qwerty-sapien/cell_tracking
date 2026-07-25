# CGL D-14 learned-representation fast-fail

**FAIL-CLOSED**

The goal-owner-amended CPU guard admitted the fit-only rerun, then stopped it exactly as
contracted when cumulative swap growth exceeded **1.5 GiB**. The process reached 91/105
fit-train feature videos but did not begin head optimization or held-fold evaluation.
Therefore no honest GO/PARK/DEAD model verdict is available.

## Contract amendment recorded before the run

The D-14 candidate manifest was amended before launch and validated under content SHA-256
`d62554218a1c7081327e5de889429907aae4e69b43d8169465db9d8a0d2f7056`.
The preregistration record is
[`cgl_d14_runtime_guard_amendment_preregistered_v2.json`](cgl_d14_runtime_guard_amendment_preregistered_v2.json).

Only the swap limit changed:

| guard | amended contract | observed | result |
|---|---:|---:|---|
| cumulative swap growth | ≤1.5 GiB | **1,617.50 MiB / 1.5796 GiB** | **BREACH** |
| process-tree RSS | ≤18 GiB | 0.838 GiB | pass |
| system free memory | ≥25% | 31% minimum | pass |
| wall time | ≤360 min | 231.97 s | pass |

Swap started at 14,982.75 MiB and peaked at 16,600.25 MiB. Growth exceeded the ceiling
by **81.50 MiB**, so the guard terminated the child with
`swap_growth_ceiling_exceeded`. The full telemetry is in
[`cgl_d14_fastfail_guard_amended_v2.json`](cgl_d14_fastfail_guard_amended_v2.json).
Swap affects speed rather than the deterministic numerical outputs, but no numerical
outputs exist because the contracted runtime guard stopped the process first.

## Frozen candidate

| field | frozen value |
|---|---|
| core | released TemporalUNet3D V45 32-D node embeddings |
| core weight SHA-256 | `12f6881e…e771` |
| trainable head | `Linear(135,32) → GELU → Linear(32,1)` |
| trainable parameters | 4,385 |
| objective | same-video budget-tail pairwise softplus plus distributional ranking |
| hard negatives | top `max(1,024, 64 × video FP budget)`, refreshed once |
| generic BCE | **not used** |
| folds | five deterministic held-video folds, seed 250725 |

The objective remains grounded in focal, AP-loss, and distributional-ranking transfers
for dense foreground/background imbalance. The core remained frozen and the head never
trained in this attempt.

## Scope and certification

- Fit-only input: fit-train105 feature construction.
- Forbidden `holdout` and `CV57` path tokens were rejected before reads.
- validation12, holdout, audit, test, CV57, graph replay, GPU, Kaggle, RunPod,
  competition submission, and full T-03 training remained closed.
- Full per-video r10/k8 backgrounds were being processed, not sampled.
- 91/105 video features completed; full-pool held-fold certification did **not** complete.

| family | held-fit retention | full-pool FP certification | status |
|---|---:|---:|---|
| 44b6 | — | — | **NOT RUN** |
| 6bba | — | — | **NOT RUN** |

No head artifact was created.

## Verdict and cascade

**FAIL-CLOSED**

- No T-03 handoff or GPU launch is authorized.
- Option D is not triggered because the learned mechanism was not trained or falsified.
- No further local guard relaxation is authorized by this run.
- The exact goal-owner amendment has been exhausted; a new goal-owner decision is
  required for any resume.

Machine-readable result:
[`cgl_d14_fastfail_v1.json`](cgl_d14_fastfail_v1.json).
