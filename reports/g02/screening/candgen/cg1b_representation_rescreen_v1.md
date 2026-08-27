# CG1b stronger-frozen-representation abstention re-screen

**PARK**

The existing frozen appearance and geometry representations do not lift the extreme-tail
retention needed to justify Stage 2. The strongest exact-budget 44b6 result retained
**14/766 = 1.83%** of known-correct edges, versus the preregistered **20% GO** floor,
the **15% PARTIAL** floor, and CG1's sampled **11.88%** result. The hardest-family stop
therefore fired before 6bba, the Trackastra 49-D cache, or CG2.

## Preregistration and scope

Before execution, the Stage-1 rule was sharpened as follows:

- **GO:** one identical representation/mechanism reaches at least 20% known-correct
  retention in both families while every video respects its exact FP budget.
- **PARTIAL:** one identical representation/mechanism reaches at least 15% in both
  families and improves by at least 2 percentage points over CG1 in each family.
- **PARK:** neither condition holds.
- Evaluate 44b6 first and stop before 6bba if the best budget-compliant 44b6 candidate
  misses the PARTIAL floor.

The run used only the 105 fit-training videos. Fit-validation12 remained closed for
selection. Holdout, audit, test, and paths containing `holdout` or `CV57` were rejected
before read. There was no model training, graph replay, GPU, Kaggle, or RunPod action.
No new feature cache was created.

## Full-pool certification

CG1's 8,668-row 44b6 sample was replaced with the exact per-video r10/k8 deployment
pool. Each video received

`floor(CG0 44b6 background-acceptance rate × exact video raw background)`

false-positive slots. This conservative integer allocation totals **742 FP** over the 36
fit-train105 44b6 videos, below the proportional fit117 family budget of 848.

| certification quantity | 44b6 fit-train105 |
|---|---:|
| videos | 36 |
| exact r10/k8 candidate pairs | 3,617,372 |
| exact raw background candidates | 3,606,536 |
| selected-graph conflict proposals before geometry | 2,437,802 |
| finite frozen-geometry proposals | 84,883 |
| exact correct finite proposals | 283 |
| known-correct denominator | 766 |
| known-correct finite proposals | 281 (36.68%) |
| conservative per-video FP budget sum | 742 |
| sampled background used | **no** |

All candidates outside selected-graph-conflict and frozen-geometry support were explicit
abstentions. For candidates receiving a finite score, background status was counted
against the exact ground-truth match, per video.

## Stage-1 results

`Pass` requires every held video to stay within its own FP allocation. The hard top-k
mechanism is count-safe by construction. Conformal mechanisms use five deterministic
video folds and one threshold learned from the other videos; an aggregate FP total below
742 does not rescue a held-video budget violation. PU retains unlabeled semantics but
collapses to complete abstention at this tail.

| representation | mechanism | known correct kept | retention | background FP / budget | pass |
|---|---|---:|---:|---:|:---:|
| APNCC + geometry rank | hard per-video top-k | **14/766** | **1.83%** | 728 / 742 | yes |
| APNCC + motion rank | hard per-video top-k | 12/766 | 1.57% | 730 / 742 | yes |
| APNCC margin | hard per-video top-k | 8/766 | 1.04% | 734 / 742 | yes |
| geometry confidence | hard per-video top-k | 6/766 | 0.78% | 736 / 742 | yes |
| APNCC + geometry + motion + V45 rank | hard per-video top-k | 5/766 | 0.65% | 737 / 742 | yes |
| V45 logit margin | hard per-video top-k | 1/766 | 0.13% | 741 / 742 | yes |
| APNCC + geometry rank | held-family conformal | 5/766 | 0.65% | 374 / 742 | **no** |
| APNCC + motion rank | held-family conformal | 5/766 | 0.65% | 298 / 742 | **no** |
| APNCC + geometry + motion + V45 rank | held-family conformal | 2/766 | 0.26% | 167 / 742 | **no** |
| geometry confidence | held-family conformal | 1/766 | 0.13% | 342 / 742 | **no** |
| APNCC margin | held-family conformal | 0/766 | 0.00% | 46 / 742 | **no** |
| V45 logit margin | held-family conformal | 0/766 | 0.00% | 138 / 742 | **no** |
| each of the six representations | PU family threshold | 0/766 | 0.00% | 0 / 742 | yes |

### Per-family gate

| family | status | best exact result | reason |
|---|---|---|---|
| **44b6** | **PARK** | APNCC + geometry hard top-k: 14/766 = 1.83% | 13.17 points below PARTIAL and 10.05 points below CG1's sampled result |
| 6bba | NOT RUN | — | preregistered hardest-family falsification made it decision-irrelevant |

The geometry support ceiling of 36.68% means sufficient correct edges still exist inside
the finite APNCC population to pass the 20% gate in principle. The actual 1.83% tail
retention shows that the banked appearance/geometry/motion signals order that population
incorrectly at the required abstention rate. Adding these representations therefore
does not repair CG1's transferable-tail problem.

## Verdict and cascade

**PARK**

Stage 2 is not authorized: Stage 1 produced no material lift, so the 44b6 Trackastra
pipeline is not built and the human-gated 6bba Kaggle cache completion is not requested.
CG2 is not authorized. The required cascade is
[`cg_escalation_v1.md`](cg_escalation_v1.md).

Machine-readable evidence:
[`cg1b_representation_rescreen_v1.json`](cg1b_representation_rescreen_v1.json).
