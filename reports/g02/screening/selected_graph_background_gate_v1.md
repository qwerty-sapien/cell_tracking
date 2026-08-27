# Selected-graph background abstention gate

Date: 2026-07-24

Status: **PARK**

## Decision

Explicit deployment-background regularization repaired the support failure of the
first selected-graph gate, but the resulting sparse policy regressed on the exact
CV57 top-10 screen. It therefore does not qualify for a full 57-video confirm,
packaging, GPU work, or a public submission.

| Exact top-10 arm | Adjusted edge Jaccard | TP / FP / FN |
|---|---:|---:|
| Banked backward-motion `c=0.25` control | 0.721796529 | 5,117 / 1,133 / 730 |
| Background-regularized selected-graph gate | 0.720934715 | 5,113 / 1,136 / 734 |
| Delta | **-0.000861814** | -4 / +3 / +4 |

The primary screen gate was `>= +0.005`. The candidate failed it in the wrong
direction.

## Leakage-safe deployment support

The background builder audited all 105 fit-train videos and no validation,
holdout, audit, or test video. It enumerated 4,747,100 graph-conflict proposals,
removed known labeled pairs, and retained a deterministic 19,398-proposal
background sample: 9,696 highest-confidence hard examples and 9,702
hash-uniform examples.

The initial objective selected checkpoints by labeled utility before deployment
sparsity. That ordering was rejected because it preserved a 57.59% background
acceptance rate. The corrected selector first required labeled admissibility,
then minimized fit-background acceptance, then used labeled utility/loss as
tie-breakers. At background weight 0.50 it reached 2.459% fit-background
acceptance.

Natural validation-video support still failed: the best threshold exceeded the
registered sparse budget by 2.37x. A fail-closed cap made the policy admissible:

- frozen gate artifact threshold: `0.605`;
- validation threshold override: `0.845`;
- labeled validation: 10 accepted, 10 correct, 0 harmful;
- per-video cap: `max(16, ceil(0.005 * selected_raw_edges))`;
- confidence-ordered, node-disjoint proposals only.

That capped rule was frozen before the CV57 screen.

## Exact identity and observation audit

An initial report in
`selected_graph_background_sparse_screen10_w050_v1/report.json` used the older
`bhcv57out_v3` fragments as its displayed control. It is **superseded** and must
not be used: the candidate itself was built from the backward-motion `c=0.25`
graph.

Version 2 explicitly points both scoring and reconciliation at
`phase1_e3_backward_motion_coefficient_cv57_v1/motion_error_weight_0p25`.
The frozen scorer reproduced both arms exactly. Observation comparison classified
the candidate as **DISTINCT**:

- node Jaccard: `0.996025`;
- raw/scorer-eligible edge Jaccard: `0.988035`;
- edge disagreement fraction: `0.006019`;
- fork agreement: `1.0` with no forks in either arm.

The screen accepted 1,729 proposals after capping (3,332 before the cap), of
which 1,173 challengers were realized. The negative score is therefore an
informative failure, not an identity-null replay.

## Artifacts

- `reports/g02/selected_graph_background_prepare_v1/prepare.json`
- `reports/g02/selected_graph_background_sparse_w050_v1/report.json`
- `reports/g02/selected_graph_background_sparse_w050_v1/gate.pt`
- `reports/g02/selected_graph_background_sparse_support_w050_v2.json`
- `reports/g02/selected_graph_background_sparse_screen10_w050_v2/report.json`
- `reports/g02/screening/obs_diff_selected_graph_background_sparse_screen10_w050_v1.json`

No Kaggle submission, Kaggle GPU, RunPod, or paid compute was used.
