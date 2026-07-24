# P0R — Observation-aware transfer calibration

> Load `_shared_context.md`, `DAG.md`, and `P1_framework_and_rubric.md`.
> P0R is the P0B/CAL workstream, not the root of all model development.

## Status and scope

- `P0A` scorer semantic parity: **PASS**.
- `P0B` local-to-Kaggle transfer calibration: **STILL-UNRESOLVED**.
- P0B gates fine public-score forecasting, public-probe interpretation, and S1.
- P0B does not halt leakage-safe local decomposition, diagnostics, candidate
  development, or the association/detection GPU gates.
- No Kaggle submission or RunPod launch without explicit human approval.
- Ordinary commits and pushes to this private repository are permitted.

Current execution evidence:

- Phase 0 canonical comparison: **BANKED** in `obs_diff_baseline_v1.json`.
- Phase 1 six-rung dispersion ladder: **BANKED** in
  `dispersion_ladder_v1.md`, with all 15 full-CV57 pairwise observation vectors
  in `dispersion_ladder_observation_matrix_v1.json`.
- The prepared `lb16`/`c0` and `lb4`/`c2` matched pairs are not vacuous; both
  are observation-distinct. They remain below the high-range APNCC-direct
  submission priority and do not establish a general family effect.
- Next measurement: one APNCC-direct public probe, only after explicit human
  approval, followed by complete redesign before any second submission.

## Frozen identities

Do not mix these two local baselines:

| Identity | Prediction source | Score | TP / FP / FN | W |
|---|---|---:|---:|---:|
| `banked_calibration_baseline_c0p25` | `reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/motion_error_weight_0p25/fragments/` | 0.8731495935 | 34,205 / 2,685 / 2,063 | 38,953 |
| `frozen_submission_path_cv57_v3` | `kaggle_kernel/bhcv57out_v3/cv57.csv` | 0.8711885823 | 34,174 / 2,736 / 2,094 | 39,004 |

The three learned-bonus directories in `kaggle_kernel/bhp0-lb{4,8,16}/` are
packages, not predictions. Their local predictions are:

- `reports/g02/screening/p0_lb_replay_cv57_v1/learned_bonus_4__raw_distance_weight_0p05/fragments/`
- `reports/g02/screening/p0_lb_replay_cv57_v1/learned_bonus_8__raw_distance_weight_0p05/fragments/`
- `reports/g02/screening/p0_lb_replay_cv57_v1/learned_bonus_16__raw_distance_weight_0p05/fragments/`

Historical public predictions are the three artifact-linked `submission.csv` files under
`bhp0-c0-out-v1`, `bhp0-c2-out-v2`, and `bhsubmitout_v3`. Compare local with local and
public with public; never cross prediction domains.

## Why the current calibration is insufficient

There are only three scored pairs, all one graph-cost mechanism:

| Configuration | Local CV57 | Displayed public |
|---|---:|---:|
| backward motion `c=0` | 0.8483945156 | 0.860 |
| backward motion `c=2` | 0.8721499475 | 0.878 |
| backward motion `c=0.25` | 0.8731495935 | 0.877 |

The distant `c=0` anchor drives the high affine R². The two near-baseline points reverse
rank, the observed range ends below 0.88, and displayed public scores are rounded to
0.001. Treat every displayed value `s` as the interval `[s-0.0005, s+0.0005)`.

## Mechanism labels

Pre-register the changed mechanism; do not call every structural change a model family:

- detector/front-end;
- association scorer;
- graph cost/solver;
- lineage/division postprocessor;
- learned architecture.

A mechanism/family effect requires repeated score-separated anchors. One matched pair
can reveal an interaction but cannot establish a general family law.

## Phase 0 — Canonical observation comparison

Build `scripts/compare_submissions.py` plus focused tests.

### Identity and matching order

1. Record the raw SHA-256 and package/source identity.
2. Canonicalize semantic graph content, ignoring CSV row order and the presentation `id`.
3. Compare exact node IDs and coordinates.
4. If coordinates or IDs differ, perform deterministic time-aware one-to-one matching:
   same dataset and timestep, physical voxel scale, maximum-cardinality/minimum-distance,
   deterministic tie breaking.
5. Report both exact matching and the official 7 µm tolerance.

### Output vector

Keep distances as a vector, per video and aggregate:

- node shared/A-only/B-only counts and node Jaccard;
- raw and scorer-eligible edge shared/A-only/B-only counts and edge Jaccard after node
  correspondence;
- structural fork/lineage shared/A-only/B-only counts and agreement;
- canonical semantic hashes;
- with GT, each prediction's official score, node adjustment/recall, edge TP/FP/FN, and
  division TP/FP/FN.

Prediction-to-prediction edge overlap is **edge Jaccard**, not “adjusted edge Jaccard.”
Official adjusted edge Jaccard is a separate GT-based metric.

A source with outdegree >=2 is only a candidate fork. Division comparison must mirror the
official parent/divider/children/grandchildren, distinct-branch, and bipartite-pairing
semantics where GT is available. On hidden public data report structural agreement only.

### Classification

Do not collapse the vector into an arbitrary scalar. Classify comparisons as:

- `IDENTICAL`: all semantic components are zero-distance;
- `LOW-INFORMATION`: nonzero but within a preregistered reproducibility/null envelope;
- `DISTINCT`: exceeds that envelope in one or more named components.

Calibrate the envelope from identical reruns/reserialization, deterministic numerical
jitter, and known node/edge/fork perturbations. Only exact semantic equality is vacuous.

Deliver:

- `scripts/compare_submissions.py`;
- focused tests;
- `reports/g02/screening/obs_diff_baseline_v1.json`.

## Phase 1 — Evidence consolidation and dispersion ladder

Before rerunning B1/B2/E1, consolidate:

- `division_sandbox_and_holdout_decomposition_v1.*`;
- `source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.*`;
- fixed-node oracle evidence;
- existing graph-cost/learned-bonus sweeps;
- node-threshold and division-failure reports.

Enumerate CPU/cached configurations only where an exact prediction identity can be
recovered. Select approximately five or six leverage points that:

- span the widest feasible local range;
- include useful coverage below the operating point and, when attainable, above 0.88;
- are `DISTINCT` from adjacent rungs;
- change a preregistered mechanism when testing transfer dependence.

Deliver `dispersion_ladder_v1.md` with local score, identity, observation vector,
mechanism, expected information gain, and an adaptive submission order.

## Phase 2 — Mechanism-diverse anchors

Use cheapest structurally distinct anchors first:

1. association-cost replacement rather than a small additive bonus;
2. detector/front-end threshold change;
3. lineage/division postprocessing when it produces scorer-valid forks.

For a mechanism claim, target at least two score-separated anchors. A matched-local
comparison is admissible only when:

- it uses the same frozen prediction/evaluation domain;
- local intervals overlap or preregistered `abs(delta_local) < 0.004`;
- both predictions have exact identities;
- the comparison is `DISTINCT`;
- the mechanism labels were fixed before seeing the public result.

Small nonzero differences are low-information, not vacuous. The learned-bonus probes
must pass this rule before consuming a submission.

## Phase 3 — Sequential public measurement

- Maximum five Kaggle submissions per competition day and two final selections.
- Calibration probes count against the cap.
- Submit at most one newly approved, observation-distinct probe at a time.
- Obtain explicit human approval for each submission.
- Record the expected information gain before submission.
- Treat the returned score as a rounding interval and update the remaining probe order.
- Do not spend a submission merely to reproduce the already-known near-tie resolution
  floor.

## Phase 4 — Identifiable calibration models

Fit only models supported by the accumulated design:

1. **Now:** descriptive global affine and monotone/interval feasibility over the observed
   range.
2. **After repeated mechanisms:** common slope plus mechanism intercept.
3. **Only with >=3 score-separated points per family/mechanism:** separate slopes, so
   leave-one-out retains at least two fit points.
4. **Covariates:** only when pair count materially exceeds the effective parameter count
   and the design matrix is full rank.

Evaluate predictions against displayed-score intervals. Report interval coverage and miss
distance, not point residual SD alone. With the current three pairs, per-family affine,
multivariable regression, and a precise LOO uncertainty estimate are not admissible.

Verdicts:

- `GLOBAL-ADEQUATE-OBSERVED-RANGE`: one global mapping is adequate over the measured
  mechanisms and score range.
- `MECHANISM-SPECIFIC`: repeated anchors show a mechanism shift material to decisions.
- `STILL-UNRESOLVED`: identify the exact missing score range/mechanism anchor.

None of these verdicts triggers or vetoes GPU work by itself.

## Phase 5 — Independent model-development gates

Association and detection escalation follow GA and GD in `DAG.md`. They depend on local
headroom, error mechanism, proxy improvement, runtime, and budget. Use private Kaggle GPU
quota first. RunPod requires explicit human approval for each exact launch after
predeclaring GPU-hours, maximum cost, and artifact identity.

## Phase 6 — S1 portfolio

Maintain the candidate table with local grouped-CV/audit evidence, mechanism label,
observation identity, generalization spread, rounded public interval, and submission
count. Keep both final selections uncommitted.

- Pick A: highest robust floor / lowest generalization risk.
- Pick B: highest evidence-supported upside from the appropriate observed mechanism/range.

Do not simply choose the two highest local scores.

## Evidence basis

Literature motivates policy direction, not repository thresholds:

- Dwork et al., *The reusable holdout*, and Blum & Hardt, *The Ladder*, motivate guarding
  against repeated adaptive leaderboard reuse.
- Varma & Simon motivate separating model selection from performance assessment.
- The Cell Tracking Challenge analyses motivate dataset-aware generalization and learned
  spatial/temporal context.

The numerical `0.004`, `0.002`, `+0.005`, no-GNN decision, submission cap, and compute
budgets are repository measurements or policy—not claims derived from those papers.

Primary references:

- https://pubmed.ncbi.nlm.nih.gov/26250683/
- https://arxiv.org/abs/1502.04585
- https://pubmed.ncbi.nlm.nih.gov/16504092/
- https://www.nature.com/articles/s41592-023-01879-y
- https://www.nature.com/articles/nmeth.4473
