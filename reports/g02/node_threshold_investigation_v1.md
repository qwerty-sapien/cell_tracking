# Node-threshold investigation (fold 0)

Status: completed analysis; no runtime threshold change; no Kaggle action.

## Finding 1 — the scored holdout run used 0.970, not 0.985 (verified)

The one-pass detector threshold is frozen at `0.970` in
`src/celltrack_alpha/bh_model.py:118` and again in the configuration validator
at `src/celltrack_alpha/bh_model.py:136`. The batched peak detector applies
that value at `src/celltrack_alpha/bh_model.py:1101`.
`src/celltrack_alpha/v45_fast.py:438` calls that detector without an override,
and `scripts/v45feval.py:477` calls `infer_fast` without a threshold argument.

The decisive provenance check is the execution hash. Recomputing

```text
v45_config_sha(pair_batch_size=1, view_batch_size=1, d4_views=(0,))
```

with the current frozen `0.970` configuration returns
`66009cdfc2cd74f4f9906073bdafe9201deba0eaf7fda1899cbd0aeb489bd575`.
That exactly equals `execution.video_config_sha256` in
`reports/g02/source_softmax_edge_f0_fit117_holdout_dry_v1/report.json` and in
every available per-video source-softmax holdout report checked. Therefore the
52 scored samples in
`reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json`
are 0.970 results. The previously fit-selected 0.985 threshold was not wired
into this evaluator.

This is also locked by
`tests/test_source_softmax_node_threshold.py:30-42`.

## Finding 2 — completed fit-only grid selects 0.985, globally and by scenario (verified, provisional)

The completed grid in `reports/g02/fit_node_threshold_14_v1.json` evaluates
thresholds 0.970 through 0.999 on 14 fold-fit videos only. The hardened report
`reports/g02/fit_node_threshold_14_v2.json` enforces no held-out, audit, or
test exposure and selects the highest mean adjusted edge Jaccard subject to a
maximum per-video node-recall loss of 0.002. The rule is implemented at
`scripts/analyze_fit_node_threshold.py:107-127`; its fit-only checks are at
`scripts/analyze_fit_node_threshold.py:175-202` and `:225-237`.

Selected thresholds:

| Slice | Videos | Selected | Mean adjusted edge Jaccard at 0.970 | At 0.985 | Delta | Mean raw delta | Aggregate node ratio, 0.970 -> 0.985 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 14 | 0.985 | 0.824831419 | 0.829188086 | +0.004356667 | +0.001631982 | +6.6905% -> +0.7048% |
| 44b6 | 5 | 0.985 | 0.753033073 | 0.760315099 | +0.007282026 | +0.002755756 | -1.7524% -> -8.9560% |
| 6bba | 9 | 0.985 | 0.864719389 | 0.867450856 | +0.002731468 | +0.001007664 | +20.6723% -> +16.7035% |

Thus the safe per-scenario rule does not differ from the safe global rule:
both scenarios select 0.985. Relative to the actual 0.970 holdout execution,
0.985 is the first fit-supported candidate that should be evaluated fresh.

These are per-video means because the completed v1 artifact did not persist
`edge_tp/fp/fn`; see the aggregation at
`scripts/analyze_fit_node_threshold.py:87-104`. They must not be represented as
the official edge-event-weighted run score.

## Finding 3 — the uncapped bonus creates a tempting but unsafe 0.9975 optimum (verified on the same fit grid)

Without the recall guard, the mean-adjusted optimum is 0.9975 globally and in
both scenarios:

| Slice | Mean adjusted at 0.985 | At 0.9975 | Delta | Mean raw delta | Aggregate node ratio at 0.9975 |
|---|---:|---:|---:|---:|---:|
| all | 0.829188086 | 0.834789711 | +0.005601625 | -0.004317069 | -20.1959% |
| 44b6 | 0.760315099 | 0.763125927 | +0.002810828 | -0.012485869 | -31.6540% |
| 6bba | 0.867450856 | 0.874602924 | +0.007152068 | +0.000221153 | -1.2208% |

The gain is substantially driven by the node multiplier rather than better
edges. Across all 14 videos, mean node recall falls from 0.987264 to 0.981058;
5/14 videos exceed the 0.002 recall-loss guard and the worst loss is 0.046282.
For example, `6bba_ebff6e76` falls from 0.976480 recall at 0.985 to 0.930956 at
0.9975. `44b6_c771cb04` falls from 0.995595 to 0.973568.

The above-1.0 behavior is real: for fit video `6bba_d5eae175`, raw edge
Jaccard remains 1.0 while adjusted edge Jaccard rises from 1.008298 at 0.985
to 1.013344 at 0.9975 as the node ratio moves from -8.2983% to -13.3437%.
This follows the uncapped formula at
`external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py:353-363`.

Conclusion: do not promote 0.9975 from this evidence. Its apparent gain is
paired with lower raw edge quality and a material recall tail.

## Representativeness and transfer limits

- Verified fit boundary: all 14 calibration videos are fold-fit only; held-out,
  audit, and test flags are false.
- Sample size: 14/117 fit videos (12.0%). It is the cache plan's
  `already_complete_video_ids` convenience subset, not a documented random or
  stratified sample. The report itself marks the result
  `provisional_until_full_117_fit` at
  `scripts/analyze_fit_node_threshold.py:296-304`.
- Scenario mix is close: the subset is 5/14 44b6 (35.7%) and 9/14 6bba
  (64.3%); the full fit corpus is 41/117 (35.0%) and 76/117 (65.0%); the full
  holdout is 21/58 (36.2%) and 37/58 (63.8%). This supports scenario coverage,
  not execution equivalence.
- Execution mismatch: the grid uses released sequential-D4 detections,
  inherited legacy motion association, active divisions, and no image
  refinement (`scripts/analyze_fit_node_threshold.py:240-276`). The scored
  holdout uses the one-pass fast detector and the fit-117 source-softmax edge
  artifact. Threshold transfer is therefore provisional.

The three verified high-ratio holdout tails at the actual 0.970 execution are:
`6bba_705ec2c9` +0.707321, `6bba_07e24132` +0.672423, and
`44b6_996155de` +0.500594. Their response to 0.985 is not measured.

## Why frozen holdout CSVs cannot answer the 0.985 question

Submission fragments persist coordinates and graph IDs but no node confidence;
the complete CSV schema is at `scripts/run_bh.py:93-104`. Nodes removed by a
higher threshold therefore cannot be identified post hoc, and changing the
threshold also changes sparse candidates, edges, gap synthesis, and graph
pruning. A valid 0.985 comparison requires fresh one-pass image inference and
official rescoring. No such replay was launched while the full-58 baseline was
still running.

`scripts/analyze_source_softmax_node_threshold.py` is a reproducible future
fit-side replay path. Its default grid starts at the actual runtime 0.970 and
includes the fit-supported 0.985 candidate. It validates the exact fit-117
manifest and edge artifact, filters detections and pair features together,
disables divisions, computes official edge-event-weighted summaries, and
applies the same 0.002 worst-video recall-loss guard while also reporting the
unguarded optimum. A one-video 0.970-versus-0.985 smoke replay completed
successfully and selected 0.985 (+0.003665 weighted adjusted edge Jaccard on
that fit video, with no recall loss). At approximately 52 seconds per video per
threshold on local CPU, even the two-point 117-video comparison is about 3.4
hours and the default nine-point grid is about 13.7 hours, so neither full run
was launched during the active holdout run.

## Commands run

```text
sed -n '1,260p' reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json
sed -n '261,9999p' reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv313/bin/python -c '<print threshold and one-view config hash>'
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python scripts/analyze_source_softmax_node_threshold.py --limit 1 --thresholds 0.970 0.985 --output /tmp/source_softmax_node_threshold_0970_0985_smoke.json
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python -m unittest tests.test_fit_node_threshold tests.test_source_softmax_node_threshold
```

No Kaggle submission, network access, paid compute, public artifact, or runtime
configuration change was performed.
