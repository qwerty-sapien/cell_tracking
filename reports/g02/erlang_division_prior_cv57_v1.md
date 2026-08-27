# Erlang division-opening prior: fit117 → CV57

Decision: **REJECT. Keep production divisions OFF.**

The best tested prior (`k=1`) produced one genuine true-positive division under
the patched official metric, but also 104 false positives. Its division Jaccard
was `0.0068493151`, for a weighted gain of only `0.0006849315`. This is below
the pre-specified `0.002` threshold. The combined score rose by `0.0004892035`,
which does not change the rejection decision.

## Fit

Fold-0 fit scope contained 117 videos and 83 annotated dividing cells. The GT
has no multigeneration daughter-to-division lineage among these events: all 83
dividing tracks begin at annotation roots. Three roots appear only in their
division frame (`dt=0`) and were excluded because an Erlang density has positive
support. The usable sample size is therefore **n=80**.

Operational lifetime is the number of frames from the earliest annotated
ancestor of the cell to its division. This is an annotation-span proxy, not a
directly observed biological lifetime; root truncation and the small sample are
material uncertainties.

- Mean: `21.55` frames (nonparametric bootstrap 95% interval
  `[17.5750, 25.7753]`)
- Median: `15.5`; sample SD: `19.7528`
- Range: `1–66`; P05/P25/P75/P95: `1 / 4 / 34.25 / 60.05`
- Free Gamma diagnostic: shape `0.9081` (bootstrap 95% interval
  `[0.7541, 1.1519]`), rate `0.04214/frame`
- Maximum-likelihood integer Erlang shape: `k=1`

For each swept integer shape, the fixed-shape MLE is
`lambda = k / 21.55`:

| k | lambda/frame | Patched TP | Patched FP | Patched FN | Division Jaccard | 0.1 × gain | Adjusted-edge delta | Combined delta |
|---:|-------------:|-----------:|-----------:|-----------:|-----------------:|-----------:|--------------------:|---------------:|
| 1 | 0.0464037 | 1 | 104 | 41 | 0.0068493 | 0.0006849 | -0.0001957 | +0.0004892 |
| 2 | 0.0928074 | 1 | 108 | 41 | 0.0066667 | 0.0006667 | -0.0002638 | +0.0004029 |
| 3 | 0.1392111 | 1 | 108 | 41 | 0.0066667 | 0.0006667 | -0.0002167 | +0.0004500 |
| 4 | 0.1856148 | 1 | 109 | 41 | 0.0066225 | 0.0006623 | -0.0001915 | +0.0004707 |
| 5 | 0.2320186 | 1 | 109 | 41 | 0.0066225 | 0.0006623 | -0.0002138 | +0.0004484 |

## Evaluation contract

- Exact 57-video fold-0 holdout; fit, audit, and test videos were not used for
  evaluation-time fitting.
- Exact division-off baseline reproduced:
  adjusted edge Jaccard `0.8711885823490648`, TP/FP/FN divisions `0/0/42`.
- Re-vendored patched official division scorer, validated by the frozen green
  parity ledger.
- CPU-only graph postprocessing; no detector inference, training, GPU, Kaggle
  action, or production configuration change.

The full machine-readable fit, bootstrap uncertainty, per-video selected fork
pairs, solver counters, scorer hashes, and input hashes are in
`erlang_division_prior_cv57_v1.json`.
