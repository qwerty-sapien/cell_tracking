# §6.2 PEFT DELETE terminal report

The sole authorized Kaggle T4 version completed and **failed the frozen 44b6
graph gate**. No 6bba, CV57, transport packaging, or submission followed.

## Execution and integrity

- Version count changed exactly `0 -> 1`; there was one push and one remote
  version.
- Authoritative quota changed from `5.02/24.98` to `5.06/24.94` T4-hours
  used/remaining, an exact `0.04` T4-hour debit.
- Inline Step 0 completed in `1.500973 s`, authenticated all 90 declared
  members, and made zero competition-tree visits.
- Training used the frozen `112,918`-row sample: all `1,118` beneficial
  actions plus `111,800` deterministic harmful actions at 100:1. The sample
  digest is `a1e1f638…51571`.
- Every reported AUC, retention, and graph metric used the complete
  `1,345,497`-action background.
- The adapter contained 2,665 trainable parameters, `0.009706%` of the
  authenticated 27,456,880-parameter frozen base. It completed 3,744 optimizer
  steps; the measured step-250 projection was `0.053206` T4-hours and the
  scientific execution took `145.680 s`.

## Capacity probe

Full-population OOF AUC by epoch was:

| Epoch | AUC | Delta from 0.502 |
|---:|---:|---:|
| 1 | 0.448881 | -0.053119 |
| 2 | 0.515572 | +0.013572 |
| 3 | 0.598619 | +0.096619 |
| 4 | 0.622770 | +0.120770 |
| 5 | 0.630757 | +0.128757 |
| 6 | 0.633477 | +0.131477 |
| 7 | 0.634590 | +0.132590 |
| 8 | 0.634749 | +0.132749 |

The late-epoch trajectory is descriptive only: the preregistration forbids
choosing an epoch by AUC. Nested inner-OOF graph delta selected epochs
`1/2/2/2/2` across the five outer folds. That graph-selected complete OOF
prediction has AUC `0.496495`, movement `-0.005505` from the `0.502`
incumbent, and therefore triggers the frozen **case B** rule (`<0.552`):
negligible selectable AUC movement and no indication for more capacity on
this frozen backbone.

Even the nonselectable epoch-8 diagnostic remains `0.215251` below the
measured `0.85` requirement. CG0-budget retention is `3/1,118 =
0.002683`, versus the measured `0.020394` requirement; it does not improve
across epochs.

## Frozen graph verdict

The selected 129 deletes removed two false edges and zero true edges:

- adjusted edge-Jaccard delta: `+0.000133826958`;
- raw edge-Jaccard delta: `+0.000133160113`;
- node-recall delta: `0`.

Raw and node clauses pass, and every outer adjusted delta is nonnegative, but
the aggregate adjusted delta misses the binding `+0.005` line by about
`37.36x`. The exact PEFT realization is **FAIL / PARK-C**. Under the
precommitted cascade, 6bba and CV57 remain closed, no transport package is
triggered, and submission remains forbidden without explicit human approval.

Independent replay passes every recorded invariant in
`RUN_V1_VALIDATION.json`.
