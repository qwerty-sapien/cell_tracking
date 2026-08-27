# A7 frozen separation measurement — terminal report

Status: **FAIL / PARK A7 ON EVIDENCE / NO CONDITIONAL REALIZATION** (2026-08-03).

## Decision

A7 does not clear the existing frozen occupied-swap separation gate. The exact
45,961-candidate universe contains 134 oracle-beneficial actions, but adding the
original sealed A7 phase/template `action_margin` to the incumbent representation
does not rank them adequately. A1 retention at the CG0 harmful-action budget is
zero in every fold, all five A1 AUCs are below the binding `0.90` floor, and no
fold clears either AUC lift clause.

The applicable action class is **occupied swap**, so the binding measured absolute
requirements are AUC `>=0.90` and retention `>=0.06791044776119404`. The unchanged
per-fold lift requirements are A1-B0 AUC/retention `>=0.05/0.05` and A1-A0
AUC/retention `>=0.03/0.03`. Every clause must pass in every fold.

Because separation fails, no FIT-only acceptance threshold is derived, no edit set
is sealed, and the one-shot 44b6 held-fit graph gate is not opened. Edit-set size
and graph delta are therefore **N/A (not executed)**, not zero-valued graph results.

## Exact five-fold result

All retention values for B0, A1, and A0 are `0` in all folds.

| Fold | positives / actions | B0 AUC | A1 AUC | A0 AUC | A1-B0 AUC | A1-A0 AUC | A1 retention | Fold pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 34 / 9,247 | 0.625542 | 0.627866 | 0.626292 | +0.002324 | +0.001574 | 0 | No |
| 1 | 7 / 3,403 | 0.837077 | 0.828496 | 0.834974 | -0.008582 | -0.006478 | 0 | No |
| 2 | 22 / 9,919 | 0.622301 | 0.610208 | 0.621961 | -0.012093 | -0.011753 | 0 | No |
| 3 | 31 / 11,263 | 0.664797 | 0.659067 | 0.664524 | -0.005730 | -0.005457 | 0 | No |
| 4 | 40 / 12,129 | 0.656014 | 0.654961 | 0.656483 | -0.001053 | -0.001522 | 0 | No |

The unweighted fold means, reported descriptively and not used for the every-fold
verdict, are B0 `0.681146`, A1 `0.676120`, and A0 `0.680847` AUC. A1 improves B0
only in fold 0 and regresses in folds 1--4.

## Protocol integrity

- B0 was executed and sealed first. Its 45,961 actions, 134 positives, fold counts,
  AUCs, and retentions match the historical pre-A7 B0 seal exactly.
- A1 is B0 plus the single original A7 action-level ranking score,
  `action_margin`; no feature bundle or sign/transform search was performed.
- A0 deterministically permutes the one-column supplier vector within each
  video/source-frame stratum: 45,816 rows moved and 145 singleton rows remained
  fixed.
- The hash-pinned evaluator supplied the unchanged class-balanced ridge probe,
  folds, AUC, and atomic-tie CG0 retention implementation.
- A path-only B0 startup repair occurred before any shard was read. A later
  provenance-path formatting failure occurred after metric computation but before
  result publication; the exact deterministic score was replayed once under a
  serialization-only recovery receipt. Neither repair changed scientific semantics,
  and no configuration or threshold was searched.

## Campaign accounting

A7 is a CPU diagnostic, not a GPU training candidate, and its failed separation
gate opens no graph score for allocation classification. R1 remains the only G-03
model score: `0.7637338804169099` versus neighborhood floor
`0.868299238910600`, Branch B. Wave-R GPU allocation/spend remains `0/30 T4-h`;
best known remains `0.881522069960`. No edge or division graph term was run for A7.

Primary result: `run_v1/SEPARATION_RESULT.json`, SHA-256
`3143a31db3af952bc0c02d9583f1a9d7911b9bc335cb446ff9adedf4bdadf96d`.

