# OCC-SLICED-W1-v2 local score and submission decision

**Decision: do not make a competition submission.** The remote feature run is a
clean, complete scientific execution, but the resulting supplier fails the frozen
44b6 occupied-swap separation gate and is not a competition artifact.

## Remote-run authentication

- Kernel: `onebitbrain/g02-occ-sliced-w1-repair2-v1`, exact Version 1.
- Owner-supplied log SHA-256: `2e157dbd188e235a0ae37dcc63ce6de239b8ce0fd59b65e1cbb1bca0d30068a1`.
- Step 0: PASS; exact dataset Version 1, 36 request files, 36 population files,
  and 45,961 requested actions.
- Terminal event: `status=COMPLETE` after `3026.241221044 s`; 36/36 identities,
  45,961/45,961 actions, 91,922 endpoint crops, and 45,815 moved A0 rows.
- There is no `_run_failure`, `KernelError`, or traceback. The final Mistune and
  nbconvert messages are export-time warnings, not application failures.
- The output download succeeded once through CMUX. It contains 36 feature shards,
  `FEATURE_MANIFEST.json`, `SEMANTIC_RECEIPT.json`, and the persisted run log.
- Every file and semantic-array hash matches the manifest/receipt. Population
  coverage is 100%; there are zero nonfinite, duplicate, or invalid rows.
- Feature-manifest SHA-256: `b8801845f3277c191772b519e0580b7b36e85a6c048bbf3f2a0a9752c34adfa3`.
- Semantic-receipt SHA-256: `b698efb7df3c2943ba5c4908c018b5c47b7d7298839599d9635e4a6608614723`.

The Kaggle API's downloaded log is a JSON stream serialization, while the supplied
UI log is a numbered plain-text rendering, so their container-file hashes differ.
Their scientific events and terminal hashes agree exactly.

## Frozen local score

The scorer reused the pinned five-fold, identity-grouped, class-balanced ridge probe
and sealed B0. A1 is B0 plus the 14 sliced-W1 features; A0 is B0 plus the pre-frozen
within-target-frame permutation control. No hyperparameter search, feature selection,
CV57 read, or graph tuning was performed.

| Fold | Beneficial | A1 AUC | B0 AUC | A0 AUC | A1 retention |
|---:|---:|---:|---:|---:|---:|
| 0 | 34 | 0.570830 | 0.625542 | 0.580098 | 0.000000 |
| 1 | 7 | 0.821975 | 0.837077 | 0.816507 | 0.000000 |
| 2 | 22 | 0.607195 | 0.622301 | 0.574779 | 0.000000 |
| 3 | 31 | 0.625241 | 0.664797 | 0.616531 | 0.000000 |
| 4 | 40 | 0.622125 | 0.656014 | 0.627868 | 0.000000 |

Aggregate A1 results:

- mean AUC `0.649473481` versus required `0.90`;
- empirical fold p05 AUC `0.578103180` versus robust reference `0.95`;
- mean and every-fold retention `0`, versus required mean `0.067910448`;
- selected beneficial/harmful actions at the exact CG0 budget: `0 / 0`;
- A1 AUC is below B0 in all five folds; the lift ranges from `-0.054712` to
  `-0.015102`;
- only fold 2 clears the A1-minus-A0 AUC lift clause, while no fold clears the
  absolute AUC or any retention clause.

Verdict: **`FAIL_PARK_C`**. The gate selects no actions, so its deployable graph is a
literal no-op with implied adjusted delta `0.0`; the graph preregistration is not
unlocked. An official graph score is therefore correctly recorded as not run, not as
a missing positive result.

Machine-readable result:
`local_score_v1/RESULT.json` (`0e9e2b6a55aa56d131690f6e048cb76b1dda294f84d4890b09b54bcfd324fe32`).

## Next steps

1. Park this exact OCC-SLICED-W1-v2 supplier. Do not retune its ridge, projection
   bins, crop, or thresholds after seeing this result; the frozen rerun rule requires
   a materially new mechanism.
2. Execute `CRW-COMPONENT-REWRITE-v1/A0` first. It is the cheapest decisive test and
   has the strongest banked realistic headroom: replay 459/459 complete transactions,
   validate 36/36 graphs and five retained anchors, and require exact T2 equality.
   Dispatch its label-free A1 supplier only on that structural GO.
3. In parallel, execute `DIV-UOT-FORK-BAYES-v1/B0`, the legal division-fork ceiling.
   Continue to UOT/Optuna/Gurobi only if the exact 44b6 ceiling is at least `+0.005`,
   raw/node deltas are nonnegative, and at least 20 beneficial legal events exist.
4. Keep GPU allocation at zero until a frozen 44b6 policy clears its exact graph gate.
   Use the portfolio Gurobi barrier only after at least one branch is scientifically GO;
   then open 6bba and CV57 in that order. No current result licenses a competition
   submission.

