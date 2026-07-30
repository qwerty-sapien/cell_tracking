# Two-supplier separation bake-off — terminal cross-read

Both repaired supplier attempts are terminal **INVALID**, not scientific FAILs.
Neither supplier produced a complete authenticated per-class feature manifest, so
the sealed B0 was not followed by A1/A0 and the frozen separation gate issued no
PASS/FAIL verdict.

| Supplier | DELETE | OCCUPIED_SWAP | Frozen gate | run_v6 requirement | State |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 XY-beta | `INVALID_RESOURCE_NO_VERDICT` after 22/41 quarantined shards | `INVALID_NOT_RUN_BINDING_RESOURCE_STOP` | Not evaluated | Not evaluated | R5 remains PARK-B |
| CELLECT R7 | `INVALID_INFRASTRUCTURE_NO_VERDICT` | `INVALID_INFRASTRUCTURE_NO_VERDICT` | Not evaluated | Not evaluated | R7 remains PARK-B |

The common identity stayed fixed: population `4ab7bb23…76f0c`, sealed B0
`a9ce8c0b…b4ff2`, five folds, weighted ridge probe, and preregistration
`9fb030bd…3310e`. The frozen per-fold gate remains A1 AUC ≥0.65 with lifts
≥0.05 over B0 and ≥0.03 over A0, plus retention ≥0.15 with the same lifts and
per-video FP integrity.

The reporting-only run_v6 requirements are delete mean/robust-p05 AUC
0.85/0.90 and retention 0.020394/0.044946; occupied swap AUC 0.90/0.95 and
retention 0.067910/0.147388. No supplier measurement exists to compare against
them. DELETE remains the primary class: its compiler-valid oracle ceiling
`+0.081897667` is about 4.41× occupied swap's `+0.018571397`.

DINOv2 passed its fresh zero-swap preflight and a 4.362940 h MPS projection,
then stopped at +140.875 MiB swap after 7,877.258 s. Its 724,634 partial rows
are quarantined and non-consumable. CELLECT passed local content discovery, but
the sole bounded T4 run failed in step 0 at 3.956 s because the exact private
mount yielded zero matches for the pinned bundle-manifest content. Its
conservative debit is 0.0188 GPU-h; weekly budget is 2.9035 used / 27.0965
remaining with no active authorization.

There are zero PASS unlocks, zero scientific FAIL/PARK-C moves, and no graph
preregistration. The graph-contract barrier remains closed for human review.
