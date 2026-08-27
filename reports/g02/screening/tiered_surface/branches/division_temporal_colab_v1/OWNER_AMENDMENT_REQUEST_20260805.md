# Owner amendment request: division parent-event weight refinement

Status: **SUPERSEDED — NO WAIVER REQUIRED**. The owner subsequently clarified that the longer-context input and temporal architecture define a new model, not refinement of V32. See `OWNER_NEW_ARCHITECTURE_CLARIFICATION_20260805.md`.

The current campaign rule blocks weight refinement below local `0.916`. That rule was adopted while the remaining score gap was treated as a broad structural gap. The current evidence is narrower:

- V32 is a faithful immutable-weight replication of public `0.908--0.911` behavior and remains untouched by this lane.
- Detection/ILP parameter probes are flat at `0.908`; seed and ensemble gains are only about `+0.002` and `+0.001`.
- The existing conditional daughter-pair head is useful (`AUROC 0.897081`, `AP 0.368520`), while parent-event detection is random and V32's two-frame division logic is architecturally short of event context.
- The proposed work adds a named missing capability: a longer-context, division-weighted **parent-event detector**. It does not rebuild daughter assignment or reopen general V32 tuning.

Requested amendment:

> After the APNCC-vs-V32 taxonomy is published and hash-bound, permit one preregistered longer-temporal-context parent-event training program on the real competition training data, subject to the already-frozen deployment gate (`top 22: >=5 TP and <=17 FP`, `AP lift >=8462x`, AUROC never qualifying), complete Colab receipts, and reuse of the frozen conditional daughter-pair head.

Scope before amendment: rules adjudication, preprocessing, integrity tests, and checkpoint/resume infrastructure are allowed. Scientific training, loss finalization, checkpoint selection, and weight deployment remain blocked.

Owner decision: `SUPERSEDED_BY_NEW_ARCHITECTURE_SCOPE_CLARIFICATION`

Owner/date: `____________________`
