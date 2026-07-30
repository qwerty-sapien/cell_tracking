# Two-supplier frozen separation gate — terminal report

Status: **BOTH SUPPLIERS INVALID / NO SCIENTIFIC VERDICT / HALT**

The frozen gate was not altered: preregistration
`9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e`,
sealed B0 `a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2`,
and population manifest
`4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c`.
Both suppliers were assigned the same 1,345,497-delete / 45,961-occupied-swap
population, five identity-grouped OOF folds, deterministic ridge rank probe,
A1 incumbent-plus-supplier arm, and within-frame node-consistent A0 permutation.
B0 was consumed first from the sealed artifact and was not recomputed.

| Supplier | Delete | Occupied swap | A1 / A0 | State |
|---|---|---|---|---|
| DINOv2 ViT-S/14 R5 XY | `INVALID_RESOURCE_NO_VERDICT` | `INVALID_RESOURCE_NO_VERDICT` | unrun / unrun | PARK-B unchanged |
| CELLECT R7 v4 | `INVALID_NO_VERDICT` | `INVALID_NO_VERDICT` | unrun / unrun | PARK-B unchanged |

DINOv2 produced zero complete feature shards because both authorized local paths
tripped the binding zero-swap guard before materialization. CELLECT's one
authorized Kaggle run authenticated checkpoint SHA `f3e8e730…89ce5` byte-exact
at step 0, then stopped before feature computation with
`FeatureError: no frozen feature requests found`. The CELLECT run debited
0.3086 kernel-h; the weekly budget is 2.8847 used / 27.1153 remaining.

DAG 2 run_v6 subsequently completed and supersedes the stale null reference in
the original report. Its hash-pinned requirement surface measures the +0.005
mean/robust-p05 AUC requirements as **0.85/0.90 for delete** and **0.90/0.95
for occupied swap**. Mean retention requirements are 0.020394 and 0.067910;
robust-p05 retention requirements are 0.044946 and 0.147388. Neither supplier
can be assessed against them in this historical v1 result because both supplier
attempts stopped before A1/A0.

No class received PASS or FAIL, so neither supplier-class graph-gate
preregistration is unlocked and neither instantiation moves to PARK-C. No
training, full-pool cache, graph contract/run, 6bba, CV57, submission, or RunPod
action occurred. The barrier is closed pending human review of any new supplier
compute or transport design.
