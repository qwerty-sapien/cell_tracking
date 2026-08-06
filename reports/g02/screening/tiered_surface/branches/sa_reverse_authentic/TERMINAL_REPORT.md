# Authentic reverse Trackastra supplier — N0–N3 terminal report

Disposition: **AUTHENTIC SUPPLIER PASS; FROZEN SEPARATION FAIL FOR BOTH CLASSES**

The owner-cmux Kaggle SDK download authenticated the exact historical H1
checkpoint by content, without using the permanently forbidden
`ListDatasetFiles` gate. `model.pt` is
`24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f`;
the uploaded `ctc.zip.bin` is
`2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e`;
and its `ctc/model.pt` member is byte-identical to the standalone model.

The supplier ran genuine reversed-time Trackastra inference. For each frozen
forward packing, time was mapped as `min(t)+max(t)-t`; coordinates and node IDs
were preserved; and canonical `u_t→v_t+1` evidence was returned from the newly
inferred `A_reverse[v,u]`. No forward-margin negation, matrix transposition,
duplication, recalibration, alternate checkpoint, or synthetic reverse was
used.

Coverage is exact: **36 videos / 3,539 transitions / 84,883 proposals /
121,534 rows**. All rows are finite. The 99-call witness
`train:44b6_0db75fae` is `np.array_equal` for every emitted array. Forward and
reverse margin signs agree on **95,666/121,534 = 0.787154212** rows; Pearson
margin correlation is **0.649644** and median absolute disagreement is
**0.196472**. This is label-free, **NON-GT-IDENTIFIABLE** evidence.

The first long guard correctly stopped on 140.875 MiB of external swapout
growth after 29 files. Those files were quarantined, then authenticated by
schema, frozen joins, forward-margin equality, finiteness, hashes, and
recomputed packing counts. A resume materialized only seven missing videos and
passed with 0 MiB swap growth. No incomplete manifest was ever cited.

The unchanged v3 five-fold per-class separation gate then evaluated the frozen
feature surface against sealed B0. Delete had authentic support on
109,569/1,345,497 actions and 367/1,118 beneficial actions; actions without
supplier evidence were explicitly zero with support=0. Its A1 AUC was
0.663796–0.769853, but retention was only 0–0.003268 and failed all five
folds. Occupied-swap had 45,961/45,961 support and all 134 beneficial actions;
its A1 AUC was 0.642024–0.844186, retention was zero in every fold, and all
five folds failed. The exact evaluator verdict token is `FAIL_PARK_C` for both
supplier×class units, but **no tier move is applied or proposed**.

AE-P6 and R8 remain unchanged. There is no graph-preregistration unlock, graph
run, 6bba, CV57, kernel launch, GPU use, budget debit, RunPod action,
submission, DEAD action, plan amendment, or tier-census mutation. The lane
halts at the human-review barrier.
