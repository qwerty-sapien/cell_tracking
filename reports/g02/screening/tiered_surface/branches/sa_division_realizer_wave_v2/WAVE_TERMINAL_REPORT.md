# Division realizer wave v2 — full-38 barrier

Status: **no passing arm; downstream barrier remains closed**.

The human-approved §6.5 correction makes all 38 repair-target events across
both families the binding division screen. Family slices below are diagnostics,
not gates. A, clean B, and C were sealed before the target was opened, GT
bundles were never features, and every arm has zero duplicate survivors. A
postrun provenance audit fail-closed clean D as nonbinding; its empty content
remains diagnostic only.

| Arm | Sealed predictions | Binding TP / FP / FN | Division Jaccard | Adjusted / raw / node delta | Exact combined delta | Verdict |
|---|---:|---:|---:|---:|---:|---|
| A geometry/motion | 23 | 0 / 23 / 38 | 0 | -0.000022335 / -0.000022463 / 0 | -0.000022335 | FAIL-PARK-C |
| B persistence | 48 | 0 / 48 / 38 | 0 | +0.000029258 / +0.000028813 / 0 | +0.000029258 | FAIL-PARK-C |
| C appearance | 22 | 0 / 22 / 38 | 0 | -0.000022335 / -0.000022463 / 0 | -0.000022335 | FAIL-PARK-C |
| D exact 3-of-3 | 0 | 0 / 0 / 38 | 0 | 0 / 0 / 0 | 0 | INVALID-NONBINDING |

Arm B's tiny positive edge delta does not rescue it: it recovered no frozen
division target and therefore fails the binding Jaccard `>=0.30` clause. Arm
D's target-free content is an exact empty 3-of-3 intersection, but its clean
preregistration SHA is not the D SHA pinned by either pre-outcome
implementation freeze. It is therefore invalid/nonbinding; no retroactive
freeze was created. See
`arm_d_consensus_clean/PROVENANCE_ADJUDICATION.json`.

## Per-family generalization diagnostics

| Arm | 44b6 TP / FP / FN; Jaccard | 6bba TP / FP / FN; Jaccard |
|---|---|---|
| A | 0 / 12 / 9; 0 | 0 / 11 / 29; 0 |
| B | 0 / 19 / 9; 0 | 0 / 29 / 29; 0 |
| C | 0 / 8 / 9; 0 | 0 / 14 / 29; 0 |
| D (nonbinding diagnostic) | 0 / 0 / 9; 0 | 0 / 0 / 29; 0 |

The official full-57 scorer counted division TP/FP/FN of `0/1/42`,
`0/3/42`, `0/1/42`, and `0/0/42` for A through D respectively. Those scorer
counts cover every division in the 57 graphs; the table above is the
human-approved 38-event repair-target falsifier.

## Appearance and resource discipline

Arm C is now genuinely tested. It built 57 atomic per-video cache files from
385 graph-local candidates. The cache is 476,307 bytes total and stores only
patch summaries; it contains no voxels, embeddings, labels, or full-pool
inventory. C sealed 22 predictions after the frozen mass-conservation,
daughter-balance, and parent-roundness filters.

A, B, and C each passed a fresh 45-second idle preflight and completed with
exactly zero cumulative swapout growth. No Kaggle CPU fallback, GPU, RunPod,
submission, or full-pool cache was used.

Arm B was authored and generated in a fresh no-history worker. Its firewall
receipt records 57 fragment reads, one generic compiler-interface read, and
exactly zero target, label, scorer, evaluator, outcome, central-state, or
other-arm reads. Seven target-free tests and an independent structural replay
passed. The coordinator's earlier `0/53/38` B arithmetic and its consensus are
retained only as diagnostic artifacts after fail-closing the coordinator's
prior target-schema exposure; they do not enter this report.

## FP discipline

Every arm enforces one event per video and one-to-one target matching; duplicate
survivors are exactly zero. The preregistered sensitivity remains:

- 25 TP / 0 FP: `+0.0595`
- 25 TP / 5 FP: `+0.0532`
- 25 TP / 10 FP: `+0.0481`

Neither valid arm A, clean B, nor C passes, and D is nonbinding. Nothing
unlocks fitting, graph continuation, CV57, submission, GPU, RunPod, or a DEAD
action. Hold at the full-38 human barrier.
