# A7 phase-template occupied-swap screen — terminal report

## Decision

**KILL on 44b6.** The sealed five-fold OOF selector abstained completely: zero
held actions passed into the node-disjoint edit set. The compiled candidate is
byte-identical to APNCC for all 36 videos, so it cannot beat APNCC at the graph
gate. Per the preregistration, no 6bba, CV57, Kaggle, GPU, or submission action
was run.

## Exact 44b6 graph result

| Quantity | APNCC | A7 candidate | Delta |
|---|---:|---:|---:|
| Adjusted edge Jaccard | 0.8665345260361912 | 0.8665345260361912 | 0.0000000000000000 |
| Raw edge Jaccard | 0.8625960574674240 | 0.8625960574674240 | 0.0000000000000000 |
| Node recall | 0.9926180002844743 | 0.9926180002844743 | 0.0000000000000000 |

- Frozen action population: 45,961 occupied-endpoint conflict swaps across 36 videos.
- Threshold-pass support after sealed per-fold choice: 0.
- Applied edit-set size: **0** across 0 videos.
- Postselection correct/harmful: 0 / 0; precision is undefined because the selector abstained.
- Candidate edge counts: TP 10,327; FP 958; FN 687, identical to APNCC.
- APNCC comparison: failed. Historical APNCC banked gain context is +0.008372476430831166; A7 added 0.

## Governing-lesson compliance

| Property | Satisfied? | Evidence |
|---|---|---|
| Appearance-grounded | Yes | The only proposed evidence was apodized 3-D phase correlation over multi-frame APNCC tracklet templates. |
| Narrow | Yes | The supplier was restricted to the 45,961 frozen occupied-endpoint conflict swaps; final support was 0, never the raw shell or full action pool. |
| Node-disjoint | Yes | The frozen selector used strict touched-node disjointness; the empty final set satisfies it vacuously. |
| Recall-preserving | Yes | All 36 candidate fragments are byte-identical to APNCC, so nodes and edges—and therefore raw score and node recall—are unchanged. |

The design encoded the observed ranking that smaller incumbent replacement is
safer: it could only perform local N1-valid occupied-conflict swaps, used a
small edit budget and per-video cap, and abstained when sealed training evidence
did not transfer. It never rescored or reranked the whole pool.

## Audit chain

The first complete guarded scientific execution ran 327.936 seconds with zero
swap growth, completed the official exact score, exact/algebra parity, and APNCC
parity, then failed only while serializing aggregate edge counts because the
official summary omits those count keys. The scorer was not replayed for the
terminal recovery. `EXECUTION_REPAIR_3.json` admitted recovery only if all 36
sealed candidate fragment hashes equalled their pinned APNCC control hashes;
that condition passed. `GUARD_RECOVERY_3.json` records a second clean 45-second
zero-swap preflight and zero swap growth during result serialization.

Primary artifacts:

- `PREREGISTRATION.json` — frozen design, thresholds, budgets, gates, and kill line.
- `run_v1/FEATURE_MANIFEST.json` — 36 sealed label-blind feature shards / 45,961 actions.
- `run_v1/RESULT.json` — exact terminal vector, edit count, properties, and 36 fragment hash records.
- `run_v1/GUARD_RETRY_2.json` — prior complete zero-swap exact execution ending at the reporting exception.
- `run_v1/GUARD_RECOVERY_3.json` — guarded terminal recovery receipt.
- `VALIDATION.json` — independent result and fragment validation.
