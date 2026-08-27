# C1 N4 — 6bba replication

> Load `DAG.md`, the GO-qualified `outputs/c1_n3_44b6_graph.{md,json}`, all frozen
> upstream contracts, authenticated 6bba fit-only fragments, and the frozen scorer.

## Objective / gate

Does the unchanged C1 arm replicate on 6bba at adjusted edge-Jaccard delta
`>=+0.005` vs APNCC with raw edge-Jaccard and node-recall deltas each `>=0`?

## Tasks

1. Verify the model/configuration/inference hashes are identical to N3; do not refit,
   retune, or family-specialize outside the preregistered grouped-OOF procedure.
2. Reconstruct and score the 69 fit-only 6bba graphs against exact APNCC and C0.
3. Report adjusted/raw/node deltas, TP/FP/FN, observation distinctness, runtime, and
   proposal diagnostics in a non-gating appendix.
4. Confirm the minimum-family adjusted delta and the unchanged C1-minus-C0 44b6 alpha
   attribution from N3.

## Output

Write `outputs/c1_n4_6bba_graph.{md,json}` plus proposed LEDGER row
`C1-TA-HOCT-LITE-6BBA | REPLICATION | >=+0.005 | exact score vector | identity |
0 external usage | verdict | reason`. Do not edit LEDGER; update LIVE_STATE after the
node verdict.

## Verdict

- **GO** — adjusted delta `>=+0.005`, raw delta `>=0`, node-recall delta `>=0`, and
  observation-distinct predictions.
- **PARK** — adjusted delta is positive but below `+0.005`, or raw/node regresses.
- **DEAD** — adjusted delta is `<=0`; the cross-family mechanism failed.
- **ESCALATE** — identity, leakage, scorer, or reproducibility cannot be certified.

## Cascade

- On **GO** → proceed directly to `N5_cv57_closeout.md` in the same run; do not wait for
  a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
