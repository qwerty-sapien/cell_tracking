# R0 N4 — 6bba replication

> Load `DAG.md`, GO-qualified `outputs/r0_n3_44b6_graph.{md,json}`, all frozen
> contracts, authenticated 6bba fit-only fragments, exact budgets, and the scorer.

## Objective / gate

Does unchanged R0 replicate on 6bba with total adjusted delta vs APNCC `>=+0.005`,
incremental adjusted delta vs the frozen upstream no-shell arm `>epsilon_null`, raw
edge-Jaccard and node-recall deltas vs that arm `>=0`, and no admission-budget breach?

## Tasks

1. Verify every representation/model/prior/group/risk/threshold hash is identical to
   N3; apply only the preregistered cross-fit procedure.
2. Score APNCC, upstream no-shell, and R0 on all 69 fit-only 6bba graphs.
3. Report the complete total/incremental score vector, shell TP/FP/FN, groupwise
   diagnostics, support, and exact family/per-video budget headroom.
4. Compute the minimum-family total and incremental gain; do not route by family.

## Output

Write `outputs/r0_n4_6bba_graph.{md,json}` plus proposed LEDGER row
`R0-TA-PU-SHELL-6BBA | REPLICATION | total >=+0.005, incremental >null, caps | exact
score vector | identity/support | 0 external usage | verdict | reason`. Do not edit
LEDGER or LIVE_STATE.

## Verdict

- **GO** — every gate in the Objective is met and output is observation-distinct.
- **PARK** — total or incremental gain is positive but below gate, or raw/node
  regresses without a broad-admission breach.
- **DEAD** — incremental adjusted delta is `<=0`, or any background budget is breached.
- **ESCALATE** — identity, leakage, scorer, or reproducibility cannot be certified.

## Cascade

- On **GO** → proceed directly to `N5_cv57_closeout.md` in the same run; do not wait for
  a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.

