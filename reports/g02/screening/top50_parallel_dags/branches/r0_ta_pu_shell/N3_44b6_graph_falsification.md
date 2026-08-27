# R0 N3 — 44b6-first shell graph falsification

> Load `DAG.md`, all GO-qualified N0–N2 outputs, the frozen upstream no-shell graph,
> exact APNCC, authenticated 44b6 fit-only fragments, CG0 budgets, and the frozen
> scorer. Proposal/PU metrics remain diagnostic.

## Objective / gate

Does R0 produce a graph-level 44b6 gain without broad background admission? **GO**
requires total adjusted delta vs APNCC `>=+0.005`, incremental adjusted delta vs the
frozen upstream no-shell arm `>epsilon_null`, raw edge-Jaccard and node-recall deltas vs
that arm each `>=0`, observation-distinct output, and every family/per-video admission
budget satisfied.

## Tasks

1. Authenticate all hashes and reconstruct only the preregistered 36-video 44b6 OOF
   graphs for APNCC, the upstream no-shell arm, and R0.
2. Report total and incremental adjusted/raw/node deltas, TP/FP/FN, shell correct/false
   admissions, per-video support, groupwise risk bounds, and exact budget headroom.
3. Assert that no seed, group, prior, risk level, threshold, radius, architecture, or
   graph cost was changed after N1.
4. Label proposal precision, PU risk, p-values, and FDR estimates diagnostic; they
   cannot override the exact graph verdict.

## Output

Write `outputs/r0_n3_44b6_graph.{md,json}` plus proposed LEDGER row
`R0-TA-PU-SHELL-44B6 | SCREEN | total >=+0.005, incremental >null, caps | exact score
vector | identity/support | 0 external usage | verdict | reason`. Do not edit LEDGER;
update LIVE_STATE after the node verdict.

## Verdict

- **GO** — every gate in the Objective is met.
- **PARK** — total or incremental adjusted gain is positive but below its gate, or
  raw/node regresses without a broad-admission breach.
- **DEAD** — incremental adjusted delta is `<=0`, or any family/per-video background
  budget is breached; do not run 6bba/CV57 without a new named mechanism.
- **ESCALATE** — identity, leakage, scorer, or reproducibility cannot be certified.

## Cascade

- On **GO** → proceed directly to `N4_6bba_replication.md` in the same run; do not wait
  for a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
