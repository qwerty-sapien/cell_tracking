# C1 N3 — 44b6-first graph falsification

> Load `DAG.md`, all N0–N2 outputs, the authenticated 44b6 fit-only graph fragments,
> exact APNCC/C0 controls, and the frozen scorer. This is the first outcome-opening
> node; proposal metrics remain diagnostic.

## Objective / gate

Does C1 improve the exact held-fit 44b6 graph objective enough to be both useful and
differentiating? **GO** requires adjusted edge-Jaccard delta vs APNCC `>=+0.005`, raw
edge-Jaccard delta `>=0`, node-recall delta `>=0`, observation-distinct predictions,
and adjusted C1-minus-C0 `>=max(0.001, epsilon_null)`.

## Tasks

1. Authenticate N1/N2 hashes, repeat the clean resource preflight, and reconstruct only
   the preregistered 36-video 44b6 grouped-OOF graphs.
2. Score APNCC, C0, and C1 with the exact same frozen scorer and fragments. Report the
   complete adjusted/raw/node vector and TP/FP/FN counts.
3. Compute the preregistered identical-repeat null and assert no adaptive retry, seed,
   threshold, architecture, or cost choice occurred.
4. Report proposal precision/AUC/support only in a clearly labeled diagnostic section.
5. Produce one verdict from the thresholds below before touching 6bba.

## Output

Write `outputs/c1_n3_44b6_graph.{md,json}` plus proposed LEDGER row
`C1-TA-HOCT-LITE-44B6 | SCREEN | >=+0.005 and >=+0.001 over beta | exact score vector |
identity/observation class | 0 external usage | verdict | reason`. Do not edit LEDGER;
update LIVE_STATE after the node verdict.

## Verdict

- **GO** — every gate in the Objective is met.
- **PARK** — adjusted delta is positive but `<+0.005`, raw/node regresses, or C1 fails
  the alpha margin over C0 despite remaining positive.
- **DEAD** — adjusted delta is `<=0`; do not run 6bba or CV57 without a new named
  mechanism.
- **ESCALATE** — identity, leakage, scorer, or reproducibility cannot be certified.

## Cascade

- On **GO** → proceed directly to `N4_6bba_replication.md` in the same run; do not wait
  for a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
