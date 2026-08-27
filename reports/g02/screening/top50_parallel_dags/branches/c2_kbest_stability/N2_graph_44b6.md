# C2-N2 — 44b6-first held-fit graph gate

> Load `DAG.md`, C2-N0/N1 GO artifacts, the locked U0/C0 contracts, and the frozen
> held-fit graph/scorer identities. This is the first objective-scoring node.

## Objective / gate

Does fixed C2 assignment stability improve 44b6 adjusted edge Jaccard over APNCC by
at least `+0.003`, with raw edge-Jaccard and node-recall deltas both `>=0` and an
observation-distinct graph?

## Tasks

1. Authenticate every upstream hash and prove no 44b6 held label affected costs, K,
   min-margins, stability, fallback, or arm selection.
2. Run exactly APNCC, same-cost C0 rank-1/MAP, and C2 on the identical frozen 44b6
   held-fit set. Do not run 6bba in the same invocation.
3. Report adjusted/raw edge Jaccard, node recall, TP/FP/FN, changed components,
   fallback/support, and wall/RSS. Proposal metrics are diagnostics only.
4. Compare C2 to APNCC (campaign objective) and the stronger of APNCC/C0 (mechanism
   attribution). Record the exact null/replay envelope.

## Output

Write `results/C2_N2_44b6_graph.md` plus a hashed score/result JSON. Include a proposed
`C2-KBEST-44B6` ledger row (`SCREEN / ALPHA`, exact graph deltas, zero external usage)
for the root coordinator; update live state through the coordinator.

## Verdict

- **GO** only for adjusted delta `>=+0.003` versus APNCC, raw/node deltas `>=0`,
  distinct output, and a positive attributable delta versus same-cost C0.
- PARK for adjusted delta in `(0,+0.003)`, any raw/node regression, identity-null
  output, or no gain over C0.
- **DEAD** at adjusted delta `<=0`; halt before 6bba/CV57.
- ESCALATE on scorer, split, contract, or provenance mismatch.

## Cascade

- On **GO** → proceed directly to `N3_graph_6bba.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
