# N2 — 44b6-first held-fit graph gate

> Load `DAG.md`, `../LIVE_STATE.md`, the approved N1 tolerance contract,
> `../h1_zero_wrfeat_graph_gate_contract.json`, and the authenticated cache.

## Objective / gate

Does frozen zero-wrfeat H1 improve the held-fit frozen-scorer graph objective
by adjusted edge Jaccard `>=+0.005` on both families, with raw edge Jaccard and
node recall deltas each `>=0`?

## Tasks

1. Run the 45-second zero-swap preflight and the pinned graph coordinator.
2. Execute 44b6 first. If its adjusted delta is `<=0`, kill immediately and do
   not call the 6bba evaluator.
3. If 44b6 is positive but below `+0.005`, or raw/node recall regresses, PARK
   without 6bba.
4. Only after a complete 44b6 pass, run the same frozen gate on 6bba. Report
   proposal precision/support but never use them as gates.

## Output

Write the per-family graph reports plus `../h1_graph_gate_result.json`; update
`../LIVE_STATE.md`. The root coordinator appends exactly one
`H1-ZWRFEAT-GRAPH` ledger row.

## Verdict & cascade

- **GO** only when both families have adjusted delta `>=+0.005`, raw delta
  `>=0`, node-recall delta `>=0`, and observation-distinct outputs. Proceed
  directly to `N3_full_cv57.md`.
- PARK for a weak positive or any non-kill regression; halt.
- DEAD when 44b6 adjusted delta is `<=0`; halt before 6bba/CV57 and report that
  `H3-MASK-SOURCE` is now eligible. Do not fabricate masks or start it here.
