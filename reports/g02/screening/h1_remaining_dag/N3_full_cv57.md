# N3 — Full CV57 confirmation

> Load `DAG.md`, `../LIVE_STATE.md`, the GO result from
> `N2_held_fit_graph.md`, `../p0_parity.md`, and the frozen scorer contract.
> This is the first node authorized to access CV57.

## Objective / gate

Does the observation-distinct, graph-passing H1 candidate reach local CV57
`>=0.89` with no node-recall regression under the frozen official scorer?

## Tasks

1. Authenticate the N2 GO hashes and ensure all model/head choices were frozen
   before CV57 access.
2. Run the exact full CV57 package once; do not tune or select on CV57.
3. Compare against banked APNCC `0.8815220700`, report exact TP/FP/FN,
   adjusted/raw edge Jaccard, node recall, and observation identity.
4. Apply no public-score forecast beyond the coarse, rounding-bounded transfer
   statement in `p0_parity.md`.

## Output

Write `../h1_cv57_result.json` and the frozen public-package preregistration if
GO; update `../LIVE_STATE.md`. The root coordinator appends exactly one
`H1-ZWRFEAT-CV57` ledger row.

## Verdict & cascade

- **GO** only for local CV57 `>=0.89`, observation-distinct output, and no
  node-recall regression. Proceed directly to `N4_public_probe.md`.
- PARK below `0.89`; halt. H2 becomes eligible because H1 showed graph signal,
  but do not improvise or execute H2 inside this DAG.
- DEAD on scorer/identity failure; halt.
