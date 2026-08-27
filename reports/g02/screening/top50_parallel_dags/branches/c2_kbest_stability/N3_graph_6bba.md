# C2-N3 — 6bba transfer and standalone gate

> Load `DAG.md`, the C2-N2 GO result, all frozen hashes, and the held-fit 6bba
> graph/scorer identities. No setting may change after observing 44b6.

## Objective / gate

Does the identical C2 rule achieve adjusted edge-Jaccard delta `>=+0.005` on 6bba
and `>=+0.005` retrospectively on 44b6, with raw/node deltas `>=0` in each family?

## Tasks

1. Prove byte-identical C2 contract/implementation hashes and no post-44b6 tuning.
2. Run APNCC, same-cost C0, and C2 once on the frozen 6bba held-fit graph set.
3. Report the same graph vector as N2 plus minimum-family delta and attribution over
   C0. Do not aggregate families to hide a weak 44b6 result.
4. Classify a reproducible `>=+0.003` but `<+0.005` minimum-family result only as
   stack evidence; it does not open CV57.

## Output

Write `results/C2_N3_6bba_graph.md` and hashed combined-family result JSON. Include a
proposed `C2-KBEST-6BBA` ledger row (`SCREEN / ALPHA`, both exact family vectors,
zero external usage) for the root coordinator.

## Verdict

- **GO** only if adjusted delta is `>=+0.005` in each family, raw/node deltas are
  `>=0`, output is distinct, and attribution over C0 remains positive.
- PARK if both families are `>=+0.003` but either is `<+0.005`, or on any non-kill
  raw/node/attribution failure. Preserve as fixed stack evidence only.
- DEAD if 6bba adjusted delta is `<=0` or a structural identity failure is proven.
- ESCALATE on scorer, split, provenance, or frozen-contract drift.

## Cascade

- On **GO** → proceed directly to `N4_cv57.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
