# C2-N1 — Implement and prove the enumerator

> Load `DAG.md`, `results/C2_N0_preregistration.md`, the frozen C2 contract, and the
> U0/C0 implementation manifests. Do not open held labels or run a graph score.

## Objective / gate

Does the C2 implementation return the exact ordered feasible assignments and
min-marginals deterministically? **GO** requires equality with exhaustive enumeration
on every small fixture, exact replay, and unchanged input costs/control outputs.

## Tasks

1. Implement deterministic component decomposition, Murty/Lawler enumeration,
   constrained include/exclude solves, and whole-component APNCC fallback.
2. On exhaustive synthetic matrices with ties, no-link choices, forbidden edges, and
   one-to-one conflicts, compare the complete first-eight order/cost set and every
   min-marginal to brute force.
3. Test invariance to row/column relabeling after inverse mapping and to a permissible
   constant shift of fixed-cardinality fixtures; test deterministic tie breaking.
4. Assert no coefficient, cost normalization, K, stability rule, or fallback is
   configurable at graph-evaluation time. Negative-test all validation/CV paths.
5. Run a label-blind fit105 feasibility audit. GO requires coverage of all eligible
   components under the preregistered local wall/RSS limit and zero swap growth.

## Output

Write `results/C2_N1_implementation_verification.md` and a hashed implementation/test
manifest. Include a proposed `C2-KBEST-IMPLEMENT` ledger row (`IMPLEMENT / ALPHA`,
`NOT SCORED`, zero external usage) for the root coordinator.

## Verdict

- **GO** only for `100%` brute-force fixture agreement, deterministic replay,
  immutable costs/rule, complete feasibility, and zero forbidden access.
- PARK on a correctable resource or implementation failure before scoring.
- DEAD on algorithmic disagreement, non-determinism, or inability to preserve costs.
- ESCALATE if feasibility requires relaxing K/rule, split, or resource governance.

## Cascade

- On **GO** → proceed directly to `N2_graph_44b6.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
