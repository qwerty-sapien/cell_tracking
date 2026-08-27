# N1 — feasibility-exact action compiler

> Load `DAG.md`, `N0_ACTION_ORACLE.md`, the P0 JSON, and the final APNCC graph schema.

## Objective / gate

Can one frozen compiler accept a fully specified atomic action and produce a valid
final graph for every unit fixture, every single retained action, and every combined
batch? GO requires 100% validity before labels or held outcomes can freeze support.

## Tasks

1. Define add/delete/swap/fork/restore schemas with explicit no-op.
2. Resolve endpoints against the final graph, carrying all incident conflicts.
3. Enforce time direction, <=1 parent, <=2 children, no merge/dangling endpoint.
4. Hash action and graph identities; test unit, single-action, and batch application.

## Output

Frozen schema/implementation/test hashes, full validity audit, LIVE_STATE update, and
a proposed ledger row.

## Verdict & cascade

- **GO** (100% valid and deterministic) -> launch
  `N2_ASSOCIATION_REALIZERS.md` and `N3_REPRESENTATION_ROBUSTNESS.md` in parallel.
- PARK / DEAD / ESCALATE / HUMAN-DECISION -> halt and report.

