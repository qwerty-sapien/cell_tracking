# N4 division dependency and audit preregistration

Frozen: 2026-07-28, before inspection of banked association-arm outcome artifacts.

## Decision question

Does frozen root evidence contain one banked association-continuity arm that passes
both required families and therefore opens division fitting?

## Eligibility predicate

For a candidate arm `a`, define:

```text
eligible(a) =
  committed_or_root_banked(a)
  and resolvable_artifact_path(a)
  and sha256_authenticated(a)
  and one_immutable_arm_identity(a)
  and adjusted_delta(a, 44b6) >= 0.005
  and raw_delta(a, 44b6) >= 0
  and node_delta(a, 44b6) >= 0
  and adjusted_delta(a, 6bba) >= 0.005
  and raw_delta(a, 6bba) >= 0
  and node_delta(a, 6bba) >= 0
```

All comparisons are against the arm's frozen graph beta under the unchanged
official scorer. Missing family results fail eligibility; they are not zero and are
not imputed. Proposal, utility, oracle, AUC, candidate-count, and public/CV57-only
values are nonqualifying. Vectors from different arm identities may not be spliced.

## Audit sources and order

After this file is frozen, inspect only frozen root evidence in this order:

1. `LIVE_STATE.md` for the current banked frontier and explicit dependency count.
2. The banked D1 persistent-fork `N2_association_dependency.md` and terminal report
   for its frozen dependency inventory and preservation boundary.
3. Root portfolio synthesis/owner-resolution artifacts only as needed to authenticate
   arm identity and exact family vector provenance.
4. The cited arm result artifact itself for exact vectors and SHA-256.

Do not inspect any Phase-1 sibling subtree or use active sibling claims.

## Predetermined outcomes

- **No eligible arm:** `PARK_DEPENDENCY`. Do not fit or execute either family.
  Proceed only with synthetic N1 division-fork/no-op structural validation and
  immutable official scorer semantic fixtures. Keep `DIV-F1` at its existing
  dependency-blocked tier; propose no tier move.
- **At least one eligible arm:** `STOP_AND_REPORT_ROOT`. Record path, SHA-256,
  immutable arm ID, and exact both-family vectors; notify the root coordinator
  before any fitting. This lane does not infer authority to fit.

## Fixture-only acceptance

The safe audit may confirm:

- N1 accepts a complete binary fork as one atomic action;
- explicit no-op leaves the graph hash unchanged;
- omitted incident conflicts, merges, backward-time edges, missing endpoints, or
  outdegree greater than two fail closed;
- official division fixtures preserve the already-banked direct/late/persistent
  fork semantics and duplicate-survivor constraints.

These fixtures cannot establish association continuity, open held outcomes, move a
tier, or supersede D1.

## Quarantine

The external memo's C2/K-best values and Branch-B `66/223` audit are unbanked and
remain quarantined. They are excluded even if numerically above the gate.

