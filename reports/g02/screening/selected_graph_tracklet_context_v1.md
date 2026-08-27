# Selected-graph tracklet context

**Decision: PARK the linear tracklet-context rule as built.**

This fit-only screen tested whether predecessor/successor context can safely
extend the successful pairwise NCC direct-swap rule. It used:

- 105 fit-training videos for deterministic five-fold video-grouped
  cross-validation and the final threshold;
- 12 fit-validation videos only for the registered support check;
- no CV57, audit, test, public, GPU, or submission signal.

The feature was local and symmetric:

- shared-source conflicts: predecessor-to-target NCC margin and acceleration
  residual;
- shared-target conflicts: source-to-successor NCC margin and velocity
  continuation residual.

Only selected edges already represented in the prepared conflict artifacts
were used as context, giving conservative context coverage of
`3,575 / 8,515 = 41.98%` training rows and
`126 / 439 = 28.70%` validation rows.

## Result

No coarse context/motion combination met the preregistered `0.95`
out-of-fold precision floor. Across the 16 combinations:

- out-of-fold precision ranged from `0.9428` to `0.9469`;
- pair-only NCC itself reproduced `0.9469` out-of-fold precision;
- the highest-utility context model used context weight `1.0` and motion
  weight `0.5`, reaching OOF utility `2,233` at precision `0.9469`;
- on fit-validation that model improved conservative utility by `+24`
  (`148 → 172`) but precision was only `0.9463`.

One context-only grid point happened to clear validation precision
(`0.9560`, utility `158`), but it failed independently in video-grouped
cross-validation (`0.9409`). It is not promoted.

The failure is not a claim that temporal context is useless. It establishes
that a coarse linear addition over a partial selected-edge context cannot
provide the fail-closed precision needed for deployment. Reconsider only with
full selected-graph context plus an explicit abstention/background objective;
do not spend a full CV57 replay on this implementation.

Authoritative machine report:
`reports/g02/selected_graph_tracklet_context_audit_v2.json`
(`sha256 cd0d72c7a5b4a0ed6060c77b34940694f2124ebe090ddb5b7c68c7f4500f0c59`).
