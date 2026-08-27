# N4 — 44b6 held-fit graph kill

> Load `DAG.md`, the approved `runs/<run-id>/N3_cache_attestation.md`, the frozen arm
> manifests, and the authenticated 44b6 held-fit graph/scorer. Do not alter features,
> encoder, folds, head, or thresholds.

## Objective / gate

Does tri-view dense identity improve the hardest-family frozen graph and add value
beyond the single-view beta? **GO** requires tri-view adjusted graph delta `>=+0.005`,
raw edge delta `>=0`, node-recall delta `>=0`, and tri-view adjusted score strictly
greater than XY-only dense beta. Adjusted delta `<=0`, any protected regression, or
failure to beat the beta is a kill of the tri-view alpha.

## Tasks

1. Fit the frozen `<=100k` heads on five grouped fit folds and score the four arms on
   44b6 with identical graph/scorer manifests.
2. Report exact graph deltas, event counts, per-video effects, runtime, and the
   predeclared permutation-ablation result; proposal metrics remain diagnostics.
3. Do not choose another encoder/view/crop from the held result. Preserve any
   independently valid XY-only beta result as a separate branch fact.

## Output

Write `runs/<run-id>/N4_44b6_graph.md`, exact scorer artifacts and hashes, all four arm
results, and a proposed root-coordinator ledger row.

## Verdict

- **GO** — absolute graph gate and strict tri-view-over-XY alpha attribution pass.
- **PARK** — tri-view passes the absolute gate but does not beat XY; retain the beta,
  PARK the alpha.
- **DEAD** — tri-view adjusted delta `<=0`, raw/node regresses, provenance/leakage
  fails, or the permutation ablation reproduces the claimed gain.

## Cascade

- On **GO** → proceed directly to `N5_6bba_replication.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not run 6bba.

