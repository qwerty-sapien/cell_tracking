# N1 — Authenticate cache and M5↔T4 tolerance

> Load `DAG.md`, `../LIVE_STATE.md`, `../h1_cache_prep.json`, the human-returned
> `trackastra_h1_run_manifest.json`, and all downloaded cache manifests.
> **HUMAN-GATE:** tolerance acceptance is a human judgment.

## Objective / gate

Is the downloaded T4 cache complete, fit105-only, repeatable, and free of
structural cross-device drift? Pass analytically only with 105 authenticated
videos (36 44b6 + 69 6bba), exact integer/geometry fields, T4↔T4 equality, and
per-column float/neural differences inside the preregistered probe
`atol=1e-7, rtol=1e-5`.

## Tasks

1. Authenticate the run manifest, contract, implementation, checkpoint, and
   every atomic video output; reject partial or resumed identity drift.
2. Confirm no validation12/CV57/holdout identity or path appears.
3. Compare the named M5 and T4 witnesses. Treat any integer, identity,
   geometry, coherent-column offset, or out-of-probe difference as structural.
4. For benign float drift only, emit one tolerance proposal per column using
   the locked 2× observed-error rule. Do not approve it on the human's behalf.

## Output

Write `../h1_cache_attestation.json` and `../h1_m5_t4_tolerance_proposal.json`;
update `../LIVE_STATE.md`. The root coordinator appends exactly one
`H1-ZWRFEAT-CACHE-ATTEST` ledger row.

## Verdict & cascade

- **HUMAN-DECISION** if coverage and structure pass and only human tolerance
  approval remains. Halt. After explicit approval, dispatch
  `N2_held_fit_graph.md`.
- PARK on incomplete cache or benign but unapproved drift.
- DEAD on authenticated structural drift or T4 nondeterminism.
