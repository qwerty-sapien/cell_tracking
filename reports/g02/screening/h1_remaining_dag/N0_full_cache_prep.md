# N0 — Pin the full-cache package

> Load `DAG.md`, `../LIVE_STATE.md`, `../h1_timing_smoke_3.json`,
> `../h1_smoke_attempt_3.json`, and `../h1_execution.md`. **HUMAN-GATE:** the
> next action versions a Kaggle dataset and launches a GPU kernel.

## Objective / gate

Can the timing-only package be rotated to a fail-closed full-cache package
without changing model, data scope, or environment? Pass only if both timing
artifacts authenticate, T4 repeatability is PASS, projected runtime is
`<=7 h`, and the package still covers exactly 36 44b6 + 69 6bba fit105 videos.

## Tasks

1. Pin the timing artifact and coordinator attestation in the Option-C contract.
2. Set full-cache authorization only for
   `PASS_BANKED_OPTION_C_CUDA_TIMING`; keep output quarantined pending N1.
3. Rotate contract, source-dependency, assets-manifest, and kernel hashes.
4. Verify the staged Option-C assets contain the two timing records and all
   pinned dependencies. Run focused tests after the zero-swap preflight.
5. Emit exact human dataset-version, registration-check, kernel-push, status,
   and output-download commands. Do not execute them.

## Output

Write `../h1_cache_prep.json`, update `../h1_execution.md` and
`../LIVE_STATE.md`, and have the root coordinator append exactly one
`H1-ZWRFEAT-CACHE-PREP` row to `../LEDGER.md`.

## Verdict & cascade

- **HUMAN-DECISION** when every analytical gate passes: halt at the Kaggle
  dataset-version/kernel-push boundary. After the human returns a completed
  full-cache output, dispatch `N1_cache_attest_tolerance.md`.
- PARK if any identity, projection, environment, or coverage pin fails.
- ESCALATE if full-cache authorization would require widening scope.
