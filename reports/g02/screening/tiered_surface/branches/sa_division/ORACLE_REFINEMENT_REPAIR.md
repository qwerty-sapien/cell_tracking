# SA-DIVISION oracle refinement authentication repair

Frozen: 2026-07-28, after failed attempt 1 and before retry 1.

## Observed failure

Attempt 1 stopped in `authenticate()` before P0 was parsed. The script's
expected P0 SHA-256 contained a duplicated `a`:

`8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5f18c8889c06`

The observed digest was:

`8ad548c134c419e2e76b8661848e90493d674731ceb34ab231c5f18c8889c06`

The observed digest is identical to the P0 pin already authenticated by the
banked SA-DIVISION fixture audit and `EVIDENCE_INVENTORY.md`. Attempt 1 recorded
zero swap growth, wrote no refinement result, and opened no fit or scientific
outcome.

## Authorized repair

Correct only the one-character expected-digest typo in
`division_oracle_refinement.py` and the displayed table in
`ORACLE_REFINEMENT_PREREGISTRATION.md`. Do not change paths, algorithms,
thresholds, expected self-test values, conclusions, or execution permissions.

Retry only after:

1. committing this repair;
2. running a fresh 45-second zero-swap preflight;
3. retaining the failed guard artifact;
4. writing the successful retry to new output paths.

The retry remains diagnostic and fail-closed. Division fitting remains
unauthorized.
