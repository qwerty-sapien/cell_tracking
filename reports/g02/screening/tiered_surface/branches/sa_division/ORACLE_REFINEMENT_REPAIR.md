# SA-DIVISION oracle refinement authentication repair — superseded

Frozen: 2026-07-28, after failed attempt 1 and before retry 1.

Status: **SUPERSEDED by `ORACLE_REFINEMENT_REPAIR_V2.md`.**

## Observed failure

Attempt 1 stopped in `authenticate()` before P0 was parsed. The script's
expected P0 SHA-256 contained a duplicated `a`:

`8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5f18c8889c06`

This note incorrectly diagnosed the observed digest as:

`8ad548c134c419e2e76b8661848e90493d674731ceb34ab231c5f18c8889c06`

The actual banked digest is
`8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06`.
The original mismatch was `f` versus `c` at zero-based index 53, not a
duplicated `a`. Attempt 1 recorded zero swap growth, wrote no refinement result,
and opened no fit or scientific outcome.

## Authorized repair

This repair is not authorization for another retry. Its mistaken change was
preserved in git and failed closed. Only the V2 repair governs the next attempt.

Retry only after:

1. committing this repair;
2. running a fresh 45-second zero-swap preflight;
3. retaining the failed guard artifact;
4. writing the successful retry to new output paths.

The retry remains diagnostic and fail-closed. Division fitting remains
unauthorized.
