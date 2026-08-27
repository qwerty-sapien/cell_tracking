# Git path allowlist audit — SA-DIVISION-DEEP

Branch: `codex/sa-division-deep-wave2`
Base: `5a805c2`
Allowed prefix:
`reports/g02/screening/tiered_surface/branches/sa_division_deep/`

Result: **PASS**.

Every path added by the lane is below the allowed prefix. No path under
`LIVE_STATE.md`, `LEDGER.md`, `tier_census.json`, `kaggle_budget.json`, CELLECT,
another lane, source code, tests, folds, N1, P0, or a shared cache is modified.
The frozen coordinator root was read only. Python bytecode caches were moved
out of the worktree before staging.

The root coordinator must repeat the allowlist assertion against the exact
terminal commit before barrier merge.
