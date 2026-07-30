# Pointer — Kaggle reliability + stop-loss doctrine now live on `main`

**This branch forked before `reports/g02/screening/LIVE_STATE.md` existed on `main`, so it
cannot take a clean merge.** Read this before resuming work on this branch, and
specifically before any further Kaggle kernel push or repeat-compute request.

As of `main@955e8ea` (2026-07-30), two standing process documents apply to every
lane/branch in the G-02 campaign, including this one:

1. **`reports/g02/screening/MARGINAL_VALUE_GATE.md`** (SHA-256
   `c2a3e8ad8d047ef41d86c11a3b9ab552d2c41c70004eab50668df4010c25af95`) — a mandatory
   check before requesting a second or later round of compute on any direction: state
   the *realistic* (precedent-adjusted, not best-case) ceiling, compare it to the
   actionability floor, and STOP at the mechanism-*class* level when the limiting factor
   is structural. Worked example: `dag_prompts/2026-07-30-recall-and-division/recall/runs/N0B_RECONCILIATION.md`
   (SHA-256 `eae05d62ce1329cbd635bafaf24235e6821d72586a84ac3ffd1a760baa195915`).

2. **`reports/g02/screening/tiered_surface/kaggle_prelaunch/RELIABILITY_RUNBOOK.md`**
   (SHA-256 `3a8112f5371af327134e4cbabdb303434975c5604201f1153a8edcfe2053b1fb`) — every
   Kaggle failure this week was 100% infrastructure (never reached scientific code). The
   fix already exists (`kaggle_prelaunch_guard.py`, 18 root-cause families, 14+2 passing
   regression fixtures) but was never structurally enforced — no hook, no wrapper, zero
   external call sites — and its `dataset_ownership` registry had gone stale. **No
   kernel push from this branch is valid without a fresh, logged guard receipt.** Run
   `python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py`
   and the guard itself (see the runbook for the exact command) before any push. Do not
   write a new bespoke per-lane preflight script — extend the shared module instead;
   that duplication is exactly what let already-solved failure modes recur.

Do not fork copies of these two files onto this branch — that would recreate the same
registry-staleness problem they fix. Read them from `main` (or cherry-pick this exact
commit if this branch is later rebased forward) rather than maintaining a local copy.

This pointer is informational only: no fitting, tier, gate, budget, or campaign-state
change is implied for this branch by its addition.
