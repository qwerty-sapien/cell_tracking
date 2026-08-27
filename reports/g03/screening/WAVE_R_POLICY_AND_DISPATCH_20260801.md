# G-03 Wave R — owner allocation policy and parallel dispatch

Date: 2026-08-01  
Status: **R1 active; R2–R4 complete; GPU allocation blocked on R1**

## Binding owner amendment

The owner authorizes read-only research/diagnostics, CPU experiments and sweeps,
preregistrations, tier moves, Kaggle GPU work within the weekly cap, daily probe
submissions, and downstream execution without another approval.  RunPod remains
forbidden.  Each kernel retains a hard wall-clock kill, the weekly cumulative cap and
shared serial lock remain binding, and no uncited number may set a decision.  Only the
final-slot selection returns to the owner.

This amendment supersedes the earlier G-03 Phase-2 preparation hold once the new
crop/adapter/cost/output/checkpoint contract and allocation gates are hash-pinned.  It
does not retroactively turn G-02 into authority or reopen any named DEAD mechanism.

## Allocation policy

The banked best-known score is `0.881522069960`, sourced from
`../../g02/screening/TERMINAL_CAMPAIGN_REPORT.md`.  The exact current floor is therefore:

`neighborhood_floor = 0.881522069960 * 0.985 = 0.868299238910600`.

The owner's `~0.8678` parenthetical is treated as a loose approximation; the exact
formula controls.

- **Branch A:** score `>=0.868299238910600`, or new best.  Eligible to compete for the
  30 T4-h allocation only if further training has a feasible outsized-impact case.
- **Branch B:** score `<0.868299238910600`.  Ineligible for the 30 T4-h allocation until an
  audited primary-literature dossier supports both comparability within `<=30 T4-h` and
  a realistic path through `0.915` to `0.945` with further runway.
- Development may continue on CPU in either branch.  Eligibility is not an instruction
  to spend; highest-ranked eligible evidence receives the single allocation.

As `best_known_score` rises, both the reference and exact `0.985` floor must be
recomputed before the next wave decision.

## Isolated Wave R dispatch

| lane | isolated worktree | deliverable | external spend |
|---|---|---|---:|
| R1 | `/private/tmp/g03_wave_r_r1_20260801` at `2241c72…` | one sealed zero-shot real-`wrfeat` 44b6 graph measurement; no tuning after score read | 0 T4-h |
| R2 | `/private/tmp/g03_wave_r_r2_20260801` at `c493a67…` | primary-source Branch-B literature/cost/headroom dossier | 0 T4-h |
| R3 | `/private/tmp/g03_wave_r_r3_20260801` at `c493a67…` | path/hash/interface/staleness inventory of reusable G-02 assets | 0 T4-h |
| R4 | `/private/tmp/g03_wave_r_r4_20260801` at `c493a67…` | finite division-decode lattice and nested/grouped-OOF plus sealed one-shot protocol | 0 T4-h |

R1 fixes one adapter seed, architecture, crop normalization/padding, checkpoint,
compiler, and scorer identity before opening the 44b6 score.  It may not train, inspect
CV57, or choose among alternatives.  R4 may design and later run CPU candidate-bank
work, but sealed confirmation remains inaccessible until nested OOF has frozen one
configuration.

## Wave-R reporting frame

Until R1 returns, the only truthful model classification is **PENDING**.  Current
accounting is: candidate score `PENDING`; floor `0.868299238910600`; branch `PENDING`;
Wave-R GPU consumed `0`; latest previously authenticated global quota
`1.01/28.99 T4-h used/remaining` with refresh `2026-08-08 00:00 UTC`; division
TP/FP/FN/Jaccard `NOT YET EVALUATED`; updated best `0.881522069960`.

No GPU, Kaggle mutation, submission, CV57 access, final-slot action, or RunPod action
occurred in this dispatch.
