# N0 — Submission transport forensics (ROOT GATE)

> Load `.agents/skills/seeking-alpha/SKILL.md` (Phase 1), `DAG.md`,
> `reports/g02/screening/p0_parity.md`, `reports/g02/screening/LIVE_STATE.md` (log entries
> 2026-08-03, ~line 2553). READ-ONLY: Kaggle read commands only, no push/submit/launch.

## Objective / gate

**Why do 6 of our 13 submissions return `COMPLETE` with an empty public *and* private score?**

This is the Phase-1 "make the objective computable" gate. Until it passes, no leaderboard
number means anything and no local gain can be validated in the units that decide the
competition.

| | refs |
|---|---|
| **Unscored** | 55272130, 55251996, 55212065, 55201829, 54893629, 54890576, 54717551 |
| **Scored** | 55221497 (0.881), 55203439 (0.881), 54942864 (0.881), 54923074 (0.878), 54921344 (0.860), 54910497 (0.877) |

`55272130` is the flagship: *"V32 unsharded replication proof 0.914358 CV57"*, 2026-08-05
14:02:35 — unscored. Our LB entry (`onebitbrain`, rank 1111) still reads **0.881** with
`LastSubmissionDate` equal to that submission's timestamp.

Prior evidence (LIVE_STATE, 2026-08-03): ref `55201829` returned *"COMPLETE with a blank
score, `total_bytes=0`, and the generic notebook-output incorrect-format diagnostic. The
downloaded output remains byte-identical and organizer graph parsing passes, so transport
versus format is under diagnosis; no blind retry."*

**Preregistered GO threshold:** a named root cause supported by evidence that separates
**all** scored refs from **all** unscored refs — not a plausible story that fits some of
them — together with a concrete minimal fix. Anything less is PARK.

## Tasks

1. Recover every artifact tied to each ref (grep the repo for the ref numbers). Start at
   `scripts/kaggle_submission_lifecycle.py`, `scripts/cmux_ops.py`,
   `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/`,
   `submission_result/`.
2. Classify each ref's **transport**: direct CSV upload vs code-competition notebook output.
   Leading hypothesis: this is a *code* competition and the two transports are not
   equivalent. `total_bytes=0` points at transport, not model.
3. Diff the submission artifacts themselves: column order, header, `id` index column,
   dtypes, `-1` sentinel int-vs-float, row_type vocabulary, node/edge counts, byte size.
4. Establish the organizer's expected format from `external/kaggle-cell-tracking-competition/`
   and any `sample_submission` under `data/kaggle/`.
5. State the minimal change that makes the next submission score.

## Output

`runs/N0_SUBMISSION_FORENSICS.md` — scored-vs-unscored table, ranked root-cause hypotheses
with confidence, the discriminating evidence, the minimal fix. A LEDGER row.

## Verdict & cascade

- **GO** (root cause separates all 7 scored from all 6 unscored + minimal fix stated) →
  proceed directly to `N3_TRANSFER_CALIBRATION.md`.
- **PARK** (hypothesis fits only some refs, or the deciding artifact is missing) → halt;
  report exactly which artifact is missing and the read-only command that would produce it.
- **ESCALATE** (the failure is on Kaggle's side / needs a host forum post) → halt; hand to
  owner.
