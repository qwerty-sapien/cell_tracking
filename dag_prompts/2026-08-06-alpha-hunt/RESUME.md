# RESUME POINT — paused 2026-08-06 by owner

**Everything is paused awaiting owner resume. No Kaggle action was taken. Nothing was
submitted, pushed, or launched. No pipeline code was modified.**

## State at pause

| node | status | artifact |
|---|---|---|
| N0 submission transport forensics | ✅ **complete, independently verified** | `runs/N0_SUBMISSION_FORENSICS.md` |
| N1 division alpha | ⏸️ **stopped mid-run, no output written** | — |
| N2 clean port plan | ⏸️ **stopped mid-run, no output written** | — |
| N3 transfer calibration | not started (blocked on N0=GO) | — |
| N4 kernel package + submit | not started (**HUMAN-GATE**) | — |

N1 was killed shortly after start (it had also died once on an API error). N2 was killed
while replaying our V32 patches against the support-pack source to verify port anchors.
Neither wrote a deliverable, so both must be **restarted from scratch** — their node
contracts in `N1_DIVISION_ALPHA.md` / `N2_CLEAN_PORT_PLAN.md` are complete and sufficient
to relaunch as-is.

## Files changed this session

- `reports/g02/screening/public_solution_landscape_survey_v1.md` (new, previous turn)
- `reports/g02/screening/LIVE_STATE.md` (modified — 2 log entries + Key numbers + one
  correction to the 2026-08-06 survey entry)
- `dag_prompts/2026-08-06-alpha-hunt/` (new — DAG.md, 5 node files, this file,
  `runs/N0_SUBMISSION_FORENSICS.md`)

Nothing is staged or committed. `LEDGER.md` was **not** written (root coordinator is the
sole writer).

## The one finding that matters

Verified root cause of the six unscored submissions:
`kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:100,182-184` pins
`EXPECTED_TEST_COUNT = 4` and raises at import when the hidden inventory is not exactly
four datasets. Kaggle's own `error_description` for ref `54890576`: *"the hidden dataset can
be larger/smaller/different than the public dataset"*, `total_bytes=0`.

Our verified public standing is **0.881, rank 1111/2019**. Local CV57 `0.914358` has never
been publicly confirmed — it was never tested, not disproven.

## ⚠️ Standing warning that survives this pause

**Do not launch `scripts/v32_submission_bisect.py`'s eight-stage ladder.** It templates the
identical `EXPECTED_TEST_COUNT` gate into every stage (`:115,163-164`), so under the N0 root
cause all stages fail identically and burn up to seven submission slots. This is recorded in
`LIVE_STATE.md` for Codex.

## To resume

1. Relaunch N1 and N2 (contracts are ready; both need a fresh start).
2. Owner decisions still open:
   - apply the N0 minimal fix to `scripts/public_temporalunet_baseline.py`? (rerun contract
     only — weights, thresholds, graph policy, CV57 untouched)
   - pull `errorDescription` / `totalBytes` for `55272130` from the raw REST
     `competitions/submissions/list` endpoint, which would reveal the true hidden test count
   - whether four artifact-less refs (`54717551, 54893629, 55212065, 55251996`) are
     auto-submissions silently spending daily slots
3. N4 remains a HUMAN-GATE and requires fresh explicit approval against a specific plan.
