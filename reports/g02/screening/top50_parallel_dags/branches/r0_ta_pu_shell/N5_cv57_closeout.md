# R0 N5 — frozen CV57 closeout

> Load `DAG.md`, GO-qualified N3/N4 outputs, the once-frozen R0 artifact, CV57 identity
> manifest, upstream no-shell candidate, exact APNCC, and the frozen scorer. No fitting,
> prior estimation, calibration, thresholding, or arm selection is permitted.

## Objective / gate

Does frozen R0 remain observation-distinct on CV57 with total adjusted delta vs APNCC
`>=+0.005`, incremental adjusted delta vs the frozen upstream no-shell arm
`>epsilon_null`, raw edge-Jaccard and node-recall deltas vs that arm each `>=0`, and all
admission caps satisfied?

## Tasks

1. Authenticate that all choices were frozen before CV57 opened.
2. Run one exact local CV57 reconstruction for APNCC, upstream no-shell, and R0. No
   retry, rerouting, recalibration, threshold change, or family-specific selection.
3. Report total/incremental adjusted/raw/node vectors, shell TP/FP/FN, per-video/group
   support, cap headroom, observation identity, and score relative to the `0.89`
   submission-worthiness floor and top-50 planning range.
4. Prepare but do not execute any submission or external mutation.

## Output

Write `outputs/r0_n5_cv57.{md,json}` plus proposed LEDGER row
`R0-TA-PU-SHELL-CV57 | CONFIRM | total >=+0.005, incremental >null, caps | exact score
vector | observation identity | 0 submissions/GPU/RunPod | verdict | reason`. Do not
edit LEDGER or LIVE_STATE.

## Verdict

- **HUMAN-DECISION** — all Objective gates pass; halt at the public-probe boundary and
  present evidence. This is not a claim that public `0.91` is assured.
- **PARK** — total or incremental gain is positive but below gate, or raw/node
  regresses without a cap breach.
- **DEAD** — incremental adjusted delta is `<=0` or an admission cap is breached.
- **ESCALATE** — leakage, identity, scorer, or reproducibility cannot be certified.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner whether to authorize one Kaggle public
  probe; never submit automatically.
- On PARK / DEAD / ESCALATE → halt and report; do not advance.

