# C1 N5 — frozen CV57 closeout

> Load `DAG.md`, GO-qualified N3/N4 outputs, the single frozen candidate artifact, the
> CV57 identity manifest, and the frozen scorer. No fitting or selection is permitted.

## Objective / gate

Does the once-frozen C1 candidate remain observation-distinct on CV57 and improve
adjusted edge Jaccard by `>=+0.005` vs exact APNCC, with raw edge Jaccard and node
recall deltas each `>=0`?

## Tasks

1. Authenticate that N3/N4 fixed every model and inference choice before opening CV57.
2. Run exactly one local CV57 reconstruction and exact frozen-score comparison against
   APNCC and C0. No retry, family router, threshold change, or arm selection.
3. Report adjusted/raw/node deltas, TP/FP/FN, per-family/video decomposition,
   observation identity, runtime, and the local score relative to the `0.89`
   submission-worthiness floor and `0.9255–0.9422` top-50 planning range.
4. Prepare but do not execute any submission command or outward mutation.

## Output

Write `outputs/c1_n5_cv57.{md,json}` plus proposed LEDGER row
`C1-TA-HOCT-LITE-CV57 | CONFIRM | >=+0.005 | exact score vector | observation identity
| 0 submissions/GPU/RunPod | verdict | reason`. Do not edit LEDGER; update LIVE_STATE
after the node verdict.

## Verdict

- **HUMAN-DECISION** — candidate is observation-distinct and meets adjusted
  `>=+0.005`, raw `>=0`, and node-recall `>=0`; halt at the public-probe permission
  boundary and present the evidence. This does not imply the top-50 target is reached.
- **PARK** — positive but `<+0.005`, or raw/node regresses; no public probe.
- **DEAD** — adjusted delta `<=0`; no public probe.
- **ESCALATE** — leakage, identity, scorer, or reproducibility cannot be certified.

## Cascade

- On **HUMAN-DECISION** → halt and ask the owner whether to authorize a single Kaggle
  public probe; never submit automatically.
- On PARK / DEAD / ESCALATE → halt and report; do not advance.
