# SA-NODETOPO-GATE terminal report

## Outcome

`BLOCKPEAK-RESTORE-V1` is terminal **KILL_44B6_STOP** at the unchanged
44b6 held-fit graph gate. The exact adjusted/raw/node deltas are
**`-0.000003436150368130164 / 0 / 0`** against the preregistered
`+0.005 / 0 / 0` floors. Raw edge Jaccard and node recall pass at equality;
adjusted edge Jaccard fails.

The candidate and control have identical aggregate edge counts:
TP/FP/FN **10,341 / 1,064 / 673**. The adjusted score changes despite identical
aggregate raw counts because the official adjusted metric is summarized from
per-video node-adjusted edge scores.

The exact result is
`reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/RESULT_44B6.json`
at SHA-256
`fb9202586242f506a7cd8437d780bc93ba1fcfe5de957e22de8d6a2e73f94834`.

## Population and replay

- Exactly **50/50** committed `restore_topology` actions were consumed.
- Exactly **24/24** affected-video batches compiled through frozen N1.
- All **36/36** frozen fit105 44b6 videos were scored; the 12 unaffected
  videos replayed the baseline.
- Every batch before/after/action-set hash matched the committed budget-50
  receipt in `FALSE_NODE_STRESS_CURVE.json`.
- The committed action population remained
  `6be1c77b4869c0a83eb3e540f13b16d71ba097b30bd808647d688bea0a5a5324`;
  N1 remained `2a72950c…f865`, grouped folds `ddb313cb…8769`, and P0
  `8ad548c1…9c06`.

The lane independently loaded the committed population and frozen root inputs.
It did not read or use the pre-existing untracked root graph-gate output.

## Decision

Propose **PARK-C for this exact `BLOCKPEAK-RESTORE-V1` instantiation only**.
`NODETOPO-O0` remains PARK-A as the broader positive-oracle topology surface;
`NODERESTORE-O0` remains PARK-B as the broader catalogued mechanism.
No family demotion and no DEAD action are proposed.

Because the 44b6 gate failed, the authentic universe does **not** unblock
`DET-F0`, `DET-F2`, `DET-F3`, or `NODERESTORE-O0`. 6bba and CV57 remained
closed.

## Resource and failure chronology

The terminal retry followed a fresh 45.120-second preflight at zero swap growth.
Its guard completed in **111.186 s**, with **0.667 GiB** peak process-tree RSS,
**34%** minimum system free memory, and **0 MiB** swap growth. It used CPU/local
only.

An earlier full scorer pass failed closed before emitting a result because JSON
serialization rejected undefined per-video division diagnostics. Its guard is
preserved at
`reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/local_run_guard.json`
at SHA-256
`942e5e97d4f616a5610d3e642fa4242c9975dd3d55922130c7808203180e2ad8`.
The serialization-only `NaN`→`null` repair was frozen in commit `108b18e`
before the retry; it changed no population, metric, aggregation, gate, or tier
semantics.

Terminal resource receipts:

- preflight:
  `idle_preflight_retry.json` @
  `41d71457d4f495ee9a4f6eaf2ce72329c6e345150f7ec2b2144be503d3c92586`
- guard:
  `local_run_guard_retry.json` @
  `002d672c4652222e57cf04a7cf970fb68a51874ffc3f718aa3607bb492916363`

## Governance

No GPU, Kaggle action, budget debit, RunPod action, submission, external action,
DEAD promotion, plan amendment, model fit, threshold selection, 6bba run, CV57
access, CELLECT write, root single-writer edit, sibling-live-output read, or
uncited tier move occurred.
