# Archived organiser TemporalUNet3D + SimpleNodeTransformer

Status: **DEAD / ARCHIVED** on 2026-07-24 by explicit compute-governance decision.

Agents must not resume, package, evaluate, or launch this model unless the user explicitly
reopens the archived line and changes the governing compute decision.

## Reason

- The last validated checkpoint completed epoch 7 on the frozen fold-0 fit117 set.
- Measured Kaggle T4x2 epoch time was about 1.63 hours. Reaching epoch 300 projected
  approximately 478 additional T4 session-hours.
- Under the user's 6 GiB local MPS ceiling, the largest stable batch was 3 and one epoch
  projected to 8.38 hours. Epochs 8–300 projected to about 2,455 M5 GPU-hours.
- Both projections exceed the locked maximum of 30 cumulative Kaggle T4 GPU-hours per
  model for initial training.
- The official fit monitor was not comparable to the patched scorer. A fit-only
  inference-like audit of the best checkpoint measured 3.96% precision and F1 0.0752,
  providing no evidence that more training would clear the promotion gate.

## Preserved evidence

- Last checkpoint SHA-256:
  `26ea5682fbc4df7b68ebbd5608e0ee7af5eb5adbc9993345545a4ba8903f50c5`
- Best checkpoint SHA-256:
  `9549a2e3425872473b9541fb69e0ca62c350ddcac338a2c513e115e677a3865f`
- Training boundary: fit117 only; holdout, audit, and test unused.
- Holdout-57 was never evaluated. There is no organiser holdout score to compare with
  the `0.873150` baseline.
- Remote Kaggle versions `onebitbrain/org-f0-00` and `onebitbrain/org-f0-01` remain
  private; all continuation sessions were completed or cancelled.

The archive retains:

- `reports/`: checkpoints, logs, provenance, benchmarks, and feasibility records;
- `scripts/`: fit-only trainer, evaluator, Kaggle builders, and MPS benchmark;
- `tests/`: organiser-specific boundary and evaluator tests;
- `kaggle_kernel/`: all private kernel packages and relaunch variants.

## Governing compute rule

- Initial training: at most 30 cumulative Kaggle T4 GPU-hours per model.
- Hard baseline improvement gate: 3.5%.
- RunPod: not considered before an accepted score reaches 0.89; thereafter only for a
  higher-quality-GPU-dependent optimization with literature/theory-backed expected score
  improvement greater than 5%, a $10 total cap, and explicit approval for the exact job.
