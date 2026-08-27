# Trackastra 6bba one-video parity diagnostic v1

Finalized: `2026-07-25 17:03:36 +0800`

Status: **STOP — budget preserved; parity decision not measured**

## Outcome

The minimal private, internet-disabled T4 diagnostic and its local
launch/download/comparison tool are implemented. The authenticated preflight
passes, the original kernel source remains pinned at
`caf3e02e…613f8`, and seven focused tests pass.

No T4 `row_features` array was produced. Therefore:

- T4↔T4 exact repeatability is **not measured**.
- No M5↔T4 column can be classified as `BENIGN_FP` or `STRUCTURAL`.
- `ADOPT_TOLERANCE_CONTRACT` is **not recommended or enacted**.
- `REAL_BUG_STOP` is also not asserted: the diagnostic stopped before the
  evidence needed to distinguish floating-point drift from a bug.

The exact missing artifacts are:

- `trackastra_parity_t4_run1_row_features.npy`
- `trackastra_parity_t4_run2_row_features.npy`

## Fail-closed attempts and budget

Attempt 1 stopped after `15.080610 s`: the new guard incorrectly required exactly
one visible CUDA device. Kaggle's `NvidiaTeslaT4` shape can expose more than one
T4, and the pinned v3 guard correctly accepts one or more with device 0 named T4.
The guard is fixed in the final implementation.

Attempt 2 stopped after `215.429856 s`. Stage A run 1 completed in approximately
130 seconds, but the over-broad design attempted a second full Stage A and stopped
that subprocess after `64.755296 s`, before either Stage-B pass.

Total observed diagnostic runtime is `230.510466 s` =
`0.064030685 T4-h`. Only `129.489534 s` = `0.035969315 T4-h` remains under
the task-wide `0.1 T4-h` ceiling. That is less than one measured Stage-A pass,
before the two required Stage-B extractions, so no third compute attempt is
authorized in this task.

Both attempts remained one-video-only and label-free. No 44b6, holdout,
validation12/CV57, training, competition submission, or RunPod action occurred.
The fail-closed manifests and logs are preserved under
`reports/g02/trackastra_parity_diagnostic_t4_v1/`.

## Corrected minimal implementation

The final kernel now performs the feasible minimal design:

1. authenticate and build Stage A once for exactly
   `train:6bba_edf14583`;
2. run the identical pinned Stage-B enrichment twice in the same T4 kernel
   session;
3. persist both complete 49-D arrays before testing `numpy.array_equal`;
4. list, download, SHA-authenticate, and compare all 49 columns locally using
   the requested formulas and initial
   `rtol=1e-5, atol=1e-7, equal_nan=True` probe;
5. recommend—but never enact—per-column tolerances only if T4↔T4 is exact and
   every M5↔T4 column is benign.

Kernel:
`kaggle_kernel/trackastra-parity-diagnostic/trackastra_parity_kernel.py`,
SHA-256 `ac6195ff…de58a6`.

Launcher:
`scripts/trackastra_parity_diagnostic.py`,
SHA-256 `0c679b3a…e915d8b`.

The corrected implementation was not launched because doing so would exceed the
current task's compute ceiling. A fresh, explicitly approved parity-only budget
is required to obtain the two missing arrays and populate the 49-column
`DECISION`.
