# Organizer checkpoint reconciliation v1

Status: **PARK; fit monitor is structurally optimistic**

The repository contains a valid fit-only organizer
`TemporalUNet3D + SimpleNodeTransformer` checkpoint:

- seven completed training epochs over fit117;
- best checkpoint at epoch 5;
- best fit-monitor product `edge_accuracy * detection_recall = 0.907895`;
- checkpoint SHA-256
  `9549a2e3425872473b9541fb69e0ca62c350ddcac338a2c513e115e677a3865f`;
- no holdout, audit, test, or public-score exposure.

This number is **not** adjusted edge Jaccard.  It is a training-monitor proxy
computed on 12 videos that are also members of fit117.  The training run is
resumable rather than converged: epoch 7 of a minimum 300-epoch convergence
contract.

## Monitor audit

A CPU-only audit ran 12 deterministic frame-pair windows from the locally
available fit117 member `train:6bba_d3da753b`.  It reused the exact epoch-5
checkpoint and official organizer code, without opening holdout, audit, or test
data.

The monitor's official sparse mask produced:

- accuracy `0.998959`;
- all-negative accuracy `0.997627`;
- positive-edge precision/recall/F1 `0.8161 / 0.7245 / 0.7676`.

That mask ignores predictions between detected nodes when neither row nor column
contains a matched GT edge.  Over the complete detected-node matrices, the same
windows had `71 TP / 2,000 FP / 27 FN`, precision `0.0343`, and F1 `0.0655`.

An inference-like, no-TTA replay used the packaged `0.99` sigmoid detection
threshold, 3 um pooling, source-softmax edge threshold `0.5`, and greedy
two-child/one-parent limits.  It emitted 1,816 edges for 100 GT edges:
`72 TP / 1,744 FP / 28 FN`, precision `0.0396`, recall `0.72`, and F1 `0.0752`.
This is not scorer-comparable and omits detection TTA, but it falsifies the claim
that the `0.907895` product is credible promotion evidence.

## Compute evidence

- The existing local-MPS benchmark projects about 8.38 hours per full epoch at
  the stable batch size, so completing 300 epochs locally is infeasible.
- A bounded CPU end-to-end proxy on one fit video was terminated after at least
  900 seconds during official TemporalUNet3D inference.  It wrote no prediction
  and produced no score.
- The prior Kaggle stage was canceled after the account displayed only 8h36m
  remaining, preserving quota.  No new Kaggle or RunPod action occurred here.

## Gate interpretation

The branch satisfies:

- learned model class;
- classical local ceiling evidence (`0.874290`, two-arm oracle `0.876831`);
- association-bound E1 evidence;
- fit-only checkpoint and split provenance.

It does **not** satisfy the required comparable `>= +0.005` CPU/proxy scorer
gain.  More importantly, the positive-edge audit shows that the high monitor
product is driven by sparse-matrix accuracy and an evaluation mask that excludes
a large false-positive population.  The current checkpoint and training
objective are therefore **PARK**, not candidates for more local MPS or external
GPU work.  Kaggle GPU, RunPod, full CV57 inference, packaging, and submission
remain locked.

A successor would need an explicit no-link/background objective and a
scorer-aligned validation metric before accelerated training is reconsidered.

Artifacts:

- `reports/g02/organizer_unet_transformer_f0/stage00_v3/organizer_f0/run.json`
- `reports/g02/organizer_unet_transformer_f0/stage00_v3/organizer_f0/history.json`
- `reports/g02/organizer_unet_transformer_f0/local_mps_feasibility.json`
- `reports/g02/organizer_f0_fit_proxy_cpu_timeout_v1.json`
- `reports/g02/organizer_fit_monitor_audit_v1.json`
