# G-03 R1 zero-shot preregistration

Status: **SEALED BEFORE ANY R1 HELD-FIT SCORE OR GROUND TRUTH READ**  
Date: 2026-08-01 (Asia/Singapore)  
Population: the exact 36 fit-train `44b6` videos listed in the machine contract  
Repeats/alternatives: one; no post-result tuning or replacement

## Question and decision rule

Measure the official 44b6 held-fit graph objective produced by authentic,
frozen pretrained Trackastra when its required 12-dimensional `wrfeat` is
populated by one deterministic, untrained, image-conditioned crop adapter.
The candidate is Branch A iff its combined official score is at least
`0.868299238911` (`0.881522069960 * 0.985`) or it is a new best; otherwise it
is Branch B. The same-population APNCC score `0.866534526036` is a descriptive
reference only and never replaces the stated best-known reference.

## Frozen estimator

- Raw input: authenticated competition archive, exact fit-train 44b6 members.
- Detections/coordinates/IDs: authenticated fit105 BH9 sparse caches.
- Crop: intensity-only `(channels=3, z=16, y=32, x=32)`, channels
  `[t-1,t,t+1]`, endpoint temporal clamp, nearest-integer center, replicate
  spatial padding, float32.
- Normalization: per-crop global mean subtraction, global RMS division with
  floor `1e-6`, then clamp to `[-5,5]`.
- Adapter seed `20260801`: `AvgPool3d(4,8,8) -> Conv3d(3,8,k3,p1) -> exact
  GELU -> global average -> Linear(8,12) -> LayerNorm(no affine)`. It has 764
  parameters, all frozen at initialization, and zero training steps.
- Trackastra: authentic frozen CTC checkpoint/config, `feat_dim=12`, window 4,
  CPU deterministic inference, at most 1024 tokens.
- Candidate bank: adjacent-frame target neighbors within 12 micrometres, at
  most 16 per source. Every returned probability, distance and node coordinate
  is retained in a hash-sealed association bank before scoring.
- Graph compiler: restrict to 8 micrometres and the nearest 8 targets per
  source, probability threshold `0.35`, global descending greedy one-in/one-out,
  divisions disabled. Physical scale is `(1.625,0.40625,0.40625)` micrometres.
- Scorer: the hash-pinned official local wrapper and metric sources in the
  machine contract. Required output is combined score, adjusted and raw edge
  Jaccard, node recall, edge TP/FP/FN, and division TP/FP/FN/Jaccard.

## Integrity and stopping rule

The preflight may read archive metadata, raw images, caches, model weights,
code and scorer parity receipts, but not GEFF/ground truth or a held-fit score.
Inference must emit all 36 fragments and their complete probability banks,
then seal their hashes. Only the separate score command may open ground truth.
After the first official candidate score is exposed, the result is final for
R1 even if poor or surprising: no retry, alternate seed, threshold sweep,
normalization change or compiler change is permitted. CV57, Kaggle submission,
GPU use and training are forbidden for this measurement.

## Null and validity checks

Null expectation: random untrained appearance features may not transfer and
may underperform APNCC. That outcome is valid Branch-B evidence, not a reason
to rerun. The measurement fails closed if any hash drifts, any video is
missing, the adapter has a zero row or no image-conditioned channel variance,
the model/config changes, ground truth is touched before the inference seal,
or the 3-hour CPU wall is exhausted before all 36 identities are sealed.
