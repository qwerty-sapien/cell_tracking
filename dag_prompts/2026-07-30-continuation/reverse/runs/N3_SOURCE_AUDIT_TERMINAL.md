# MCC contour-return witness — N3 source audit terminal receipt

Date: 2026-07-30 (Asia/Singapore)
Verdict: **TERMINAL BEFORE KAGGLE COMPUTE — no deterministic bounded pilot exists**

The owner authorized the bounded, public-source-only MCC pilot with “Reverse:
go ahead.” This receipt records the source/rights audit performed before any
GPU setup, data download, asset upload, kernel creation, or execution.

## Authenticated public provenance

| Object | Result |
|---|---|
| Source | `https://github.com/JunbongJang/contour-tracking.git` at `1219b669cc8f66b84f4edf7a23eaa7bb42ff9041` |
| Code license | Apache-2.0; `LICENSE` SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| Linked data record | Zenodo `7754589`, metadata license `cc-by-4.0` |
| Data manifest exposed by record | `Phase Contrast.zip` (6,410,053,381 B, `md5:6553193bb45b700c8eef77f44a292d4c`); `Confocal Fluorescence.zip` (6,404,743,572 B, `md5:fdd5214064f9d9332870f81574f57ed02`); `Jellyfish.zip` (28,533,970 B, `md5:78c9b3d5b6b93c7e3d22ad7589f24353`) |

The Zenodo metadata was read directly from its public API. No data file was
downloaded, mounted, copied, or supplied to Kaggle. No campaign object was
read or named by this audit.

## Why the prescribed 2-hour witness cannot be executed faithfully

The pinned repository releases source only—no MCC checkpoint or deterministic
inference artifact. Its own README requires: (1) manually or pseudo-labeled
ordered contour points, (2) author-format preprocessing into images,
contours, and tracked contour-index labels, (3) TFRecord conversion, and
(4) training. Its declared working environment is Python 3.6.9, TensorFlow
2.4.3, CUDA 11.0, Ubuntu 18.04. This is not a self-contained 200-frame
inference witness; it is a new source-training project.

Consequently there is no compliant way to obtain a deterministic contour output
within the preregistered one-T4, <=2 GPU-hour ceiling. Substituting a model,
using campaign masks, omitting the ordered-contour requirement, or expanding
the budget would violate N2. The stop occurs before the protocol’s data/model
execution step and consumes **0 GPU-h**.

## Terminal scope

This exact source/record/slice/budget configuration is terminal with no retry.
It neither changes nor reopens authentic Reverse Trackastra, and it yields no
campaign feature, association, graph action, or score. A future contour route
would need a newly named mechanism with a released immutable checkpoint (or a
separately human-approved training budget and rights/input contract) and a new
N0–N3 review.
