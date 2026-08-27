# E4 — appearance separability on the exact CV57 holdout

Date: 2026-07-23

Fresh machine-readable rerun: `reports/g02/phase1_e4_e5_holdout57_rerun_20260723_v1.json`

## Decision

**PASS — P4 is justified.** Appearance beats physical distance on the scored
2–4-candidate decisions. Full 3-D normalized cross-correlation improves top-1
accuracy by 1.389 percentage points over distance on all confusable events and
by 26.571 points on the FN-only confusable subset.

The user-specified rule requires any positive margin. The rerun also clears the
stricter predeclared check for 3-D NCC: its paired video-cluster bootstrap 95%
interval for the all-event improvement is entirely positive
(+0.581 to +2.439 percentage points).

## Exact scope

- 57 fold-0 holdout videos, divisions off.
- Reconstructed official edge counts: 34,174 TP, 2,736 FP, and 2,094 FN.
- Patches: 16x32x32 voxels at raw GT-matched FrameDet centres at `t` and `t+1`.
- Candidate set: raw next-frame detections within the existing 10 um gate.
- NCC: zero-mean, unit-norm full patch; lateral NCC uses the axial-max 32x32
  projection.
- PCA: 16 components over normalized block-mean descriptors, with five-fold
  video grouping and the unsupervised basis fit only on the other videos. No
  supervised appearance model was trained.
- CPU only; no production geometry, graph, model, submission, or Kaggle state
  was changed.

## Top-1 results

| Cohort / ranker | Correct / N | Top-1 | Delta vs distance | Paired delta 95% CI |
|---|---:|---:|---:|---:|
| All 2–4: distance | 13,815 / 14,759 | 93.604% | — | — |
| All 2–4: 3-D NCC | 14,020 / 14,759 | 94.993% | +1.389 pp | +0.581 to +2.439 pp |
| All 2–4: lateral NCC | 13,479 / 14,759 | 91.327% | -2.277 pp | -3.214 to -1.092 pp |
| All 2–4: 3-D PCA | 13,891 / 14,759 | 94.119% | +0.515 pp | -0.358 to +1.634 pp |
| All 2–4: lateral PCA | 13,470 / 14,759 | 91.266% | -2.338 pp | -3.282 to -1.096 pp |
| FN 2–4: distance | 263 / 907 | 28.997% | — | — |
| FN 2–4: 3-D NCC | 504 / 907 | 55.568% | +26.571 pp | +17.896 to +35.826 pp |
| FN 2–4: lateral NCC | 487 / 907 | 53.693% | +24.697 pp | +16.409 to +33.542 pp |
| FN 2–4: 3-D PCA | 487 / 907 | 53.693% | +24.697 pp | +16.057 to +34.258 pp |
| FN 2–4: lateral PCA | 475 / 907 | 52.370% | +23.374 pp | +14.990 to +32.453 pp |

## All 2,094 FNs

Top-1 ranking is defined for 1,232 FNs whose raw matched endpoints exist and
whose true successor is inside the 10 um gate. The remaining 862 comprise 317
above-gate FNs, 464 with an official-unmatched endpoint, and 81 with a
synthetic/pruned raw endpoint. Treating every one of those 862 unrankable cases
as incorrect gives the strict full-FN accounting below.

| Ranker | Rankable in-gate correct / 1,232 | Strict correct / 2,094 |
|---|---:|---:|
| Distance | 400 (32.468%) | 400 (19.102%) |
| 3-D NCC | 690 (56.006%) | 690 (32.951%) |
| Lateral NCC | 670 (54.383%) | 670 (31.996%) |
| 3-D PCA | 674 (54.708%) | 674 (32.187%) |
| Lateral PCA | 657 (53.328%) | 657 (31.375%) |

This full-denominator view does not change the comparison: 3-D NCC finds 290
more true successors than distance among the currently rankable FNs.

## Anisotropy conclusion

The discriminative power is **not all lateral**. Lateral NCC carries strong
signal on the FN subset, but it is 1.875 points below full 3-D NCC there and it
regresses by 2.277 points versus distance on the complete confusable cohort.
Full axial context is required for the only robust all-event improvement.

## Reproducibility check

The fresh report is byte-identical to the frozen prior result. Both files have
SHA-256 `e2ec483c6338e68051af38456d755f46499c642c487f127e2caba2714c1ae7e7`.
