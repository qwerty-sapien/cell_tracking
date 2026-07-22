# Phase 1 E4/E5 — exact CV57 appearance and division morphology

Date: 2026-07-22

Pre-edit checkpoint: `d6c5ace`

Machine-readable report: `reports/g02/phase1_e4_e5_holdout57_v1.json`

## Outcome

- **E4 appearance separability: PASS.** Three-dimensional normalized
  cross-correlation beats physical distance on the predeclared 2–4-candidate
  cohort, and its paired video-cluster bootstrap interval excludes zero.
- **E5 division morphology separability: PASS.** The fixed local-moment feature
  set clears the predeclared video-grouped held-out AUC and balanced-accuracy
  thresholds.
- These are falsification results, not production changes. Divisions remain
  off; no tracking geometry, segmenter, mask, production model, or submission
  artifact was changed.

## Verified scope and reconciliation

The script independently reconstructed the exact divisions-off baseline from
the 57 active fragments:

| Quantity | Count |
|---|---:|
| Official edge TP | 34,174 |
| Official edge FP | 2,736 |
| Official edge FN | 2,094 |
| Raw-FrameDet-eligible FN | 1,549 |
| Raw distance <= 10 um | 1,232 |
| Raw distance > 10 um | 317 |
| Official-unmatched FN endpoint | 464 |
| Officially matched FN with a synthetic/pruned raw endpoint | 81 |

The initial final-CSV audit had produced 1,187 in-gate and 443 above-gate FNs.
That was a coordinate-stage mismatch, not a scorer mismatch. Among the 1,549
common raw-eligible FNs, raw-to-final transitions were 1,063 in-to-in, 169
in-to-above, 52 above-to-in, and 265 above-to-above. The 81 raw-absent cases
contributed another 72 final-in and 9 final-above cases. E4 therefore uses raw
FrameDet centers and candidates, exactly matching the association-gate stage;
final topology is used only to label TP versus FN.

The newly derived exact in-gate FN candidate-count distribution is 108 with one
candidate, 354 with two, 327 with three, 226 with four, 122 with five, 63 with
six, 19 with seven, 9 with eight, and 4 with nine. Thus the requested 2–4
confusable FN cohort contains 907 events.

## E4 — appearance separability

Patches are 16x32x32 voxels centered on raw matched detections at `t` and raw
next-frame candidates. Full-patch NCC uses separately zero-mean, unit-norm
patches. The lateral test uses an axial-max 32x32 projection. PCA uses either
4x8x8 block means or 16x16 lateral block means, 16 components, and five-fold
video-grouped fitting; each PCA mean and basis is fit only on training-video
patches.

Primary decision rule: PASS only if a predeclared appearance ranker beats
distance over all 2–4-candidate confusable events and the paired video-cluster
bootstrap 95% interval for the improvement excludes zero.

| Cohort / ranker | Correct / N | Top-1 accuracy | 95% video-cluster CI | Delta vs distance | Delta 95% CI |
|---|---:|---:|---:|---:|---:|
| All 2–4: distance | 13,815 / 14,759 | 0.93604 | 0.92254–0.94795 | — | — |
| All 2–4: 3D NCC | 14,020 / 14,759 | 0.94993 | 0.93823–0.95971 | +0.01389 | +0.00581–+0.02439 |
| All 2–4: lateral NCC | 13,479 / 14,759 | 0.91327 | 0.90096–0.92497 | -0.02277 | -0.03214–-0.01092 |
| All 2–4: 3D PCA | 13,891 / 14,759 | 0.94119 | 0.92920–0.95134 | +0.00515 | -0.00358–+0.01634 |
| All 2–4: lateral PCA | 13,470 / 14,759 | 0.91266 | 0.90031–0.92433 | -0.02338 | -0.03282–-0.01096 |
| FN 2–4: distance | 263 / 907 | 0.28997 | 0.25694–0.32157 | — | — |
| FN 2–4: 3D NCC | 504 / 907 | 0.55568 | 0.47995–0.63611 | +0.26571 | +0.17896–+0.35826 |
| FN 2–4: lateral NCC | 487 / 907 | 0.53693 | 0.46246–0.61225 | +0.24697 | +0.16409–+0.33542 |
| All in-gate FN: distance | 400 / 1,232 | 0.32468 | 0.29032–0.36713 | — | — |
| All in-gate FN: 3D NCC | 690 / 1,232 | 0.56006 | 0.48014–0.64996 | +0.23539 | +0.15629–+0.32577 |

Only 3D NCC passes the primary all-edge rule. The lateral-only representations
are informative on the error cohort, but they regress on the full confusable
population. The leakage-safe 3D PCA descriptor is directionally positive on the
full cohort but its improvement interval crosses zero.

## E5 — division morphology separability

The E5 cohort contains all 42 GT division parents in the exact fold-0 selection
and 168 controls. Controls are one-child adjacent nodes selected without
replacement in the same video, up to four per division, nearest in time. All 26
videos containing a division are held out as whole groups in five-fold CV.

Moments are measured without masks in a fixed 4.5 um sphere, with a 4.5–6.0 um
shell for local background. Features include raw and background-excess
intensity sums, raw and relative intensity variance, physical inertia-eigenvalue
ratios, anisotropy, and their `t` to `t+1` ratios/deltas. The future sphere is
centered at the mean successor centroid (two daughters for a division, one
successor for a control). The classifier is a fixed L2-ridge logistic regression
with class-balanced training weights; scaling and fitting occur inside each
training fold.

Decision rule: PASS only if grouped out-of-fold AUC is at least 0.70 with lower
95% video-cluster bound above 0.60, and balanced accuracy is at least 0.60 with
lower bound above 0.50.

| Metric | Result | 95% video-cluster CI |
|---|---:|---:|
| Out-of-fold ROC-AUC | 0.94019 | 0.89753–0.97969 |
| Out-of-fold balanced accuracy | 0.87798 | 0.81784–0.93314 |

Per-fold held-out AUCs are 0.99383, 1.00000, 0.93359, 0.91016, and 0.99609.
Each fold contains 8–9 divisions and 32–36 controls.

Strongest orientation-free univariate effects:

| Feature | AUC |
|---|---:|
| Background-excess intensity ratio, `t+1 / t` | 0.90377 |
| Raw intensity-sum ratio, `t+1 / t` | 0.89909 |
| Inertia middle/maximum eigenvalue delta | 0.84340 |
| Anisotropy delta | 0.82696 |
| Inertia minimum/maximum eigenvalue delta | 0.80740 |

## Interpretation boundary

Verified: both stated falsification gates pass on held-out videos with
video-level leakage controls and cluster-bootstrap uncertainty.

Inferred: E4 supports appearance-aware edge costs, especially on current
in-gate FNs; E5 supports building morphology features into the existing
division scorer. Neither result establishes official tracking score gain. Per
the project constraint, divisions stay off until a separate patched-metric test
demonstrates a real division TP.
