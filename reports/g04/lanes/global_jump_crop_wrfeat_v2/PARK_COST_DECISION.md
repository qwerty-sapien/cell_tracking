# Crop-level computed wrfeat — PARK-COST decision

Date: 2026-08-03

## Status

`PARK-COST` — not `PARK-EVIDENCE` and not `DEAD`.

The lane has three independent throughput results against the same computed
wrfeat interface:

| Realization | Measured/projected cost |
|---|---:|
| Whole-frame GPU segmentation | `195.59 T4-h` |
| Fixed-crop CPU segmentation | `766.2 CPU-h` |
| Recentered, enlarged, batched-crop GPU segmentation | `659.856 T4-h` |

Each realization is operationally infeasible against the authenticated
`27.24 h` remaining weekly GPU budget. The last bounded pilot proves that crop
geometry is no longer the blocker: recentered `32x64x64` crops with one
`48x96x96` fallback produced `914/916` valid masks, and the two independent
frames produced `673/673` valid masks with zero boundary or empty selections.

## Banked reusable work

- Real computed-object size distribution and the diagnosis of why
  `16x32x32` crops truncate masks.
- Intensity recentering capped at 4 um.
- `32x64x64` primary crop and single `48x96x96` fallback.
- Genuine per-frame CUDA batching with serial/batch-one label parity.
- Authenticated coverage and throughput receipts.

These artifacts remain reusable if Trackastra's wrfeat interface becomes
relevant again. The lane is parked for cost and base relevance, not for lack
of feature evidence.

## Fail-closed reopening rule

Do not attempt a fourth throughput configuration, batch-size variation, crop
tuning pass, or equivalent implementation tweak. Reopening requires a new,
explicitly named architectural mechanism that removes redundant
per-detection, tri-orientation 2D segmentation—for example, an authenticated
3D-native pretrained segmenter or another genuinely amortized 3D feature
source. Such a mechanism must receive its own preregistration and bounded cost
gate before any full run.

V32 is the active base. This PARK decision does not alter the standing
frame-level global-jump preprocessing or the requirement to measure its score
directly on an authenticated V32 base.
