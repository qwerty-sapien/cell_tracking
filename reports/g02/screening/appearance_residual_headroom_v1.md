# Appearance direct-swap residual and 0.90 headroom

**Decision: BANK as a diagnostic; do not tune a production router from these
CV57 labels.**

The frozen `0.8815220699597315` appearance direct-swap candidate leaves
`2,459 FP + 1,948 FN = 4,407` edge residual events. Its fixed node-adjustment
factor is `0.994717518`, so an adjusted score of `0.900000` requires edge
Jaccard `0.904779482`.

Under transparent idealized mechanisms, the remaining gap is equivalent to one
of:

- `720` FN→TP recoveries with denominator unchanged;
- `796` correct FP prunes;
- `378` perfect wrong-edge→right-edge swaps, where each swap adds one TP and
  removes one FP and one FN.

The last quantity is the useful scale for the next association mechanism. The
current direct-swap line already produced `+115 TP / -226 FP / -115 FN`, so
`0.90` is not arithmetically remote: it needs roughly three more times the
current perfect-swap yield, not a wholesale detector replacement.

## Residual concentration

| worst residual videos | FP+FN | share of remaining residual |
|---:|---:|---:|
| 1 | 548 | 12.43% |
| 3 | 1,112 | 25.23% |
| 5 | 1,549 | 35.15% |
| 10 | 2,306 | 52.33% |
| 20 | 3,263 | 74.04% |

The concentration persists after the successful appearance correction. Family
prefix is still not a valid router, but it sizes the affected population:
`6bba` carries `3,555 / 4,407 = 80.67%` of residual events and `44b6` carries
`852 / 4,407 = 19.33%`.

The largest remaining video, `train:6bba_786893ac`, alone contains
`325 FP + 223 FN = 548` residual events. The current mechanism made 308
proposals there but yielded only `+4 TP / -23 FP / -4 FN`. This is evidence that
centroid-pair NCC is useful but insufficient in the most crowded regime.
Video identity and residual rank are diagnostic labels only and are forbidden
production features.

## Stability of the current mechanism

Across all 57 videos:

- 43 improved in event utility;
- 8 were neutral;
- 6 regressed;
- 5,047 proposals produced aggregate event utility `+456`;
- the mean yield was `9.03` utility events per 100 proposals.

The six regressions are small (`-8` worst; the other five total `-13`) relative
to the aggregate gain. This supports retaining the current candidate rather
than replacing it while the next mechanism is developed.

## Next mechanism constraint

The next screen must be selected exclusively from fit-only data and must address
occupied-endpoint association conflicts beyond the current single-pair,
10 µm-candidate, centroid-NCC support. Pre-register:

1. tracklet predecessor/successor motion consistency;
2. multi-frame appearance consistency rather than another centered-pair NCC
   threshold;
3. an extended-radius reachability audit for correct edges outside the frozen
   candidate cap;
4. at least `+0.005` on a fit-only affected-slice proxy before any full CV57
   confirmation.

No CV57 identity, residual rank, scorer label, or per-video outcome may be used
as a production router or threshold-selection input.

Machine report:
`reports/g02/selected_graph_appearance_direct_residual_v1.json`
(`sha256 9d8934672847197f56cdce1cbc14223bef4ac52d93979dab049b4faa544047c1`).
The audit is read-only, performed no Kaggle/RunPod/GPU action, and consumed no
submission.
