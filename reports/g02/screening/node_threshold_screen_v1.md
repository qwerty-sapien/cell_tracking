# Node-threshold CPU screen v1

Status: **PARK; no public submission**

## Rationale and identity

This is a detector/front-end mechanism probe, not a new learned architecture.
The fit-only 14-video grid selected threshold `0.985` over the executed `0.970`
control with +0.004357 mean adjusted edge Jaccard and no video exceeding the
0.002 node-recall-loss guard. The evidence is provisional because the fit grid
used released sequential-D4 detections.

The held-out screen uses persisted one-pass CV57 `FrameDet` scores and matching
`PairContext` features. Raising the threshold is an exact cache filter for retained
detections: node rows and pair features are filtered together, association
probabilities and graph topology are recomputed, and the frozen scorer is rerun.
No detector forward pass or new inference is needed.

Frozen setting:

- threshold: `0.985` (selected before this screen);
- graph: banked backward motion coefficient `0.25`;
- fitted association artifact: the same hash-pinned conflict residual used by the
  banked graph replay;
- learned bonus: `1`;
- divisions: OFF;
- image refinement and projection: unchanged;
- control: exact banked `c=0.25` fragments, score `0.8731495935`.

## Selection and gate

Use the same pre-registered ten largest weighted-residual CV57 contributors listed
in `association_cost_screen_v1.md`. This reuses a diagnostic slice across mechanisms,
so its delta is screening evidence rather than an unbiased estimate.

- `>= +0.005`: confirm the immutable threshold on full CV57.
- `+0.002 .. < +0.005`: HUMAN-DECISION.
- `< +0.002`: PARK.

Observation comparison against the exact control is required before any promotion.

## Result

The ten-video cached replay completed:

| quantity | `c=0.25` control | threshold 0.985 | delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | 0.721796529 | 0.724542416 | **+0.002745888** |
| raw edge Jaccard | 0.733094556 | 0.732319610 | **-0.000774946** |
| node recall | 0.992105048 | 0.987979569 | **-0.004125479** |
| edge TP | 5,117 | 5,105 | -12 |
| edge FP | 1,133 | 1,124 | -9 |
| edge FN | 730 | 742 | +12 |

The filter removed 24,156 of 436,512 cached detections (5.53%). Observation
comparison is **DISTINCT**: node Jaccard 0.926864, edge Jaccard 0.902125, and fork
agreement 1.0.

Although the adjusted score lands in the nominal HUMAN-DECISION band, the held-out
mechanism contradicts the fit-side safety premise: raw edge quality and node recall
both worsen, TP falls, and FN rises. The apparent gain comes from the metric's
node-count adjustment rather than better tracking. This setting is therefore PARK,
not promoted to full CV57 and not eligible for submission.

Artifacts:

- `reports/g02/node_threshold_0p985_screen10_v1/report.json`
- `reports/g02/screening/obs_diff_node_threshold_0p985_screen10_v1.json`
