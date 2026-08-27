# Association-cost CPU screen v1

Status: **PARK; no public submission**

## Mechanism

Use the existing fitted source-softmax association scorer as a full-cost assignment
where its cached candidate components are complete:

- edge cost: `-log(P(target | source))`;
- unmatched-source cost: `-log(P(no-link | source))`;
- topology: `scored_partial`;
- incomplete candidate components: explicit legacy fallback, counted in the report;
- divisions: OFF;
- detector inference: none;
- frozen scorer: unchanged.

This is structurally different from the learned-bonus probes, which retained legacy
motion-cost assignment and only subtracted a scalar learned bonus.

## Control

The control is the exact banked `c=0.25` fragment set:

`reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/`
`motion_error_weight_0p25/fragments`

Its full-CV57 identity is score `0.8731495935289003`, edge TP/FP/FN
`34,205 / 2,685 / 2,063`.

## Screen

The affected slice is the ten largest weighted residual contributors under that
control, selected before running the new topology:

1. `train:44b6_71a4179f`
2. `train:6bba_786893ac`
3. `train:44b6_87bba6c4`
4. `train:6bba_3abfe10a`
5. `train:6bba_ebdf3b34`
6. `train:44b6_cf2536e8`
7. `train:44b6_e57ff5c6`
8. `train:44b6_d5e7d891`
9. `train:6bba_f17befbc`
10. `train:44b6_267148e4`

The ranking uses `estimated_number_of_nodes × (1 − adjusted edge Jaccard)` from
`obs_diff_c0p25_vs_c2_v1.json`; it is not selected using the new model's outcomes.

## Decision rule

- `>= +0.005` adjusted edge Jaccard on the pre-registered slice: run the same
  immutable setting on full CV57.
- `+0.002 .. < +0.005`: HUMAN-DECISION before a full confirm.
- `< +0.002`: PARK this association-cost artifact/topology pair.

Regardless of score, the report must show nonzero `scored_partial_components` and
must record the fallback share. Observation comparison against the control precedes
any calibration or submission claim.

## Result

The pre-registered ten-video screen completed on CPU from cached `FrameDet`,
`PairContext`, and `PairEdges` inputs:

| quantity | `c=0.25` control | scored partial | delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | 0.721796529 | 0.688537189 | **-0.033259340** |
| edge TP | 5,117 | 5,026 | -91 |
| edge FP | 1,133 | 1,336 | +203 |
| edge FN | 730 | 821 | +91 |

The mechanism was genuinely active: 43,764 components and 911,366 candidate pairs
used learned partial matching, producing 298,707 learned-topology edges. Only 198
components fell back to legacy assignment (0.45% of considered components), so the
negative result is not explained by silent fallback.

Phase-0 observation comparison classifies the output as **DISTINCT** from the
control:

- node Jaccard: 0.946488;
- raw edge Jaccard: 0.799901;
- fork agreement: 1.0 (both division-off).

The worst single video improved in the smoke run, but the pre-registered aggregate
lost 0.03326 and crossed the PARK boundary decisively. Per the decision rule, no
full-CV57 run and no public submission will be made for this immutable
source-softmax/full-cost topology pair.

Artifacts:

- `reports/g02/scored_partial_source_softmax_screen10_v1/report.json`
- `reports/g02/screening/obs_diff_association_cost_screen10_v1.json`

## Interpretation

The existing probabilities are not calibrated well enough to replace geometric
assignment wholesale: relative to the control they lose true links while adding
more false links. This parks this artifact/topology pairing, not learned association
as a direction. The next CPU screen should test a conservative, fit-only calibrated
blend or rejection rule before considering new training or GPU work.

## Follow-on pre-registration: conservative additive blend

Parameter selection is moved off CV57 to the 12 fit-only validation videos recorded
by the source-softmax training report. The immutable selection is
`source_softmax_fit_validation12_v1.json`; it contains no audit or held-out CV57
videos.

The CPU replay grid is fixed before scoring:

- topology: `legacy`;
- divisions: OFF;
- backward motion coefficient: `0.25`;
- source-softmax learned bonus: `0, 0.25, 0.5, 1, 2, 4`;
- all other graph parameters frozen.

`bonus=0` is the explicit geometry-only control. Choose the single validation-best
bonus, then evaluate only that setting on the already registered CV57 worst-residual
slice. A full CV57 confirm still requires `>= +0.005` versus the exact banked
`c=0.25` control. The six-way validation search and subsequent slice reuse must be
reported as model-selection evidence, not an unbiased performance estimate.

### Follow-on result

Fit-only validation selected `bonus=2`:

| bonus | validation adjusted edge Jaccard | delta vs bonus 0 |
|---:|---:|---:|
| 0 | 0.867296766 | — |
| 0.25 | 0.866535958 | -0.000761 |
| 0.5 | 0.866781131 | -0.000516 |
| 1 | 0.867927173 | +0.000630 |
| **2** | **0.868712822** | **+0.001416** |
| 4 | 0.867803572 | +0.000507 |

Only `bonus=2` was then evaluated on the registered CV57 slice:

| quantity | `c=0.25` control | source-softmax bonus 2 | delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | 0.721796529 | 0.723802728 | **+0.002006199** |
| edge TP | 5,117 | 5,124 | +7 |
| edge FP | 1,133 | 1,124 | -9 |
| edge FN | 730 | 723 | -7 |

Observation comparison is **DISTINCT**: node Jaccard 0.979215, raw/scorer-eligible
edge Jaccard 0.936087, and fork agreement 1.0. The result lands in the registered
`+0.002 .. < +0.005` HUMAN-DECISION band. It is not promoted to a full CV57 run
automatically and makes no public-score claim.

Artifacts:

- `reports/g02/source_softmax_bonus_fit_validation12_v1/report.json`
- `reports/g02/source_softmax_bonus2_screen10_v1/report.json`
- `reports/g02/screening/obs_diff_source_softmax_bonus2_screen10_v1.json`
