# P0B mechanism-diverse anchors v1

Decision: **BANK cached Stage-A evidence; submit no calibration-only anchor.**

This file retains the requested legacy `cross_family_anchors` name, but uses
the revised mechanism labels from `P0R_recalibration_wide_dispersion.md`.
A detector threshold, association scorer, graph solver, division postprocessor,
and learned architecture are not assumed to be interchangeable “families.”

No Kaggle submission, Kaggle GPU, RunPod, or other external action occurred.

## Full-CV57 anchors

All scores and observation vectors below use the same attested 57-video scorer
and the exact fragment identities frozen in
`dispersion_ladder_observation_matrix_v1.json`.

| Anchor | Mechanism | Same-range control | Local scores | `abs(delta local)` | Node J / edge J | Full identity | Public package | Decision |
|---|---|---|---:|---:|---:|---|---|---|
| P4 no-attention control | learned association architecture/control | graph-cost `c=0.25` | 0.874290 / 0.873150 | 0.001140 | .991023 / .974946 | exact 57/57 | none reproduced | BANK as cached matched-local interaction; do not rebuild the harmful P4 attention layer |
| APNCC global re-solve | appearance association postprocessor + global solver | P4 no-attention | 0.877401 / 0.874290 | 0.003111 | .990483 / .973725 | exact 57/57 | none reproduced | PARK for accuracy; retain as a solver-control anchor |
| APNCC direct swap | appearance association postprocessor + exact local solver | graph-cost `c=0.25` | **0.881522 / 0.873150** | 0.008372 | .992972 / .984736 | exact 57/57 | **READY**: `bhap-submit` | BANK as the high-range performance and transfer anchor |
| APNCC direct swap | same appearance proposals, local solver | APNCC global re-solve | 0.881522 / 0.877401 | 0.004121 | .994074 / .988688 | exact 57/57 | direct only | useful score-separated solver comparison; not a matched-local pair |

The no-attention/P4 control supplies one cached matched-local
mechanism-diverse interaction, but one such pair cannot establish a general
mechanism effect. APNCC-direct supplies the missing local `>0.88` leverage and
is the only anchor with a reproduced fail-closed public package.

## Stage-A candidates that failed before full anchoring

| Candidate | Scope reached | Evidence | Consequence |
|---|---|---|---|
| full source-softmax association-cost replacement | preregistered ten-video residual slice | adjusted delta **-0.033259**; DISTINCT, node J .946488 / edge J .799901 | PARK this artifact/topology pair; no full-CV57 anchor or package |
| source-softmax additive bonus 2 | fit-selected, ten-video residual slice | adjusted delta +0.002006; HUMAN-DECISION band | not a full-cost replacement and not a full-CV57 anchor |
| detector threshold 0.985 | fit-selected, ten-video residual slice | adjusted +0.002746, but raw -0.000775 and node recall -0.004125; DISTINCT | PARK; apparent gain came from node adjustment, so no full-CV57 anchor |
| divisions ON / `add_safe_divisions` | sandbox and lineage audit | no added edge on 13 valid fixtures; 0/51 removed forks restored; current emitted forks fail the official lineage objective | PARK and keep divisions OFF |

These negative screens are informative for model development, but they cannot
be promoted into transfer-calibration pairs because no exact full-CV57
prediction identity was authorized for them.

## Package-readiness gate

`kaggle_kernel/bhap-submit` is the only Stage-A anchor ready for a public
measurement:

- local score `0.8815220699597315`;
- TP/FP/FN `34,320 / 2,459 / 1,948`;
- package-path proposal parity exact on 57/57 videos;
- fragment tree byte-identical to the banked candidate;
- manifest identity
  `779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b`;
- expected 7 minutes wall time and at most 0.25 Kaggle GPU-hours;
- one competition submission if, after generation and output inspection, the
  artifact is explicitly approved for submission.

## Sequential decision

1. If public measurement is authorized, measure APNCC-direct alone.
2. Treat the displayed score as a 0.001-wide rounding interval and update P0B
   before selecting any second probe.
3. If transfer is unsurprising and the score improves, conserve submissions
   for materially stronger local candidates.
4. If direct swap under-transfers, reproduce APNCC-global as a fail-closed
   solver-control package before considering P4 no-attention or lower-range
   learned-bonus probes.

Stage B learned-model work remains governed by the independent GA/GD gates.
Neither a favorable nor unfavorable calibration verdict is, by itself, a GPU
trigger.
