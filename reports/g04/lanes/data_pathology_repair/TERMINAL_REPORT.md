# G04 data-pathology repair — terminal report

Date: 2026-08-03  
Node: `G04-DATA-PATHOLOGY-REPAIR-V1`  
Decision: **PROMOTE G1 LOCALLY / BANK F1 / STRIDE ALREADY CORRECT / KILL CROP-WRFEAT-V1**

## Outcome

On the fixed Prompt-1 control—the exact APNCC CV57 artifact—frame-global
registration before candidate construction raises the exact official score from
`0.8815220699597315` to `0.8867644202595942`, a delta of
`+0.0052423502998628`.  The frozen-frame identity repair is score-neutral.
The combined result equals registration alone.  No GT row was changed.

| arm | exact adjusted score | delta | edge TP / FP / FN | decision |
|---|---:|---:|---:|---|
| APNCC Prompt-1 base | 0.881522069960 | — | 34,320 / 2,459 / 1,948 | control reproduced exactly |
| F1 exact frozen identity | 0.881522069960 | +0.000000000000 | 34,320 / 2,459 / 1,948 | bank; score-neutral |
| G1 frame-global registration | **0.886764420260** | **+0.005242350300** | **34,389 / 2,308 / 1,879** | promote locally |
| F1 + G1 | **0.886764420260** | **+0.005242350300** | **34,389 / 2,308 / 1,879** | same score as G1 |
| S1 stride repair | 0.881522069960 | +0.000000000000 | 34,320 / 2,459 / 1,948 | identity; no mismatch existed |
| crop computed-segmentation `wrfeat` | not scored | not admissible | — | pilot failed completeness/fidelity gate |

Exact candidate aggregates are recorded in `score/REPORT.json`.  G1 removes
9,099 raw APNCC edges on the 75 flagged CV57 transitions, applies the frozen
unchanged edge scorer and legacy local association solve to registered geometry,
and adds 7,995 registered raw edges.  All three candidate fragment sets have
57/57 coverage, adjacent-time edges, indegree at most one, and outdegree at most
one.  Final node rows and evaluator coordinates remain the original APNCC rows.

## D0 — exact frozen frames

The complete archive census covers 203 videos and 20,097 transitions:

- 967 transitions are exactly frozen, affecting 116 archive videos.  They are
  all `6bba`: 947/12,672 train transitions in 114/128 train videos, plus 20/198
  test transitions in 2/2 test videos.  The two affected test identities
  duplicate train identities, so the unique-dataset count is 947 transitions
  in 114/199 datasets.
- Every frozen candidate was first identified by equal ZIP size/CRC and then
  confirmed by byte equality of the canonical lossless Zarr frame chunks:
  967/967 confirmed, zero rejections.  Thus every decoded `uint16` voxel
  difference is exactly zero.
- Across the archive, frozen-run transition lengths are 792 runs of length 1,
  11 of length 2, and 51 of length 3 (respectively two-, three-, and four-frame
  repeats).  CV57 contains 282 frozen transitions in 32/57 videos.

Current APNCC has no frozen-frame branch.  Identical raw frames enter different
two-frame inference windows and do not generally yield identical detections:
only 36/282 CV57 frozen transitions have identical complete coordinate sets.
There are 41,970 exact-coordinate counterparts among 47,467 source detections.
The learned pool contains 120,615 candidates, 41,970 at zero distance.  The
final graph contains 44,510 adjacent edges on frozen transitions, only 5,667 of
which have zero displacement in final output coordinates; 903 final edges are
outside the learned cap-8 pool but inside the graph's uncapped 10-um geometry.
The evaluator never reads images and gives frozen pairs no special treatment.

F1 installs 39,953 retained-node exact-coordinate pairs: 39,928 were already
selected by node identity, 25 were novel, and 26 conflicts were removed; 2,017
raw pairs referred to nodes pruned from the final APNCC fragment and were
correctly skipped.  Despite the changed fragment aggregate, the exact score and
all official counts are unchanged.

The official GT is retained verbatim.  As an audit only, 2,515/2,522 CV57 GT
edges crossing frozen transitions have exactly zero displacement; seven are
nonzero, consistent with the small bad-annotation warning, and none exceeds
10 um.

## D1/D2 — global jumps and candidate repair

Registration uses each complete origin-0 strided volume
`volume[:, ::4, ::4]` (64-cubed, isotropic 1.625-um samples), robust q01/q99
clipping, median centering, a separable Hann window, and integer 3-D phase
correlation.  Frozen transitions short-circuit to zero.  For 19,130 nonfrozen
transitions, shift magnitude in micrometres is:

| quantile | p00 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| magnitude | 0.000 | 1.625 | 2.298 | 3.980 | 5.390 | 10.405 | 54.504 |

Histogram counts are 11,438 at `[0,1.625]`, 4,393 at `(1.625,3.25]`, 2,080
at `(3.25,5]`, 809 at `(5,7.5]`, 183 at `(7.5,10]`, 112 at `(10,12]`, 60
at `(12,15]`, 32 at `(15,20]`, and 23 above 20 um.  The preregistered
pool-breaking criterion flags 227 transitions in 82 archive videos: 21 train
`44b6`, 205 train `6bba`, and one test `6bba`.  CV57 contains 75 flagged
transitions in 23 videos, with magnitude 10.017–47.153 um.

The current sparse pool is built in unregistered physical coordinates by a
10-um cKDTree radius and deterministic cap 8; the official scorer sees only the
submitted graph.  Therefore a coherent camera shift over 10 um removes
stationary-cell true edges before the learned scorer can inspect them.

G1 cumulatively applies only the flagged frame translations to detection
geometry before both the 10-um/cap-8 candidate gate and association scoring.
This keeps every unflagged pair's relative geometry unchanged after a prior
jump.  Only association geometry changes; original node IDs, times, and final
coordinates are retained.  On the 75 flagged CV57 transitions:

- candidate count changes from 27,276 to 25,553;
- 311 GT continuations have both endpoints matched to raw detections;
- reachable continuations improve from 209 to 259, recovering 50 net true
  edges and reducing pool absence from 102 to 52.

The independent GA-R12REACH fit-only117 bank contains 456 correct edges in the
10–12-um shell.  Ninety-three lie on transitions whose measured global shift
exceeds 10 um; G1 moves 90/93 back inside 10 um.  The other 363 shell edges are
on non-pool-breaking transitions and remain evidence for natural long jumpers
or other non-global mechanisms.  This partition is fit-only117, not CV57.

The unchanged-GT CV57 audit similarly finds 314 GT edges on flagged
transitions.  Eighty-one are over 10 um before correction; 74 of those return
inside 10 um, and 263/314 are inside after correction.  This is an audit, not a
GT edit or selection rule.

This mechanism is causally distinct from terminal A7.  A7 used local
phase/template residual as a per-conflict candidate feature.  G1 estimates one
label-blind transform from the entire frame and applies it before candidate
construction.  A7 remains closed and was not re-screened.

## D3 — downsampling parity

No repair was needed.  The pinned organizer training and prediction sources,
the materialized APNCC CV57 code, current inference, and sparse training loader
all use literal origin-0 striding with spatial stride `(1,4,4)`, i.e.
`frame[:, ::4, ::4]`.  None uses a block mean.  APNCC's full-resolution raw
patches are an intentional later feature path, not a downsampling mismatch.
The scorer is graph-only.  S1 is therefore byte-identical by construction with
exact delta zero; no cache needs invalidation for stride parity.

## D4 — crop-level computed-segmentation `wrfeat`

The owner-requested `CROP-COMPUTED-SEGMENTATION-WRFEAT-V1` pilot reused Lane
B's authenticated Cellpose3 `cyto3` weights/settings and exact native
`wrfeat12` contract, but segmented one fixed APNCC-size `(16,32,32)` crop per
detection.  Intensity uses whole-frame p1/p99.8 normalization, border distance
uses original full-frame XY coordinates, and every mask is explicitly labelled
`COMPUTED_SEGMENTATIONS`.

The fixed real-frame pilot on `train:44b6_40c45f5a`, timepoint 0, is terminal:

- 231/243 detections receive finite nonzero features; 12 crops are empty and
  unmatched;
- 199/243 selected masks (81.893%) touch a crop boundary, so their morphology
  is truncated rather than faithful whole-object `wrfeat`;
- 483.937 CPU seconds are required for the frame (median 1.962 s/detection),
  projecting to 766.233 CPU-hours for CV57.

The completeness and fidelity gates fail.  No full CV57 feature production,
Trackastra score, GPU, Kaggle, RunPod, or submission was run.  This rejects the
fixed crop realization; it does not overwrite Lane B's whole-frame timing
evidence or claim a cross-device CPU/T4 speed ratio.

## Integrity and artifacts

- Census: `CENSUS.json`, SHA-256
  `f56fee41667159d681f31147dfbb18943d418cbb45d76a30daf169885403cdad`.
- Exact score/candidate/shell report: `score/REPORT.json`, SHA-256
  `2144019cca1da3dc10452b5f2dd0d2d8c01669852b0fec3816424e1fcc878a64`.
- Crop pilot: `../computed_segmentation/crop_wrfeat_v1/PILOT_44b6_40c45f5a_T000.json`,
  SHA-256 `6be0744c04777f8cfa4e4d0447d9c0e2cbc1a7d0fa43e47c4d62160e8d2e67e1`;
  NPZ SHA-256 `56bb64906fb077fb22aa78913067d5e24aa925da49750a4aa42d432963ee4058`.
- Focused verification: 34 tests passed.  Four expected scikit-image warnings
  exercise the legacy nonfinite-solidity fixture; the replacement voxel-cell
  feature remains finite.
- Execution was local CPU only.  No Kaggle, GPU, submission, or RunPod action
  occurred.

G1 passes the preregistered strict local promotion gate.  External transfer is
not attempted while the campaign's serial submission/public-baseline work is
active.
