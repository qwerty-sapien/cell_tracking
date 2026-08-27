# Global-jump integration and crop-wrfeat repair — terminal report

Date: 2026-08-03

## Decision

Global-jump correction is now a standing transform-once preprocessing seam
before candidate construction for APNCC, Trackastra/computed-wrfeat,
synthetic-division-integrated, and a fresh V32 replay. Original coordinates
remain authoritative for image/feature sampling and evaluator output;
registered coordinates are used only for candidate topology and association
geometry. This is frame-level registration, not an A7 candidate feature.

The repaired crop-wrfeat implementation passed its bounded GPU coverage gate
but failed its compute-budget gate. The feature lane is therefore
`PARK-COST`, not `PARK-EVIDENCE` or `DEAD`; its geometry and coverage proof are
banked for reuse. No full feature extraction or crop-wrfeat score run was
launched or authorized, and no fourth throughput configuration is allowed
without a new named architecture that removes redundant per-detection
tri-orientation segmentation.

## Crop diagnosis and repair

The real-frame computed-object audit found 171 unique objects. Z/Y/X extents
were p50 `9/18/16`, p90 `15/31/26`, p95 `19.5/34/28`, p99
`23.3/39.3/34.9`, and maximum `27/46/42` voxels. The old fixed
`16x32x32` crop could literally contain only `57/243` pilot instances
(`23.4568%`), explaining the earlier `199/243` boundary contacts.

The preregistered repair uses intensity recentering capped at 4 um, a
`32x64x64` primary crop, one `48x96x96` fallback, and batched CUDA execution
across detections in each frame. The batch path runs the same Cellpose network
and mask dynamics as the serial path; their one-detection label arrays were
byte-identical in the remote pilot.

## Bounded T4 result

| Quantity | Result |
|---|---:|
| Independent coverage frames | `673/673` valid; `0` boundary; `0` empty |
| All three frames | `914/916` valid; `1` boundary; `1` empty |
| Fallback recovery | `50/52` |
| Worker / kernel elapsed | `902.346 / 918.830 s` |
| Throughput | `0.938601 s/detection` |
| CV57 exact-count projection, before margin | `293.262 T4-h` |
| CV57 dense-frame projection, before margin | `599.869 T4-h` |
| CV57 preregistered conservative projection | `659.856 T4-h` |
| Held-fit105 + CV57 conservative projection | `1875.361 T4-h` |
| Authenticated weekly balance after pilot | `2.76 used / 27.24 remaining h` |

The geometry/coverage defect and CPU-versus-GPU implementation defect are
fixed, but three-orientation segmentation per detection remains much more
expensive than whole-frame reuse. The conservative `659.856 T4-h` CV57
projection cannot fit the remaining `27.24 h`; the full-run gate therefore
fails closed.

## Exact-score accounting

| Repair/base | Exact CV57 effect | Status |
|---|---:|---|
| APNCC + standing G1 preprocessing | `0.881522069960 -> 0.886764420260` (`+0.005242350300`) | Measured |
| V32 + G1 preprocessing | — | Pending all-six-shard immutable V32 base authentication; must be remeasured directly |
| Synthetic-division parent + G1 | — | Pending authenticated parent base |
| Crop-level real wrfeat | — | Not measured: budget gate failed, so full extraction/scoring stayed closed |

## Authoritative evidence

- Transform and base-family contract: `INTEGRATION_CONTRACT.md`
- Geometry audit: `CROP_GEOMETRY_44b6_40c45f5a_T000.json`
- Pilot preregistration: `PREREGISTRATION.md` and
  `PREREGISTRATION_AMENDMENT_1.md`
- Terminal machine adjudication:
  `CROP_GPU_PILOT_TERMINAL_ADJUDICATION.json`
- Remote atomic result:
  `crop_gpu_pilot_remote_output_v1/G04_CROP_WRFEAT_GPU_PILOT.json`

Focused verification: `45 passed`; adjacent public-TemporalUNet coverage:
`27 passed, 3 parameterized subtests`.
