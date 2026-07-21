# Architecture and Algorithm Notes

## Audited repository structure

The repository has three active paths plus historical duplicate-route evidence:

1. `src/celltrack_baseline/rule_based.py` is the dependency-light DoG plus adjacent Hungarian baseline used by local and Kaggle scripts.
2. `src/celltrack_alpha/` profiles a video, routes it, fuses point proposals, selects edges/divisions, and evaluates sparse graphs.
3. `src/celltrack_ultrack/` creates foreground/contour clips and lazily invokes the optional Ultrack backend.
4. Historical duplicate-route reports document a removed GEFF-copy shortcut that matched ZIP member names, uncompressed sizes, and CRC32. This remains leakage evidence only; the executable duplicate-copy source and script are no longer active.

No packaging manifest or original automated test suite was present. The local `.venv` uses Python 3.14; Ultrack is not installed, and the existing runbook records that published Ultrack releases used by this project require an older supported Python. The repository contains an 81 GB ZIP and extensive derived reports; these remain untouched.

## Coordinate contract

The canonical array contract is `T/Z/Y/X`. Node coordinates are voxel-center indices stored as integer `(t, z, y, x)` values for competition export. Physical association distances are calculated as:

```text
delta_um = (target_zyx - source_zyx) * scale_zyx
distance_um = ||delta_um||2
```

`parse_scale` now validates case-insensitive OME-NGFF names, requires the raw-array metadata order `T/Z/Y/X`, composes dataset- and multiscale-level scale transforms, and requires exactly one `z`, `y`, and `x` axis with a consistent recognized metric unit. Meter, millimeter, micrometer, and nanometer values are normalized to micrometers. The historical competition scale is retained only when `multiscales` metadata is absent. Present-but-malformed, unitless, mixed-unit, unsupported-unit, or noncanonical physical metadata fails rather than being silently misinterpreted.

The current implementation does not apply arbitrary affine, rotation, displacement, or origin transforms. Such transforms require an explicit coordinate object and must not be approximated as voxel spacing.

The `t` axis is validated as canonical time, but motion models still use integer frame differences and `max_gap` is measured in frames. Uneven timestamps or physical seconds are not modeled; adding them requires per-frame time coordinates and speed gates with explicit time units.

## Proposal features

`CandidateNode` currently represents point detections. Available association evidence is:

- physical centroid;
- detector score and source provenance;
- local point density;
- intensity sampled at the candidate center;
- optional scalar features supplied by future proposal generators.

There are no instance masks in the fallback path, so volume, covariance, elongation, surface, and robust region-intensity summaries are unavailable. Shape costs are therefore not fabricated. A future instance-feature stage should compute those quantities in physical coordinates and define behavior for disconnected labels.

## Edge cost and hard gates

Candidate generation first applies a physical radius gate and a per-source neighbor cap. The base cost is:

```text
distance / physical_radius
+ density_penalty
+ gap_penalty
+ appearance_weight * symmetric_relative_intensity_difference
- node_score_bonus
```

The symmetric intensity difference is bounded and safe for zero or non-finite values. Its compatibility default is `0.0`: it is implemented but must be calibrated on held-out data before being enabled. This prevents an unvalidated appearance heuristic from silently changing every competition route.

For gap candidates with a predecessor, the second stage optionally adds a bounded constant-velocity residual in physical coordinates. The residual and cost contribution are retained in edge diagnostics.

## Globally optimal adjacent assignment

Each adjacent frame pair is solved with `scipy.optimize.linear_sum_assignment` over an augmented square matrix:

```text
[ feasible link costs | diagonal death costs ]
[ diagonal birth costs | zero dummy block    ]
```

Failed gates and off-diagonal dummy choices are forbidden. A real link must be below both the hard edge threshold and the combined birth/death alternative. Birth and death costs may be configured explicitly; otherwise each is half the existing edge threshold, preserving its interpretation.

Inputs and duplicate candidates are canonically sorted by node ID and complete edge key before the solver is called. This produces byte-identical results under candidate-list permutations in the tested SciPy environment without numerically perturbing meaningful costs. Exact-tie choice is not formally guaranteed to remain identical across different SciPy implementations or versions, so solver version is part of reproducibility provenance; a future cross-version guarantee would need an exact secondary optimization rather than floating-point noise.

## Tracklet gap closing

Gap edges are no longer mixed into the first greedy edge list. The algorithm now:

1. solves all adjacent assignments;
2. identifies tracklet endpoints and starts;
3. keeps only forward-time gap candidates within `max_gap`;
4. adds constant-velocity residuals when history exists;
5. decomposes the sparse endpoint/start candidate graph into independent bipartite components;
6. solves one augmented global assignment per component, which is exactly equivalent because births/deaths are node-local and no link crosses components;
7. rechecks degree constraints before committing each join.

This avoids a cheap long-gap candidate stealing a detection that already has a valid adjacent predecessor. Results inside the gap horizon are conceptually provisional; the current alpha still retains all candidates in memory rather than exposing a streaming commit API.

## Divisions and lineage invariants

Division remains a separate conservative one-parent/two-child selector and is disabled in normal routes unless explicitly enabled. The graph validator enforces:

- unique node IDs and edge pairs;
- existing endpoints;
- strict forward time and exact declared `dt`;
- in-degree at most one, so merges are forbidden;
- out-degree at most one without divisions and at most two with divisions;
- two division children at the same immediately following timepoint.

A gap child cannot be combined with a next-frame child to masquerade as a division. Strict forward-time edges make the graph acyclic by construction.

## Validation boundaries

The alpha pipeline now rejects empty/non-4D arrays, non-real dtypes, invalid windows, malformed output offsets, invalid physical scales, invalid solver parameters, and non-finite per-frame volumes. Validation operates before graph construction and avoids printing image data or metadata values beyond the invalid field.

The optional Ultrack import is lazy. A missing package now produces an actionable optional-backend error. A third-party import-time type-annotation crash is wrapped as an incompatible Ultrack/Python error rather than leaking the original `builtin | None` `TypeError`. This does not make an unsupported Ultrack release compatible with Python 3.14; it makes the failure controlled and diagnosable.

## Evaluation harness

`src/celltrack_alpha/synthetic.py` and `scripts/benchmark_synthetic_tracking.py` implement a deterministic fixed-detection test on an anisotropic `T/Z/Y/X` volume. It contains moving tracks, one missed frame, and a z-displaced distractor whose assignment changes when anisotropy is ignored. The acceptance gate checks exact edges, gap recovery, graph invariants, repeat determinism, a golden fixture/config manifest digest, a golden output digest, and runtime.

Segmentation is intentionally bypassed. This lane tests association software only. End-to-end detection/segmentation evaluation and representative biological validation remain open work.

## Memory, visualization, and export

- ZIP-backed Zarr and GEFF inputs are read without extracting the 81 GB archive.
- Classical proposal generation materializes one volume at a time, but retains all candidate nodes and edges for a window.
- Ultrack input construction materializes complete foreground and contour arrays for the requested clip; it is not safe to treat as whole-archive streaming.
- Existing overlay scripts use derived PNGs and sparse coordinates. No interactive UI exists in the audited source.
- Submission CSV validation checks duplicate IDs/edges and missing endpoints. Atomic export and spreadsheet-injection defenses are not yet implemented.

The duplicate-GEFF production route was re-run to temporary files after these changes and remained byte-identical to the existing production submission.
