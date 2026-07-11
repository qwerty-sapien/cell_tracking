# Delegation-Ready Work Queue

Statuses: `DONE` is implemented and locally verified; `READY` can be delegated without another design decision; `BLOCKED` needs the stated dependency. File scopes are exclusive starting points—delegates must preserve all reports, data, environments, and unrelated files.

## CT-001 — Named-axis physical calibration

- **Status:** DONE
- **Evidence:** Actual archive metadata reports axes `T/Z/Y/X`, scale `[1.0, 1.625, 0.40625, 0.40625]`, parsed as `(1.625, 0.40625, 0.40625)`. Calibration tests pass.
- **Dependencies:** None.
- **File scope:** `src/celltrack_baseline/rule_based.py`, `tests/test_parse_scale.py`.
- **Acceptance tests:** Canonical named TZYX maps to z/y/x; recognized consistent m/mm/µm/nm units normalize to micrometers; noncanonical, unitless, mixed, unsupported, absent-axis, non-positive, and non-finite metadata fails; wholly absent `multiscales` uses the historical fallback.
- **Risks:** OME-NGFF affine/non-linear transforms remain unsupported; time is still measured in frames rather than physical seconds or uneven timestamps. Do not approximate either as spatial scale.
- **Primary sources:** [OME-NGFF specification](https://ngff.openmicroscopy.org/latest/), [OME-NGFF paper](https://doi.org/10.1038/s41592-021-01326-w).

## CT-002 — Global adjacent LAP and births/deaths

- **Status:** DONE
- **Evidence:** The original greedy counterexample returned one link; the augmented assignment returns the globally optimal two links. The short competition smoke retains its node/edge counts and official metrics.
- **Dependencies:** CT-001.
- **File scope:** `src/celltrack_alpha/solver.py`, `src/celltrack_alpha/types.py`, `tests/test_alpha_solver.py`.
- **Acceptance tests:** Greedy-dead-end fixture yields `(1,4)` and `(2,3)`; failed gates never link; input permutation produces identical pairs; unassigned nodes resolve through birth/death dummies.
- **Risks:** Link costs and unmatched penalties still need held-out calibration; a larger matrix can cost more than greedy selection in dense frames.
- **Primary sources:** [Jaqaman et al.](https://doi.org/10.1038/nmeth.1237), [TrackMate LAP documentation](https://imagej.net/plugins/trackmate/trackers/lap-trackers).

## CT-003 — Tracklet gap closing and graph invariants

- **Status:** DONE
- **Evidence:** Tests recover one allowed missing-frame link, prevent a gap from replacing an adjacent predecessor, reject merges, and reject mixed-time divisions.
- **Dependencies:** CT-002.
- **File scope:** `src/celltrack_alpha/solver.py`, `tests/test_alpha_solver.py`.
- **Acceptance tests:** Gaps join endpoints to later starts only; `dt <= max_gap`; unique IDs/edges; existing endpoints; strict forward time; in-degree <= 1; out-degree <= 1 or 2 when divisions are enabled; division children share the next timepoint.
- **Risks:** Constant velocity can be wrong under abrupt motion; no uncertainty covariance is estimated; long horizons increase false joins.
- **Primary sources:** [Jaqaman et al.](https://doi.org/10.1038/nmeth.1237), [btrack lineage approach](https://doi.org/10.3389/fcomp.2021.734559).

## CT-004 — Deterministic synthetic 4D association harness

- **Status:** DONE
- **Evidence:** Ten repetitions give 14 nodes, 10/10 truth/predicted edges, edge precision/recall/F1 `1.0`, and one recovered gap. Golden fixture/config digest: `51b98139c2650ee2b0b2e57fe2049d8e510e177aad51e74077c1c06c484c1569`; golden edge digest: `eac380a9a04d3155b222df8c557d9840d90274c9dd545912ad6364b1fbceb99b`.
- **Dependencies:** CT-002, CT-003.
- **File scope:** `src/celltrack_alpha/synthetic.py`, `scripts/benchmark_synthetic_tracking.py`, `tests/test_synthetic_benchmark.py`.
- **Acceptance tests:** Exact graph; anisotropic z distractor fails under the wrong scale; reviewed fixture/config and output digests are golden-gated; valid invariants; positive repetition validation; runtime reported.
- **Risks:** Fixed detections bypass segmentation; engineered trajectories are easier than microscopy.
- **Primary sources:** [CTC evaluation methodology](https://celltrackingchallenge.net/evaluation-methodology/), [CTC Cell Linking Benchmark](https://public.celltrackingchallenge.net/documents/Cell%20Linking%20Benchmark.pdf).

## CT-005 — Ultrack import failure control

- **Status:** DONE
- **Evidence:** Missing-backend and simulated `builtin | None` type-crash tests return actionable `RuntimeError` messages.
- **Dependencies:** None.
- **File scope:** `src/celltrack_ultrack/core.py`, `tests/test_ultrack_backend.py`.
- **Acceptance tests:** Missing Ultrack and import-time `TypeError` are wrapped; unrelated missing subdependencies are not misreported as an absent top-level package.
- **Risks:** This controls failure; it does not make unsupported Python/Ultrack combinations compatible.
- **Primary sources:** [Ultrack repository](https://github.com/royerlab/ultrack), [Ultrack paper](https://doi.org/10.1038/s41592-025-02778-0).

## CT-101 — Fixed-detection biological linker benchmark

- **Status:** READY
- **Evidence:** The two-frame sparse smoke cannot distinguish greedy and global edges; 16 links changed while its official metric stayed identical.
- **Dependencies:** Representative train sequences and frozen detections; sequence-level split plan.
- **File scope:** New `scripts/evaluate_fixed_detections.py`, `src/celltrack_alpha/evaluation.py`, new tests; derived outputs under a new reports subdirectory only.
- **Acceptance tests:** Same detections supplied to greedy baseline, global LAP, and optional btrack; report edge precision/recall, LNK-compatible score, ID switches, gap recovery by length, division metrics, runtime, and peak memory per complete sequence; bootstrap by sequence.
- **Risks:** Sparse labels can hide wrong links; random frame splitting leaks temporal context; hyperparameter tuning can overfit dataset IDs.
- **Primary sources:** [CTC 10-year benchmark](https://doi.org/10.1038/s41592-023-01879-y), [CTC Cell Linking Benchmark](https://public.celltrackingchallenge.net/documents/Cell%20Linking%20Benchmark.pdf).

## CT-102 — Instance feature extraction and calibrated costs

- **Status:** BLOCKED
- **Evidence:** Current fallback nodes have point intensity and detector scores but no masks, volumes, covariance, elongation, or robust region statistics. Appearance weight is intentionally `0.0` by default.
- **Dependencies:** A mask/instance proposal source and CT-101 held-out evaluation.
- **File scope:** New `src/celltrack_alpha/features.py`, `src/celltrack_alpha/types.py`, `src/celltrack_alpha/proposals.py`, focused tests.
- **Acceptance tests:** Physical centroid/volume and covariance match analytic synthetic shapes; disconnected-label policy documented; intensity median/MAD/quantiles finite; feature scales estimated on training sequences only; ablation improves held-out linker metrics without detection changes.
- **Risks:** Tiny or near-isotropic instances have unstable orientation; intensity drift is modality-dependent; unnormalized features can dominate costs.
- **Primary sources:** [btrack feature guidance](https://btrack.readthedocs.io/en/stable/), [TrackMate LAP documentation](https://imagej.net/plugins/trackmate/trackers/lap-trackers).

## CT-103 — Supported Ultrack environment and real short-clip run

- **Status:** BLOCKED
- **Evidence:** Current Python is 3.14.6 and Ultrack is absent; only foreground/contour and controlled-error paths are locally testable.
- **Dependencies:** A separate supported Python environment approved by the user; no mutation of the current `.venv`; an already local short clip.
- **File scope:** Environment manifest outside source, `src/celltrack_ultrack/core.py`, `scripts/run_ultrack_zip_cv.py`, new tests/reports.
- **Acceptance tests:** Import/version recorded; no unexpected download; candidate count nonzero; solver status and optimality gap captured; graph invariants pass; foreground/contour quality inspected; runtime and peak memory reported.
- **Risks:** Solver licensing, package incompatibility, full-clip materialization, poor contour maps, and windowed suboptimality.
- **Primary sources:** [Ultrack repository](https://github.com/royerlab/ultrack), [Ultrack paper](https://doi.org/10.1038/s41592-025-02778-0).

## CT-104 — Streaming and memory-bounded alpha

- **Status:** READY
- **Evidence:** Classical detection loads one timepoint at a time but retains every node and candidate edge. Sparse bipartite components now prevent a whole-window dense matrix when the candidate graph is disconnected, and a 200-pair test bounds each disconnected solve to a 2x2 augmented matrix. One connected component can still span the window.
- **Dependencies:** CT-003 and frozen batch outputs.
- **File scope:** New `src/celltrack_alpha/streaming.py`, `src/celltrack_alpha/pipeline.py`, performance tests.
- **Acceptance tests:** Buffer at most `max_gap + 1` frames plus active track state; `flush()` equals batch graph byte-for-byte on synthetic and small archive clips; no monotonic retained-memory growth; checkpoints include calibration/config hash and next stable ID.
- **Risks:** Premature finalization changes gap decisions; checkpoint compatibility; one temporally percolating component can still allocate a dense square matrix and remains a release blocker until peak component size/RSS is bounded or a sparse fallback exists.
- **Primary sources:** [OME-NGFF paper](https://doi.org/10.1038/s41592-021-01326-w), [Ultrack scalability evidence](https://doi.org/10.1038/s41592-025-02778-0).

## CT-105 — CTC DET/SEG/TRA evaluation lane

- **Status:** BLOCKED
- **Evidence:** Current evaluation is the competition's sparse graph metric; the fallback emits points, not segmentation masks.
- **Dependencies:** CTC-format masks/tracks and a clearly separated end-to-end method.
- **File scope:** New `scripts/evaluate_ctc.py`, isolated adapters/tests; do not alter competition scoring code.
- **Acceptance tests:** Ground truth as prediction scores exactly `1`; empty/corrupted outputs behave as specified; official DET/SEG/TRA tools used; report per-sequence scores and volumetric metrics separately.
- **Risks:** Arbitrary spheres around points would make SEG misleading; CTC gold segmentation is sparse; leaderboard parameters may be dataset-specific.
- **Primary sources:** [CTC evaluation methodology](https://celltrackingchallenge.net/evaluation-methodology/), [CTC SEG definition](https://public.celltrackingchallenge.net/documents/SEG.pdf).

## CT-106 — Visualization and export correctness

- **Status:** READY
- **Evidence:** Static overlays exist, but no automated axis/orientation or round-trip export test exists.
- **Dependencies:** CT-001.
- **File scope:** `scripts/plot_rule_based_overlay.py`, `src/celltrack_baseline/rule_based.py`, new landmark tests.
- **Acceptance tests:** An asymmetric `t/z/y/x` landmark maps to the correct slice and exported coordinate; anisotropic aspect ratio shown; deterministic ordering; duplicate/self/missing endpoints rejected; temporary-file plus atomic rename; explicit overwrite behavior.
- **Risks:** Coordinate flips may be dataset-specific; spreadsheet formula injection in user-controlled CSV text; partial writes.
- **Primary sources:** [OME-TIFF specification](https://docs.openmicroscopy.org/ome-model/6.2.1/ome-tiff/specification.html), [GEFF specification](https://liveimagetrackingtools.org/geff/latest/specification/).

## CT-107 — Packaging and reproducibility manifest

- **Status:** READY
- **Evidence:** No project packaging manifest, dependency lock, declared Python range, README, LICENSE, or CI test configuration was present at the audited root.
- **Dependencies:** Maintainer decision on supported Python versions and licenses.
- **File scope:** New `pyproject.toml`, README/LICENSE/CITATION as approved, CI configuration, dependency lock.
- **Acceptance tests:** Clean wheel build/install; imports and unittest pass outside source tree; optional backends separated; `pip check` passes; unsupported Python rejected by metadata; model versions/checksums recorded.
- **Risks:** Python 3.14 versus Ultrack constraints; native solver dependencies; licensing decisions cannot be inferred.
- **Primary sources:** [Python packaging specifications](https://packaging.python.org/en/latest/specifications/), [Ultrack repository](https://github.com/royerlab/ultrack).

## CT-108 — Safe archive and output handling

- **Status:** READY
- **Evidence:** ZIP inputs are streamed without extraction and subprocess calls use argument lists, but writers overwrite directly and no uniform resource preflight exists.
- **Dependencies:** Maintainer overwrite policy and expected maximum dataset dimensions.
- **File scope:** New `src/celltrack_baseline/io.py`, submission/report writers, focused security tests.
- **Acceptance tests:** Canonical path checks; no archive extraction; member/dimension/decompressed-size limits; restrictive temporary files; atomic replace; cleanup on failure; no pickle; logs exclude pixel data, credentials, and full environment dumps.
- **Risks:** Symlink escapes, ZIP bombs, output/input aliasing, and behavior changes for scripts that currently overwrite reports.
- **Primary sources:** [Python `zipfile` documentation](https://docs.python.org/3/library/zipfile.html), [OME-NGFF specification](https://ngff.openmicroscopy.org/latest/).
