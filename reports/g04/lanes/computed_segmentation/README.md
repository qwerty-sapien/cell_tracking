# G-04 Lane B: computed 3-D segmentations

Status: **whole-frame CUDA timing v2 passed but is Kaggle-infeasible; the
owner-requested fixed crop-level wrfeat v1 pilot is terminal on incomplete,
boundary-truncated features; no production feature run followed**.

## Crop-level owner amendment (2026-08-03)

`CROP-COMPUTED-SEGMENTATION-WRFEAT-V1` reuses the authenticated backend and
native feature contract below but changes segmentation work to one fixed
`(16,32,32)` APNCC-size crop per detection.  The real-frame all-243 pilot is
`FAIL_CROP_INTERFACE_INCOMPLETE`: 231 finite nonzero matches, 12 empty/unmatched
crops, and 199 selected masks touching the crop boundary.  It took 483.937 CPU
seconds for one frame and projects to 766.233 CPU-hours for CV57.  The fixed
crop is therefore neither faithful whole-object morphology nor a viable cheap
CPU production path.  No full feature production or Trackastra score ran.
Evidence is under `crop_wrfeat_v1/`; the combined pathology report is
`../data_pathology_repair/TERMINAL_REPORT.md`.

Every mask and derived feature in this lane is labelled
`COMPUTED_SEGMENTATIONS`. These are model-computed labels from raw intensities;
they are **not** authentic H3 masks, organizer-provided masks, or a claim that
the competition supplied segmentations.

## Authenticated backend

- Backend: Cellpose3 `cyto3`, `cellpose==3.1.1.2` (the final Cellpose3 release),
  with `do_3D=True`, grayscale channels `[0,0]`, and OME-Zarr anisotropy.
- Weight: `assets/computed-segmentation-cellpose3-v1/cyto3`, 26,566,255 bytes,
  SHA-256 `2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c`.
- License: BSD-3-Clause text from the pinned source tag,
  SHA-256 `fa908286fef07e1e8acd05adcdcd7bbb8c7ca4c02f563a4722dad28b94ce0b49`.
- Full receipt: `assets/computed-segmentation-cellpose3-v1/PROVENANCE.json`.
- Inference is offline-only. `scripts/g04_computed_segmentation.py` requires
  both expected hashes before importing or constructing the model and never
  invokes Cellpose's automatic downloader.

The local carrier uses Python 3.13.7 / NumPy 2.4.6. Cellpose 3.1.1.2 declares
`numpy<2.1`; installing it normally therefore attempted an unavailable source
build. The package was installed `--no-deps` against the already-tested binary
carrier. This declared-bound exception is recorded in every preflight and
segmenter provenance block; a successful import/model/pilot is not represented
as upstream support for that dependency combination.

## Feature contract

The checkpoint matrix `features` / `trackastra_native_wrfeat12` is never
overloaded. Its exact native order is:

1. `equivalent_diameter_area`
2. `intensity_mean`
3. the complete 3x3 `inertia_tensor` in row-major order
4. `border_dist`

Equivalent diameter and inertia remain in voxel-coordinate units. Intensity is
normalized exactly like the authenticated Trackastra source: independently for
each float32 frame using p1/p99.8 and `(x-p1)/(p99.8-p1+1e-8)`. `border_dist`
uses only the XY frame boundary, cutoff 5 pixels, and region maximum.

A separate `descriptive_regionprops` matrix carries the requested morphology:
voxel volume, physical major/intermediate/minor ellipsoid axes, elongation,
raw intensity mean/std/min/max, voxel-cell 3-D convex-hull solidity, equivalent
diameter, and border distance. V2 defines solidity as foreground voxel-cell
volume divided by the physical 3-D convex-hull volume of every voxel's eight
half-voxel-offset corners. It has no undefined-value sentinel; each object also
records the affine dimension of its voxel centers. This separate matrix is
suitable for division decoding and does not change checkpoint channel meanings.

Detections are assigned first by the label containing their rounded ZYX point,
then by deterministic nearest physical centroid. The mapping-only pilot curve
at radii 5/10/15/20/30 matched 192/220/234/240/243 of 243 detections. Production
therefore preregisters radius 30; the largest required pilot distance was
21.709619. Match mode, physical distance, and label are retained per row, and
duplicate label assignments are explicitly allowed and counted (97 at radius
30) rather than misrepresented as unique segment identities. Unmatched rows
remain `matched=false` with NaN features. Production publication fails closed
unless every detection is matched and every native and descriptive row is
finite and nonzero. Exact evidence is in `MAPPING_RADIUS_CURVE.json`.

## Verified local preflight

`PREFLIGHT.json` authenticates the runtime, asset bytes, one extracted raw
OME-Zarr, and its existing point cache. The checked video is
`train:44b6_40c45f5a`: shape `[100,64,256,256]`, physical ZYX scale
`[1.625,0.40625,0.40625]`, and 37,929 detections over frames 0-99. Preflight is
`READY`; this is an interface/provenance result, not a score.

## Real-volume pilot result

`PILOT_44b6_40c45f5a_T000.json` is a completed
`PASS_INTERFACE_NONEMPTY_COMPUTED_SEGMENTATION` on one real
`[64,256,256]` frame. It produced 171 masks and 224,324 foreground voxels in
881.161 CPU seconds. At the original radius-15 diagnostic, 234/243 detections
matched; the separately sealed radius curve establishes complete 243/243 pilot
coverage at the production radius 30. The label NPZ is explicitly
`PILOT_PARTIAL_ONE_FRAME`, SHA-256
`9aa3ce1fe4cf4f358010935b8cdfe6f6ef57e3166e6c556d527d963d6076abac`,
and cannot satisfy production coverage.

The CPU measurement projects to 24.477 h per 100-frame video and 1,395.171 h
for CV57's 5,700 frames. It is therefore an interface result, not a viable
production route. `KAGGLE_GPU_PRODUCTION_PLAN.json` defines the CV57-first CUDA
timing gate, offline runtime carrier, atomic shard worker, and hard deadline
rules. The pilot itself used no CUDA or Kaggle action; the later raw-only timing
kernel was launched by the root coordinator and is monitored separately.

Timing v1 attached the organizer competition source and the already-published
private runtime dataset only. Its pure-stdlib Step 0 authenticated all 30
remote runtime files plus the exact three raw inputs for
`train:44b6_24264f12`, timepoint 0 before pip or scientific imports. It then
failed at 75.597 s in the combined region-feature finite guard, so it produced
no timing bound.

The exact-raw/config CPU reproduction isolated three planar objects (labels
26, 42, and 99; 47, 18, and 42 voxels) whose native equivalent diameter,
intensity mean, and inertia tensor are finite while scikit-image 0.26 legacy
center-hull solidity is `inf`. The failed GPU receipt did not persist labels,
so those IDs are explicitly CPU-scope evidence; the exact field/mechanism is
device-independent. See `TIMING_V1_NONFINITE_DIAGNOSIS_V2.json`.

The changed-signature v2 timing package at
`kaggle_kernel/g04-computed-segmentation-timing-v2` keeps the exact raw-only
Step 0 and two repeats, uses voxel-cell solidity, and reports exact GPU labels,
bounds, and affine dimensions for every subdimensional/legacy-nonfinite object.
It separately reports model initialization, the second repeat as steady-state,
the maximum repeat as worst-case, projected T4 GPU-hours for CV57 (5,700
frames) and held-fit105 (10,500 frames), plus planning-quota feasibility and a
mandatory fresh authoritative quota reconciliation. It reads no GEFF, learned
detections, targets, metrics, or submissions. V2 is prepared but unlaunched.

## Commands

Bounded one-frame real-data pilot (never production-manifest eligible):

```bash
.venv313/bin/python scripts/g04_computed_segmentation.py pilot \
  --zarr data/derived/t03_fold0_fit10_images_v1/train/44b6_40c45f5a.zarr \
  --identity train:44b6_40c45f5a \
  --cache-root reports/g02/fitcache_117/bh9out/cache \
  --pilot-timepoint 0 \
  --weight assets/computed-segmentation-cellpose3-v1/cyto3 \
  --weight-sha256 2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c \
  --license-file assets/computed-segmentation-cellpose3-v1/CELLPOSE_LICENSE.txt \
  --license-sha256 fa908286fef07e1e8acd05adcdcd7bbb8c7ca4c02f563a4722dad28b94ce0b49 \
  --pilot-labels reports/g04/lanes/computed_segmentation/pilot_44b6_40c45f5a_t000_labels.npz \
  --pilot-report reports/g04/lanes/computed_segmentation/PILOT_44b6_40c45f5a_T000.json
```

Production per-video run (all timepoints represented in its detection cache):

```bash
.venv313/bin/python scripts/g04_computed_segmentation.py run \
  --zarr /path/to/train/44b6_ID.zarr \
  --identity train:44b6_ID \
  --cache-root reports/g02/fitcache_117/bh9out/cache \
  --weight assets/computed-segmentation-cellpose3-v1/cyto3 \
  --weight-sha256 2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c \
  --license-file assets/computed-segmentation-cellpose3-v1/CELLPOSE_LICENSE.txt \
  --license-sha256 fa908286fef07e1e8acd05adcdcd7bbb8c7ca4c02f563a4722dad28b94ce0b49 \
  --output reports/g04/lanes/computed_segmentation/features/train__44b6_ID.npz
```

The run streams one frame at a time, atomically publishes a persistent
`*.computed_segmentations.zarr`, and writes the feature NPZ plus a JSON sidecar.
It does not hold a whole video's uint32 label volume in memory.
For a coordinator-authorized CUDA carrier, add `--device cuda`; the adapter
requires CUDA availability, fixes all RNG seeds, disables TF32, enables PyTorch
deterministic algorithms, and uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Any
unsupported nondeterministic operation fails closed. This implementation task
did not launch GPU compute.

The rejected 595 MB learned-detection cache is not a remote input and must not
be republished. `scripts/g04_build_segmentation_production_package.py` now
prepares an arbitrary root-supplied shard without selecting or freezing a shard
count. It refuses nonterminal or v1 timing and derives the hard per-frame bound
only as `ceil(max(two terminal v2 repeat seconds) * 1.25)`. The generated kernel reads
only organizer raw Zarrs plus the published runtime dataset and emits one
computed-object row per mask: label id/hash/count, centroid ZYX, native
wrfeat12, richer morphology, and center-affine dimension. It remains
fail-closed until a v2 timing run terminates PASS and the root then supplies a
shard; v1 cannot unlock this builder.

Build one future shard only after terminal v2 timing:

```bash
.venv313/bin/python scripts/g04_build_segmentation_production_package.py \
  --timing-result /path/to/G04_SEGMENTATION_TIMING.json \
  --timing-supervisor /path/to/G04_SEGMENTATION_TIMING_SUPERVISOR.json \
  --shard-manifest /root/supplied/SHARD.json \
  --competition-archive data/kaggle/raw/biohub-cell-tracking-during-development.zip \
  --kernel-id onebitbrain/ROOT_SUPPLIED_FUTURE_SLUG \
  --output /path/to/empty/package-directory
```

After all exact frames are downloaded, map the existing local learned
detections to deterministic nearest computed centroids at radius 30 and create
the Trackastra-compatible manifest:

```bash
.venv313/bin/python scripts/g04_assemble_segmentation_objects.py \
  --population-manifest /root/supplied/EXACT_POPULATION.json \
  --cache-root kaggle_kernel/g02-trackastra-native-cv57-cache-dataset-v1 \
  --shard-output /path/to/downloaded/shard-a \
  --shard-output /path/to/downloaded/shard-b \
  --output-dir reports/g04/lanes/computed_segmentation/assembled
```

The assembler requires exact `(identity,timepoint)` set equality, authenticates
every shard artifact, and rejects any unmatched, non-finite, or all-zero
wrfeat row. No concrete production package has been built or launched.

The Trackastra adapter consumes only the fixed production manifest:
`reports/g04/lanes/computed_segmentation/wrfeat_manifest.json`. Build it only
after all expected identities have complete NPZs:

```bash
.venv313/bin/python scripts/g04_computed_segmentation.py manifest \
  --output reports/g04/lanes/computed_segmentation/wrfeat_manifest.json \
  --expected-identities-file /path/to/exact_expected_identities.json \
  --feature-npz reports/g04/lanes/computed_segmentation/features/train__44b6_ID.npz \
  --feature-npz reports/g04/lanes/computed_segmentation/features/train__6bba_ID.npz
```

Manifest construction checks every NPZ hash and sidecar, rejects any unmatched
or non-finite feature row, and requires exact set equality with the expected
identity file. A pilot or partial artifact cannot pass this constructor.

## Verification

Focused tests cover native channel order, exact per-frame normalization,
physical morphology separation, deterministic mapping/unmatched failure,
explicit non-authentic provenance, exact production-manifest coverage, and
real Zarr preflight:

```text
The final local receipt includes feature, timing-v2 package, production,
legacy timing-v1 integrity, and Trackastra-adapter tests. No GPU or external
action is part of this verification.
```
