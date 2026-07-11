# Biohub Cell Tracking Implementation Plan

Status: draft, based on public resources verified on 2026-07-08.

## Alignment Next Steps

1. Acquire the official Kaggle competition data or confirm that it is already mounted locally.
2. Mirror the official baseline repository as the reference implementation for I/O, metrics, and submission validation.
3. Build a data inventory before modeling: image shapes, chunking, voxel scale, intensity statistics, graph node/edge counts, division counts, sparsity, and submission schema.
4. Establish a rule-based baseline first: multi-scale 3D DoG/blob detection, physical-unit non-max suppression, Hungarian frame-to-frame linking, and division logic off by default until CV proves it helps.
5. Add learning only after the baseline has a trusted local CV harness: sparse-label 3D detection, learned edge scoring, and global graph optimization.
6. Validate every output graph before scoring or submission: axis order, node IDs, edge endpoints, duplicate edges, degree constraints, node count ratio, and CSV schema.

## Sources Checked

- Kaggle competition page: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
- Official baseline repo: https://github.com/royerlab/kaggle-cell-tracking-competition
- Official baseline metrics: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- Cell Tracking Challenge datasets: https://celltrackingchallenge.net/3d-datasets/
- Cell Tracking Challenge data terms and annotation description: https://celltrackingchallenge.net/datasets/ and https://celltrackingchallenge.net/annotations/
- GEFF format: https://github.com/live-image-tracking-tools/geff
- TracksData: https://github.com/royerlab/tracksdata
- Motile: https://github.com/funkelab/motile
- Ultrack: https://github.com/royerlab/ultrack
- Trackastra: https://github.com/weigertlab/trackastra

Chrome note: Chrome plugin control was requested, but no Chrome-backed browser was available in this environment. Public web verification was used instead. Kaggle pages that require a logged-in account cannot be fully verified from this environment.

## Confirmed Competition Constraints

The official baseline repository describes the competition as sparse-ground-truth 3D+time cell tracking. Images are OME-Zarr volumes with dimensions `(T, Z, Y, X)`. Training tracks are paired `.geff` graphs where nodes carry approximate cell centers `(t, z, y, x)`, edges link cells over time, and a division is one source node with two outgoing target edges.

Spatial scale from the baseline repository is `(z, y, x) = (1.625, 0.40625, 0.40625)` microns per voxel. Node matching in the metric uses centroid distance with a maximum of 7 um in physical units.

The scoring target is graph-level, not just detection-level. The baseline metrics define:

- Edge Jaccard: `TP / (TP + FP + FN)` after sparse-aware node matching.
- Adjusted edge Jaccard: edge Jaccard penalized by excess predicted node count using alpha `0.1`.
- Division Jaccard: division TP/FP/FN with tolerance around split timing and daughter-lineage coverage.
- Final score: adjusted edge Jaccard plus `0.1 * division_jaccard`.

This makes detection quality the largest lever, but over-detecting can hurt through the node-count penalty and annotated-region false positive edges.

## Data Acquisition Plan

### Official Kaggle Data

Target local layout:

```text
data/
  kaggle/
    raw/
      biohub-cell-tracking-during-development.zip
    manifests/
    interim/
    predictions/
```

Current instruction for this workspace: keep the Kaggle archive zipped. Do not extract it. Use ZIP central-directory inspection and targeted streaming reads only. If a modeling step later needs random access to individual zarr chunks, implement an archive-aware adapter or create derived, explicitly documented caches from selected members; do not run a blanket unzip.

Preferred acquisition into the workspace:

```bash
kaggle competitions download \
  -c biohub-cell-tracking-during-development \
  -p data/kaggle/raw
python3 scripts/inspect_zip_archives.py \
  data/kaggle/raw/biohub-cell-tracking-during-development.zip \
  --json-output reports/kaggle_zip_inspection.json \
  --markdown-output reports/kaggle_zip_inspection.md
```

Manual step likely required: the Kaggle competition rules must be accepted by a logged-in Kaggle account before the API or web download will work. If `~/.kaggle/kaggle.json` is absent, expired, or not authorized for the competition, download must be done manually from the Kaggle web UI, then placed under `data/kaggle/raw/`.

Kaggle data must be treated as the source of truth for:

- dataset names expected in submission rows
- train/test split names
- sample submission columns and row conventions
- hidden-test target distribution

### Official Baseline Code

Clone as a dependency/reference rather than rewriting metric code:

```bash
git clone https://github.com/royerlab/kaggle-cell-tracking-competition external/kaggle-cell-tracking-competition
cd external/kaggle-cell-tracking-competition
uv sync
```

If cloning is not allowed in the execution environment, vendor only the needed APIs after reviewing license compatibility:

- `tracking_cellmot.io.open_dataset`
- graph save/load helpers
- `tracking_cellmot.metrics.evaluate`, `evaluate_datasets`, `summarise`
- submission writer/reader if present

### Cell Tracking Challenge External Data

Use CTC data as auxiliary pretraining/validation data, not as a substitute for Kaggle CV. It differs in imaging modality, annotation format, cell type, scale, and evaluation rules.

Highest-priority CTC candidates:

- `Fluo-N3DL-DRO`: developing Drosophila embryo, light-sheet, voxel size roughly `(x, y, z) = (0.406, 0.406, 2.03)` um, training size about 5.8 GB. This is the closest public analog by developmental embryo and lateral voxel scale.
- `Fluo-N3DH-CE`: C. elegans developing embryo, training size about 3.1 GB. Useful for lineage/division behavior but different voxel scale and organism.
- `Fluo-N3DL-TRIC`: developing Tribolium embryo cartographic projection, training size about 20.6 GB. Useful only if storage/time allows; note projection scale caveats.
- `Fluo-N3DH-CHO`: nuclei, small training size about 98 MB. Useful for smoke tests and fast conversion checks.
- `Fluo-N3DH-SIM+`: simulated nuclei, training size about 3.1 GB. Useful for controlled segmentation/tracking pretraining.
- `Fluo-C3DL-MDA231`: small 3D real microscopy, training size about 182 MB. Useful for detector robustness, less relevant biologically.

Suggested staged download:

```bash
scripts/download_ctc_archives.sh --recommended
python3 scripts/inspect_zip_archives.py data/ctc/raw/*.zip \
  --json-output reports/ctc_zip_inspection.json \
  --markdown-output reports/ctc_zip_inspection.md
```

The CTC site says training ZIPs include original image data and available reference annotations; test ZIPs do not include reference annotations. It also states usage restrictions and forbids cloning/rehosting the datasets or annotations. Keep download scripts as links, not redistributed copies.

## ZIP-Only Data Profiling Requirements

For every Kaggle dataset, derive and cache metadata or small tabular manifests from the ZIP without archive-wide extraction:

- image metadata: dataset name, path, zarr format, axes, shape, chunks, dtype, min/max, intensity quantiles, scale, timepoint count
- per-timepoint image profile: foreground proxy, intensity quantiles, number of z slices with signal, artifacts/outlier slices
- graph metadata: node count, edge count, estimated total node count if present, node attributes, edge attributes
- node table: `dataset`, `node_id`, `t`, `z`, `y`, `x`, physical coordinates, in-degree, out-degree, matched track id if recoverable
- edge table: `dataset`, `source_id`, `target_id`, `source_t`, `target_t`, delta time, physical displacement
- division table: source nodes with out-degree 2, daughter distances, daughter separation, timepoint distribution
- split table: embryo/sample grouping, time-window grouping, and whether a dataset has divisions

For CTC external data, convert into a Kaggle-compatible intermediate schema:

- stream TIFF time series members and `TRA`/tracking annotations from ZIPs
- convert mask centroids to `(t, z, y, x)` in voxel and physical units
- convert lineage tables into graph edges
- record original scale/time-step metadata
- keep a `source_dataset` field so CTC-derived samples cannot leak into Kaggle CV reporting

## Preliminary Analysis Plan

Use the local skills as follows:

- `explore-data`: profile image and graph structure before any model work.
- `analyze`: compare baseline variants, threshold sweeps, and CV/LB behavior.
- `statistical-analysis`: characterize displacement distributions, density shifts, intensity normalization, outliers, and division rarity.
- `data-visualization`: create CV-vs-LB scatter, metric decomposition plots, density-over-time plots, and edge failure overlays.
- `validate-data`: run pre-submission QA on graph validity, metric calculations, and conclusions.
- `scientific-problem-selection`: keep a risk matrix for sparse labels, domain shift, division rarity, and leaderboard overfitting.

Initial EDA outputs:

- `reports/data_profile.md`
- `reports/plots/intensity_quantiles_by_dataset.png`
- `reports/plots/node_count_by_time.png`
- `reports/plots/displacement_histograms_um.png`
- `reports/plots/division_counts_by_time.png`
- `reports/plots/cv_lb_scatter.csv`
- `data/kaggle/manifests/dataset_manifest.parquet`
- `data/kaggle/manifests/node_manifest.parquet`
- `data/kaggle/manifests/edge_manifest.parquet`

Critical checks:

- Verify no `(z, y, x)` axis swap by overlaying ground-truth nodes on image slices.
- Verify physical-unit conversion before any radius threshold or metric use.
- Verify that graph node IDs are stable only within dataset unless the submission requires global uniqueness.
- Verify edge endpoints always exist in node rows.
- Verify duplicate source-target edges are removed before evaluation.
- Verify the official sample submission schema exactly before any Kaggle submission.

## Baseline Pipeline

### Detector V0: Multi-Scale 3D DoG

Goal: reproduce or exceed the public rule-based direction before learning.

Algorithm:

1. Load one timepoint volume with zarr chunk-aware I/O.
2. Apply robust normalization per dataset/timepoint: percentile clipping, background subtraction, optional z-slice normalization.
3. Run anisotropic 3D Difference-of-Gaussians over multiple physical scales.
4. Use scale-space maxima, not a single fixed scale.
5. Non-max suppress in physical units; start with 4-7 um exclusion radius.
6. Calibrate threshold by CV edge score and predicted node count ratio.
7. Emit integer voxel centroids after keeping internal float/physical coordinates for scoring/linking.

Parameters to sweep:

- DoG sigma in physical units
- z/y/x anisotropy handling
- local maximum window size
- score threshold or top-k per timepoint
- minimum distance between detections
- optional border/slice artifact filters

### Linker V0: Hungarian t-to-t+1

Goal: produce a coherent graph without expensive global optimization.

Algorithm:

1. For each adjacent timepoint pair, generate candidate edges within a physical radius.
2. Score continuation edges by distance, local density, intensity continuity, and optional detection confidence.
3. Solve linear assignment with Hungarian matching.
4. Enforce at most one predecessor and one normal continuation successor.
5. Start with no gap closing. Add gap closing only if CV shows missed detections dominate.
6. Start with divisions disabled or highly conservative. Public anecdotal evidence suggests naive division edges may raise CV but lower LB.

Parameters to sweep:

- max linking radius, e.g. 6-14 um
- distance cost scale
- detection confidence penalty
- time-specific radius scaling
- gap closing max delta t
- whether to allow births/deaths freely

### Division V0: Conservative Forks

Enable only after detector/linker are stable.

Candidate criteria:

- parent has no high-confidence continuation
- two daughters are close to parent in physical space
- daughter-daughter separation is plausible and not too large
- local intensity/morphology suggests a split or paired nuclei
- both daughter lineages remain linkable for at least one future step

Because the final score weights division Jaccard by only 0.1, division logic should be precision-first. A false positive fork in annotated regions can hurt more than a missed rare division during early iterations.

## Learned Pipeline

### Detector V1: Sparse-Supervised 3D Heatmap

Base architecture:

- 3D U-Net or MONAI UNet on patches
- input: normalized 3D timepoint or short temporal stack
- target: Gaussian heatmaps centered on annotated nodes
- loss: positive-label heatmap loss plus ignore masks for unlabeled background; avoid treating all unlabeled cells as negatives
- optional pseudo-labels from DoG V0 filtered by high confidence and consistency across time

Key design:

- Train with sparse supervision: annotated nodes are positives; unlabeled tissue is mostly unknown.
- Use validation node recall, edge Jaccard, and adjusted edge Jaccard, not voxel loss, as the model-selection signal.
- Match detection density to estimated total node count when available.

### Linker V1: Learned Pairwise Edge Scoring

Candidate generation remains radius/kNN based. Score candidates with:

- physical displacement and direction
- detection confidence
- local image patch embeddings at source/target
- local density
- parent/daughter morphology features
- short motion history when available

Possible models:

- gradient-boosted classifier for quick iteration
- MLP on engineered features
- transformer/cross-attention node-pair scorer following the official baseline repository
- Trackastra-style association if segmentation masks are available or can be produced

### Graph Optimizer V1

After edge scores are calibrated, solve a global graph problem:

- at most one predecessor per node
- one continuation successor or two daughter successors
- no cycles
- births/deaths allowed with learned penalties
- gap closing optional with explicit cost
- use `motile`, `tracksdata` ILP tools, or a custom min-cost flow if the graph remains simple

This stage should be introduced only after the local Hungarian baseline is trusted, because global solvers can hide metric regressions behind many interacting costs.

## Cross-Validation Strategy

The competition train set appears very small by embryo count. Use multiple CV views:

1. Leave-one-dataset/embryo-out if there are two or more named embryos.
2. Time-window holdout inside each embryo to detect temporal overfitting.
3. Annotation-subset bootstrap for metric stability, while clearly marking it as weaker than embryo holdout.
4. CTC-only validation for pretraining sanity, never as Kaggle score evidence.

Track every run with:

- config hash
- detector parameters
- linker parameters
- division parameters
- per-sample edge TP/FP/FN
- per-sample division TP/FP/FN
- node recall
- predicted node count and node count ratio
- adjusted edge Jaccard
- public LB if submitted

Submission rule:

- Submit only when local CV improves on the previous best or when testing one explicitly documented uncertainty.
- Record `(CV, LB)` for every submission to detect CV/LB divergence.
- Treat graph-level add-ons such as gap closing and divisions as suspect until CV/LB scatter supports them.

## Submission Serialization

Required output table should follow the official sample submission exactly. The expected columns are:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Node rows:

- `row_type = node`
- fill `dataset,node_id,t,z,y,x`
- leave `source_id,target_id` empty

Edge rows:

- `row_type = edge`
- fill `dataset,source_id,target_id`
- leave `node_id,t,z,y,x` empty unless the sample submission requires placeholders

Validation before writing:

- IDs unique within each dataset.
- All edge endpoints exist in node rows.
- No duplicate edge rows.
- Edges point forward in time.
- No impossible in-degree/out-degree except permitted divisions.
- Integer `t,z,y,x` in bounds.
- Dataset names match Kaggle test names exactly.
- CSV has deterministic row order and stable `id`.

## Risk Matrix

| Risk | Impact | Mitigation |
|---|---:|---|
| Axis or scale mismatch | Catastrophic | Overlay nodes on volumes; unit tests for physical distance; compare known scale from baseline repo |
| Sparse labels treated as full negatives | High | Positive-unlabeled losses; ignore unlabeled background; use metric-aware validation |
| Detector overproduces nodes | High | Track node count ratio and adjusted edge score; threshold by sample/timepoint |
| CV overfits two embryos | High | Leave-embryo-out, time-window holdout, and CV/LB scatter |
| Division false positives | Medium-high | Keep divisions off by default; add precision-first fork detector only after baseline |
| CTC domain shift | Medium | Use CTC for pretraining/smoke tests; never use CTC as Kaggle CV proxy |
| Large data I/O bottleneck | Medium | Zarr chunk-aware reads, patch caches, lazy profiling, per-timepoint manifests |
| Invalid submission graph | High | Dedicated schema/graph validator and dry-run scorer before submission |

## First Implementation Milestones

1. `scripts/download_kaggle.sh`: checks for Kaggle credentials, prints manual instructions if unavailable, and never assumes data exists.
2. `scripts/download_ctc.sh`: downloads only selected CTC ZIPs by explicit dataset name.
3. `scripts/profile_kaggle_data.py`: builds image/graph manifests and validates axes/scale.
4. `scripts/convert_ctc_to_graph.py`: converts CTC tracking annotations into the intermediate node/edge schema.
5. `scripts/run_dog_baseline.py`: emits detections and detection diagnostics.
6. `scripts/run_hungarian_linker.py`: builds prediction graphs from detections.
7. `scripts/evaluate_predictions.py`: wraps official metrics and writes per-sample/run summaries.
8. `scripts/validate_submission.py`: validates CSV schema and graph consistency.
9. `configs/baseline_dog.yaml`: all detector/linker/division parameters in one reproducible file.
10. `reports/run_log.csv`: persistent CV/LB tracking table.
