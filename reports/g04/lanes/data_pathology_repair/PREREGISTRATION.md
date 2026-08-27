# G-04 data-pathology repair — preregistration

Date: 2026-08-03  
Node: `G04-DATA-PATHOLOGY-REPAIR-V1`  
Compute: local CPU only

## Objective

Measure whether deterministic acquisition-pathology handling improves the frozen
official graph objective on the Prompt-1 base.  In the absence of a repository
artifact literally named “Prompt-1”, that base is fixed here as the exact APNCC
CV57 graph: combined score `0.8815220699597315`, adjusted edge Jaccard
`0.8815220699597315`, raw edge TP/FP/FN `34320/2459/1948`, and divisions off.
The pathology census covers every raw video in the organizer archive (train and
test); score deltas use only the fixed 57-video CV57 population.

## Immutable inputs and conventions

- Raw archive: `data/kaggle/raw/biohub-cell-tracking-during-development.zip`.
- APNCC caches/fragments: `kaggle_kernel/bhcv57out_v3/`.
- Exact scorer: the pinned organizer `metrics-fix` checkout already attested in
  `reports/g02/screening/p0_parity.md`.
- Raw shape and scale must be exactly `(100,64,256,256)` and
  `(1.625,0.40625,0.40625)` micrometres in `T,Z,Y,X` / `Z,Y,X` order.
- Every image-space calculation uses the organizer convention
  `volume[:, ::4, ::4]`, equivalently spatial stride `(1,4,4)`.  Block means are
  forbidden.
- The existing candidate contract is physical radius `10.0 um`, cap `8`, with
  deterministic distance/node-ID ordering.

## D0 — exact frozen-frame census

A transition `t -> t+1` is frozen iff the decoded `uint16` arrays are exactly
equal, i.e. every voxel difference is zero.  The archive has one canonical,
fixed-codec Zarr chunk per frame; candidate equalities may be found from the ZIP
central directory but must be confirmed by byte equality of the encoded Zarr
chunks.  Identical canonical chunk bytes imply identical decoded arrays.  Report
transition count, affected video count, family/split counts, run lengths, and
the APNCC candidate/selected-edge/official-score accounting on CV57 frozen
transitions.

Frozen repair `F1` is label-blind and parameter-free: only on confirmed frozen
transitions, form one-to-one edges between detections with exactly equal original
voxel coordinates; install only node-disjoint pairs and remove their conflicting
same-transition raw edges.  No nearest-neighbour radius is tuned.  Score `F1`
alone against the unchanged APNCC base.

## D1 — whole-volume global-translation census

For every non-frozen adjacent transition, read both full frames, take the exact
stride view `(Z, Y/4, X/4)`, robustly centre each volume, apply a separable Hann
window, and estimate the single 3-D translation that aligns `t+1` to `t` by
phase correlation.  Use integer-voxel registration on the strided isotropic grid;
each registered-grid voxel is exactly `1.625 um` on all three axes.  Record the
signed shift, Euclidean magnitude, pre/post non-wrapping overlap NCC, and whether
the shift is pool-breaking.

The pool-breaking threshold is fixed before observation at magnitude `>10.0 um`.
No percentile, family-specific threshold, confidence threshold, or shift sweep is
allowed.  Frozen transitions have exactly zero shift and are excluded from the
phase-correlation distribution while being reported separately.

## D2 — frame-level registered candidate repair

Global repair `G1` applies the one registered transform once to the complete
target frame geometry before candidate generation.  It is not a per-candidate
feature.  For each pool-breaking transition, target detection coordinates are
translated into the source-frame coordinate system, then the unchanged physical
`10.0 um` / cap-`8` candidate gate and unchanged scorer are applied.  Node IDs,
times, and final coordinates written for the evaluator remain byte-identical to
the original APNCC detections; only association geometry changes.  Unflagged
transitions are graph-identical to APNCC.

Report candidate count changes and matched-GT continuation-edge reachability
before/after.  Cross-check the fit-only GA-R12REACH bank by partitioning its 456
recoverable `10-12 um` correct edges into transitions with registered magnitude
`<=10` and `>10 um`, and report how many move inside `10 um` after correction.

Score `G1` alone and `F1+G1` on exact CV57.  There is no threshold selection from
CV57 labels.  The official GT is never edited, including suspected bad
annotations; incompatible frozen/jump-transition GT edges are reported as an
audit only.

## D3 — downsampling parity

Trace the pinned organizer training/inference loaders and every APNCC feature
path.  If all base paths already use exact striding, the downsampling repair is
an identity with exact score delta zero.  If any Prompt-1 path uses a block mean,
replace only that operation with exact striding, freeze the code change, and
score it as ablation `S1` before combination.  No other normalization or model
change is allowed.

## D4 — crop-level computed-segmentation `wrfeat`

Owner amendment (2026-08-03): reopen Lane B only as the new named mechanism
`CROP-COMPUTED-SEGMENTATION-WRFEAT-V1`.  The terminal whole-frame Cellpose
design (measured `43.460/42.774 s` per frame and projected to as much as
`195.586 T4-h`) remains closed.  Reuse its authenticated Cellpose3 `cyto3`
weight, deterministic inference settings, feature names, finite voxel-cell
solidity, and explicit `COMPUTED_SEGMENTATIONS` provenance, but change the unit
of segmentation to one detection-centred crop.

- Crop shape is fixed before observation at `(16,32,32)` raw voxels in Z/Y/X,
  matching the existing APNCC crop.  Boundary crops use deterministic zero
  padding and retain their origin in the full frame.
- Segment every crop independently.  Select the label containing the crop's
  detection centre, otherwise the deterministic nearest physical centroid
  under Lane B's already frozen `30 um` mapping bound.  No GT or score enters
  mask selection.
- Checkpoint-native `wrfeat12` retains its exact Trackastra channel order.
  Its intensity channel uses the whole-frame p1/p99.8 normalization, never a
  crop-specific feature normalization.  Centroids and XY border distance are
  evaluated in original full-frame coordinates; translation-invariant shape
  fields come from the crop label.  Computed masks are never called authentic.
- Report mask-empty, unmatched, duplicate, and selected-mask crop-boundary
  contact rates.  Boundary contact is evidence of truncation, not silently
  accepted proof of faithful whole-object morphology.
- First run one fixed real-frame CPU timing/fidelity pilot on
  `train:44b6_40c45f5a`, timepoint 0, with all 243 banked detections.  Project
  the observed per-detection and per-frame cost to CV57 before any full run.
  Full CV57 is admissible only if the crop pipeline is materially cheaper than
  the terminal whole-frame upper bound and every detection obtains finite,
  nonzero `wrfeat`; otherwise stop and report the measured blocker.
- If admitted, run the already authenticated Trackastra computed-`wrfeat`
  adapter once on exact CV57 and report its exact official score and delta to
  APNCC.  It is an observation-distinct feature-engineering arm, not a claim
  that APNCC itself consumes Trackastra features.  No threshold or crop-size
  sweep is permitted.

## Gates and stop conditions

- Fail closed on raw shape/scale, cache identity, scorer hash, CV57 identity, or
  baseline reproduction drift.
- No GPU, Kaggle, submission, RunPod, training, or parameter search.
- Report every exact score, including negative and zero deltas.
- Promotion requires strict exact combined-score improvement over
  `0.8815220699597315`; otherwise bank the pathology census and reject the repair.
- A7's candidate-level phase/template mechanism remains terminal.  This node is
  separately named because its estimand is a label-blind frame transform applied
  before candidate generation.
