# G-04 standing global registration + crop-wrfeat v2 — preregistration

Date: 2026-08-03  
Node: `G04-GLOBAL-JUMP-STANDING-CROP-WRFEAT-V2`  
Expensive resource: one bounded Kaggle T4 timing/coverage pilot only

## Objective and hard stops

The graph objective remains the exact official adjusted edge/division score.  The
already measured APNCC result (`0.8815220699597315 -> 0.8867644202595942`) is
evidence for the named APNCC realization only.  It is not an assumed delta for
TemporalUNet3D-v32, a detection blend, Trackastra, or a synthetic-division child.
Each newly authenticated base must be scored both without and with the standing
registration transform on the same detections and frozen scorer.

No full computed-wrfeat run may launch in this node.  It remains closed unless
the bounded pilot passes both the coverage and live-budget gates below.  RunPod
is forbidden.  A pending/running v32 shard or submission owns the serial lane and
blocks this pilot until terminal reconciliation.

## G0 — base-agnostic standing registration contract

- Source: the hash-pinned whole-volume census under
  `reports/g04/lanes/data_pathology_repair/`.
- A transition contributes its exact integer phase shift only when the frozen
  `shift_magnitude_um > 10.0` rule marked it `pool_breaking`; all other steps are
  exactly zero.  No threshold or confidence tuning is permitted.
- Candidate coordinates for frame `t` equal original detection coordinates plus
  the cumulative target-aligning shift through frame `t`.  One constant video
  offset may be added for nonnegative storage; it cancels from all relative
  geometry.
- Original coordinates remain the only coordinates used to sample image/UNet
  features and the only coordinates written to evaluator graph rows.  Registered
  coordinates are used for candidate topology, displacement/distance features,
  positional geometry, and the association solve.
- The transform occurs exactly once after detections exist and before any edge or
  division candidate construction.  Synthetic-division integration consumes the
  registered parent candidate bank; it never applies a second transform.
- APNCC, v32, and synthetic-division children use the same interface and emit a
  provenance receipt naming the census SHA, event count, and coordinate-view
  contract.  A missing receipt fails closed.
- The immutable v32 reproduction remains unmodified until its unregistered base
  is authenticated.  Its registered arm is a distinct replay and must report its
  own exact score and candidate-topology delta.

Local validation: unit tests must prove cumulative composition, unchanged
original/evaluator coordinates, transform-once enforcement, source-order
invariance, and identical output when a video has no pool-breaking transition.

## C0 — real-object geometry diagnosis and deterministic crop choice

Design evidence is restricted to the existing computed full-frame Cellpose label
volume for `train:44b6_40c45f5a`, frame 0.  These are explicitly
`COMPUTED_SEGMENTATIONS`, not organizer masks or ground truth.

For every unique full-frame object and every detection mapped by the frozen
contains-else-nearest-centroid rule, report voxel and physical bounding-box
extents, equivalent diameter, detection-to-centroid offset, and the crop half
extent required to contain the complete selected object plus a `2.0 um` context
margin on every axis.  Compare these distributions with `(16,32,32)`.

Choose the smallest fixed even crop from this ladder whose per-axis half extent
covers the empirical p99 detection-anchored requirement:

- Z: `16,24,32,48,64`
- Y/X independently: `32,48,64,80,96,128,160,192,256`

If the selected shape exceeds `(48,128,128)`, stop before GPU: the design crop is
not a bounded replacement.  Re-centering is allowed only if the diagnosis shows
that the p95 detection-to-centroid offset alone exceeds the `2.0 um` margin.  In
that case the frozen recenter is the intensity-weighted centroid of positive
signal inside the original `(16,32,32)` anchor crop, clipped to at most `4.0 um`
from the original detection.  No label-derived production anchor is allowed.

## C1 — genuine batched GPU implementation

Cellpose 3.1.1.2's public list/5-D `eval` path loops over volumes serially, so it
does not satisfy this node.  The implementation must concatenate corresponding
YX, ZY, and ZX plane stacks across equal-shaped detection crops, call the frozen
Cellpose network on those plane batches, split the flow outputs back per crop,
and run the unchanged 3-D dynamics independently per crop.  It must retain:

- authenticated cyto3 weights and package/license provenance;
- `do_3D=True`, `diameter=30`, `anisotropy=4`, threshold 0, min size 15;
- whole-frame Trackastra p1/p99.8 intensity normalization for wrfeat;
- original-frame border distance and evaluator coordinates;
- deterministic batch order and identical crop membership for any batch size.

Required local tests: batch size one matches the serial segmentation contract;
multi-crop batching preserves order, shapes, and independent label namespaces;
CPU fixture fallback remains supported; malformed/missing CUDA fails closed.

## C2 — bounded real T4 pilot

Frozen frames:

1. `train:44b6_40c45f5a`, frame 0 (design-frame audit only),
2. `train:44b6_40c45f5a`, frame 50 (independent 44b6 gate),
3. `train:6bba_debd7bfa`, frame 0 (independent 6bba gate).

All detections in each frame are processed.  Network plane batch size is 64;
detection crops are grouped in deterministic batches of at most 32.  The pilot
has a 30-minute wall-clock kill, may use at most one T4, and cannot score a graph
or open any held/CV57 outcome.

Coverage passes only if, on the two independent frames jointly:

- at least 98% of detections are matched to finite, nonzero wrfeat;
- selected-mask boundary contact is at most 10%;
- empty-mask crops are at most 2%; and
- no schema, provenance, NaN/Inf, crop-order, or output-shape violation occurs.

Throughput uses the slower of (a) the independent-frame p95 seconds/frame and
(b) total pilot steady seconds divided by detections then multiplied by the
exact CV57 detection count.  Add measured initialization and 10% contingency.
The report must state T4-hours for CV57 and held-fit105+CV57.

The full-run budget gate passes only if the conservative CV57 projection is
strictly below the owner-authenticated remaining weekly T4 quota after all active
or preregistered v32 shard reservations.  Passing records eligibility only; this
node never launches the full run.  Any failed coverage or budget clause terminates
the crop-wrfeat realization without a full score.

## Marginal-value and crowding checks

The prior negative was structural but movable: `(16,32,32)` supplied almost no
margin around a configured 30-pixel-XY object, and CPU per-detection execution did
not test the requested GPU batching mechanism.  This node changes both limiting
factors directly.  Its realistic ceiling is not an assumed graph delta: first it
must establish complete real wrfeat coverage and affordable production.  The
alpha is the two-view integration seam—raw coordinates for feature sampling and
registered coordinates for topology—while GPU crop batching itself is necessary
engineering beta.  No score or promotion claim follows from throughput alone.
