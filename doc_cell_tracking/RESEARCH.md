# Research Review: 3D+t Cell Tracking

## Scope and conclusion

This repository targets sparse cell-center graphs in `(t, z, y, x)` rather than full instance masks. Its production competition route currently copies a train GEFF graph after paired train/test Zarrs match by member name, uncompressed size, and ZIP CRC32. That is strong duplicate evidence for the current archive, not collision-resistant proof of byte identity. The classical fallback detects 3D points and links them; the Ultrack branch builds foreground/contour maps for short clips.

The most defensible near-term improvement is therefore a calibrated, deterministic linking baseline over fixed detections. Joint segmentation/tracking methods such as Ultrack become appropriate when the inputs contain genuine alternative segmentation hypotheses. No method reviewed below establishes biological lineage accuracy for this repository without representative held-out ground truth.

## Benchmarks and metrics

The Cell Tracking Challenge (CTC) separates related but non-identical questions:

- `DET` is a normalized acyclic-oriented-graph edit measure over detections.
- `SEG` is based on matched-object Jaccard overlap. It evaluates segmentation, which this repository's sparse-center fallback does not produce.
- `TRA` is a normalized graph-edit measure combining detection and linking errors.
- `OP_CSB = 0.5 * (DET + SEG)` and `OP_CTB = 0.5 * (SEG + TRA)`.
- The newer linking benchmark's `LNK`/`BIO` measures are more diagnostic when detections are fixed.

CTC gold segmentation for real 3D data is sparse and does not by itself establish full-volume boundary quality. Results should be reported per biological sequence, with sequence-level uncertainty; random frame splits leak temporal context. A synthetic association test is useful for software correctness but cannot support a CTC, biological, or state-of-the-art claim.

Primary and official sources:

- [CTC evaluation methodology](https://celltrackingchallenge.net/evaluation-methodology/)
- [CTC SEG definition](https://public.celltrackingchallenge.net/documents/SEG.pdf)
- [CTC Cell Linking Benchmark](https://public.celltrackingchallenge.net/documents/Cell%20Linking%20Benchmark.pdf)
- [CTC 3D+t datasets](https://celltrackingchallenge.net/3d-datasets/)
- [Cell Tracking Accuracy Measurement Based on Comparison of Acyclic Oriented Graphs](https://doi.org/10.1371/journal.pone.0144959)
- [An objective comparison of cell-tracking algorithms](https://doi.org/10.1038/nmeth.4473)
- [The Cell Tracking Challenge: 10 years of objective benchmarking](https://doi.org/10.1038/s41592-023-01879-y)

## Assignment and tracklet linking

Jaqaman et al. formulate two global linear-assignment stages: frame-to-frame linking followed by track-segment linking for gaps and other events. This is materially stronger than sorting candidate edges greedily. TrackMate documents its LAP trackers as stripped-down versions using physical-distance gates and a separate gap-closing stage; the simple tracker does not model splits or merges.

Repository implication:

1. Solve each adjacent-frame gated assignment globally with explicit unmatched choices.
2. Form tracklets.
3. Close gaps only between unmatched tracklet endpoints and starts.
4. Treat division as a separate one-to-two event; ordinary Hungarian assignment cannot encode it.
5. Keep merges forbidden unless the data model and metric explicitly require fusion.

Sources:

- [Robust single-particle tracking in live-cell time-lapse sequences](https://doi.org/10.1038/nmeth.1237)
- [Official TrackMate LAP tracker documentation](https://imagej.net/plugins/trackmate/trackers/lap-trackers)
- [TrackMate: An open and extensible platform for single-particle tracking](https://doi.org/10.1016/j.ymeth.2016.09.016)

## Bayesian and global graph methods

`btrack` combines motion prediction, optional normalized visual features, lost-track buffering, and global hypothesis selection. These are useful design references, but its primary biological validation does not prove performance for dense 3D embryos.

Min-cost flow is efficient for ordinary one-to-one associations with births and deaths. Coupled divisions, mutually exclusive segment hypotheses, and overlap constraints generally require a coupled flow, LP, or ILP formulation. That complexity should be added only after fixed-detection evaluation shows that local LAP is the bottleneck.

Sources:

- [Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach](https://doi.org/10.3389/fcomp.2021.734559)
- [btrack documentation](https://btrack.readthedocs.io/en/stable/)
- [Coupled minimum-cost flow cell tracking for high-throughput quantitative analysis](https://doi.org/10.1016/j.media.2010.07.006)
- [Conservation Tracking](https://doi.org/10.1109/ICCV.2013.364)

## Ultrack and segmentation uncertainty

Ultrack is not just a point linker. It consumes foreground and contour evidence, generates nested candidate segments, and jointly selects segmentation/tracking hypotheses with an integer program. It is a strong fit for crowded 3D tissue when boundary or instance hypotheses are genuinely ambiguous. It is not automatically a better replacement for the current sparse-point competition route.

Before interpreting an empty or poor Ultrack result, measure foreground recall, contour quality, segment candidate counts, physical size bounds, solver status, optimality gap, window overlap, runtime, and memory. Windowed solving improves scalability but is not whole-movie global optimization.

Sources:

- [Ultrack: pushing the limits of cell tracking across biological scales](https://doi.org/10.1038/s41592-025-02778-0)
- [Official Ultrack repository](https://github.com/royerlab/ultrack)

## Validated 3D lineage reference

TGMM jointly propagates hierarchical supervoxels and a Bayesian 3D Gaussian-mixture model, with temporal priors and local division correction. It was validated on large light-sheet embryo data and is a useful 3D lineage design reference. Its reported linkage accuracy is not CTC `TRA`; fluorescent nuclei also need to fit its modeling assumptions. The older native stack should not be introduced as a dependency without a focused comparison.

Sources:

- [Fast, accurate reconstruction of cell lineages from large-scale fluorescence microscopy data](https://doi.org/10.1038/nmeth.3036)
- [Official TGMM software page](https://www.janelia.org/lab/keller-lab/software/fast-accurate-reconstruction-cell-lineages-large-scale-fluorescence)

## 4D microscopy conventions

Array position must not be used as a substitute for metadata. OME-NGFF identifies axes, units, and coordinate transformations explicitly; OME-TIFF records dimension order and physical sizes separately. This repository's raw-array API requires canonical `T/Z/Y/X` metadata and normalizes recognized consistent spatial units to micrometers. Noncanonical axes must be transposed by a future explicit loader before entering the tracker. GEFF is the relevant directed graph exchange format for cell-center tracks and lineages.

Sources:

- [OME-NGFF paper](https://doi.org/10.1038/s41592-021-01326-w)
- [OME-NGFF specification](https://ngff.openmicroscopy.org/latest/)
- [OME-TIFF specification](https://docs.openmicroscopy.org/ome-model/6.2.1/ome-tiff/specification.html)
- [GEFF paper](https://doi.org/10.21105/joss.10143)
- [GEFF specification](https://liveimagetrackingtools.org/geff/latest/specification/)

## Recommended external evaluation set

Use fixed detections first, then end-to-end inputs where masks exist:

- `Fluo-N3DH-CHO`: small 3D smoke test.
- `Fluo-N3DH-SIM+`: controlled synthetic truth with divisions.
- `Fluo-N3DH-CE`: embryo lineages and frequent division.
- `Fluo-N3DL-DRO`: anisotropic developmental imaging.
- `Fluo-C3DL-MDA231`: real 3D cancer-cell morphology.

Keep the Kaggle competition metric, CTC metrics, and the repository's synthetic fixed-detection metric in separate result tables. They answer different questions.
