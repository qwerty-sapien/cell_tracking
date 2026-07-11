# Biohub Cell Tracking Research Decomposition

Status: research-backed execution plan, updated 2026-07-08.

This document continues the strategy review beyond the DoG plus Hungarian baseline. The premise is that the Biohub competition is a sparse-label 3D+time graph reconstruction task: predict cell-center nodes and directed temporal edges, including division forks. The best path is therefore not a single detector plus local assignment, but a candidate graph with learned or multi-hypothesis evidence and a global lineage solver.

## Source Takeaways

- The official baseline repository describes sparse 3D+time tracking with GEFF graphs: nodes are approximate `(t, z, y, x)` cell centers, edges link cells, and divisions are one node linked to two nodes at the next timepoint. Its baseline is already learned: a 3D U-Net detector plus a cross-attention transformer linker trained under sparse supervision. Source: <https://github.com/royerlab/kaggle-cell-tracking-competition>
- The official metric matches predicted nodes to ground-truth nodes within 7 microns, then scores edge Jaccard, an adjusted edge Jaccard penalizing excess total node count, and division Jaccard with small final weight. Source: <https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md>
- GEFF is a graph-exchange format for spatial graph data, with Python readers/writers that convert to common in-memory graph structures. This confirms that graph validity and graph IO are core infrastructure, not peripheral details. Source: <https://github.com/live-image-tracking-tools/geff>
- Ultrack, published in Nature Methods in 2025, explicitly targets the failure mode of dense/crowded tissues where committing to one segmentation per frame compounds errors. It combines multiple segmentation hypotheses, temporal consistency, biological constraints, and an ILP; it was validated on zebrafish, fly, and nematode embryo recordings. Source: <https://www.nature.com/articles/s41592-025-02778-0>
- The earlier Ultrack arXiv preprint reports large-scale 3D tracking using segmentation selection with ultrametric contour maps and a faster ILP formulation, including competitive Cell Tracking Challenge performance with or without deep learning. Source: <https://arxiv.org/abs/2308.04526>
- The Malin-Mayor et al. Nature Biotechnology paper is the closest template for sparse point annotation plus learned features plus global optimization: train center heatmaps and backward movement vectors, build a candidate graph, then solve a binary-forest ILP with division constraints and learned costs. Source: <https://www.nature.com/articles/s41587-022-01427-7>
- The Linajea MICCAI/arXiv extension adds a learned division detector and structured SVM tuning for ILP weights on C. elegans whole-embryo lineages. Source: <https://arxiv.org/abs/2208.11467>
- Trackastra learns pairwise associations in temporal windows, accounts for divisions, and can link greedily or with ILP via Motile. This is a practical second-stage association model once candidate nodes are reasonable. Source: <https://arxiv.org/abs/2405.15700> and <https://github.com/weigertlab/trackastra>
- OrganoidTracker 2.0, published in Nature Methods in 2025, supports the learned edge/division probability idea and adds calibrated error probabilities through graph-context marginalization. For Kaggle, the useful lesson is not manual curation, but probability calibration and confidence-aware edge/division costs. Source: <https://www.nature.com/articles/s41592-025-02845-6>
- The Cell Tracking Challenge Nature Methods comparison found that methods using learning strategies or global spatiotemporal context performed better than more local approaches. Source: <https://www.nature.com/articles/nmeth.4473>
- Cellpose and Cellpose3 are useful proposal generators, not full solutions for this task. Cellpose provides broad cellular segmentation and 3D extension; Cellpose3 improves noisy, blurry, and undersampled microscopy images. Sources: <https://www.nature.com/articles/s41592-020-01018-x> and <https://www.nature.com/articles/s41592-025-02595-5>
- StarDist-3D is relevant as a nuclear proposal source because it handles 3D star-convex nuclei, anisotropic voxels, and NMS. Source: <https://arxiv.org/abs/1908.03636>

## Ranked Strategy

1. Learned sparse point detector plus backward displacement plus global binary-forest solver.

This should be the final architecture target. It directly matches the official annotation type and metric: point centers, temporal edges, and division forks. It also avoids requiring dense segmentation masks that the competition does not provide.

2. Ultrack-style multi-hypothesis proposal selection plus global optimization.

This is the strongest robust near-term system, especially if the learned detector is not ready or if classical/Cellpose/StarDist proposals recover cells missed by one model. It is particularly well supported by the Biohub/Royer context and dense 3D embryo evidence.

3. Learned association and division scoring on candidate graphs.

This should be layered on top of either strategy 1 or 2. It is most useful once Node F1 is not the bottleneck and edge/division errors dominate.

DoG plus Hungarian remains a floor and a debugging tool. The current local notes already report poor sparse recall for the bright DoG baseline, so it should not absorb much more effort except as a proposal source and sanity check.

## Mandatory Task Decomposition

### 1. Metric-Faithful Local Evaluation

Why it is mandatory: without a trusted local evaluator, every modeling decision is underdetermined. Public leaderboard feedback is too sparse and too late, and frame-level CV leaks temporal structure.

Necessary subtasks:

- Load official train `.zarr` and paired `.geff` graphs without axis ambiguity.
- Convert GEFF nodes into a canonical table with `dataset`, `node_id`, `t`, `z`, `y`, `x`, physical coordinates, in-degree, and out-degree.
- Convert edges into a canonical table with endpoint coordinates, displacement, time delta, and division/fork flags.
- Wrap the official metric implementation rather than reimplementing it from memory.
- Build video-level CV splits, not random frame splits.
- Reproduce official counts on known toy or baseline examples.
- Implement prediction validation: schema, duplicate node IDs, missing endpoints, duplicate edges, illegal degrees, coordinate bounds, invalid times, and excessive node count.
- Generate failure overlays for matched TP, FP, FN edges and division neighborhoods.

Done when:

- A prediction CSV can be round-tripped through local graph IO and official metrics.
- A small train run reports edge TP/FP/FN, division TP/FP/FN, adjusted edge Jaccard, division Jaccard, and final score.
- Visual overlays confirm the coordinate convention and scale conversion.

### 2. Data Audit and Coordinate Calibration

Why it is mandatory: the metric uses physical distance, the images are anisotropic, and an axis/scale mistake can make a plausible detector score zero.

Necessary subtasks:

- Inventory every train/test dataset: shape, chunks, dtype, axis order, timepoints, intensity quantiles, foreground extent, and estimated total node count if available.
- Profile labels: nodes per timepoint, edge displacement distribution, division count distribution, birth/death rates, and annotated-region sparsity.
- Estimate reasonable radii in microns for NMS, linking, gap edges, and division motifs.
- Verify whether sparse labels tend to be bright local maxima, dim centers, membrane-like features, or mixed appearances.
- Make per-dataset normalization diagnostics for depth attenuation, z-slice artifacts, and time drift.
- Define physical-coordinate helpers used by every detector, linker, and metric.

Done when:

- There is a manifest for every dataset and a compact report of intensity, density, displacement, and division statistics.
- Ground-truth points overlay correctly on image volumes at multiple timepoints and depths.
- A written table specifies default physical radii and their empirical source.

### 3. Baseline Replication and Error Attribution

Why it is mandatory: the baseline is the control experiment. It tells us whether the bottleneck is detection recall, node overcount penalty, edge assignment, or division logic.

Necessary subtasks:

- Keep DoG plus Hungarian as a reproducible script with fixed configs.
- Add threshold and polarity sweeps, including bright blobs, dark centers, and contrast-normalized variants.
- Sweep physical NMS radius and top-k-per-frame caps.
- Evaluate with divisions disabled and enabled separately.
- Produce per-dataset error attribution: node recall around GT, unmatched GT edges, wrong matched edges, extra predicted node penalty, and division TP/FP/FN.
- Compare against the official learned baseline if compute and dependency installation allow.

Done when:

- We have a stable floor submission and a local CV table showing which errors dominate.
- DoG proposals are retained only if they add recall not already covered by better proposal sources.

### 4. High-Recall Candidate Node Generation

Why it is mandatory: global optimization cannot recover a cell if no plausible node exists near it. Candidate recall sets the upper bound for edge and division score.

Necessary subtasks:

- Classical proposals: multi-scale DoG/LoG, h-maxima, anisotropy-aware local maxima, watershed seeds, z-normalized variants, and foreground filtering.
- Learned proposals: 3D U-Net or MONAI U-Net center heatmap trained from sparse points with Gaussian targets.
- Segmentation proposals: StarDist-3D, Cellpose/Cellpose3 nuclei mode, watershed over restored/normalized images, and optional SAM-derived masks only if they pass 3D sanity checks.
- Proposal fusion: physical-coordinate NMS with source-specific confidence, radius, and provenance.
- Candidate density calibration: match total predicted node count to the adjusted-Jaccard penalty while keeping high GT recall.
- Store all candidates in a table with source, score, radius/uncertainty, physical coordinates, and local image features.

Done when:

- Candidate node recall near sparse GT is high at a tolerable overcount ratio.
- Removing any proposal source has measured impact, so the ensemble is justified.

### 5. Sparse-Supervised Detector

Why it is mandatory: Biohub labels are sparse points, and the strongest paper template trains from sparse point annotations rather than dense masks.

Necessary subtasks:

- Generate Gaussian center targets in physical units, not raw voxel units.
- Build ignore masks around annotated centers so unlabeled tissue is not treated as background.
- Sample training patches around annotated tracks plus hard negative/uncertain regions.
- Train a 3D U-Net with outputs for center heatmap, radius/uncertainty, optional foreground, and optional division state.
- Add temporal input context if memory allows: centered short stacks, backward/forward neighboring frames, or late feature fusion.
- Validate by post-NMS candidate recall and adjusted edge score, not voxel MSE alone.
- Calibrate heatmap thresholds per dataset or per intensity regime.

Done when:

- The learned detector beats classical proposals on sparse GT recall at comparable node count.
- Inference can run on Kaggle free compute or in a documented chunked local mode.

### 6. Movement Vector and Edge Prior Model

Why it is mandatory: nearest-neighbor distance is weak when cells move collectively, divide, or crowd. A backward vector is division-friendly because daughters point back to the parent.

Necessary subtasks:

- Build targets for backward displacement from each annotated node at `t` to its parent/same-cell node at `t-1`.
- Train vector outputs only where sparse lineage labels define the correspondence.
- Normalize vector targets in physical units and record voxel conversion only at IO boundaries.
- At inference, attach predicted backward vector to each candidate node.
- Generate edge candidates by agreement between predicted parent location and actual previous-frame candidates.
- Combine vector agreement with Euclidean radius and local density to avoid missing links when vectors are uncertain.

Done when:

- Candidate edge recall covers most GT edges with far fewer alternatives than pure radius/kNN.
- Edge feature histograms separate GT edges from hard negatives better than distance alone.

### 7. Candidate Graph Construction

Why it is mandatory: every global solver, Trackastra/GNN scorer, and division module consumes the same candidate graph. Bad graph construction silently caps all downstream models.

Necessary subtasks:

- Build nodes from fused candidates with unique per-dataset IDs and stable provenance.
- Connect adjacent timepoints with radius/kNN edges using physical coordinates.
- Add optional gap edges over 2-3 frames only after measuring missed-detection errors.
- Add candidate division motifs: one parent at `t`, two children at `t+1` or within tolerated timing.
- Add features per node: detector score, source, intensity patch stats, foreground confidence, local density, depth, time, uncertainty.
- Add features per edge: distance, vector agreement, intensity/crop similarity, source confidence, local candidate competition, motion-field residual.
- Add features per division motif: daughter-parent distances, daughter separation, angle, daughter score symmetry, parent division score, future survivability.
- Enforce basic graph limits so candidate counts remain solvable.

Done when:

- GT edge and division recall of the candidate graph is high before optimization.
- Candidate graph size is bounded enough for solver experiments on whole videos or documented blocks.

### 8. Global Lineage Optimization

Why it is mandatory: the task rewards coherent graph reconstruction. Local Hungarian assignment cannot enforce long-range consistency, division constraints, or rescue ambiguous detections.

Necessary subtasks:

- Choose a solver backend: Motile/ILP first if dependencies fit; SCIP fallback if Gurobi is unavailable; min-cost flow for a restricted no-division baseline.
- Define binary variables for selected nodes, selected edges, track starts, track ends, and divisions.
- Add constraints: selected edge implies selected endpoints, at most one parent per node, at most two children per selected parent, no merges, no more-than-binary split, optional gap-edge consistency.
- Add costs: node confidence, edge cost, division prior, track-start penalty, track-end penalty, false-positive node penalty proxy, and dataset-specific density calibration.
- Support blockwise solving with temporal/spatial overlap if full videos are too large.
- Validate selected graph against official metric and biological constraints.
- Run ablations: local assignment vs min-cost flow vs ILP; no divisions vs divisions; no gap vs gap.

Done when:

- Global solve improves edge Jaccard or division precision over Hungarian at fixed node candidates.
- Solver output is deterministic and finishes within Kaggle or local compute limits.

### 9. Ultrack-Style Multi-Hypothesis Branch

Why it is mandatory: this is the fastest robust alternative if sparse-supervised training underperforms or takes too long. It also gives a strong proposal-selection mechanism for mixed candidate sources.

Necessary subtasks:

- Convert classical, Cellpose, StarDist, and learned heatmap outputs into foreground and contour/hypothesis maps usable by Ultrack-like selection.
- Generate multiple parameterizations per dataset: thresholds, gamma/intensity transforms, smoothing scales, and foreground filters.
- Run Ultrack or a reduced Ultrack-inspired selector on short train clips first.
- Convert selected segments to cell-center nodes and temporal edges compatible with GEFF/submission CSV.
- Tune for official edge score, not segmentation IoU.
- Compare three modes: classical-only hypotheses, learned-only hypotheses, and combined hypotheses.

Done when:

- The branch produces valid graph predictions on full train/test samples.
- Combined hypotheses recover GT nodes/edges that any single proposal source misses.

### 10. Learned Association and Division Reranking

Why it is mandatory: once candidate recall is acceptable, the main remaining leaderboard loss is wrong edges and wrong division forks.

Necessary subtasks:

- Build positive edge examples from GT edges and hard negative examples from nearby wrong candidates.
- Build division examples from GT fork neighborhoods and hard false forks.
- Train a small practical model first: gradient boosting or MLP over engineered features.
- Add image-crop models second: pair/triplet 3D crop CNN, Trackastra-style temporal transformer, or graph neural network.
- Calibrate probabilities with held-out videos, using reliability curves for link and division scores.
- Feed calibrated probabilities as costs into the global solver.
- Evaluate whether a learned scorer improves edge Jaccard and division Jaccard beyond distance/vector costs.

Done when:

- Learned edge scores improve ranking of true edges over hard negatives.
- Learned division scores improve precision without causing broad false-positive fork penalties.

### 11. Division-Specific Logic

Why it is mandatory: the final metric includes division Jaccard, and division mistakes can also corrupt lineage connectivity. But because division weight is small, division prediction should be precision-first until edge tracking is strong.

Necessary subtasks:

- Extract GT divisions as out-degree-2 nodes and compute timing/daughter geometry distributions.
- Build candidate forks from parents with two plausible child links.
- Train or hand-code a conservative division score from parent morphology, daughter geometry, vector evidence, and future lineage survival.
- Respect metric tolerance around split timing but avoid creating forks merely to chase rare events.
- Add solver costs/constraints so divisions are selected only when they beat continuation plus separate births.
- Validate division TP, FP, FN separately from edge score.

Done when:

- Division-enabled runs improve final CV score, not just raw division recall.
- False-positive forks in annotated regions remain rare.

### 12. Probability Calibration and Error Confidence

Why it is mandatory: the official score is a graph score with sparse labels. Poorly calibrated thresholds can overproduce nodes, overlink, or hallucinate divisions even if raw model rankings are good.

Necessary subtasks:

- Calibrate detector thresholds to the adjusted node penalty and per-dataset density.
- Calibrate edge probabilities using video-held-out reliability curves.
- Calibrate division probabilities separately because divisions are rare and class imbalance is severe.
- Track confidence by dataset/time/depth/local density to detect failure regimes.
- Optional later: implement OrganoidTracker-style local marginalization to estimate context-aware link confidence and prune low-confidence edges.

Done when:

- Chosen thresholds are derived from CV curves, not one-off visual inspection.
- Public/private submission choices can be explained by metric components.

### 13. Kaggle Deployment and Reproducibility

Why it is mandatory: a good local method is useless if it cannot run within Kaggle's environment or cannot produce a valid submission reliably.

Necessary subtasks:

- Package dependencies or implement installation in the Kaggle script without relying on fragile network steps where possible.
- Add modes for smoke, train-CV, and submit.
- Chunk zarr reads and model inference to avoid memory blowups.
- Cache only derived artifacts allowed by the competition and workspace policy.
- Produce submission CSV with exact schema and stable row ordering.
- Download outputs and logs after each run; store metrics/config summaries locally.
- Keep every leaderboard submission tied to a config hash, code snapshot, and local CV result.

Done when:

- The same config runs locally on a small split and on Kaggle test.
- Each completed Kaggle run produces a valid submission and a retrievable log.

## Execution Order

1. Finish metric/data audit and coordinate overlays.
2. Run official baseline and current DoG floor through the same evaluator.
3. Build high-recall candidate node and edge graph generation.
4. Implement restricted global solve without divisions.
5. Train sparse center heatmap and backward movement vector model.
6. Feed learned node/vector costs into the global solve.
7. Add conservative division candidates and division scoring.
8. Add Ultrack-style hypothesis selection as a parallel branch or proposal source.
9. Add learned association/division reranking after candidate recall is no longer the main bottleneck.
10. Calibrate thresholds per video-held-out CV and submit only configs with explained metric gains.

## Immediate Next Tasks for This Repo

1. Create the official metric harness around `external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py`.
2. Generate dataset/node/edge/division manifests from the Kaggle ZIP or mounted Kaggle data.
3. Produce GT overlays for at least three train videos and several timepoints/depths.
4. Re-run the current rule-based baseline after polarity, scale, and coordinate checks.
5. Build the candidate graph table abstraction before writing any new model code.

