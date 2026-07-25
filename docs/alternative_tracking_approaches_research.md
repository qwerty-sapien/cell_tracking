# Alternative Cell-Tracking Approaches Research Ledger

Updated: 2026-07-08

> Superseded for learned-model selection and compute policy by
> [`reports/g02/model_landscape_20260724.md`](../reports/g02/model_landscape_20260724.md)
> and `reports/g02/compute_governance_v1.json`. The historical notes below retain the
> older `$25` policy only as provenance; they are not current launch authority.

Purpose: durable research notes for the Biohub Cell Tracking During Development Kaggle competition. This file is designed to survive `/clear` and let the next session continue from a concrete source-backed strategy.

## Current Alignment

- Budget policy remains local compute plus Kaggle free compute first. Paid GPU work is allowed only for a scoped deep-learning pilot whose measured extrapolated training cost is under `$25`; stop if the pilot implies the full experiment will exceed that cap.
- The Kaggle ZIP must remain zipped locally. Continue using ZIP streaming / central-directory inspection only.
- Current local baseline is a structural DoG plus Hungarian pipeline, but sparse-GT recall was zero in the initial smoke tests. That means we should treat it as an IO/submission/debug baseline, not as a serious scoring endpoint until coordinate/detection calibration improves.
- Kaggle free-compute rule-based kernel `onebitbrain/biohub-rule-based-dog-hungarian` completed successfully and produced `data/kaggle/kernel_output/submission.csv`. The run reported a valid CSV with 12,144 node rows, 9,399 edge rows, and 4 datasets.
- Browser/Chrome plugin surfaces were unavailable earlier in this session; research below used direct web access instead.
- Web-search checkpoint: 30 `search_query` searches completed. Context-budget polling was requested every 5 searches; `get_goal` was called at 8, 10, 15, 20, 25, and 30 searches, but it returned no active goal/budget object and no numeric remaining-token value.

## High-Level Conclusion

This competition is not purely a compute-brute-force contest. The public anecdote showing a rule-based pipeline reaching a strong LB, plus the official metric's emphasis on graph edge agreement, implies there is real alpha in data/coordinate correctness, high-recall center proposals, conservative node-count calibration, and better graph optimization.

The strongest realistic path under a small paid-compute budget is:

1. Fix/verify detection coordinates and sparse-GT alignment.
2. Generate higher-recall candidates with multi-source proposals.
3. Replace local Hungarian with a stronger graph/linking layer.
4. Add learned scoring only where it is cheap: edge/division reranking over detected nodes, not end-to-end giant 3D training.
5. Treat pretrained or small deep-learning models as a fourth branch only if a timed pilot extrapolates below `$25` total training cost.

Full 3D U-Net or Trackastra-style training may help, but should only be attempted after candidate recall and local metric infrastructure are trustworthy.

## Source-Backed Method Matrix

| Method / family | Core idea | Fit for this Kaggle task | Budget fit | Practical recommendation |
|---|---|---:|---:|---|
| DoG/LoG + Hungarian | Multi-scale blob proposals and local one-step assignment. | Medium as floor, low as final. | Excellent | Keep as smoke-test and proposal source. Do not over-invest until sparse-GT overlay is fixed. |
| LapTrack | Linear-assignment tracking with tunable metrics; examples include bright-spot detection and cell divisions. | High for fast upgrade over Hungarian. | Excellent | Try first among external linkers. It maps cleanly to centroid detections and may add gap/division handling cheaply. Source: <https://github.com/yfukai/laptrack> |
| btrack | Bayesian multi-object tracking with lineage use cases and C++ core. | Medium-high if package installs cleanly. | Good | Test as a global linker over our detections. Likely useful for lineage consistency and hypothesis handling. Source: <https://github.com/quantumjot/btrack> |
| Motile | Python global multi-object tracking by solving an optimization problem. | High conceptually. | Good if installable | Use if we need explicit constraints: one parent, up to two children, start/end penalties. Source: <https://github.com/funkelab/motile> |
| Linajea | Sparse point annotations, learned center/motion/division evidence, ILP lineage extraction. | Very high conceptually. | Medium/low for full replication | Borrow architecture ideas: center heatmap, backward displacement, division detector, learned costs. Full stack is likely too heavy for immediate Kaggle work. Sources: <https://arxiv.org/abs/2208.11467>, <https://github.com/funkelab/linajea> |
| Ultrack | Multi-hypothesis segmentation selection under temporal consistency and biological constraints; scales to large 3D+t data. | High if we can produce foreground/contour maps. | Medium | Use as an alternate branch only after fast candidate/linker work. Its key alpha is not one mask per cell, but multiple plausible hypotheses plus ILP/windowed solving. Sources: <https://github.com/royerlab/ultrack>, <https://ar5iv.labs.arxiv.org/html/2308.04526> |
| Trackastra | Transformer learns pairwise cell associations in temporal windows from detections and supports dividing cells. | High once detections are decent. | Medium | Check pretrained/installation viability. Training association on sparse edges may be cheaper than volumetric detector training. Sources: <https://arxiv.org/abs/2405.15700>, <https://github.com/weigertlab/trackastra> |
| Cell-TRACTR | DETR-style transformer that segments and tracks cells end-to-end, with published checkpoints for DeepCell and mother-machine data. | Medium conceptually, low direct fit | Medium/poor for full adaptation | Use as architectural evidence for attention-based tracking and division handling. Do not adapt directly until sparse-point/Kaggle graph infrastructure is solved, because the published setup targets 2D mask-style segmentation/tracking rather than sparse 3D center graphs. Sources: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013071>, <https://zenodo.org/records/14509424> |
| OrganoidTracker 2.0 | CNN center detection, learned link/division probabilities, min-cost flow, calibrated context-aware error probabilities. | High as design pattern. | Low for full tool integration | Borrow probability calibration and local-subgraph marginalization ideas; full GUI/manual workflow is not Kaggle-friendly. Sources: <https://www.nature.com/articles/s41592-025-02845-6>, <https://github.com/jvzonlab/OrganoidTracker> |
| CELLECT | Paused and archived under `_LATER/cellect/`. | Not currently active. | Deferred | Do not act on this direction unless explicitly asked and all conditions in `_LATER/cellect/PAUSE_RATIONALE.md` hold. |
| Segment Anything for Cell Tracking / SAM2 | Zero-shot SAM2-based tracking framework for 2D and large-scale 3D time-lapse microscopy. | Uncertain | Medium/low | Interesting proposal generator, but likely too expensive and brittle for whole-volume competition inference. Test only on short clips if installation is easy. Source: <https://arxiv.org/abs/2509.09943> |
| Cellpose / Cellpose3 | Generalist cell segmentation, including 3D usage/restoration improvements. | Medium as proposal source. | Medium | Use only as candidate proposal/restoration branch, not the main tracker. Sources: <https://www.nature.com/articles/s41592-020-01018-x>, <https://github.com/MouseLand/cellpose> |
| StarDist-3D | 3D star-convex object detection/segmentation for nuclei, handles anisotropic voxels and NMS. | Medium as proposal source. | Medium | Worth testing on small clips if dependencies do not derail Kaggle. Source: <https://arxiv.org/abs/1908.03636> |
| Micro-SAM | SAM-based microscopy segmentation/tracking tooling. | Low/medium for competition automation. | Medium/low | Useful for ideas or local inspection; likely not immediate whole-volume pipeline. Source: <https://github.com/computational-cell-analytics/micro-sam> |
| ByoTrack | Modular 2D/3D bio-object tracking framework with wavelet detector, Kalman/optical-flow linkers, wrappers, gap closing. | Medium | Good | Good source of lightweight linker/refiner ideas; may be too frame-memory-oriented for full 3D volumes. Source: <https://github.com/raphaelreme/byotrack> |
| ASCENT | Self-supervised contrastive embeddings for 3D neuron tracking. | Medium conceptually, domain mismatch | Medium | Borrow self-supervised embedding idea only. Source: <https://github.com/lu-lab/ascent> |
| TrackMate 7 | Fiji plugin integrating segmentation algorithms into tracking pipelines. | Low direct, high historical relevance | Poor for Kaggle automation | Use as conceptual reference only. Source: <https://www.nature.com/articles/s41592-022-01507-1> |

## Specific Potential Alphas

### Alpha 1: Coordinate and polarity correction before modeling

Current DoG smoke tests had zero sparse-GT recall, which is suspicious because the public anecdote reports a strong rule-based result. Before any deep training, verify:

- whether the target cells are bright local maxima, dark centers, rings, or membrane-like structures;
- whether the labels are in raw `(z, y, x)` voxel coordinates or require transform/scale interpretation;
- whether a train label overlays on the same apparent image structure after z/y/x flips/swaps;
- whether sparse GT points sit on dim centers while surrounding membranes/nuclei are bright.

This is the cheapest possible alpha. A single axis/polarity fix can matter more than a day of GPU training.

### Alpha 2: Detection-count calibration against adjusted edge Jaccard

The official metric penalizes excess predicted nodes. The rule-based public table improved mostly through detection changes, and the LB tracked CV closely for detection-style changes. We should tune node density per dataset, not just maximize candidate recall.

Concrete approach:

- produce per-dataset candidate count curves by threshold quantile / NMS radius;
- report sparse-GT recall at 7 microns plus predicted-node overcount ratio;
- choose operating points that preserve high GT recall without flooding nodes;
- submit only when local adjusted edge Jaccard improves, not merely raw recall.

### Alpha 3: Multi-source candidate fusion

No single proposal source is likely robust across 44b6 and 6bba. Fuse:

- classical multi-scale DoG/LoG;
- h-maxima on z-normalized and locally contrast-normalized volumes;
- dark-center or inverted-intensity variants;
- watershed/foreground connected components if cells appear ring-like;
- optional StarDist/Cellpose proposals on small validation clips.

Use physical-unit NMS and source-specific confidence. Store source provenance so ablations can identify which source is actually adding recall.

### Alpha 4: Replace Hungarian with candidate-graph optimization

Hungarian gives one local assignment per adjacent frame and cannot enforce a lineage graph. The task rewards edges and divisions, so add a graph layer:

- start with LapTrack or a custom min-cost flow over candidate nodes;
- add one-parent and up-to-two-children constraints;
- include start/end penalties and node-confidence penalties;
- only add gap closing after measuring missed-detection errors;
- use local density to adjust max displacement and birth/death costs.

This is likely the best near-term improvement per hour.

### Alpha 5: Learned edge/division scoring over detections, not full dense images

Trackastra and OrganoidTracker 2.0 both point to the same practical lesson: learning association probabilities over detected cells can outperform hand-tuned distance rules without requiring full end-to-end segmentation.

Cheap model progression:

1. engineered features + logistic regression / gradient boosting;
2. small MLP over distance, intensity patch stats, local density, detector source/score, and candidate competition;
3. optional tiny 3D crop-pair CNN only if Kaggle-free GPU is enough.

Feed calibrated scores into the graph solver. This should cost far less than training a full 3D detector and may directly improve edge Jaccard.

### Alpha 6: Backward-vector parent prediction

Linajea-style backward movement vectors are a good fit for daughters: each detected node can point back to its likely parent/same-cell previous position. Even a weak vector field can reduce candidate edges compared with pure radius/kNN.

Budget version:

- train only on sparse annotated edges;
- use small patches around GT nodes;
- predict physical displacement from child at `t` to parent at `t-1`;
- use vector agreement as an edge feature rather than relying on it exclusively.

### Alpha 7: Conservative division precision

Division Jaccard is important but appears lower-weighted than edge Jaccard. False positive forks can damage graph structure. Near term:

- keep division disabled or very conservative until edge score is strong;
- select divisions only when parent has two high-confidence child links, geometry is plausible, and daughters survive in subsequent frames;
- use division-specific GT statistics: daughter separation, parent-child distances, timing, intensity/morphology cues.

### Alpha 8: Budget-gated pretrained/small deep-learning branch

This is the fourth approach requested after reviewing Cell-TRACTR. The technical direction is valid, but it needs a cost gate because full transformer or large 3D detector training can burn GPU budget without fixing the current first-order bottleneck: candidate recall and graph scoring.

Preferred implementation order:

1. Official competition baseline first: sparse-supervised 3D U-Net detector plus cross-attention transformer linker from `external/kaggle-cell-tracking-competition`. It matches the Kaggle data model: sparse center nodes, temporal edges, and divisions.
2. Trackastra second: use pretrained `ctc` or `general_2d`/`general_2d_w_SAM2_features` only after producing usable candidate instance masks or synthetic labeled spheres around detections. Trackastra is attractive because it learns associations over detections instead of dense images.
3. Cell-TRACTR third: use its released checkpoints as evidence/reference, not as the first adaptation target. Its paper trained on a bacterial mother-machine dataset with V100 and on DynamicNuclearNet/DeepCell with A100, using 2D image/mask-style data; Kaggle is sparse 3D center-graph tracking.
4. Cheap learned scorer fallback: train logistic regression, gradient boosting, or a small MLP over candidate edge/division features. This is the most likely paid-compute-safe learned component.

Cost gate:

- Run a timed pilot on one fold or one representative video block for 30-60 minutes.
- Record GPU type, hourly price, wall time, epochs/iterations, validation metric, peak VRAM, and dataset fraction.
- Extrapolate full run cost with a 1.5x safety factor for failed starts, environment setup, validation, and reruns.
- Continue only if `estimated_total_cost <= $25` and local CV improves the relevant metric component.
- Stop immediately if candidate recall is still low; a better tracker cannot recover missing nodes.

Current rough paid-GPU envelope from Runpod on 2026-07-08:

| GPU | Listed pod price | `$25` wall-clock envelope | Use here |
|---|---:|---:|---|
| RTX A5000 24GB | `$0.27/hr` | ~92 h | cheapest small model, likely enough for learned scorer or small U-Net patches |
| L4 24GB | `$0.39/hr` | ~64 h | efficient inference/pilot option |
| A40 48GB | `$0.44/hr` | ~57 h | better memory for 3D patches |
| RTX 4090 24GB | `$0.69/hr` | ~36 h | fast 24GB option |
| L40S 48GB | `$0.99/hr` | ~25 h | strong 48GB option |
| A100 80GB | `$1.39-$1.49/hr` | ~17-18 h | use only if memory blocks 24-48GB GPUs |

Storage/data-transfer risk: the Kaggle archive is around 81 GB locally, so paid runs should avoid copying the whole archive unless needed. Prefer Kaggle free notebooks for training against mounted competition data, or paid pilots on small extracted/converted blocks only if competition rules and workspace policy allow it.

Sources:

- Cell-TRACTR paper and training hardware: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013071>
- Cell-TRACTR checkpoints: <https://zenodo.org/records/14509424>
- Trackastra pretrained models and CTC model: <https://github.com/weigertlab/trackastra>
- Runpod GPU pricing checked 2026-07-08: <https://www.runpod.io/pricing>

## What Not To Chase First

- Do not spend paid GPU money on B200/H200/H100-style brute-force training. The early bottleneck is likely pipeline correctness and graph formulation.
- Do not train a large 3D U-Net from scratch before label overlays and candidate recall are sane.
- Do not integrate full GUI-centric tools like TrackMate or OrganoidTracker as primary Kaggle inference pipelines.
- Do not run whole-volume SAM/SAM2 blindly. It may be valuable on curated clips, but whole-competition inference risk is high.
- Do not optimize division prediction before reliable node and edge reconstruction.

## Immediate Implementation Queue

1. Build a local coordinate/polarity diagnostic notebook/script:
   - overlays sparse GT on raw z-slices;
   - plots bright, dark, LoG, inverted-LoG response at GT locations;
   - sweeps `(identity, xflip, yflip, xyflip, yxswap)` and any physical-scale mapping assumptions.

2. Build a threshold/NMS grid runner:
   - inputs: zipped Kaggle train data;
   - outputs: candidate recall at 7 microns, node overcount ratio, local edge score if possible;
   - tracks configs in W&B only if setup is already frictionless.

3. Add a LapTrack or LapTrack-inspired linker experiment:
   - use current candidate CSVs;
   - compare against Hungarian on the same candidates;
   - test gap closing and division handling only after basic links improve.

4. Test install viability on Kaggle-free:
   - `laptrack`;
   - `btrack`;
   - `motile`;
   - `trackastra`;
   - `ultrack`;
   - do this as a diagnostic kernel, not a full submission kernel.

5. If candidate recall remains poor:
   - implement sparse-supervised 3D U-Net center heatmap on Kaggle GPU;
   - train on short clips/patches;
   - validate by candidate recall and adjusted edge score, not voxel loss.

## Research Notes By Source

### Official Kaggle Baseline

The Royer Lab baseline repository describes the task as sparse 3D+time tracking with GEFF graphs: cell center nodes `(t, z, y, x)`, temporal edges, and divisions as one node linking to two nodes at the next timepoint. Their baseline is a learned 3D U-Net detector plus a cross-attention transformer linker under sparse supervision. This is important: the official benchmark itself says sparse supervision is expected, and a learned linker is a reasonable baseline.

Source: <https://github.com/royerlab/kaggle-cell-tracking-competition>

### Cell Tracking Challenge

The Cell Tracking Challenge remains the right external dataset/benchmark reference. It includes 2D+time and 3D+time datasets, segmentation/tracking benchmarks, and a newer linking-only benchmark. It is useful both for data and for understanding which algorithm families generalize.

Source: <https://celltrackingchallenge.net/>

### Ultrack

Ultrack’s key idea is multi-hypothesis segmentation under uncertainty. It avoids committing to one segmentation and uses temporal consistency plus biological constraints. The arXiv/ECCV method explicitly reports scalability to very large 3D+t data, competitive Cell Tracking Challenge performance, and a faster ILP formulation. For this Kaggle, the most relevant ideas are hypothesis ensembles, windowed solving, and perturbing contour maps to avoid oversegmentation artifacts.

Sources:

- <https://github.com/royerlab/ultrack>
- <https://ar5iv.labs.arxiv.org/html/2308.04526>

### Trackastra

Trackastra learns pairwise associations over a temporal window from detections and explicitly handles dividing cells. It avoids dense image processing by operating on detection context. This is attractive under budget if our detections improve: train or fine-tune a linker rather than training a full detector.

Sources:

- <https://arxiv.org/abs/2405.15700>
- <https://github.com/weigertlab/trackastra>

### OrganoidTracker 2.0

OrganoidTracker 2.0 combines neural networks for center detection, link probabilities, and division probabilities with min-cost flow and calibrated error estimates. Its most transferable Kaggle insight is context-aware link confidence: a low local link probability may become high-confidence when surrounding alternatives are excluded. We can approximate this with local subgraph marginalization or calibrated graph costs even without full OrganoidTracker integration.

Sources:

- <https://www.nature.com/articles/s41592-025-02845-6>
- <https://github.com/jvzonlab/OrganoidTracker>

### CELLECT (paused)

The CELLECT-specific research note was moved to
`_LATER/cellect/research/alternative_tracking_approaches_excerpt.md`. Do not act on
it unless explicitly asked and the resume conditions in
`_LATER/cellect/PAUSE_RATIONALE.md` hold.

### Segment Anything for Cell Tracking

The 2025 arXiv paper proposes a zero-shot cell tracking framework using SAM2, claiming competitive results for 2D and large-scale 3D time-lapse microscopy without finetuning. This is promising but risky for the current budget: SAM2-style inference can be expensive and may not map cleanly to all volumes. Treat it as a proposal source to test on short clips, not a first-line full-pipeline bet.

Source: <https://arxiv.org/abs/2509.09943>

### LapTrack / btrack / Motile

These are the most immediate low-cost linking upgrades:

- LapTrack: linear assignment particle tracking with tunable metrics, examples for bright spots and dividing cells. Source: <https://github.com/yfukai/laptrack>
- btrack: Bayesian multi-object tracking with lineage use cases and automatic lineage determination examples. Source: <https://github.com/quantumjot/btrack>
- Motile: global multi-object tracking optimization package. Source: <https://github.com/funkelab/motile>

Recommendation: try these before large learned models, because they directly target the graph bottleneck and have much lower compute risk.

### Cellpose / StarDist / Micro-SAM

These are primarily candidate generators:

- Cellpose is a widely used generalist cell segmentation method with 3D usage. Source: <https://www.nature.com/articles/s41592-020-01018-x>
- StarDist-3D targets 3D microscopy nuclei with anisotropic voxels and star-convex shape priors. Source: <https://arxiv.org/abs/1908.03636>
- Micro-SAM provides SAM-based microscopy segmentation and tracking tools. Source: <https://github.com/computational-cell-analytics/micro-sam>

Use them only if they improve candidate recall or provide useful masks/contours for Ultrack-style selection.

## Unresolved / Continue After `/clear`

- `Cell-TRACTR` source is now verified. It is relevant scientifically but not a direct first implementation target for this Kaggle task because it is end-to-end 2D segmentation/tracking, while the competition uses sparse 3D center graphs.
- Trackastra has pretrained `ctc` support for 2D+3D CTC linking, plus general 2D models. Still need to test whether synthetic masks around sparse detections are enough for useful Kaggle association scores.
- Need test package install/runtime constraints for `laptrack`, `btrack`, `motile`, `trackastra`, and `ultrack` inside Kaggle notebooks.
- Kaggle kernel status/output is resolved: completed successfully, output downloaded to `data/kaggle/kernel_output/submission.csv`, log downloaded to `data/kaggle/kernel_output/biohub-rule-based-dog-hungarian.log`.
- Need update the implementation plan after the coordinate/polarity diagnostic produces a nonzero sparse-GT recall baseline.

## Recommended Next Session Prompt

Continue from `docs/alternative_tracking_approaches_research.md`. First, check the Kaggle kernel status/output. Then implement the coordinate/polarity diagnostic and threshold/NMS grid runner over the zipped Kaggle train archive without extraction. Prioritize getting nonzero sparse-GT recall before integrating LapTrack or any learned method.
