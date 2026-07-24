# Learned-model landscape: bounded local and Kaggle candidates

Date: 2026-07-24  
Status: compact point-context and crop-linker branches PARKed after grouped
evidence; frozen pretrained Trackastra qualified for grouped-OOF head design;
no external training authorized or launched  
Data boundary: all proposed training and falsification use only the frozen fold-0 fit
set of 117 videos. The 57-video holdout is inaccessible until a candidate has completed
its preregistered fit-only path.

## Decision

The next implementation candidate is now **frozen pretrained Trackastra plus a
small fit-only association head**. The exact released CTC body runs locally,
and its zero-wrfeat challenger/incumbent margin passed a preregistered
five-video fit-only signal gate at ROC AUC 0.9185 versus 0.8412 for pair-NCC.
This authorizes grouped-OOF design only. Full deployed-proposal coverage and
exact held-fit graph improvement must pass before validation12 or CV57.

The compact point-context and calibrated 3-D crop-linker branches are PARKed.
Both showed useful fit-only recall, but neither transferred the required
precision across grouped held videos. They remain mechanism evidence rather
than active candidates.

LoRA and Kaggle-T4 work remain behind the frozen-head and external-compute
gates. No T4 candidate is approved here.

No paper result is directly comparable to this competition's patched adjusted-edge
Jaccard. Literature establishes mechanism plausibility, not a promised 3.5% score gain.
The pretrained candidate is
`QUALIFIED_FOR_GROUPED_OOF_IMPLEMENTATION`. Its label-free complete-pool audit
covered all 69 fit-train `6bba` videos: 110,471 geometry-admissible proposals,
5,824 transition calls, maximum 870 required endpoints under the 1,024-token
cap, and zero identity loss. Safety-factored frozen inference projects to
1.213 hours. This state authorizes only the nested fit-only frozen-head path;
it does not authorize validation12, CV57, LoRA, or external compute.

## Measured problem to solve

- Banked score: **0.881522070**, versus the plain **0.873150** baseline.
- Residual after the banked appearance swap: 4,407 false-positive plus false-negative
  edge events. At fixed node adjustment, 378 ideal wrong-to-right swaps would reach 0.90.
- The 10 µm/cap-8 proposal pool excludes 719 fit-only correct edges; 12 µm recovers 456
  at 1.678 times candidate cost.
- The 10–12 µm shell contains 5.34 million background candidates. Geometry plus 3-D NCC
  cannot abstain safely.
- An explicit background-aware compact gate achieved high precision but over-abstained.
  A temporal MLP gained validation utility but fell to 0.9004 precision; per-family
  training worked on 6bba and failed on 44b6.
- The organiser TemporalUNet3D plus SimpleNodeTransformer fit-only checkpoint emitted
  1,816 links for 100 ground-truth links in an inference-like audit. Extrapolated training
  to epoch 300 is about 478 T4 hours or 2,455 M5 hours, so it is archived.

The useful model must learn **competition and abstention**, not merely score isolated
pairs, and it must do so without a large dense 3-D segmentation backbone.

## Ranked local lane

All estimates are engineering projections, not measured runtimes. The initial pilot must
run under `scripts/run_local_training_guard.py`: at most 18 GiB for training, at least
6 GiB background reserve and 25% system memory free, no swap growth, and at most 7 hours
unattended.

| rank | candidate | proposed small form | projected peak | first pilot | continuation ceiling |
|---:|---|---|---:|---:|---:|
| 1 | Compact point-context transformer | 3-frame window; `d=96`; 2 encoder + 2 decoder layers; 4 heads; 512-token training cap; shallow intensity, geometry, motion and family-normalized features; parental softmax with no-parent | 2–6 GiB | 60–90 min | 7 h |
| 2 | Calibrated 3-D Siamese crop linker | Shared 3-D crop encoder, relative geometry/context head, explicit no-link/background class, held-fit calibration | 4–10 GiB | 90 min | 7 h |
| 3 | Frozen pretrained Trackastra + small head/LoRA | Released model, frozen body first; supervised fit-only head, then minimal adapters only if the head passes | 6–12 GiB, provisional | 120 min | 7 h |

### Local rank 1 preregistration

Training split is grouped 105/12 inside fit117. No holdout path may be imported or opened.
Use the existing fit-only proposal/background tables, with joint window-level spatial,
intensity and temporal augmentations. Reject if any of these occurs:

- peak training memory exceeds 18 GiB, system free memory falls below 25%, or swap grows;
- no validation capped-utility improvement after 500 optimizer steps;
- overall or either-family validation precision falls below 0.93;
- the 90-minute projection cannot finish a useful continuation inside 7 hours.

Promotion requires all of:

- grouped out-of-fold precision at least 0.95;
- held-fit validation precision at least 0.95 for both 44b6 and 6bba;
- held-fit capped utility at least 5 above the exact pair-NCC control;
- non-negative utility change on both families;
- reproducible fit117-only artifact provenance.

The full local continuation stops on a 20% validation-patience window. Passing the local
pilot does not approve Kaggle compute or holdout evaluation.

### Local rank 2 preregistration

Positive pairs come only from fit GT. Guaranteed negatives are same-target competing
parents, same-frame pairs, and deployment-background proposals that cannot be positives
under the fit matching rule; uncertain sparse-label pairs are masked, not called negative.
Calibration uses a disjoint fit-only group. It has the rank-1 precision/utility gates and
also fails if calibration error does not improve over the raw head.

## Ranked Kaggle T4 lane: maximum 30 cumulative GPU-hours/model

These are feasibility envelopes, not approvals. Before a long launch, run exactly one
30–60 minute private fit-only probe, count setup and validation in the cumulative budget,
measure peak VRAM and steps/second, and extrapolate with at least a 1.25 safety factor.
Archive if the credible total exceeds 24 hours; the remaining 6 hours are recovery reserve
inside the hard 30-hour ceiling.

| rank | candidate | projected T4 envelope | VRAM envelope | decision |
|---:|---|---:|---:|---|
| 1 | Pretrained Trackastra supervised head/LoRA | 8–24 h | 10–16 GiB | CONDITIONAL PILOT |
| 2 | Compact Trackastra from scratch (`d=128`, 3+3 layers, window 3–4) | 8–20 h | 8–14 GiB | CONDITIONAL PILOT |
| 3 | Linajea-lite frozen features + backward displacement/no-link head | 12–28 h | 12–16 GiB | HOLD |
| 4 | ASCENT-style projected-patch ResNet18 embedding | 6–20 h | 6–12 GiB | HOLD: domain mismatch |

For Trackastra probes, pass only if peak VRAM is at most 14.5 GiB, the safety-factored
projection is at most 24 hours, held-fit loss and capped utility have a useful learning
slope, and the local precision requirements remain achievable. Full Trackastra from
scratch at the paper's `d=256`, 6+6 layers, window 6, 2,048 tokens and batch 8 is not
presumed to fit a T4.

Linajea-lite is not permission to train full Linajea. It may use precomputed/frozen
features and a small displacement/no-link head only. Any organiser-like background
over-linking or a safety-factored projection above 24 hours archives it.

## Rejected, paused and watchlisted

| family | status | reason |
|---|---|---|
| Full organiser TemporalUNet3D + SimpleNodeTransformer | ARCHIVE | Fit-only false-positive audit failed; about 478 T4 hours to epoch 300 |
| Full Linajea | ARCHIVE | 4-D U-Net, seven-frame input, 400,000 iterations; incompatible with the T4 ceiling |
| Full HOCT | WATCHLIST, NOT TRAINABLE | Excellent edge-centric mechanism, but only a 2026 preprint; its full run used an H200 with about 60 GiB peak VRAM |
| Compact HOCT-inspired derivative | WATCHLIST | Revisit after peer review/code release; current primary-peer-reviewed gate is unmet |
| CELLECT | PAUSED | Strong sparse 3-D design, but clean-room/licensing and engineering conditions in `_LATER/cellect/PAUSE_RATIONALE.md` remain unmet |
| Full Ultrack/SAM/Cell-TRACTR/dense segmentation | ARCHIVE FOR THIS FUNNEL | Adds detection expense while the measured bottleneck is sparse association and abstention |
| Topology-message-passing GNN | REJECT | Prior local evidence failed; HOCT reports near-random label agreement among incident candidate edges |
| Mitosis-heavy tracker | REJECT FOR NOW | Divisions are off and measured score headroom is ordinary association |

RunPod has no eligible candidate. The accepted score is below 0.89, and no candidate has
fit-only evidence for a greater-than-5% quality-GPU-dependent gain.

## Literature basis

1. Gallusser and Weigert, **Trackastra: Transformer-based cell tracking for live-cell
   microscopy**, ECCV 2024. It predicts associations among detection tokens in temporal
   windows, uses a quiet parental softmax with an implicit no-parent choice, and reports
   good results for windows 3–6. On its bacteria test, shallow features reduced AOGM
   about 70% from the cited state of the art, while points-only remained competitive.
   Paper: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>.
   Code: <https://github.com/weigertlab/trackastra>.
2. Kok et al., **OrganoidTracker 2.0: an algorithm for tracking cells in 3D organoids**,
   Nature Methods 2025. It separates center, link and division networks, restricts link
   candidates spatially, calibrates probabilities, and uses global optimization. Its
   documented RTX 2080 Ti/11 GiB training supports a compact local interpretation.
   Paper: <https://www.nature.com/articles/s41592-025-02845-6>. Training notes:
   <https://jvzonlab.github.io/OrganoidTracker/TRAINING_THE_NETWORK.html>.
3. Lalit et al., **An Investigation of Unsupervised Cell Tracking and Interactive
   Fine-Tuning**, ICCV 2025 BIC Workshop. It supports association-matrix fine-tuning and
   low-rank adaptation around Trackastra. Paper:
   <https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Lalit_An_Investigation_of_Unsupervised_Cell_Tracking_and_Interactive_Fine-Tuning_ICCVW_2025_paper.html>.
4. Bragantini, Theodoro and Royer, **Higher-Order Cell Tracking Transformer**, arXiv
   2607.11754v1, 2026. It is not peer-reviewed, so cannot promote a candidate, but it
   motivates edge competition, no-parent handling, multi-gap normalization and tracklet
   solving. Paper: <https://arxiv.org/pdf/2607.11754>.
5. Gallusser et al., **Automated reconstruction of whole-embryo cell lineages by learning
   from sparse annotations**, Nature Biotechnology 2022. It supports sparse point
   supervision/backward displacement, but its full 400,000-iteration 4-D U-Net is outside
   budget. Paper: <https://www.nature.com/articles/s41587-022-01427-7>.
6. Ma et al., **CELLECT: contrastive embedding learning for large-scale efficient cell
   tracking**, Nature Methods 2025. It reports over 50-fold faster processing than
   Linajea, but remains paused by the clean-room gate. Paper:
   <https://www.nature.com/articles/s41592-025-02886-x>.
7. Han et al., **ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D
   Neuron Tracking in Electron Microscopy**, ICCV 2025. It supports a projected-2-D
   contrastive branch, but transfer to dividing light-microscopy cells is not established.
   Paper:
   <https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html>.

## Next authorized choice

The compact point-context dry-data audit passed: 25,289 tokens form 8,382
transition windows, the largest window contains 28 tokens, all 4,007 positive
tokens survive the declared cap, and no target has multiple positive parents.
Its strengthened guarded pilot completed: validation utility improved by 39
and overall precision reached 0.9636, but 44b6 precision was only 0.8333 versus
the locked 0.93 pilot floor. The exact model PARKs. Its 6bba slice was strong
(182/184, 0.9891). The new retrospective held-fit graph diagnostic now
quantifies why proposal evidence is insufficient: the earlier 6bba temporal
ranker reproduced proposal OOF precision 0.9648, yet exact graph scoring lost
0.000612 adjusted edge Jaccard to APNCC on fit69 by adding 49 FP for only two
TP. The graph gate correctly PARKed the model, matching its known CV57
regression. A new family-scoped point-context identity is admissible only
after deployment-complete runtime proposal generation, nested held-fit
predictions, and at least +0.005 actual graph gain over exact APNCC.

The calibrated crop-linker completed its guarded 500-step CPU pilot. It improved
validation capped utility by 33 and reached 0.9846 overall precision, but PARKed
because 44b6 precision was only 0.8889 against the preregistered 0.93
per-family floor. Its separately identified 6bba-only branch then completed a
five-model nested grouped OOF confirmation on 69 videos without validation12 or
CV57. OOF utility improved by 471 and all fold gains were positive, but
precision was only 0.9233 overall, with two folds below 0.93. This stronger
evidence PARKs the entire crop-linker branch: the representation adds recall
but does not solve cross-video operating-threshold transfer.

Kaggle training, RunPod use, and public submission remain behind their
corresponding gates. The organiser model remains archived and this work is
separate from V45.
