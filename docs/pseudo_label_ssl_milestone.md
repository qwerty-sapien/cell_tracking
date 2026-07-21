# G-02 pseudo-label SSL prototype milestone

## Status and boundary

This is a bounded implementation milestone inside active G-02. It is **not G-02 completion** and it does not replace the original native goal, score thresholds, audit, runtime, package, router, or displayed Kaggle-score requirements. No full-fold evaluation, Kaggle action, network call, W&B run, external GPU, or scored submission was performed for this milestone.

The prototype adds:

- `src/celltrack_alpha/pseudo_ssl.py`: fit-only provenance, compact EMA teacher state, task curricula, physical sparse pseudo-graph construction, transform mapping, losses, and accounting;
- `scripts/train_pseudo_ssl.py`: an armed, time-capped local/private-GPU entrypoint;
- `tests/test_pseudo_ssl.py`: focused mechanism and guard tests.

The exact released checkpoint remains the parent. The default universal prototype exposes only 110,950 parameters: the 33-parameter released detector, 27,845-parameter compact auxiliary head, and 83,072 released temporal-block parameters. EMA shadows contain exactly those `requires_grad` parameters. Frozen spatial weights and buffers are neither copied into EMA nor allowed to drift. The three specialists remain frozen because specialist routing belongs to later G-02 work.

## Mechanism

The teacher receives a weak view (small intensity noise). The student receives a strong view (gamma, gain, stronger noise, and independent z/y/x flips). The EMA teacher is evaluated sequentially by swapping only its compact shadow parameters into the model, so there is no duplicate copy of the 3D released core.

Pseudo detections are local maxima from the teacher center output. Detection uncertainty is the bounded Bernoulli confidence entropy `4p(1-p)`; continuation uncertainty additionally includes the teacher motion-variance output and normalized physical residual. Continuation candidates are generated only by a per-video physical-coordinate `cKDTree` query around each child's teacher-predicted parent location. With `K` child detections and neighbor cap `k`, there are at most `K*k` continuation records. Edges are stored as COO-like records; there is no object `N x N`, frame `N x N`, `cdist`, or full-frame pair-logit allocation.

The weak-to-strong transform maps each selected coordinate before indexing the student output and flips the sign of the corresponding physical motion-vector components. Axis permutation is intentionally absent: preserving z/y/x axes avoids silently changing the meaning of anisotropic per-video voxel scales.

The loss is the weighted sum of:

1. sparse soft detection BCE at accepted transformed points only;
2. candidate-only continuation BCE, where the student logit is derived from physical backward-motion residual for that declared edge;
3. sparse Smooth L1 backward-motion consistency for the same accepted edges;
4. parent-point division BCE for an accepted two-child motif.

Unselected voxels are not treated as negative detections.

## Adaptive selection and accounting

Each task starts strict and relaxes over 2,000 optimizer steps, subject to hard floors/ceilings and a deterministic top-confidence coverage budget:

| Task | Confidence | Uncertainty | Target coverage |
| --- | ---: | ---: | ---: |
| Detection | `0.98 -> 0.75` | `0.08 -> 0.35` | `2% -> 35%` |
| Continuation | `0.99 -> 0.80` | `0.05 -> 0.25` | `1% -> 25%` |
| Division | `0.995 -> 0.95` | `0.03 -> 0.10` | `0.2% -> 5%` |

Continuation candidates are additionally rejected when either endpoint was rejected. A division requires its parent, both children, and both continuation edges to have been accepted. Every candidate is assigned exactly one outcome such as `accepted`, `below_confidence`, `above_uncertainty`, `endpoint_rejected`, `motif_dependency_rejected`, `curriculum_budget`, or `gate_closed`. Counts reconcile per task and per video in `metrics.json`.

The division implementation is present but the experiment runner keeps it closed. Current evidence does not satisfy the locked prerequisite because the observed `0.9026` edge-only result is explicitly a two-video rung, not full grouped CV, and its own artifact records `division_activation_allowed=false`. The CLI rejects every attempt to pass `--division-gate-open`, including a bare claimed score; a later G-02 integration must verify an artifact-backed full grouped edge-only CV score of at least `0.90`. The later retention rule still applies: division must improve combined score while reducing adjusted edge Jaccard by no more than `0.005`.

## Fit-only and physical provenance

Before image access, the runner hashes and parses the frozen split, fit-provenance, and registry manifests. For the selected fold it requires all training/calibration/router/threshold source sets to equal the fold fit groups, rejects held-out and audit identities, rejects test IDs, and binds each train video to its registry `exact_identity_sha256`.

Each streamed video separately provides its effective downsampled z/y/x scale. Every node is checked against that video's shape. Every continuation must be forward one frame, have a finite teacher-motion residual inside the configured `4–10 um` physical gate, and respect neighbor cap `2`, `4`, or `8`. Division motifs must reference two distinct adjacent children and two existing sparse continuations.

## Exact dry run

The dry run restores the released checkpoint, validates all immutable manifests, resolves the 12 locally materialized fold-0 fit videos, checks the 110,950-parameter scope, and writes a prelaunch manifest. It does not open image voxels or optimize:

```bash
.venv313/bin/python scripts/train_pseudo_ssl.py \
  --fold 0 \
  --data-root data/derived/t03_fold0_fit10_images_v1/train \
  --output-dir /private/tmp/g02-pseudo-ssl-dry \
  --dry-run
```

Expected output includes `status: not_launched`, `source_count: 12`, `maximum_hours: 1.75`, and `dense_nxn_association: false`.

## Exact 1–2 GPU-hour experiment

Run on one already-available local workstation GPU. Do not use Kaggle, Runpod/another external GPU, enable networking, start W&B, or attach a submission path. The command hard-stops by 1.75 wall-clock hours and therefore consumes at most 1.75 GPU-hours on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 .venv313/bin/python scripts/train_pseudo_ssl.py \
  --fold 0 \
  --data-root data/derived/t03_fold0_fit10_images_v1/train \
  --output-dir reports/g02/pseudo_ssl_f0_v1 \
  --lr 3e-5 \
  --ema-decay 0.995 \
  --candidate-floor 0.20 \
  --max-points-per-frame 512 \
  --edge-radius-um 8.0 \
  --neighbor-cap 4 \
  --ramp-steps 2000 \
  --empty-patience 64 \
  --epochs 6 \
  --max-steps 6000 \
  --maximum-hours 1.75 \
  --seed 17 \
  --armed
```

This is one bounded mechanism experiment, not a hyperparameter sweep. Its outputs are `prelaunch.json`, `checkpoint.pt`, and `metrics.json`. The checkpoint binds the exact fold sources, three immutable manifest hashes, initial teacher identity, compact trainable names, EMA state, optimizer state, local effects, and reconciled per-video accounting. `metrics.json` explicitly says `improvement_claimed: false`.

## Stop and promote gates

The run stops immediately on forbidden or wrong-hash provenance, per-video scale/coordinate failure, candidate duplication/topology failure, neighbor/radius violation, non-finite teacher/student/loss/gradient state, frozen-state drift, 64 consecutive windows with no accepted detection or continuation, 6,000 steps, or the 1.75-hour cap. Any evidence of a dense pair path is a hard rejection, not a tuning opportunity.

After the run, reject the candidate if accounting does not reconcile, status is `coverage_stop`, fewer than 500 steps complete for a reason other than the time cap, the frozen-state check is false, or late-ramp aggregate coverage lies outside detection `2–35%` or continuation `1–25%`. A closed division gate is expected and must be visible as `gate_closed`, not silently omitted.

Promote only to the existing fixed fold-0 two-video non-acceptance rung—not to full CV, audit, packaging, or submission—when all mechanism gates pass. On that rung require, relative to the same frozen parent and compute/view protocol:

- combined official score improvement of at least `+0.005`;
- edge Jaccard decline no worse than `0.005`;
- node recall decline no worse than `0.005`;
- legal sparse graph and no more than `1.5x` parent inference time.

If the rung passes, the next work is a separately controlled 10% G-02 promotion experiment. Full grouped CV, audit, division activation, router, runtime package, and Kaggle evidence remain governed by the original G-02 contracts.

## Why direct alpha-motion reranking was rejected

The repository already contains two decisive pieces of local evidence:

- `reports/g02/t12cmp.json` compares the trained motion-only auxiliary on the frozen two-video rung. Its score is `0.5603938136` versus `0.8320255362` for V6, a delta of `-0.2716317227`; threshold `0.70` did not rescue it.
- `reports/g02/f4gtr.json` finds 12 reachable-but-unselected true edges, but under the locked raw-distance-plus-motion cost the true candidate had lower cost in `0/9` source conflicts and only `1/9` target conflicts. Thus a direct motion-cost reorder rarely chooses the missed true edge even when the endpoint exists.

The earlier one-video compact-head comparison in `reports/g02/f4cmp.json` also showed identical node recall and a combined-score delta of `-0.002544` versus its zero-head control. These results point to representation/consistency and topology calibration, not another scalar alpha-motion reranking coefficient. The SSL prototype therefore tests EMA-stabilized, uncertainty-filtered weak/strong consistency while preserving sparse physical candidates and the global-solver boundary.

## Primary literature

- Tarvainen and Valpola, [Mean Teacher](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html), NeurIPS 2017: EMA model weights as consistency targets.
- Sohn et al., [FixMatch](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html), NeurIPS 2020: confidence-filtered pseudo labels from weak views supervise strong views.
- Zhang et al., [FlexMatch](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html), NeurIPS 2021: curriculum pseudo-label thresholds tied to learning progress.
- Liu et al., [Unbiased Teacher](https://openreview.net/forum?id=MJIve1zgR_), ICLR 2021: EMA teacher/student training for sparse detection with different augmentations.
- Wang, Peng, and Zhang, [Uncertainty-Aware Pseudo Label Refinery](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Uncertainty-Aware_Pseudo_Label_Refinery_for_Domain_Adaptive_Semantic_Segmentation_ICCV_2021_paper.html), ICCV 2021: uncertainty-aware pseudo-label filtering.
- Malin-Mayor et al., [Automated reconstruction of whole-embryo cell lineages by learning from sparse annotations](https://doi.org/10.1038/s41587-022-01427-7), *Nature Biotechnology* 2023: learned sparse center/motion evidence plus globally constrained lineage optimization.
- Gallusser and Weigert, [Trackastra](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf), ECCV 2024: learned cell association with explicit division handling.

These works motivate the mechanism; none is treated as evidence that this repository candidate improves the official G-02 metric.
