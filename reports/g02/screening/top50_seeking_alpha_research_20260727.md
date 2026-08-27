# Top-50 seeking-alpha research and screening DAG

Date: 2026-07-27  
Status: **RESEARCH DECISION / NO EXECUTION AUTHORITY**  
Public anchor: APNCC `0.881`; user-supplied top-50 bar `>=0.910`  
Live-state basis: H1-ZWRFEAT N1 is `DEAD_STRUCTURAL_M5_T4_DRIFT`; no H1 graph score exists

## Executive decision

There is no credible single edge-reranking tweak that can *ensure* a public `0.91`.
The current-pool edge-only projection tops out around local `0.896`, and the observed
local-to-public improvement transfer is only about `0.48–0.66`. Starting from the
banked local/public pair `0.881522 → 0.881`, that coarse range maps a public `0.910`
planning target to approximately **local `0.9255–0.9422`**. This is a planning range,
not a forecast; division-enabled models have no public calibration point.

The best portfolio is therefore a stack:

1. **Unblock the already-paid Trackastra evidence without weakening the failed node.**
   Create a new, human-locked **T4-canonical** arm that fits and evaluates entirely on
   the authenticated T4 cache. The old cross-device N1 verdict remains DEAD. This is
   the fastest route to the missing 44b6 graph evidence and consumes no new model
   inference if the existing cache is sufficient.
2. **Screen a compact higher-order conflict head on frozen Trackastra features.**
   This is the best near-term alpha: it directly models competing *edges as a set*,
   uses pretrained features to avoid the proven 44b6 from-scratch failure, and goes
   directly to the held-fit graph scorer.
3. **Exploit division credit only after association continuity improves.** CV57 has
   42 division events and currently earns zero division credit. At zero division FP,
   12 TP add `0.02857`; 20 TP add `0.04762`; 25 TP add `0.05952`. A topology-aware
   pretrained fork decoder is the only locally plausible single component large
   enough to bridge most of the top-50 gap.
4. **Use Trackastra-conditioned unlabeled-background learning for recall expansion.**
   Revisit the 10–12 µm shell only with a new representation and explicit no-link
   learning; geometry/NCC, constant velocity, generic PU calibration, and the old
   compact tail head remain negative controls, not candidates.
5. **Reserve expensive feature generation for a second wave.** The strongest
   representation bet is multi-view microscopy dense embeddings fused into
   Trackastra, followed by real morphology/wrfeat. Full HOCT, full CELLECT, full
   Ultrack, and end-to-end 3-D training are not budget-feasible betas here.

No Kaggle submission, Kaggle GPU launch, RunPod launch, CV57 evaluation, tolerance
approval, or graph-gate execution is authorized by this report.

## Phase 1 — deterministic objective and disqualifiers

### Equation

For candidate graph \(G\), optimize the exact frozen official score:

`S(G) = weighted_adjusted_edge_jaccard(G) + 0.1 * division_jaccard(G)`.

The local selection objective is the exact scorer on the locked held-fit graph gate
and then CV57. The external objective is public score `>=0.910`. Proposal precision,
AUC, utility, LNK, TRA, AOGM, and paper benchmark scores are diagnostics only.

### Hard disqualifiers

- Any proposal-only promotion; H0 found no proxy with `|rho| >= 0.5`.
- Any 44b6 held-fit graph delta `<=0` for an association candidate.
- Raw edge-Jaccard or node-recall regression at the existing H1 gate.
- Re-screening a DEAD mechanism without a genuinely new named mechanism.
- A from-scratch ranker without pretraining or explicit domain adaptation.
- Geometry/NCC/constant-velocity shell abstention.
- Scalar recalibration, threshold-only repair, or hand-tuned graph-cost sweeps.
- CV57/validation/test leakage into fitting, routing, threshold selection, or arm choice.
- More than 30 cumulative T4-h/model, full organizer-scale training, or any RunPod use
  before an accepted `>=0.89`.
- Any Kaggle GPU/submission or RunPod launch without explicit human approval.

### Candidate sieve

Every funded candidate must pass all five mostly orthogonal criteria:

1. **Objective pertinence:** acts on graph association, division credit, or reachable
   candidate recall; not merely an intermediate metric.
2. **44b6 mechanism:** names why the proven cross-family failure should improve.
3. **Deployment abstention:** models no-link/background or preserves a trusted base
   graph under uncertainty.
4. **Local kill test:** exact preregistered 44b6-first held-fit graph test.
5. **Budget/runtime:** cache-first or bounded fit-only execution under current governance.

## Target arithmetic

### Edge-only planning bar

Using the coarse OLS slope `0.6594275648`:

`0.881522 + (0.910 - 0.881) / 0.659428 = 0.925500 local`.

Using the one high-range APNCC improvement ratio
`0.004 / 0.0083724764 = 0.477756`:

`0.881522 + (0.910 - 0.881) / 0.477756 = 0.942223 local`.

These values make an edge-only current-pool strategy non-credible: its projected
perfect-ranker ceiling is about `0.896`.

### Division leverage

With 42 CV57 GT divisions and no predicted-fork FP:

| division TP | division Jaccard | score contribution | total over APNCC |
|---:|---:|---:|---:|
| 12 | 0.2857 | +0.0286 | 0.9101 |
| 20 | 0.4762 | +0.0476 | 0.9291 |
| 25 | 0.5952 | +0.0595 | 0.9410 |

False forks matter: with `TP=25`, five division FP reduce the contribution from
`+0.0595` to `+0.0532`; ten FP reduce it to `+0.0481`. The right design target is
therefore **high-precision connected two-branch motifs**, not broad fork recall.

## Phase 2 — dependency DAG

```text
U0 human decision: new T4-canonical contract
  |
  +--> C0 cache-only frozen Trackastra linear-head graph control (beta)
  |      |
  |      +--> C1 higher-order conflict-set head (alpha)
  |      +--> C2 k-best/min-marginal stability wrapper (alpha)
  |      +--> C3 forward/reverse consensus (alpha; extra inference if cache lacks it)
  |
  +--> if any association arm passes both families
         |
         +--> D0 native pretrained fork decode (beta)
         +--> D1 scorer-semantic persistent fork decoder (alpha)
         +--> R0 Trackastra-conditioned learned shell no-link model (alpha)
         |
         +--> if local ceiling remains below the 0.925 stretch bar
                |
                +--> F0 dense microscopy embeddings into Trackastra (alpha)
                +--> F1 authenticated real morphology/wrfeat (beta)
```

Critical path: `U0 → C0/C1 → D0/D1 → CV57 → approved public probe`.

U0 is a governance/parity decision, not model alpha. C1 is the main association alpha.
D1 is the main score-leverage alpha. R0/F0 are recall and representation reserves.

## Phase 3–6 — ranked beta and alpha portfolio

Scores are `pertinence × feasibility × defensibility`, each on a 1–5 scale. They rank
research priority; they are not estimated leaderboard deltas.

### Necessary betas

| rank | id | candidate | P/F/D | crowding verdict | decision |
|---:|---|---|---:|---|---|
| B1 | `C0-T4TA-LINEAR` | Frozen zero-wrfeat Trackastra + existing tiny head on a new T4-canonical contract | 5/5/1 | Obvious beta; already paid for | **First control after U0** |
| B2 | `D0-TA-FORK` | Native Trackastra parental-softmax/greedy division decode after continuity passes | 5/4/2 | Competent teams will try it | **Conditional high priority** |
| B3 | `F1-REAL-WRFEAT` | Intended authenticated regionprops/morphology interface | 4/2/2 | Direct paper/interface completion | **Conditional; no pseudo-mask shortcut** |
| B4 | `C3-TA-TTA` | Rotation/reflection/time-direction consensus around frozen Trackastra | 3/3/2 | Standard ensemble idea | **Cheap only if Stage B can be reused** |

Beta is necessary here because the campaign has not yet obtained a Trackastra graph
score. It should not consume the entire search budget.

### Differentiating alphas

| rank | id | candidate | P/F/D | why it is a defensible edge | preregistered first kill |
|---:|---|---|---:|---|---|
| A1 | `C1-TA-HOCT-LITE` | Component-local edge-to-edge attention over frozen Trackastra/APNCC conflict rows, with an explicit no-link token; no graph-topology message passing | **5/5/5** | July-2026 HOCT identifies edge competition and division entanglement; few teams will distill that mechanism onto an already authenticated competition cache. Pretraining directly addresses 44b6. | Five fixed grouped folds; exact 44b6 held-fit graph delta `>=+0.005`, raw/node nonnegative; kill at `<=0`. |
| A2 | `D1-PERSISTENT-FORK` | Pretrained fork scorer over scorer-valid parent→two-branch motifs, allowing a one-frame-late fork and requiring daughter/grandchild persistence plus duplicate suppression | **5/4/5** | Derived from the metric's actual topology and the local finding that 35/37 audited division stages exist but are fragmented. Most teams optimize exact-time mitosis, not the scorer's late-fork equivalence class. | Unlock only after association pass. Held-fit division Jaccard `>=0.30`, zero edge/raw/node regression, and no duplicate-fork explosion; kill at division Jaccard `<=0.15`. |
| A3 | `C2-KBEST-STABILITY` | Murty k-best assignments/min-marginals used as an uncertainty measure; alter APNCC only when a Trackastra-supported edge is stable across near-optimal assignments | **4/5/5** | Cross-disciplinary transfer from operations research; replaces a fragile pairwise winner with assignment-level sensitivity without reopening hand-tuned costs. | Cache-only 44b6 graph test. Kill if `<=0`; pass at `>=+0.003` as a stackable wrapper or `>=+0.005` standalone, with raw/node nonnegative. |
| A4 | `R0-TA-PU-SHELL` | Trackastra/HOCT-conditioned no-link head trained on the full unlabeled deployment population, using non-negative PU risk and groupwise conformal/FDR selection inside occupied-endpoint 10–12 µm conflicts | **5/3/5** | The novelty is the combination of a materially stronger pretrained representation, deployment exposure, and graph-level selection. Generic PU and frozen-score conformal calibration already failed and are not re-proposed. | Only after C0/C1 signal. 44b6 shell graph delta `>0`, bounded accepted support, then both-family `>=+0.005`; kill on any broad background admission or graph regression. |
| A5 | `F0-TRIVIEW-DENSE-TA` | Three orthogonal projected microscopy embeddings at each 3-D point, linearly compressed and fused into Trackastra/HOCT-lite | **4/3/5** | Combines ASCENT-style projected 3-D identity, the May-2026 result that dense microscopy embeddings improve Trackastra, and the competition's missing-mask interface. | Before full extraction, fixed density-quantile runtime/identity smoke; promotion still requires exact 44b6 graph `>=+0.005`. |
| A6 | `C3-BIDIR-CYCLE` | Forward/reverse Trackastra association consensus as label-free uncertainty; accept changes only when time reversal returns the same endpoints | **3/4/4** | Transfers cycle consistency from correspondence learning to sparse biological assignment and attacks domain shift without labels. | Two-family held-fit graph gate; kill if consensus merely over-abstains or delta `<=0`. |
| A7 | `A7-PHASE-TEMPLATE` | Apodized 3-D phase-correlation displacement plus a multi-frame tracklet template, used only inside occupied endpoint conflicts | **3/4/4** | Registration is shift/illumination robust and materially different from centered pair NCC; multi-frame identity matches the residual specification. | Fixed top-residual fit slice, then mandatory 44b6 graph gate. Kill if it cannot beat APNCC at the graph. Never use on the raw shell. |

### Explicitly rejected or deferred

- **Full HOCT:** research signal only; the local landscape estimates roughly H200/60-GiB
  scale and it is an unreviewed preprint. Distill its interaction mechanism instead.
- **Full CELLECT, Ultrack, Linajea, or organizer training:** too much detection/3-D
  compute for the measured association bottleneck and current 30-T4-h ceiling.
- **Another generic compact ranker:** DEAD unless it consumes frozen pretrained
  representation and is selected at the graph objective.
- **Generic PU/conformal thresholding:** already PARKed with near-zero useful retention.
- **Pseudo masks to sneak through H3:** explicitly forbidden by the current contract.
- **Per-family scalar normalization/CORAL alone:** too close to DEAD scalar repair and
  lacks a nameable graph mechanism.
- **Metric gaming through node pruning:** prior apparent gain degraded raw score/recall.

## High-throughput screen

The screen is high-throughput in *graph evaluations*, not in adaptive hyperparameter
search. Every arm is fixed before its held fold is opened.

### Wave 0 — governance unblock

`U0-T4CANON` is a proposed new node and requires explicit human locking.

Inputs:

- authenticated 105-video T4 cache: 195,354 proposals / 280,095 rows;
- exact challenger/conflict/offset identity across M5 and T4 witnesses;
- byte-identical T4 repeat arrays on one 44b6 and one 6bba witness;
- old N1 evidence: 41/49 columns classified benign under its rule, eight outside its
  preregistered probe, maximum Trackastra-margin difference `1.7643e-5`.

Output:

- a **new** T4-native fit/calibration/held contract; no amendment to old N1;
- M5 values become diagnostic only; all model fitting and candidate evaluation use the
  same authenticated T4 feature domain;
- exact deterministic CPU fitting, graph reconstruction, and scorer hashes.

Kill:

- any T4 nondeterminism, identity mismatch, cache contamination, or inability to
  reconstruct exact repeat graphs.

Pass:

- exact T4 repeatability plus exact graph/scorer reproducibility under the new contract.

This does not authorize N2 under the old DAG and does not assert that the cross-device
differences are benign. It creates a separate production-device experiment.

### Wave 1 — cache-only association screen

Run no more than these three fixed arms:

1. `C0-T4TA-LINEAR` — current ridge/direct-swap control.
2. `C1-TA-HOCT-LITE` — one fixed compact edge-set attention architecture,
   `<=100k` trainable parameters, frozen Trackastra body, no GNN.
3. `C2-KBEST-STABILITY` — one fixed `K` and stability rule selected without held labels
   from operations-research sensitivity, not a threshold grid.

For each arm:

`manifest → leakage/provenance → 44b6 held-fit graph → 6bba held-fit graph → CV57`.

Common pass:

- adjusted graph delta `>=+0.005` in **each** family;
- raw delta and node-recall delta `>=0`;
- observation-distinct fragments;
- no candidate selection by proposal precision.

Common kill:

- 44b6 delta `<=0`; do not run 6bba.

Winner:

- maximize the minimum family graph delta, not aggregate proposal utility.
- keep C0 as the beta control even if an alpha wins.

### Wave 2 — division

Unlock only after an association arm passes both families and improves connected
two-branch opportunities.

Run:

1. `D0-TA-FORK` native pretrained fork control.
2. `D1-PERSISTENT-FORK` one fixed scorer-semantic arm.

Required report vector:

`edge adjusted/raw delta, node recall delta, division TP/FP/FN/Jaccard, combined score`.

Minimum survival:

- division Jaccard `>=0.30`;
- edge adjusted/raw/node deltas all nonnegative;
- no duplicate fork remains after one-to-one event suppression.

Stretch target:

- division Jaccard `>=0.50`, which contributes at least `+0.05` before edge gains.

### Wave 3 — recall and representation

Run `R0` only after Trackastra/HOCT-lite graph signal proves the representation. Run
`F0` only if Wave 1+2 cannot plausibly reach local `0.925`.

No shell or dense-feature arm reaches CV57 until it passes both family graph gates.

## Crowding test

- `C0`, native fork decoding, real wrfeat, and TTA are beta: hundreds of strong teams
  can find them.
- `C1` is defensible because it requires connecting a week-old edge-centric result to
  this campaign's exact failure—proposal winners collapse after graph competition—and
  distilling it onto frozen cache rather than training the published large model.
- `D1` is defensible because the edge comes from re-deriving the competition scorer:
  late forks and grandchild-supported branches can score, and 12–25 precise events are
  leaderboard-sized.
- `C2` is defensible because it imports ranked assignment sensitivity, not another
  tracking-cost sweep.
- `R0` is defensible only with the stronger Trackastra/HOCT representation; without it,
  it is the already-failed PU/conformal beta.
- `F0` is defensible because it combines three recent/adjacent results into a
  competition-specific, point-compatible 3-D representation without full segmentation.

## Evidence basis

Primary/official sources:

1. Gallusser & Weigert, *Trackastra: Transformer-based cell tracking for live-cell
   microscopy*, ECCV 2024:
   <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>.
2. Bragantini, Theodoro & Royer, *Higher-Order Cell Tracking Transformer*, arXiv
   2607.11754 (2026; preprint, not peer-reviewed):
   <https://arxiv.org/abs/2607.11754>.
3. Dalinghaus, Archit & Pape, *Dense Embeddings from Self-Supervision and Foundation
   Models Improve Cell Linking Performance*, MIDL 2026 short paper:
   <https://openreview.net/pdf/2d2323e7e5b7ecc9da47e2e748e46f75876e5d04.pdf>.
4. Kok et al., *Cell tracking with accurate error prediction*, Nature Methods 2025:
   <https://www.nature.com/articles/s41592-025-02845-6>.
5. Zhou et al., *CELLECT: contrastive embedding learning for large-scale efficient cell
   tracking*, Nature Methods 2025:
   <https://www.nature.com/articles/s41592-025-02886-x>.
6. Malin-Mayor et al., *Automated reconstruction of whole-embryo cell lineages by
   learning from sparse annotations*, Nature Biotechnology:
   <https://www.nature.com/articles/s41587-022-01427-7>.
7. Kiryo et al., *Positive-Unlabeled Learning with Non-Negative Risk Estimator*,
   NeurIPS 2017:
   <https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html>.
8. Angelopoulos et al., *Conformal Risk Control*, ICLR 2024:
   <https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html>.
9. Murty, *An Algorithm for Ranking all the Assignments in Order of Increasing Cost*,
   Operations Research 1968:
   <https://doi.org/10.1287/opre.16.3.682>.
10. Wang et al., *Learning Correspondence from the Cycle-Consistency of Time*, CVPR
    2019:
    <https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.pdf>.

Local decisive evidence:

- `LIVE_STATE.md` and `h1_cache_attestation.json` for the current no-executable-lever
  frontier and exact T4/M5 evidence.
- `h0_proxy_validity_v1.md` for the proposal-proxy ban.
- `appearance_residual_headroom_v1.md` and `achievable_headroom_v1.md` for the
  378-swap scale and current-pool ceiling.
- `division_sandbox_and_holdout_decomposition_v1.md` for scorer-valid late/persistent
  fork topology, 42 CV57 events, and the association prerequisite.
- `extended_radius_reachability_v1.md` and
  `extended_radius_shell_appearance_support_v1.md` for shell headroom/background burden.
- `model_landscape_20260724.md` for budget and prior model failures.

## Owner decisions requested

1. **Lock or reject `U0-T4CANON`.** This is the only immediate path that can use the
   completed cache without pretending old N1 passed.
2. If locked, authorize local/cache-only implementation of `C0`, `C1`, and `C2` under
   the new contract. This is not authorization for new Kaggle GPU or submission work.
3. Pre-authorize only the *design* of `D0/D1`; execution remains conditional on an
   association graph pass.
4. Keep `R0/F0/F1` closed until the earlier gates justify their cost.

