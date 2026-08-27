# Tiered surface — representation, robustness, feasibility, and governance

Date: 2026-07-28  
Owner: representation/governance branch  
Status: inventory only; no experiment, external action, or shared-state mutation  
Objective: maximize the frozen official
`weighted_adjusted_edge_jaccard + 0.1 * division_jaccard`

## Scope and evidence rule

This branch inventories pretrained representations, `H3`/`F0`/`F1`, domain
robustness, the feasibility-exact action compiler, and compute/submission governance.
It does not execute an oracle, train a model, score a graph, alter `LIVE_STATE.md`, or
write `LEDGER.md`.

A numerical result affects a tier only when it exists in the banked root record and is
cited below by exact repository path and SHA-256. A literature claim or an uncommitted
external memo may define a `PARK-B` hypothesis, but it cannot promote or demote it.
Proposal precision, AUC, and utility remain diagnostic only.

The common scientific cascade is:

```text
authenticated artifact and immutable contract
  -> cheapest fit-only feasibility/oracle test
  -> 44b6 held-fit graph gate
  -> unchanged 6bba held-fit graph gate
  -> CV57
  -> fail-closed package
  -> explicit-human-approval public submission
```

For every representation or decision arm that reaches graph scoring:

- pass each family only at adjusted delta `>= +0.005` with raw edge and node-recall
  deltas both nonnegative;
- kill the instantiation at 44b6 adjusted delta `<= 0`, and do not open 6bba;
- use proposal-level metrics only as diagnostics;
- keep the baseline graph and no-op action explicit.

## Initial census

This census includes the 18 nodes defined in this file.

| tier | count | nodes |
|---|---:|---|
| `LIVE` | 0 | none |
| `PARK-A / PROMISING` | 4 | `G0-U0-T4CANON`, `G3-KAGGLE-GPU`, `G4-LOCAL-ZEROSWAP`, `G5-SUBMISSION-GATE` |
| `PARK-B / UNCERTAIN` | 12 | `R2-H3-SOURCE`, `R3-F1-REAL-WRFEAT`, `R4-F0-ENCODER-PROVENANCE`, `R5-F0-XY-BETA`, `R6-F0-TRIVIEW`, `R7-F2-MICRO-SSL`, `R8-C3-BIDIR-CYCLE`, `R9-ENV-GROUPDRO`, `R10-PEFT-44B6`, `G1-FEASIBILITY-COMPILER`, `G2-DFL-ACTION-REGRET`, `G6-RUNPOD` (governance-closed) |
| `PARK-C / UNLIKELY` | 1 | `R1-CG1B-LEGACY-TAIL` |
| `DEAD` | 1 | `R0-H1-ZWRFEAT` |

`C2-KBEST-STABILITY` is owned by the association branch and is not counted here. Its
unbanked claimed graph vector cannot alter its banked `NOT SCORED` state; this branch
therefore treats the claim as `PARK-B` pending a root-banked artifact. The same rule is
applied to the unbanked “Branch B 66/223 infeasible” assertion: it cannot demote
decision-focused learning or establish a compiler defect.

## Dependency DAG

```text
G0 U0-T4CANON [PARK-A, locked enabling asset]
  |
  +--> G1 feasibility-exact compiler [PARK-B]
  |      |
  |      +--> action-class oracle interface (owned by P0 branch)
  |             |
  |             +--> G2 decision-focused exact-regret action model [PARK-B]
  |
  +--> R4 F0 encoder provenance [PARK-B]
  |      |
  |      +--> R5 XY dense beta [PARK-B]
  |      +--> R6 physical tri-view alpha [PARK-B]
  |
  +--> R2 H3 authentic source [PARK-B]
  |      |
  |      +--> R3 F1 real-wrfeat ablation [PARK-B]
  |
  +--> R7 microscopy-specific SSL [PARK-B]
  +--> R8 bidirectional cycle consistency [PARK-B]
  +--> R9 environment-group robustness [PARK-B]
  +--> R10 frozen-backbone PEFT [PARK-B]

Any branch needing heavy feature generation
  -> G3 Kaggle GPU [PARK-A; <=30 h/week]
  -> graph cascade
  -> G5 public submission [PARK-A; explicit human approval]

Local/cache-only work
  -> G4 zero-swap gate [PARK-A]

G6 RunPod [PARK-B / forbidden by owner; no outgoing edge]
```

`R0` and `R1` are retained as negative controls. Neither can be silently reused as a
new branch.

## Representation nodes

### R0 — `H1-ZWRFEAT`: frozen Trackastra with zero-filled morphology

- **Tier:** `DEAD` (already promoted by root/human authority; this file does not make
  the promotion).
- **Banked evidence:** 44b6 adjusted/raw/node deltas
  `-0.0111480766/-0.0110575321/-0.0002879009`, despite report-only proposal precision
  `0.9565217391`; 36/36 fragments were observation-distinct.
- **Artifact:** `reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json`,
  SHA-256 `530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d`.
- **Resume/new-name condition:** never re-screen the zero-filled 12-channel
  instantiation. A successor must introduce a new named evidence mechanism, such as
  authenticated real morphology, dense image embeddings, or an explicitly different
  temporal representation.
- **44b6 kill:** already fired; do not run 6bba or CV57.
- **Primary source / transfer domain:** Gallusser and Weigert, Trackastra, ECCV 2024,
  <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf>. Direct
  live-cell microscopy tracking method, but not evidence that zero-filled morphology
  transfers to G-02.

### R1 — `CG1B-LEGACY-TAIL`: existing frozen appearance/geometry/motion tail

- **Tier:** `PARK-C / UNLIKELY` for this exact instantiation; the broader pretrained
  representation family remains untested and is not demoted.
- **Banked evidence:** at the exact 44b6 full-pool FP budget, the best arm retained
  `14/766 = 1.83%`, below the preregistered `15%` partial and `20%` GO floors. 6bba
  stopped per the hardest-family rule.
- **Artifact:** `reports/g02/screening/candgen/cg1b_representation_rescreen_v1.json`,
  SHA-256 `0c9c085c436dbf455b199d39117645ef3c15e45b112a4c7fdc5b838eaa0e68a0`.
- **Promote:** only a new named representation, tested on the exact full unlabeled
  deployment pool, reaches at least `15%` retained known-correct edges and improves
  by at least two percentage points over its preregistered control in both families.
- **Demote:** a second score-separated failure of the same named representation may
  be proposed to the root/human for DEAD; this branch cannot promote it.
- **44b6 kill:** miss the frozen full-pool retention/FP-budget prerequisite; no graph
  model or 6bba spend.
- **Primary source / transfer domain:** this is a local negative control rather than a
  literature-derived method.

### R2 — `H3-MASK-SOURCE`: authentic morphology provenance

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** the banked launch audit found no authenticated per-detection
  masks or aligned 12-D `wrfeat-regionprops2` arrays. This is absence of an input,
  not a negative morphology result.
- **Artifact:** `reports/g02/screening/h1_execution.json`,
  SHA-256 `31dbb13a7cff5cba30d0b290dff866cee5f6ead8c2a3523b6ff917e5d49ca3e9`.
- **Resume:** owner supplies or separately authorizes acquisition of an artifact with
  origin, license, immutable SHA-256, exact `(identity, frame, node_id)` join,
  complete fit105 coverage, and a fail-closed offline loader. Pseudo-masks are
  forbidden. Raw-Zarr image features may support a separately named branch but may
  not be relabeled H3 masks.
- **Promote:** provenance, licensing, join, coverage, leakage, and repeatability all
  pass before any label or held outcome opens.
- **Demote:** a banked complete provenance audit that again finds no qualifying
  source makes this acquisition instantiation `PARK-C`; it still does not falsify
  morphology.
- **44b6 kill:** not applicable to provenance. Its child `R3` owns the graph kill.
- **Primary source / transfer domain:** Trackastra ECCV 2024, above. It establishes
  the object-feature interface in microscopy, not the existence or validity of G-02
  masks.

### R3 — `F1-REAL-WRFEAT`: authenticated regionprops/morphology beta

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** catalogued as a conditional beta, but no authentic input and no
  F1 graph outcome are banked.
- **Artifact:** `reports/g02/screening/top50_seeking_alpha_research_20260727.md`,
  SHA-256 `e64b11b8c074c8de44f68cbb2494fc3a3c5c34aa75443e65124fc6f3e0ea1715`;
  input absence is independently banked in `h1_execution.json` at SHA-256
  `31dbb13a7cff5cba30d0b290dff866cee5f6ead8c2a3523b6ff917e5d49ca3e9`.
- **Resume:** `R2` passes and a preregistration freezes the real-wrfeat arm, zero-wrfeat
  DEAD control, no-feature/APNCC control, grouped folds, and no pseudo-mask rule.
- **Promote:** positive 44b6 graph delta, followed by the unchanged 6bba pass; raw and
  node metrics must remain nonnegative.
- **Demote:** 44b6 adjusted delta `<=0` makes this exact F1 instantiation `PARK-C`.
- **44b6 kill:** adjusted `<=0`, raw `<0`, or node recall `<0`; do not run 6bba.
- **Primary source / transfer domain:** Trackastra ECCV 2024, direct microscopy
  tracking. The paper motivates morphology inputs but does not validate G-02 masks,
  this join, or this scorer.

### R4 — `F0-ENCODER-PROVENANCE`: frozen encoder acquisition and authentication

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** zero compatible local DINO/SAM checkpoints; exact weight hash,
  source revision, and weight-license authentication were unavailable. No F0 score
  was opened.
- **Artifacts:** `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/f0_triview_dense_ta/N0_weights_license_manifest.json`,
  SHA-256 `b15c491b1166e9f6c5a1bef3d776bd84016f29eb5273603dc1dd2d0bf06803e6`;
  terminal report SHA-256
  `d4d2aa09a315b7ca6578412140aee0f00de1706536887da343d7fc8a54ea117c`.
- **Resume:** owner-provided or separately approved acquisition of the exact encoder
  weights, followed by weight SHA-256, pinned repository tree/revision, code and
  weight licenses, and an offline no-network loader proof.
- **Promote:** provenance passes and the fixed N1 fit-only runtime/identity smoke is
  feasible under the weekly budget.
- **Demote:** authenticated weights that fail the frozen offline interface/runtime
  smoke move this acquisition/encoder pairing to `PARK-C`.
- **44b6 kill:** not applicable to provenance; children `R5`/`R6` own it.
- **Primary source / transfer domain:** Oquab et al., DINOv2, TMLR 2024,
  <https://openreview.net/forum?id=a68SUt6zFt>. Natural-image self-supervision,
  not microscopy or cell tracking.

### R5 — `F0-XY-DENSE-TA`: single-view dense-feature beta

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** architecture, physical crop, normalization, grouped folds, and
  four-arm contract are frozen, but no weights, runtime result, representation cache,
  or graph outcome exists.
- **Artifact:** `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/f0_triview_dense_ta/N0_feature_contract.json`,
  SHA-256 `5a41c125e66f2e6611ef262f8e3a03970921831a4fa1144ed36274ab567be547`.
- **Resume:** `R4` passes, then the fixed density-quantile fit-only smoke validates
  coordinates, crop identity, encoder outputs, determinism, runtime, and no leakage.
- **Promote:** 44b6 adjusted delta `>=+0.005`, raw/node nonnegative; then identical
  6bba pass.
- **Demote:** fail the runtime/identity smoke or 44b6 graph gate.
- **44b6 kill:** adjusted `<=0`, raw `<0`, or node recall `<0`.
- **Primary source / transfer domain:** Dalinghaus, Archit, and Pape, *Dense
  Embeddings from Self-Supervision and Foundation Models Improve Cell Linking
  Performance*, MIDL 2026,
  <https://openreview.net/forum?id=4OLrHi9ROr>. Direct cell-linking evidence on two
  microscopy datasets, not G-02 or its 3-D scorer.

### R6 — `F0-TRIVIEW-DENSE-TA`: physically aligned XY/XZ/YZ alpha

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** the tri-view hypothesis and deterministic within-frame
  XZ/YZ-permutation ablation are preregistered, but no F0 result exists.
- **Artifact:** `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/f0_triview_dense_ta/N0_literature_contract.md`,
  SHA-256 `7466483bb870469e938abb203a4c966b2a99910c54c61aab22cc817c634e9598`.
- **Resume:** `R4` passes and `R5` produces a valid XY beta cache under the identical
  encoder and crop contract.
- **Promote:** pass the common graph gate and strictly beat both XY beta and
  correspondence-destroying permutation control outside the exact rerun envelope.
- **Demote:** 44b6 adjusted `<=0`, protected regression, or the permutation arm
  reproduces the purported tri-view gain.
- **44b6 kill:** common kill plus failure of `tri-view > XY` and
  `tri-view > permutation`.
- **Primary source / transfer domain:** Han and Lu, ASCENT, ICCV 2025,
  <https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html>.
  Direct 3-D neuron-tracking representation work, but not orthogonal projections,
  G-02 cells, or the frozen scorer. The tri-view synthesis is novel and unvalidated.

### R7 — `F2-MICRO-SSL`: microscopy-specific dense encoder alternative

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** the top-50 report identifies dense microscopy embeddings and
  CELLECT as relevant but defers a full model; no compatible F2 checkpoint/interface
  audit or G-02 outcome is banked.
- **Artifact:** `reports/g02/screening/top50_seeking_alpha_research_20260727.md`,
  SHA-256 `e64b11b8c074c8de44f68cbb2494fc3a3c5c34aa75443e65124fc6f3e0ea1715`.
- **Resume:** define one immutable, small, frozen microscopy encoder and a point-level
  adapter; authenticate weight/source/license; compute a fit-only action-class or
  representation oracle before training any new head.
- **Promote:** positive oracle headroom plus common 44b6/6bba graph passes against
  both APNCC and the F0 natural-image beta.
- **Demote:** no usable local checkpoint/interface, zero oracle headroom, or a failed
  44b6 graph instantiation.
- **44b6 kill:** adjusted `<=0`, raw `<0`, node recall `<0`, or equality to the
  no-representation control within exact repeat tolerance.
- **Primary source / transfer domain:** Zhou et al., CELLECT, Nature Methods 2025,
  <https://www.nature.com/articles/s41592-025-02886-x>. Direct cell-tracking
  representation learning, but not G-02, this point-only interface, or this scorer.

## Domain-robustness nodes

### R8 — `C3-BIDIR-CYCLE`: forward/reverse correspondence consistency

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** catalogued as an alpha in the top-50 research report; no
  reverse-inference cache, graph delta, or cycle-consistency outcome is banked.
- **Artifact:** `reports/g02/screening/top50_seeking_alpha_research_20260727.md`,
  SHA-256 `e64b11b8c074c8de44f68cbb2494fc3a3c5c34aa75443e65124fc6f3e0ea1715`.
- **Resume:** prove the required forward and reverse evidence can be generated or
  reused without held labels; freeze the endpoint-return rule and APNCC no-op before
  opening 44b6.
- **Promote:** common both-family graph pass and a non-null consistency ablation.
- **Demote:** cycle filtering only over-abstains, is graph-null, or 44b6 delta `<=0`.
- **44b6 kill:** common kill; no threshold sweep after outcomes.
- **Primary source / transfer domain:** Wang et al., *Learning Correspondence from
  the Cycle-Consistency of Time*, CVPR 2019,
  <https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.pdf>.
  Generic natural-video correspondence, not microscopy lineages.

### R9 — `ENV-GROUPDRO-TRACKLET`: video-environment robust event representation

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** 44b6 is the documented cross-dataset failure axis, but no
  group-DRO tracklet arm or graph outcome is banked. The negative 44b6 history cannot
  be used as positive evidence for this mechanism.
- **Artifact:** `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`,
  SHA-256 `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`.
- **Resume:** introduce new tracklet/event evidence rather than reuse only the DEAD
  scalar or frozen pair-row mechanisms; freeze video-level groups, loss, capacity,
  no-link control, and grouped folds.
- **Promote:** positive 44b6 graph delta, common both-family pass, and improvement
  over ordinary ERM with identical features/capacity.
- **Demote:** graph-null group reweighting, protected-metric regression, or 44b6
  adjusted `<=0`.
- **44b6 kill:** common kill plus no improvement over the capacity-matched ERM beta.
- **Primary source / transfer domain:** Sagawa et al., *Distributionally Robust
  Neural Networks for Group Shifts*, ICLR 2020,
  <https://openreview.net/forum?id=ryxGuJrFvS>. General supervised group-shift
  robustness, not cell tracking; video identities provide the proposed environments.

### R10 — `PEFT-44B6-ADAPTER`: frozen-backbone, parameter-efficient specialization

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** the locked plan permits a `<1%` parameter adapter only after
  frozen inference fails; the zero-wrfeat frozen arm did fail, but no PEFT contract,
  new representation, or PEFT graph result is banked.
- **Artifact:** `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`,
  SHA-256 `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`.
- **Resume:** pair the adapter with genuinely new authenticated image/morphology
  evidence (`R3`, `R5`, `R6`, or `R7`), freeze `<1%` trainable parameters, grouped
  video folds, ordinary frozen-head beta, and no full-backbone fine-tune.
- **Promote:** common both-family pass and a positive adapter-minus-frozen-beta graph
  increment outside repeat tolerance.
- **Demote:** any 44b6 regression, family memorization, or no gain over the frozen
  capacity-matched beta.
- **44b6 kill:** common kill plus adapter-minus-frozen adjusted delta `<=0`.
- **Primary source / transfer domain:** Jia et al., *Visual Prompt Tuning*, ECCV 2022,
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf>.
  Generic natural-image transfer, not microscopy tracking; it supports the PEFT
  principle only.

## Feasibility and decision-governance nodes

### G0 — `U0-T4CANON`: immutable T4-domain governance

- **Tier:** `PARK-A / PROMISING` as a completed enabling asset, not as performance
  signal.
- **Banked evidence:** owner locked contract
  `08cd90e5d67f0bf9c7822caa1a057f767b5e7e66e0889fcde04ef317cfc016c1`;
  two 105-video replays were byte-identical, the disabled challenger reproduced
  APNCC 105/105, and all five 44b6 folds retained the negative
  `NO_ADMISSIBLE_THRESHOLD` result.
- **Artifacts:** `reports/g02/screening/top50_parallel_dags/branches/u0_t4canon/results/u0_t4canon_exact_replay_attestation.json`,
  SHA-256 `b5916b077f8ed66cd6a726e7404946354ceab423ce038877c239411fe2e8fcc1`;
  owner resolution
  `reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/G2_owner_resolution.md`,
  SHA-256 `2333d2ff7971fcbceafe67e77d1005f1bfd006112c7ca3683cb414e76f4cdf54`.
- **Resume/use:** downstream local/cache-only branches may consume the exact locked
  interface. U0 does not authorize a threshold change, revive old H1 N1, create a
  performance claim, or authorize external work.
- **Demote:** any changed row/ID/fold/schema/hash, forbidden path, scorer input, or M5
  decision input fails closed.
- **44b6 kill:** not applicable; integrity-only node.
- **Primary source / transfer domain:** local governance contract; no literature
  transfer claim.

### G1 — `FEASIBILITY-EXACT-ACTION-COMPILER`

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** no compiler result, invalid-action count, or action-vector
  score exists in the root bank. The owner brief requires exact final-graph endpoints,
  all incident conflicts, and degree constraints, but that requirement is not an
  experimental result.
- **Authority source:** direct owner brief,
  `/Users/agent/.codex/attachments/3ae87751-357f-4dba-9bef-92928e2be0ce/pasted-text.txt`,
  SHA-256 `34ae8aa4ee481f83ee2155b5311afcaf2653e8f8c123669081492b3e325b5cc3`
  (not yet a root-banked result).
- **Resume:** freeze an action schema for add/delete/swap/gap-bridge/division-fork/
  node-add/node-restore; validate against the complete final baseline graph before
  support freezes; include explicit no-op and deterministic graph hash.
- **Promote:** 100% of retained actions are executable, endpoint-complete, degree-valid,
  and repeat-identical on fit-only fixtures; then expose the compiler to the P0
  action-class oracle branch.
- **Demote:** any retained action is invalid or compilation is nondeterministic. That
  demotes this compiler instantiation only, not decision-focused learning.
- **44b6 kill:** structural pre-gate: any invalid retained action prevents 44b6 from
  opening. Once an action model exists, the common graph kill applies.
- **Primary source / transfer domain:** Amos and Kolter, OptNet, ICML 2017,
  <https://proceedings.mlr.press/v70/amos17a.html>. General differentiable
  optimization layers, not cell tracking; cited only for explicit constrained
  decision structure.

### G2 — `DFL-ACTION-REGRET`: exact scorer regret versus no-op

- **Tier:** `PARK-B / UNCERTAIN`.
- **Banked evidence:** no root-banked Branch-B graph utility or infeasibility result
  exists. The reported “66/223 infeasible” assertion is quarantined and cannot tier
  this node.
- **Hypothesis source:** direct owner brief at SHA-256
  `34ae8aa4ee481f83ee2155b5311afcaf2653e8f8c123669081492b3e325b5cc3`;
  it establishes the evidence rule, not a DFL result.
- **Resume:** `G1` passes; the P0 branch banks positive oracle headroom for at least one
  action class; then freeze exact frozen-scorer regret, no-op, grouped support,
  learner, and controls before held outcomes open.
- **Promote:** common both-family graph pass and positive exact regret versus no-op
  without feasibility or protected-metric failure.
- **Demote:** zero action-class oracle headroom or a failed 44b6 graph instantiation.
- **44b6 kill:** adjusted `<=0`, raw `<0`, node recall `<0`, any invalid action, or
  no-op outperforming the learned action.
- **Primary source / transfer domain:** Donti, Amos, and Kolter,
  *Task-based End-to-end Model Learning in Stochastic Optimization*, NeurIPS 2017,
  <https://proceedings.neurips.cc/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html>.
  General stochastic decision optimization, not graph tracking.

## Compute and external-action gates

These are permission/resource nodes. Their tiers describe readiness or closure, not
scientific merit.

### G3 — `KAGGLE-GPU`

- **Tier:** `PARK-A / PROMISING` (authorized resource, no job currently LIVE).
- **Owner update:** Kaggle GPU may be used as needed within a cumulative
  **30 GPU-hour/week** budget. Every run must be recorded by the root ledger writer;
  stop before the cap. Heavy feature generation/inference should prefer Kaggle over
  the swap-fragile host.
- **Authority source:** direct owner brief at SHA-256
  `34ae8aa4ee481f83ee2155b5311afcaf2653e8f8c123669081492b3e325b5cc3`.
- **Banked operational evidence:** the prior corrected T4 timing smoke projected
  `5.538884 h`; `reports/g02/screening/h1_smoke_attempt_3.json`,
  SHA-256 `9d21b48195439ad3baf702d4cba31266cc05b829bd1c1366e765bb3bd73bb88e`.
- **Resume/open:** an upstream node freezes code, data/version hashes, runtime
  projection, output quarantine, and requested hours; root confirms cumulative weekly
  hours remain below 30.
- **Demote/stop:** projected or cumulative budget exhaustion, source/version drift,
  missing output quarantine, or an upstream scientific kill.
- **44b6 kill:** GPU generation is not a scoring gate; generated evidence must still
  go through 44b6 first.

### G4 — `LOCAL-ZEROSWAP`

- **Tier:** `PARK-A / PROMISING` as a reusable execution discipline.
- **Banked evidence:** the locked campaign requires a 45-second clean idle preflight,
  two-stage isolation, and never co-loading V45 with Trackastra.
- **Artifact:** `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`,
  SHA-256 `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`.
- **Resume/open:** current preflight reports zero swap growth and the job has an
  immutable memory/time guard.
- **Demote/stop:** any swap growth or cross-stack co-residency; preserve partial atomic
  outputs and route heavy work to `G3`.
- **44b6 kill:** not applicable; execution-safety gate.

### G5 — `PUBLIC-SUBMISSION-GATE`

- **Tier:** `PARK-A / PROMISING` as a proven packaging path, blocked on candidate
  evidence and explicit human approval.
- **Banked evidence:** APNCC produced an accepted public `0.881`, but fine transfer is
  unresolved; submission-worthy candidates must be observation-distinct and survive
  all local graph gates. Public `0.89` is a milestone, not a forecast from local score.
- **Artifact:** `reports/g02/screening/p0_parity.md`,
  SHA-256 `59b2f60cd3708b86a75b7e0d4a5d242e6ba3823907f196d69876cc20ba2b8058`.
- **Owner update:** submission remains separately human-gated even though Kaggle GPU
  compute is authorized. Prepare fail-closed packages and stop.
- **Authority source:** direct owner brief at SHA-256
  `34ae8aa4ee481f83ee2155b5311afcaf2653e8f8c123669081492b3e325b5cc3`.
- **Resume/open:** candidate survives 44b6, 6bba, and CV57; package identity,
  provenance, scorer vector, observation distinctness, and remaining daily slots are
  recorded; human explicitly approves that exact package.
- **Demote/stop:** no qualifying candidate, package/hash drift, missing vector, or no
  explicit approval.
- **44b6 kill:** upstream common kill; no package preparation can override it.

### G6 — `RUNPOD`

- **Tier:** `PARK-B / GOVERNANCE CLOSED`. It is not assigned scientific `DEAD`
  because no repeated score-separated mechanism failure exists.
- **Owner update:** RunPod is forbidden because no funds are available. Do not design
  a branch around it and do not launch it.
- **Authority source:** direct owner brief at SHA-256
  `34ae8aa4ee481f83ee2155b5311afcaf2653e8f8c123669081492b3e325b5cc3`.
- **Resume:** only a new explicit owner instruction can reopen this resource.
- **44b6 kill:** not applicable; no execution edge exists.

## Quarantined claims and cross-branch interfaces

1. **C2/K-best claimed graph deltas:** no matching result exists in the banked root
   ledger, whose C2 row is `PARK_DEPENDENCY / NOT SCORED`. The association branch must
   keep C2 at `PARK-B` until it produces a root-banked exact result. This file neither
   repeats nor interprets the unbanked numbers.
2. **Branch-B claimed 66/223 infeasible actions:** no root-banked artifact establishes
   that count. `G1` and `G2` remain `PARK-B`; the claim does not justify a “repair”
   contract or a demotion.
3. **Action-class oracle table:** owned by the P0/oracle branch. `G1` supplies only
   the required exact compiler interface. No representation trains until the relevant
   action class has positive banked oracle headroom or the branch has a separately
   justified representation-only falsifier.
4. **Division:** this file does not reopen division fitting. The division-action oracle
   may run as a diagnostic, but fitting remains behind the locked association-
   continuity dependency unless the human explicitly amends the plan.

## Cheapest-next ordering

No node is moved to `LIVE` by this inventory. When the root coordinator allocates work,
the cheapest valid order for this branch is:

1. bank the owner’s 2026-07-28 governance update in the root record;
2. freeze and unit-test `G1` without scoring;
3. authenticate an encoder or authentic morphology source (`R4` or `R2`) before any
   GPU work;
4. run the fixed fit-only identity/runtime smoke;
5. run the P0 action-class oracle or the representation’s preregistered 44b6 graph
   falsifier;
6. spend 6bba, CV57, and submission attention only after the preceding gate passes.

This preserves a broad surface without confusing plausibility with evidence.
