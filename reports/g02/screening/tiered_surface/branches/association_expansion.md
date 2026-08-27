# Association-expansion tiered surface

Date: 2026-07-28  
Status: **INVENTORY / NO EXECUTION AUTHORITY**  
Scope: association continuity, candidate/action expansion, and representations that
serve association. Division fitting remains locked behind an association-continuity
pass. No model, scorer, CV57, GPU, submission, external service, `LIVE_STATE.md`, or
`LEDGER.md` was touched by this inventory.

## 0. Evidence boundary

Facts below come only from banked root artifacts. Git blob hashes are against root
`HEAD=2c74eed32c6f080faff045ae22847b98bbde8db0`.

Two supplied claims are explicitly **not evidence**:

- The claimed C2 deltas (`+0.00261` versus C0 and `-0.01003` versus APNCC) are
  rejected for tiering. The root record has no committed C2 graph artifact; C2 is
  therefore still an unscored mechanism and is `PARK-B`.
- The claimed Branch-B audit (`66/223` infeasible actions) is rejected for tiering.
  No committed/banked root artifact records that experiment. Decision-focused action
  learning remains `PARK-B`, and its first node is a fresh feasibility-exact contract,
  not a presumed repair of an established result.

A single failed instantiation does not kill a family. Existing human-promoted DEAD
mechanisms are inherited unchanged; this branch creates no new DEAD verdict.

## 1. Objective, oracle framing, and near-term order

The exact objective is

`S(G) = weighted_adjusted_edge_Jaccard(G) + 0.1 * division_Jaccard(G)`.

Every model/action arm follows:

`action-class oracle -> frozen preregistration -> 44b6 graph -> 6bba graph -> CV57`.

The universal graph survival line is deliberately stricter than mere positive sign:

- **44b6 first:** adjusted delta `>= +0.005`, raw delta `>= 0`, node-recall
  delta `>= 0`; otherwise KILL the instantiation and demote it to `PARK-C`.
- **Then 6bba:** the same adjusted `>= +0.005`, raw `>= 0`, node `>= 0`;
  otherwise do not open CV57.

Proposal precision, AUC, action utility, assignment stability, and oracle score are
diagnostics only.

### Three ceilings that must not be conflated

1. **Current-output/current-policy reranking projection:** local `0.895678`.
   This is the narrow near-term path to the soft local `~0.895-0.900` requirement for
   an accepted public `0.89`, but it leaves essentially no margin. Evidence:
   `reports/g02/screening/achievable_headroom_v1.md`
   (git blob `92037876cd640c84e8dd93c6f59d2a40fc74d372`).
2. **Literal current r10/k8 candidate-pool perfect selection:** local `0.960918`.
   This is not a realizable forecast; it proves candidate presence, not selector
   transfer. Evidence:
   `reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md`
   (git blob `69448d160137d1729b9c0b59a3729f8032eafcc9`).
3. **Expanded r20/k32 plus one-gap maximum:** local `0.975750`
   (`44b6=0.997711`, `6bba=0.971718`). This is an expansion ceiling, not permission
   to absorb its much larger background. Same committed CG0 artifact.

Therefore the funding order is: first try to realize the narrow current-policy
headroom for the near-term `0.89` milestone; in parallel, cheaply measure which
expanded action classes contain the headroom needed for `0.91`. High ceilings alone
never promote a learned arm.

## 2. Dependency DAG

```text
AE-P0 feasibility-exact action compiler + class oracles
  |
  +--> AE-P1 current-policy exact-regret selector          [near-term 0.89 path]
  |
  +--> AE-P2 r10/k8 full-pool occupied-conflict selector
  |      +--> AE-P3 pretrained PU/no-link shell
  |      +--> AE-P5 K-best/min-marginal stability
  |
  +--> AE-P4 one-gap / extended-pool bridge actions
  |
  +--> representation/evidence suppliers
         AE-P6 bidirectional cycle evidence
         AE-P7 phase/template tracklet evidence
         AE-P8 dense pretrained microscopy embeddings
         AE-P9 44b6-specific PEFT adapter

AE-P10 is the parked coarse track-context instantiation; it is a negative control,
not a reusable model.
```

## 3. Nodes

### AE-P0 — feasibility-exact action compiler and action-class oracle

- **Tier:** `PARK-A / PROMISING`.
- **Why:** committed perfect-selection headroom is positive, but no committed
  add/delete/swap/gap-bridge class decomposition exists. This is the common enabling
  falsifier, not a performance claim.
- **Committed evidence:** CG0 current/full expansion ceilings,
  `reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md`
  (`69448d160137d1729b9c0b59a3729f8032eafcc9`); proposal-only gating is banned by
  `reports/g02/screening/h0_proxy_validity_v1.md`
  (`6492c45cca465044c0b41daba6104530d0b53ced`).
- **Output:** for each family and action class `{add, delete, swap, gap-bridge}`,
  exact perfect-action `Δadjusted/Δraw/Δnode`, TP/FP/FN changes, support, and a stated
  FP/background budget. Every action carries all existing incident edges, endpoints,
  degree constraints, and a post-application graph-validity proof before support
  freezes.
- **Promote/resume/demote:** resume by freezing schemas, final-graph endpoint rules,
  scorer hash, and family-isolated inputs. Promote a class to `PARK-A` only if its
  44b6 oracle is at least `+0.005` with nonnegative raw/node. Zero/negative 44b6
  oracle demotes that class directly to `PARK-C` without training.
- **Exact graph kill:** any later realized arm is killed at 44b6 unless adjusted
  `>=+0.005`, raw `>=0`, node `>=0`; only then require the same on 6bba.
- **Primary source / transfer:** feasibility-aware decision-focused learning treats
  feasibility and decision quality jointly
  ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ef158a0d4f5364741a571b8d1d44fb1b-Abstract-Conference.html)).
  Transfer domain is generic constrained prediction/optimization, not cell tracking;
  here the frozen graph invariants supply the constraints.

#### Initial P0 action-class table — evidence inventory, not an oracle result

No class-separated exact oracle is banked, so no numeric family cell is imputed.
The entries below are the required first outputs if AE-P0 is authorized.

| action class | committed partial clue | 44b6 exact oracle | 6bba exact oracle | required frozen budget/constraint | initial class tier |
|---|---|---|---|---|---|
| add edge | Full r10/k8 pool ceiling `0.960918`, not attributable to adds | `NOT MEASURED` | `NOT MEASURED` | background FP plus destination/source-degree conflicts | `PARK-B` pending class oracle |
| delete edge | Residual audit says 796 perfect FP prunes reach local `0.90`; this is CV57 arithmetic, not a family oracle | `NOT MEASURED` | `NOT MEASURED` | maximum removed true edges and node-recall non-regression | `PARK-A` partial headroom |
| swap edge | Residual audit says 378 ideal wrong-to-right swaps reach local `0.90`; this is CV57 arithmetic, not a family oracle | `NOT MEASURED` | `NOT MEASURED` | complete incumbent/challenger incident-edge bundle | `PARK-A` partial headroom |
| gap bridge | Extended maximum `0.975750` includes one-gap candidates but does not isolate their contribution | `NOT MEASURED` | `NOT MEASURED` | gap length, endpoint availability, and background FP | `PARK-B` pending class oracle |

Any future populated cell must report
`Δadjusted/Δraw/Δnode`, TP/FP/FN changes, action support, invalid-action count
(`0` required), and the immutable scorer/input hashes.

### AE-P1 — current-policy feasibility-exact regret selector

- **Tier:** `PARK-A / PROMISING`; **first association fallback for public 0.89**.
- **Why:** APNCC direct swap is the sole banked transferred association win, and 378
  ideal residual swaps were sufficient to reach local `0.90`; the current-policy
  projection is `0.895678`. This is positive headroom but only a narrow realization
  margin.
- **Committed evidence:** APNCC `+0.008372` graph win in
  `reports/g02/screening/selected_graph_appearance_direct_v1.md`
  (`f0a4e3fc4ae2becf2934d3a1e367e5a757ce1d2a`); public `0.881` in
  `reports/g02/screening/q_public_probe_selected_graph_appearance_v1.md`
  (`c9d032009ca966490e889d90199868dcd8d59b4f`); residual scale in
  `reports/g02/screening/appearance_residual_headroom_v1.md`
  (`7b03f95d90bb7d02f024abdc949f1877ef25af1c`).
- **Hypothesis:** among feasibility-exact current-policy actions, directly learning
  frozen-scorer regret versus explicit no-op realizes enough of the 378-swap residual
  to reach the near-term local `~0.895-0.900` planning band.
- **Promote/resume/demote:** resume only after AE-P0 proves nonzero current-policy
  swap/delete headroom and freezes valid actions. Promote to `LIVE` with a
  preregistered grouped 44b6 run. Demote to `PARK-C` if the class oracle is below
  `+0.005`, support is insufficient, or the graph gate fails.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then 6bba
  adjusted `>=+0.005`, raw `>=0`, node `>=0`. No public-score forecast is permitted.
- **Primary source / transfer:** Donti, Amos & Kolter optimize prediction models for
  downstream task loss
  ([NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/3fc2c60b5782f641f76bcefc39fb2392-Abstract.html)).
  Their demonstrations are operations problems, not tracking; the structural transfer
  is “prediction inside a discrete downstream optimizer.”

### AE-P2 — full r10/k8 occupied-conflict regret/no-link selector

- **Tier:** `PARK-A / PROMISING`.
- **Why:** the committed literal r10/k8 oracle is `0.960918`, but existing frozen
  selectors do not realize it. The alpha is not another pairwise ranker: it is
  feasibility-exact no-op/action regret restricted to occupied-endpoint conflicts,
  with exposure to the full deployment population.
- **Committed evidence:** CG0 (`69448d160137d1729b9c0b59a3729f8032eafcc9`);
  the strongest existing 44b6 full-tail screen retained only `14/766=1.83%` in the
  root ledger, so the old representation/selector is a negative control rather than
  a reusable arm.
- **Promote/resume/demote:** resume after AE-P0 separately quantifies add/swap/delete
  headroom and background budgets. Promote to `LIVE` only with a new pretrained or
  explicitly domain-adapted representation and a fixed no-op control. Demote to
  `PARK-C` if its class oracle is zero or if exact full-population support cannot be
  constructed without leakage.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line.
- **Primary source / transfer:** Trackastra learns pairwise cell associations in
  full spatiotemporal detection context
  ([ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf)).
  This source is directly in live-cell tracking, but it does not establish G-02
  scorer improvement or full-pool abstention.

### AE-P3 — pretrained deployment-PU shell no-link

- **Tier:** `PARK-A / PROMISING`, dependency-blocked.
- **Why:** radius expansion recovers 456/719 excluded correct links at 12 µm, while
  the old geometry/NCC shell sees 5.34M background candidates. Positive reachability
  makes the class promising; the old deterministic shell mechanism remains DEAD.
- **Committed evidence:** `reports/g02/screening/extended_radius_reachability_v1.md`
  (`d3df8d40122c3d0340da764851df48a85a272651`) and
  `reports/g02/screening/extended_radius_shell_appearance_support_v1.md`
  (`9b7e870b9c1ebba96372faf0f27fa2d82ee9e90a`).
- **Distinct mechanism:** pretrained association evidence + non-negative PU risk on
  the actual unlabeled shell population + explicit no-link + occupied-endpoint
  action constraints. Geometry/NCC or constant velocity alone are forbidden controls.
- **Promote/resume/demote:** resume after AE-P2 or another pretrained association arm
  passes both families and AE-P0 shows positive shell-add oracle at a frozen FP
  budget. Demote to `PARK-C` if oracle headroom is zero, safe support is null, or
  background admission exceeds the frozen budget.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line.
- **Primary source / transfer:** non-negative PU risk estimation
  ([Kiryo et al., NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)).
  Transfer domain is generic positive-unlabeled classification, not microscopy; it
  supports the risk estimator, not the G-02 representation or graph policy.

### AE-P4 — feasibility-exact one-gap and extended-pool bridge actions

- **Tier:** `PARK-A / PROMISING`.
- **Why:** the r20/k32 plus one-gap maximum is `0.975750`, but the committed report
  does not isolate how much comes from gap bridges. This node first separates that
  action-class oracle before any bridge model.
- **Committed evidence:** CG0 (`69448d160137d1729b9c0b59a3729f8032eafcc9`);
  residual concentration in
  `reports/g02/screening/appearance_residual_headroom_v1.md`
  (`7b03f95d90bb7d02f024abdc949f1877ef25af1c`).
- **Hypothesis:** short missing-link events can be proposed with longer pre/post
  tracklet evidence and applied as complete graph actions, realizing new associations
  outside current output selection.
- **Promote/resume/demote:** resume only if AE-P0 reports 44b6 gap-bridge oracle
  `>=+0.005` under an explicit FP budget. Demote the class to `PARK-C` on zero
  class-specific headroom; otherwise preregister one fixed bridge mechanism.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line.
- **Primary source / transfer:** whole-embryo lineage reconstruction uses global
  spatiotemporal and biological context to recover lineages from noisy observations
  ([Nature Biotechnology](https://www.nature.com/articles/s41587-022-01427-7)).
  This is direct microscopy-lineage evidence, but not this benchmark’s action
  compiler or scorer.

### AE-P5 — assignment K-best/min-marginal stability

- **Tier:** `PARK-B / UNCERTAIN`.
- **Why:** plausible cross-disciplinary mechanism, but there is no accepted C2 graph
  result in the root evidence. Supplied external deltas are ignored.
- **Committed evidence:** the need for non-pairwise realization is supported by the
  current/full-pool gap in CG0 (`69448d160137d1729b9c0b59a3729f8032eafcc9`);
  no committed C2 outcome exists.
- **Promote/resume/demote:** resume from a fresh N0 only after a complete C0
  edge/no-link assignment-cost and MAP contract is frozen and AE-P0 validates every
  emitted action. A positive action-class oracle promotes to `PARK-A`; a first
  score-separated graph failure demotes this instantiation to `PARK-C`, not DEAD.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line. Stability metrics never gate.
- **Primary source / transfer:** Murty ranks assignments by increasing cost
  ([Operations Research, 1968](https://doi.org/10.1287/opre.16.3.682)).
  Transfer domain is ranked linear assignment in operations research; it supplies an
  uncertainty construction, not evidence that APNCC costs are correctly specified.

### AE-P6 — bidirectional temporal-cycle association evidence

- **Tier:** `PARK-B / UNCERTAIN`.
- **Why:** no committed G-02 graph falsifier exists for a genuinely bidirectional
  frozen representation. This is distinct from scalar calibration and from the coarse
  one-direction track-context instantiation.
- **Committed evidence:** APNCC residual requires multi-frame consistency
  (`reports/g02/screening/appearance_residual_headroom_v1.md`,
  `7b03f95d90bb7d02f024abdc949f1877ef25af1c`); the old coarse track-context result is
  retained only as a negative control.
- **Promote/resume/demote:** first freeze forward/reverse inputs, endpoint return
  identity, no-link behavior, and a disabled-cycle beta. Promote to `PARK-A` only if
  AE-P0 shows the affected action classes have headroom; demote this instantiation to
  `PARK-C` after one negative 44b6 graph run.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line. Cycle agreement is diagnostic only.
- **Primary source / transfer:** temporal cycle consistency as self-supervision for
  visual correspondence
  ([CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.html)).
  Transfer domain is natural-video correspondence, not dividing microscopy cells;
  no-link and division break naive cycles and must be modeled explicitly.

### AE-P7 — phase-correlation multi-frame tracklet template

- **Tier:** `PARK-B / UNCERTAIN`.
- **Why:** centered pair NCC transferred, while a shift/illumination-robust
  multi-frame template is untested. Restricting it to AE-P0-valid occupied conflicts
  avoids reopening the DEAD raw-shell geometry mechanism.
- **Committed evidence:** APNCC banked gain
  (`reports/g02/screening/selected_graph_appearance_direct_v1.md`,
  `f0a4e3fc4ae2becf2934d3a1e367e5a757ce1d2a`) and residual tracklet recommendation
  (`reports/g02/screening/appearance_residual_headroom_v1.md`,
  `7b03f95d90bb7d02f024abdc949f1877ef25af1c`).
- **Promote/resume/demote:** freeze apodization, physical crop, temporal window, and
  APNCC beta before opening 44b6. Promote to `PARK-A` if its affected action-class
  oracle is positive; demote the exact template to `PARK-C` on graph failure.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line.
- **Primary source / transfer:** efficient subpixel image registration by
  cross-correlation
  ([Guizar-Sicairos et al., Optics Letters 2008](https://doi.org/10.1364/OL.33.000156)).
  Transfer domain is image registration, not biological identity; it supports robust
  displacement evidence only.

### AE-P8 — frozen dense microscopy embeddings fused into association

- **Tier:** `PARK-B / UNCERTAIN`.
- **Why:** image signal and pretrained Trackastra signal are banked, but the proposed
  dense-embedding fusion has no committed G-02 graph result and authenticated local
  encoder weights are not established by a committed artifact.
- **Committed evidence:** Trackastra qualification
  `reports/g02/screening/trackastra_pretrained_qualification_v1.md`
  (`c361e8f66b10066ab9e9076a97a97e9253e2f476`); the 6bba crop-linker added recall but
  failed transferable precision in
  `reports/g02/screening/crop_linker_6bba_oof_v1.md`
  (`50509bc8c0ec1f87919d0b896688538ed12a9243`).
- **Promote/resume/demote:** resume only with authenticated frozen weights,
  source/license/hash, offline loader proof, fixed physical crops, and an XY beta.
  A positive AE-P0 action-class oracle plus a bounded fit-only identity/runtime smoke
  promotes to `PARK-A`; missing provenance or negative graph delta demotes/parks.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line. Encoder benchmarks cannot gate.
- **Primary source / transfer:** dense self-supervised/foundation embeddings improve
  cell linking
  ([MIDL 2026 short paper](https://openreview.net/pdf/2d2323e7e5b7ecc9da47e2e748e46f75876e5d04.pdf)).
  Transfer is directly to microscopy cell linking, but not to G-02’s 3-D point
  interface or scorer.

### AE-P9 — parameter-efficient 44b6 adapter on a frozen pretrained backbone

- **Tier:** `PARK-B / UNCERTAIN`.
- **Why:** the frozen zero-wrfeat Trackastra instantiation failed 44b6, but full
  fine-tuning/retraining is excluded. A sub-1% adapter is a distinct explicit
  domain-adaptation mechanism, not a from-scratch ranker.
- **Committed evidence:** negative 44b6 graph report
  `reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json`
  (`84d02f94db5000f06ced076de5bb3d77e3e5595a`) and earlier pretrained signal
  (`reports/g02/screening/trackastra_pretrained_qualification_v1.md`,
  `c361e8f66b10066ab9e9076a97a97e9253e2f476`).
- **Promote/resume/demote:** only after AE-P0 identifies a positive action class and
  a frozen-backbone control is reproducible. Freeze adapter placement/parameter cap,
  grouped folds, early stop, and no-link beta before outcomes. Demote the exact
  adapter to `PARK-C` on 44b6 graph failure; do not infer the whole PEFT family DEAD.
- **Exact graph kill:** 44b6 adjusted `>=+0.005`, raw `>=0`, node `>=0`; then the
  identical 6bba line.
- **Primary source / transfer:** LoRA supplies low-rank parameter-efficient adaptation
  ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).
  Its demonstrated domain is large language models, not microscopy; it supports the
  bounded adaptation method, while Trackastra supplies the cell-tracking backbone.

### AE-P10 — coarse linear predecessor/successor track context, v1

- **Tier:** `PARK-C / UNLIKELY` for this exact instantiation.
- **Why:** it showed weak/negative evidence at one gate; this is not enough to kill
  temporal consistency as a family. AE-P6 and AE-P7 are new named mechanisms.
- **Committed evidence:** no fixed rule cleared the precision floor in
  `reports/g02/screening/selected_graph_tracklet_context_v1.md`
  (`d10a38c2332c846c7b25a4a138e823049206b309`); the related temporal OOF graph gate
  regressed in `reports/g02/screening/temporal_rank_oof_graph_gate_v1.md`
  (`f1669641169efc8130b939bb4e7ba9510f032107`).
- **Promote/resume/demote:** do not resume the same coarse model. Only a new named
  representation/mechanism with AE-P0-positive action-class headroom can re-enter as
  `PARK-B`.
- **Exact graph kill for any new identity:** 44b6 adjusted `>=+0.005`, raw `>=0`,
  node `>=0`; then the identical 6bba line.
- **Primary source / transfer:** Trackastra’s temporal-window association mechanism
  ([ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf))
  is direct cell-tracking evidence for richer context, but does not rehabilitate this
  failed coarse linear instantiation.

## 4. Tier census and execution order

| tier | count | nodes |
|---|---:|---|
| `LIVE` | 0 | none; this artifact authorizes no run |
| `PARK-A` | 5 | AE-P0, AE-P1, AE-P2, AE-P3, AE-P4 |
| `PARK-B` | 5 | AE-P5, AE-P6, AE-P7, AE-P8, AE-P9 |
| `PARK-C` | 1 | AE-P10 |
| `DEAD` | 0 new | human-promoted DEAD register inherited unchanged |

Cheapest-first order if the root coordinator opens execution:

1. AE-P0 action-class oracle and 100%-valid compiler.
2. AE-P1 current-policy regret arm for the near-term public-0.89 path.
3. In parallel, AE-P2 and AE-P4 class-specific expansion oracles.
4. Open at most one representation supplier (AE-P6–P9) whose target action class
   actually has positive 44b6 oracle headroom.
5. AE-P3 only after a pretrained association arm passes both families; its 5.34M
   background makes it a poor first model despite genuine recall headroom.

CV57, GPU, and a public package remain downstream of the complete both-family graph
gate. Public submission always stops for separate human approval; RunPod is forbidden.
