# Event and node-action surface — tiered DAG inventory

Date: 2026-07-28  
Scope: division forks, one-missed-frame gap bridges, node addition/restoration, and
detection-derived node actions  
Status: **INVENTORY ONLY / NO ORACLE OR MODEL RUN / NO PLAN AMENDMENT**

## Authority and evidence rule

This branch inherits the exact frozen objective

`S(G) = weighted_adjusted_edge_Jaccard(G) + 0.1 * division_Jaccard(G)`

and the sequence `44b6 held-fit graph → 6bba held-fit graph → CV57`. Proposal
precision, AUC, utility, candidate recall, and oracle ceilings are diagnostic only.
Every fitted arm must be frozen before a held fold opens and must clear adjusted
edge-Jaccard `>= +0.005` on each family with raw edge-Jaccard and node recall
nonnegative, unless a stricter node below is stated.

Only committed local artifacts establish tiers. The uncommitted
`top50_seeking_alpha_research_20260727.md` is used only as a hypothesis catalogue,
not as result evidence. In particular, no unbanked C2 or “Branch B” number is used.

The two locked sequencing rules remain intact:

1. A division-action oracle may run now as a diagnostic, but **division fitting
   remains closed** until an association-continuity arm passes both families and
   creates connected fork opportunities. The banked reason is `0/51` restored fork
   edges.
2. A node-add/node-restore oracle may run as a diagnostic, but **detection fitting
   remains conditional-PARK** until the locked plan's association prerequisite is
   satisfied. Crossing either boundary requires the root coordinator and, where it
   changes the locked plan, the human owner.

### Committed evidence pins

| Pin | Artifact | SHA-256 | What it can establish |
|---|---|---|---|
| `E-DIV` | `reports/g02/division_sandbox_and_holdout_decomposition_v1.md` | `2478f76f1e6ab5ba962bf4c2379323cae368898e61c8888a5df06e05c266e9cc` | Scorer-valid direct/late/persistent fork semantics; 35/37 stages present but fragmented; legacy helper 0/51. |
| `E-B1` | `reports/g02/screening/B1_division_tp.md` | `c9f2b4d1b8e33f5dae456de9e0ac1649b08f2cda8591a67b9d00d4a355dffbb1` | Association-continuity dependency and division-fitting PARK. |
| `E-CG0` | `reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md` | `486142315d7b9f47749a1164571aa0f1603db439c884b8e628d2ad6806614f51` | Candidate-policy ceilings and one-missed-frame gap opportunity; not a realizable action-class result. |
| `E-FIXNODE` | `reports/g02/t04_v45f4_fold0_fixed_node_oracle/report.json` | `853a5df0f2409ac0131c1b6a2d1b4793d53ee41357f5e3a964238f68a4e148a9` | Fixed-prediction-node edge upper bound only; it cannot establish node-add or node-restore headroom. |
| `E-GD985` | `reports/g02/screening/node_threshold_screen_v1.md` | `4736215dfb7bf04fe4da7ad312d676cbabeacebe08a83925a25f9aa267348b99` | One detection-pruning instantiation: adjusted positive but raw and recall negative. |
| `E-PLAN` | committed `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md` | `f1d2720e55cc806a3db5d036fa43e9972344e32c47c65a4720032868c79f2638` | Human-locked conditional-PARK status and graph-gate order. |
| `E-LIT` | `docs/research_strategy_decomposition.md` | `53612cd0e11dd6c4b75ac621123b8a9c8313c2489d0d9e49ff7e842b960ceaf2` | Literature inventory and transfer hypotheses only; no local score evidence. |

## Initial branch census

This census covers only the twelve nodes in this file, not the campaign-wide census.

| Tier | Count | Nodes |
|---|---:|---|
| LIVE | 0 | — |
| PARK-A / PROMISING | 2 | `DIV-F1`, `GAP-O0` |
| PARK-B / UNCERTAIN | 8 | `DIV-O0`, `DIV-F2`, `GAP-F1`, `NODEADD-O0`, `NODERESTORE-O0`, `DET-F0`, `DET-F2`, `DET-F3` |
| PARK-C / UNLIKELY | 2 | `DIV-F0`, `DET-F1` |
| DEAD | 0 | No DEAD promotion is made here. |

## Shared oracle and feasibility contract

Every action-class oracle starts from the exact frozen APNCC graph and reports, for
`44b6` and `6bba` separately:

- `delta_S`, adjusted/raw edge-Jaccard deltas, node-recall delta, and division
  `TP/FP/FN/Jaccard`;
- action count, videos affected, complete endpoint/incident-edge mutation, and an
  observation-distinct graph hash;
- a **zero-harm budget** first: zero false forks for division, zero incorrect bridge
  edges for gap actions, and zero unmatched/background added nodes for node actions;
- a stress curve that adds harmful actions in a fixed deterministic order. No
  deployment claim may be inferred from the perfect zero-harm point.

Before any support set freezes, the shared feasibility-exact compiler must prove that
every retained action:

- references endpoints present in the final APNCC graph or explicitly introduces a
  provenance-pinned node;
- carries every incident conflict and removes/replaces it atomically;
- preserves temporal direction, at most one parent, at most two children, no merge,
  and no dangling endpoint;
- passes full-graph validation after each single action and after the combined batch.

Failure to reach 100% valid retained actions is `PARK`/contract repair, not evidence
against the scientific family.

```text
committed-evidence audit
  ├─ division oracle ──> [association-continuity dependency] ──> fork fitting
  ├─ gap oracle ───────> feasibility-exact bridge builder ─────> gap fitting
  └─ node oracles ─────> [locked detection dependency] ───────> detection fitting

any fitted arm: 44b6 first ──GO──> 6bba ──GO──> CV57
                              └──PARK/ESCALATE: stop
```

## Division-fork branch

### `DIV-O0` — exact division-action oracle

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `reports/g02/division_sandbox_and_holdout_decomposition_v1.md`
  @ `2478f76f1e6ab5ba962bf4c2379323cae368898e61c8888a5df06e05c266e9cc`.
- **Why / evidence:** scorer semantics and a nonempty event surface are banked in
  `E-DIV`, but no committed artifact isolates the exact 44b6 and 6bba
  division-action deltas under the current APNCC graphs. The older 42-event CV57
  count is not a family-first oracle result.
- **Diagnostic action:** enumerate scorer-valid binary fork actions, including the
  direct fork, one-frame-late fork, and grandchild-supported branch classes proven by
  the official sandbox. Apply only complete atomic fork actions.
- **FP budget:** first point is zero false forks; additionally report a fixed
  `1/5/10` false-fork stress curve, never selecting a threshold from held outcomes.
- **Promote:** PARK-A only if the exact oracle is observation-distinct, its
  zero-false-fork `delta_S >= +0.005` on both families, and edge raw/adjusted and
  node recall are nonnegative. Promotion does **not** open fitting.
- **Demote:** PARK-C if 44b6 zero-false-fork `delta_S <= 0`; stop before 6bba.
- **Resume:** diagnostic may resume when the root action-table contract and frozen
  APNCC/scorer hashes are pinned. Fitting remains behind `E-B1`.
- **Preregistered 44b6-first kill:** kill the oracle cascade at
  `delta_S_44b6 <= 0`, any raw/adjusted edge or node-recall regression, invalid
  action, or null graph; do not open 6bba.
- **Primary source / transfer domain:** the competition's official
  [`division_metrics.py`](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/src/tracking_cellmot/division_metrics.py)
  is the direct metric domain (checkout and source hash are pinned in `E-DIV`), not
  an external benchmark transfer.

### `DIV-F0` — legacy immediate-orphan `add_safe_divisions`

- **Tier:** PARK-C / UNLIKELY for this instantiation; the division family is not DEAD.
- **Evidence artifact:** `reports/g02/screening/B1_division_tp.md`
  @ `c9f2b4d1b8e33f5dae456de9e0ac1649b08f2cda8591a67b9d00d4a355dffbb1`.
- **Why / evidence:** `E-DIV` and `E-B1` record zero edges added on 13 valid fixtures
  and `0/51` exact counterfactual restores. This is one score-semantic mechanism
  failure, not two score-separated failures of every fork generator.
- **Promote:** only through a **new named mechanism** that constructs connected
  daughter lineages rather than widening the legacy distance constants, then passes
  `DIV-O0` and the association-continuity dependency.
- **Demote:** no DEAD promotion here; root/human authority would require repeated
  score-separated failures of the same named mechanism.
- **Resume:** do not resume this exact helper or tune its inherited geometry. It may
  remain only as the negative control for a new fork builder.
- **Preregistered 44b6-first kill:** any new instantiation is killed at
  adjusted edge delta `<= 0`, raw/node regression, division `delta_S <= 0`, or any
  duplicate/merge survivor; 6bba stays closed.
- **Primary source / transfer domain:** official competition division fixtures and
  scorer, direct domain; the local helper itself is the tested implementation, not a
  literature-derived method.

### `DIV-F1` — scorer-semantic late/persistent fork family

- **Tier:** PARK-A / PROMISING, blocked.
- **Evidence artifact:** `reports/g02/division_sandbox_and_holdout_decomposition_v1.md`
  @ `2478f76f1e6ab5ba962bf4c2379323cae368898e61c8888a5df06e05c266e9cc`.
- **Why / evidence:** `E-DIV` proves that direct, one-frame-late, and
  grandchild-supported connected forks can receive credit, and reports necessary
  matched stages for 35/37 audited GT divisions. The same artifact shows those stages
  are fragmented, so this is positive mechanism evidence with an explicit upstream
  block—not fitted performance evidence.
- **Mechanism:** propose one parent-to-two-branch action with daughter persistence and
  one-to-one duplicate suppression; compare an exact-time beta with the preregistered
  late/persistent alpha.
- **Promote:** eligible for LIVE fitting only after one association arm has immutable
  adjusted `>= +0.005` on 44b6 and 6bba, raw/node nonnegative, and a measured positive
  delta in connected fork opportunities. This is the banked dependency, not an
  inferred plan amendment.
- **Demote:** PARK-C if `DIV-O0` has no 44b6 headroom or, after the dependency opens,
  the first fixed scorer-semantic instantiation fails the 44b6 line.
- **Resume:** from the oracle/continuity gate; never from a relaxed distance constant.
- **Preregistered 44b6-first kill:** adjusted edge delta must be `>= +0.005`,
  raw/node nonnegative, combined `delta_S > 0`, division Jaccard `>= 0.30`, and zero
  duplicate fork survivors. Any miss halts before 6bba.
- **Primary source / transfer domain:** Trackastra
  ([ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf))
  directly models associations and divisions in live-cell microscopy, but its
  pretrained datasets and decoder are not this competition's scorer. The late-fork
  equivalence is grounded separately by the direct-domain official scorer in `E-DIV`.

### `DIV-F2` — native pretrained Trackastra fork beta

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** committed
  `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`
  @ `f1d2720e55cc806a3db5d036fa43e9972344e32c47c65a4720032868c79f2638`.
- **Why / evidence:** `E-LIT` and Trackastra establish a plausible pretrained
  division-capable beta, but no committed family graph result exists for its native
  fork decode. It cannot inherit association evidence from a different head.
- **Promote:** PARK-A after `DIV-O0` is positive and a qualifying association arm
  increases connected fork opportunities; LIVE only after the locked dependency is
  satisfied.
- **Demote:** PARK-C after one fixed native-decoder instantiation fails 44b6; do not
  demote the entire pretrained-division family.
- **Resume:** only with a frozen checkpoint/interface, exact action compiler, native
  decode contract, and unchanged no-division APNCC beta.
- **Preregistered 44b6-first kill:** adjusted edge delta `< +0.005`, raw/node
  regression, `delta_S <= 0`, division Jaccard `< 0.30`, or duplicate fork survivor;
  no 6bba.
- **Primary source / transfer domain:** Trackastra
  ([ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf));
  direct task class (cell-tracking lineage), external microscopy datasets and metric.

## One-missed-frame gap branch

### `GAP-O0` — one-missed-frame bridge oracle

- **Tier:** PARK-A / PROMISING as a candidate-policy oracle; exact action delta is
  still pending.
- **Evidence artifact:** `reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md`
  @ `486142315d7b9f47749a1164571aa0f1603db439c884b8e628d2ad6806614f51`.
- **Why / evidence:** `E-CG0` banks 231 one-missed-frame events recovering 462 correct
  edges. Its perfect-selection policy rises from `0.960918` for current
  `r10/k8` to `0.966728` for `r10/k8 + gap`; family opportunity counts are 35
  (`44b6`) and 196 (`6bba`). These are candidate-policy ceilings, not a realizable
  bridge model and not an exact APNCC intervention result.
- **FP/background budget:** zero incorrect bridges first. The only banked broader
  policy budget is at most 3,855 background FP among 29,651,239 background
  candidates to retain the `0.92` ceiling (`99.986999%` rejection); report it as a
  realization stress bound, never as achieved performance.
- **Promote:** retain PARK-A and open `GAP-F1` feasibility work only if the isolated
  exact action oracle clears adjusted `>= +0.005` on both families with raw/node
  nonnegative.
- **Demote:** PARK-C if isolated 44b6 zero-FP delta `<= 0`; stop before 6bba.
- **Resume:** after the shared action-table contract pins the definition of a
  one-missed-frame bridge as the atomic pair of edges and all displaced conflicts.
- **Preregistered 44b6-first kill:** adjusted delta `<= 0`, raw/node regression,
  fewer than two valid time-directed edges, any dangling endpoint/merge, or null
  graph; do not open 6bba.
- **Primary source / transfer domain:** Malin-Mayor et al.
  ([Nature Biotechnology](https://www.nature.com/articles/s41587-022-01427-7))
  use sparse annotations, temporal displacement evidence, and global lineage
  optimization in whole-embryo microscopy. This supports long-range lineage
  constraints, but not this competition's exact gap action or scorer.

### `GAP-F1` — feasibility-exact temporal bridge realizer

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md`
  @ `486142315d7b9f47749a1164571aa0f1603db439c884b8e628d2ad6806614f51`.
- **Why / evidence:** `E-CG0` establishes theoretical gap opportunity but no committed
  artifact demonstrates a transferable proposer or 100%-valid final-graph bridge
  batch. Geometry/NCC/constant-velocity shell mechanisms are DEAD and cannot be
  relabeled as this arm.
- **Mechanism boundary:** a new named mechanism must use longer temporal lineage
  evidence and emit the two-edge bridge atomically; it may not be a hand-tuned motion
  cost or threshold-only repair.
- **Promote:** PARK-A after `GAP-O0` passes and the compiler validates 100% of retained
  bridges; LIVE only under a frozen, leakage-safe realization contract.
- **Demote:** PARK-C if the first fixed new mechanism fails 44b6; the broader gap
  family remains PARK-B unless repeated evidence supports stronger action.
- **Resume:** exact bridge-oracle output, complete endpoint/conflict inventory, and
  fixed APNCC fallback must all be pinned.
- **Preregistered 44b6-first kill:** adjusted delta `< +0.005`, raw/node regression,
  any invalid retained bridge, or failure to beat unchanged APNCC; no 6bba.
- **Primary source / transfer domain:** the same Malin-Mayor et al. primary source
  above; biological whole-embryo lineage reconstruction is adjacent, while the
  proposed atomic gap edit and frozen metric are competition-specific.

## Node-add and node-restore diagnostics

Node addition and restoration are distinct action classes:

- **node-add** creates a new coordinate/provenance identity from raw image evidence;
- **node-restore** reactivates a pre-existing, provenance-pinned candidate that was
  omitted or pruned.

Neither may cite the `0.974994` fixed-node oracle as its own headroom. `E-FIXNODE`
holds prediction nodes fixed and optimizes only GT-supported edges.

### `NODEADD-O0` — exact new-node oracle

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `reports/g02/t04_v45f4_fold0_fixed_node_oracle/report.json`
  @ `853a5df0f2409ac0131c1b6a2d1b4793d53ee41357f5e3a964238f68a4e148a9`.
- **Why / evidence:** no committed artifact isolates the 44b6/6bba score effect of
  adding new nodes. `E-FIXNODE` explicitly fixes nodes, so it is negative evidence
  against misusing that number, not evidence for this class.
- **FP/background budget:** zero unmatched/background added nodes first; report a
  deterministic harmful-node stress curve and the adjusted node-count penalty.
- **Promote:** PARK-A only if exact zero-background node-add headroom is adjusted
  `>= +0.005` on each family, raw/node nonnegative, and every new node has an
  immutable image-source and coordinate provenance.
- **Demote:** PARK-C if 44b6 zero-background delta `<= 0`; no 6bba.
- **Resume:** diagnostic oracle may run under P0. Any fitting/build remains behind the
  locked detection prerequisite in `E-PLAN`.
- **Preregistered 44b6-first kill:** adjusted delta `<= 0`, raw/node regression,
  unmatched added node, duplicate identity, dangling edge, or null graph.
- **Primary source / transfer domain:** the official competition
  [metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
  is the direct source for matching radius and node-count penalty; it provides no
  realization method.

### `NODERESTORE-O0` — exact banked-candidate restoration oracle

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `reports/g02/t04_v45f4_fold0_fixed_node_oracle/report.json`
  @ `853a5df0f2409ac0131c1b6a2d1b4793d53ee41357f5e3a964238f68a4e148a9`.
- **Why / evidence:** no committed artifact isolates restoration of omitted
  provenance-pinned candidates. The fixed-node oracle in `E-FIXNODE` does not change
  membership and cannot establish this tier.
- **FP/background budget:** zero incorrectly restored nodes first; all restored nodes
  must come from a frozen pre-held candidate bank, and incident edges are scored as
  part of the same atomic action.
- **Promote:** PARK-A if both-family exact adjusted headroom is `>= +0.005`,
  raw/node nonnegative, and the candidate bank plus join is complete and
  leakage-safe.
- **Demote:** PARK-C at 44b6 zero-FP delta `<= 0`; stop before 6bba.
- **Resume:** after the root action table defines candidate-bank identity and the
  shared compiler supports atomic node-plus-incident-edge restoration. Fitting
  remains detection-gated.
- **Preregistered 44b6-first kill:** adjusted delta `<= 0`, raw/node regression,
  candidate absent from the frozen bank, ambiguous provenance, or invalid graph.
- **Primary source / transfer domain:** Ultrack
  ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0))
  selects among multiple segmentation hypotheses using temporal and biological
  constraints in 2-D/3-D+t microscopy. That supports keeping alternate node
  hypotheses, but it uses segmentation hypotheses and its own objectives—not sparse
  GEFF center-node restoration under this scorer.

## Detection-derived node branch

### `DET-F0` — detection family parent

- **Tier:** PARK-B / UNCERTAIN and conditional-PARK.
- **Evidence artifact:** committed
  `reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md`
  @ `f1d2720e55cc806a3db5d036fa43e9972344e32c47c65a4720032868c79f2638`.
- **Why / evidence:** `E-PLAN` locks detection behind association. `E-GD985` is one
  negative pruning instantiation only; it cannot demote the whole detection family.
  No isolated node-add/node-restore oracle has run.
- **Promote:** PARK-A only after `NODEADD-O0` or `NODERESTORE-O0` provides positive
  both-family headroom. LIVE fitting additionally requires the locked association
  prerequisite or an explicit human amendment.
- **Demote:** retain PARK-B if an individual child fails; PARK-C for a named child
  after its first negative 44b6 result. No family DEAD promotion here.
- **Resume:** oracle diagnostics first; then a root/human decision if the plan
  dependency is still unmet.
- **Preregistered 44b6-first kill:** for every child, adjusted delta `< +0.005`,
  raw/node regression, or non-distinct output halts before 6bba.
- **Primary source / transfer domain:** official competition baseline
  ([repository](https://github.com/royerlab/kaggle-cell-tracking-competition))
  is direct-domain sparse 3-D+t detection/linking; it does not establish a local
  improvement for any new detector.

### `DET-F1` — detector-threshold pruning at `0.985`

- **Tier:** PARK-C / UNLIKELY for this instantiation; detection family remains
  PARK-B.
- **Evidence artifact:** `reports/g02/screening/node_threshold_screen_v1.md`
  @ `4736215dfb7bf04fe4da7ad312d676cbabeacebe08a83925a25f9aa267348b99`.
- **Why / evidence:** `E-GD985` records adjusted `+0.002746`, raw `-0.000775`, and
  node recall `-0.004125`. It misses both the effect-size and non-regression gates.
- **Promote:** do not retune this scalar threshold. A successor must be a new named
  node-action mechanism and begin at the exact oracle/feasibility gates.
- **Demote:** no DEAD promotion; this is one negative instantiation.
- **Resume:** negative control only.
- **Preregistered 44b6-first kill:** unchanged common line—adjusted `< +0.005` or
  raw/node negative halts; the banked instantiation already fails that rule.
- **Primary source / transfer domain:** direct local official-scorer evidence in
  `E-GD985`; no external method claim is attached.

### `DET-F2` — Ultrack-style multi-hypothesis node selection

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `docs/research_strategy_decomposition.md`
  @ `53612cd0e11dd6c4b75ac621123b8a9c8313c2489d0d9e49ff7e842b960ceaf2`
  (literature hypothesis only).
- **Why / evidence:** `E-LIT` supplies a primary-method rationale, but there is no
  committed local mask/hypothesis bank, no exact node-action oracle, and no family
  graph result.
- **Mechanism boundary:** generated segmentation hypotheses would be new
  detection-proposal evidence. They must never be called authentic H3 masks or used
  to satisfy H3 provenance.
- **Promote:** PARK-A after a provenance-pinned local hypothesis bank improves
  `NODEADD-O0` or `NODERESTORE-O0` on both families and passes 100% feasibility.
- **Demote:** PARK-C if a single frozen multi-hypothesis instantiation fails 44b6;
  do not demote all image-derived detection.
- **Resume:** only after the locked detection prerequisite or explicit amendment;
  external weights/data and GPU remain separate permission gates.
- **Preregistered 44b6-first kill:** adjusted `< +0.005`, raw/node regression,
  any unmatched-node/background-budget breach, invalid graph, or null output.
- **Primary source / transfer domain:** Ultrack
  ([Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0));
  close microscopy domain including large 3-D+t embryonic data, but segmentation-
  hypothesis inputs and evaluation differ from this sparse-center competition.

### `DET-F3` — sparse center/restoration lineage model

- **Tier:** PARK-B / UNCERTAIN.
- **Evidence artifact:** `docs/research_strategy_decomposition.md`
  @ `53612cd0e11dd6c4b75ac621123b8a9c8313c2489d0d9e49ff7e842b960ceaf2`
  (literature hypothesis only).
- **Why / evidence:** `E-LIT` supports sparse center heatmaps, backward displacement,
  and global lineage constraints as a mechanism; no committed local oracle or graph
  result exists. Full organizer-scale training remains outside this node.
- **Mechanism boundary:** a bounded node proposer/restorer plus atomic incident-edge
  action, not the DEAD full `TemporalUNet3D` organizer and not a from-scratch
  pair-ranker.
- **Promote:** PARK-A only after an exact node-action oracle clears both families and
  a bounded preregistered implementation is feasible under current governance.
- **Demote:** PARK-C if its first fixed bounded instantiation fails 44b6; the family
  remains untested beyond that instantiation.
- **Resume:** locked detection prerequisite or explicit plan amendment, then separate
  compute authorization if GPU is needed.
- **Preregistered 44b6-first kill:** adjusted `< +0.005`, raw/node regression,
  background-node budget breach, provenance failure, or invalid graph; no 6bba.
- **Primary source / transfer domain:** Malin-Mayor et al.
  ([Nature Biotechnology](https://www.nature.com/articles/s41587-022-01427-7))
  reconstruct whole-embryo lineages from sparse annotations using learned center/
  motion evidence and global optimization. This is a close sparse-microscopy
  transfer, but not the same organisms, annotations, graph format, or scorer.

## Cascade and stop policy

1. Root evidence audit and exact action definition.
2. Run only the zero-harm 44b6 action-class oracle.
3. On positive but `< +0.005`, retain diagnostic evidence and PARK-C the
   instantiation; on `<= 0`, halt immediately.
4. On `>= +0.005` with raw/node nonnegative, run the unchanged 6bba oracle.
5. Only a both-family oracle pass may open feasibility implementation.
6. Only 100% graph-valid actions may support a fitted arm.
7. Division and detection fitting still halt at their locked dependencies.
8. A fitted arm that passes 44b6 auto-advances to 6bba; any non-GO halts. CV57 opens
   only after both-family success.
9. Public submission, external mutation, plan amendment, and DEAD promotion are
   HUMAN-GATEs. RunPod remains forbidden.

No node in this file authorizes training, scoring, GPU use, external acquisition,
submission, a LIVE_STATE/LEDGER edit, or a change to the human-locked campaign.
