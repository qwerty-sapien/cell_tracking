# GLOBAL-REASSIGN-ORACLE-V1 preregistration

**State:** frozen before any new oracle value is read. This file will be SHA-256
hashed, committed on `main`, and authenticated by the execution harness before it
can open the frozen GT archive. Every scientific quantity in this document is
`[GT-DEPENDENT]`.

## Why this node exists

Two banked facts have not previously been priced together. The frozen r10/k8
candidate tensor contains `73,999/74,718 = 99.04% [GT-DEPENDENT]` of clean
one-to-one GT edges, while `626/673 = 93.02% [GT-DEPENDENT]` of held-44b6 edge
false negatives are absent from the deployed action population. The candidate
edge is usually present; the pointwise free-add/node-disjoint-swap vocabulary
cannot represent the incumbent-displacement chain needed to install it. This
node measures that action-vocabulary ceiling. It is not a selector, a training
run, a representation study, or a realized gain estimate.

## Frozen scope and order

- Local CPU only. No Kaggle asset, dataset, kernel, push, GPU debit, Kaggle CPU
  kernel, competition submission, RunPod action, or other external action.
- Experimental scope: the full frozen 44b6 APNCC-DIRECT population, exactly all
  `36 [GT-DEPENDENT]` videos, with no subsampling. The original T1 run_v6 delete
  control is an integrity-only replay of its exact frozen `41 [GT-DEPENDENT]`
  video population; it is not used in the experimental T2-minus-T0 arithmetic.
- Tier order is T3, then T0, then T1, then T2 at every
  `L in {2, 3, 4, unbounded} [GT-DEPENDENT]`. No tier may be selected after an
  outcome is seen.
- No AUC, selector, training, representation, threshold search, 6bba, CV57,
  held-fold, public-score, tier-move, DEAD/PARK promotion, plan amendment, or
  LEDGER mutation is in scope.

## Frozen inputs

Repository pre-preregistration HEAD is
`8ca1b73331be8e6fef24ec9c6b3d7ff641f98a4e` on `main`.

| Input | SHA-256 |
|---|---|
| `dag_prompts/2026-08-01-global-reassign-oracle/CODEX_DISPATCH.md` | `cf215c655c00a1bd5b54b6ce13df185c98641d198656e5ffae4675644102de62` |
| `reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json` | `530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d` |
| APNCC 36-fragment logical aggregate | `48b99c5db7024c0998b779d2dfcf30bade9c12d74ec679325cc1974ceb9f7ccb` |
| `reports/g02/extended_radius_reachability_v1.json` | `22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803` |
| `reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json` | `8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06` |
| `reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1/population_manifest.json` | `4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c` |
| `reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json` | `28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a` |
| `reports/g02/screening/tiered_surface/branches/sa_swap/PREREGISTRATION.json` | `00e6036ef076482a39266d797c5fea3e2e4845ba4262c8532cfe70bb8ccb4628` |
| raw 44b6 archive, banked P0 pin; exact bytes `87,393,127,165` | `538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19` |
| frozen N1 interface | `2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865` |
| `src/celltrack_alpha/action_compiler.py` | `457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5` |
| `src/celltrack_alpha/official_eval.py` | `22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6` |
| `scripts/scoreall.py` | `7b92fed9fe66e9a2faf062528a78b04a65bc1be50f806dff44f51c780da5ac6c` |
| official `tracking_cellmot/metrics.py` | `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444` |
| official `tracking_cellmot/division_metrics.py` | `0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9` |
| `reports/g02/screening/selector_requirement_44b6/selector_requirement.py` (control replay only) | `f9a9cc60ae0c199a8a39e20be50a45803ed4ad166947c8f07cef86131fea1555` |
| `reports/g02/screening/selector_requirement_44b6/selector_requirement_class.py` (control replay and GT topology only) | `0f5f6ca1e29616f4e2a0d7fe4ad79f5311eaac6d0353fde8f60aeab535100755` |
| `dag_prompts/2026-08-01-global-reassign-oracle/global_reassign_oracle.py` | `674a44b82bb862369a47d1745d442f8c40d0a668149ed5ecfbf4bc21384a4db0` |
| `dag_prompts/2026-08-01-global-reassign-oracle/run_guarded.py` | `c5fbbc5b0036e07340f4b66a200a78a0fa1abe1e3c95b145019039324bfc1758` |
| `dag_prompts/2026-08-01-global-reassign-oracle/test_global_reassign_oracle.py` | `1cecab618ad717df4f336dc645d188be7967f14bdae13faf610e33b004c4c471` |
| `scripts/run_local_training_guard.py` (resource observation only) | `520c14db2e53b0053a931c708a533d87691eda82885a817d617c230839f86ced` |

The harness rehashes all scorer/compiler/core input files and every APNCC graph
fragment before opening GT. The raw archive's banked SHA and exact byte size are
cross-checked against P0; the 87-GB archive is not redundantly rehashed.

## Exact estimands

### T3: free global reassignment

For each frozen baseline node set, reconstruct exactly the physical-distance
r10/k8 next-frame candidate tensor: radius `10.0 um [GT-DEPENDENT]`, at most
`8 [GT-DEPENDENT]` candidates per source, ordered deterministically by distance,
target node ID, then target-array position. Reproduce each video's banked pool
size and clean reachable count. Use the frozen official one-to-one node matching
to retain every and only GT-correct candidate edge. The resulting all-correct
subgraph is the GT-optimal legal subgraph because it has no false-positive edge,
preserves forward time, has indegree at most one, and permits GT division
outdegree at most two. It ignores baseline edges entirely and is scored by the
frozen official scorer.

If `T3 adjusted delta < +0.030 [GT-DEPENDENT]`, emit `STOP_CLASS`, skip T0/T1/T2,
and terminate. Equality does not stop.

### T0 and T1: integrity controls

- T0 replays the exact original node-disjoint occupied-swap oracle at candidate
  prefix `32,768 [GT-DEPENDENT]`; required adjusted delta is
  `+0.018571397 [GT-DEPENDENT]`.
- T1 replays the exact original delete oracle at prefix
  `2,048 [GT-DEPENDENT]`; required adjusted delta is
  `+0.081897667 [GT-DEPENDENT]`.

Each official replay must agree with its fixed value to absolute tolerance
`1e-9 [GT-DEPENDENT]`. T1 runs only after T0 passes. Any mismatch emits
`INVALID_HARNESS` and stops before T2; tolerance and targets cannot be changed.

### T2: closure-complete chained conflict swap

The target edge set is the GT-correct subset of the same frozen r10/k8 tensor.
A chain begins with one correct target edge absent from the baseline. Its atomic
replacement boundary owns every baseline edge whose source is an added-edge
source or whose target is an added-edge target. Each displaced wrong edge links
recursively to every missing correct tensor edge that re-homes its source or
target. Each displaced baseline-true edge is re-added as a retained anchor; its
endpoints recursively expand the same closure. Correct division daughters that
share a source therefore belong to one closure. Iterate to a fixed point.

One fixed-point connected component is one indivisible chain. Chain length is
the number of missing correct tensor edges installed; retained true anchors do
not count. A finite-L tier admits every complete chain with length `<= L` and
rejects the whole chain otherwise. Partial chains are forbidden. Baseline edges
outside selected transaction boundaries remain unchanged. Selected chains are
compiled individually and as an order-independent batch through the frozen N1
compiler; the analysis graph must hash-identically to the compiled graph. Exact
graph validation forbids duplicate edges, backward-time edges, indegree above
one, and outdegree above two.

Report L=`2`, `3`, `4`, and `unbounded [GT-DEPENDENT]` in that order. For every
tier report exact frozen-official adjusted, raw, node, and combined deltas versus
the same banked 36-video APNCC baseline, plus correct edges installed, false
positives removed, compiled chains, and the complete chain-length histogram.
If any winning chain is inexpressible in the frozen compiler, continue only far
enough to preserve the analysis finding, then emit `ESCALATE_COMPILER_INEXPRESSIBLE`;
the scorer and compiler remain untouched.

## Decision arithmetic

Only after all controls and T2 tiers complete, compute the raw oracle increment
`T2(unbounded adjusted delta) - T0 adjusted delta [GT-DEPENDENT]`. Report first
the campaign-precedent expectation:

`0.2449 [GT-DEPENDENT] * (T2(unbounded) - T0)`.

If that precedent-adjusted value is `< +0.005 [GT-DEPENDENT]`, emit
`STOP_CLASS`. Otherwise emit `CONTINUE_PROPOSAL_ONLY`; no follow-on action is
authorized. Equality continues. Thresholds and arithmetic are immutable after
any outcome is read.

## Resource, integrity, and terminal rules

- Require a 45-second clean-idle preflight before GT access. Abort with ESCALATE
  on immediate swap growth at least 384 MiB, or growth at least 256 MiB sustained
  for 15 seconds. Apply the same swap guard throughout the child run.
- Hard wall-clock limit is 6 hours. After at least three T2 videos, project total
  wall time by completed r10/k8 candidate-pair work. If projection exceeds 6
  hours, emit `ESCALATE_PROJECTED_WALL_CAP`; never reduce coverage.
- Write T3, T0, T1, and T2 fragments one video at a time to a temporary sibling
  and atomically rename. A resumed fragment must match schema, stage, video,
  preregistration SHA, and (for controls) the selected-action hash.
- Report measured total wall-clock, preflight duration, peak process-tree RSS,
  and swap growth. Preserve all fragments on every terminal path.
- Do not modify the official scorer, N1 compiler, campaign plan, or
  `reports/g02/screening/LEDGER.md`. Produce proposals only. Update
  `reports/g02/screening/LIVE_STATE.md` once with the terminal result.

The outcome-free unit suite passed 7/7 checks before this seal. No new T3 or T2
oracle value was computed or read before this preregistration.
