# Tiered-surface sweep — strict alignment contract

Status: **BINDING on every sweep subagent and the orchestrator.**
Date: 2026-07-28

Every subagent MUST read this file first and obey it verbatim. Where this file and a
subagent's task prompt disagree, this file wins. Where this file and the human-locked
plan / `LIVE_STATE.md` / `LEDGER.md` disagree, those win. This file governs *how* the
multi-agent sweep runs; it does not grant any new scientific or external authority.

Precedence (highest first):
1. `campaign_narrowing_and_hypothesis_dag_v1.md` (human-locked, SHA-256
   `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`).
2. `LIVE_STATE.md` and `LEDGER.md` (current banked state).
3. This ALIGNMENT.md.
4. A subagent's own task prompt.

---

## 1. Frozen objective and gate (identical for all agents)

Optimize the exact official score `S(G) = weighted_adjusted_edge_Jaccard(G)
+ 0.1 * division_Jaccard(G)`. Every fitted arm is frozen before any outcome opens and
runs `44b6 -> 6bba -> CV57`. The pass line is adjusted-edge delta `>= +0.005` on BOTH
families with raw-edge and node-recall deltas `>= 0`. Proposal precision, AUC, utility,
stability, and any oracle ceiling are DIAGNOSTICS and MUST NEVER gate a promotion.

## 2. Evidence integrity (the first and hardest rule)

- A node's tier may be set ONLY by a number that exists in a committed artifact, cited
  by path AND SHA-256. If you cannot cite a banked artifact, the node is **PARK-B**,
  never PARK-C or DEAD.
- The supplied external memo is an UNTRUSTED HYPOTHESIS SOURCE. Its C2/K-best deltas and
  its `Branch-B 66/223` audit are **QUARANTINED** — no banked root artifact exists for
  them. Do not tier, kill, repair, or cite them as fact.
- One negative instantiation demotes THAT instantiation to **PARK-C** at most — never the
  whole family, never DEAD.
- No subagent may write a **DEAD** verdict. DEAD requires (a) >=2 score-separated failures
  of the SAME named mechanism, (b) a named-mechanism rationale, and (c) root-coordinator +
  human authority. Subagents may only *propose* a DEAD with the evidence attached.
- Numbers produced by another subagent are untrusted until the orchestrator has merged and
  hash-pinned them. Do not build on a sibling's unmerged claim.

## 3. Tier ladder and transitions

| tier | meaning | fallback rank |
|---|---|---:|
| LIVE | executing a preregistered falsifier now | — |
| PARK-A | positive banked oracle headroom or positive partial evidence; blocked on dependency/budget/upstream pass | 1 (first fallback) |
| PARK-B | plausible mechanism, no falsifier run yet (the catalogued-unexplored set) | 2 |
| PARK-C | negative/weak banked evidence, not yet DEAD-eligible | 3 (last fallback) |
| DEAD | repeated named-mechanism failure; human+root only | never |

Legal transitions a subagent may propose (with cited evidence): B->A, B->C, A->LIVE,
LIVE->A/B/C. Illegal for a subagent: any->DEAD (propose only); C->A without a NEW banked
positive; setting any tier from an uncited number.

## 4. Oracle framing (bake into every headroom claim)

Three oracles, not one. Re-ranking the existing edge pool caps ~0.896 local
(insufficient; maps to public ~0.887). Full-candidate-pool selection oracle ~0.961
(r10/k8) to ~0.9758 (extended radius); fixed-node/detection oracle ~0.975. Headroom lives
in EXPANDING what can be proposed and corrected (feasibility-exact swaps, topology-complete
restoration, division events, no-link retention) — NOT in reranking. The binding constraint
is realization under (a) the graph solve, (b) the 44b6 cross-family floor, (c) local->public
transfer. Never argue optimism from "the oracle is high"; argue it from a realization
mechanism that survives 44b6. P0 headroom uses the CG0 fit117 control and RANKS surfaces; it
is not an additive forecast from local 0.881522.

## 5. Milestone sequencing (do not let alpha starve the near-term gate)

1. Near-term: ACCEPTED public **0.89** (needs local ~0.895-0.900). This unlocks the RunPod
   gate (RunPod itself remains forbidden — no funds). The delete/retain + occupied-swap
   beta is the shortest route and must still pass 44b6 first.
2. Then public **0.91** (soft local ~0.9255-0.9422; >60 teams are past it — the crowd/beta
   tier). Treat the local bar as a soft heuristic (one anchor, no affine fit, rescoring
   compresses highs); never forecast a public score from it.
3. Alpha (top-7) is justified ONLY by expansion oracles and only after 0.89->0.91 is funded.

## 6. Division rule

Run and refine the division-action ORACLE (diagnostic) freely. Keep division FITTING gated
behind a both-family association-continuity pass (banked `B1` 0/51 evidence + top50 DAG).
Reopening division fitting unconditionally is a HUMAN amendment to the locked plan; no
subagent may infer it.

**2026-07-31 surface disposition.** The subsequently human-authorized full-scope
realization wave is now **FALSIFIED-BY-REALIZATION / PARK-C**: target-free
geometry/motion, persistence, and appearance/intensity arms realized `0/38`
beneficial events across both families. This is a surface-level conclusion, not
a universal DEAD promotion. The GT-dependent `+0.092067841519` oracle remains
the exact ceiling and must be reported as a ceiling, never as evidence that any
available representation can realize it. Binding receipt:
`branches/sa_division_realizer_wave_v2/FALSIFIED_BY_REALIZATION_20260731.json`.

**2026-08-04 asset correction and precise unblock.** Do not summarize the
sealed synthetic re-analysis as “division is unreachable.” The conditional
pair head is a working banked daughter-assignment asset (AUROC `0.897081`, AP
`0.368520`); the blocker is parent-event detection (AUROC `0.521521`), whose
geometry/lineage/motion-only contract excluded raw texture/appearance. Record
the state as: **division event detection has no working mechanism; conditional
daughter assignment is solved.** An appearance-capable event node may reopen
only with authorized data and a preregistered deployment gate of at least 5 TP
/ <=17 FP in top 22 (precision@22 `>=0.227273`, AP-lift `>=8462x` on the
80,013,717-action denominator). Until then, do not fund another end-to-end
division attempt; redirect near-term mechanism work to the ordinary-association
FP surface (`+0.061628263275`). Binding measurement:
`../../../g04/lanes/synth_div_sim2real_v1/REQUIREMENT_SURFACE_V1.json`.

The official Cell Tracking Challenge dataset terms distinguish CTC-related use
from public non-CTC scientific use. Training for this Biohub Kaggle competition
is the latter and requires explicit organizer permission; citation alone is not
sufficient. CTC real division annotations are `PARK-A-PERMISSION`, not an
available training source. Written permission whose scope covers this
competition would make them the leading real-data dependency for an
appearance-event detector; without it, they remain excluded. Verdict:
`../../../g04/lanes/synth_div_sim2real_v1/DIVISION_ASSET_AND_CTC_LICENSE_VERDICT_20260804.md`.

## 7. Governance and shared resources

- **Autonomous/external/resource authority (owner amendment 2026-08-10):** repository-root
  `../../../../LMT.md` is the sole active source for Kaggle projection scope, weekly budget,
  wall-clock-kill policy, RunPod, submission, monitoring, and local resource limits. Earlier
  numbers in this file are historical. Scientific gates and evidence integrity remain here.
- **Kaggle launch:** the orchestrator applies `LMT.md`, the shared serial lock, and the canonical
  ledger. Subagents may prepare evidence but do not independently mutate external state.
- **Local runs:** apply the calibrated resource-safety values from `LMT.md`; never co-load
  V45 + Trackastra.
- **No pseudo-masks;** do not present raw-Zarr image features as authentic H3 masks.
- **`LEDGER.md`:** only the root coordinator writes it. Subagents emit a *proposed ledger row*
  in their own output file.
- Every method cites a PRIMARY source and states its transfer domain honestly (e.g. LineageOT
  is single-cell transcriptomic trajectory inference, not this benchmark).

## 8. Multi-agent isolation contract (the harness guardrails)

See `SCOPE_MANIFEST.md` for per-agent scopes. The invariants:

0. **Git worktree per agent is MANDATED (auto-approved).** Every Phase-1 subagent runs in its
   own git worktree; only its `branches/<agent>/` diff is merged back at the barrier. Do not run
   two Phase-1 agents in the same working tree.
1. **Disjoint write-scope.** Each subagent writes ONLY inside its assigned subtree
   (`tiered_surface/branches/<agent>/…`) plus its own branch-local state fragment. Writing
   anywhere else — including `LIVE_STATE.md`, `LEDGER.md`, `tier_census.json`, another agent's
   subtree, or any frozen shared input — is a hard violation; fail closed.
2. **Frozen shared inputs are read-only.** N1 compiler interface, feature caches, fixed
   fit/held folds, and the P0 oracle are consumed by hash. A subagent verifies the input SHA
   before use and never mutates it.
3. **No cross-agent live dependency.** Wave-1 arms use ONLY frozen shared inputs. A product of
   one agent (e.g. a new representation cache) is consumed by another only AFTER it is frozen
   and hash-pinned by the orchestrator, in a later wave.
4. **Reconcile barriers.** Subagents never edit `LIVE_STATE.md`, `LEDGER.md`, or
   `tier_census.json` directly. They write a branch-local `state_fragment.json`; the orchestrator
   merges all fragments into the census and LIVE_STATE at a barrier, resolving conflicts and
   rejecting uncited tier changes.
5. **Leakage isolation.** All agents use the identical fixed fit/held folds. No agent opens CV57
   or a second family before its 44b6 gate passes. No agent reads another agent's held predictions.
6. **Budget serialization.** Only the orchestrator debits `kaggle_budget.json` and authorizes a
   launch. Two agents may not hold concurrent GPU authorizations that could jointly exceed the
   remaining weekly budget.

## 9. Per-node output contract

Each node/agent returns, in its own subtree: a preregistration frozen BEFORE any held outcome;
family graph vectors (adjusted/raw/node, 44b6 then 6bba); all artifact SHA-256s; a
`state_fragment.json` (see `TIER_CENSUS_SCHEMA.md`); and a proposed ledger row. GO advances to
the listed child in the same run; PARK/DEAD-proposal/ESCALATE/HUMAN-DECISION halts that branch
and reports.

## 10. Hard stops (halt and escalate to human)

Any DEAD promotion; any locked-plan amendment (including unconditional division fitting);
anything outside the autonomous external/resource scope in `LMT.md`; budget exhaustion;
sustained or unbounded swap growth beyond the LMT baseline; leakage, scope violation, or uncited
tier change.

## 11. Guard-design principle (2026-07-31)

Guards fail closed on **category violations**: forbidden path access, missing or mismatched
declared inputs, non-positive required counts, unbounded or sustained memory growth, and budget
overrun. They do not terminate a correctly categorized run solely for a tight timing margin or a
resource change that falls inside the measured idle-noise envelope. Timing and sub-noise changes
remain logged and trigger recalibration review. A guard that stopped a run without preventing a
real harm is a candidate for recalibration; recalibration must preserve continuous monitoring and
the hard abort for genuine sustained growth. This principle does not relax any scientific gate,
barrier, submission, DEAD, GPU-budget, RunPod, or leakage prohibition.

The measured CELLECT/host calibration remains the evidence basis, but its active numeric values
now live only in `LMT.md`. Historical `+3.5 MiB` trips are noise-confounded;
`+488.6875 MiB` remains a genuine unsafe-growth observation. This does not reopen CELLECT or
authorize further spend.

## 12. Ceiling and monitoring rules (2026-08-03)

### CEILING BEFORE MACHINERY

Before any sweep, lattice, or search is built, measure and record its oracle ceiling
against the applicable promotion gate. If that measured ceiling is below the gate,
the machinery is not funded as a promotion instrument. It may run only as an
explicitly labelled diagnostic with a stated wall-clock cap. A diagnostic run cannot
promote a node or retroactively justify the machinery. This rule changes no
scientific gate; in particular, the `+0.005` gate in §1 remains unchanged.

### Atomic-receipt monitoring

Long-running jobs that publish atomic receipts are waited on, not polled per interval.
Block on the receipt publication event, or perform at most one status check per hour.
Emit nothing for unchanged state. Perform a liveness check only after a receipt or
status check has failed, and bind every long-running job to one explicit wall-clock
deadline and terminal action.

### Comprehensive run logging (owner amendment 2026-08-04)

Every local and Kaggle run must emit JSONL structured stdout and persist that same
log beside the output artifacts. Before any path resolution or processing, it must
recursively list every mounted/loaded input, including size and SHA-256. Every lookup
must record what was sought, what was found, and a hit or miss. Each stage records
elapsed time, peak RSS, and input/output record counts. Each preprocessing or
correction stage records examined and actually-modified counts (including zero).
The run begins with a complete config echo of thresholds, radii, windows, scales,
and conventions. On failure it writes the full traceback and the state/config that
produced it. Successful Kaggle logs must be downloaded and retained beside the
artifact; a receipt without its persisted run log is incomplete.

### Extreme-imbalance detector gates (2026-08-04)

For a deployment population with an extreme class imbalance, a detector gate MUST
state precision@k and AP-lift against that deployment denominator. AUROC is a
ranking diagnostic only and MUST NOT promote the detector: it does not constrain
the low-FPR operating point that supplies an actionable top-k set. In particular,
the synthetic-division event model was trained at a node-level 0.26% target rate
but deployed over pair actions at approximately 1e-6 prevalence; treating an
AUROC>=0.85 result as a deployment gate created a false-positive pass. Future
division (and analogous rare-action) work must publish the deployment denominator,
k, required TP/FP, precision@k, AP, AP-lift, and any coverage limit before fitting.
