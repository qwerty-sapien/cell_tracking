# Cruise Goal reference

> User-only document. Cruise writes this file algorithmically and must never
> read, summarize, fingerprint, or use its contents as agent context.

## 2026-07-28T16:08:46.047Z — completed: Dispatch and complete a CPU/cache-only Wave-2 in isolated worktrees, bank reproducible lane artifacts, and reconcile only fully cited fragments at a root-controlled barrier without opening graph, GPU, CELLECT, Kaggle-budget, CV57 fitting, submission, RunPod, or DEAD gates.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-28T16:08:46.047Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
CELLECT-R7 is blocked on a packaging fix (separate track). Dispatch a CPU-only Wave-2 that does
NOT touch the Kaggle budget or CELLECT. Same governance: ALIGNMENT.md + SCOPE_MANIFEST.md; each
lane in its OWN git worktree; disjoint write-scope; frozen shared inputs (N1 2a72950c…, folds, P0)
read-only; orchestrator is sole writer of LIVE_STATE/LEDGER/tier_census; barrier-merge at the end;
reject uncited tier moves; quarantine memo C2/Branch-B numbers; no DEAD; 44b6-first.

DISPATCH IN PARALLEL (worktrees):
1. SA-NODETOPO-GEN (branches/sa_nodetopo_gen/) — build a deterministic AUTHENTIC node-proposal
   universe + false-node budget stress curve from the raw Zarrs, fit-only, to unblock the PARK-A
   NODETOPO-O0 surface. Emit the generator contract, coverage/identity audit, and a proposed
   graph-action population compiled through frozen N1. It builds node-RESTORATION actions only;
   it does NOT modify the detector (that is a separate human gate). No graph run.
2. SA-DIVISION-DEEP (branches/sa_division_deep/) — refine the division-action ORACLE and freeze
   the D1-PERSISTENT-FORK preregistration (late/persistent-fork equivalence classes, duplicate
   suppression, kill line). Quantify how many of the 42 CV57 divisions are present-but-fragmented.
   FITTING STAYS LOCKED behind a both-family association-continuity pass; oracle/prereg only.
3. SA-PARKB-SWEEP (branches/sa_parkb/) — run the cheapest identity/support falsifier on each
   CPU-testable PARK-B node; propose B→A or B→C with cited evidence. Skip weight/GPU-gated
   representation nodes (they wait with CELLECT). Proposes tier moves in its fragment only.
4. (optional) SA-ASSOC-CYCLE (branches/sa_assoc_cycle/) — bidirectional forward/reverse Trackastra
   consensus on the EXISTING cache, screened through the frozen separation-falsifier design (B0 vs
   this signal) BEFORE any graph spend. Park cheaply if it doesn't beat B0 separation on 44b6.

COMPUTE
- CPU/cache-only; 45s idle preflight; zero swap growth; NO Kaggle launch, NO budget debit, NO GPU.
- If any lane thinks it needs GPU, it STOPS and reports rather than launching (Kaggle serialization
  is reserved for the CELLECT track).

BARRIER
- Merge fragments into tier_census.json + LIVE_STATE; append proposed LEDGER rows; hold any graph
  gate for human review. Report per-lane outcomes and the updated tier census.
```

### Goal

**Outcome:** Dispatch and complete a CPU/cache-only Wave-2 in isolated worktrees, bank reproducible lane artifacts, and reconcile only fully cited fragments at a root-controlled barrier without opening graph, GPU, CELLECT, Kaggle-budget, CV57 fitting, submission, RunPod, or DEAD gates.

**Verification**

- Every lane reads LIVE_STATE.md, ALIGNMENT.md, and SCOPE_MANIFEST.md before acting and writes only its assigned subtree.
- Every lane authenticates frozen N1 interface 2a72950c..., folds, fit provenance, P0, and any cache it consumes before producing evidence.
- Every compute lane performs a 45-second idle preflight and fails closed on any swap growth.
- SA-NODETOPO-GEN emits a deterministic authentic fit-only raw-Zarr generator contract, coverage/identity audit, false-node budget stress curve, and N1-compiled node-restoration action population without detector modification or graph execution.
- SA-DIVISION-DEEP freezes late/persistent-fork equivalence, duplicate suppression, and kill lines; reports the cited present-but-fragmented count among the 42 CV57 divisions; and performs no fitting.
- SA-PARKB-SWEEP evaluates every CPU-testable PARK-B node, explicitly skips weight/GPU-gated nodes, and cites path plus SHA-256 for every proposed B-to-A or B-to-C move.
- Optional SA-ASSOC-CYCLE uses only the existing Trackastra cache and frozen separation design, compares forward/reverse consensus with B0 on 44b6, and stops before graph spend.
- Root rejects uncited moves and quarantined C2 or Branch-B numbers, accepts no DEAD verdict, merges validated fragments into tier_census.json, and appends root-authored LIVE_STATE and LEDGER records.
- Kaggle budget and CELLECT artifacts remain byte-identical; no GPU, Kaggle, submission, RunPod, graph, 6bba, or division-fitting action occurs.

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/
- reports/g02/screening/tiered_surface/branches/sa_division_deep/
- reports/g02/screening/tiered_surface/branches/sa_parkb/
- reports/g02/screening/tiered_surface/branches/sa_assoc_cycle/
- reports/g02/screening/tiered_surface/wave2_cpu_barrier.json
- reports/g02/screening/tiered_surface/wave2_cpu_barrier_validation.json
- reports/g02/screening/tiered_surface/reconcile_wave2_cpu_barrier.py
- reports/g02/screening/tiered_surface/tier_census.json
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md

**Non-goals**

- CELLECT packaging or feature work
- Kaggle-budget mutation or any external compute launch
- detector modification, graph scoring, 6bba, CV57 fitting, or public submission
- division fitting before a banked both-family association-continuity pass
- rescreening a DEAD mechanism or creating a DEAD verdict
- consuming prior incomplete NODETOPO partial metrics or quarantined memo numbers

**Invariants**

- The human-locked plan, LIVE_STATE.md, and LEDGER.md retain precedence over lane prompts.
- Frozen shared inputs are read-only and hash-pinned; sibling live outputs are never consumed.
- Each lane runs in its own git worktree and changes only its assigned subtree.
- Root is the sole writer of LIVE_STATE.md, LEDGER.md, tier_census.json, and kaggle_budget.json.
- Every tier move requires a committed artifact cited by path and SHA-256; graph gates remain human-held.
- Existing dirty-worktree changes and unrelated artifacts are preserved.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The terminal barrier matches the exact CPU-only Wave-2 request and all scope and authority fences.
  - Evidence: Exact CPU-only Wave-2 intent and authority fences are represented in commit a8c65b4 and the root barrier.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: All requested lane deliverables validate from frozen inputs and the root barrier deterministically reconciles cited fragments.
  - Evidence: Deterministic root validation passed 48 of 48 checks with all lane validations and evidence hashes resolving.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Frozen inputs, CELLECT, Kaggle budget, graph gates, and unrelated dirty work remain unchanged.
  - Evidence: Kaggle budget and CELLECT tree hashes are unchanged and the root commit contains only authorized Wave-2 paths.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Swap growth, missing input hashes, scope violations, GPU needs, uncited moves, memo leakage, and dependency failures stop or park fail-closed.
  - Evidence: All lanes passed the 45-second zero-swap guard and parked fail-closed before every unavailable or human-held gate.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no missing deliverable, unsupported claim, or unauthorized state change.
  - Evidence: Final root review reproduced the census, barrier, validation hashes, 48 checks, clean lane worktrees, and held graph gate.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate reviewer attempts to falsify lane isolation, evidence identity, tier adjudication, and no-external-action claims.
  - Evidence: Independent reviewer found governance and chronology gaps, verified their fixes, and returned final PASS on all barrier claims.

### Goal detail

# Flight plan

Freeze exact root hashes and isolated worktrees, dispatch three mandatory lanes while the
coordinator evaluates the optional cache-only association-cycle lane, then validate all
artifacts and commit-scoped changes. At the barrier, accept only self-contained, cited results,
update the census and root records, hold graph gates for human review, and finish with an
independent skeptical review plus deterministic Cruise attestation.

## 2026-07-28T17:45:35.420Z — blocked: CONTEXT Three corrections to the CELLECT diagnosis, all provable from banked artifacts — no new API calls: 1. AUTH: The 403 on ListDatasetFiles is NOT rules/entitlement. onebitbrain/g02-cellect-r7- published-assets-v1 is the user's OWN private dataset; rules were accepted ~3 weeks ago and the CLI has worked all along (see Q-APNCC-PUBLIC submission in LEDGER). It is an agent-shell token/scope problem. STANDING RULE: all authenticated CLI work (Kaggle list/push/submit) goes through computer-use cmux where the real session lives — not the agent's own token. Re-run `kaggle datasets files ...` via cmux. 2. FILENAME BUG (real, fix now): locate_one() uses Path("/kaggle/input").rglob(pattern), which treats the argument as a GLOB, not a literal. Fix: resolve the weight by CONTENT — glob "*.pth" (and *.bin/*.tar members), select the file whose SHA-256 == f3e8e7303976dc5fb6e52ade12d1ff795 3d2a28e30c06274b9f309562cd89ce5, fail closed if none. Make it step 0 of the kernel (before dep install) so a miss costs ~1s, not the observed ~650s. 3. BYTE MISMATCH (the deeper defect, previously missed): manifest stages 7 files / 19,467,606 B with the .pth at 10,197,221 B. Remote observed total is 18,508,409 B; non-checkpoint files are 9,270,385 B, implying a remote checkpoint of 9,238,024 B — 959,197 B SMALLER than authenticated. Kaggle's dataset pipeline is transforming the file, so its SHA can never match. FIX: upload the checkpoint inside an UNCOMPRESSED .tar (or as .bin), extract in-kernel, THEN verify SHA-256 against the pin. Also rename to safe ASCII (cellect_uext_x3rd_149_0_4p6540.pth) inside the tar. Verify via cmux that the stored object round-trips to the exact SHA BEFORE any launch. TASK — do 1-3 above, no kernel launch until cmux confirms a byte-exact round-trip. IN PARALLEL (CPU, no Kaggle, no budget): run the human-held NODETOPO 44b6 graph gate on its 50 N1-valid restoration actions. Rationale: unlike delete/swap (393 positives in 4.02M — the sparse needle regime that has killed every selector), NODETOPO is a small dense action set that avoids that failure mode entirely. Preregistered gate unchanged: adjusted >= +0.005, raw/node >= 0, 44b6 first. Report at the barrier. HARD STOPS unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod.

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-28T17:45:35.420Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Three corrections to the CELLECT diagnosis, all provable from banked artifacts — no new API calls:

1. AUTH: The 403 on ListDatasetFiles is NOT rules/entitlement. onebitbrain/g02-cellect-r7-
   published-assets-v1 is the user's OWN private dataset; rules were accepted ~3 weeks ago and
   the CLI has worked all along (see Q-APNCC-PUBLIC submission in LEDGER). It is an agent-shell
   token/scope problem. STANDING RULE: all authenticated CLI work (Kaggle list/push/submit)
   goes through computer-use cmux where the real session lives — not the agent's own token.
   Re-run `kaggle datasets files ...` via cmux.

2. FILENAME BUG (real, fix now): locate_one() uses Path("/kaggle/input").rglob(pattern), which
   treats the argument as a GLOB, not a literal. Fix: resolve the weight by CONTENT — glob "*.pth"
   (and *.bin/*.tar members), select the file whose SHA-256 == f3e8e7303976dc5fb6e52ade12d1ff795
   3d2a28e30c06274b9f309562cd89ce5, fail closed if none. Make it step 0 of the kernel (before dep
   install) so a miss costs ~1s, not the observed ~650s.

3. BYTE MISMATCH (the deeper defect, previously missed): manifest stages 7 files / 19,467,606 B
   with the .pth at 10,197,221 B. Remote observed total is 18,508,409 B; non-checkpoint files are
   9,270,385 B, implying a remote checkpoint of 9,238,024 B — 959,197 B SMALLER than authenticated.
   Kaggle's dataset pipeline is transforming the file, so its SHA can never match. FIX: upload the
   checkpoint inside an UNCOMPRESSED .tar (or as .bin), extract in-kernel, THEN verify SHA-256
   against the pin. Also rename to safe ASCII (cellect_uext_x3rd_149_0_4p6540.pth) inside the tar.
   Verify via cmux that the stored object round-trips to the exact SHA BEFORE any launch.

TASK — do 1-3 above, no kernel launch until cmux confirms a byte-exact round-trip.

IN PARALLEL (CPU, no Kaggle, no budget): run the human-held NODETOPO 44b6 graph gate on its 50
N1-valid restoration actions. Rationale: unlike delete/swap (393 positives in 4.02M — the sparse
needle regime that has killed every selector), NODETOPO is a small dense action set that avoids
that failure mode entirely. Preregistered gate unchanged: adjusted >= +0.005, raw/node >= 0, 44b6
first. Report at the barrier.

HARD STOPS unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod.
```

### Goal

_No current Flight was available for this run._

## 2026-07-28T18:29:54.860Z — completed: CONTEXT Three corrections to the CELLECT diagnosis, all provable from banked artifacts — no new API calls: 1. AUTH: The 403 on ListDatasetFiles is NOT rules/entitlement. onebitbrain/g02-cellect-r7- published-assets-v1 is the user's OWN private dataset; rules were accepted ~3 weeks ago and the CLI has worked all along (see Q-APNCC-PUBLIC submission in LEDGER). It is an agent-shell token/scope problem. STANDING RULE: all authenticated CLI work (Kaggle list/push/submit) goes through computer-use cmux where the real session lives — not the agent's own token. Re-run `kaggle datasets files ...` via cmux. 2. FILENAME BUG (real, fix now): locate_one() uses Path("/kaggle/input").rglob(pattern), which treats the argument as a GLOB, not a literal. Fix: resolve the weight by CONTENT — glob "*.pth" (and *.bin/*.tar members), select the file whose SHA-256 == f3e8e7303976dc5fb6e52ade12d1ff795 3d2a28e30c06274b9f309562cd89ce5, fail closed if none. Make it step 0 of the kernel (before dep install) so a miss costs ~1s, not the observed ~650s. 3. BYTE MISMATCH (the deeper defect, previously missed): manifest stages 7 files / 19,467,606 B with the .pth at 10,197,221 B. Remote observed total is 18,508,409 B; non-checkpoint files are 9,270,385 B, implying a remote checkpoint of 9,238,024 B — 959,197 B SMALLER than authenticated. Kaggle's dataset pipeline is transforming the file, so its SHA can never match. FIX: upload the checkpoint inside an UNCOMPRESSED .tar (or as .bin), extract in-kernel, THEN verify SHA-256 against the pin. Also rename to safe ASCII (cellect_uext_x3rd_149_0_4p6540.pth) inside the tar. Verify via cmux that the stored object round-trips to the exact SHA BEFORE any launch. TASK — do 1-3 above, no kernel launch until cmux confirms a byte-exact round-trip. IN PARALLEL (CPU, no Kaggle, no budget): run the human-held NODETOPO 44b6 graph gate on its 50 N1-valid restoration actions. Rationale: unlike delete/swap (393 positives in 4.02M — the sparse needle regime that has killed every selector), NODETOPO is a small dense action set that avoids that failure mode entirely. Preregistered gate unchanged: adjusted >= +0.005, raw/node >= 0, 44b6 first. Report at the barrier. HARD STOPS unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-28T18:29:54.860Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Three corrections to the CELLECT diagnosis, all provable from banked artifacts — no new API calls:

1. AUTH: The 403 on ListDatasetFiles is NOT rules/entitlement. onebitbrain/g02-cellect-r7-
   published-assets-v1 is the user's OWN private dataset; rules were accepted ~3 weeks ago and
   the CLI has worked all along (see Q-APNCC-PUBLIC submission in LEDGER). It is an agent-shell
   token/scope problem. STANDING RULE: all authenticated CLI work (Kaggle list/push/submit)
   goes through computer-use cmux where the real session lives — not the agent's own token.
   Re-run `kaggle datasets files ...` via cmux.

2. FILENAME BUG (real, fix now): locate_one() uses Path("/kaggle/input").rglob(pattern), which
   treats the argument as a GLOB, not a literal. Fix: resolve the weight by CONTENT — glob "*.pth"
   (and *.bin/*.tar members), select the file whose SHA-256 == f3e8e7303976dc5fb6e52ade12d1ff795
   3d2a28e30c06274b9f309562cd89ce5, fail closed if none. Make it step 0 of the kernel (before dep
   install) so a miss costs ~1s, not the observed ~650s.

3. BYTE MISMATCH (the deeper defect, previously missed): manifest stages 7 files / 19,467,606 B
   with the .pth at 10,197,221 B. Remote observed total is 18,508,409 B; non-checkpoint files are
   9,270,385 B, implying a remote checkpoint of 9,238,024 B — 959,197 B SMALLER than authenticated.
   Kaggle's dataset pipeline is transforming the file, so its SHA can never match. FIX: upload the
   checkpoint inside an UNCOMPRESSED .tar (or as .bin), extract in-kernel, THEN verify SHA-256
   against the pin. Also rename to safe ASCII (cellect_uext_x3rd_149_0_4p6540.pth) inside the tar.
   Verify via cmux that the stored object round-trips to the exact SHA BEFORE any launch.

TASK — do 1-3 above, no kernel launch until cmux confirms a byte-exact round-trip.

IN PARALLEL (CPU, no Kaggle, no budget): run the human-held NODETOPO 44b6 graph gate on its 50
N1-valid restoration actions. Rationale: unlike delete/swap (393 positives in 4.02M — the sparse
needle regime that has killed every selector), NODETOPO is a small dense action set that avoids
that failure mode entirely. Preregistered gate unchanged: adjusted >= +0.005, raw/node >= 0, 44b6
first. Report at the barrier.

HARD STOPS unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod.
```

### Goal

_No current Flight was available for this run._

## 2026-07-28T19:29:38.633Z — blocked: Measure the selector quality required for frozen Phase-1 delete, occupied-swap, and restore-topology actions to reach exact 44b6 graph delta +0.005 and breakeven, without changing tiers or thresholds

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-28T19:29:38.633Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Same governance as DAG 1. CPU-only, no Kaggle. This is a DIAGNOSTIC: it sets no tier and promotes
nothing.

GOAL
Answer the question that explains three near-zero results: HOW GOOD must a selector be, per action
class, before the exact 44b6 graph delta reaches +0.005? Delete/swap/restore all had large oracle
headroom and realized ~0. Trackastra had 0.918 AUC and realized -0.011. The required precision has
never been measured.

NODES
N0 DOSE-RESPONSE. For each action class (delete, occupied_swap, restore_topology) on the frozen
   Phase-1 44b6 population, apply the top-N ORACLE-ranked actions through the frozen N1 compiler for
   N across a fixed log grid, and record the exact graph delta. This gives the ceiling curve:
   realized delta vs support at perfect selection.
N1 NOISE INJECTION. Degrade the oracle ranking to a controlled target AUC in {0.55,0.65,0.75,0.85,
   0.90,0.95,0.99} (fixed seed, deterministic corruption preserving marginals) and recompute exact
   graph delta at each support level. Repeat >=20 seeds; report mean and spread.
N2 REQUIREMENT SURFACE. For each class emit the minimum (AUC, retention-at-CG0-budget) needed to
   reach +0.005 and to reach breakeven 0. State explicitly whether the CURRENT separation-gate
   thresholds (AUC>=0.65, retention>=0.15) are SUFFICIENT or TOO LENIENT for each class.
N3 CONSEQUENCE. If the measured requirement exceeds the frozen gate thresholds, do NOT silently
   change them -- emit a written amendment proposal with the evidence and HALT for human review
   (thresholds are frozen pre-outcome; changing them is a human decision).

OUTPUT: dose-response curves, requirement surface, explicit sufficiency verdict per class,
state_fragment, proposed ledger row. No tier moves. Barrier and report.
```

### Goal

**Outcome:** Measure the selector quality required for frozen Phase-1 delete, occupied-swap, and restore-topology actions to reach exact 44b6 graph delta +0.005 and breakeven, without changing tiers or thresholds

**Verification**

- run focused deterministic unit tests for target-AUC corruption, ranking, budget retention, N1 batching, and requirement extraction
- authenticate every frozen input by hash and reproduce banked baselines
- run the CPU-only guarded diagnostic with at least 20 seeds per AUC and exact compiled graph scoring
- independently recompute selected curve points and review all artifacts, hard stops, and campaign state

**Touchpoints**

- reports/g02/screening/selector_requirement_44b6/**
- reports/g02/screening/LIVE_STATE.md
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- No tier move, promotion, or DEAD action
- No modification of frozen separation-gate thresholds
- No write to reports/g02/screening/LEDGER.md; emit a proposed row only
- No 6bba, CV57, test, audit, public submission, Kaggle, GPU, or RunPod action
- No re-screening of a DEAD mechanism

**Invariants**

- Use only the frozen fit-only Phase-1 44b6 action populations and the frozen public N1 compiler
- Oracle labels remain exact single-action delta S greater than zero; exact zero is harmful
- Noise corruption preserves action membership, class prevalence, and the score-rank marginal and attains each requested empirical AUC deterministically
- Every emitted action batch is accepted by the frozen N1 compiler and scored against the exact frozen 44b6 control
- Current AUC 0.65 and retention 0.15 thresholds remain unchanged; any stricter requirement becomes proposal-only evidence at a human-review barrier
- Only LIVE_STATE and the new diagnostic directory change outside Cruise bookkeeping

**Quality gates**

- **core.intent** — failed
  - Dimension: intent
  - Criterion: Outputs answer N0-N3 for all three requested action classes with no tier move or frozen-threshold change
  - Evidence: Requested N0-N3 measurement is unavailable because both guarded runs failed the binding zero-swap rule before output
- **core.functional** — unavailable
  - Dimension: functional-correctness
  - Criterion: Oracle and target-AUC rankings compile through frozen N1 and produce exact 44b6 dose-response and requirement surfaces over at least 20 seeds
  - Evidence: Seventeen tests passed but no complete valid curve or requirement surface survived guarded execution
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Frozen populations, compiler, controls, gates, tiers, and unrelated dirty-worktree content remain unchanged
  - Evidence: Frozen sources were hash-pinned, compiler tests passed 17 of 17, and scoped review found no tier threshold ledger or external mutation
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Hash drift, malformed populations, unattainable AUC, class-degenerate populations, conflicting batches, overwrite attempts, and nonfinite metrics fail closed or receive an explicit infeasible diagnostic status
  - Evidence: Wrapper failure and both resource violations failed closed with empty quarantines and explicit no-verdict artifacts
- **core.completion-review** — failed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no unsupported exactness claim, missing requested output, or unauthorized campaign mutation
  - Evidence: Skeptical review confirms the barrier is supported but the requested exact selector requirements are missing
- **core.independent-review** — unavailable
  - Dimension: independent-review
  - Criterion: A genuinely separate review pass reproduces critical points and finds no unresolved flaw in AUC control, N1 selection, scoring, requirement extraction, or governance
  - Evidence: Separate barrier validation passed but critical scientific curve points cannot be reproduced because no valid result was emitted
- **backend.data-integrity** — passed
  - Dimension: data-integrity
  - Criterion: Frozen population identities, counts, labels, deltas, budgets, action payloads, compiler pins, and scorer pins are hash-authenticated and fully covered
  - Evidence: Frozen manifests population files compiler scorer archive and preregistration were hash-checked before guarded processing
- **backend.security** — passed
  - Dimension: security
  - Criterion: The diagnostic is local CPU-only and does not access credentials, external services, forbidden splits, or root underscore-reference documents
  - Evidence: Commands remained local CPU-only with all GPU Kaggle RunPod submission credential and external-action flags false
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: Deterministic seeds and atomic artifact writes make retries reproducible and prevent partial or concurrent runs from masquerading as complete
  - Evidence: Deterministic seeds and overwrite refusal prevented either terminated process from publishing a partial scientific result
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Runtime receipts expose progress, class/seed/AUC/support coverage, resources, empirical-AUC error, compiler failures, and exact barrier status
  - Evidence: Preflights guards repair receipts quarantine checks resource barrier state fragment report and live-state entry expose the terminal path

### Goal detail

# Flight plan

Authenticate and pin the existing Phase-1 population banks, N1 compiler, 44b6 controls,
and scorer. Freeze a deterministic ranking/noise/scoring contract and focused tests.
Run the oracle curve first, then the requested seven target AUCs over at least 20
seeds, deriving the exact graph response at a fixed support grid and the CG0-budget
retention for every run. Extract the minimum observed requirements, issue explicit
threshold verdicts, and if necessary stop at a written amendment proposal for human
review. Finish with a separate verifier, resource and hard-stop barrier, short
LIVE_STATE update, and no LEDGER mutation.

## Blocker

The binding local zero-swap rule stopped both the original run and the single
streamed-lifetime repair before any scientific result file was emitted. The
second attempt grew swap by 1,567.69 MiB despite a lower 1.453 GiB process-tree
RSS. A complete dose/noise/requirement surface therefore does not exist, and a
new execution design or retry requires human review.

## 2026-07-29T14:58:36.927Z — blocked: Execute the window-inference-repair DAG cascade from N0_offset_and_auc_gate.md through its truthful terminal state

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-29T14:58:36.927Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
Read .agents/dag/window-inference-repair/DAG.md and execute the cascade starting at N0_offset_and_auc_gate.md
```

### Goal

**Outcome:** Execute the window-inference-repair DAG cascade from N0_offset_and_auc_gate.md through its truthful terminal state

**Verification**

- read the exact requested DAG and N0 node before executing any node action
- obey every node dependency, gate, permission stop, and terminal condition
- update reports/g02/screening/LIVE_STATE.md after every campaign state change
- preserve unrelated dirty-worktree changes and the human-locked campaign plan

**Touchpoints**

- .agents/dag/window-inference-repair/**
- reports/g02/screening/LIVE_STATE.md
- node-authorized paths declared by the requested DAG
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- No substitution of a different DAG or inferred node contract
- No rescreening of a DEAD mechanism without a new named mechanism
- No Kaggle GPU or submission and no RunPod launch without explicit human approval
- No modification of unrelated pre-existing workspace changes

**Invariants**

- The human-locked campaign plan remains unchanged unless a requested node explicitly reaches its human decision gate
- Only the root coordinator may write reports/g02/screening/LEDGER.md
- LIVE_STATE remains the shared source of truth and is updated only when campaign state changes

**Quality gates**

- **core.intent** — pending
  - Dimension: intent
  - Criterion: The exact requested DAG is executed from N0 through its declared terminal condition without substituting another cascade
  - Evidence: None
- **core.functional** — pending
  - Dimension: functional-correctness
  - Criterion: Every executed node produces and verifies the artifacts and verdicts required by its contract
  - Evidence: None
- **core.regression** — pending
  - Dimension: regression
  - Criterion: Frozen campaign inputs, unrelated dirty-worktree changes, and protected governance artifacts remain intact
  - Evidence: None
- **core.failure-paths** — pending
  - Dimension: failure-paths
  - Criterion: Missing inputs, failed gates, resource violations, and permission stops fail closed at the exact declared boundary
  - Evidence: None
- **core.completion-review** — pending
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no skipped dependency, stale state, unsupported verdict, or unauthorized action
  - Evidence: None
- **core.independent-review** — pending
  - Dimension: independent-review
  - Criterion: A separate review pass independently verifies routing, critical results, and hard-stop compliance
  - Evidence: None

### Goal detail

# Flight plan

The requested cascade cannot begin until
`.agents/dag/window-inference-repair/DAG.md` and
`.agents/dag/window-inference-repair/N0_offset_and_auc_gate.md` are present.
Once available, read both completely, execute N0, follow only the DAG-declared
edges whose gates pass, verify every node, and synchronize `LIVE_STATE.md` after
each campaign state change.

## 2026-07-29T18:41:02.860Z — completed: Repair and locally verify the withdrawn selector CPU packaging lane, then complete exactly one eligible bounded GPU CELLECT/DINOv2 44b6 separation-population bake-off and halt at the supplier-specific human graph-preregistration barrier.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-29T18:41:02.860Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Two Kaggle kernel failures diagnosed from logs, both packaging (not scientific):
  v1: ModuleNotFoundError 'kaggle_selector_step0' — `kernels push` ships ONLY the code_file from
      kernel-metadata.json; sibling .py files are not uploaded.
  v2: Step0Error "expanded gzip transport is not byte-reversible" — Kaggle expands archives on
      materialization and gzip recompression is nondeterministic, so container-SHA round-trip is
      structurally unsatisfiable.

TASKS
1. Report the status/log of the already-pushed v3 before changing anything.
2. Fix v1: inline the step-0 helper into the single code_file (or ship helpers as a dataset and
   sys.path them). Verify by asserting the module resolves at t<5s in the next run.
3. Fix v2 by PORTING the proven CELLECT v4 step-0 resolver: use an UNCOMPRESSED tar, accept the
   Kaggle-expanded directory representation, and verify by MEMBER SHA-256 — never by container
   round-trip. Keep fail-closed on a genuine member-hash mismatch.
4. RESOURCE POLICY: do not run CPU-only jobs on Kaggle while the local selector replay is healthy.
   If the local DAG-2 run is still progressing under its zero-swap guard, let it finish there and
   withdraw the Kaggle port. Reserve Kaggle GPU exclusively for the CELLECT/DINOv2 bake-off.
5. THEN launch the bake-off: ONE bounded run, authenticated projection <5h, hard 5h kill, serial
   authorization, prelaunch budget check, postrun debit (27.4239 h remain). Compute ONLY the 44b6
   separation-population endpoint features — no full-pool cache. Sealed B0 first, then A1/A0.

Governance unchanged. HALT at the barrier: PASS unlocks only a supplier-specific 44b6 graph-gate
preregistration for human review; FAIL parks that supplier PARK-C. No training, graph, 6bba, CV57,
submission, or RunPod.
```

### Goal

**Outcome:** Repair and locally verify the withdrawn selector CPU packaging lane, then complete exactly one eligible bounded GPU CELLECT/DINOv2 44b6 separation-population bake-off and halt at the supplier-specific human graph-preregistration barrier.

**Verification**

- python -m pytest reports/g02/screening/selector_requirement_44b6
- python -m pytest reports/g02/screening/tiered_surface/branches/sa_dinov2
- python -m pytest reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate

**Touchpoints**

- reports/g02/screening/selector_requirement_44b6/**
- reports/g02/screening/tiered_surface/branches/sa_dinov2/**
- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/**
- reports/g02/screening/tiered_surface/kaggle_budget.json
- reports/g02/screening/tiered_surface/tier_census.json
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- No scientific threshold, population, fold, probe, B0, or scorer changes
- No CPU-only Kaggle selector rerun while the completed local zero-swap replay is healthy
- No more than one newly authorized GPU bake-off run
- No training, graph execution, full-pool cache, 6bba, CV57, submission, RunPod, or DEAD move

**Invariants**

- Frozen separation gate SHA 9fb030bd…3310e and sealed B0 remain unchanged
- Selector kernel is a single-file code package whose step-0 symbol resolves before five seconds
- Uncompressed-tar transport accepts Kaggle-expanded members and authenticates member SHA-256, never reconstructed container SHA
- Genuine missing, duplicate, or mismatched members fail closed before imports or scientific work
- Exactly one newly authorized bounded GPU supplier run is allowed after an authenticated projection under five hours
- The run computes only frozen 44b6 separation-population endpoint features, seals B0 first, then evaluates A1/A0
- Root remains sole writer for LIVE_STATE, LEDGER, budget, and barrier state
- PASS or FAIL affects only the exact supplier instantiation and halts before graph-contract opening

**Quality gates**

- **core.intent** — pending
  - Dimension: intent
  - Criterion: Delivered artifacts and actions match every numbered task, resource rule, and governance fence in the exact request.
  - Evidence: None
- **core.functional** — pending
  - Dimension: functional-correctness
  - Criterion: The single-file selector step 0 authenticates expanded uncompressed-tar members, and the one GPU run produces complete frozen-population endpoint features plus unchanged B0-first A1/A0 evaluation.
  - Evidence: None
- **core.regression** — pending
  - Dimension: regression
  - Criterion: Existing frozen inputs, evaluator semantics, thresholds, local selector result, and unrelated campaign state remain intact.
  - Evidence: None
- **core.failure-paths** — pending
  - Dimension: failure-paths
  - Criterion: Missing, duplicate, or mismatched member hashes, incomplete features, projection overflow, timeout, partial outputs, and budget failures stop closed without scientific verdict leakage.
  - Evidence: None
- **core.completion-review** — pending
  - Dimension: completion-review
  - Criterion: A skeptical final evidence review finds no unresolved in-scope gap or unauthorized state transition.
  - Evidence: None
- **backend.data-integrity** — pending
  - Dimension: data-integrity
  - Criterion: Versioned assets, mounted files, feature shards, manifests, and ledger debits are content-addressed, complete, and internally consistent.
  - Evidence: None
- **backend.security** — pending
  - Dimension: security
  - Criterion: Authenticated Kaggle access uses the owner session without persisting credentials or enabling internet.
  - Evidence: None
- **backend.concurrency** — pending
  - Dimension: concurrency
  - Criterion: Serial authorization and root-only barrier writes prevent overlapping runs or racing state changes.
  - Evidence: None
- **backend.observability** — pending
  - Dimension: observability
  - Criterion: Remote status, step-0 timing, progress, projection, kill, output, and terminal receipts make any failure self-diagnosing.
  - Evidence: None
- **hpc.complexity** — pending
  - Dimension: complexity
  - Criterion: Per-video inference and aggregation cost models fit the bounded run and avoid full-pool work.
  - Evidence: None
- **hpc.memory** — pending
  - Dimension: memory
  - Criterion: Separation-population-only endpoint materialization, per-video load-and-release, and bounded aggregation avoid full-pool work and GPU or host exhaustion.
  - Evidence: None
- **hpc.numerical** — pending
  - Dimension: numerical-correctness
  - Criterion: Endpoint embeddings, XY-beta features, A0 permutation, folds, probe, and metrics preserve the frozen numerical contract.
  - Evidence: None
- **hpc.performance** — pending
  - Dimension: performance-measurement
  - Criterion: Representative measured throughput yields a conservative projection below five hours and the process stops before the hard limit.
  - Evidence: None
- **hpc.portability** — pending
  - Dimension: toolchain-portability
  - Criterion: The single-file selector and self-contained bake-off kernel work on Kaggle Python and accelerator tooling without local-path, sibling-module, or archive-container assumptions.
  - Evidence: None
- **domain.accelerator** — pending
  - Dimension: accelerator
  - Criterion: CUDA execution uses the pinned supplier model and deterministic frozen semantics while partial data never becomes a scientific result.
  - Evidence: None
- **core.independent-review** — pending
  - Dimension: independent-review
  - Criterion: A fresh skeptical review independently attempts to falsify launch safety, data completeness, evaluator immutability, and final verdict scope.
  - Evidence: None

### Goal detail

# Flight plan

Confirm and freeze the v3 terminal evidence before product mutation. Withdraw the
selector CPU lane because the local zero-swap DAG-2 result is already healthy, while
repairing its code-file and uncompressed-tar/member-hash contracts locally. Reconcile
the current supplier, budget, serial authorization, and authenticated projection;
launch no more than one eligible GPU run with a five-hour hard kill. Consume only a
complete frozen 44b6 separation-population result, evaluate sealed B0 before A1/A0,
debit actual usage, update root state, and halt at the supplier-specific human review
barrier without executing a graph or any downstream work.

