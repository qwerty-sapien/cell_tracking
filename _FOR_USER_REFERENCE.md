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

## 2026-07-30T04:57:21.141Z — completed: Reconcile selector transport, withdraw CPU work when local DAG-2 is healthy, and terminally account the bounded 44b6 supplier bake-off without crossing the scientific barrier.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-30T04:57:21.141Z
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

**Outcome:** Reconcile selector transport, withdraw CPU work when local DAG-2 is healthy, and terminally account the bounded 44b6 supplier bake-off without crossing the scientific barrier.

**Verification**

- selector transport and existing selector tests pass 18/18
- DINOv2 Kaggle GPU package tests pass 5/5
- terminal, comparison, budget, and validation JSON invariants pass
- owner-authenticated latest status and postrun quota are banked
- git diff check passes

**Touchpoints**

- reports/g02/screening/selector_requirement_44b6/**
- reports/g02/screening/tiered_surface/branches/sa_dinov2/kaggle_gpu_bakeoff_v1/**
- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/two_supplier_comparison_v3_gpu_terminal/**
- reports/g02/screening/tiered_surface/kaggle_budget.json
- reports/g02/screening/LEDGER.md
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- no further Kaggle CPU or GPU launch
- no CELLECT launch
- no training or full-pool cache
- no graph contract or graph run
- no 6bba or CV57
- no submission or RunPod
- no threshold tier DEAD or human-locked plan change

**Invariants**

- frozen 44b6 population folds probe thresholds and sealed B0 remain unchanged
- genuine member hash mismatch fails closed before heavy imports
- no partial MPS rows or incomplete class manifest can be consumed
- infrastructure INVALID is never converted into a scientific PASS or FAIL
- only root writes LEDGER and LIVE_STATE remains the shared current source
- all external compute usage is honestly debited and no authorization remains active

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: Every requested task and governance barrier is represented truthfully, including the accidental two-version launch race.
  - Evidence: Exact-request review confirms v3-first reporting, selector repair and withdrawal, bounded bake-off terminalization, honest two-run race reconciliation, and barrier halt.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Single-file step 0, expanded uncompressed-tar member verification, and endpoint-only GPU package behavior are evidenced.
  - Evidence: Selector transport plus existing suites pass 18/18, GPU package passes 5/5, and latest remote v2 step 0 resolves all 77 requests before the pip-only failure.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Frozen science identities and forbidden downstream actions remain unchanged.
  - Evidence: Frozen population, preregistration, sealed B0, folds, probe, thresholds, and PARK-B states remain unchanged with zero forbidden downstream actions.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing duplicate or mismatched members fail closed and the remote wheel error halts before model or features.
  - Evidence: Transport negatives fail closed on missing, duplicate, and mismatched members, while exact remote chronology shows zero worker, model, feature, class-manifest, or B0/A1/A0 work.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff and evidence review finds no unresolved in-scope gap or unsupported scientific claim.
  - Evidence: Fresh final diff and evidence review found no unresolved in-scope gap, unsupported scientific verdict, or open compute authorization.
- **backend.data-integrity** — passed
  - Dimension: data-integrity
  - Criterion: Quota, debit, ledger, terminal, comparison, and evidence hashes are internally consistent.
  - Evidence: Budget debit sum equals 3.6435 h, remaining equals 26.3565 h, owner capacity is 25.47 h, and all pinned evidence hashes and comparison counts validate.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Owner authentication stays in cmux, no credential is exposed, and no submission or RunPod action occurs.
  - Evidence: All authenticated Kaggle reads used the owner cmux session, banked evidence contains no credential, and submission plus RunPod remained untouched.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: The concurrent v1/v2 push race is detected, counted, conservatively debited, closed, and cannot trigger a retry.
  - Evidence: The v1/v2 race is explicitly recorded as a serial-authorization violation with two runs, a full 0.74 h quota debit, closed authorization, and no retry.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Terminal failure, launch race, quota, and barrier state are diagnosable from hash-pinned artifacts without secrets.
  - Evidence: Exact status, quota, run-list time, version-probe failures, pushed code, inventory, step-0 receipt, terminal payload, and logs are banked and hash-pinned.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate skeptical review pass reproduces the terminal classification, accounting, and closed barrier.
  - Evidence: A separate skeptical pass reproduced two triggered versions, latest-v2 infrastructure failure before science, the conservative 0.74 h accounting, and zero PASS, FAIL, PARK-C, or graph unlock.

### Goal detail

# Flight plan

The selector v3 observation precedes all repairs. The locally repaired v4
transport remains withdrawn because the local DAG-2 result is valid and
zero-swap. The DINOv2 bake-off is terminal infrastructure-INVALID; a concurrent
push race created two byte-identical versions under the one-run authorization.
The reconciliation must preserve that violation, debit the full authenticated
quota delta, and stop without a scientific verdict or downstream unlock.

## 2026-07-30T14:18:44.205Z — completed: Produce separate governed DINOv2-local and one-attempt CELLECT verdict outcomes after fixing transport, serial launch, and budget controls.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-30T14:18:44.205Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Attempt #6 root cause: hash-prefixed wheel filenames violate PEP 427, so pip rejected them after all
77 requests resolved correctly. Governance unchanged. TWO material issues also surfaced: (a) local
kaggle_budget.json drifted from Kaggle's authoritative quota (tracked 2.88h vs actual 4.53h used /
25.47h remaining); (b) a concurrency race created two kernel versions under a one-run authorization.

TASK 1 — Fix the wheel packaging (do NOT rename wheels)
- Restore CANONICAL PEP 427 wheel filenames. Verify each wheel by SHA-256 against the bundle
  manifest by READING BYTES before install; never encode a hash in a wheel filename.
- Install with --no-index --find-links against the canonical-named directory.
- Keep SHA-addressed resolution for the checkpoint only. Add a regression test asserting every wheel
  in the bundle has a PEP 427-valid filename and a manifest-matching SHA.

TASK 2 — Make step 0 fast AND targeted (it took 225s, blowing the 5s envelope)
- Step 0 must stat/hash ONLY the declared input paths (checkpoint, request manifest, population
  manifest, wheels). It must NEVER walk or inventory the competition dataset (24,968 files / 87.7 GB
  is why it took 225s). Assert request_count > 0 and manifest SHA 4ab7bb23…76f0f. Target <5s.

TASK 3 — Enforce serial authorization for real, and reconcile the budget
- Before any push: query Kaggle for in-flight/existing kernel versions and acquire an exclusive lock;
  abort if a version is already running. After: assert the version count increased by exactly 1.
- Reconcile kaggle_budget.json against Kaggle's AUTHORITATIVE quota before and after every launch.
  Record the authoritative balance (now 25.47 h) as the source of truth; the local file is derived,
  never canonical. Log the 1.65h drift as a correction entry.

TASK 4 — Get the first verdict WITHOUT Kaggle: run DINOv2 locally
- DINOv2 ViT-S/14 is ~21M params (~84MB) -- far lighter than the Trackastra+V45 co-residence that
  caused historical swap failures. Extract R5 XY features for the 77 frozen request shards LOCALLY
  (CPU/MPS) using the existing two-stage isolation + 45s idle preflight + zero-swap guard, one shard
  at a time with atomic checkpointing so a guard trip resumes instead of restarting.
- Then run the frozen gate for DINOv2: sealed B0 (a9ce8c0b…b4ff2), then A1 and A0, per class.
  Frozen prereg 9fb030bd…3310e unchanged. This consumes NO Kaggle quota.

TASK 5 — CELLECT: ONE more attempt, with a pre-committed stop rule
- After Tasks 1-3 verify, ONE bounded GPU run (<5h projection, hard 5h kill, serial lock held,
  authoritative budget check). STOP RULE (human pre-committed): if this run fails again for TRANSPORT
  reasons rather than producing a scientific verdict, the GPU bake-off track HALTS -- mark CELLECT
  PARK-B blocked-on-infrastructure and do NOT request a seventh run. Escalate for a human decision on
  whether the representation lever is worth further spend.

HOLD AT BARRIER. Report DINOv2's verdict (if obtained) and CELLECT's outcome separately. Hard stops
unchanged: submission, DEAD, plan amendment, >=5h GPU, RunPod, full-pool cache.
```

### Goal

**Outcome:** Produce separate governed DINOv2-local and one-attempt CELLECT verdict outcomes after fixing transport, serial launch, and budget controls.

**Verification**

- .venv313/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_attempt_v7 -p 'test_*.py'
- .venv313/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_dinov2/kaggle_gpu_bakeoff_v1 -p 'test_*.py'
- .venv313/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_dinov2/local_bakeoff_v3 -p 'test_*.py'
- .venv313/bin/python reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/verify_cellect_step0_v7.py
- .venv313/bin/python reports/g02/screening/tiered_surface/branches/sa_dinov2/local_bakeoff_v3/verify_run.py

**Touchpoints**

- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/**
- reports/g02/screening/tiered_surface/branches/sa_dinov2/**
- reports/g02/screening/tiered_surface/kaggle_budget.json
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md
- .cruise/**

**Non-goals**

- No submission
- No DEAD promotion
- No plan amendment
- No GPU run projected at or above five hours
- No RunPod
- No full-pool feature cache
- No graph, training, 6bba, or CV57 work

**Invariants**

- The frozen request, population, B0, and preregistration artifacts remain byte-identical.
- Wheel filenames remain canonical PEP 427 names and wheel bytes must match the bundle manifest before installation.
- Step 0 touches only declared inputs and fails closed on a true member SHA mismatch.
- Kaggle authoritative quota is canonical and every launch is serialized with exactly one new version.
- DINOv2 uses local CPU or MPS only and resumes per shard after a zero-swap guard stop.
- CELLECT receives at most one additional bounded run under this authorization.
- A transport failure parks CELLECT PARK-B and ends the GPU bake-off track without a seventh request.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The delivered outcomes match all five tasks and preserve every stated hard stop.
  - Evidence: All five requested tasks completed with separate DINOv2 and CELLECT outcomes and every hard stop preserved.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Canonical wheel verification, targeted step 0, resumable local extraction, frozen gates, and bounded CELLECT supervision behave as specified.
  - Evidence: Twenty-five focused tests and both terminal verifiers passed; targeted step 0 remained below 0.06 seconds.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing frozen scientific inputs, evaluator contracts, and unaffected branch tests remain intact.
  - Evidence: Preregistration, sealed B0, and population hashes match frozen pins and all affected regression suites pass.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Invalid wheel names or hashes, broad input discovery, concurrent launches, running kernels, quota mismatch, swap growth, timeout, and transport failure all fail closed.
  - Evidence: Negative wheel, member-hash, lock, running-kernel, version-delta, swap, and transport probes fail closed.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff and evidence review finds no unsupported verdict or remaining in-scope gap.
  - Evidence: Final diff-and-evidence review found no unsupported verdict or unresolved authorized execution.
- **backend.data-integrity** — passed
  - Dimension: data-integrity
  - Criterion: Atomic shard checkpoints, manifest hashes, version counts, and derived budget history cannot silently corrupt or double count.
  - Evidence: Seventy-seven atomic checkpoints verify; version delta is exactly one and authoritative budget debit is uniquely recorded.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Kaggle authentication remains in the owner environment and no credential is copied into repository artifacts or logs.
  - Evidence: Credential-pattern scan passed across 311 scoped artifacts and no authentication material entered repository receipts.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: Exclusive launch authorization prevents duplicate pushes and post-push version arithmetic detects any race.
  - Evidence: Lock contention and version mismatch tests pass; recovered supervision proved zero-to-one with no second push.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Preflight, per-shard progress, guard stops, quota reconciliations, push state, terminal category, and stop-rule decision are diagnosable.
  - Evidence: Structured preflight, shard, guard, push, quota, terminal, stop-rule, LIVE_STATE, and ledger records agree.
- **independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate skeptical review finds no credible packaging, resource, concurrency, quota, or governance defect.
  - Evidence: Fresh adversarial review reproduced the post-push parser interruption and transport failure and verified their fail-closed dispositions.

### Goal detail

# Flight plan

First repair and test the transport, targeted step-0, launch serialization, and authoritative budget model. Then run the DINOv2 shard-isolated local path to its frozen gate. Only after the controls pass, consume the single CELLECT authorization, monitor to a terminal outcome, apply the pre-committed stop rule, and halt at the barrier.

## 2026-07-30T16:10:54.609Z — blocked: Produce a zero-Kaggle local CELLECT frozen-gate verdict, bank the operating-point methodology, and emit an oracle-backed division amendment dossier while fitting and amendment remain locked. The methodology and dossier are complete; the CELLECT verdict is blocked before shard 0 by the strict zero-swap invariant.

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-30T16:10:54.609Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
DINOv2 returned the campaign's first real separation verdict: AUC 0.670/0.682 (vs incumbent ~0.502)
but retention 0.0013/0.000 against a 0.15 gate -- real rank signal, zero performance at the operating
point. With Trackastra (AUC 0.918 -> -0.0111 graph) and the banked CG1/PU/CRC retention results
(0.4-1.8%), four independent representations now fail identically at the FP-budget operating point.
CELLECT's GPU track is halted by the pre-committed stop rule. Governance unchanged.

TASK 1 — Run CELLECT LOCALLY (no Kaggle, honors the GPU stop rule)
- CELLECT's checkpoint is 10,197,221 bytes -- smaller than DINOv2, which just completed all 77 shards
  locally with zero swap growth and 0 Kaggle cost. Reuse the exact DINOv2 local harness (two-stage
  isolation, 45s preflight, per-shard atomic checkpointing, zero-swap guard).
- Run the unchanged frozen gate: sealed B0 (a9ce8c0b…b4ff2), then A1/A0, per class, prereg
  9fb030bd…3310e. This completes the two-supplier comparison with zero further GPU spend.

TASK 2 — Bank the operating-point finding as reusable methodology
- Record explicitly that ROC-AUC is not diagnostic at ~1e-4 prevalence and that retention at the CG0
  FP budget is the binding criterion. Propose (for human approval) that future supplier gates lead
  with precision/recall-at-budget rather than an AUC floor. Do not alter any frozen gate.

TASK 3 — Prepare the DIVISION PLAN-AMENDMENT DOSSIER (decision-grade, no fitting)
Assemble, for human decision, the case to unlock division fitting from the association-continuity
prerequisite. Include:
- The arithmetic: local 0.881522, target ~0.9255-0.9422, gap +0.044-0.061; division 12/20/25 TP =
  +0.0286/+0.0476/+0.0595; edge-only pool caps ~0.896.
- The 38/42 exact repair bundles, per-event: required edges, motif class, late-fork equivalence,
  duplicate-suppression risk.
- Why the prerequisite is stale: B1's 0/51 measured postprocessing on a fragmented graph, before
  repair bundles existed and before 6 association mechanisms failed at ~0.
- An ORACLE-ONLY measurement (allowed, diagnostic): apply the 38 repair bundles through frozen N1
  and report the exact combined-score delta (edge + 0.1*division). This quantifies the true ceiling.
- Explicit FP risk: division FP cost (25 TP with 5 FP drops +0.0595 to +0.0532; 10 FP -> +0.0481).
- The preregistered arm that WOULD run if unlocked, with its 44b6-first kill line.
- FITTING REMAINS LOCKED until the human decides. Emit the dossier and HALT.

HOLD AT BARRIER. Hard stops unchanged: submission, DEAD, >=5h GPU, RunPod, full-pool cache, and any
plan amendment (Task 3 prepares the decision; it does not make it).
```

### Goal

**Outcome:** Produce a zero-Kaggle local CELLECT frozen-gate verdict, bank the operating-point methodology, and emit an oracle-backed division amendment dossier while fitting and amendment remain locked. The methodology and dossier are complete; the CELLECT verdict is blocked before shard 0 by the strict zero-swap invariant.

**Verification**

- python3 -m unittest discover reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1 -p 'test_*.py'
- python3 reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1/verify_blocked.py
- python3 reports/g02/screening/tiered_surface/branches/sa_division_value/verify_dossier.py

**Touchpoints**

- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1/**
- reports/g02/screening/tiered_surface/branches/sa_division_value/**
- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/OPERATING_POINT_METHODOLOGY.md
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- No Kaggle, RunPod, submission, full-pool cache, training, graph execution, 6bba, CV57, or >=5h GPU work.
- No frozen gate modification, plan amendment, fitting unlock, DEAD promotion, or supplier graph-gate execution.
- No mutation of the 38 repair bundles or frozen N1 scorer.

**Invariants**

- CELLECT uses exactly the authenticated 10,197,221-byte checkpoint and pinned safe loader.
- All 77 frozen request shards run one child at a time after a fresh 45-second idle preflight, with atomic checkpoints and a strict global zero-swap baseline.
- Evaluation order is sealed B0, then A1 and A0 per class under preregistration 9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e.
- The division measurement is oracle-only and diagnostic; fitting and the plan amendment remain human-locked.
- Campaign governance and all hard stops remain unchanged.

**Quality gates**

- **core.intent** — pending
  - Dimension: intent
  - Criterion: The result exactly covers the local CELLECT verdict, methodology proposal, and decision dossier without crossing the barrier.
  - Evidence: None
- **core.functional** — pending
  - Dimension: functional-correctness
  - Criterion: All 77 CELLECT shards and the unchanged per-class frozen gate complete, and the 38-bundle oracle replay reproduces an exact combined-score delta.
  - Evidence: None
- **core.regression** — pending
  - Dimension: regression
  - Criterion: Frozen identities, supplier math, DINOv2 harness properties, and campaign hard stops remain intact.
  - Evidence: None
- **core.failure-paths** — pending
  - Dimension: failure-paths
  - Criterion: Hash drift, incomplete shards, checkpoint/output splits, stale preflight, and any swap growth fail closed and remain resumable.
  - Evidence: None
- **core.completion-review** — pending
  - Dimension: completion-review
  - Criterion: A skeptical final evidence and diff review finds no unsupported number, missing requested field, or barrier breach.
  - Evidence: None
- **core.independent-review** — pending
  - Dimension: independent-review
  - Criterion: A separate skeptical review pass attempts to falsify the verdict, oracle arithmetic, and governance claims.
  - Evidence: None
- **backend.data-integrity** — pending
  - Dimension: data-integrity
  - Criterion: Every frozen input and generated shard is SHA-addressed; atomic writes and manifest counts prevent partial evidence from being accepted.
  - Evidence: None
- **backend.security** — pending
  - Dimension: security
  - Criterion: The run loads only the authenticated local checkpoint and safe loader and performs no network or external compute action.
  - Evidence: None
- **backend.concurrency** — pending
  - Dimension: concurrency
  - Criterion: Exactly one shard child runs at a time and atomic finalization makes interruption/retry safe without duplicate acceptance.
  - Evidence: None
- **backend.observability** — pending
  - Dimension: observability
  - Criterion: Preflight, each shard guard, each checkpoint, evaluation, oracle replay, and terminal state have diagnosable receipts without secret material.
  - Evidence: None

### Goal detail

# Flight plan

First reproduce the DINOv2 isolation contract for the pinned CELLECT supplier and validate it with focused negative tests. Run a fresh preflight and the resumable 77-shard feature stage, then evaluate only the unchanged sealed B0/A1/A0 gate. Separately replay the already frozen 38-bundle N1 oracle path, generate the per-event decision dossier and methodology note, update campaign state, and halt with fitting and amendment still locked.

Blocked outcome: five independent 45-second preflights passed with zero swap growth, but every immediate shard-0 start caused positive system swapout and was killed by the invariant. MPS batch 8, MPS batch 1 (cold and warm), CPU batch 1, and CPU batch 1 with one Torch thread all produced the same blocking class. No atomic feature or checkpoint exists and the frozen gate did not run. Repeating under the same ambient host state or weakening the zero-swap guard is prohibited; continuation requires a stable zero-swap host window.

## 2026-07-30T17:26:18.782Z — completed: One fresh APNCC transport kernel version proves or falsifies valid exact-parity submission production without submission.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-30T17:26:18.782Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
GOAL: prove the end-to-end submission kernel path works TODAY, before division needs it.
- Rebuild and launch the banked APNCC-direct package (manifest 779aa8d3…606b) as a Kaggle kernel.
- Success = a valid submission.csv is produced with exact expected row counts and CV57-parity
  fragment identity. DO NOT SUBMIT -- re-submitting a known 0.881 candidate teaches nothing about
  transfer and burns a slot.
- Apply the hardened packaging learned this week: canonical PEP 427 wheels, byte-SHA verification,
  --no-index --find-links, targeted <5s step 0 asserting every declared input, ONE consolidated
  dataset, serial launch lock (assert version delta == 1), authoritative Kaggle quota reconcile
  before and after (balance 25.47 h).
- Report: does the current transport stack produce a valid submission.csv end to end, yes or no.
  If no, the failing layer is the single most important thing to fix this week.
```

### Goal

**Outcome:** One fresh APNCC transport kernel version proves or falsifies valid exact-parity submission production without submission.

**Verification**

- python -m unittest discover -s reports/g02/screening/apncc_transport_smoke_v1 -p 'test_*.py'
- python reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- python reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py --kernel-dir reports/g02/screening/apncc_transport_smoke_v1/kernel --contract reports/g02/screening/apncc_transport_smoke_v1/CONTRACT.json --registry reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json --phase static

**Touchpoints**

- reports/g02/screening/apncc_transport_smoke_v1
- reports/g02/screening/tiered_surface/kaggle_prelaunch
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- No competition submission or leaderboard measurement.
- No candidate, model, scorer, threshold, graph, 6bba, or CV57 scientific change.
- No retry after an identical transport failure.

**Invariants**

- The banked package identity remains 779aa8d3cefd701024cfb1870d56f3fbb0b29ad1b72f120838951932017d606b.
- The remote kernel has exactly one fresh version and uses one declared support dataset plus the competition source.
- Step 0 uses only standard-library imports, verifies each declared support input byte-for-byte, and fails at five seconds or later.
- All package dependency installation is offline through no-index and find-links.
- Output acceptance requires all four expected fragments and the combined CSV to be byte-identical to the frozen CV57-parity reference.
- The run is never submitted.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The completed smoke run proves or falsifies transport only, never a public resubmission or a scientific change.
  - Evidence: v1 terminal receipts record submission_action=false; the reviewer confirmed no submission or retry.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: The one remote run either produces the exact combined CSV and parity fragments or emits a definitive pre-heavy-work failure receipt identifying the transport layer.
  - Evidence: The remote result decisively falsified output production: Step 0 failed closed at 5.061967561s before the runner.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: The shared launch guard and its existing fixtures remain valid.
  - Evidence: APNCC tests passed 8/8, shared guard tests 15/15, registry freshness and inline guard passed.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Invalid wheel, missing mount, and stale-version paths fail closed before heavy work.
  - Evidence: Missing-mount negative test and remote five-second overrun both failed closed before heavy work.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical review finds no gap in launch serialization, quota reconciliation, output acceptance, or no-submit scope.
  - Evidence: Independent completion review PASS confirmed evidence-aligned no-submit, no-output, and zero-debit reporting.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: An independent reviewer attempts to find a transport or scope breach in the final package and receipts.
  - Evidence: Independent reviewer PASS after adversarial checks of lifecycle, cross-lane locking, terminal evidence, and wording.

### Goal detail

# Flight plan

Pin the historical APNCC-direct package and old accepted output before changing only
the execution transport. The kernel must be self-contained, use the established
single support dataset, remain offline and unsubmitted, and create exactly one fresh
remote version while holding the campaign launch lock. A terminal receipt will state
the decisive transport outcome and the authoritative pre/post GPU quota.

## 2026-07-30T17:46:44.348Z — completed: Record the approved division dependency amendment, park CELLECT R7, execute and independently audit four frozen CPU-first 44b6 division-realizer arms, characterize the four oracle false negatives, and halt at the 44b6 barrier.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-30T17:46:44.348Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
HUMAN DECISION — PLAN AMENDMENT APPROVED
Amend campaign_narrowing_and_hypothesis_dag_v1.md: division fitting is UNLOCKED from the
association-continuity prerequisite. Evidence basis: frozen-N1 division oracle 38 TP / 0 FP / 4 FN,
division Jaccard 0.904762, exact combined delta +0.092068; edge-only pool ceiling ~0.896 cannot reach
the target at any realization; six association mechanisms realized ~0; four representations failed at
the ~1e-4-prevalence operating point. Record the amendment with its evidence in LEDGER + LIVE_STATE.
Unchanged: 44b6-first kill lines, frozen scorer, submission/GPU/RunPod human gates, DEAD authority.

TASK 1 — Park CELLECT
- CELLECT R7 → PARK-C, blocked-on-infrastructure (Kaggle transport stop rule + local memory: five
  preflights passed, every shard-0 start tripped the swap guard). No further spend. Note that
  checkpoint size does not predict volumetric-inference memory, for future supplier triage.

TASK 2 — Division realizer wave (parallel arms, one worktree each, CPU-first)
Screen multiple FROZEN arms against the same 38-event target, each preregistered before outcomes,
each 44b6-first. Do NOT use GT bundles as features -- they define the target, not the input.
  A. Geometry/motion arm: parent→two-branch motif from displacement, splitting geometry, local
     cell-count change, and motion discontinuity.
  B. Persistence arm: daughter/grandchild survival + late-fork equivalence (scorer-semantic D1).
  C. Appearance arm: intensity redistribution across the fork using the existing frozen cache.
  D. Combined high-precision arm: intersection/consensus of A-C, tuned for precision not recall.
Report per arm: division TP/FP/FN, division Jaccard, edge adjusted/raw/node deltas, and the exact
combined delta. Pass = combined delta > 0 with division Jaccard >= 0.30 and NO edge regression.

TASK 3 — FP discipline (this is where the lever dies if mishandled)
- Enforce one-to-one event suppression; zero duplicate survivors. Report the realized-FP sensitivity
  explicitly (25 TP: 0/5/10 FP → +0.0595/+0.0532/+0.0481). Prefer fewer, higher-precision forks over
  recall. Any arm producing duplicate forks fails closed.

TASK 4 — Characterize the 4 FN
- The oracle repaired 38/42. Report why the remaining 4 were not repairable; if they form a distinct
  motif class, propose it as a separate node (propose only).

HOLD AT BARRIER for the 44b6 results. Hard stops unchanged: submission, DEAD, >=5h GPU, RunPod,
full-pool cache.
```

### Goal

**Outcome:** Record the approved division dependency amendment, park CELLECT R7, execute and independently audit four frozen CPU-first 44b6 division-realizer arms, characterize the four oracle false negatives, and halt at the 44b6 barrier.

**Verification**

- .venv/bin/python -m json.tool reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave/WAVE_TERMINAL_RESULT.json
- .venv/bin/python reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave/arm_c_appearance/test_arm_c.py
- git diff --check

**Touchpoints**

- reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md
- reports/g02/screening/LEDGER.md
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/tiered_surface/tier_census.json
- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1
- reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave

**Non-goals**

- No 6bba, full CV57, graph, submission, Kaggle/GPU launch, RunPod, full-pool cache, DEAD action, or further plan amendment.
- No use of GT repair bundles as model features and no outcome-driven tuning or rerun.

**Invariants**

- Every arm is separately preregistered before its prediction outcome and stops at the frozen 44b6 barrier.
- The frozen N1/scorer and pass rule remain unchanged.
- One-to-one event suppression yields zero duplicate survivors or the arm fails closed.
- Invalid information access, missing frozen inputs, and invalid component inputs remain non-promotable.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The amendment, CELLECT decision, four-arm barrier result, FP sensitivity, and four-FN characterization exactly match the owner request and preserve every hard stop.
  - Evidence: The approved dependency amendment, CELLECT PARK-C decision, terminal four-arm wave, FP sensitivity, and four-FN characterization are all recorded with every hard stop preserved.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Each arm has an authenticated preregistration and bounded 44b6 terminal state; governed metrics and the pass rule are computed and reported correctly.
  - Evidence: All four arm receipts parse and hash correctly; 44b6 scope, governed metrics, score arithmetic, preregistrations, and zero-duplicate outputs were verified.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Frozen scorer/N1 contracts, tier census, central Markdown, and focused arm tests remain internally consistent after final integration.
  - Evidence: Eight CELLECT tests, three appearance-arm tests, Python compilation, all JSON parses, family checks, and the final diff check passed.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Firewall contamination, inadequate cache inputs, invalid consensus inputs, duplicate survivors, and forbidden downstream actions all fail closed.
  - Evidence: Persistence contamination, missing appearance inputs, invalid consensus inputs, and every forbidden downstream action are explicitly fail-closed.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final evidence review finds no unsupported PASS, tier move, retry, or downstream unlock.
  - Evidence: Final skeptical reconciliation found no unsupported PASS, tier move, retry, stale open-wave claim, broken ledger link, or downstream unlock.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: An independent reviewer verifies arm integrity, governed status, exact metrics, and barrier closure.
  - Evidence: Independent review verifies A as the sole valid FAIL, B/C/D as fail-closed or invalid, zero valid passes, zero duplicates, and HOLD_44B6.

### Goal detail

# Flight plan

Preserve the human-approved dependency amendment and frozen campaign contracts.
Treat the four arm outputs according to their information and input integrity,
not merely their raw arithmetic. Bank only valid evidence, explicitly quarantine
diagnostic-only results, reconcile the root ledger and live state, and stop with
all downstream gates closed.

## 2026-07-31T03:39:05.282Z — completed: Adjudicate APNCC v2 against the authenticated public-0.881 T4 archive, activate no compute fallback unless a real regression exists, adopt the measured swap guard without reopening CELLECT, and preserve the active full-38 division/PEFT work at the barrier.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T03:39:05.282Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
TASK 1 — Decisive zero-cost parity diagnosis (do this first)
- Diff the v2 remote output (263,282 rows, four fragment hashes) against the BANKED successful
  submission archive 031aa5f6…7219 from Q-APNCC-PUBLIC (which also ran on T4 and scored public 0.881).
- Report: identical / same-drift-signature / genuinely different. Quantify how many of the 1,096
  missing rows sit within 1e-6 of the geometry 0.605, NCC -0.04768145, or per-video cap boundaries.
- If the deficit matches known M5->T4 threshold-boundary drift, classify BENIGN (precedent: the
  Option-C BENIGN_FP classification) and declare the transport path VALID.

TASK 2 — Only if Task 1 shows a real regression
- (a) Rebuild the submission kernel as Kaggle CPU-only (enable_gpu: false). APNCC is geometry/NCC/
  assignment; verify it needs no GPU. CPU avoids cuDNN/BLAS nondeterminism and debits no GPU quota.
- (b) If CPU still drifts, adopt T4-CANONICAL scoring using the U0-T4CANON precedent: re-derive the
  CV57 reference on T4, verify T4<->T4 repeatability, and retire M5<->T4 equality as a gate.

TASK 3 — Swap guard: adopt the calibration
- Accept max(256 MiB, 5x baseline) = 256 MiB, 0.25s monitoring, 15s sustained abort, 384 MiB immediate.
- Note CELLECT's +488.7 MiB still exceeds the immediate-abort bound, so it remains blocked in practice;
  do NOT spend further on it. Its +3.5 MiB trips are retired as noise-confounded, as recorded.

TASK 4 — Keep the science moving in parallel (unchanged from the prior dispatch)
- Division: apply the scoped screening amendment (full 38-event population, both families; per-family
  reporting as diagnostic not gate), build the intensity cache, run Arm C, re-run Arm B target-blind.
- Open the PEFT escalation node per the locked plan §6.2 (its "escalate only if frozen fails"
  precondition is satisfied): frozen backbone + <1% adapter, selection on nested-OOF 44b6 graph delta.

HOLD AT BARRIER. Hard stops unchanged.
```

### Goal

**Outcome:** Adjudicate APNCC v2 against the authenticated public-0.881 T4 archive, activate no compute fallback unless a real regression exists, adopt the measured swap guard without reopening CELLECT, and preserve the active full-38 division/PEFT work at the barrier.

**Verification**

- python3 -B -m unittest discover -s reports/g02/screening/apncc_transport_smoke_v2 -p 'test_*v2.py'
- python3 -m unittest discover -s reports/g02/screening/tiered_surface/kaggle_prelaunch -p 'test_kaggle_prelaunch_guard.py'
- python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- git diff --check

**Touchpoints**

- results-bhap.zip
- reports/g02/screening/apncc_transport_smoke_v2
- reports/g02/screening/tiered_surface/kaggle_prelaunch
- reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1
- reports/g02/screening/tiered_surface/ALIGNMENT.md
- reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2
- reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- No public submission, DEAD action, RunPod, full-pool cache, or GPU action projected at five hours or more.
- No APNCC CPU/T4 fallback when the public archive proves identity; no APNCC relaunch or retry.
- No CELLECT rerun, spend, scientific verdict, or tier move.
- No full PEFT fine-tune or target-derived prediction feature/threshold.

**Invariants**

- Q-APNCC-PUBLIC parity is bound to archive SHA 031aa5f6…7219 and its own appearance-contract/output identities, never a baseline mechanism archive.
- A false parity reference cannot authorize new compute; fallback opens only on a genuine candidate-vs-candidate regression.
- The adopted swap guard continuously monitors and still aborts sustained growth at 256 MiB or immediate growth at 384 MiB.
- Division uses all 38 target events across both families; per-family reporting is diagnostic and association remains 44b6-first.
- GT repair bundles remain evaluation labels only; clean Arm B is sealed under its active forbidden-read firewall.
- PEFT remains a preregistered frozen-backbone <1% adapter selected on nested-OOF 44b6 graph delta.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: Parity, swap adoption, division/PEFT continuation, and unchanged hard stops match the current owner request.
  - Evidence: Final review confirms archive identity, calibrated no-spend guard, full-38 barrier, open-but-unlaunched PEFT, and every owner hard stop.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: The archive comparison discriminates identity from drift and the corrected builder selects the authenticated APNCC output reference.
  - Evidence: Ten focused tests and an independent byte comparison prove identical submission, four fragments, and normalized run semantics.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Step-0 hard checks, serial launch fences, active division firewall, and PEFT preregistration remain intact.
  - Evidence: APNCC, prelaunch, wave, clean-B, and clean-D suites pass; registry freshness and diff checks pass; frozen science receipts remain intact.
- **core.failure** — passed
  - Dimension: failure-paths
  - Criterion: Wrong-reference parity, missing archive identity, forbidden input, duplicate launch, and genuine swap growth fail closed without spending compute.
  - Evidence: Negative tests, KFG-20, terminal immutability, D provenance fail-close, and 256/384 MiB guard rules cover the relevant failure categories.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A skeptical review finds no unsupported drift claim, fallback launch, CELLECT reopening, scientific PASS, or barrier violation.
  - Evidence: Fresh skeptical final review passes and confirms D nonbinding, no fallback launch, no CELLECT reopening, and no scientific or downstream PASS.
- **backend.integrity** — passed
  - Dimension: data-integrity
  - Criterion: Archive, submission, fragment, mechanism, calibration, and state identities are exact and schema-valid.
  - Evidence: JSON validation and byte-SHA tests pass for the archive, submission, fragments, swap receipt, clean-B evidence, and D provenance adjudication.
- **backend.security** — passed
  - Dimension: security
  - Criterion: No credential, leaderboard submission, network mutation, or target-label leakage occurs.
  - Evidence: Command and firewall audit records no submission, credential use, network mutation, GPU launch, or clean-B target/scorer/evaluator read.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: Terminal APNCC cannot be rebuilt/relaunched and the clean division rerun remains isolated from this adjudication.
  - Evidence: Terminal-lock and serial tests pass, APNCC was not relaunched, and clean-B sealing remained isolated under its explicit read firewall.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Reference-selection false positive and real resource failures are classified at their actual layer.
  - Evidence: KFG-20 and LIVE_STATE distinguish the wrong parity oracle, real 488.7 MiB pressure, clean-B scientific FAIL, and clean-D provenance invalidity.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: An independent skeptical review reproduces public-archive identity and finds no barrier or guard regression.
  - Evidence: Independent reviewer reproduces archive identity and confirms the calibrated guard, current quota, A/B/C no-PASS barrier, and D fail-close.

### Goal detail

# Flight plan

Treat the authenticated public T4 archive as the candidate equality oracle,
retire the wrong baseline comparison, and make no external compute mutation.
Adopt the measured swap guard while preserving CELLECT's permanent no-spend
status. Do not duplicate the already-active clean division B/D firewall rerun;
reconcile it and the existing §6.2 PEFT preregistration at the barrier.

## 2026-07-31T03:43:34.239Z — completed: Correct the division falsifier to the full 38-event population, run target-blind A/B/C and a governed 3-of-3 D, preregister the plan-authorized PEFT escalation without launching it, and prove APNCC Step-0-to-submission.csv transport while holding every downstream barrier.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T03:43:34.239Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Division wave HELD at 44b6: one valid arm (geometry/motion, 0 TP / 1 FP), one governance-invalid arm,
one fail-closed (no intensity cache), one abstained. Scope covered only 9 of 38 repair-target events.
Governance unchanged.

TASK 1 — Fix the division screening scope (policy correction, evidence-based)
- 44b6-first is a cross-family GENERALIZATION guard, appropriate for association. Division's target
  population is 29/38 in 6bba, so a 9-event screen cannot falsify it.
- Amend the DIVISION falsifier scope ONLY: screen on the full 38-event population across both
  families; retain per-family reporting as a generalization diagnostic, not a gate. Association
  arms keep 44b6-first unchanged. Record as a scoped amendment with this rationale.
- Re-classify Arm A's 0/9 as INCONCLUSIVE-UNDERPOWERED rather than a surface kill.

TASK 2 — Actually run the appearance arm
- Build the bounded intensity cache required by Arm C (mitotic rounding / intensity redistribution
  across the fork is the strongest biological prior and it has never been tested). CPU-first;
  Kaggle CPU kernel if the local guard trips.
- Re-run Arm B cleanly with a target-blind design process (its predictor was target-free; the
  design search was not). Then re-run the 3-of-3 consensus with valid inputs.

TASK 3 — OPEN THE PEFT ESCALATION (the plan's own §6.2 next step)
- The §6.2 precondition "escalate only if frozen fails" is now satisfied: 4 representations and
  4 action classes at or below zero. Open a preregistered PEFT node under the §6.2 recipe:
  frozen backbone + LoRA/bottleneck adapter <1% of params; NEVER a full fine-tune; selection on
  nested-OOF 44b6 held-fit GRAPH delta (not proposal loss); layer-wise LRs, warmup+cosine, weight
  decay, dropout, grad clipping, short schedule with early stopping, fold/seed ensembling or SWA.
- Budget it explicitly: projected T4-h, <5h auto / >=5h human, against the authoritative 25.47h.
  Preregister the kill line before any outcome. This is the plan's designated escalation, not a
  new mechanism -- cite §6.2 in the preregistration.

TASK 4 — Finish the transport smoke
- Relaunch the APNCC transport smoke with the corrected Step-0 envelope. Until a valid submission.csv
  is produced end to end, no science can be delivered regardless of outcome.

HOLD AT BARRIER. Hard stops unchanged: submission, DEAD, >=5h GPU, RunPod, full-pool cache.
```

### Goal

**Outcome:** Correct the division falsifier to the full 38-event population, run target-blind A/B/C and a governed 3-of-3 D, preregister the plan-authorized PEFT escalation without launching it, and prove APNCC Step-0-to-submission.csv transport while holding every downstream barrier.

**Verification**

- .venv/bin/python -m unittest reports/g02/screening/apncc_transport_smoke_v2/test_transport_v2.py reports/g02/screening/apncc_transport_smoke_v2/test_serial_launch_v2.py
- .venv/bin/python -m unittest reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2/test_wave_v2.py
- .venv/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2/arm_b_persistence_clean -p 'test_*.py'
- .venv/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2/arm_d_consensus_clean -p 'test_*.py'
- .venv/bin/python -m unittest discover -s reports/g02/screening/tiered_surface/kaggle_prelaunch -p 'test_kaggle_prelaunch_guard.py'
- .venv/bin/python reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- git diff --check

**Touchpoints**

- reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md
- reports/g02/screening/tiered_surface/branches/sa_division_realizer_wave_v2
- reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1
- reports/g02/screening/apncc_transport_smoke_v2
- reports/g02/screening/tiered_surface/kaggle_prelaunch
- reports/g02/screening/LEDGER.md
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- No public submission, DEAD action, RunPod, full-pool cache, or GPU action projected at five hours or more.
- No APNCC CPU/T4 fallback, relaunch, retry, or scientific re-evaluation after authenticated public-archive identity passes.
- No full PEFT fine-tune or target-derived prediction feature or threshold.

**Invariants**

- Q-APNCC-PUBLIC parity is bound to archive SHA 031aa5f6…7219 and its own appearance-contract/output identities, never a baseline mechanism archive.
- Division uses all 38 target events across both families; per-family reporting is diagnostic and association remains 44b6-first.
- GT repair bundles remain evaluation labels only; clean Arm B is sealed under its active forbidden-read firewall.
- Any consensus lacking a pre-outcome implementation freeze fails closed as nonbinding; no retroactive freeze is permitted.
- PEFT remains a preregistered frozen-backbone <1% adapter selected on nested-OOF 44b6 graph delta.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The division scope correction, full-population arm results, PEFT preregistration, APNCC transport result, and unchanged hard stops match the current owner request.
  - Evidence: Plan §6.5, clean full-38 results, §6.2 PEFT preregistration, APNCC transport terminal, LEDGER, and LIVE_STATE all match the requested scope and hard stops.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: A/B/C are evaluated on all 38 targets, D fails closed if provenance is invalid, PEFT pins the real grouped split, and APNCC produces a valid end-to-end CSV identical to the authenticated public archive.
  - Evidence: Independent recount and integrity checks reproduce A/B/C failures, D nonbinding provenance, the exact three-fold PEFT pin, and APNCC public-archive identity.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Step-0 hard checks, serial launch fences, target firewall, association 44b6-first policy, and PEFT restrictions remain intact.
  - Evidence: Final suites pass 10 APNCC, 7 wave, 12 clean-B, 6 clean-D, and 16 prelaunch tests; registry freshness and git diff checks pass.
- **core.failure** — passed
  - Dimension: failure-paths
  - Criterion: Wrong-reference parity, missing archive identity, forbidden target reads, unfrozen consensus, duplicate launch, and malformed inputs fail closed without opening downstream compute.
  - Evidence: Negative tests and receipts fail closed on duplicate launch, forbidden reads, unfrozen D provenance, malformed transport inputs, and wrong-reference parity.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A skeptical review finds no unsupported scientific PASS, retroactive freeze, fallback launch, PEFT launch, or barrier violation.
  - Evidence: Fresh skeptical review 9ec635c9…d1fe5 finds no binding pass, retroactive freeze, PEFT launch, fallback compute, submission, or barrier violation.
- **backend.integrity** — passed
  - Dimension: data-integrity
  - Criterion: Plan, target, predictions, scorer results, PEFT folds, archive, submission, fragments, and state identities are exact and schema-valid.
  - Evidence: Final integrity script validates terminal result hashes, D nonbinding state, PEFT path/hash/fold count, JSON schemas, and APNCC submission/fragment equality.
- **backend.security** — passed
  - Dimension: security
  - Criterion: No credential exposure, leaderboard submission, unauthorized network mutation, or target-label feature or design leakage occurs.
  - Evidence: Firewall and governance receipts show zero target-design leakage for clean B, zero submission, no credential exposure, and no unauthorized external mutation.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: Terminal APNCC cannot be duplicated or relaunched and the clean Arm B process remains isolated from target and outcome evidence.
  - Evidence: Serial-launch and terminal-lock tests pass; the sole APNCC version is terminal and clean B's isolated provenance records zero forbidden or outcome reads.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Underpowered scope, target-design contamination, unfrozen consensus provenance, and wrong-reference parity are classified at their actual layers.
  - Evidence: LIVE_STATE and terminal reports distinguish underpowering, design contamination, D freeze provenance, schema validity, and wrong-reference parity at the correct layers.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: An independent skeptical review reproduces the final division, PEFT, and APNCC evidence and finds no barrier or governance regression.
  - Evidence: Independent review 9ec635c9…d1fe5 reproduces full-38 counts, D fail-closed provenance, PEFT three-fold integrity, APNCC identity, and the held barrier.

### Goal detail

# Flight plan

Use the full 9+29 repair-target population for the binding division falsifier,
preserve family slices as diagnostics, and fail closed on any
information-firewall or pre-outcome-freeze defect. Open §6.2 only through a
hash-pinned, bounded PEFT preregistration; do not fit or launch it. Treat the
authenticated public T4 archive as APNCC's transport equality oracle and make
no submission or fallback compute mutation. Finish at the human barrier.

## 2026-07-31T04:00:17.796Z — blocked: GOAL: launch the preregistered §6.2 PEFT node — the last untried lever. TARGET SELECTION (apply the measured requirement surface) - Train on the DELETE action-selection task: beneficial-vs-harmful over the frozen 1,345,497-action 44b6 population (1,118 beneficial). Ceiling +0.0819; required AUC 0.85, retention 0.020. - Rationale for not choosing others: swap ceiling only +0.0186 (below the needed gap even at oracle); restore generator yields 0 beneficial actions; division falsified 0/38 across three arms. DESIGN (unchanged from the frozen preregistration; §6.2 recipe) - Frozen backbone + LoRA/bottleneck adapter <1% of params. NEVER a full fine-tune. - Selection on nested-OOF 44b6 HELD-FIT GRAPH delta -- never training loss, never proposal metrics. - Layer-wise/discriminative LRs (tiny on backbone), warmup + cosine decay, weight decay, dropout, gradient clipping, short schedule with early stopping, fold/seed ensembling or SWA. - Report AUC and retention-at-CG0-budget against the MEASURED bar (0.85 / 0.020), not the old 0.65/0.15. BUDGET AND GATES - Authenticated projection first. <5h auto-approve with hard 5h kill; >=5h stops for human approval. - Serial launch lock (assert version delta == 1); reconcile the authoritative quota before and after (24.98 T4-h). Step 0 asserts all declared inputs with the 60s warning-only envelope. - Kill line preregistered before any outcome: 44b6 adjusted >= +0.005 with raw/node >= 0. IF IT PASSES 44b6 - Proceed 6bba -> CV57. If CV57 clears local ~0.895-0.900, package through the NOW-PROVEN transport path (byte-identical parity confirmed) and STOP for explicit human submission approval. ALSO - Record division as FALSIFIED-BY-REALIZATION: 0/38 across three independent arms at full scope, oracle ceiling +0.0921 unreachable from geometry/persistence/appearance evidence. Surface-level tier per your rules; the oracle result stands as the ceiling, not a promise.

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T04:00:17.796Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
GOAL: launch the preregistered §6.2 PEFT node — the last untried lever.

TARGET SELECTION (apply the measured requirement surface)
- Train on the DELETE action-selection task: beneficial-vs-harmful over the frozen 1,345,497-action
  44b6 population (1,118 beneficial). Ceiling +0.0819; required AUC 0.85, retention 0.020.
- Rationale for not choosing others: swap ceiling only +0.0186 (below the needed gap even at oracle);
  restore generator yields 0 beneficial actions; division falsified 0/38 across three arms.

DESIGN (unchanged from the frozen preregistration; §6.2 recipe)
- Frozen backbone + LoRA/bottleneck adapter <1% of params. NEVER a full fine-tune.
- Selection on nested-OOF 44b6 HELD-FIT GRAPH delta -- never training loss, never proposal metrics.
- Layer-wise/discriminative LRs (tiny on backbone), warmup + cosine decay, weight decay, dropout,
  gradient clipping, short schedule with early stopping, fold/seed ensembling or SWA.
- Report AUC and retention-at-CG0-budget against the MEASURED bar (0.85 / 0.020), not the old 0.65/0.15.

BUDGET AND GATES
- Authenticated projection first. <5h auto-approve with hard 5h kill; >=5h stops for human approval.
- Serial launch lock (assert version delta == 1); reconcile the authoritative quota before and after
  (24.98 T4-h). Step 0 asserts all declared inputs with the 60s warning-only envelope.
- Kill line preregistered before any outcome: 44b6 adjusted >= +0.005 with raw/node >= 0.

IF IT PASSES 44b6
- Proceed 6bba -> CV57. If CV57 clears local ~0.895-0.900, package through the NOW-PROVEN transport
  path (byte-identical parity confirmed) and STOP for explicit human submission approval.

ALSO
- Record division as FALSIFIED-BY-REALIZATION: 0/38 across three independent arms at full scope,
  oracle ceiling +0.0921 unreachable from geometry/persistence/appearance evidence. Surface-level
  tier per your rules; the oracle result stands as the ceiling, not a promise.
```

### Goal

_No current Flight was available for this run._

## 2026-07-31T04:01:31.613Z — blocked: Launch and supervise the preregistered §6.2 PEFT DELETE node under its <5-hour authenticated budget, measured supplier and graph gates, and conditional 6bba/CV57 continuation, without submission.

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T04:01:31.613Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
GOAL: launch the preregistered §6.2 PEFT node — the last untried lever.

TARGET SELECTION (apply the measured requirement surface)
- Train on the DELETE action-selection task: beneficial-vs-harmful over the frozen 1,345,497-action
  44b6 population (1,118 beneficial). Ceiling +0.0819; required AUC 0.85, retention 0.020.
- Rationale for not choosing others: swap ceiling only +0.0186 (below the needed gap even at oracle);
  restore generator yields 0 beneficial actions; division falsified 0/38 across three arms.

DESIGN (unchanged from the frozen preregistration; §6.2 recipe)
- Frozen backbone + LoRA/bottleneck adapter <1% of params. NEVER a full fine-tune.
- Selection on nested-OOF 44b6 HELD-FIT GRAPH delta -- never training loss, never proposal metrics.
- Layer-wise/discriminative LRs (tiny on backbone), warmup + cosine decay, weight decay, dropout,
  gradient clipping, short schedule with early stopping, fold/seed ensembling or SWA.
- Report AUC and retention-at-CG0-budget against the MEASURED bar (0.85 / 0.020), not the old 0.65/0.15.

BUDGET AND GATES
- Authenticated projection first. <5h auto-approve with hard 5h kill; >=5h stops for human approval.
- Serial launch lock (assert version delta == 1); reconcile the authoritative quota before and after
  (24.98 T4-h). Step 0 asserts all declared inputs with the 60s warning-only envelope.
- Kill line preregistered before any outcome: 44b6 adjusted >= +0.005 with raw/node >= 0.

IF IT PASSES 44b6
- Proceed 6bba -> CV57. If CV57 clears local ~0.895-0.900, package through the NOW-PROVEN transport
  path (byte-identical parity confirmed) and STOP for explicit human submission approval.

ALSO
- Record division as FALSIFIED-BY-REALIZATION: 0/38 across three independent arms at full scope,
  oracle ceiling +0.0921 unreachable from geometry/persistence/appearance evidence. Surface-level
  tier per your rules; the oracle result stands as the ceiling, not a promise.
```

### Goal

**Outcome:** Launch and supervise the preregistered §6.2 PEFT DELETE node under its <5-hour authenticated budget, measured supplier and graph gates, and conditional 6bba/CV57 continuation, without submission.

**Verification**

- python3 -B -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1 -p 'test_*.py'
- python3 -B -m unittest discover -s reports/g02/screening/tiered_surface/kaggle_prelaunch -p 'test_kaggle_prelaunch_guard.py'
- python3 -B reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- git diff --check

**Touchpoints**

- reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1
- reports/g02/screening/tiered_surface/branches/sa_assignment_separation_v1
- reports/g02/screening/tiered_surface/kaggle_prelaunch
- reports/g02/screening/tiered_surface/tier_census.json
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md

**Non-goals**

- No full fine-tune, trainable fraction at or above one percent, RunPod, full-pool cache, or automatic public submission.
- No GPU launch when the authenticated projection is five hours or more.
- No 6bba, CV57, transport packaging, or submission path unless every preceding measured and graph gate passes.
- No resurrection of swap, restore-topology, or division realizers.

**Invariants**

- The DELETE population is exactly 1,345,497 frozen 44b6 actions with exactly 1,118 beneficial labels and the measured AUC/retention bar is 0.85/0.020.
- Backbone weights remain frozen and exact trainable adapter parameters are strictly less than one percent of authenticated base parameters.
- Model or checkpoint selection uses nested-OOF held-fit graph adjusted-edge delta, never training loss or proposal metrics.
- The 44b6 graph kill line remains adjusted delta at least +0.005 with raw and node deltas nonnegative.
- Step 0 hard-fails declared-input category violations but logs elapsed time after full verification under the 60-second warning envelope.
- One serial launch produces exactly one new remote version; quota is authoritatively reconciled before and after.
- A passing CV57 candidate is packaged through the proven transport and then stops before submission for explicit human approval.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: Execution matches the authorized PEFT DELETE node, measured bars, conditional cascade, and explicit submission stop.
  - Evidence: Execution stayed within the authorized PEFT DELETE scope and stopped before mutation when the frozen three-fold envelope proved invalid.
- **core.functional** — failed
  - Dimension: functional-correctness
  - Criterion: The package trains only the sub-one-percent adapter and reports authentic nested-OOF supplier and graph outcomes on the frozen population.
  - Evidence: The frozen 1,345,497-action population has zero grouped-fold-0 support, so no authentic three-outer-fold nested-OOF result can be produced.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing prelaunch, scoring, transport, tier, and campaign hard-stop contracts remain intact.
  - Evidence: No dataset or kernel mutation, GPU debit, downstream gate, or submission occurred; targeted state validation passed.
- **core.failure** — passed
  - Dimension: failure-paths
  - Criterion: Projection, identity, parameter fraction, Step-0, five-hour kill, graph gate, and conditional cascade failures stop at the correct layer.
  - Evidence: The prelaunch category guard fired at the missing outer-fold layer before projection, upload, or optimization; verifier 4c73d4f3...47701 passes.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no outcome-aware retuning, hidden full fine-tune, quota drift, duplicate launch, unsupported tier move, or submission.
  - Evidence: Fresh root completion review found no outcome-aware change, fine-tune, quota drift, duplicate launch, unsupported DEAD move, or submission.
- **backend.integrity** — failed
  - Dimension: data-integrity
  - Criterion: All source, population, feature, label, fold, wheel, checkpoint, output, and state identities are exact and schema-valid.
  - Evidence: All byte identities are exact, but the declared three-fold result schema is unsatisfiable because grouped fold 0 has zero population rows.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Credentials remain outside artifacts; the kernel is offline and no submission or unauthorized external action occurs.
  - Evidence: Only a read-only authenticated quota query was made; no credential material entered artifacts and no unauthorized action occurred.
- **backend.concurrency** — not_applicable
  - Dimension: concurrency
  - Criterion: The serial lock prevents duplicate in-flight G-02 work and asserts a remote version delta of exactly one.
  - Evidence: The serial launch stage was never entered because the data-integrity gate failed earlier; no remote mutation occurred.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Projection, Step 0, training, supplier gates, graph gates, quota, and terminal failure layers are independently diagnosable.
  - Evidence: The hard stop is isolated to the outer-fold contract and records exact counts, byte-SHAs, quota, zero-debit state, and required decision.
- **core.independent** — unavailable
  - Dimension: independent-review
  - Criterion: An independent skeptical review reproduces the frozen identities, budget decision, outcome classification, and downstream stops.
  - Evidence: No experiment outcome exists to review independently; the deterministic verifier replays the blocking fold intersection.

### Goal detail

# Flight plan

Authenticate the frozen DELETE population and PEFT implementation surface,
prove the complete T4 projection is below five hours, and only then acquire the
serial launch lock. Supervise exactly one remote version through terminal output,
reconcile quota, apply supplier and graph kill lines without retuning, continue
conditionally through 6bba/CV57 only on PASS, and stop before any submission.

## Blocking condition

The frozen action population has no grouped-fold-0 rows, while the frozen
preregistration requires three outer-fold graph results and fail-closes a
missing fold. Substituting the separate five embedded hash folds would amend
the design after freeze. No authenticated projection or launch is valid until
the human owner chooses a preregistration amendment or a complete
three-outer-fold data contract.

## 2026-07-31T04:30:49.636Z — completed: Launch and supervise the preregistered §6.2 PEFT DELETE node under its <5-hour authenticated budget, measured supplier and graph gates, and conditional 6bba/CV57 continuation, without submission.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T04:30:49.636Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
GOAL: launch the preregistered §6.2 PEFT node — the last untried lever.

TARGET SELECTION (apply the measured requirement surface)
- Train on the DELETE action-selection task: beneficial-vs-harmful over the frozen 1,345,497-action
  44b6 population (1,118 beneficial). Ceiling +0.0819; required AUC 0.85, retention 0.020.
- Rationale for not choosing others: swap ceiling only +0.0186 (below the needed gap even at oracle);
  restore generator yields 0 beneficial actions; division falsified 0/38 across three arms.

DESIGN (unchanged from the frozen preregistration; §6.2 recipe)
- Frozen backbone + LoRA/bottleneck adapter <1% of params. NEVER a full fine-tune.
- Selection on nested-OOF 44b6 HELD-FIT GRAPH delta -- never training loss, never proposal metrics.
- Layer-wise/discriminative LRs (tiny on backbone), warmup + cosine decay, weight decay, dropout,
  gradient clipping, short schedule with early stopping, fold/seed ensembling or SWA.
- Report AUC and retention-at-CG0-budget against the MEASURED bar (0.85 / 0.020), not the old 0.65/0.15.

BUDGET AND GATES
- Authenticated projection first. <5h auto-approve with hard 5h kill; >=5h stops for human approval.
- Serial launch lock (assert version delta == 1); reconcile the authoritative quota before and after
  (24.98 T4-h). Step 0 asserts all declared inputs with the 60s warning-only envelope.
- Kill line preregistered before any outcome: 44b6 adjusted >= +0.005 with raw/node >= 0.

IF IT PASSES 44b6
- Proceed 6bba -> CV57. If CV57 clears local ~0.895-0.900, package through the NOW-PROVEN transport
  path (byte-identical parity confirmed) and STOP for explicit human submission approval.

ALSO
- Record division as FALSIFIED-BY-REALIZATION: 0/38 across three independent arms at full scope,
  oracle ceiling +0.0921 unreachable from geometry/persistence/appearance evidence. Surface-level
  tier per your rules; the oracle result stands as the ceiling, not a promise.
```

### Goal

**Outcome:** Launch and supervise the preregistered §6.2 PEFT DELETE node under its <5-hour authenticated budget, measured supplier and graph gates, and conditional 6bba/CV57 continuation, without submission.

**Verification**

- python3 -B -m unittest discover -s reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1 -p 'test_*.py'
- python3 -B -m unittest discover -s reports/g02/screening/tiered_surface/kaggle_prelaunch -p 'test_kaggle_prelaunch_guard.py'
- python3 -B reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- git diff --check

**Touchpoints**

- reports/g02/screening/tiered_surface/branches/sa_peft_adapter_v1
- reports/g02/screening/tiered_surface/branches/sa_assignment_separation_v1
- reports/g02/screening/tiered_surface/kaggle_prelaunch
- reports/g02/screening/tiered_surface/tier_census.json
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md

**Non-goals**

- No full fine-tune, trainable fraction at or above one percent, RunPod, full-pool cache, or automatic public submission.
- No GPU launch when the authenticated projection is five hours or more.
- No 6bba, CV57, transport packaging, or submission path unless every preceding measured and graph gate passes.
- No resurrection of swap, restore-topology, or division realizers.

**Invariants**

- The DELETE population is exactly 1,345,497 frozen 44b6 actions with exactly 1,118 beneficial labels and the measured AUC/retention bar is 0.85/0.020.
- Backbone weights remain frozen and exact trainable adapter parameters are strictly less than one percent of authenticated base parameters.
- Model or checkpoint selection uses nested-OOF held-fit graph adjusted-edge delta, never training loss or proposal metrics.
- The 44b6 graph kill line remains adjusted delta at least +0.005 with raw and node deltas nonnegative.
- Step 0 hard-fails declared-input category violations but logs elapsed time after full verification under the 60-second warning envelope.
- One serial launch produces exactly one new remote version; quota is authoritatively reconciled before and after.
- A passing CV57 candidate is packaged through the proven transport and then stops before submission for explicit human approval.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: Execution matches the authorized PEFT DELETE node, measured bars, conditional cascade, and explicit submission stop.
  - Evidence: The owner-authorized five-fold runtime amendment was frozen before outcomes; one bounded DELETE-only version ran and stopped at the failed 44b6 gate without downstream execution or submission.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: The package trains only the sub-one-percent adapter and reports authentic nested-OOF supplier and graph outcomes on the frozen population.
  - Evidence: The 2,665-parameter adapter completed five-fold nested graph-only selection and full 1,345,497-action evaluation; independent replay matches the AUC, retention, and graph result exactly.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing prelaunch, scoring, transport, tier, and campaign hard-stop contracts remain intact.
  - Evidence: Focused PEFT tests pass 6/6, prelaunch guard tests 16/16, registry freshness and diff checks pass, and the failed 44b6 gate left 6bba/CV57/package/submission closed.
- **core.failure** — passed
  - Dimension: failure-paths
  - Criterion: Projection, identity, parameter fraction, Step-0, five-hour kill, graph gate, and conditional cascade failures stop at the correct layer.
  - Evidence: Inline Step 0 authenticated 90 members in 1.501 s with zero competition visits; calibration, hard-kill, serial, and graph failure paths remained active, and the adjusted graph miss stopped the cascade.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no outcome-aware retuning, hidden full fine-tune, quota drift, duplicate launch, unsupported tier move, or submission.
  - Evidence: Fresh closeout found no outcome-aware retuning, full fine-tune, quota drift, duplicate version, unsupported tier move, downstream gate violation, or submission.
- **backend.integrity** — passed
  - Dimension: data-integrity
  - Criterion: All source, population, feature, label, fold, wheel, checkpoint, output, and state identities are exact and schema-valid.
  - Evidence: Byte-SHA controls, canonical wheel, fixed sample digest, 41 population shards, five folds, full score shapes, selected effects, outputs, and quota receipts validate exactly.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Credentials remain outside artifacts; the kernel is offline and no submission or unauthorized external action occurs.
  - Evidence: The private offline kernel used only the authorized dataset and T4 version; credentials are absent from artifacts and no submission, RunPod action, or unauthorized external mutation occurred.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: The serial lock prevents duplicate in-flight G-02 work and asserts a remote version delta of exactly one.
  - Evidence: The exclusive lock covered prelaunch through terminal accounting; zero in-flight jobs and zero target versions preceded one push, and the post-push version delta is exactly one.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Projection, Step 0, training, supplier gates, graph gates, quota, and terminal failure layers are independently diagnosable.
  - Evidence: Projection, Step 0, optimizer steps, all epoch AUCs, selected retention, per-fold graph choices, aggregate deltas, remote status, and pre/post quota are recorded separately.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: An independent skeptical review reproduces the frozen identities, budget decision, outcome classification, and downstream stops.
  - Evidence: Fresh local replay passes 15/15 checks and reproduces the sample, score shapes, eight AUCs, retention, selected edge effects, graph failure, version delta, and submission stop.

### Goal detail

# Flight plan

Authenticate the frozen DELETE population and PEFT implementation surface,
prove the complete T4 projection is below five hours, and only then acquire the
serial launch lock. Supervise exactly one remote version through terminal output,
reconcile quota, apply supplier and graph kill lines without retuning, continue
conditionally through 6bba/CV57 only on PASS, and stop before any submission.

## Terminal result

The sole authorized v1 completed with exact version delta one and a 0.04
T4-hour debit. Graph-only nested selection produced AUC 0.496495, retention
0.002683, and adjusted/raw/node deltas
+0.000133827/+0.000133160/0. The adjusted +0.005 line failed, so the exact
realization is PARK-C / frozen capacity case B. The conditional 6bba, CV57,
transport, and submission stages did not run.

## 2026-07-31T05:00:42.099Z — completed: Bank the PEFT capacity/proxy/runtime findings, deliver an evidence-based exactly two-option human decision dossier on scoped Trackastra fine-tuning versus a compact 3D model, and persist the owner's standing Kaggle authorization without wasting quota on a duplicate submission.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T05:00:42.099Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
GOAL: close out the frozen-correction era and make ONE evidence-based decision about training.

TASK 1 — Bank the PEFT finding properly
- Record: adapter converged at AUC 0.635 (plateau over epochs 6-8) vs required 0.85; five approaches
  (incumbent 0.502, C1 0.502, PEFT 0.635, DINOv2 0.670) converge in 0.50-0.67. Conclusion: the
  limiting factor is representational information content, not adapter capacity or backbone choice.
- Record the selection paradox: graph-only nested selection preferred epochs 1-2 (AUC ~0.50) over
  epoch 8 (0.635); all epoch deltas ~1e-4. At this effect size the graph objective cannot guide model
  selection. This extends the H0 proxy ban to within-trajectory model selection.
- Record that the 4.8h projection resolved to 145s actual (~100x conservative) -- future compute
  projections must be recalibrated before they are used to rule anything out.

TASK 2 — Re-scope TRAINING feasibility (analysis only, NO training, NO GPU)
The DEAD verdict on training rests on a single figure: ~478 T4-h for organizer TemporalUNet3D to
epoch 300. Re-derive feasibility for the variants that were never costed:
  (a) Trackastra FINE-TUNE (27M params, already pretrained on cell tracking) on the 105-video fit set
      -- estimate T4-h to a useful checkpoint, not to full convergence.
  (b) A compact 3D association/detection model trained to a useful checkpoint.
  (c) For each: literature-grounded expected score range, T4-h estimate WITH the projection
      recalibration from Task 1, overfitting risk on 44b6, and the minimum viable experiment.
- Cite primary sources; state transfer domains honestly. Compare against what is publicly known about
  approaches at the 0.91+ tier.
- Output a decision dossier with exactly two options and their evidence: (1) open a scoped training
  node under a new preregistration, or (2) consolidate at 0.881 and stop. Recommend one. HALT for
  human decision -- this is a locked-plan amendment either way.

No GPU, no Kaggle, no submission, no DEAD promotion in this task.
```

### Goal

**Outcome:** Bank the PEFT capacity/proxy/runtime findings, deliver an evidence-based exactly two-option human decision dossier on scoped Trackastra fine-tuning versus a compact 3D model, and persist the owner's standing Kaggle authorization without wasting quota on a duplicate submission.

**Verification**

- python3 -B reports/g02/screening/tiered_surface/branches/training_feasibility_v1/validate_dossier.py
- git diff --check

**Touchpoints**

- reports/g02/screening/tiered_surface/branches/training_feasibility_v1
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md
- reports/g02/screening/KAGGLE_STANDING_AUTHORIZATION_20260731.md
- AGENTS.md
- /Users/agent/.codex/AGENTS.md

**Non-goals**

- No training, GPU kernel push, Kaggle dataset mutation, RunPod, full-pool cache, or observation-identical public resubmission.
- Authenticated Kaggle quota, submission-history, competition, and kernel-status reads are allowed.
- No edit to the human-locked campaign plan and no plan amendment decision.
- No DEAD promotion, resurrection, or removal.
- No third decision option or automatic launch recommendation.

**Invariants**

- PEFT measurements and graph-selection outcomes remain byte-faithful to the authenticated v1 result.
- Compute estimates separate measured facts, scaling calculations, recalibration assumptions, and uncertainty.
- Expected score ranges cite primary sources and explicitly disclose transfer-domain mismatch.
- The dossier contains exactly two mutually exclusive options and halts for human decision.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The dossier banks every requested PEFT finding, assesses both uncosted training variants, presents exactly two options, recommends one, and stops at the amendment barrier.
  - Evidence: Dossier, LIVE_STATE, and LEDGER bank every requested PEFT finding, assess both variants, contain exactly two options, recommend one, and halt at the human barrier.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: All local measurements, arithmetic, estimate ranges, option count, recommendation, and barrier statements validate deterministically.
  - Evidence: validate_dossier.py passed 25 deterministic checks, including source hashes, arithmetic, option count, recommendation, and no-action scope.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: The locked plan, DEAD register, existing verdicts, and external hard stops remain unchanged.
  - Evidence: Scoped diff review and git diff --check found no dossier formatting defect, plan mutation, DEAD change, training, kernel push, or duplicate submission.
- **core.failure** — passed
  - Dimension: failure-paths
  - Criterion: Unsupported transfer claims, false precision, the PEFT selector paradox, and projection uncertainty are surfaced rather than hidden.
  - Evidence: Adversarial review labels numeric score ranges as campaign envelopes, discloses every transfer mismatch, forbids a universal 118.6x factor, and preserves the selector paradox.
- **core.completion** — passed
  - Dimension: completion-review
  - Criterion: A fresh skeptical review finds no training action, missing requested evidence, third option, premature amendment, or unsupported score promise.
  - Evidence: Fresh completion review found no missing requested evidence, third option, automatic launch, premature amendment, unsupported 0.91-tier attribution, or unresolved in-scope gap.
- **backend.integrity** — passed
  - Dimension: data-integrity
  - Criterion: Campaign measurements, source identities, calculations, and state cross-links are exact and internally consistent.
  - Evidence: Evidence JSON replays the authenticated PEFT SHA, trajectory, selected epochs, graph deltas, runtime factor, and both cost calculations exactly.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Research is read-only; no credentials, private data, training, remote mutation, or submission is exposed or performed.
  - Evidence: No credential contents were read or emitted; remote calls were authenticated read-only queries, and no training, dataset mutation, kernel push, or submission occurred.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: No launch authorization, job, queue, or concurrent external state exists in this analysis-only task.
  - Evidence: Authenticated status checks found all recent campaign kernels terminal; the budget has no active authorization and no concurrent external work was created.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Measured facts, source-derived priors, estimates, uncertainty, kill lines, and the human barrier are separately auditable.
  - Evidence: EVIDENCE.json, the decision dossier, standing-authorization receipt, LIVE_STATE log, and LEDGER separate measurements, assumptions, sources, options, and barriers.
- **core.independent** — passed
  - Dimension: independent-review
  - Criterion: A separate skeptical pass challenges the recommendation, cost model, source transfer, and two-option framing and finds no unresolved material flaw.
  - Evidence: A separate skeptical reread challenged the recommendation, projection model, transfer claims, public-tier comparison, and standing authorization and found no unresolved material flaw.

### Goal detail

# Flight plan

Reconstruct the local PEFT and TemporalUNet cost evidence; gather primary
sources for Trackastra transfer, compact 3D training, and public 0.91+ methods;
derive bounded useful-checkpoint estimates with explicit recalibration; write
and deterministically validate an exactly two-option decision dossier; update
the shared state and halt for human amendment choice.

## 2026-07-31T07:41:44.511Z — completed: Open and execute the preregistered RAW3D-NOLINK-MVE information gate, stop downstream spend on its frozen verdict, and explicitly protect the banked Kaggle final candidate.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-07-31T07:41:44.511Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
HUMAN DECISION — OPTION 1 APPROVED. Open RAW3D-NOLINK-MVE and execute under the standing Kaggle
authority.

- Preregister before any outcome: population, folds, probe, metric, and the kill line. Selection on
  nested-OOF 44b6 held-fit GRAPH delta only -- never training loss, never AUC (the proxy ban now
  explicitly covers within-trajectory checkpoint selection).
- Report against the MEASURED requirement: AUC 0.85 / retention 0.020 at the CG0 budget, and against
  the current best raw-pixel result (DINOv2-XY 0.670).
- RECOMMENDED first step inside the <=0.25h calibration: a zero-weight crude-3D information probe
  (local intensity stats, 3D gradients/texture at action endpoints) through the existing frozen
  separation gate. If crude 3D cannot beat 0.670, report that before spending the 2-6h GPU budget.
- Budget: 2-6 T4-h provisional, <5h auto-approve with hard kill; >=5h stops for human approval.
  Serial launch lock, authoritative quota reconcile before and after (24.94 h, refreshes 1 Aug 08:00
  SGT). Do not widen scope because quota is expiring.
- If it passes 44b6 -> 6bba -> CV57. At local >=0.895 with an observation-distinct prediction, package
  through the proven transport path and submit under the standing authority. Below that bar, do not
  submit -- the scientific rule stands even though the permission gate is lifted.

STILL OPEN (bookkeeping, please answer): the competition's final-submission selection state and
deadline. Confirm the banked 0.881522/public-0.881 candidate is locked as a final rather than relying
on a platform default. This protects the only result you currently hold.
```

### Goal

**Outcome:** Open and execute the preregistered RAW3D-NOLINK-MVE information gate, stop downstream spend on its frozen verdict, and explicitly protect the banked Kaggle final candidate.

**Verification**

- ./.venv313/bin/python -m unittest -v reports/g02/screening/tiered_surface/branches/raw3d_nolink_mve_v1/test_stage0.py
- ./.venv313/bin/python reports/g02/screening/tiered_surface/branches/raw3d_nolink_mve_v1/validate_stage0.py
- ./.venv313/bin/kaggle competitions submissions biohub-cell-tracking-during-development --format json

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md
- reports/g02/screening/tiered_surface/branches/raw3d_nolink_mve_v1/
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/LEDGER.md
- reports/g02/screening/KAGGLE_FINAL_SELECTION_20260731.json
- reports/g02/screening/KAGGLE_FINAL_SELECTION_20260731.md

**Non-goals**

- Do not reopen a DEAD learned crop-linker or inspect outcome before preregistration.
- Do not launch RAW3D GPU work after a failed zero-weight continuation gate.
- Do not submit below local 0.895 or without an observation-distinct prediction.
- Do not use RunPod, a persistent full-pool representation cache, or quota expiry to widen scope.

**Invariants**

- The complete frozen DELETE background and all five video-group folds determine separation metrics.
- Training loss, proposal metrics, and AUC never select a learned checkpoint.
- A projected RAW3D GPU run at or above five hours returns to human review.
- Final-selection bookkeeping does not create a new competition submission.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The terminal action and report match the approved RAW3D scope and Kaggle final-protection request.
  - Evidence: Owner scope, preregistered CPU-first execution, frozen stop, and explicit final protection agree.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: The frozen zero-weight probe scores all declared actions and applies its pre-outcome continuation rule exactly.
  - Evidence: Seven synthetic tests and all thirty-one immutable terminal checks passed independently.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Frozen inputs, scorer order, downstream gates, and unrelated campaign authority remain unchanged.
  - Evidence: All implementation and input pins match; the plan edit is scoped to decision 6 and preserves later authority.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: A failed Stage-0 continuation rule closes acquisition, training, GPU, graph, 6bba, CV57, and RAW3D submission.
  - Evidence: Result authority fields are false downstream and current Kaggle history contains no RAW3D kernel or submission.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no unresolved in-scope scientific, compute, or bookkeeping gap.
  - Evidence: Final review confirms pre-outcome freeze, exact arithmetic, zero RAW3D GPU spend, and durable final selection.
- **backend.data-integrity** — passed
  - Dimension: data-integrity
  - Criterion: Population, feature, preregistration, result, and selection identities remain internally consistent and hash-bound.
  - Evidence: Recomputed hashes match all frozen pins and validation reports 31 of 31 checks passing.
- **backend.security** — passed
  - Dimension: security
  - Criterion: Authentication secrets are not persisted in campaign receipts or command output and no unauthorized external write occurs.
  - Evidence: Reviewed artifacts contain identities and outcomes only; the sole external mutation was the explicitly authorized final selection.
- **backend.concurrency** — passed
  - Dimension: concurrency
  - Criterion: RAW3D launches no kernel and any separate active kernel is not misattributed to this node.
  - Evidence: RAW3D used CPU only; the later running g02-trackastra-native-fit-v1 is separately preregistered and serially launched.
- **backend.observability** — passed
  - Dimension: observability
  - Criterion: Result metrics, resource use, final selection, deadline, and downstream authority are recorded in terminal and shared state artifacts.
  - Evidence: Terminal report, result, validation, ledger, live state, and selection receipts contain the required facts.
- **web.accessibility** — passed
  - Dimension: accessibility
  - Criterion: Final-selection state is represented in a durable text and JSON receipt independent of visual-only UI state.
  - Evidence: Text and JSON receipts record 0-of-2 to 1-of-2, exact submission identity, and Kaggle final-selection semantics.
- **web.responsive** — passed
  - Dimension: responsive-ui
  - Criterion: Final-selection bookkeeping does not depend on a viewport-specific control interpretation.
  - Evidence: Receipt binds selected count and submission reference rather than coordinates or viewport layout.
- **web.browser-runtime** — passed
  - Dimension: browser-runtime
  - Criterion: The authenticated Kaggle page action and current API submission state agree on the banked candidate.
  - Evidence: Authenticated selection receipt and fresh Kaggle API history agree on COMPLETE ref 54942864 at public 0.881.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: An independent skeptical pass finds no post-outcome gate movement, leakage, false GPU attribution, or unsupported final-selection claim.
  - Evidence: Skeptical pass confirmed six-minute freeze-to-result chronology, zero feature label interface, fixed gate, no RAW3D GPU, and consistent selection evidence.

### Goal detail

# RAW3D-NOLINK-MVE flight

The approved node begins with a bounded, zero-weight crude-3D endpoint-information
probe. Its preregistered failure closes the RAW3D GPU branch. Final-selection
bookkeeping protects the already scored APNCC candidate without creating a new
submission. The active native-base kernel observed later is a separate authorized
node and is not part of this Flight's scientific result.

## 2026-07-31T15:44:28.185Z — aborted

- **Status:** aborted
- **Ended:** 2026-07-31T15:44:28.185Z
- **Trace:** unavailable
## 2026-08-01T01:44:15.771Z — blocked: <task> Mid-task course correction on the global-reassign-oracle node (repo /Users/agent/Desktop/cell_tracking, branch main). The prior blocker ESCALATE_PREREG_COMMIT_DENIED is resolved: the commander performed the commit from an unsandboxed session. Commit c493a67 on main now contains the sealed preregistration and full harness: - dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_PREREGISTRATION.md - dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_PREREGISTRATION.sha256 - dag_prompts/2026-08-01-global-reassign-oracle/CODEX_DISPATCH.md - dag_prompts/2026-08-01-global-reassign-oracle/global_reassign_oracle.py - dag_prompts/2026-08-01-global-reassign-oracle/run_guarded.py - dag_prompts/2026-08-01-global-reassign-oracle/test_global_reassign_oracle.py Independently verified by the coordinator: N0_PREREGISTRATION.md hashes to 6fe5ec6e6b11ad833f84988a2535d0dcfea1900c6a962db25c4ba6da97d7e879, matching the declared seal. Verify this yourself against N0_PREREGISTRATION.sha256 before proceeding. The preregistration is now immutable and must NOT be edited -- if execution requires any change to it, that is ESCALATE, not an edit. Note the pinned "pre-preregistration HEAD" of 8ca1b73 in the preregistration remains correct as written; c493a67 is the preregistration commit itself. Resume the node from the T3 tier and run it to completion under the unchanged contract. Everything else is as before and still binding: - Local CPU only -- no Kaggle asset/dataset/kernel/GPU debit/submission, no RunPod, under any circumstance. - Do not modify the frozen official scorer or the frozen N1 feasibility compiler. - Tier order: T3 (with its preregistered +0.030 early stop) first, then integrity controls T0 and T1 (must reproduce to 1e-9 or emit INVALID_HARNESS), then T2 at chain lengths L in {2,3,4,unbounded}. - Full 36-video 44b6, no subsampling. - 45-second clean-idle preflight; abort on sustained swapout >=256 MiB; per-video fragments written atomically for resumability; 6h wall-clock cap with ESCALATE_PROJECTED_WALL_CAP (not reduced coverage) if T2 projects past it. - Label every number [GT-DEPENDENT]. No selector, no AUC, no training, no representation. - The preregistration correctly identified that the T1 delete control replays a 41-video run_v6 population distinct from the 36-video experimental scope, and excludes T1 from the T2-minus-T0 arithmetic -- hold that exactly as sealed. - Section 4 decision arithmetic: report the precedent-adjusted number (0.2449 x (T2(unbounded) - T0)) BEFORE the raw ceiling; emit STOP_CLASS if that precedent-adjusted figure is < +0.005. - Do not write reports/g02/screening/LEDGER.md -- propose the row only. Update reports/g02/screening/LIVE_STATE.md per its protocol (refresh frontier, append one dated Log line, newest first). - If git operations are still sandbox-denied for anything besides the already-landed commit, do not block on them -- write artifacts to the working tree, report exactly what needs committing, and continue the science. </task> <default_follow_through_policy> Do not re-derive or re-edit the sealed preregistration; treat it as immutable input. Follow the contract's own tier sequencing and stop-loss rules as final authority. Only pause for ESCALATE where the contract mandates it (6h wall-clock overrun projection, compiler inexpressibility, sustained swapout >=256 MiB, or any required edit to the sealed preregistration). Treat STOP_CLASS and INVALID_HARNESS as complete, valid terminal outcomes to report plainly, never as failures to soften or route around. </default_follow_through_policy> <action_safety> Local CPU only -- never touch Kaggle or RunPod in any form. Never modify the frozen official scorer or the frozen N1 feasibility compiler; if the contract's chained actions cannot be expressed in the frozen compiler, ESCALATE and record it as a finding rather than editing the compiler. Do not edit N0_PREREGISTRATION.md. Do not write reports/g02/screening/LEDGER.md directly -- only propose the row. </action_safety> <structured_output_contract> Final report must state exactly: - Exact adjusted/raw/node/combined deltas for T3, T0, T1, and T2 at all four L values, each labeled [GT-DEPENDENT]. - Correct edges installed, false positives removed, compiled chains, and the chain-length histogram. - The precedent-adjusted figure 0.2449 x (T2(unbounded) - T0), stated BEFORE the raw increment. - The section 4 verdict: STOP_CLASS or CONTINUE_PROPOSAL_ONLY. - The compiler-expressibility finding. - Measured wall-clock, preflight duration, peak process-tree RSS, and swap growth. - Any ESCALATE token triggered, stated plainly with no hedging. </structured_output_contract>

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-01T01:44:15.771Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
<task>
Mid-task course correction on the global-reassign-oracle node (repo /Users/agent/Desktop/cell_tracking, branch main). The prior blocker ESCALATE_PREREG_COMMIT_DENIED is resolved: the commander performed the commit from an unsandboxed session. Commit c493a67 on main now contains the sealed preregistration and full harness:
- dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_PREREGISTRATION.md
- dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_PREREGISTRATION.sha256
- dag_prompts/2026-08-01-global-reassign-oracle/CODEX_DISPATCH.md
- dag_prompts/2026-08-01-global-reassign-oracle/global_reassign_oracle.py
- dag_prompts/2026-08-01-global-reassign-oracle/run_guarded.py
- dag_prompts/2026-08-01-global-reassign-oracle/test_global_reassign_oracle.py

Independently verified by the coordinator: N0_PREREGISTRATION.md hashes to 6fe5ec6e6b11ad833f84988a2535d0dcfea1900c6a962db25c4ba6da97d7e879, matching the declared seal. Verify this yourself against N0_PREREGISTRATION.sha256 before proceeding. The preregistration is now immutable and must NOT be edited -- if execution requires any change to it, that is ESCALATE, not an edit. Note the pinned "pre-preregistration HEAD" of 8ca1b73 in the preregistration remains correct as written; c493a67 is the preregistration commit itself.

Resume the node from the T3 tier and run it to completion under the unchanged contract. Everything else is as before and still binding:
- Local CPU only -- no Kaggle asset/dataset/kernel/GPU debit/submission, no RunPod, under any circumstance.
- Do not modify the frozen official scorer or the frozen N1 feasibility compiler.
- Tier order: T3 (with its preregistered +0.030 early stop) first, then integrity controls T0 and T1 (must reproduce to 1e-9 or emit INVALID_HARNESS), then T2 at chain lengths L in {2,3,4,unbounded}.
- Full 36-video 44b6, no subsampling.
- 45-second clean-idle preflight; abort on sustained swapout >=256 MiB; per-video fragments written atomically for resumability; 6h wall-clock cap with ESCALATE_PROJECTED_WALL_CAP (not reduced coverage) if T2 projects past it.
- Label every number [GT-DEPENDENT]. No selector, no AUC, no training, no representation.
- The preregistration correctly identified that the T1 delete control replays a 41-video run_v6 population distinct from the 36-video experimental scope, and excludes T1 from the T2-minus-T0 arithmetic -- hold that exactly as sealed.
- Section 4 decision arithmetic: report the precedent-adjusted number (0.2449 x (T2(unbounded) - T0)) BEFORE the raw ceiling; emit STOP_CLASS if that precedent-adjusted figure is < +0.005.
- Do not write reports/g02/screening/LEDGER.md -- propose the row only. Update reports/g02/screening/LIVE_STATE.md per its protocol (refresh frontier, append one dated Log line, newest first).
- If git operations are still sandbox-denied for anything besides the already-landed commit, do not block on them -- write artifacts to the working tree, report exactly what needs committing, and continue the science.
</task>

<default_follow_through_policy>
Do not re-derive or re-edit the sealed preregistration; treat it as immutable input. Follow the contract's own tier sequencing and stop-loss rules as final authority. Only pause for ESCALATE where the contract mandates it (6h wall-clock overrun projection, compiler inexpressibility, sustained swapout >=256 MiB, or any required edit to the sealed preregistration). Treat STOP_CLASS and INVALID_HARNESS as complete, valid terminal outcomes to report plainly, never as failures to soften or route around.
</default_follow_through_policy>

<action_safety>
Local CPU only -- never touch Kaggle or RunPod in any form. Never modify the frozen official scorer or the frozen N1 feasibility compiler; if the contract's chained actions cannot be expressed in the frozen compiler, ESCALATE and record it as a finding rather than editing the compiler. Do not edit N0_PREREGISTRATION.md. Do not write reports/g02/screening/LEDGER.md directly -- only propose the row.
</action_safety>

<structured_output_contract>
Final report must state exactly:
- Exact adjusted/raw/node/combined deltas for T3, T0, T1, and T2 at all four L values, each labeled [GT-DEPENDENT].
- Correct edges installed, false positives removed, compiled chains, and the chain-length histogram.
- The precedent-adjusted figure 0.2449 x (T2(unbounded) - T0), stated BEFORE the raw increment.
- The section 4 verdict: STOP_CLASS or CONTINUE_PROPOSAL_ONLY.
- The compiler-expressibility finding.
- Measured wall-clock, preflight duration, peak process-tree RSS, and swap growth.
- Any ESCALATE token triggered, stated plainly with no hedging.
</structured_output_contract>
```

### Goal

_No current Flight was available for this run._

## 2026-08-01T03:45:53.883Z — completed: Complete the frozen G03 Phase-1 B-v2 continuation from an exact immutable A handoff within the justified eight-T4-hour ceiling, then produce a decision-ready audit of every unexplored or PARK-A direction under corrected 30h and 90h continuation gates.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-01T03:45:53.883Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
# Response annotations:
Each item contains text selected from an earlier Codex response and may include a user comment. Treat items as Annotation 1, Annotation 2, and so on in array order. Use every selection as context and address every comment. When addressing multiple comments, label each answer with its annotation number (for example, `Annotation 1`) so the user can match it to the numbered annotation.
<response-annotations>
[{"text":"Finish G03 Phase 1 with B v2. Publish the exact A handoff as the immutable dataset and launch the already-frozen B package. It is budgeted at roughly one T4-hour. If B mismatches, stop—no v3.If B passes, prepare rather than rush Phase 2. Freeze the raw-crop contract, adapter census, VRAM/examples-per-second calibration, output inventory, and 30/60/90-hour graph checkpoints. The current architecture still needs real crop-handling and cost evidence before scientific training."}]
</response-annotations>

## My request for Codex:
30h continuation already should require at least a 0..01 improvement, so 90h will require minimally 0.035 (3 weeks of compute)
let's pursue 1 and use as much t4 hours as needed where justified (max 8h)

consider all other unexplored/parka dirs
```

### Goal

**Outcome:** Complete the frozen G03 Phase-1 B-v2 continuation from an exact immutable A handoff within the justified eight-T4-hour ceiling, then produce a decision-ready audit of every unexplored or PARK-A direction under corrected 30h and 90h continuation gates.

**Verification**

- .venv/bin/python -m unittest -q tests.test_validate_g03_phase1_b_prelaunch tests.test_cmux_ops
- .venv/bin/python reports/g02/screening/tiered_surface/kaggle_prelaunch/test_kaggle_prelaunch_guard.py -q
- .venv/bin/python reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
- .venv/bin/python -m py_compile scripts/cmux_ops.py scripts/validate_g03_phase1_b_prelaunch.py
- node /Users/agent/.codex/plugins/cache/agent-local/cruise/0.4.1/scripts/cruise.mjs attest --json

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- scripts/validate_g03_phase1_b_prelaunch.py
- tests/test_validate_g03_phase1_b_prelaunch.py
- reports/g03/screening/
- reports/g02/screening/
- /private/tmp/g03_phase1_handoff_v2_dataset*/
- /private/tmp/g03_phase1_b_v2_output*/
- Kaggle dataset onebitbrain/g03-phase1-handoff-v2
- Kaggle kernel onebitbrain/g03-phase1-resume-b-v2

**Non-goals**

- No Phase-2 scientific training launch in this run.
- No official competition submission from an infrastructure receipt.
- No RunPod use.
- No rerun of any DEAD mechanism without a newly named observation-distinct mechanism.
- No B-v3 after a B-v2 mismatch.
- No use of more than eight T4-hours; unused allowance is preserved.

**Invariants**

- The published handoff is byte-identical to A-v2's verified atomic handoff and is round-trip verified before B launch.
- The frozen B-v2 code, metadata, and embedded contract hashes remain unchanged.
- The shared Kaggle serial-launch lock, authoritative quota reconciliation, prelaunch integrity checks, runtime kill limits, exact version accounting, and terminal monitoring remain binding.
- The corrected owner gates are at least +0.010 by 30 T4-hours and at least +0.035 by 90 T4-hours; no intermediate 60-hour gate is invented.
- A B-v2 mismatch terminates Phase 1 with no v3.
- Only the root coordinator updates screening ledgers and live state after state changes.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The result completes B-v2 or records its terminal failure, preserves the corrected gates and eight-hour ceiling, and audits all unexplored/PARK-A directions.
  - Evidence: B-v2 completed, corrected gates and scope fences are preserved, and the final state/ledger/audit reconcile to the exact request.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: A handoff survives remote round-trip byte-for-byte, frozen B-v2 passes its prelaunch validator, and its terminal artifacts validate against the locked contract.
  - Evidence: Remote round-trip and frozen package guards passed; B terminal replay is deterministic and 41 focused tests pass.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing CMUX and G02 prelaunch protections remain green and DEAD/no-rerun governance is unchanged.
  - Evidence: CMUX/G03 tests, 17 legacy G02 guard tests, registry freshness, compilation, and diff hygiene all pass.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing or altered handoff members, stale package hashes, remote upload loss, kernel failure, timeout, quota drift, and B mismatch all fail closed with no v3 or scientific launch.
  - Evidence: Negative fixtures and code review confirm altered trees, stale hashes, semantic exit-zero errors, timeouts, mismatches, and unauthorized repeats fail closed.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final evidence review finds no unresolved in-scope gap or overclaim.
  - Evidence: Fresh post-mutation review found and corrected one train/evaluation wording ambiguity; no unresolved in-scope gap remains.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A genuinely separate skeptical review of the costly external action and scientific recommendation finds no unresolved defect.
  - Evidence: Separate adversarial review challenged false-pass, quota, graph-interpretation, submission, leakage, and census risks and resolved the sole wording finding.
- **domain.compute-governance** — passed
  - Dimension: compute-governance
  - Criterion: The B launch is observation-distinct, preregistered, serialized, quota-bracketed, terminally monitored, and consumes no more compute than justified within eight T4-hours.
  - Evidence: The sole version-1 push was preregistered, predicate-bound, serialized, terminally monitored, and used 1.01 T4-hours of the 8-hour maximum.
- **domain.direction-census** — passed
  - Dimension: direction-census
  - Criterion: Every current unexplored or PARK-A direction is accounted for once with evidence, prerequisites, expected information gain, cost, and a pursue/defer/retire recommendation.
  - Evidence: Machine cross-check accounts for all 14 PARK_A and 18 PARK_B IDs with prerequisites, cost, information value, and disposition.

### Goal detail

# Flight plan

This run executes only the already frozen G03 Phase-1 B-v2 transport continuation. The owner-corrected economic gates bind future scientific continuation: at least +0.010 improvement by 30 T4-hours and at least +0.035 by 90 T4-hours. The old 60-hour threshold is not silently retained or replaced; it requires an explicit later amendment before it can control a decision.

The A-v2 handoff must be published as a private immutable Kaggle dataset and downloaded back for exact verification before B-v2 is launched. B-v2 remains a roughly one-hour deterministic transport test; the eight-hour allowance is a hard ceiling available only if concrete evidence justifies it, not a compute target. A mismatch ends Phase 1 with no v3. If B passes, this run prepares Phase 2 and prioritizes the remaining direction space but does not launch scientific training or submit an infrastructure-only result.

## 2026-08-02T16:12:17.370Z — completed: Terminalize the R4 nested-OOF run at sealed receipts or the owner deadline, record its decision-grade finding, and encode the two durable campaign rules without changing a scientific gate

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-02T16:12:17.370Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
The R4 division-decode nested OOF run has all four folds in the final sealed scorer pass after
~11h. Liveness was confirmed twice (processes runnable at 79–156% CPU, RSS 0.2–2.7 GiB). The
per-interval polling has emitted several hundred "no change" messages carrying zero information.

TASK 1 — STOP POLLING IMMEDIATELY
Do not poll per-interval. Either block on the atomic receipt files, or check ONCE every 60 minutes.
No liveness checks unless a check has already failed. No narration of unchanged state.

TASK 2 — HARD DEADLINE
Set a single deadline 6 hours from now (~13k unique graph scores per fold at ~1s is several hours).
If all four receipts publish before then, read them once and report. If the deadline passes without
publication, KILL the run and record the wall-clock runtime itself as the finding — an 18,145-config
lattice that cannot complete its scorer pass in a working day is not a viable screening instrument.

TASK 3 — INTERPRET AGAINST THE KNOWN CEILING
When receipts land, compare the winning configuration's exact 44b6 delta against BOTH:
- the +0.005 gate, and
- DAG 4's measured tier-(a) ceiling of +0.002406 (1/38 divisions inferable from current features).
If the winner is below +0.005, park it on evidence AND record explicitly that the pre-measured
ceiling predicted this outcome before the sweep was built. That record is the point of the exercise.

TASK 4 — ENCODE TWO DURABLE RULES IN ALIGNMENT.md
  (a) CEILING BEFORE MACHINERY: no sweep, lattice, or search may be built before its oracle ceiling
      is measured. If the measured ceiling sits below the promotion gate, the sweep is not funded —
      it may run only as an explicitly-labelled diagnostic with a stated wall-clock cap.
  (b) MONITORING: long-running jobs that publish atomic receipts are WAITED ON, not polled. Maximum
      one status check per hour; no emission of unchanged state.
Report both edits; change no scientific gate.
```

### Goal

**Outcome:** Terminalize the R4 nested-OOF run at sealed receipts or the owner deadline, record its decision-grade finding, and encode the two durable campaign rules without changing a scientific gate

**Verification**

- rg -n CEILING reports/g02/screening/tiered_surface/ALIGNMENT.md
- rg -n 25:56:56 reports/g03/screening/LIVE_STATE.md reports/g03/screening/LEDGER.md reports/g03/screening/wave_r/R4_RUNTIME_KILL_RECEIPT_20260803.json
- node /Users/agent/.codex/plugins/cache/agent-local/cruise/0.4.1/scripts/cruise.mjs attest --json

**Touchpoints**

- reports/g02/screening/tiered_surface/ALIGNMENT.md
- reports/g03/screening/LIVE_STATE.md
- reports/g03/screening/LEDGER.md
- reports/g03/screening/wave_r/R4_RUNTIME_KILL_RECEIPT_20260803.json
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- Do not change the +0.005 scientific gate or any other scientific gate
- Do not infer a score, winner, or tier from unsealed transient spools
- Do not restart R4, open held-fit data, use GPU/Kaggle, or launch another scientific node

**Invariants**

- Only atomic R4 receipts and seals are admissible for a scientific result
- No unchanged monitoring state is emitted and no periodic receipt polling is introduced
- The measured tier-(a) ceiling remains +0.002406490 and the promotion gate remains +0.005
- LIVE_STATE and LEDGER tell the same terminal story and root remains their sole writer

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The R4 terminal action and durable rules match the exact owner monitoring, deadline, ceiling, and no-gate-change instructions
  - Evidence: ORIGIN, terminal records, and both durable rules match the exact owner request with no scientific gate change
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: R4 is terminal with either four valid atomic receipts or a recorded deadline/runtime kill finding, never a transient score
  - Evidence: Zero of four atomic receipts existed and the terminal runtime receipt records the fail-closed working-day kill finding
- **core.regression** — passed
  - Dimension: regression
  - Criterion: No scientific gate, tier, held-access marker, GPU state, or existing campaign evidence changed
  - Evidence: Scoped review found no scientific gate, tier, held-access, GPU, Kaggle, submission, or existing evidence mutation
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing receipts fail closed, unsealed spools stay scientifically inadmissible, and no absent process is represented as a scored outcome
  - Evidence: Missing receipts produced no score or winner and transient spool access remained structurally bounded and scientifically quarantined
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: Skeptical final review finds both durable rules present and no unresolved in-scope monitoring, runtime, ceiling, or state-consistency gap
  - Evidence: Final diff and evidence review found the runtime, ceiling, state consistency, and both durable-rule obligations closed
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate skeptical review confirms the terminal interpretation, runtime arithmetic, quarantine boundary, and absence of gate changes
  - Evidence: Separate skeptical reread confirmed runtime arithmetic, absent receipts, no score inference, quarantine wording, and unchanged gates

### Goal detail

# Flight plan

Wait only on atomic receipts and one absolute deadline. If the run is already terminal before the waiter is armed, fail closed immediately: do not revive it, do not infer a score from transient storage, and record the observed wall-clock interval as the operational finding. The known tier-(a) ceiling is contextual governance evidence, not a substitute score. Encode ceiling-first funding and event-driven receipt monitoring as durable rules while leaving every scientific gate unchanged.

## 2026-08-02T17:24:24.532Z — completed: Zero-initialize the G-03 Phase-2 48-to-12 adapter, prove immutable h000 graph parity, audit Branch-B effect sizes, and run h005 only if the allocation gate opens.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-02T17:24:24.532Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
G-03 R1 zero-shot measured 0.763734 vs floor 0.868299 (Branch B). This is BELOW the Trackastra
native score of 0.829194 — the randomly-initialized crop adapter is injecting noise into the 12
wrfeat channels that were previously zero-filled, costing ~0.066 before training begins.

TASK 1 — ZERO-INIT THE ADAPTER OUTPUT (blocking, before any scientific hours)
Change the final projection `Linear(48->12)` to ZERO initialization for BOTH weight and bias. With
tanh(0)=0 the adapter emits exact zeros, so the model starts bit-identical to the established
zero-feature interface and can only improve. Gradients still flow (nonzero input activations), so
trainability is unaffected. Precedent: LoRA B=0 init; ControlNet zero-convolutions. Record the
change in the Phase-2 contract.

TASK 2 — RE-MEASURE h000
Publish a new immutable h000 checkpoint and emit its graph score. VERIFY it now reproduces the
zero-feature baseline within numerical tolerance. If it does not, the discrepancy is a decoding-path
difference, not an init problem — diagnose before proceeding. Reclassify under the allocation policy
with the corrected number.

TASK 3 — EARLY TRAJECTORY-SLOPE READ (cheap insurance)
Do NOT commit 30 T4-h blind. Publish an immutable checkpoint at ~5 T4-h and emit its graph score,
using the same protocol as h000/h030 — this is a FIXED checkpoint measurement, not within-trajectory
selection, so it does not violate the no-proxy-selection rule. Report the slope from h000. If the
trajectory is flat or negative at h005, stop and save 25 T4-h. If it is on track, continue to h030
under the unchanged gate.

TASK 4 — BRANCH-B DOSSIER (owner policy requires it)
G-03 is Branch B, so per the allocation rule it owes an audited, literature-validated case that
<=30 T4-h reaches comparability with 0.881522, plus a realistic path to 0.915 then 0.945. Directional
support is insufficient — the dossier must state EFFECT SIZES. The MIDL dense-embeddings result is
cited as "improved linking"; quantify by how much, on what data, at what training budget. Deliver
this before the 30-hour trajectory is funded.
```

### Goal

**Outcome:** Zero-initialize the G-03 Phase-2 48-to-12 adapter, prove immutable h000 graph parity, audit Branch-B effect sizes, and run h005 only if the allocation gate opens.

**Verification**

- PYTHONPATH=/Users/agent/.cache/uv/archive-v0/AqUrADnwBTTu67kTkt7Dt/lib/python3.13/site-packages .venv313/bin/python -m pytest -q tests/test_g03_phase2_adapter.py
- .venv313/bin/python scripts/validate_g03_phase2.py

**Touchpoints**

- scripts/g03_phase2_adapter.py
- scripts/g03_phase2_h000.py
- scripts/g03_phase2_h000_canonical.py
- scripts/g03_phase2_h000_score_recovery.py
- scripts/validate_g03_phase2.py
- tests/test_g03_phase2_adapter.py
- reports/g03/screening/phase2_zero_init_v1/
- reports/g03/screening/PHASE2_PREPARATION_GATE_20260801.md
- reports/g03/screening/LIVE_STATE.md
- reports/g03/screening/LEDGER.md
- reports/g02/screening/LIVE_STATE.md
- .cruise/ORIGIN.md
- .cruise/FLIGHT.md

**Non-goals**

- No CV57 read, competition submission, final-slot action, or RunPod action.
- No h005 or h030 GPU spend unless the corrected Branch-B allocation gate is satisfied.
- No rescreening of a DEAD mechanism or alteration of the unchanged h030 graph gate.

**Invariants**

- Historical R1 artifacts and the code identity that produced 0.7637338804169099 remain immutable.
- The Phase-2 final projection is exactly Linear(48,12), with weight and bias exactly zero at h000 and tanh output exactly zero.
- Adapter hidden activations and projection gradients remain nonzero on a deterministic nonconstant crop batch.
- h000, h005, and h030 are fixed cumulative-hour checkpoints; no within-trajectory checkpoint selection occurs.
- Every external action retains the shared serial lock, quota reconciliation, hard runtime kill, exact version accounting, and atomic receipt rules.
- The population, compiler, scorer, and model identities are identical across h000, h005, and h030 graph reads.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: All four requested tasks are either completed or truthfully stopped by the binding Branch-B gate.
  - Evidence: All four tasks closed: zero-init and h000 completed; h005/h030 truthfully stopped by the failed Branch-B funding gate.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: h000 emits exact-zero wrfeat, reproduces the same-protocol zero-feature graph, and h005 obeys the preregistered stop rule if funded.
  - Evidence: Five focused tests, checkpoint validation, 36 graph-pair rehashes, and frozen scoring establish exact h000 parity and fail-closed h005.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Historical R1 validation and unaffected G-03 controls remain unchanged.
  - Evidence: Historical R1 contract and source retain their sealed hashes and score; focused regression tests pass 5/5.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Nonzero h000 output, graph mismatch, artifact overwrite, dossier failure, or flat h005 trajectory fail closed.
  - Evidence: Negative tests reject overwrite and reference drift; runtime mismatch paused scoring; dossier FAIL held GPU allocation at zero.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff-and-evidence review finds no unresolved in-scope gap or overstated scientific claim.
  - Evidence: Fresh final review reconciled code, immutable hashes, corrected score arithmetic, primary-source effects, allocation, ledger, and both live-state files.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: An independent skeptical review finds no gate bypass, data leakage, numerical mismatch, or unsupported dossier claim.
  - Evidence: A separate root validator and fresh frozen-scorer replay found no bypass, leakage, numerical claim drift, or unsupported funding claim.
- **cxx-hpc.complexity** — passed
  - Dimension: complexity
  - Criterion: Adapter and checkpoint paths state and respect their per-batch time and auxiliary-space cost.
  - Evidence: Fixed-shape O(N) batch complexity, activation sizes, MAC counts, and single-thread CPU batch timings are recorded in the resource note.
- **cxx-hpc.memory** — passed
  - Dimension: memory
  - Criterion: Peak accelerator memory, checkpoint bytes, swap growth, and crop-batch bounds stay within the frozen resource contract.
  - Evidence: Activation bounds and 101075-byte checkpoint are recorded; the failed funding gate prevented any unsupported accelerator-memory claim or spend.
- **cxx-hpc.numerical** — passed
  - Dimension: numerical-correctness
  - Criterion: Projection tensors and h000 output are bitwise zero; parity tolerance and deterministic hashes are explicit.
  - Evidence: Projection tensors and output are bitwise zero, gradients are finite/nonzero, 36 graphs match bytewise, and score parity error is exactly zero.
- **cxx-hpc.performance** — passed
  - Dimension: performance-measurement
  - Criterion: The T4 throughput/cost model is measured before funding and h005 actual debit is reconciled.
  - Evidence: Branch-B failed before funding, so T4 calibration remains a mandatory future precondition; h005 debit reconciles to exactly 0.0 and no launch occurred.
- **cxx-hpc.portability** — passed
  - Dimension: toolchain-portability
  - Criterion: Pinned Python, PyTorch, CUDA/T4, dtype, and deterministic-algorithm assumptions are recorded and validated.
  - Evidence: Pinned local versions and the explicit CPU/CUDA threshold diagnosis led to a sealed same-runtime CUDA-bank decode without a new GPU action.

### Goal detail

# Flight plan

The first ready micro-goal is a versioned Phase-2 adapter and h000 checkpoint
publisher that does not mutate historical R1 evidence. The corrected h000 graph
must be compared on the exact same population and decode/scorer path as its
zero-feature control; the separately reported CV57 score is descriptive only
unless the population is explicitly identical.

The Branch-B dossier is a funding gate, not advocacy. It must report source-native
effect sizes, datasets, controls, variance where available, and disclosed training
budgets. If the evidence still fails the owner policy, h005/h030 remain unfunded and
the task stops truthfully at that policy barrier. If it passes, h005 is fixed before
training, published immutably at about five cumulative T4-hours, scored once, and
continued only under the preregistered slope rule and unchanged h030 gate.

## 2026-08-02T23:17:58.076Z — blocked: Publish the fixed held-fit36 h005 slope read and complete truthful ceiling-first standalone screens for dense motion and trained fork evidence without crossing any conditional funding gate.

- **Status:** blocked
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-02T23:17:58.076Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
Owner approved: h005 slope read for G-03 (5 T4-h), AND parallel exploration of models drawing on
ORTHOGONAL EVIDENCE TYPES. Hybridisation is deferred until a candidate reaches >=0.900 — keep every
lane standalone and scored on its own. ~25 T4-h remain this week after h005; ~255 through Sept 29.
ALIGNMENT's CEILING BEFORE MACHINERY rule is binding: measure before building, and do not fund a
sweep or training run whose measured ceiling sits below the gate.

LANE 0 — G-03 h005 (funded, 5 T4-h)
Publish an immutable checkpoint at ~5 T4-h from the zero-init h000 (0.763088, parity error 0) and
emit its graph score on the SAME held-fit36 population. Report the slope. Fund h030 only if the
trajectory projects to closing ~0.10 on that population; otherwise stop with 25 T4-h preserved.
NOTE: h000 is held-fit36 and the campaign best 0.881522 is CV57 — never subtract across populations.
Every gate reference must state its population explicitly (APNCC on held-fit36 is 0.866535).

LANE 1 — DENSE MOTION FIELD (new evidence type)
Step 1, MEASURE FIRST (CPU): derive dense deformable displacement fields between consecutive
volumes (classical 3-D registration is sufficient for the screen) and turn them into per-candidate
motion-consistency features. Run them through the EXISTING frozen separation gate — sealed B0, then
A1, plus the within-frame permuted control A0 — on the frozen 44b6 population. This is a distinct
evidence type from A7's rigid global phase correlation; state that distinction explicitly in the
preregistration.
Step 2, CONDITIONAL: fund a compact learned displacement predictor (VoxelMorph-class, in-domain)
ONLY if Step 1 clears its class requirement. Cite primary sources and state transfer domain
honestly (Linajea is whole-embryo lineage reconstruction on zebrafish — same domain, different
detection interface).

LANE 2 — TRAINED FORK DETECTOR (the zero-credit term)
DIV-F2 is blocked on AUTHENTIC_PRETRAINED_FORK_INPUT, which has never been supplied. The 0/38
realizer failure used hand-crafted decode rules, NOT a trained detector — say so explicitly so this
is not treated as re-screening a falsified mechanism.
Step 1, MEASURE FIRST: acquire or train a mitosis/fork detector operating on raw 3-D+t intensity
crops. Evaluate it directly against the 38 known present-but-fragmented division events plus a
matched negative set: report precision/recall BEFORE any graph integration.
Step 2, CONDITIONAL: integrate only if precision at the required operating point clears the banked
FP sensitivity (TP=25 gives +0.0595 at 0 FP, +0.0532 at 5 FP, +0.0481 at 10 FP). This term is
ADDITIVE to the edge score — it does not need to beat 0.881522, only to add on top of it.

CONSTRAINTS
Standalone scoring only; no hybridisation until a candidate reaches >=0.900. Each lane preregisters
its ceiling and kill line before building. Any from-scratch training must validate cross-family from
the start — every from-scratch model in this campaign passed 6bba and failed 44b6. Serial GPU
authorization, <5h auto with hard kill, weekly cap enforced, RunPod forbidden, submission human-gated.
```

### Goal

**Outcome:** Publish the fixed held-fit36 h005 slope read and complete truthful ceiling-first standalone screens for dense motion and trained fork evidence without crossing any conditional funding gate.

**Verification**

- python3 -m pytest tests/test_g03_phase2_adapter.py tests/test_g03_phase2_campaign_lanes.py -q
- python3 scripts/validate_g03_phase2.py
- python3 scripts/validate_g03_phase2_campaign_lanes.py

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- reports/g03/screening/**
- reports/g02/screening/LIVE_STATE.md
- reports/g02/screening/tiered_surface/branches/g03_orthogonal_lanes_20260803/**
- reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json
- kaggle_kernel/g03-h005-*/**
- scripts/g03_phase2_*.py
- scripts/g03_dense_motion_*.py
- scripts/g03_fork_detector_*.py
- scripts/validate_g03_phase2_campaign_lanes.py
- tests/test_g03_phase2_*.py
- tests/test_g03_dense_motion_*.py
- tests/test_g03_fork_detector_*.py

**Non-goals**

- No hybridization of lanes before a standalone candidate scores at least 0.900 on its declared population.
- No h030 funding unless the fixed held-fit36 h005 slope linearly projects to the held-fit36 APNCC score 0.8665345260361912 by 30 T4-hours.
- No learned dense-displacement model unless the classical dense-field supplier passes its frozen 44b6 class gate.
- No graph integration of the fork term unless its direct fixed-population operating point clears the banked TP and FP value line.
- No CV57 access for Lane 0 and no cross-population subtraction.
- No RunPod action and no competition submission.

**Invariants**

- H005 starts from immutable zero-init h000 SHA-256 603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95 and is measured once at cumulative 4.75 to 5.25 T4-hours.
- H000 and h005 graph scoring use the same 36 held-fit36 identities, association bank, compiler, and scorer; every reported numerical gate names its population.
- Dense motion uses the existing frozen 44b6 population, folds, sealed B0, A1 consumer, and within-frame permuted A0 control without rebaselining.
- Dense deformable registration is recorded as distinct from A7 rigid global phase correlation and from any DEAD hand-tuned motion mechanism.
- Lane 2 evaluates a trained or authentically pretrained raw 3-D+t detector on the frozen 38 present-but-fragmented events plus a sealed matched negative set before graph access.
- The prior 0 of 38 result is described only as a hand-crafted decode-rule failure, not a trained detector falsification.
- Any from-scratch model has cross-family validation built in from the first training run.
- All GPU work is serialized, authoritative quota-checked, automatically hard-killed below five hours per run, and remains inside the weekly cap.

**Quality gates**

- **core.intent** — failed
  - Dimension: intent
  - Criterion: Deliver exactly the owner-amended three-lane ceiling-first campaign with conditional stops and population labels intact.
  - Evidence: The two ceiling-first CPU lanes are terminal, but the requested immutable h005 and slope do not exist after the sole repaired run failed before training.
- **core.functional** — failed
  - Dimension: functional-correctness
  - Criterion: Fixed-checkpoint scoring and both standalone screen evaluators reproduce their declared metrics from immutable inputs.
  - Evidence: Lane 1 and Lane 2 evaluators reproduce, while fixed-checkpoint Lane 0 scoring is impossible because no checkpoint was produced.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Zero-init parity, frozen scorer contracts, existing separation gate identity, and prior immutable decisions remain unchanged.
  - Evidence: Zero-init h000 validation, preregistration hashes, frozen consumer identity, and prior decisions remain exact.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing assets, population drift, partial outputs, nonfinite scores, runtime overflow, quota conflicts, and conditional-gate failures stop closed.
  - Evidence: Both remote failures stopped before scientific work, missing outputs are rejected, and no evaluation or retry opened.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff and evidence review finds no unmeasured claim, cross-population arithmetic, forbidden hybrid, or unauthorized conditional spend.
  - Evidence: Fresh terminal review found no invented score, cross-population arithmetic, forbidden hybrid, or unauthorized external action.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A genuinely separate adversarial review pass finds no unresolved scientific-governance or implementation defect.
  - Evidence: A separate raw-artifact adversarial pass verified the empty V2 inventory, absent h005 outputs, three allowed pushes, and zero submissions.
- **hpc.complexity** — passed
  - Dimension: complexity
  - Criterion: Dense-field extraction and crop-detector evaluation state and obey bounded time and auxiliary-space costs on the frozen population.
  - Evidence: Dense motion completed 4059 streamed pairs and 1391458 actions; the fork probe and both remote failure paths stayed bounded.
- **hpc.memory** — passed
  - Dimension: memory
  - Criterion: Volume streaming, field storage, batch sizing, and checkpoint writes remain within measured CPU RAM, disk, swap, and T4 VRAM limits.
  - Evidence: Dense motion peaked at 229.94 MiB RSS, fork fitting remained CPU-bounded, and V2 failed before CUDA allocation.
- **hpc.numerical** — passed
  - Dimension: numerical-correctness
  - Criterion: Registration coordinates, physical units, interpolation, normalization, score tolerances, slope arithmetic, and permutations are deterministic and numerically valid.
  - Evidence: Synthetic transform tests, exact hashes, fixed fold metrics, and null h005 slope semantics all validate without nonfinite values.
- **hpc.performance** — passed
  - Dimension: performance-measurement
  - Criterion: Representative release-path timings and throughput establish that each authorized run fits its declared CPU or GPU hard limit without a sweep.
  - Evidence: Lane 1 completed in 6386.171 CPU seconds, the runtime probe in 9.223 seconds, and both GPU failures terminated within their hard caps.
- **hpc.portability** — failed
  - Dimension: toolchain-portability
  - Criterion: Required Python, PyTorch, registration, scorer, CPU, and T4 assumptions are explicit and validated on the actual execution paths.
  - Evidence: The single-source Kaggle CPython 3.12 runtime probe passed, but the exact four-source GPU worker could not authenticate the same runtime manifest.
- **domain.ceiling-first** — passed
  - Dimension: ceiling-first-allocation
  - Criterion: Every machinery or training action is preceded by a measured ceiling at or above its population-specific gate.
  - Evidence: Both orthogonal machinery gates were measured first, and V2 was preceded by a CPU release-path gate plus static and quota guards.
- **domain.population** — passed
  - Dimension: population-integrity
  - Criterion: All scores, slopes, controls, and gates are attached to immutable held-fit36, frozen 44b6, or frozen division identities with no population mixing.
  - Evidence: Lane 1 stayed on frozen 44b6, Lane 2 trained on 6bba and validated on 44b6, and Lane 0 never accessed held-fit36 or CV57.
- **domain.external** — passed
  - Dimension: external-compute-governance
  - Criterion: Scientific GPU execution is productive, serial, quota-reconciled, version-accounted, hard-killed, and within owner authorization.
  - Evidence: All external actions were version-accounted, serially guarded, quota-reconciled, hard-capped, and ended with no submission or RunPod use.

### Goal detail

# Flight plan

The first micro-goal seals an owner amendment that supersedes the earlier Branch-B h005 non-funding decision without rewriting that immutable history. Lane 0 uses a same-population linear projection: with h000 `0.7630876414338513`, held-fit36 APNCC `0.8665345260361912`, and actual checkpoint hours `h`, h030 is fundable only when `h000 + 30 * (h005 - h000) / h >= 0.8665345260361912`; at exactly five hours this requires h005 delta at least `0.0172411474337233` and slope at least `0.00344822948674466` adjusted graph score per T4-hour.

Lane 1 may spend CPU only on a preregistered classical dense-deformable displacement supplier and the unchanged frozen B0/A1/A0 consumer. A learned VoxelMorph-class predictor remains outside the ready scope until a class-level PASS exists. Lane 2 first seals the 38 positives and matched negatives and directly measures a trained raw-intensity detector; graph integration remains outside the ready scope until a banked operating point passes. Literature claims must use primary sources and state domains, interfaces, budgets, and effect sizes honestly.

No task may convert an infrastructure failure into a scientific verdict. Partial or unauthenticated outputs remain invalid, and any failed conditional gate terminates that lane while preserving compute.

Concrete terminal blocker: the sole dependency-repaired h005 V2 failed its exact four-source remote-mount manifest check before training, produced no checkpoint or score, and the frozen repair contract authorizes no V3. Publishing the requested h005 and computing its held-fit36 slope now require explicit owner supersession plus a new exact-four-source preflight contract; the current run cannot complete that outcome within its authority.

## 2026-08-03T09:06:17.389Z — completed: Build and verify a local self-contained CPU visualiser for the banked APNCC CV57 tracks, delivering all 57 score-sorted scrubbable MP4s with the top-10 residual videos available first.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-03T09:06:17.389Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
PROMPT 4 — TRACK VISUALISER FOR HUMAN INSPECTION (parallel, CPU)

CONTEXT
Competitors are visually inspecting how tracks form over time; we never have. The owner will watch the
output and report observations directly. Our residual is concentrated (top-10 videos = 52.3%), so
prioritise the worst videos.

BUILD
A local, self-contained visualiser:
1. Per frame, render max-intensity projections (XY primary; XZ/YZ optional) at native resolution.
2. Overlay predicted tracks coloured by track ID, with a short motion tail showing recent history.
3. Overlay GT annotations where present (only ~2.8% of nuclei are annotated — mark unannotated
   explicitly so absence of GT is never read as a false positive).
4. Highlight error classes in distinct colours: FP edges, FN edges, track fragmentation (breaks),
   missed divisions, false divisions.
5. Emit one scrubbable animation per video (MP4 or GIF) plus an HTML index listing all videos SORTED BY
   ERROR CONTRIBUTION, so the worst offenders are watched first. Show per-video score, edge TP/FP/FN,
   and division TP/FP/FN in the index.
6. Flag frames detected as frozen or as global jumps (Prompt 3) directly on the timeline, so visual
   discontinuities are not misread as tracking failures.
Deliver the index plus the top-10 worst videos first so the owner can begin inspecting immediately.
```

### Goal

**Outcome:** Build and verify a local self-contained CPU visualiser for the banked APNCC CV57 tracks, delivering all 57 score-sorted scrubbable MP4s with the top-10 residual videos available first.

**Verification**

- .venv313/bin/python -m pytest -q tests/test_track_visualizer.py tests/test_holdout_fragmentation.py tests/test_official_scorer_parity.py
- .venv/bin/python scripts/build_track_visualizer.py validate --bundle reports/g04/track_visualizer_v1 --min-rendered 57 --ffprobe /opt/homebrew/bin/ffprobe

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- scripts/build_track_visualizer.py
- tests/test_track_visualizer.py
- reports/g04/track_visualizer_v1/**
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- No change to predictions, scores, scorer contracts, campaign gates, tiers, DEAD mechanisms, Kaggle state, or submissions.
- No GT-dependent model selection, fitting, or candidate promotion; this is human-inspection output only.
- No extraction or duplication of the 81 GB raw competition ZIP.
- No GPU, Kaggle kernel, RunPod, or network action.
- XZ and YZ projections are optional and are not required for the top-10-first delivery.

**Invariants**

- Input predictions are the banked APNCC CV57 graph fragments and are never mutated.
- Video ordering is descending residual error contribution with deterministic identity tie-breaking; the registered top-10 sum to the banked 52.325845 percent residual share.
- XY projections retain the raw 256 by 256 pixel field at native resolution before any browser scaling.
- Sparse GEFF annotation coverage is explicit in every animation and in the index; an unannotated prediction is never labelled as a false positive merely because GT is absent.
- Error overlays derive from frame-local 7 micrometer matching with physical voxel scale and retain separate semantics for FP edge, FN edge, fragmentation break, missed division, and false division.
- Per-video summary metrics are official scorer metrics, not counts inferred from pixels or visual overlays.
- Prompt 3 discontinuity flags are consumed when available; unavailable flags are disclosed rather than fabricated or silently treated as all-clear.
- Every emitted MP4 is locally referenced by a relative path from the self-contained HTML bundle and is seekable in an HTML5 video element.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The delivered bundle matches all six requested visual inspection capabilities and contains the top-10 worst videos first.
  - Evidence: Complete bundle contains 57 residual-sorted MP4s and all requested overlays; the registered top ten were rendered and shared first.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Matching, sparse-label semantics, track tails, five error classes, official per-video metrics, MP4s, and sorted index work on synthetic and real CV57 inputs.
  - Evidence: 25 tests plus 26 subtests passed; strict validation probed and hash-checked all 57 videos; representative rendered frames were visually inspected.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing banked predictions, score artifacts, and campaign state remain unmodified except for one diagnostic live-state log entry.
  - Evidence: HEAD remained c493a676; only the new visualizer/tests/bundle and diagnostic LIVE_STATE entry changed, while all 57 input fragment hashes reconciled.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing or malformed ZIP, fragments, GT, ffmpeg, flags, and incomplete bundle outputs fail clearly without converting unknown annotation state into an error label.
  - Evidence: Regression tests reject malformed and out-of-range flags, stale contracts, metric mismatch, incomplete bundles, zero-edge corner cases, and same-size media tampering.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff and generated-output review finds no missing requested overlay, misleading sparse-GT claim, broken video, or unsorted index.
  - Evidence: Independent adversarial review found no bundle misclassification; its four P2 and one P3 implementation findings were fixed, regression-tested, re-rendered, and re-reviewed.
- **web.accessibility** — passed
  - Dimension: accessibility
  - Criterion: Index structure, controls, labels, legend, contrast, keyboard use, focus, and reduced-motion behavior support human inspection.
  - Evidence: Playwright confirmed semantic labels, 57 native controls, keyboard link activation and video focus, reduced-motion media behavior, and explicit sparse-GT language.
- **web.responsive** — passed
  - Dimension: responsive-ui
  - Criterion: Video cards, metrics, legends, and timeline markers remain usable without accidental horizontal clipping at representative narrow and wide viewports.
  - Evidence: Playwright measured no page overflow at 1440x1000 or 390x844; the narrow table scrolls intentionally and the 337 px video remains inside its 367 px card.
- **web.runtime** — passed
  - Dimension: browser-runtime
  - Criterion: All top-10 videos load through local HTTP, scrub, and show synchronized frame and discontinuity state without console errors.
  - Evidence: Range-capable localhost Playwright run loaded 57/57 media, scrubbed every top-ten video to 2 s/frame 14 with synchronized 14% playheads, and reported zero console errors.
- **domain.provenance** — passed
  - Dimension: artifact-provenance
  - Criterion: Manifest records exact raw ZIP, prediction fragment set, residual ranking, scorer implementation, and optional Prompt 3 flag provenance.
  - Evidence: Observed 87 GB raw SHA-256 matched the registered archive; manifest pins its receipt plus ranking, score, Prompt-3 state, and every prediction fragment hash.
- **domain.sparse-gt** — passed
  - Dimension: sparse-annotation-clarity
  - Criterion: Visual and textual language makes annotated, matched, and unannotated states distinguishable in every video.
  - Evidence: Tests and independent review confirm only official pred_valid unmatched edges are FP; gray-ringed unknown predictions remain neutral and every frame carries the sparse-GT warning.

### Goal detail

# Flight plan

The first ready micro-goal resolves all input contracts and freezes a deterministic ranking from the existing residual audit. The second implements pure-Python/Pillow overlays and streams raw Zarr frames from the existing ZIP into ffmpeg, with process-level parallelism by video and no raw extraction. Matching uses physical scale and sparse-annotation-aware edge validity. Official scorer rows remain the source of index metrics, while overlay classifications are event-local visual explanations.

The initial delivery renders only the registered ten largest-residual videos, writes an all-57 metadata index with unavailable animations clearly marked, and places those ten first. A later invocation of the same CLI can fill the remaining videos without changing ranking or semantics. Prompt 3 flags are an optional input contract: if no compatible artifact exists at render time, the index and timeline disclose that discontinuity flags are unavailable.

## 2026-08-03T10:34:36.366Z — completed: Implement and verify a default post-scoring visual-review hook with the exact curated four-category selection, 15-second clip pacing by default, fully CLI-configurable HUD/speed, content-hash caching, a refreshed latest index, and a manual uncapped full-catalog scope.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-03T10:34:36.366Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
PROMPT 3 (REVISED) — DEFAULT POST-SCORING VISUAL REVIEW, CLI-CONFIGURABLE HUD + SPEED

CONTEXT
Supersedes the prior slowdown-to-3-minutes instruction — that was a wrong resolution of an
ambiguity. Correct spec: SPEED UP, not down. 100 frames should play back in ~15s by default so an
entire video is skimmable in seconds and a full-catalog review (all scored videos) is feasible in
one sitting. ALL playback/HUD behavior must be CLI flags, not hardcoded choices.

SELECTION (unchanged from before)
Default hook remains the 4-category curated set: top-3/bottom-3 edge adjusted Jaccard, top-3
highest division Jaccard (GT_divisions>0), and bottom-3 by "highest GT division count with ~0
achieved division credit" (the missed-opportunity substitute for the degenerate "lowest division
Jaccard" ranking). Deduplicate videos across categories; tag each clip with every category it
qualifies for.

PLAYBACK SPEED
Implement `--clip-seconds` (default 15) as the primary control: encoded fps = num_frames /
clip_seconds. This is symmetric — small values speed up, large values (up to native ~1 fps at
clip_seconds=num_frames) slow down for detailed frame-by-frame study. Provide `--fps` as a direct
override for users who want exact frame rate instead of a target duration. Do NOT hardcode a
specific interpretation of "3 minutes" — this replaces that entirely.

HUD OVERLAY (CLI-CONFIGURABLE)
Build a metrics registry (edge_adjusted_jaccard, edge_raw_jaccard, edge_tp_fp_fn, division_jaccard,
division_tp_fp_fn, gt_division_count, node_recall, combined_score, frame_index, frozen_flag,
jump_flag, video_id, family). Render selected metrics as: semi-transparent rounded background panel
(so it never fully obscures frame content) + bold, outlined/drop-shadowed text in a fixed
high-contrast color per metric type, so numbers stay legible against any content. Expose:
  --hud / --no-hud            toggle overlay entirely
  --hud-corner {tl,tr,bl,br}  default tr
  --metrics LIST              default: edge_adjusted_jaccard,division_jaccard,gt_division_count
  --metric-corner MAP         optional per-metric corner override for splitting groups
Aggregate (whole-video) metrics may render as a constant badge; if cheap to compute, ALSO offer a
running/cumulative variant (e.g. FP/FN accumulated up to the current frame) as an additional
registry entry, since it shows error accumulation over time rather than only the final number --
gate this behind a metric name like `edge_fp_fn_cumulative`, not a separate mode.

SCOPE
`--scope default` (unchanged: curated ~12-clip hook, <1GB, fires automatically after every
scoring event) vs `--scope all` (full-catalog on-demand mode, same pacing/HUD flags apply,
NOT triggered automatically — the owner runs this manually when they want a full review session).
Re-assert the <1GB budget for `--scope default`; `--scope all` reports its own total size but is
not budget-capped the same way since it's explicitly on-demand.

INVOCATION
Unchanged: hook into the scoring harness so `--scope default` fires automatically on every
submission candidate, training checkpoint eval, or snapshot; cached by input hash; refreshes one
"latest" index.
```

### Goal

**Outcome:** Implement and verify a default post-scoring visual-review hook with the exact curated four-category selection, 15-second clip pacing by default, fully CLI-configurable HUD/speed, content-hash caching, a refreshed latest index, and a manual uncapped full-catalog scope.

**Verification**

- .venv313/bin/python -m pytest -q tests/test_track_visualizer.py tests/test_scoreall.py tests/test_holdout_fragmentation.py tests/test_official_scorer_parity.py
- .venv/bin/python scripts/build_track_visualizer.py validate --bundle reports/g04/post_scoring_visual_review/latest --min-rendered 1 --ffprobe /opt/homebrew/bin/ffprobe

**Touchpoints**

- .cruise/ORIGIN.md
- .cruise/FLIGHT.md
- scripts/build_track_visualizer.py
- scripts/scoreall.py
- tests/test_track_visualizer.py
- tests/test_scoreall.py
- reports/g04/post_scoring_visual_review/**
- reports/g02/screening/LIVE_STATE.md

**Non-goals**

- No change to official scorer math, prediction graphs, model weights, campaign gates, Kaggle state, submissions, or external services.
- No automatic full-catalog render; scope all remains an explicit owner CLI action.
- No GPU use, raw ZIP extraction, or duplicate image corpus.
- No removal of sparse-GT/error overlays or reinterpretation of unannotated predictions as false positives.

**Invariants**

- scoreall writes the authoritative score artifact before invoking visual review, and review failure never corrupts or rewrites scorer metrics.
- Scope default uses exactly four deterministic top/bottom-three category rules, deduplicates identities, and preserves every qualifying category tag.
- Scope default remains below 1000000000 bytes and garbage-collects stale latest-bundle media; scope all only reports size.
- clip-seconds defaults to 15 and yields frames divided by clip-seconds FPS unless an explicit positive fps override is supplied.
- All HUD behavior is represented in CLI/config and the render-content hash; no selected metric or corner is silently hardcoded.
- Unknown metrics, malformed corner maps, invalid durations, stale media, and malformed score/flag inputs fail clearly.
- Prompt-3 data-pathology transitions map source frame t to target frame t+1; missing detector data remains unavailable rather than clear.
- The existing official FP/FN/division localization and sparse-GT semantics remain unchanged.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: The implementation matches the revised speed, HUD, selection, scope, cache, latest-index, and hook requirements without retaining the superseded slowdown interpretation.
  - Evidence: Reconciled ORIGIN, CLI help, generated manifest/index, and scoreall; all revised requirements are represented and no superseded three-minute interpretation remains.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Default and all selection, effective FPS, HUD registry/corners, cumulative errors, real flags, cache hits, generated media, and hook execution work on synthetic and APNCC score inputs.
  - Evidence: All 47 pytest cases and 27 scorer subtests passed; the real nine-clip bundle rendered, validated, and played correctly in browser checks.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Official scorer outputs and existing track-overlay semantics remain unchanged, and relevant scorer/visualizer tests pass.
  - Evidence: Scoreall, official-scorer parity, fragmentation, and visualizer suites all passed with scorer math and existing localization semantics preserved.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Bad metrics/maps/speed/scope, missing predictions/flags, budget overflow, stale cache, hook failure, and incomplete media have deliberate fail-closed behavior.
  - Evidence: Focused tests cover invalid metrics, maps, speed, scope, census inputs, budget overflow, stale hashes, hook failure, and tampered video validation.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final diff and artifact review finds no unresolved revised-Prompt-3 requirement or misleading metric/selection claim.
  - Evidence: Final skeptical review of code, CLI, tests, rendered artifacts, manifests, and validation receipts found no unresolved revised-Prompt-3 requirement.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate adversarial pass attempts to break selection ties, cache identity, hook lifecycle, duration math, flag alignment, and HUD layout, with credible findings resolved.
  - Evidence: An isolated adversarial pass found and resolved marker-frame rounding and prefixed aggregate-graph materialization issues, followed by full reverification.
- **web.accessibility** — passed
  - Dimension: accessibility
  - Criterion: Latest index and media controls retain semantic labels, keyboard operation, focus, contrast, and reduced-motion behavior with configurable HUD.
  - Evidence: Semantic controls, ARIA labels, visible focus styling, contrast, native keyboard-operable video controls, and reduced-motion CSS were reviewed in the generated page and browser.
- **web.responsive** — passed
  - Dimension: responsive-ui
  - Criterion: Curated and all-scope cards, category tags, metrics, videos, and timelines remain usable at narrow and wide viewports without accidental page overflow.
  - Evidence: At a 390-pixel viewport the document stayed within 390 pixels, the main/card widths fit, and the metrics table scrolled internally; wide layout also rendered correctly.
- **web.runtime** — passed
  - Dimension: browser-runtime
  - Criterion: Generated default media loads, scrubs, reports configured duration/FPS, synchronizes playheads/flags, and emits no console errors.
  - Evidence: Range-capable browser checks loaded all media without errors, measured 15-second duration and seekability, and synchronized a seek to frame 5 with the frame readout.
- **domain.selection** — passed
  - Dimension: curated-selection
  - Criterion: Four category definitions, three-per-category limits, GT-division eligibility, missed-opportunity ordering, deduplication, and multi-tagging are exact and deterministic.
  - Evidence: Tie-heavy synthetic selection tests and the real manifest confirm exact category limits, eligibility, ordering, deduplication, and retained multi-tags.
- **domain.pacing-hud** — passed
  - Dimension: pacing-and-hud
  - Criterion: clip-seconds/fps precedence and every registered metric/corner control are explicit, hash-bound, and accurately rendered, including edge_fp_fn_cumulative.
  - Evidence: Registry, per-metric corners, cumulative values, and hash binding passed tests; every real clip probes as 100 frames at 20/3 FPS and 15 seconds, and a rendered HUD frame was inspected.
- **domain.hook-cache** — passed
  - Dimension: scoring-hook-and-cache
  - Criterion: scoreall automatically invokes only scope default after output write, unchanged inputs reuse media, changed inputs/settings invalidate reuse, and latest refresh remains coherent.
  - Evidence: Hook lifecycle tests passed and an identical real rerun reported nine of nine cache hits with zero clips rerendered while refreshing the coherent latest index.
- **domain.budget** — passed
  - Dimension: artifact-budget
  - Criterion: Default scope reports and enforces total bundle bytes below 1GB while all scope reports bytes without applying that cap.
  - Evidence: Manifest and filesystem accounting agree at 16558395 bytes with default budget_pass true; tests prove strict default rejection and uncapped all-scope reporting.

### Goal detail

# Flight plan

First normalize the score-row contract and implement the deterministic curated selector plus CLI parsing. Then make pacing/HUD/flag state part of the per-video render contract, generalize manifests and validation to default/all scopes, and preserve sparse-GT overlays. Finally attach the default-only hook to scoreall after score output publication, prove content-hash reuse and latest-bundle budget behavior, render the real default bundle, and close browser/review gates.

## 2026-08-03T15:10:13.366Z — completed: Audit every G04 submission-path kernel and builder for fixed runtime identities, repair any additional hardcoded identity assumptions with runtime discovery, and verify the repaired pipeline terminal Kaggle blind-rerun canary before any future score-bearing candidate relies on it.

- **Status:** completed
- **Provider:** codex
- **Started:** Not started
- **Ended:** 2026-08-03T15:10:13.366Z
- **Session:** direct-host
- **Trace:** unavailable
- **Reason:** None

### Original request

```text
CONTEXT
The hardcoded-identity/dynamic-discovery diagnosis for the g04-division-permissive scoring error is
high-confidence but ARTIFACT-BASED, not confirmed via a recovered hidden traceback. Verify before
trusting it repo-wide.

TASKS
1. Grep every submission-path kernel/script (not just the one that errored) for the same
   hardcoded-identity anti-pattern -- anything that assumes a fixed set of video/sample IDs rather
   than discovering whatever is actually mounted at kernel runtime.
2. Fix any additional instances found using the same dynamic-discovery pattern already applied.
3. Run ONE cheap canary submission with the repaired pipeline to confirm it survives Kaggle's actual
   blind rerun end-to-end, before relying on it for the next real (score-bearing) submission.
4. Report pass/fail plainly -- if the canary also errors, the diagnosis needs revisiting, not another
   patch-and-hope cycle.
```

### Goal

**Outcome:** Audit every G04 submission-path kernel and builder for fixed runtime identities, repair any additional hardcoded identity assumptions with runtime discovery, and verify the repaired pipeline terminal Kaggle blind-rerun canary before any future score-bearing candidate relies on it.

**Verification**

- pytest -q tests/test_g04_division_hidden_submit.py tests/test_g04_division_permissive_sweep.py tests/test_g04_division_hidden_guard.py tests/test_kaggle_submission_lifecycle.py
- python3 scripts/g04_division_hidden_guard.py --help
- git diff --check
- The single terminal code-output submission receipt and public calibration show the repaired pipeline survived the Kaggle blind rerun end to end.

**Touchpoints**

- kaggle_kernel/g04-division-permissive-submit-v1/**
- kaggle_kernel/g04divcap5-v1/**
- kaggle_kernel/g04-division-permissive-cap5-v1/**
- scripts/g04_build_division_hidden_submit.py
- scripts/g04_hidden_kaggle_code_submit.py
- scripts/g04_division_hidden_guard.py
- scripts/g04_division_hidden_postprocess.py
- scripts/g04_division_permissive_sweep.py
- scripts/kaggle_submission_lifecycle.py
- tests/test_g04_division_hidden_submit.py
- tests/test_g04_division_hidden_guard.py
- tests/test_g04_division_permissive_sweep.py
- tests/test_kaggle_submission_lifecycle.py
- reports/g02/screening/LIVE_STATE.md
- reports/g04/**

**Non-goals**

- No additional competition submission; the existing terminal code-output canary is the one permitted probe.
- No duplicate canary or patch-and-retry cycle if the canary evidence were to show an error.
- No model, threshold, scorer, or scientific-candidate change.
- No RunPod action, re-screening of a DEAD mechanism, or interference with V32 continuation work.

**Invariants**

- Every runtime test-data identity is discovered from the mounted Kaggle dataset, never assumed from a local/public manifest.
- The canary is bounded, has a hard runtime kill, and is bound to the shared serial/quota guard and an exact code-output version.
- Exactly one terminal remote canary is accepted; the historical exact Version 1 receipt is used rather than launching an identical duplicate.
- A canary failure terminates the diagnosis as unconfirmed; no follow-up patch or launch occurs in this Flight.
- Existing unrelated worktree changes are preserved.

**Quality gates**

- **core.intent** — passed
  - Dimension: intent
  - Criterion: All changes and the one remote launch are confined to the requested G04 identity audit and canary verification.
  - Evidence: Audit report and LIVE_STATE record only the requested identity verification and existing canary evidence.
- **core.functional** — passed
  - Dimension: functional-correctness
  - Criterion: Every submission-path runtime identity source is dynamically discovered and the canary terminalizes with valid output.
  - Evidence: 21 focused dynamic-discovery, guard, sweep, and lifecycle tests passed; ref 55203439 is terminal COMPLETE at public 0.881.
- **core.regression** — passed
  - Dimension: regression
  - Criterion: Existing G04 builder, guard, and submission lifecycle contracts continue to pass.
  - Evidence: Focused 21-test suite passed and the scoped diff check reported no whitespace error.
- **core.failure-paths** — passed
  - Dimension: failure-paths
  - Criterion: Missing or mismatched mounted identities fail closed before output/submission; an errored canary prevents retry.
  - Evidence: Focused tests cover unseen identities, mounted inventory mismatch rejection, output drift rejection, exact-version binding, and one-shot consumption.
- **core.completion-review** — passed
  - Dimension: completion-review
  - Criterion: A skeptical final review finds no unaudited submission surface, second canary, score-bearing side effect, or unrecorded terminal result.
  - Evidence: Final review re-enumerated code-output entrypoints, separated historical parity fixtures from runtime sources, and found no unrecorded external mutation.
- **core.independent-review** — passed
  - Dimension: independent-review
  - Criterion: A separate skeptical pass finds no remaining fixed runtime identity source or launch-governance bypass.
  - Evidence: Independent review pass traced the embedded APNCC submit runner and G04 postprocessor to their mounted test discovery functions and confirmed fixed public lists are local parity or historical-only.

### Goal detail

# Flight plan

Enumerate the G04 code-kernel build and execution chain, distinguish legitimate fit/provenance identities from mounted-test identities, and add coverage for every runtime discovery branch. The repository already contains one exact terminal code-output submission from the repaired pipeline; verify that receipt as the binary blind-rerun canary rather than launching an identical retry. Never patch and relaunch in this Flight.

