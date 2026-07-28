# G-02 Live State — shared source of truth

**Update protocol.** Single live status doc for the G-02 cell-tracking campaign, shared by
**Claude and Codex**. READ IT FIRST each session. UPDATE IT after any state change (a node
result, a gate flip, a DEAD promotion, a decision). Keep it SHORT — current state only;
push detail to the linked reports. Append a dated line to the Log (newest first). This doc
points at the authoritative artifacts; it does not replace them.

## Authoritative artifacts
- Plan / frontier: `campaign_narrowing_and_hypothesis_dag_v1.md` (human-locked 2026-07-25).
- Bet ledger (root coordinator is the sole writer): `LEDGER.md`.
- Scorer / CV57→public transfer gate: `p0_parity.md`.

## Current frontier (as of 2026-07-29)
- **N1 FEASIBILITY COMPILER GO / PHASE 1 BARRIER COMPLETE.** The immutable
  compiler accepts explicit no-op/add/delete/swap/fork/topology-restore actions,
  proves complete incident-conflict ownership, validates endpoints/time/degree/no-
  merge constraints, and compiles deterministic single actions and batches. It
  passed 10/10 tests, 6/6 action kinds, both batch orders, and 7/7 negative
  fail-closed fixtures with zero swap growth. Interface `2a72950c…f865`;
  implementation `457f6607…24f5`; tests `09107306…e97`; terminal report
  `tiered_surface/n1_feasibility_compiler/N1_FREEZE_REPORT.json`.
- **TIER CENSUS / NO LIVE RUN:** machine state
  `tiered_surface/tier_census.json` at SHA-256
  `a52d3dd32a46e1dbf19007f4d2fb07562ddffb7b8b7509b07ea987d5080a8759`
  records **LIVE 0 / PARK-A 13 / PARK-B 22 / PARK-C 9 / DEAD 1**. The five
  mandatory disjoint worktrees passed the collective root barrier at exact clean
  commits and were merged scope-only. DELETE-O0 moved PARK-A→PARK-C after its
  exact 44b6 failure. SWAP-O0 now records its exact-regret learner's banked
  `−0.000310288/−0.000296592/0` 44b6 failure as a PARK-C instantiation
  breadcrumb while the high-headroom surface correctly remains PARK-A pending
  the representation separation gate. Proposed demotions of the broader
  AE-P1/AE-P2 families remain rejected. R7's CELLECT supplier remains PARK-B.
  Manifest `f8852468…7a63`; barrier `5fef02ee…9349`; decisions
  `edba055b…0da1`; **uncited tier moves 0**.
- **CPU-ONLY WAVE-2 BARRIER COMPLETE / NODETOPO GRAPH HUMAN-HELD.**
  `SA-NODETOPO-GEN` authenticated all 36 fit105 44b6 Zarrs and 3,600 frames,
  froze 107,370 label-free proposals plus an exact 50-action
  `restore_topology` population across 24 videos, and passed frozen N1 for
  50/50 singles and 55/55 nonempty budget batches. `SA-DIVISION-DEEP` found
  **38/42 CV57 divisions present-but-fragmented**, one role-present
  component-spanning event without a local fork, and three role-missing events;
  39/39 diagnostic actions compiled, while fitting remains locked behind a
  both-family association-continuity pass. `SA-PARKB-SWEEP` audited 14
  CPU-testable PARK-B nodes, skipped eight weight/GPU-gated representation
  nodes, and proposed no move. Optional `SA-ASSOC-CYCLE` authenticated 36/36
  forward-cache files but found no immutable reverse supplier, so separation
  and graph stayed closed. Barrier `c60799d9…c63f3`; validation
  `6a467695…5c0f8` (**48/48 PASS**). Accepted and uncited tier moves are 0/0;
  census stays 0/13/22/9/1. No graph, GPU, Kaggle launch/debit, CELLECT write,
  RunPod, submission, or DEAD action occurred.
- **CONCURRENT CPU WAVE / ROOT BARRIER PASS.** PARK-B support/identity sweep
  `3bb73ef`, representation population/provenance `cefd106`, division refinement
  `0bfa879`, and authentic raw-node universe `27b7220` are terminal. Representation
  froze 154 exact 44b6 population shards (44,729,626 bytes) but found no new
  authenticated non-R7 supplier; division still has zero both-family continuity
  arms. Node topology authenticated 36 videos but completed zero: the serial
  path projected beyond five hours and the bounded retry stopped at +24.5 MiB
  swap, so no partial metric was consumed. Root rejected PARK-B moves for
  `GAP-F1`, `DET-F0`, and `G2` because their node-specific transition evidence
  was absent. The newer independent R7 terminal state and budget were preserved.
  Barrier `9fed2429…8f8d`; validation `9dae1cb1…b522`; accepted moves 0, uncited
  moves 0, census unchanged.
- **OWNER COMPUTE GOVERNANCE:** Kaggle GPU is authorized as needed within
  **30 h/week cumulative**, with hours banked every run. **RunPod is forbidden.**
  Public submission remains a separate explicit human gate. Local runs retain the
  45-second idle preflight and zero-swap rule.
- **CELLECT R7 SEPARATION / INVALID TERMINAL INFRA HOLD.** The authenticated-weight
  per-class amendment is frozen at `9fb030bd…3310e`; exact fit-only 44b6 populations completed under
  guard with delete **1,345,497 / 1,118 beneficial** and occupied-swap
  **45,961 / 134 beneficial** actions. The official checkpoint matches pinned
  commit `35860709…c63236` at SHA-256 `f3e8e730…89ce5`, is GPL-2.0-only,
  passed guarded tensor-only local loading, and was registered in private Kaggle
  asset v3. Both-class B0 is sealed at `a9ce8c0b…b4ff2`. The one authorized v3
  kernel still observed zero mounted files under the exact checkpoint name and
  stopped before checkpoint load or supplier features. No A1/A0 metric or
  scientific verdict exists across any attempt; total CELLECT invalid-run debit
  is **0.4591 kernel-h**. R7 remains PARK-B and no further launch is authorized.
- **H0 DONE** — proposal-only screening **BANNED** (no proxy `|ρ|≥0.5`). Every candidate now
  routes straight to the **44b6-first held-fit graph gate**; proposal metrics may be reported, never gate.
- **GA-H1-ZWRFEAT-GRAPH DEAD — 44b6 GRAPH KILL.** The human-authorized parity
  repair supersedes N1's false M5↔T4 fail: T4 is canonical, both T4 witnesses are
  array-exact, and the one-time M5 comparison is **BENIGN_FP**. The existing
  105-video cache was not regenerated. On 44b6, adjusted edge Jaccard
  **−0.01114808**, raw edge Jaccard **−0.01105753**, and node recall
  **−0.00028790**; held proposal precision was report-only **0.956522** (242/253).
- **NEXT / STOP** — H1 zero-wrfeat is falsified at the graph level. **6bba and
  CV57 were not run.** Do not build a submission kernel. `H3-MASK-SOURCE` is
  eligible only as a separately authorized provenance node for authentic
  per-detection masks; pseudo-masks remain forbidden and H3 is not started.
- **TOP-50 RESEARCH DECISION / HUMAN-GATE** — `top50_seeking_alpha_research_20260727.md`
  ranks a separately locked T4-canonical arm as the immediate enabler, followed by a
  cache-only higher-order conflict head and, only after association passes, a precise
  pretrained fork decoder. This opens no execution gate and does not amend N1.
- **TOP-50 RUN 01 G2 RESOLVED / C1-N3 DEAD** — owner locked
  U0 contract `08cd90e5…016c1` and authorized the frozen local/cache-only C1-N3
  cascade. This accepts T4-domain governance only; it does not convert U0 into
  44b6 signal or alter old H1 N1.
  **U0-N0/N1/N2 GO → N3 HUMAN-DECISION**: two 105-video T4-native replays
  are byte-identical (`1d1a8ac2…4758`) with exact APNCC disabled control and
  zero swap growth; all five 44b6 folds deterministically have no admissible
  threshold under the unchanged ≥100/≥0.95 rule. **C1 N0/N1/N2 GO → N3 PARK**
  after authenticating 105 videos / 195,354 proposals / 160,490 deterministic
  conflict components, locking one 20,097-parameter attention/no-link arm, and
  passing deterministic dry validation. **N3 is now DEAD**: all five C1 folds
  had admissible unchanged thresholds and 36/36 graphs were observation-distinct,
  but adjusted/raw/node deltas were
  `−0.00760293/−0.00759107/−0.00018539`; exact C0 equaled APNCC and
  `epsilon_null=1e−12`. N4 6bba and CV57 were not run;
  **C2-N0 PARKED** because no owner-recorded U0 lock or frozen C0 edge/no-link
  assignment-cost contract exists. **D1 N0/N1 GO → N2 PARK**: its fixed scorer
  decoder passed 14/14 official fixtures and 51/51 invariant counterfactuals with
  zero duplicate survivors, but no association arm has passed the required
  both-family graph/opportunity gate. **R0 N0 PARK**: zero pretrained association
  arms have immutable `>=+0.005` adjusted graph deltas on both families with
  nonnegative raw/node deltas; N1–N5 stayed closed. **F0 N0 PARK** after freezing its
  physical tri-view/XY/permutation contract: zero compatible frozen DINO/SAM encoder
  checkpoints are local, so weight hash/source/license authentication failed and
  N1–N5 stayed closed. GPU, Kaggle, RunPod,
  submission, push, and external mutation remain closed.

## The one live lever
Phase 1 is closed. The supplier-agnostic **R-SEPARATION-FALSIFIER-44B6**
per-class amendment is frozen at `tiered_surface/branches/sa_repr/phase2_separation_gate/`
(`9fb030bd…3310e`). Its exact 44b6 populations and both-class B0 are banked, but
the feature run is **INVALID / terminally infrastructure-blocked** before supplier
outcomes: v1 lacked `zarr`; v2 and the authenticated-weight v3 could not find the
mounted CELLECT checkpoint.
On the exact fit-only Phase-1
N1-valid delete and occupied-swap populations, every class/fold must clear
A1 AUC `≥0.65`, `≥B0+0.05`, `≥A0+0.03` and CG0-budget retention `≥0.15`,
`≥B0+0.05`, `≥A0+0.03`, with exact per-video budgets and full integrity.
Trackastra's prior AUC `0.918491` still led to a failed graph, so PASS only
unlocks a supplier-class-specific 44b6 graph contract; it is not a promotion.
No further retry, A1/A0 evaluation, full cache, graph, 6bba, or CV57 run is authorized. Division still
requires one association-continuity arm passing both families. C2 remains
PARK-B / NOT SCORED.

## Census / fallback queue

| Queue | Surface | Tier | 44b6 evidence | Headroom / next condition |
|---:|---|---|---|---|
| 1 | `AE-P1` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.075828`; newly named realizer only |
| 2 | `AE-P2` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.075828`; newly named realizer only |
| gated | `SWAP-O0` | PARK-A | exact-regret instantiation `−0.000310288/−0.000296592/0` (PARK-C breadcrumb) | `+0.075828`; representation falsifier PASS |
| gated | `R7` CELLECT | PARK-B | official weight + B0 authenticated; A1/A0 INVALID/unrun, graph unrun | materially new authorized mount-path resolution, then falsifier |
| closed | `DELETE-O0` | PARK-C | exact retention beta `−0.000127119/−0.000131807/0` | materially new named deletion mechanism |

## Key numbers
- Baseline local `0.873150` → public `0.877`. Banked APNCC local `0.881522` → **public `0.881`**.
- User-supplied top-50 bar is public `0.910`; coarse transfer maps it to a planning
  range of local `≈0.9255–0.9422`. Edge-only current-pool projection (`~0.896`) is
  insufficient; 12/20/25 zero-FP division TP would add `+0.0286/+0.0476/+0.0595`.
- In-pool re-ranking oracle `~0.896` (projection). Transfer slope `≈0.48–0.66` (soft, rounding-bounded).
- **Accepted (public) 0.89 needs local `≈0.895–0.900`**; zero-wrfeat H1 cannot
  supply it after the 44b6 graph kill. Kaggle GPU cap is **30 h/week cumulative**;
  RunPod is forbidden.
- P0 action-class surface (fit117 CG0 control, not APNCC): 44b6/6bba `ΔS` is
  delete `+0.088774/+0.074840`, swap `+0.075828/+0.059735`, division
  `+0.014723/+0.021405`, topology restore `+0.009812/+0.024040`, one-gap
  `+0.003639/+0.004424`, and free-add/node-add exactly `0`. Guard runtime
  **818.157 s**, peak RSS **1.151 GiB**, swap growth **0 MiB**.
- N1 compiler: **GO**; 10/10 unit tests, 6/6 single-action kinds, 2/2 batch
  orderings, 7/7 invalid fixtures rejected; fixed grouped folds confirm 175
  remainder groups and 20 untouched audit groups with zero leakage intersections.
  Freeze guard runtime **0.920 s**, peak RSS **0.068 GiB**, swap growth **0 MiB**.
- Phase-1 graph outcomes: DELETE-O0 adjusted/raw/node
  `−0.0001271190/−0.0001318070/0`; SA-SWAP exact-regret
  `−0.0003102875/−0.0002965922/0`; both stopped at 44b6.
- Tier census is **0/13/22/9/1** over 45 nodes. Weekly Kaggle budget artifact
  `tiered_surface/kaggle_budget.json` at SHA-256
  `2e02bc4f…1fc6` reconciles **2.5761/30.0 kernel-h used**, **27.4239 h remaining**,
  no active authorization; RunPod forbidden and submission human-gated.
- CPU-only Wave-2 NODETOPO prerequisite: **36/36 videos, 3,600/3,600 frames,
  107,370 proposals, 106,668 false proposals, 100,983 topology-joinable,
  50 restoration actions across 24 videos**, with 0 MiB swap growth. The graph
  gate remains human-held.
- CPU-only Wave-2 division diagnostic: **38/42 present-but-fragmented,
  1/42 present/component-spanning without a local fork, 3/42 role-missing**;
  39/39 diagnostic N1 actions compiled, 36 persistent actions selected, three
  missing-persistence actions suppressed, and zero duplicate survivors.
- H1 external usage: two human-created datasets; two failed timing-kernel versions, one
  passing smoke, and one completed full-cache kernel. Banked completed use is
  **2.1170 kernel-h** (≤**4.2340 allocated T4-GPU-h**), **0 submissions**. The
  44b6 adjusted/raw/node vector is
  `−0.01114808/−0.01105753/−0.00028790`; 6bba remains unrun.
  Package covers exactly **36 44b6 + 69 6bba**.
- H1-ZWRFEAT full-cache pins: contract `4da06d48…34b`, kernel `28a3801a…537`, source
  dependencies `a252e8ef…9e9`, graph contract `d35e2d3d…7f48`; focused verification
  **22/22 PASS**, staged v2 assets 55 files, swap 0 MiB.
- N1 pins: run manifest `548bbac4…8d10`, feature report `d52a4a42…c0cf`,
  cache attestation `e7c0ca58…ce3b`, tolerance evidence `8264bb9a…e621`;
  attestation preflight **45.152 s**, swap 0 MiB.
- Parity/graph pins: repair contract `f0beba3c…aa24`, rotated graph contract
  `09b67cdd…4fb`, coordinator result `94a3cd00…c9a`, 44b6 report
  `53011727…19d`, guard `3538f97e…a71`; runtime **2,092.348 s**, peak RSS
  **1.119 GiB**, swap growth **0 MiB**, and all 36 fragments distinct.
- Six-branch package: **6 branch DAGs / 33 branch nodes / 3 global nodes**; all node
  section-order and whitespace checks pass. External usage remains zero.
- D1 dry gate: **14/14** official scorer fixtures, **51/51** graph-invariant
  counterfactuals, **0** duplicate survivors, byte-identical repeat; **0** held graph
  runs and no CV57/GPU access.
- R0 dependency gate: **0 qualifying association arms**; no R0 model, graph, CV57,
  GPU, or external action. Banked generic PU/CRC/shell negatives were not rerun.
- C1 N3 graph gate: focused root verification **12/12**; all five folds had
  admissible ≥100/≥0.95 thresholds; 36/36 fragments distinct. APNCC/C0/C1
  adjusted scores `0.8665345260/0.8665345260/0.8589315915`; TP/FP/FN
  `10327/958/687`, identical, and `10284/1014/730`. Guard runtime **1,343.959 s**,
  peak RSS **1.133 GiB**, swap growth **0 MiB**, external usage zero.
- F0 representation contract: exactly **4 arms**, physical `13 µm³` crops,
  **32,457** trainable fusion parameters, **0** compatible local encoder checkpoints,
  downloads, GPU-hours, graph scores, or external actions.
- Root reproducibility: U0 **12/12**, C1 N3 suite **12/12**, D1 **14/14 + 51/51**, zero
  duplicate survivors; all branch JSON and whitespace checks pass.

## DEAD — do not re-propose without a new named mechanism (register: DAG §6.1)
Deterministic recall / shell abstention (`R12SHELL`, `MGSHELL`, `TEMPBG`) · from-scratch
rankers (`POINTCTX`, `CROPLINK`, `TEMPRANK`, `TEMPHYBRID`, `TRACKCTX`) · scalar recal /
threshold (`SCENCAL`, `POINTCTX-THRESH`, `APNCC-POLICY`) · hand-tuned graph-cost / motion
knobs (`GA-SP`, `LC-P4CTRL`) · full organizer `TemporalUNet3D` (~478 T4-h).
Conditional-PARK (revisitable): detection front-end · divisions · GT-conflict gates.
Frozen Trackastra zero-wrfeat H1 is DEAD at the 44b6 graph gate. Only the distinct,
separately authorized authentic-mask `H3-MASK-SOURCE` provenance question remains eligible.
Frozen C1 conflict-set attention (`C1-TA-HOCT-LITE`) is DEAD at its 44b6 graph gate;
do not run 6bba or re-screen it without a new named mechanism.

## Log (newest first)
- 2026-07-29 — CPU-only Wave-2 **ROOT BARRIER PASS / GRAPH HOLD**:
  banked exact clean lane commits `3f38a524`+`50bc0e08`,
  `cd7f24d3`+`cf703b5c`+`9151934e`, `47905928`, and
  `f92dd8d6`+`17c8d6e6`; all lane paths are
  disjoint and all four preflights exceeded 45 seconds with zero swap growth.
  NODETOPO froze an authentic 50-action restoration population, division
  quantified 38/42 strict present-but-fragmented CV57 events, PARK-B proposed
  no moves, and association-cycle stopped before separation because authentic
  reverse evidence is absent. Barrier validation passed 48/48, including
  per-lane `ALIGNMENT.md`/`SCOPE_MANIFEST.md` receipts; accepted/uncited
  moves remain 0/0 and census remains 0/13/22/9/1. The NODETOPO graph gate is
  held for human review. No GPU, Kaggle launch/debit, CELLECT action, RunPod,
  submission, DEAD promotion, or graph run occurred. (Codex root)
- 2026-07-28 — Concurrent CPU wave **ROOT BARRIER PASS**: all four clean,
  scope-only lane commits were collected; evidence hashes and memo quarantine
  validated. SA-REPR-MULTI froze 154 exact shards but queued zero new suppliers;
  SA-DIVISION remained dependency-closed; SA-NODETOPO stopped on +24.5 MiB swap
  before completing a video and consumed no partial metric. Root rejected the
  proposed `GAP-F1` B→C, `DET-F0` B→C, and `G2` B→A moves under their unchanged
  node-specific rules. Census remains 0/13/22/9/1; no GPU/Kaggle action, RunPod,
  submission, DEAD promotion, plan amendment, budget debit, 6bba, or CV57
  occurred. (Codex root)
- 2026-07-28 — CELLECT R7 authenticated-weight v3 is **INVALID / TERMINAL
  INFRA HOLD**: official commit/DOI/GPL provenance and checkpoint SHA
  `f3e8e730…89ce5` passed byte, tensor-only load, strict keyset, parameter-count,
  and zero-swap checks; per-class preregistration `9fb030bd…3310e` preceded a
  sealed both-class B0 and kernel v3. Private asset v3 reported all seven uploads
  ready, but Kaggle again mounted zero exact-name checkpoint files, so the run
  stopped before checkpoint load, features, A1/A0, or verdict and debited 0.1825
  kernel-h. R7 remains PARK-B because this is INVALID, not FAIL. No retry,
  training, full cache, graph, 6bba, CV57, submission, RunPod action, tier move,
  DEAD promotion, or plan amendment occurred; uncited tier moves remain zero.
  (Codex root)
- 2026-07-28 — CELLECT R7 v2 packaging-only retry is **INVALID / TERMINAL
  INFRA HOLD**: the pinned offline wheels installed, then kernel version 2 stopped
  before model inference because the expected CELLECT weight was absent from the
  mounted input. It emitted no complete feature manifest or scientific outcome
  and debited 0.2731 kernel-h (two-attempt total 0.2766 h). R7 remains PARK-B;
  the approved retry scope is exhausted. No further retry, full cache, graph,
  6bba, CV57, submission, RunPod action, tier move, DEAD promotion, or plan
  amendment occurred; uncited tier moves remain zero. (Codex root)
- 2026-07-28 — CELLECT R7 per-class separation is **INVALID / INFRA HOLD**:
  amendment `c3c86ec9…458c` preceded every opened population/label/outcome; exact
  44b6 delete and occupied-swap populations completed in 4,660.406 s at 1.080
  GiB peak RSS and zero swap growth. The one authorized private T4 version
  stopped before feature computation (`ModuleNotFoundError: zarr`), emitted no
  complete manifest, and debited 0.0035 kernel-h. No PASS/FAIL, tier move,
  full cache, graph, 6bba, CV57, submission, or RunPod action occurred. A local
  pinned-wheel repair requires explicit second-launch authorization. (Codex root)
- 2026-07-28 — Phase-2 **SEPARATION FALSIFIER FROZEN / HUMAN-REVIEW
  BARRIER**: SWAP-O0 now cites the authenticated Phase-1 exact-regret 44b6
  failure while remaining PARK-A; the failed realizer is retained as a PARK-C
  breadcrumb and no other tier changed. The fit-only supplier gate freezes
  identical B0/A1/A0 populations, per-class/per-fold AUC and exact CG0-budget
  retention margins, and fail-closed supplier semantics. No population,
  baseline, supplier, GPU, full cache, graph, 6bba, CV57, submission, RunPod,
  ledger append, budget debit, DEAD promotion, or plan amendment occurred;
  uncited tier moves are zero. (Codex root)
- 2026-07-28 — Phase 1 **BARRIER COMPLETE**: all five exact worktree commits
  passed path scope, frozen-input, evidence-hash, citation, memo-quarantine,
  44b6-first, budget, and forbidden-action validation. DELETE-O0 moved
  PARK-A→PARK-C; the exact SA-SWAP learner is closed, while overbroad AE-P1/AE-P2
  demotions were rejected. R7's CELLECT supplier is root-frozen for a new
  Phase-2 preregistration only. No GPU, budget debit, 6bba, CV57, submission,
  RunPod, DEAD promotion, or plan amendment occurred. (Codex root)
- 2026-07-28 — SA-SWAP corrected run terminal **KILL_44B6**: exhaustive
  frozen-N1 witnesses accepted all 45,961 candidates with zero rejects; 36/36
  alpha observations were distinct, but adjusted/raw/node deltas were
  `−0.0003102875/−0.0002965922/0`. The capacity-matched beta had the same
  aggregate vector. Guard runtime was 4,595 s at 0.861 GiB peak RSS and zero
  swap growth. 6bba, CV57, and SA-REPR integration stayed closed; clean commit
  and root validation remain pending. (SA-SWAP)
- 2026-07-28 — Root pre-barrier adjudication: DELETE's failed 13-parameter
  retention head may demote only `DELETE-O0`; its proposed `AE-P1` demotion is
  rejected because AE-P1 is the broader direct scorer-regret/no-op mechanism.
  Frozen 6bba fit rows are training inputs, not second-family graph execution;
  DELETE scored only 44b6 and kept 6bba/CV57 closed. (Independent review + root)
- 2026-07-28 — Independent review found SA-SWAP's optimized population path
  had replaced preregistered per-candidate N1 `compile_action` acceptance with
  structural checks and a hardcoded zero-invalid count. No graph outcome exists.
  A new run is blocked until every accepted candidate receives an exhaustive
  frozen-N1 locality witness (or full-baseline compile), with adversarial parity
  proof and pre-outcome implementation/test hashes. (Independent review + root)
- 2026-07-28 — SA-SWAP repair sequencing failed closed before launch:
  preflight v5 predated the execution-repair contract and no streamed parity test
  existed. No v5 family run or outcome was produced. The next attempt requires a
  real guarded parity suite, post-repair implementation/hash manifest, and a
  later fresh preflight in that order. (Codex root)
- 2026-07-28 — SA-SWAP's uncontended 44b6 attempt failed closed after
  enumerating 36/36 populations but before deployment or any graph vector:
  global swap grew 313.13 MiB over 382 s at 0.770 GiB peak process RSS. A
  pre-outcome, semantics-preserving streamed-memory repair is authorized under a
  new execution-only hash pin; scientific inputs/model/gates remain frozen.
  6bba, CV57, SA-REPR integration, and external actions stay closed. (Codex root)
- 2026-07-28 — SA-DIVISION terminal **PARK_DEPENDENCY**: the authenticated
  frozen evidence contains zero association arms passing both 44b6 and 6bba;
  fixture-only N1/scorer checks passed under a zero-swap guard. No division fit,
  family graph, CV57, GPU, submission, or external action opened. (SA-DIVISION)
- 2026-07-28 — SA-REPR's schema-only amendment is terminal: explicit
  CV57/submission/RunPod false fields and zero graph/family execution were added
  without changing frozen supplier contract `fffb09c2…3e70` or running an
  experiment. Its clean worktree now awaits the all-lane root barrier. (SA-REPR)
- 2026-07-28 — SA-REPR terminal **GO SUPPLIER-ONLY / ROOT QUARANTINE**:
  authenticated one local published CELLECT 64-D frozen microscopy encoder with
  exact source/weight/license and byte-identical synthetic CPU loads; 9/9 tests and
  zero swap growth. No cache, graph, GPU request, or integration opened. The strict
  root barrier rejected its fragment because explicit CV57/submission/RunPod false
  fields were absent; it cannot merge until amended and revalidated. SA-NODETOPO
  took the freed slot. (SA-REPR + Codex root)
- 2026-07-28 — Phase 1 **DISPATCHED / ACTIVE** with five mandatory disjoint git
  worktrees from `2c74eed…b8db0`. SA-DELETE launched first as lead, followed by
  frozen-input SA-SWAP and supplier-only SA-REPR; SA-NODETOPO and SA-DIVISION
  are queued for slots. Wave-1 pins N1 `2a72950c…f865`, folds `ddb313cb…8769`,
  and P0 `8ad548c1…9c06`. No GPU authorization, budget debit, second-family run,
  CV57, submission, RunPod, DEAD rescreen, or plan amendment occurred. (Codex root)
- 2026-07-28 — N1 feasibility-exact compiler **GO / PHASE 1 READY**: 10/10
  tests, all 6 action kinds, both batch orders, and all 7 negative fixtures passed;
  plan/P0/folds/provenance/fit117 hashes and 105+12 scope matched, with 20 audit
  groups untouched and zero swap growth. G1 moves PARK-B→PARK-A; census is
  0/14/22/8/1 and Kaggle usage reconciles to 2.117/30.0 kernel-h. No agent, model,
  graph score, CV57, GPU, RunPod, submission, or external action was launched.
  (Codex root)
- 2026-07-28 — Tiered-surface P0 **COMPLETE**: exact fit117 action-class oracles
  promote delete, occupied swap, dependency-blocked topology restore, and
  continuity-blocked division to PARK-A; free add and isolated node-add are null,
  and one-gap is weak/PARK-C. All transformed graphs validated; guard runtime
  818.157 s, peak RSS 1.151 GiB, zero swap growth. No training, CV57, GPU, RunPod,
  submission, or external action occurred. Tier census is 0/13/23/8/1; N1 compiler
  is next. (Codex + 3 subagents)
- 2026-07-27 — `C1-TA-HOCT-LITE` N3 **DEAD / 44b6 KILL**: all five
  frozen folds had admissible unchanged calibration thresholds and 36/36 C1
  fragments were observation-distinct, but adjusted/raw/node deltas were
  `−0.00760293/−0.00759107/−0.00018539` versus exact APNCC/C0.
  `epsilon_null=1e−12`; 6bba/CV57 stayed closed. Guard peak RSS was
  1.133 GiB with zero swap growth and zero external usage. (Codex)
- 2026-07-27 — `GA-H1-ZWRFEAT-GRAPH` **DEAD / 44b6 KILL**:
  Option-C parity repair classified M5↔T4 drift BENIGN_FP and accepted the
  existing exact-repeat T4 cache. The fit-only graph gate then regressed
  adjusted/raw/node by `−0.01114808/−0.01105753/−0.00028790`; 6bba/CV57
  stayed closed. No GPU, Kaggle, RunPod, submission, or forbidden-scope
  access occurred. (Codex)
- 2026-07-27 — G2 owner resolution **LOCK / RESUME**: locked U0-T4CANON
  `08cd90e5…016c1` as T4-domain governance and opened only frozen C1-N3 local
  44b6-first execution. Old H1 N1 remains DEAD; the U0 5/5 44b6 no-threshold
  negative is retained. CV57/GPU/submission/external actions remain closed. (Owner)
- 2026-07-27 — TOP-50 Run 01 G1 **GO TO G2 / HUMAN-DECISION**: joined all six
  terminal artifacts with every negative retained. U0 is exact governance but has
  `NO_ADMISSIBLE_THRESHOLD` on 5/5 44b6 folds; C1 is the only dependency-safe cheap
  next falsifier after an explicit U0 lock. No graph/CV57/GPU/external action ran.
  (Codex + subagent)
- 2026-07-27 — U0-T4CANON N3 **HUMAN-DECISION**: complete pinned dossier
  presents `LOCK U0-T4CANON`, `REJECT`, or `REQUEST REVISION`. A lock would open
  only dependency-safe local/cache work; it is not 44b6 signal and authorizes no
  CV57-before-gates, GPU, submission, RunPod, push, or old-N1 reinterpretation. (Codex)
- 2026-07-27 — U0-T4CANON N2 **GO → N3 HUMAN-DECISION**: two isolated
  105-video scorer-free replays are byte-identical (`1d1a8ac2…4758`), disabled
  challenger reproduces APNCC on 105/105, and swap growth is 0 MiB. Negative
  evidence: 44b6 has no admissible threshold in 5/5 folds under the unchanged
  calibration rule; no tolerance, scorer, CV57, or external action changed. (Codex)
- 2026-07-27 — F0-TRIVIEW-DENSE-TA N0 **PARK_MISSING_LOCAL_ENCODER_WEIGHTS**:
  froze the physical `13 µm³` XY beta, tri-view alpha, deterministic non-XY
  permutation ablation, 32,457-parameter fusion head, null, and graph kill lines.
  No compatible frozen DINO/SAM checkpoint is local, so weight/source/license
  authentication failed; N1–N5, the Wave-1+2 ceiling claim, and GPU gate stayed
  closed. No download or external action occurred. (Codex)
- 2026-07-27 — C1-TA-HOCT-LITE N2 **GO**, N3 **PARK_DEPENDENCY**: built the fixed
  20,097-parameter head; passed 8/8 tests, exact full-row C0 replay, and
  byte-identical real fit-only C1 dry OOF. No U0 owner lock exists, so 44b6,
  6bba, CV57, and scorer access stayed closed. (Codex)
- 2026-07-27 — R0-TA-PU-SHELL N0 **PARK_DEPENDENCY**: no single pretrained
  association arm has immutable `>=+0.005` adjusted graph deltas on both 44b6 and
  6bba with raw/node deltas nonnegative. The four-part full-population
  pretrained+nnPU+no-link+groupwise-risk design remains a distinct research bet;
  N1–N5 stayed closed and banked generic negatives were not rerun. (Codex)
- 2026-07-27 — U0-T4CANON N1 **GO**: new scorer-free interface passed **12/12**
  tests plus 105+105 payload/sidecar coverage; changed row/ID/fold/schema/hash/scorer,
  forbidden path, and M5 decision input all failed closed. Focused-test swap growth
  was 0 MiB; no metric or external action occurred. (Codex)
- 2026-07-27 — C1-TA-HOCT-LITE N1 **GO**: self-hashed the single 20,097-parameter
  attention/no-link architecture, grouped folds, sparse-label rules, training,
  controls, null, and graph kill lines before any score. N2 started; N3 remains
  U0-lock dependent. (Codex)
- 2026-07-27 — D1-PERSISTENT-FORK N0/N1 **GO**, N2 **PARK_DEPENDENCY**:
  froze D0/D1/persistence-ablation before outcomes; reproduced 14/14 official
  scorer fixtures and 51/51 invariant counterfactuals with zero duplicate survivors.
  No association arm yet has a qualifying both-family graph plus positive fork-
  opportunity delta, so N3–N5 stayed closed. (Codex)
- 2026-07-27 — C1-TA-HOCT-LITE N0 **GO**: authenticated 105 fit videos,
  195,354 proposals, 280,095 conflict rows, and 160,490 deterministic components;
  pinned the Trackastra/HOCT mechanism, sparse-label risk, five grouped folds, beta
  and no-attention controls. No graph score or external action occurred. (Codex)
- 2026-07-27 — U0-T4CANON N0 **GO**: froze a separate T4-native beta/governance
  contract with exactly 12 machine-checkable invariants, 105-video 36/69 scope,
  forbidden-split and M5 decision isolation, and no tolerance change. Old H1 N1
  remains DEAD; no scorer, CV57, or external action occurred. (Codex)
- 2026-07-27 — C2-KBEST-STABILITY N0 **PARK_DEPENDENCY**: froze `K=8`, exact
  membership/min-marginal rule, APNCC fallback, controls, null, and graph thresholds,
  then halted because U0 is not owner-locked and no frozen C0 assignment-cost/MAP
  contract exists. N1–N4 and all scoring stayed closed; external usage zero. (Codex)
- 2026-07-27 — TOP-50 Run 01 G0 **GO / ACTIVE**: owner triggered all six branches.
  U0/C1/C2 dispatched into three worker slots; D1/R0/F0 recorded in an immutable
  backfill queue. No GPU, Kaggle, RunPod, submission, push, or external mutation is
  authorized. (Codex)
- 2026-07-27 — TOP-50 six-branch foundations BANKED / READY FOR OWNER TRIGGER:
  U0/C1/C2/D1/R0/F0 received literature-grounded hypotheses, beta/alpha controls,
  preregistered graph kill lines, dependency locks, and permission stops. No experiment,
  CV57, GPU, submission, or external mutation occurred. (Codex + 3 subagents)
- 2026-07-27 — TOP-50 seeking-alpha research BANKED / HUMAN-DECISION: edge-only
  reranking cannot plausibly span public 0.881→0.910. Ranked a new T4-canonical
  contract, compact higher-order Trackastra conflict head, and conditional persistent
  fork decoder; no locked gate, compute, CV57, or submission was opened. (Codex)
- 2026-07-27 — H1-ZWRFEAT-CACHE-ATTEST **DEAD**: full cache/atomic integrity and
  two-family T4 repeatability passed, but exact-only M5↔T4 geometry drift and eight
  out-of-probe float/neural columns triggered `STOP_EXACT_GEOMETRY_OR_IDENTITY_DRIFT`.
  N2/CV57/submission remain closed; H3 is not eligible. (Codex)
- 2026-07-27 — H1 full-cache monitor CONTINUE: remote kernel `RUNNING`; Stage A reached
  82/105 at 5,088 s with steady warm progress and projected total ~2.3–2.5 h. No abort
  or 7 h leeway is warranted now; recorded the conditional 7 h stop rule. (Codex)
- 2026-07-27 — H1-ZWRFEAT-CACHE-PREP package ready: timing evidence, full-cache
  contract, source overlay, kernel hashes, and a five-node auto-cascade were pinned.
  Per `dag-prompts`, halted before the human Kaggle dataset-version/GPU gate. (Codex)
- 2026-07-27 — H1-ZWRFEAT-SMOKE3 PASS: authenticated T4 timing artifact projects
  **5.538884 h** with margin, with both-family T4 repeatability PASS. The notebook is
  intentionally non-submittable; it produced no `submission.csv`, full cache, or graph
  result. Next is full-cache contract preparation, not a public probe. (Codex)
- 2026-07-26 — H1-ZWRFEAT-SMOKE2 STOP after registration cleared: launcher omitted the
  pinned fit105 manifest from `/kaggle/working`, so Stage A failed before model work.
  Corrected the authenticated copy path locally; repush remains human-executed. (Codex)
- 2026-07-26 — H1-ZWRFEAT-SMOKE1 STOP at source registration: fit105 dataset is ready v1,
  but pulled remote kernel metadata omitted that source although local metadata declared it;
  code hashes match. Kernel failed in 6.7 s before CUDA/model/timing work. Added an exact
  remote-registration verifier; retry remains human-executed. (Codex)
- 2026-07-26 — H1-ZWRFEAT READY at the human upload boundary: decoupled points-only
  zero-wrfeat contract, fit105-only two-stage package, exact create/version-verify/smoke
  commands, and 44b6-first graph runner banked; 18/18 focused tests passed after a 45 s
  zero-swap preflight. H3-MASK-SOURCE is conditional and unstarted. No Kaggle/RunPod
  mutation or graph run occurred. (Codex)
- 2026-07-26 — H1 STOP / RESERVE QUOTA at launch-integrity gate: authenticated fit105
  manifests contain no real H3 mask/wrfeat artifact; authenticated Kaggle listings contain
  neither H1 dataset nor kernel. No external mutation or downstream gate ran. (Codex)
- 2026-07-26 — Human explicitly authorized the narrowly scoped H1 Kaggle timing smoke,
  conditional frozen-cache launch, and one conditional public probe; all preregistered
  runtime, identity, H3, 44b6-first, 6bba, and CV57 gates remain mandatory. (Codex)
- 2026-07-26 — Reconciled `R12SHELL`/`MGSHELL` ledger cells PARK→DEAD; added DAG §3.1
  public-target arithmetic; created this live-state doc + `AGENTS.md`/`CLAUDE.md` wiring. (Claude)
- 2026-07-26 — Cascade committed on `codex/h0-h1-campaign-cascade-v1`: H0 (`a3de734`) BAN
  verdict; H1-PREP (`0ed7983`) launch-ready but stopped at the human gate. (Codex)
