# G-03 Live State — end-to-end association training

**Update protocol.** This is the shared current state for G-03. Read it before any action. After every decision, outcome, gate change, or external action, refresh Current frontier / Key numbers and append one short dated Log line. The root coordinator is the sole writer of `LEDGER.md`.

## Authoritative artifacts

- Campaign charter: user instruction dated 2026-07-31.
- Architecture selection: `ARCHITECTURE_DOSSIER_20260731.md`.
- Phase 1 v1 terminal: dedicated-worktree commit `05c1c6de566abad84257bbdb33ff258c6ecf6f9a`,
  `PHASE1_RESUME_PREREGISTRATION_20260731.md`, and
  `PHASE1_LAUNCH_CONTRACT_20260731.json` on branch `codex/g03-phase1-resume-proof`.
- Phase 1 v2 transport-repair freeze: dedicated-worktree commit
  `2241c72b7f4bf23f1b8118183d735b99476a35bf`,
  `PHASE1_TRANSPORT_REPAIR_PREREGISTRATION_20260801.md`, and
  `PHASE1_TRANSPORT_REPAIR_CONTRACT_20260801.json` on branch
  `codex/g03-phase1-singlefile-retry`.
- Bet ledger: `LEDGER.md`.
- Legacy evidence and byte-exact transport: `../../g02/screening/`.
- Wave R owner policy and dispatch: `WAVE_R_POLICY_AND_DISPATCH_20260801.md`.

## Current frontier (as of 2026-08-03)

- **OWNER LOCK / ZERO G-03 GPU SPEND UNTIL ALIGNMENT PATCH CONFIRMATION.** No
  h030 launch and no other G-03 GPU action may launch until the pending
  `../../g02/screening/tiered_surface/ALIGNMENT.md` patch is explicitly
  confirmed landed by content inspection and SHA. The inspected current SHA
  `aff9e656…abaff` lacks this lock, so confirmation is negative and the lane
  fails closed. CPU-only inspection remains allowed; this lock does not govern
  the independent public-TemporalUNet substrate/ensembling lane.

- **OWNER-AMENDED ORTHOGONAL LANES TERMINAL / H005 UNMEASURED AFTER TWO
  PRETRAINING INFRASTRUCTURE FAILURES / LANES 1 AND 2 KILLED AT CEILINGS.**
  The owner superseded only the earlier h005
  non-funding decision and funded one fixed approximately-five-hour read;
  h030 still requires the held-fit36 slope projection to reach held-fit36
  APNCC `0.8665345260361912`. Training is fixed to 69 6bba identities, full
  end-to-end Trackastra plus the zero-init adapter, one seed, no sweep, a
  4.75-hour checkpoint, and a 4.9167-hour hard stop. All 69 development joins
  pass (`57,776` authenticated positive-edge windows); the inline Kaggle guard
  passes. Private carrier dataset `onebitbrain/g03-h005-contract-v1` version 1
  is invalid pre-GPU because its freeze timestamp used local clock text with a
  UTC suffix; corrected version 2 is the only launch input. Kaggle accepted
  training-kernel version 1 at `2026-08-02T19:33:09Z`. Its top-level inline
  STEP0 passed every pinned carrier hash, then core import failed because the
  launcher declared but did not install the canonical wheels and the base image
  lacked `zarr`. Training main, CUDA model allocation, data access, optimizer
  steps, checkpoint, held-fit36 score, and slope are absent. Authoritative quota
  moved from `1.01/28.99 h` to `1.02/28.98 h`, a rounded `0.01 h` infrastructure
  debit. Version 1 is closed as invalid. A dependency-only repair is now frozen:
  private dataset `onebitbrain/g03-h005-runtime-deps-v1` version 1 is READY with
  14 hash-pinned wheels for CPython 3.11--3.13, and a frozen CPU-only one-identity
  release-path probe must pass exact install/import/raw+GEFF decode checks before
  the sole repaired GPU version 2 becomes eligible. CPU probe kernel version 1
  completed and passed on Kaggle CPython 3.12: all 14 carrier hashes, exact
  8-wheel offline install, NumPy/SciPy/Torch/PyYAML/zarr imports, and the fixed
  6bba raw+GEFF byte hashes matched in `9.223 s`; GPU/optimizer/checkpoint/
  held-fit/CV57/metric access were zero and quota stayed `1.02/28.98 h`. The
  unchanged-core sole V2 passed its static release audit and fresh inline,
  serial-launch, and quota guards, but its exact four-source GPU worker failed
  at top-level STEP0 after `3.5865 s`: the runtime manifest was absent or its
  bytes differed from the frozen SHA. The check emitted neither presence nor
  observed hash, so the two cases cannot be distinguished. Remote output files,
  dependency installation, scientific-core import, CUDA allocation, data access,
  optimizer steps, checkpoint, score, and slope are all absent. V2 rounded debit
  is `0.00 h`; quota remains `1.02/28.98 h`. The frozen no-V3 policy closes Lane
  0 infrastructure-terminal. The held-fit36 h030 gate is not evaluable and h030
  is not funded; evaluation/CV57/hybrid/submission/RunPod remain closed. In the
  orthogonal fork lane, the one-shot trained raw-voxel model
  failed cross-family 44b6 before final-population access: at threshold `0.5`,
  TP/FP/FN/TN `5/6/4/30`, precision/recall `0.454545/0.555556`, ROC AUC/AP
  `0.777778/0.570905`; best precision at the required recall was `0.5`, below
  `0.714286`. Final 38 access, CNN, graph integration, hybridization, and GPU
  remain closed. Lane 1's classical dense-field screen completed all 41 frozen
  44b6 identities and returned `FAIL_PARK_C` for both DELETE and occupied SWAP.
  DELETE A1 AUC was `0.722147/0.770644/0.646992/0.687785/0.707490`, but
  retention was only `0.006536/0.010101/0/0/0.008547` versus the `0.15`
  floor; occupied SWAP also failed every-fold conjunction. Learned displacement
  training is closed with zero GPU spend.

- **PHASE-2 ZERO-INIT H000 COMPLETE / EXACT SAME-RUNTIME PARITY / BRANCH B /
  DOSSIER FAIL / H005 NOT FUNDED.** The versioned 22,684-parameter crop
  adapter now zero-initializes both tensors of its final `Linear(48,12)` and
  emits bitwise-zero `wrfeat` at h000 while retaining nonzero projection
  gradients. Immutable checkpoint `603e02f8…18f95` records zero optimizer steps
  and zero GPU-hours. A local CPU replay exposed a one-edge discontinuity when
  `0.3499999046` from the frozen CUDA association bank became `0.3500036001`
  and crossed the fixed `0.35` decode threshold. The score-blind canonical
  amendment therefore re-decoded the already-frozen CUDA banks under the
  unchanged compiler: all `36/36` graph fragments are byte-identical to the
  zero-feature control. Corrected held-fit36 h000 is
  `0.7630876414338513` adjusted, `0.7581383935874366` edge Jaccard,
  `0.9747454252716499` node recall, edge `9269/1212/1745`, division
  `0/0/9/0`; exact parity error is `0`. The cited native `0.829194058080` is
  CV57, not a same-population baseline. Historical random-adapter R1 is only
  `+0.0006462389830586` above h000 on held-fit36, so the claimed `~0.066`
  initialization cost is not supported. H000 is Branch B; the primary-source
  effect-size dossier fails the required target-domain/metric `<=30 T4-h`
  comparability and calibrated `0.915 -> 0.945` runway. Allocation is
  `0/30 T4-h`; h005/h030 were not launched, no slope exists, and the full
  30-hour scientific allocation is preserved. Report:
  `phase2_zero_init_v1/H000_TERMINAL_REPORT.md`.

- **WAVE R ALLOCATION COMPLETE / BRANCH B / NO GPU / R4 TERMINAL OPERATIONAL KILL / NO SCORE.** R1's single
  zero-shot pretrained Trackastra plus real-image untrained crop adapter scored
  `0.7637338804169099` on held-fit36, below the exact policy floor
  `0.868299238910600` and campaign best `0.881522069960`; classification is Branch B.
  Separate terms are adjusted/raw edge Jaccard
  `0.7637338804169099/0.7584738824299984`, node recall `0.9723341840137287`, edge
  `TP/FP/FN=9264/1200/1750`, and division `TP/FP/FN/Jaccard=0/0/9/0`. R2's nine-source
  primary literature audit fails the required `<=30 T4-h` comparability and calibrated
  `0.915 -> 0.945` runway propositions. There is therefore no eligible candidate: the
  allocation is `0/30 T4-h`, Wave-R GPU spend is `0`, and best remains
  `0.881522069960`. The latest authenticated global quota remains `1.01/28.99 h`
  used/remaining with refresh `2026-08-08 00:00 UTC`; no quota-expiry launch is allowed.
  R1 result SHA is `55300582…4372`; terminal accounting records two identical target
  attempts, one scientific configuration, and one exposed score after a strict-JSON
  NaN serialization recovery. All 36 bank/parent outputs remain reusable for R4.
  The synthesis/execution plan is now binding. Exact dev69 bank-extension preflight
  `960e3c61…c4466` passed target-blind after a pre-publication edge-API guard repair,
  and CPU inference completed 69/69 in an isolated worktree under the identical R1
  scientific configuration. Dev seal `588397b1…e6d8`, exact local105 coverage
  `aa393a71…cbc7`, V2 pretarget plan `5e357d02…10b5`, and V2 runner input
  `c629e5cd…6485` are promoted and sealed before development scoring. The fixed
  18,144-active plus null lattice then runs five-fold nested/grouped OOF; held36 is
  non-selecting and one-shot,
  hidden4 is inaccessible until local PASS, and contaminated audit20 is forbidden.
  Combined Wave-R tests pass 72/72 after promotion of the machine-gated OOF runner,
  exact-runner-binding orchestrator `4a6f967b…9cd53`, and target-blind local105 input
  repaired generator `c0937c98…d1d2d9`, runner `2196a84d…5f0`, and orchestrator
  `dc30414e…fdcc`; the post-runner-repair independent re-audit passes. The first
  target-free fold0 decode was stopped before scorer access after the live spool
  showed only 4/18,145 configurations in several minutes (roughly 15 days/fold
  projected), so development scoring was paused while an exact factorized CPU path
  was implemented and audited. The conservative factorization retains all three
  `late_window` values and all duplicate/cap settings as distinct production
  semantics; no apparently inert decoder axis is canonicalized. It may reuse only
  trusted candidate generation across fields that generation does not read, exact
  in-memory action-set-equivalent graph materializations, and the existing
  content-addressed target-free graph cache, all fail-closed against full-decoder
  graph identities. Runner `2b9cb30e…7284`, orchestrator `fcca7bf9…ab49`, V3
  pretarget plan `da2bfa61…d96d`, V3 runner input `bc815caf…0322`, and repair
  receipt `8cbc2727…8109` are sealed; 75/75 tests and the Wave-R validator pass.
  One exact target-free smallest-video benchmark represented all 18,145 rows in
  `76.084 s` via 1,008 generation calls, 47 trusted full graph decodes, and 18,098
  exact action-set reuses. Its verified scorer modules were imported while loading
  production inputs, but no scorer backend, metric function, archive member, or GT
  object was opened. Production development
  scoring remains closed at the runtime gate because the audited fold0 projection
  contains uncapped action sets with up to 11,829 accepted forks and the frozen
  sequential compiler has not demonstrated a bounded wall time for that regime.
  Independent V3/cache audit passes but estimates about 35,487 full-graph
  `compile_action` calls for that worst topology and blocks scorer opening. A
  target-free one-pass materializer prototype completed only in the isolated
  execution worktree (23 pass, 1 skip; synthetic 11,829-action disjoint topology
  `6.42 s`, about `210 MB` peak RSS), while production runner `2b9cb30e…7284`
  remained unchanged. Release additionally requires coherent frozen-API/path/hash
  binding, exact incident-edge indexing, the actual sealed worst-case hard-wall/RSS
  benchmark, production-cache and tractable-sequential differentials, an explicit
  compiler-path amendment, full resealing, and another independent PASS.
  Reproducible all-grid target-free blocker evidence `0ba158be…59a3f` audits all
  18,144 active rows on sealed `train:6bba_57b7cc1e` and finds 1,350 sequential-invalid,
  topology-affected rows (`7.44%`; 450 settings repeated across the three inert
  duplicate-window values); 1,269 change accepted-event count and 81 change topology
  at the same count. All affected rows on this identity are uncapped, while caps
  1/2/3/5/8 end before the first conflict. Representative
  `t035_g100_l0_dp0_gp0_n0_call` moves from 11,829 existing accepts to 11,775 under
  integrated ranked conflict ownership: 92 conflict candidates are rejected and 36
  replacements are newly admitted. The earlier 11,749 post-filter is not equivalent
  because it cannot reconsider node-suppressed replacements. The frozen compiler
  reproduces an `ActionCompileError` on the second claimant of baseline edge
  `(2557,3591)`. Adding frozen-baseline incident-conflict ownership inside the ranked
  acceptance loop before cap changes predictions despite preserving all 18,145 row
  names/values/folds/gates. The owner granted the exact semantic, frozen incident-index,
  one-pass compiler, graph-reuse, and receipt-contract amendment on 2026-08-01; implementation
  is active, while scorer access remains closed pending supersession of the prior amendment,
  full reseal, exact differential and actual-worst hard-wall/RSS checks, and independent
  pre-score PASS. Two target-blind actual-worst benchmark attempts (initial production
  amendment and the temporal-index optimization) failed closed at the external `60 s`
  hard kill because Python allocation tracing distorted import, generation, and
  materialization wall time. Before any PASS rerun, a target-blind stage diagnosis and
  Flight amendment preregistered production-wall plus process-RSS measurement without
  that tracer, retaining the exact `<30 s` decode, `60 s` internal/external kill,
  `<1 GiB` RSS, and `<256 MiB` swap gates. The corrected binding run passes in
  `3.648069624963682 s` decode and `17.668541667051613 s` external wall at
  `582025216` bytes peak RSS and zero swap growth, with 11,775 accepts, 92 conflict
  rejects, one graph construction/hash, direct-edge equality, and zero forbidden
  access (`fc479872…a3be9`). Independent all-grid audit `c3b3934a…317c8` passes all
  18,144 active rows against the integrated reference and reproduces byte-exactly.
  V4 source/plan/input/cache provenance is now resealed: amendment V2 `bd5a3af7…ecabcf`,
  decoder `a58c4964…521c0`, runner `cc37fc63…de304`, orchestrator
  `853e11a9…b22d7`, pretarget plan `ecbb7467…665a4`, runner input
  `be6f7a4e…6ee23`, and semantic/runtime receipt `7b2ff492…006b5` bind a fresh
  `graph_cache_v4` namespace and reject V3 cache receipts. The Wave-R validator and
  86-test regression suite pass. Final independent pre-score audit
  `0b7361ca…b3652` independently reproduces the validator, 86/86 tests, byte-exact
  all-grid check, dynamic V3-cache rejection, V4 empty-cache provenance, actual-worst
  resource gates, and zero prohibited access. Its verdict is
  `PASS_V4_PRE_SCORE_CHAIN_MAY_PROCEED_TO_FIXED_FIVE_FOLD_NESTED_GROUPED_OOF`.
  Inner folds 0--3 each completed the exact 18,145-row planning pass, with
  `13,078/13,049/13,498/13,135` full graph decodes and
  `984,897/1,003,071/1,002,622/984,840` action-set reuses, then entered the
  sealed unique-graph scorer. All four CPU-saturating scorer processes were killed
  before atomic publication: no `INNER_SCORE_RECEIPT.json`, `SEAL.json`, or runner
  diagnostics exists. From per-fold spool creation to the first root observation
  that no R4 runner remained, elapsed wall time was approximately `25 h 57 m` for
  folds 1--3 and `26 h 22 m` for fold 0. That working-day overrun is the terminal
  finding: this fixed 18,145-configuration lattice is not a viable screening
  instrument. A post-failure recovery check touched SQLite schema and aggregate row
  counts only, read no graph identity, score, or metric value, and left the private
  spools scientifically quarantined and inadmissible. R4 therefore failed closed
  with no score, division term, edge term, winner, held access, or gate flip. With
  no sealed winner, no exact 44b6 comparison can be made; separately, the
  premeasured tier-(a) ceiling `+0.002406490` (1/38 inferable divisions) was already
  below the unchanged `+0.005` promotion gate and, under the new ceiling-first rule,
  would have barred this lattice as promotion-funded machinery before construction.
  The owner's 2026-08-03 sequencing barrier released A7 only after this kill. A7's
  exact frozen 45,961-candidate / 134-positive occupied-swap separation measurement
  is now terminal **FAIL / EVIDENCE PARK**. The hash-pinned B0-first, A1, and
  within-video/frame-permuted A0 protocol used the original A7 `action_margin` as
  its sole supplier score. Per-fold A1 AUC is
  `0.627866/0.828496/0.610208/0.659067/0.654961`, every-fold CG0 retention is `0`,
  A1-B0 AUC lift is `+0.002324/-0.008582/-0.012093/-0.005730/-0.001053`, and
  A1-A0 lift is `+0.001574/-0.006478/-0.011753/-0.005457/-0.001522`. All folds
  miss the occupied-swap `0.90/0.067910` absolute requirement and unchanged lift
  clauses, so no FIT threshold, edit set, held-44b6 graph, edge/division metric,
  CV57, or GPU action opened. Result `3143a31d…f96d`. The requested future gate
  text (`adjusted >=+0.005`, `raw >=0`, `node >=-0.002`) is packaged at
  `GATE_AMENDMENT_PACKAGE_NODE_MINUS_0P002_DRAFT_20260803.md` as **DRAFT ONLY / NOT
  APPLIED**; it records the single soft APNCC transfer anchor (`0.006950 ->
  0.008372`, ratio `0.8300`) and changes no historical verdict. No Kaggle mutation,
  submission, CV57, final-slot, or RunPod action occurred.
- **PHASE 0 / ARCHITECTURE SELECTED / NO GPU.** `G03-TA-CROPADAPTER-E2E-V1` is a pretrained Trackastra association transformer, fully fine-tuned with a small 3-D intensity-crop adapter producing the formerly zero-filled twelve `wrfeat` values.
- **PHASE 1 COMPLETE / A v2 + B v2 PASS.** The one hash-pinned A v2 (`onebitbrain/g03-phase1-resume-a-v2`, version 1) independently verifies as `PASS_A_HANDOFF_READY`: exactly `3,600.000229 s` synchronized T4 training, step `1,953,711`, zero swap growth, checkpoint SHA `8a6c2867…00e134e`, and atomic tree SHA `73a6cc56…b56ac`. Its exact tree is READY as private handoff dataset version 2 and passed full remote download round-trip verification; version 1 remains an immutable pre-GPU carrier-layout failure. Frozen B v2 (`onebitbrain/g03-phase1-resume-b-v2`, version 1) was pushed exactly once, completed, and independently validates as `PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED`: final digest `816acee4…c80387b` is the exact uninterrupted reference, all 36 structural graph members emit deltas `0/0/0`, and swap growth is zero. B used `1.01 T4-h`; the remaining `6.99 h` of the owner ceiling is preserved because a repeat would be observation-identical. Phase-2 **preparation** is open, but scientific training, official graph access, CV57, submission, and the final slot remain closed until the crop/adapter/cost/output/checkpoint contract is frozen. No v3 is needed or authorized.
- **G-02 IS ARCHIVED; ITS NATIVE V1 IS RECONCILED AND REMAINS OUTSIDE G-03.**
  The legacy Trackastra v1 completed with exact version delta `0→1`, a
  `0.37 GPU-h` debit, and 105 structurally valid downloaded banks. Its original
  G-03 supersession was correct at the time; the later owner-scoped G-02
  exception below alone completed/scored it. Neither the bank nor that result is
  a G-03 training artifact or Phase 1 input.
- **SCOPED G-02 LOCAL SELECTED-ONLY CV57 ONE-SHOT TERMINAL / G-03 UNCHANGED.**
  G-02's sole sealed CV57 measurement is complete from already-downloaded bytes.
  `r08_c08_a035_off` scored `0.829194058080`, a `-0.052328011880` delta versus
  APNCC `0.881522069960`; division TP/FP/FN/Jaccard is `0/0/42/0`. All 57 native
  graphs predicted zero forks, so the preregistered hybrid is graph-identical to
  APNCC and ties it exactly. Independent pre/post audits pass. No alternative,
  tuning, repeat, GPU/Kaggle action, submission, RunPod, G-03 gate, or G-03
  artifact changed. Terminal SHA `0add0c3f…60af95a`.
- **FINAL 1/2 REMAINS LOCKED.** APNCC-DIRECT stays final slot 1; the proven byte-exact transport is retained and only the remaining final slot is open.

## Key numbers

- Phase-2 h000: corrected same-protocol held-fit36 `0.7630876414338513`;
  `36/36` canonical fragments byte-identical; parity error `0`; checkpoint
  `603e02f8…18f95`; zero-init minus historical random-adapter R1
  `-0.0006462389830586`. Branch B dossier **FAIL** remains historical; the
  later owner amendment funds h005 only. H005 kernel version 1 is invalid before
  training (`zarr` unavailable), with no checkpoint or slope and a rounded
  `0.01 h` infrastructure debit; h030 is unauthorized until a valid held-fit36
  projection gate passes.
- Lane 2 trained fork: cross-family 44b6 TP/FP/FN/TN `5/6/4/30` at `0.5`,
  precision/recall `0.454545/0.555556`, AUC/AP `0.777778/0.570905`; **STOP
  BEFORE FINAL 38**, GPU `0`, graph integration false.
- Lane 1 dense motion: all `41` frozen 44b6 identities, `4,059` frame pairs,
  and `1,391,458` actions; DELETE and occupied SWAP both `FAIL_PARK_C`; CPU
  wall `6,386.171 s`, peak RSS `229.94 MiB`, GPU `0`; VoxelMorph-class stage
  closed.
- Banked baseline: local `0.881522069960`, public `0.881`; target local `>=0.9255` and strict improvement over baseline.
- Allocation floor: exact `0.868299238910600` at current best `0.881522069960`; Branch A
  is `score >= floor` or new best, while Branch B requires the audited `<=30 T4-h`
  comparability and `0.915 -> 0.945` runway dossier before any 30-hour allocation.
- R1 zero-shot real-`wrfeat`: `0.7637338804169099`, Branch B, allocation `0/30 T4-h`;
  edge `TP/FP/FN=9264/1200/1750`, division `TP/FP/FN/Jaccard=0/0/9/0`.
- R4 terminal: `0/4` atomic inner receipts after approximately
  `25 h 57 m`--`26 h 22 m`; no sealed winner or exact 44b6 delta. The separately
  measured tier-(a) ceiling is `+0.002406490`, below the unchanged `+0.005` gate.
- A7 separation: occupied-swap universe `45,961/134`; A1 per-fold AUC
  `0.627866/0.828496/0.610208/0.659067/0.654961`, CG0 retention `0/0/0/0/0`;
  **FAIL / PARK ON EVIDENCE**, with no threshold, edit set, or graph delta.
- Budget: approximately 255 T4-h through 2026-09-29, nominal 30 h/week reset Friday 08:00 SGT; retain roughly 25 T4-h unallocated reserve.
- Latest authenticated global Kaggle quota is `1.02/28.98 h` used/remaining,
  refresh `2026-08-08 00:00 UTC`. A v2 crossed the weekly refresh, so its exact
  old-window debit cannot be reconstructed from the current balance; observed
  remote wall time was approximately `3.08 h`, within its `4 h` amendment cap.
- Selected envelope: 140–188 T4-h including infrastructure, graph-gated training, confirmation/transfer, and package.
- Phase 1 cap: 20 T4-h (expected about 4 h across A/reference and B); A requires >=60 minutes of synchronized CUDA training, then one new-kernel resume only.
- Binding endpoint: 44b6 held-fit adjusted graph delta at fixed intervals. Owner-amended continuation gates: `>=+0.010` at 30 T4-h and `>=+0.035` at 90 T4-h. The former 60-hour `+0.002` threshold no longer controls and no replacement 60-hour gate is inferred; it requires an explicit later amendment. Loss/AUC are diagnostic only.
- Carried diagnostic surface: DELETE AUC/retention `0.85/0.020394`; occupied SWAP `0.90/0.067910`.

## Controls

- Owner lock: zero h030 launch and zero G-03 GPU spend until the pending
  ALIGNMENT patch is confirmed landed by both content inspection and SHA.
- All candidates may develop in parallel. The owner later funded only the fixed
  h005 slope read despite the Branch-B dossier failure; that amendment does not
  fund h030. Recompute the exact `best_known_score * 0.985` floor whenever best
  known rises.
- No GPU before the Phase 1 contract is hash-pinned. Training may use checkpointed sessions up to the Kaggle cap; feature/inference jobs retain the 5-hour hard kill.
- No proxy or within-trajectory checkpoint selection. Do not read the 44b6 objective until its checkpoint is fixed.
- Guards fail on category violations: sustained swapout `>=256 MiB` or another integrity failure. Step-0 time is warning-only.
- Checkpoint handoff includes optimizer, scheduler, AMP scaler, Python/NumPy/Torch/CUDA RNG, dataloader state, data manifest, code/config, and artifact hashes; it is published atomically and verified before resume.
- Every G-03 code or runtime node uses a dedicated worktree from its hash-pinned preregistration commit. No experiment reads another node's uncommitted outputs; only authenticated immutable handoff artifacts cross worktree or kernel boundaries.

## Log (newest first)

- 2026-08-03 — **OWNER LOCK RECORDED / ALIGNMENT PATCH NOT YET CONFIRMED.** No
  h030 launch and no G-03 GPU spend of any kind is allowed until the pending
  ALIGNMENT patch is confirmed landed. Current inspected ALIGNMENT SHA
  `aff9e656…abaff` lacks the lock, so all G-03 GPU actions fail closed. No
  launch or spend occurred. (Owner + Codex root)

- 2026-08-03 — **H005 V2 INVALID AT REMOTE MOUNT STEP0 / NO V3 / LANE 0
  INFRASTRUCTURE-TERMINAL.** The sole repaired version 2 ended `ERROR` after
  `3.5865 s` because the runtime dependency manifest at its exact four-source
  GPU-worker mount was missing or byte-divergent; the frozen check did not emit
  enough observed facts to distinguish them. Remote file inventory is empty.
  Dependency install, core import, CUDA allocation, data access, optimizer,
  checkpoint, held-fit36/CV57 evaluation, score, and slope are absent. V2 debit
  is `0.00 h`; current quota is `1.02/28.98 h`. No V3 is authorized. H030 is not
  evaluable or funded, without calling the trajectory flat/negative; its 25 h
  remain preserved. Diagnosis `ae433c6d…518e9`. (Codex root)

- 2026-08-03 — **SOLE DEPENDENCY-REPAIRED H005 VERSION 2 LAUNCHED.** The exact
  CPU release-path probe, V2 static audit, fresh inline guard, shared serial
  launch guard, and authoritative quota bracket all passed. Kaggle accepted
  `onebitbrain/g03-h005-train-v1` version 2 at `2026-08-02T21:32:20Z`; quota
  remained `1.02/28.98 h` immediately after push. This is the unchanged single
  candidate/core with a `17,700 s` subprocess kill and `17,940 s` wrapper cap.
  It is the only live GPU job; monitor at no more than one query/hour. No V3,
  evaluation, h030, CV57, hybrid, submission, or RunPod action. (Codex root)

- 2026-08-03 — **H005 CPU RUNTIME PROBE PASS / SOLE V2 STATIC RELEASE PASS.**
  Authenticated kernel version 1 completed on CPython 3.12 and matched all 14
  runtime carrier hashes, installed the exact 8 selected wheels offline, imported
  NumPy `2.0.2`/SciPy `1.16.3`/Torch `2.10.0+cpu`/zarr `3.2.1`/numcodecs
  `0.15.1`, and byte-matched one fixed 6bba raw frame plus GEFF graph in `9.223 s`.
  GPU, optimizer, checkpoint, held-fit36, CV57, metric, and submission access were
  zero; quota remains `1.02/28.98 h`. V2's scientific core remains byte-identical
  `8c3c9b98…b9c0e9`; a fresh inline/serial/quota guard is still required. No V3.
  (Codex root)

- 2026-08-03 — **H005 CPU RUNTIME PROBE VERSION 1 LAUNCHED.** The fresh inline
  guard and authoritative quota bracket passed; Kaggle accepted the frozen
  CPU-only one-identity release-path probe at `2026-08-02T21:22:54Z`. GPU quota
  stayed `1.02/28.98 h`. The probe has no optimizer, checkpoint, held-fit36,
  CV57, metric, graph, submission, or GPU access. V2 remains ineligible pending
  an authenticated exact probe PASS. (Codex root)

- 2026-08-03 — **H005 DEPENDENCY REPAIR FROZEN / CPU RELEASE-PATH GATE NEXT.**
  Published and authenticated private runtime carrier
  `onebitbrain/g03-h005-runtime-deps-v1` version 1: READY, 14 exact offline
  wheels, no quota change from `1.02/28.98 h`. The repair preregistration binds
  one CPU-only fixed-6bba install/import/decode probe; only an exact PASS may
  make the unchanged scientific core eligible for one hard-killed GPU version
  2. Evaluation, score, slope, h030, CV57, hybrid, submission, and RunPod remain
  closed. (Codex root)

- 2026-08-03 — **H005 V1 INVALID BEFORE TRAINING / NO CHECKPOINT OR SLOPE.**
  Authenticated STEP0 passed h000, contract, core, model, archive, cache, and
  development69 bindings, then the frozen training core import failed with
  `ModuleNotFoundError: zarr`: the launcher declared but never installed the
  canonical wheel set. Training main, CUDA model allocation, data access,
  optimizer steps, h005 publication, held-fit36 evaluation, and score are absent.
  Postterminal quota is `1.02/28.98 h`, a rounded `0.01 h` debit. V1 is
  infrastructure-invalid, not a scientific verdict; no same-signature retry,
  evaluation, h030, hybrid, CV57, submission, or RunPod action follows.
  Diagnosis `V1_TERMINAL_DIAGNOSIS.json`. (Codex root)

- 2026-08-03 — **DENSE DEFORMABLE MOTION FAILS BOTH FROZEN CLASS GATES.**
  The distinct raw-voxel TV-L1 supplier completed all 41 44b6 identities,
  4,059 frame pairs, and 1,391,458 actions, then the unchanged sealed
  B0→A1→within-frame-A0 evaluator returned `FAIL_PARK_C` for DELETE and
  occupied SWAP. DELETE showed AUC lift but retention maxed at `0.010101`
  versus `0.15`; occupied SWAP also failed. Learned displacement, graph,
  hybrid, GPU, 6bba, CV57, RunPod, and submission remain closed. Result
  `8c5d9f8c…cecac9d`. (Codex root)

- 2026-08-03 — **FIXED H005 KERNEL VERSION 1 LAUNCHED / SERIAL GPU LOCK HELD.**
  Fresh inline guard and quota reconciliation passed at `28.99 T4-h` remaining;
  the single sealed `onebitbrain/g03-h005-train-v1` push succeeded at
  `2026-08-02T19:33:09Z`. An account-wide OAuth private-file endpoint 403 was
  diagnosed using both the new and a previously working carrier, so the
  unchanged top-level remote STEP0 is the canonical integrity boundary: it
  verifies h000, contract, training core, base model, archive, and cache hashes
  before heavy work and fails without retry. No second GPU job, submission,
  RunPod, CV57 access, h030 funding, or hybrid action occurred. (Codex root)

- 2026-08-03 — **ORTHOGONAL-LANE AMENDMENT ACTIVE / H005 SEALED PRELAUNCH /
  TRAINED FORK CROSS-FAMILY FAIL.** Frozen standalone preregistrations for h005,
  classical dense motion, and a trained raw-intensity fork detector. The fork
  probe trained once on 6bba with 44b6 validation from first fit and failed its
  required recall/precision class; final 38 access, GPU, CNN, and graph integration
  stopped closed. H005's 69/69 development join and end-to-end gradient preflights
  pass; corrected private contract carrier version 2 and inline launch guard pass,
  but no GPU kernel has yet launched. Dense motion remains a running CPU screen.
  No hybrid, submission, RunPod, CV57 read, or h030 action occurred. (Codex root)

- 2026-08-03 — **PHASE-2 ZERO-INIT H000 EXACT PARITY / BRANCH B DOSSIER FAIL /
  H005 NOT FUNDED.** Published immutable h000 `603e02f8…18f95` with exact-zero
  `Linear(48,12)` weight/bias/output and trainable projection gradients. A
  score-blind CPU/CUDA threshold-crossing diagnosis led to canonical decoding
  from the frozen CUDA banks; all 36 graphs match the zero-feature reference
  byte-for-byte. Corrected held-fit36 score is `0.7630876414338513`, not the
  different-population CV57 `0.829194058080`; random-adapter R1 is only
  `+0.0006462389830586` higher on the controlled population. Branch B remains;
  audited primary effects and budgets do not support `<=30 T4-h` comparability
  or calibrated `0.915/0.945` paths. Allocation stays `0/30`, h005/h030 unrun,
  GPU/CV57/submission/RunPod actions zero. (Codex root)

- 2026-08-03 — **A7 SEPARATION FAIL / EVIDENCE PARK / CONDITIONAL GRAPH CLOSED /
  GATE AMENDMENT DRAFTED BUT NOT APPLIED.** After R4's terminal kill released CPU,
  the exact inherited five-fold separation protocol scored the frozen 45,961 A7
  occupied swaps with 134 positives. B0 matched its historical pre-A7 seal exactly;
  A1 added only the original phase/template `action_margin`; A0 permuted that vector
  within video/source-frame. A1 AUC by fold was
  `0.627866/0.828496/0.610208/0.659067/0.654961`, retention was zero throughout,
  A1-B0 lift was `+0.002324/-0.008582/-0.012093/-0.005730/-0.001053`, and A1-A0
  lift was `+0.001574/-0.006478/-0.011753/-0.005457/-0.001522`. Every fold fails
  the occupied-swap `0.90/0.067910` bar and lift clauses, so A7 is evidence-parked;
  no threshold, edit set, graph score, edge/division term, CV57, GPU, or submission
  opened. Result `3143a31d…f96d`; terminal `646df32d…4ca1`. The owner-only node-gate
  amendment package is draft status only (`cfca3ce3…0803`) and explicitly changes
  zero historical verdicts. R1 remains `0.7637338804169099` versus floor
  `0.868299238910600`, Branch B; GPU allocation/spend `0/30 h`, best
  `0.881522069960`. (Codex root)

- 2026-08-03 — **R4 inner folds 0--3 KILLED / FAILED CLOSED / A7 sequencing
  released.** Each fold completed all 18,145 fixed planning rows before scorer
  access (`13,078/13,049/13,498/13,135` full decodes;
  `984,897/1,003,071/1,002,622/984,840` action-set reuses), but the four
  CPU-saturating unique-graph scorer processes were killed before any atomic
  receipt, seal, or diagnostics publication. Observed wall time from per-fold spool
  creation to terminal process absence was `25:56:56` for folds 1--3 and
  `26:21:57` for fold 0, making the 18,145-configuration lattice non-viable as a
  screening instrument. A post-failure recovery check read only SQLite schema and
  aggregate row counts—no graph identity, score, or metric value—so the transient
  spools remain quarantined and scientifically inadmissible. R4 yields no score,
  edge/division terms, winner, or gate change. Therefore no exact 44b6 winner
  comparison exists; the already measured tier-(a) ceiling `+0.002406490` (1/38)
  nevertheless sat below the unchanged `+0.005` promotion gate before this sweep
  was built. R1 remains
  `0.7637338804169099` versus floor `0.868299238910600`, Branch B, GPU allocation
  and spend `0/30 h`, best `0.881522069960`. Per the owner sequencing rule, A7 may
  now start without R4 contention. Runtime receipt:
  `wave_r/R4_RUNTIME_KILL_RECEIPT_20260803.json`, SHA
  `c46ec5dc…f06`. (Codex root)

- 2026-08-01 — **R4 final independent pre-score PASS / fixed development OOF open.**
  Independent artifact `0b7361ca…b3652` rechecked the entire V4 hash packet,
  reproduced the Wave-R validator and 86/86 suite, byte-checked all 18,144 active
  rows, dynamically proved a V3 cache receipt fails before graph load, confirmed the
  new V4 cache has zero entries, and passed actual-worst wall/RSS/swap plus all
  zero-access gates. Development target/scorer access is now authorized only through
  the fixed nested/grouped runner; no result has yet been read. R1 remains
  `0.7637338804169099` versus floor `0.868299238910600`, Branch B; GPU allocation and
  spend `0/30 h`, best `0.881522069960`; R4 division and edge terms remain unmeasured
  until the first sealed inner receipt. (Independent audit + Codex root)

- 2026-08-01 — **R4 V4 semantic/runtime chain resealed / validator + 86 tests PASS /
  final independent audit active.** Amendment V2 `bd5a3af7…ecabcf` explicitly
  supersedes the node-only/sequential-install contract and binds decoder
  `a58c4964…521c0`, runner `cc37fc63…de304`, orchestrator `853e11a9…b22d7`,
  V4 plan/input `ecbb7467…665a4`/`be6f7a4e…6ee23`, frozen compiler/model hashes,
  and new empty `graph_cache_v4` provenance; old V3 cache bytes remain immutable and
  inadmissible. Receipt `7b2ff492…006b5`, the Wave-R validator, and 86/86 relevant
  tests pass without scorer/GT/metric/archive-member access. R1 remains
  `0.7637338804169099` versus `0.868299238910600` floor, Branch B, GPU allocation and
  spend `0/30 h`, best `0.881522069960`; R4 division and edge terms remain unmeasured.
  Development scorer stays closed pending the final independent pre-score PASS.
  (Codex root)

- 2026-08-01 — **R4 amended actual-worst runtime PASS / all-grid differential PASS /
  scorer still closed for reseal.** After preregistering removal of perturbing Python
  allocation tracing while retaining the owner-requested hard-wall/RSS gates, exact
  worst-case production decode completed in `3.648069624963682 s` (external guard
  `17.668541667051613 s`), peak RSS `582025216` bytes, swap growth `0`, with
  42,235 raw/11,775 accepted/92 conflict-rejected events, one full graph
  construction/hash, and exact direct-edge algebra. Result `fc479872…a3be9` records
  zero scorer/GT/target/archive/metric/cache/network/write access. Independent
  `c3b3934a…317c8` audits all 18,144 active rows, index/full-scan equivalence,
  stale-conflict and cross-category ownership, caps, and byte-exact replay. Current
  score remains R1 `0.7637338804169099` versus floor `0.868299238910600` (Branch B);
  allocation/GPU spend `0/30 h`, updated best `0.881522069960`; R4 division and edge
  terms remain deliberately unmeasured until reseal plus independent pre-score PASS.
  (Codex root + independent audit)

- 2026-08-01 — **R4 amended actual-worst hard-wall gate FAIL-CLOSED twice / scorer
  remains closed.** The initial amended decoder and a second exact temporal-index
  optimization both reached the external `60 s` kill on sealed target-free
  `train:6bba_57b7cc1e`; neither worker returned a trustworthy receipt, so the `<30 s`
  decode, `<512 MiB` tracemalloc, `<1 GiB` RSS, and `<256 MiB` swap gates are not
  established. No result artifact, target/GT/scorer/metric/archive-member access,
  cache publication, or configuration search occurred. A bounded target-blind stage
  diagnosis is active before any further binding rerun. (Codex root + runtime audit)

- 2026-08-01 — **R4 semantic/runtime amendment explicitly owner-authorized / implementation
  active / scorer still closed.** Authorization preserves all 18,145 configuration names,
  values, folds, selection rules, and gates while permitting prediction-changing deterministic
  frozen-baseline incident-conflict ownership inside the ranked acceptance loop, a once-built
  source/target incident index, and corresponding one-pass compiler/graph-reuse/receipt changes.
  The prior implementation amendment must be superseded and the complete source/plan/input/cache
  chain resealed. Development target/scorer/GT/metric/archive-member access remains forbidden
  until exact differential, stale-conflict, cross-category, sealed-source, actual-worst
  hard-wall/RSS, regression, and independent pre-score gates pass. (Owner + Codex root)

- 2026-08-01 — **R4 all-grid target-free conflict audit BLOCK / explicit owner
  authorization required.** Reproducible audit `0ba158be…59a3f` processed all 18,144
  active rows and found 1,350 sequential-invalid, topology-affected rows, all uncapped
  on the sealed worst identity; 1,269 change accepted count and 81 change topology at
  equal count. The representative row changes 11,829→11,775 under integrated ownership
  (92 conflict rejects, 36 admitted replacements), while the preliminary 11,749
  post-filter is provably non-equivalent. Actual sequential compilation reproduces the
  stale-edge `ActionCompileError`. Script `cc8c9462…1dd65` reproduced the result
  byte-exactly; target/GT/scorer/metric/archive-member access remained zero. Production
  and scorer access remain closed pending the exact owner-authorized semantic amendment,
  reseal, hard-wall/RSS and differential gates, and independent PASS. (Independent
  audit + Codex root)

- 2026-08-01 — **R4 V3 independent structural PASS / one-pass compiler repair
  preregistered / scorer remains closed.** Audit validated all 270 content-addressed
  cache entries, 75/75 exact tests, the V3 source/plan/input chain, and corrected
  benchmark provenance (scorer modules imported, but backend/metric/archive-member/GT
  access zero). It blocks fold0 because an 11,829-fork sequential decode implies
  about 35,487 full graph copies/validations/hashes and retained intermediate graphs.
  A synthetic one-pass prototype is fast but remains non-production until coherent
  frozen-API binding, actual sealed worst-case and cache differentials, compiler-path
  amendment/reseal, and independent re-audit pass. A separate 5-minute large-video
  diagnostic stopped before materialization because incident-conflict discovery
  still scanned the full parent edge set per action. (Independent audit + Codex root)

- 2026-08-01 — **R4 preliminary actual worst-set conflict audit BLOCK.** Target-free
  audit `0c27d458…2224d` reproduced 42,235
  raw and 11,829 accepted events for `t035_g100_l0_dp0_gp0_n0_call` on
  `train:6bba_57b7cc1e`, then found 85 shared removed edges. Ranked edge ownership
  post-filtered 80 later events and retained 11,749, establishing the stale-edge
  failure. The later all-grid integrated-acceptance audit supersedes 11,749 as a repair
  prediction because a post-filter cannot admit replacements. Target/GT/scorer/
  archive/metric access remained zero. (Codex root + R4 executor)

- 2026-08-01 — **R4 V3 exact factorization sealed / small benchmark PASS / large
  compiler runtime gate closed.** Runner `2b9cb30e…7284` preserves all 18,145 rows,
  late-window semantics, and trusted per-config suppression while doing one full
  decode per exact action topology. V3 plan/input are `da2bfa61…d96d` and
  `bc815caf…0322`; receipt `8cbc2727…8109`, 75/75 tests, and validator pass. The
  exact smallest real video completed target-blind in `76.084 s` with 47 full
  decodes and 18,098 reuses; verified scorer modules imported during input loading,
  but backend/metric/archive-member/GT access remained zero. The audited 55-video projection nevertheless exposes
  uncapped topologies containing 19–11,829 forks; the frozen sequential compiler is
  not wall-bounded there, so fold0 scorer/GT remains unopened. Cache now holds 270
  valid target-free graphs; no metric, outcome, held/audit/hidden, GPU, or submission
  access occurred. (Codex root + independent audit)

- 2026-08-01 — **R4 conservative factorization boundary frozen / target still closed.**
  A proposed shortcut that would canonicalize the apparently inert late-window axis
  was not admitted to production. The active implementation retains every sealed
  late/duplicate/cap configuration, reuses only field-independent trusted decoder
  work and exact action-set-equivalent topology, and requires differential graph-SHA
  checks plus immutable target-free cache bindings. No archive member, backend,
  scorer, metric, outcome, held/audit/hidden, GPU, or submission access occurred.
  (Codex root + independent audit)

- 2026-08-01 — **R4 naive decoder runtime falsified / exact factorization active.**
  The independently cleared V2 fold0 runner passed all prior guards, but its
  target-free spool reached only 4/18,145 configs after several minutes; the
  unoptimized per-config full-graph rebuild projected roughly 15 days per fold,
  contradicting the CPU-cheap design intent. Root interrupted it after the required
  opaque archive hash verification but before any archive member, backend, scorer,
  or metric access. No output or metric
  spool survived; 227 target-free content-cache entries are reusable. The fixed
  lattice, folds, gates, and reference points remain unchanged while exact
  action/config equivalence and differential tests are developed. (Codex root)

- 2026-08-01 — **R4 V2 runner pre-score re-audit PASS / dev OOF reopened.**
  Reviewer independently reproduced 18,145 unique configuration names and the
  sealed decoder/orchestrator digest `f48e8236…f2e1`; V2 differs from V1 only in
  the repaired runner/orchestrator bindings. Both failed starts created no output
  or graph cache and stopped before archive verification/open, decode, backend, or
  scorer. 73/73 tests and validator pass; held/audit/hidden remain closed.
  (Independent audit + Codex root)

- 2026-08-01 — **R4 runner config-digest repair / 73 tests PASS / re-audit active.**
  Fold0 attempt 1 failed before target access because the system test interpreter
  lacked `zarr`; attempt 2 used the production `.venv313` but failed before decode
  because the runner's newline-terminated generic JSON digest did not reproduce the
  decoder/orchestrator no-newline configuration-name digest already sealed in the
  plan. Exact membership remained 18,145/18,145. Runner `2196a84d…5f0` now delegates
  this binding to the decoder canonicalizer; V2 plan/input are `5e357d02…10b5` and
  `c629e5cd…6485`, receipt `14a581b0…dfb6`, 73 tests and validator pass.
  Development scoring is paused for independent re-audit; decoded graphs, scorer/GT,
  outcomes, held/audit/hidden, GPU, and submissions remain zero. (Codex root)

- 2026-08-01 — **R4 final production-chain pre-score audit PASS / dev OOF open.**
  Independent recomputation verified all seven final artifact hashes and all 246
  unique target-free local105 files: exact 69 dev plus 36 held, one R1 config,
  coverage `aa393a71…cbc7`, no audit/hidden identities, exact runner/scorer source
  dependencies, and opaque archive binding. The two failed generator attempts are
  legitimate zero-publication ordering/path repairs. 72/72 tests and validator
  pass; no scorer/GT/outcome/archive member was opened. Development nested OOF is
  authorized; held36/hidden4/audit20 remain closed. (Independent audit + Codex root)

- 2026-08-01 — **R4 dev69/local105 sealed / prepublication generator repair.**
  Exact target-blind dev69 completed 69/69 with bank/parent aggregates
  `12a4fbe4…f4fc0`/`56b9e320…6763`, zero predicted forks, zero GPU, and seal
  `588397b1…e6d8`. The first two local105 generator attempts published nothing:
  one overconstrained incidental R1 row order and one duplicated the already-44b6
  APNCC path. Membership/path repairs are test-covered; exact local105, pretarget,
  and runner inputs now seal as `aa393a71…cbc7`, `ee8ee931…72d5`, and
  `c86fd43b…dae9`. Receipt `92a4f3fc…4e95`, 72 tests, and validator pass;
  production scoring waits for post-repair independent re-audit. No scorer/GT,
  outcome, archive member, held/audit/hidden, GPU, or submission access occurred.
  (Codex root + R4)

- 2026-08-01 — **R4 promoted-tree pre-score re-audit PASS.** Exact repository
  runner identity is closed at the pretarget plan, target-blind generator, and
  inner/outer receipt layers; path, hash, and byte substitution fail closed. The
  reviewer reproduced 72/72 tests and the Wave-R validator. local105 binds only
  target-free dev69 plus held36 banks/parents/APNCC predictions; no scorer, GT,
  outcome, audit20, hidden4, or archive member was accessed. Development scoring
  remains closed only until bank/manifests seal. (Independent audit + Codex root)

- 2026-08-01 — **R4 exact-runner repair promoted / 72 tests PASS / re-audit active.**
  Pretarget plans now bind the exact repository runner path and final runner bytes
  `357dc028…c5620`; inner/outer receipts reject path, byte, or hash substitution.
  Orchestrator `4a6f967b…9cd53` and target-blind local105 generator
  `3908f035…12dab` are root-pinned and the Wave-R validator passes. Production
  scoring/GT and held/audit/hidden remain unopened pending independent PASS; GPU,
  allocation, division outcome, and best are unchanged. (Codex root + R4)

- 2026-08-01 — **R4 dev69 target-blind bank extension reaches 60/69.** The
  corrected exact-runtime process remains uninterrupted with no anomaly, resume,
  overwrite, scorer/GT access, held/audit/hidden access, or GPU action. Allocation,
  branch, division outcome, and best remain unchanged. (Codex root + R4)

- 2026-08-01 — **R4 re-audit keeps scoring/held blocked on runner identity binding.**
  The machine scientific gates pass review, but receipt validation accepted any
  self-attested runner file instead of requiring the promoted repository runner
  `357dc028…c5620`. A substituted runner could fabricate a consistent five-fold PASS.
  Repair must bind exact runner path/hash in both inner and outer receipts and reject
  path/byte substitution. No production score/GT or held/hidden access occurred; bank
  inference may continue target-blind. (Independent re-audit + Codex root)

- 2026-08-01 — **R4 pre-held gate repair promoted / 62 tests PASS / re-audit active.**
  Orchestrator `deee34b6…f5097f` and CPU runner `357dc028…c5620` now bind same-video
  parent/candidate graphs and metrics, official pooling, per-fold GT division counts,
  explicit deltas, all five outer-result hashes/configs, and a recomputed machine PASS
  before any held marker. Combined score admits the official `1.1` ceiling and forged
  PASS bypass tests fail closed. No production scorer/GT/outcome was opened; held36 and
  hidden4 remain blocked pending independent re-audit. (Codex root + R4)

- 2026-08-01 — **R1 post-hoc helper provenance overstatement repaired.** Independent
  audit confirmed the recovery/finalizer helpers were preserved but omitted from the
  initial promotion hash set, so the human report could not call them pre-pinned.
  Exact bytes are now retained as `b1ca85fc…19b82c6` and `0b1afb0f…0520654`; amendment
  `f5c3084e…34bd4f` explicitly says this is post-hoc and does not retroactively pre-pin
  them. Raw/final metrics reproduce, no rerun/target access occurred, and R1 decision,
  allocation, GPU spend, and best are unchanged. (Independent audit + Codex root)

- 2026-08-01 — **Independent post-R1 audit blocks held36 pending R4 gate repair.**
  Allocation/score/hashes recompute cleanly, but the pre-target orchestrator accepted
  candidate-only outer receipts and could prepare a held marker without mechanically
  proving same-video parent deltas, positive per-fold GT divisions, and all development
  gates. Held36/hidden4 remain unopened. The orchestrator/runner repair must bind parent
  metrics and construct a machine PASS gate from all five outer results; combined-score
  validation must also permit the official maximum `1.1`. No outcome, GPU, submission,
  or best changed. (Independent audit + Codex root)

- 2026-08-01 — **R4 dev69 edge-API guard repaired before publication.** The first two
  launches sealed `0/69`: the system interpreter lacked Torch, then the correct runtime
  exposed that the target-blind divisions-off guard used `source/target` instead of the
  actual immutable `SubmissionEdge.source_id/target_id` API. The latter failure occurred
  after first-video inference but before atomic bank/parent publication; no scorer/GT was
  touched. Regression coverage now passes 43/43, corrected runner SHA is
  `7cc4c7d2…e0a9c`, and new target-blind preflight is `960e3c61…c4466`; scientific
  configuration and GPU/submission/best remain unchanged. (Codex root + R4)

- 2026-08-01 — **R1 Branch B final / no GPU allocated / R4 CPU execution dispatched.**
  R1 scored `0.7637338804169099` versus floor `0.868299238910600`; edge
  `9264/1200/1750`, division `0/0/9/0`. R2 Branch-B evidence remains FAIL, so no
  candidate receives the 30-hour allocation and Wave-R GPU spend stays `0`. Full
  validation passes, root promoted the 36 banks and issued score-free R4 receipt
  `c5713428…acc`; target-blind dev69 preflight `3b3ba581…d0ae7` passed and CPU
  inference is active. Best remains `0.881522069960`; division outcome/submissions are
  pending/zero. (Codex root + R1/R4)

- 2026-08-01 — **R1 36/36 pre-score seal complete / receipt serialization incident.**
  Exact aggregates are bank `886fbd3a…b4bd` and parent graph `e7a2aaf1…dbd6`.
  The frozen scorer completed its first GT/metric pass, but strict JSON rejected a
  legitimate per-video division-Jaccard `NaN`; no score value or artifact was exposed.
  Because the preregistered no-retry line begins after first official score exposure,
  one incident-recorded identical-input recovery may stringify only that non-finite
  diagnostic. It must report two target-access attempts, one scientific configuration,
  and one eventual score exposure; a second failure is terminal. GPU/submission spend
  remains `0` and the best is still `0.881522069960`. (Codex root + R1)

- 2026-08-01 — **R4 target-blind execution stack promoted / R1 34/36.** The decoder
  preserves the parent node set, installs atomic N1 edge actions deterministically,
  enforces the corrected dev69/held36/hidden4 access order, and refuses audit20.
  Pre-outcome amendment `34fcd826…183e` freezes the persistence meanings,
  `compile_action` installer, and literal DIRECT-before-LATE equivalence (which can make
  `late_window` output-inert) before any development outcome.
  The dev69 extension now requires a root R1-promotion receipt, seals the exact local
  105 target-free banks before scoring, and makes hidden4 inaccessible until local PASS;
  its preflight binds its own implementation hash. The OOF orchestrator seals each
  inner winner before outer evaluation, uses modal-count/lex deploy selection, and
  gates held36/hidden4 one-shot markers. Combined Wave-R tests pass 42/42
  and prerequisite validation remains PASS with only
  R1 pending. R1 remains score/GT-blind; GPU/submission spend stays `0`, division
  outcomes remain unopened, and best remains `0.881522069960`. (Codex root + R4)

- 2026-08-01 — **Independent Wave-R pre-outcome audit PASS.** Allocation arithmetic,
  citation/metric boundaries, population integrity, deterministic selection, and G-02
  authority separation pass with R1 pending. Prospective guards remain binding: seal
  all dev69 bank/parent/reference hashes before development scoring, then bind validated
  R1 score/graph SHA plus chosen config and parent/overlay hashes before the one-shot
  held-fit marker. No score, GPU, submission, or best changed. (Independent audit)

- 2026-08-01 — **R2 citation audit repaired / Branch-B verdict unchanged.** The
  Attrackt interface is now described correctly: crop embeddings are global-solver
  costs or frozen unsupervised targets, not injected Trackastra `wrfeat`. A ninth
  primary source, Chen et al. 2025, adds zero-shot SAM2 Fluo-N3DH-CE LNK `0.982` on an
  RTX 4090 and identifies an observation-distinct future candidate, but no G-03 metric
  conversion or present-Trackastra 30-hour headroom. R2 remains Branch-B FAIL; no score,
  GPU, submission, or best changed. (Codex root + independent audit)

- 2026-08-01 — **R4 population blocker repaired before any division outcome.** Audit
  proved R1-36 alone cannot instantiate nested OOF and would contaminate the held-fit
  endpoint; it also proved legacy audit20 is historically exposed. Population SHA
  `e25416da…425954` now freezes five target-blind grouped folds over exact fit105
  `6bba` development `n=69`, one non-selecting held-fit `44b6` confirmation at `n=36`,
  and a separately gated hidden-test daily probe over `n=4`; audit20 target access is
  forbidden. Local identical-config bank coverage requires 105 videos before OOF;
  hidden-test inference expands eventual coverage to 109 only after the local gate.
  Validation passes and tests
  are 11/11; no division target/outcome, GPU, submission, or best changed. (Codex root
  + independent audit)

- 2026-08-01 — **R1 preflight PASS / score-blind inference active / exact-floor
  amendment sealed.** Contract `aefe438e…50204` passed archive, checkpoint, source,
  cache, scorer, and adapter checks. After 1/36 videos had sealed, audit found its JSON
  float displayed the floor rounded to `0.868299238911`; a pre-score decision-only
  amendment now makes `0.868299238910600` authoritative and leaves the sole adapter,
  model, compiler, graph, and bank bytes unchanged. No GT/score read, GPU, submission,
  division outcome, or best change occurred. (Codex root + R1)

- 2026-08-01 — **Wave R independent-audit repairs complete / protocol sealed.** No
  allocation-invalidating issue was found. The displayed floor is now the full exact
  `0.868299238910600`; the stale G-02 native ledger row is terminally reconciled; and
  R4 now retains its full manifest (`656bb6fa…c83a`) plus a protocol/dependency contract
  (`3b7582eb…c6d02a`) binding nested OOF and one-shot access before any division
  outcome. Validation remains PASS and focused tests remain 8/8. Spend is `0 T4-h`;
  score/division terms remain pending; best is `0.881522069960`. (Codex root + audit)

- 2026-08-01 — **Wave R prerequisite validation PASS / R1 pending.** The exact
  allocation arithmetic, owner controls, R2 Branch-B fail (then eight sources; now nine
  after citation-audit repair), R3 real-`wrfeat`
  absence, and the R4 `18,145`-configuration hash reproduce under the promoted
  validator; combined focused tests passed 8/8 at that checkpoint (now 11/11 after
  population-contract coverage). Candidate score/branch and division
  metrics remain pending, spend remains `0 T4-h`, and best remains `0.881522069960`.
  (Codex root)

- 2026-08-01 — **Wave R2 literature audit complete / Branch-B dossier FAIL.** Primary
  sources establish strong related-domain 3-D embryo LNK/TRA endpoints, but those
  AOGM-family metrics do not convert to G-03's scorer. No source validates the new 3-D
  crop adapter, comparability within `<=30 T4-h`, or runway to `0.915` and `0.945`.
  Therefore a below-floor R1 result is ineligible for the 30-hour allocation; an
  at/above-floor result is evaluated under Branch A. No GPU, score, division term,
  submission, or best changed. (Codex root + R2)

- 2026-08-01 — **Wave R3 reusable-asset audit complete.** N1, frozen official scorer,
  APNCC/Trackastra controls, native bank, raw archive, transport, 38 repair bundles,
  FP curve, and requirement surface have exact G-03 plug points. No bank contains real
  12-D `wrfeat`; R1 therefore requires fresh raw-crop inference. Stale/unsafe feature and
  CV57 assets are explicitly excluded. No score, division term, GPU, submission, or best
  score changed. (Codex root + R3 explorer)

- 2026-08-01 — **Wave R4 division-decode design complete / no score access.** The
  promoted design fixes 18,144 active configurations plus `DIVISIONS_OFF`, exact
  equivalence/node-ownership suppression, grouped five-fold nested OOF, and one sealed
  audit20 access; generated configuration hash is `fe72a4c9…4adc` and tests pass 3/3.
  Division TP/FP/FN/Jaccard remain not evaluated, the best remains `0.881522069960`, and
  R4 consumed `0 T4-h`. (Codex root)

- 2026-08-01 — **Owner Wave-R autonomy/policy encoded; R1–R4 dispatched in isolated
  worktrees.** Exact floor is `0.868299238910600` from banked `0.881522069960`; R1's sealed
  zero-shot score is the GPU-allocation blocker, while literature, asset, and division
  protocol lanes run concurrently. Candidate branch and division terms remain pending;
  Wave-R consumed `0 T4-h`, updated best remains `0.881522069960`, and no Kaggle mutation,
  submission, CV57, final-slot, or RunPod action occurred. (Owner + Codex root)

- 2026-08-01 — **PARK-B rationale superseded on its gate arithmetic; the park itself is unchanged.**
  The 2026-08-01 PARK-B entry below reasoned from the then-current continuation ladder
  (`>0` at 30 T4-h, `>=+0.002` at 60, `>=+0.005` at 90) and concluded that a full pass at
  90 T4-h lands near local `0.8865`. That entry was correct when written but is **not
  current**: the owner's later gate amendment (this log, same date) raises the binding
  lines to `>=+0.010` at 30 T4-h and `>=+0.035` at 90 T4-h, with the 60-hour threshold
  superseded and no replacement inferred. The `+0.005`-based economics in that entry
  therefore understate the campaign's own success target by roughly `7x` and must not be
  cited. Note the delta is a 44b6 held-fit adjusted graph delta, not a CV57/local combined
  score, so it does not translate one-to-one onto the banked `0.881522069960` — this node
  measured the 36-video 44b6 baseline at `0.866534526036`. The owner's PARK-B decision
  pending expanded RunPod budget stands until the owner revisits it; only its recorded
  arithmetic is corrected here. (Claude)

- 2026-08-01 — **Phase 1 B v2 PASS / Phase 1 complete.** `onebitbrain/g03-phase1-resume-b-v2` version 1 reached remote `COMPLETE` after its single push and used `1.01 T4-h`. Downloaded artifacts independently validate as `PASS_B_BIT_IDENTICAL_AND_44B6_EMITTED`: resumed step `1,953,711`, exact final digest `816acee4…c80387b`, handoff tree `73a6cc56…b56ac`, zero swap growth, and 36-member structural deltas exactly `0/0/0`. The receipt explicitly says no official frozen graph objective was evaluated and selection is unauthorized. Preserve the unused `6.99 h` of the owner ceiling; Phase-2 preparation opens, while scientific training/CV57/submission/final-slot actions remain closed. Report: `PHASE1_B_V2_TERMINAL_REPORT_20260801.md`. (Codex root)

- 2026-08-01 — **B v2 version 1 launched exactly once / monitor-only.** Private handoff dataset version 2 is READY and its downloaded runtime tree is byte-identical to A (`73a6cc56…b56ac`) with the required `handoff/` prefix. The frozen B package passed isolated self-test and fresh exact-byte push guard, then was pushed under the serial/quota bracket; immediate quota stayed `0.00/30.00 h`. Phase 2, CV57, submission, and the final slot remain closed; any B mismatch is terminal with no v3. (Codex root)

- 2026-08-01 — **Handoff carrier layout repaired before GPU launch.** The first dataset-create command returned process status 0 while reporting it skipped the only directory and created nothing; CMUX now treats Kaggle mutation error text as failure. Explicit archive version 1 then exposed atomic members at dataset root, not under frozen B's required `handoff/` path. No GPU or quota was consumed. A one-level wrapper produced READY dataset version 2 with the exact required remote paths and full byte-identical download round-trip. The new KFG-08 incident and prevention are registered. (Codex root)

- 2026-08-01 — **Owner gate/cost amendment recorded before B v2.** Future scientific continuation now requires held-fit adjusted graph improvement `>=+0.010` at 30 T4-h and `>=+0.035` at 90 T4-h. The old 60-hour threshold is superseded without inventing a replacement. Up to `8 T4-h` is allowed for the present B-v2 continuation only where justified, but the frozen deterministic B remains a roughly one-hour test and unused allowance must be preserved. The owner also requested a complete unexplored/PARK-A direction audit. No external action or GPU debit occurred in this decision step. (Owner + Codex root)

- 2026-08-01 — G-03 A v2 is remotely COMPLETE and its downloaded atomic handoff independently verifies exactly: `PASS_A_HANDOFF_READY`, `3,600.000229 s` synchronized T4 segment, step `1,953,711`, zero swap growth, checkpoint `8a6c2867…00e134e`, tree `73a6cc56…b56ac`. Current quota refreshed to `0.00/30.00 h`; the old-window exact debit is not inferable across refresh. Existing authority now permits immutable handoff publication and B v2 only; Phase 2/CV57/submission/final slot remain closed. Report: `PHASE1_A_V2_TERMINAL_REPORT_20260801.md`. (Codex root)

- 2026-08-01 — G-02 completed its sole selected-only CV57 measurement locally:
  native `0.829194058080` versus APNCC `0.881522069960`; zero predicted forks
  made the hybrid an exact APNCC no-op. Independent audits pass, no GPU/Kaggle/
  submission action occurred, and G-03 Phase 1 remains unchanged. (Codex root)
- 2026-08-01 — G-02 sealed a local-only path to deliver its still-unused CV57
  measurement from already-downloaded bytes. Fit-only CPU↔T4 selected-graph parity
  passed 3/3 across both families and threshold stress; the preaccess runner loaded
  the exact model without CV57 cache/GT access. Remote v2-last remains binding and
  no Kaggle, submission, RunPod, G-03 gate, or G-03 artifact changed. (Codex root)
- 2026-08-01 — G-02 final CV57 v2 failed Step 0 before any cache/model/GT/graph/
  scorer access because none of three exact kernel-source cache roots existed.
  Measurement count remains zero, but v2 was the last repair, so G-02 closes
  without native/hybrid CV57 scores. Global quota is `7.20/22.80 h`; concurrent
  A v2 makes the `0.03 h` movement attribution ambiguous. G-03 is unchanged.
  (Codex root)
- 2026-08-01 — G-02 final selected-only CV57 v2 accepted exactly version 1 and
  first authenticated status is RUNNING. Its push bracket was `7.17/22.83 h`
  with zero immediate debit. It runs independently beside G-03 A v2; no G-03
  gate, output, authority, or phase changed. (Codex root)
- 2026-08-01 — G-02 reconciled v1 as a zero-access Step-0 failure and prepared
  its final v2 infrastructure-only repair. The code changes only exact allowlisted
  kernel-source mount resolution; scientific-function AST parity, full 57-video
  manifest validation, static+inline guards, and cmux tests pass. V2 is unpushed;
  G-03 A v2 remains independently RUNNING and unchanged. (Codex root)
- 2026-08-01 — G-02 selected-only CV57 v1 terminalized before heavy work with
  an absent exact input mount; zero fragments and zero CV57 measurements exist,
  and its no-relaunch rule closes that exception. G-03 A v2 remains independently
  RUNNING with no gate or authorization change. (Codex root)
- 2026-08-01 — The quarantined G-02 exception launched its fresh selected-only
  CV57 version 1 after a serialized guarded push; it now runs concurrently with
  A v2 as the owner requested for the expiring window. The fixed G-02 kernel has
  no training, alternative bank, submission, or G-03 effect. Push bracket was
  `7.00/23.00 h`, immediate debit `0.0`. (Codex root)
- 2026-08-01 — G-02 narrowed its unpushed package to selected-only graph output
  and re-passed both launch guards; its frozen scope and G-03's lane ownership are
  unchanged. No G-02 or G-03 remote action followed. (Codex root)
- 2026-08-01 — G-02's isolated one-shot CV57 package is locally ready and both
  launch guards pass, but it remains unpushed while G-03 A v2 owns the shared GPU
  lane. No G-03 gate, result, handoff, or authorization changed. (Codex root)
- 2026-08-01 — Quarantined G-02 completed and independently replayed its frozen
  108-name fit sweep, then sealed `r08_c08_a035_off` before CV57. Fit score is
  `0.811024400045` versus APNCC `0.860516437125`; divisions are OFF and the
  hybrid equals APNCC. The authorized one-shot CV57 action remains isolated from
  G-03 and cannot bypass the shared launch controls while A v2 runs. (Codex root)
- 2026-08-01 — **Owner PARK-B on the expiring-quota parallel dispatch / no spend.** The owner
  reviewed `dag_prompts/2026-08-01-expiring-quota/CODEX_PARALLEL_DISPATCH.md` (Lane A
  `G03-CROPBANK-CALIB-V1` calibration kill-test; Lane B `G03-DENSEEMB-ASSOC-PROBE-V1`) and
  parked both at **PARK-B pending an expanded RunPod budget**. The §0 owner amendment was
  never granted: the serial lock was not relaxed, no preregistration was hash-pinned, no
  kernel was pushed, and **0 GPU-h were spent**; remaining window quota is allowed to
  expire. Decision basis, recorded so it is not re-litigated: G-03's own continuation gates
  (`>0` at 30 T4-h, `>=+0.002` at 60, `>=+0.005` at 90) imply that a full pass at 90 T4-h
  lands near local `0.8865`, whereas `0.91` needs `+0.0285` and the charter target `0.9255`
  needs `+0.044` — i.e. `5.7x` and `8.8x` the 90-hour success threshold — against a
  140–188 T4-h envelope inside roughly 255 T4-h remaining to the 2026-09-29 deadline. This
  parks the funding decision, not the architecture selection; `G03-TA-CROPADAPTER-E2E-V1`
  remains the selected Phase 0 architecture and the crop-adapter cost factor remains
  unmeasured. Phase 1 A v2 is unaffected and continues under its own charter. (Owner + Claude)

- 2026-08-01 — Second authenticated A v2 status remains `RUNNING` (receipt SHA `2d685ae2…7813`), confirming the replacement passed the original immediate import boundary. It is still an infrastructure run, not a score or a handoff result. (Codex root)
- 2026-08-01 — First authenticated post-push status for A v2 is `RUNNING` (receipt SHA `81a3ffcf…a8c1b`). No terminal outcome, handoff, B action, graph result, or selection evidence exists. (Codex root)
- 2026-08-01 — A v2 (`onebitbrain/g03-phase1-resume-a-v2`) kernel version 1 was pushed exactly once from the self-contained hash-pinned package under the shared serial lock. Authoritative GPU quota was `5.43/24.57 h` before and after (immediate debit `0.0`); action receipt SHA `9d4f37a5…a33ed`. A is monitor-only. B v2, handoff dataset creation, scientific training, scoring, and submission remain forbidden pending a verified A terminal handoff. (Codex root)
- 2026-08-01 — **Owner-authorized Phase 1 transport repair preregistered / no GPU yet.** A v1 remains terminal at 0.70 s before CUDA/data/training, but the owner explicitly replaced only its no-retry rule. Dedicated commit `2241c72…` renders A/B v2 as standalone `script.py` code files with embedded immutable role contracts; isolated no-sibling self-tests and 9 focused tests passed. One A v2, then contingent B v2, is allowed; any v2 failure stops Phase 1 and no v3 is authorized. Fresh quota/no-running-kernel/hash preflight remains required. (Owner + Codex root)
- 2026-07-31 — Quarantined G-02 APNCC fit105 control completed under the owner exception: 105/105, preserved 66/66 byte-identical, combined/adjusted `0.860516437125`, raw `0.867176587961`, recall `0.988149479570`, division Jaccard `0`; zero GPU/Kaggle/CV57 action. The frozen native sweep is active and G-03 remains stopped independently. (Codex root)
- 2026-07-31 — Shared cmux operation guard corrected before the sole G-03 mutation: it now parses exactly one authenticated Kaggle GPU row and holds the serial lock across the full pre-quota → action → post-quota bracket. Local no-network coverage is `25/25 PASS`; the A v1 push used that corrected guard. This infrastructure repair does not reopen G-03 after the separate Phase 1 terminal hard fail. (Codex root)
- 2026-07-31 — **G-03 Phase 1 terminal hard fail / campaign stop.** The sole A v1 failed at 0.70 s with `ModuleNotFoundError: g03_resume_torch` because Kaggle executed only `code_file`; no Torch/CUDA/data/training/checkpoint/handoff/graph work occurred. Downloaded log SHA `e1c5370d…7caed`; post-terminal authoritative quota remains `5.43/24.57 h`, debit `0.0`. Per the locked no-retry rule, no package repair or B action follows. Terminal report: `PHASE1_TERMINAL_REPORT_20260731.md`. (Codex root)
- 2026-07-31 — G-03 Phase 1 A (`onebitbrain/g03-phase1-resume-a-v1`) v1 was pushed exactly once from the hash-pinned package under the shared serial lock and corrected authoritative quota bracket. Pre/post GPU quota remained `5.43/24.57 h`, so immediate debit is `0.0` and terminal reconciliation is required. A is now monitor-only; B cannot be pushed until A's completed output produces a verified immutable handoff dataset. No B action, graph result, CV57, or submission occurred. Receipts: `phase1_ops/`. (Codex root)
- 2026-07-31 — G-03 Phase 1 infrastructure contract is preregistered and committed in its dedicated worktree (`05c1c6d…`). `6/6` local tests pass: atomic publish/verify, tamper/partial/undeclared/conflicting-restart refusal, idempotent receipts, package build, and frozen source/metadata contract parity. CUDA code uses a real synchronized >=1-hour fp32 workload, periodic local checkpoints, complete optimizer/scheduler/scaler/Python/NumPy/Torch CPU+CUDA RNG/sampler handoff, source/config/runtime SHA checks, and a sustained `>=256 MiB`/15-s swapout abort. The 44b6 receipt is explicitly structural/no-op and cannot be selected or interpreted as the frozen scientific graph objective. No Kaggle operation, GPU debit, graph outcome, CV57, or submission occurred. (Codex root)
- 2026-07-31 — Owner opened a narrow exception for the quarantined G-02 native scorer: finish APNCC fit105, score the frozen existing 108-variant bank and hybrids, seal one fit selection, then take exactly one selected native+hybrid CV57 measurement. Conditional GPU authority applies only to producing that missing selected CV57 bank under standing controls. G-03 Phase 1 and its active infrastructure proof remain unchanged; no G-03 result is created. (Owner + Codex root)
- 2026-07-31 — Shared cmux launch instrument built and green: `scripts/cmux_ops.py` with `tests/test_cmux_ops.py` **23/23 PASS**, zero external calls (fake `kaggle` fixture). Bounded 9-op `command_plan()`, argv-only with `shell=False`, `O_CREAT|O_EXCL` serial lock released in `finally`, stale-break requiring both age **and** an in-flight probe, credential-scrubbed receipts that still record true output SHA-256, `kernel_push` refusing on `RUNNING`/`QUEUED`, `competition_submit` refusing without an `eligible:true` predicate receipt and no override flag, `bracketed_launch` aborting unless the authoritative quota pre-query parses, and no retry anywhere. This closes the `KFG-18` "no structural call site" gap that produced two ad-hoc launcher failures today (PEFT v1's owner `403` on first dataset status; the native-fit prepush stop on Kaggle's unparsed literal `Not found`). Placed in `scripts/` because G-02 is archived and G-03 needs the same instrument; it complements, and does not fork, `kaggle_prelaunch_guard.py`. Advisory recorded: with `24.57 h` expiring at the `2026-08-01T00:00:00Z` refresh (~13.9 h out) and Phase 1 not yet hash-pinned, **no further launch is warranted this window** — the Controls forbid GPU before the Phase 1 contract, and ~255 T4-h remain through 2026-09-29. No Kaggle asset, kernel, push, GPU debit, or submission occurred. (Claude)
- 2026-07-31 — Legacy G-02 native v1 terminal/quota bookkeeping completed: remote COMPLETE, exact `0→1`, 105/105 banks structurally valid, `0.37 GPU-h` debited, quota `5.43/24.57 h`, and serial lock released. No graph read, score, CV57, dataset publication, or submission occurred; its partial APNCC control stopped at 66/105. G-03 Phase 1 remains the next permitted compute only after its own hash-pinned contract. (Codex root)
- 2026-07-31 — G-03 charter recorded. Phase 0 selected `G03-TA-CROPADAPTER-E2E-V1`; the literal “Trackastra designed raw-crop input” premise was rejected, while the missing real-appearance mechanism was retained as an explicit crop adapter. No GPU, graph read, CV57, or submission occurred. (Codex root)
