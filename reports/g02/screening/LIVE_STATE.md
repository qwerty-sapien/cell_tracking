# G-02 Live State — archived source of truth

**Update protocol.** Single live status doc for the G-02 cell-tracking campaign, shared by
**Claude and Codex**. READ IT FIRST each session. UPDATE IT after any state change (a node
result, a gate flip, a DEAD promotion, a decision). Keep it SHORT — current state only;
push detail to the linked reports. Append a dated line to the Log (newest first). This doc
points at the authoritative artifacts; it does not replace them.

## Authoritative artifacts
- Plan / frontier: `campaign_narrowing_and_hypothesis_dag_v1.md` (human-locked 2026-07-25).
- Bet ledger (root coordinator is the sole writer): `LEDGER.md`.
- Scorer / CV57→public transfer gate: `p0_parity.md`.
- Stop-loss doctrine, mandatory before any repeat-compute request: `MARGINAL_VALUE_GATE.md`.
- Kaggle kernel launch reliability, mandatory before any push: `tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_AUDIT.md` and `tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`.

## Current global status (as of 2026-08-06)

- **COLAB DIVISION PARENT-EVENT THREE-CELL RESUME PASS / TRAINING BLOCKED.** The actual
  Biohub competition rules permit external models and therefore a privately
  transported Colab-trained weight dataset, subject to competition-data privacy
  and reproducibility obligations; the named public support pack was not used as
  authority. The exact V32 `(1,4,4)` stride carrier is complete and independently
  reverified: 199 videos / 19,900 `(64,64,64)` uint16 frames, 151 parent events,
  302 daughter edges, 10,433,331,200 image bytes, manifest
  `6048f8d7…e6a121`. G-03's complete state/atomic publication/swap boundary is
  ported into exactly three manual Colab cells: mount/install/manifest-bound
  semantic resume, training with a 600-second checkpoint callback, and terminal
  checkpoint plus full-JSONL publication. Fifteen focused tests pass. Each
  session records its assigned GPU at start and end; hours are kept in exact
  device buckets and never naively summed or mixed with Kaggle. Exact runtime
  matches retain the bit-identical claim; changed devices can resume only with
  an explicit `NO_BIT_IDENTICAL_CLAIM` continuity receipt. A disconnect-open
  ledger row is conservatively closed at the next manual start using its last
  checkpoint plus one cadence (or the six-hour ceiling if none is attributable). There is no
  activity injection, browser driver, scripted session cycling, or account
  multiplexing. Colab hours remain `{}` and Kaggle hours were not touched. The deployment gate is frozen at
  top-22 `>=5 TP / <=17 FP`, precision `>=5/22`, AP lift `>=8462x`, never
  AUROC. The owner clarification classifies this >2-frame tensor/temporal head
  with shape-compatible spatial initialization only as a new architecture, not
  V32 weight refinement; no waiver is required and V32 stays immutable. Exact
  objective finalization and all training remain fail-closed because
  `V32_RESIDUAL_ERROR_TAXONOMY.json` and a measured timing receipt have not landed. The active V32
  remote run was not queried, killed, mutated, or cadence-changed. Records:
  `tiered_surface/branches/division_temporal_colab_v1/`.

- **CAPACITY GAP NOT ESTABLISHED / RUNPOD PLANNING ONLY / PARKED-COMPUTE
  RE-ADJUDICATED.** The supplied quota snapshot leaves `24.73-7.67=17.06`
  unreserved T4-h after V32. The new detector's exact architecture and schedule
  await the taxonomy, so its measured total projection is `UNESTABLISHED` and
  committed split is Kaggle `0.0 h`, Colab `{}` by model, RunPod `0.0 h`.
  A frozen 30--60 minute same-workload T4 timing receipt decides allocation:
  `<=17.06 h` means Kaggle-only; larger projections may use a separately
  benchmarked device remainder. The owner-indicated SGD 20 RunPod envelope is
  planning input, not per-run approval. At current listed A100 rates it buys at
  most about 11.15 PCIe or 10.41 SXM hours before storage, but the locked
  campaign's existing USD 10 cap remains effective until explicitly amended
  (about 7.19/6.71 hours). DPRC-O0's distinct
  `<=4 h` preflight may resume preparation but remains launch-blocked on its
  manifest/crop-contract/5-second checks. Crop-wrfeat remains `PARK-COST`
  (`195.59/659.856 T4-h`), full organizer training remains DEAD (~478 T4-h),
  and DINOv2 R5 remains scientific `FAIL_PARK_C`; compute alone reopens none of
  them. Report: `tiered_surface/branches/division_temporal_colab_v1/CAPACITY_AND_PARKED_COMPUTE_ADJUDICATION_20260805.md`.

- **V32 UNSHARDED VERSION 1 REPLICATION PASS / CV57 0.914358 / PROOF REF
  55272130 COMPLETE WITH BLANK SCORE / REPAIR EVIDENCE BLOCKED / CAP2 VERSION 1 RUNNING.** Prompt B
  abolishes sharding: historic S00--S02 outputs are diagnostic evidence only,
  are never reassembled, and receive no retry. Fresh slug
  `onebitbrain/g02-public-temporalunet-v32-unsharded-v2` exact Version 1 was
  pushed at `2026-08-04T13:41Z` after a 162-identity immutable V32 package,
  inline prelaunch PASS, and `79 passed, 3 subtests passed` focused suite.
  The kernel is one end-to-end execution with no wall-clock kill. Its
  authenticated `6.8 h` projection passes the owner `<8 h` auto-approval gate;
  the canonical locked ledger reserved `7.67 h` against authoritative quota
  `5.27/24.73 h` used/remaining, with no launch-time debit. The first permitted
  `2026-08-04T14:44Z` and `20:01Z` status receipts report `RUNNING`. The
  permitted `2026-08-05T00:00Z` read of that same slug returned Kaggle
  `kernels.get` permission denied instead of a worker state, so the last
  authoritative state remains `RUNNING`; terminality is not established and
  the `7.67 h` reservation remains active. No retry is allowed inside the
  hourly window. At `08:11Z`, the next scheduled probe was rejected before
  process creation because the sandbox approval service disconnected; CMUX did
  not run and no new Kaggle receipt exists. This control failure will not be
  bypassed. The later exact-slug probe at `2026-08-05T12:22Z` reports
  `COMPLETE`. Exact Version 1 output downloaded through CMUX and authenticates
  to the package: 162 datasets, 6,261,255 rows, 3,191,037 nodes, 3,070,218
  edges, submission SHA `c61784c3…ab446`. Competition-faithful validation
  preserves two degree-3 parents (max outdegree 3) and logs both parent/daughter
  sets; no graph mutation occurred. Fresh quota `11.70/18.30 h`
  used/remaining establishes a `6.43 h` debit; the `7.67 h` reservation is
  released and zero active reservations remain. The downloaded Kaggle log is
  retained; the pre-existing missing persisted-JSONL caveat remains explicit.
  The one authorized frozen CV57 score is `0.914358473021`, above reported
  public `0.908--0.911` and APNCC `0.881522069960` by `+0.032836403061`.
  This clears replication/local submission eligibility but remains below the
  separate `0.916` weight-refinement gate, so no refinement is opened.
  APNCC-versus-V32 taxonomy is published and reproduces all 57 score rows with
  zero mismatches. V32 reduces ordinary-association residual events
  `2459→1931` and fragmentation FN `1482→1053`; division remains the dominant
  score ceiling and worsens as deployed error count `42→108` (`3/69/39`
  TP/FP/FN). A hidden-test-compatible immutable artifact and fresh one-shot
  lifecycle predicate are now required before submission; the public output
  may not be copied. G1 is a later independently constructed arm and its APNCC
  delta is not additive. Fresh hidden-test package `onebitbrain/g02-v32-submit`
  contains the same pinned V32 source/weights, dynamic four-video test discovery,
  no public identities or output rows, and persisted JSONL logging. Static/inline
  guards plus the `0.91` local gate pass. Exact Version 1 launched at `12:54Z`
  with a `0.30 h` projection and `0.50 h` reservation against unchanged quota
  `11.70/18.30 h`; it is exact-slug `COMPLETE` at `13:57Z`. Exact Version 1
  output passes fresh validation: four dynamically discovered hidden datasets,
  235,198 rows, 119,616 nodes, 115,582 edges, no copied public output, no
  pinned public identities, max outdegree two, and authenticated persisted
  JSONL SHA `86becd3c…f4e6d`. Submission SHA is `11ff6284…e88b1` and the fresh
  one-shot predicate passed after a fail-closed path-spelling revalidation fix
  (`73 passed, 3 subtests passed`). It was consumed exactly once by the
  code-output lifecycle; new submission ref `55272130` is authenticated
  `COMPLETE` with both score fields blank, with no retry permitted. The
  authenticated API provides neither rejection detail nor version-specific
  failed output: `/1`, `/2`, and `/3` status `COMPLETE`, but each output and
  version-qualified log request returns the same retained successful CSV/log
  (log SHA `3fea029c…354454`) and each historical source pull is HTTP 403.
  Its retained log performed full recursive input hashing and has no failure
  event, so hashing is not exonerated or implicated. No bisection package was pushed. The eight-stage source-forward ladder is locally built
  and static/inline-guarded; historical tracebacks and rejected CSVs remain
  required before its pristine control launch. Evidence:
  `tiered_surface/branches/public_temporalunet_baseline_v1/submission_repair_v32/EVIDENCE_RETRIEVAL_20260806.md`.
  Authoritative quota `11.93/18.07 h`
  used/remaining establishes a `0.23 h` hidden debit; the `0.50 h` reservation
  is released and zero active reservations remain. The automation-control API timed out on three update attempts and
  its persisted prompt remains stale; future continuation must follow this
  section and query only `onebitbrain/g02-v32-submit`, never the completed
  unsharded-public slug. The separately requested cap-to-two arm is now the
  sole active production node: unsharded
  slug `onebitbrain/g02-public-temporalunet-v32-cap2-unsharded-v1`, package
  SHA `1ac9e1c2…bbd86`, source SHA `d27e246e…e6a81`, static/inline guards PASS,
  and current focused verification `90 passed, 3 subtests passed`. It leaves the pristine safe-division
  routine unchanged, applies only the postlink `outdegree>2` top-two V32
  `edge_sort_key` rule, preserves all other edge order, and persists every
  affected parent/ranking. Full replay is required because the baseline
  CSV/logs discarded edge probabilities and raw GEFFs. Exact Version 1 launched
  at `14:06Z` after the proof submission attempt, a fresh absent-slug check, and
  unchanged quota `11.93/18.07 h`; 7.67 h is reserved and initial exact-slug
  status is `RUNNING` at `14:07Z`. Monitor at most hourly. On terminal PASS,
  score `public_v32_cap2` exactly once on frozen CV57; this experimental arm is
  not submission-authorized.
  **Provenance correction:** the pinned pristine notebook SHA
  `3f995a97…b7847d` contains `add_safe_divisions_postlink` and zero
  `outdegree` references, so the prior local outdegree-two rejection was
  stricter than the competition behavior, not a V32 defect. Baseline validation
  now preserves the emitted graph and logs every parent above degree two
  (`dataset`, parent id, outdegree, daughters); no routine is patched. A
  cap-to-two/highest-scoring-daughters arm is a separately receipted experiment
  after the baseline score, never a silent repair. Provenance:
  `tiered_surface/branches/public_temporalunet_baseline_v1/V32_OUTDEGREE_PROVENANCE_20260804.json`.
  **Immutable logging caveat:** the launched source hash `932e0b12…f8e03`
  predates the subsequent persisted `RUN_LOG.jsonl` contract. It cannot be
  retrofitted without changing Version 1; terminal authentication will retain
  every available downloaded log/receipt and label the missing structured-log
  binding explicitly rather than falsely claiming it passed. This is a logging
  finding, not a model or graph failure. Record:
  `tiered_surface/branches/public_temporalunet_baseline_v1/V32_UNSHARDED_V2_LOGGING_GAP_20260804.json`.

- **G04 DATA-PATHOLOGY G1 STANDING / CROP-WRFEAT PARK-COST.** A complete
  203-video CPU census finds 967 exact frozen transitions and 227 frame-global
  shifts over the fixed 10-um pool radius. On exact APNCC CV57, frame-level
  registration before candidate construction raises adjusted score
  `0.881522069960→0.886764420260` (`+0.005242350300`) and improves edge
  TP/FP/FN `34320/2459/1948→34389/2308/1879`; frozen identity alone and stride
  parity are exact score no-ops. Reachability on 75 flagged CV57 transitions
  improves `209/311→259/311`; 90/93 GA-R12REACH shell positives on flagged
  fit-only transitions return inside 10 um. A7 remains closed: G1 is one
  whole-frame transform before the pool, never a candidate feature. The fixed
  crop-level real-wrfeat pilot failed with 12/243 unmatched and 199/243
  boundary-truncated masks, projecting 766.233 CPU-h for CV57. Geometry audit
  now measures unique-object p99 Z/Y/X extents `23.3/39.3/34.9` voxels versus
  the old `16/32/32` crop; the preregistered repair uses recentered
  `32/64/64` crops plus one `48/96/96` fallback. The exact three-frame /
  916-detection T4 pilot passed genuine CUDA batching and serial/batch-one
  label parity. Its two independent frames are `673/673` valid with zero
  boundary/empty masks; all-frame totals are `914/916` valid, 1 boundary and
  1 empty, with fallback recovering 50/52. Throughput is `0.938601 s/detection`;
  exact-count/dense-frame CV57 projections are `293.262/599.869 T4-h`, and the
  preregistered conservative result is `659.856 T4-h` (combined held-fit105+
  CV57 `1875.361 h`) versus 27.24 h remaining. Coverage is repaired but the
  full-run budget gate fails. The lane is `PARK-COST`, not evidence-parked or
  DEAD: geometry, recentering, fallback, CUDA batching, and coverage proof are
  banked for reuse if Trackastra/wrfeat becomes relevant again. No fourth
  throughput configuration is allowed without a new named architecture that
  removes per-detection tri-orientation redundancy, so no feature production
  or score follows.
  The common transform is now standing at APNCC, Trackastra/wrfeat and
  synthetic-division construction seams; a distinct V32 replay patch is ready
  but remains score-blocked on full immutable V32 authentication. Reports:
  `../../g04/lanes/data_pathology_repair/TERMINAL_REPORT.md` and
  `../../g04/lanes/global_jump_crop_wrfeat_v2/`. The owner-approved APNCC+G1
  public-transfer package passed dynamic-inventory/raw-Zarr parity and its
  fail-closed launch guard. Fresh slug `onebitbrain/g1jump-submit` exact Version
  1 completed; its output re-authenticated byte-exact to the local raw-parity
  candidate (`366a12d6…8c8`, 263,260 rows). The canonical `6.0 h` reservation
  is released at authoritative quota `5.27/24.73 h` used/remaining (a `0.17 h`
  debit). The one-shot registry is consumed by exactly one competition
  submission, ref `55221497`, is terminal COMPLETE at displayed public `0.881`.
  Thus local `+0.005242350300` transferred as `+0.000` at public display
  precision. This is local-to-public anchor 2 and materially weakens reliance
  on CV57-only corrections for public advancement. No V32 status read or
  cadence change occurred; the terminal monitor is stopped.

- **REPOSITORY DEBUG / KAGGLE SUBMISSION LIFECYCLE HARDENED.** The invalid
  public-copy G04 transport is archived and registered as `KFG-25`; refs
  `54890576` and `55201829` are the same public-identity/hidden-rerun failure
  family. Code-output submission now requires an exact-version output receipt,
  hidden-test-compatible predicate, canonical consumed-on-attempt claim, and
  exactly one reconciled new ref. Direct predicates are also one-shot; latest
  kernel status, finite quota/ledger reservations, trusted CLI credentials,
  OS-held serial-lock ownership, and kernel-bound terminal reservation evidence
  fail closed. Active pytest collection is 1,237 with zero errors. The canonical
  uncapped suite is reviewed-green at 1,198 PASS / 39 strict XFAIL / 267
  subtests / zero failures; the earlier 25-red audit was only a capped prefix.
  Four active defects were fixed and all 39 retired nodes have exact lifecycle
  reasons. Focused
  reliability/scorer/guard/oracle verification is 100 PASS plus 7 subtests. The full/edge-only
  zero-edge scorer path now matches the evaluator. Mixed-cost sparse tie noise
  is fixed; retired banked LAPJV parity is a strict XFAIL for filler-dependent
  equal-cost tie propagation. Full
  audit: `../../../code_quality/REPO_DEBUG_AUDIT_20260803.md`. No Kaggle,
  RunPod, or full crop-wrfeat action occurred.

- **GPU GOVERNANCE AMENDMENT LANDED / SHARED 30 H LEDGER RETAINED.** ALIGNMENT
  now supersedes the former 5 h runtime-kill/sharding rule: authenticated Kaggle
  projections below 8 h are auto-approved with no per-run wall-clock kill;
  projections of 8 h or more require an explicit literature-cited human-approval
  pitch. The canonical shared ledger remains the mandatory prelaunch 30 h/week
  budget check. The amendment records that six-way sharding of a 6.8 h job added
  an error-prone reassembly layer and yielded invalid outdegree-three parents.
  Comprehensive structured stdout plus persisted/downloaded run logs are now
  mandatory for every local and Kaggle run. Inspected ALIGNMENT SHA-256 is
  `64d1f96e…4c0f9`. No GPU action occurred.

- **DEFAULT POST-SCORING VISUAL REVIEW READY / DIAGNOSTIC ONLY / NO GATE
  CHANGE.** The stable CPU `latest` bundle applies the exact four-category
  selector to APNCC CV57: 9 unique clips after deduplication, with all category
  tags retained. Each 100-frame MP4 is exactly `15.000 s` by default
  (`6.6667 fps`); HUD metrics/corners, target duration, exact FPS override, and
  default/all scope are CLI-configurable. The `16,558,395`-byte default bundle
  is below 1 GB, validates 9/9, and reuses 9/9 media on an identical input hash.
  Frozen/global-jump census boundaries are shown on video and timeline; sparse
  GT and exact edge/division error semantics are unchanged. `scoreall.py`
  refreshes default scope only after score publication. The explicit manual
  all-scope CV57 bundle is now rendered and validated at 57/57 clips,
  `90,888,617` bytes, `15.000 s`/clip; it remains diagnostic-only. Entry
  points: `../../g04/post_scoring_visual_review/latest/index.html` and
  `../../g04/post_scoring_visual_review/full_catalog_20260804/index.html`.

- **SYNTH-DIV-SIM2REAL-V2 EVENT DETECTION PARK-C / CONDITIONAL PAIR ASSET
  BANKED / CTC PARK-A-PERMISSION.** Prompt 2
  opens one observation-distinct sim-to-real division
  lane using Jose Freitas's author-declared CC0 synthetic lineage labels, not
  another sparse-real or hand-threshold ranker. Explorer v6 and generator v3
  notebook sources are locally acquired and SHA-256 pinned. The full version-3
  output authenticated at 3,716 files / 19,868,008,592 bytes, tree SHA-256
  `9a87684a…029865ec`; structural validation reproduced 1,539 static volumes,
  2,174 sequences, 4,056,226 nodes, and 165,267 divisions without materializing
  image arrays. V1 then failed closed before fitting: one `seq_0019` labelled
  daughter pair fell outside its literal nearest-8 universe. Amendment 2 parks
  that realization and freezes `SYNTH-DIV-SIM2REAL-v2-REACHABLE`: nearest-8
  stays fixed, only unreachable forks leave the pair-head loss, all parents
  remain in the event-head loss. The full authenticated audit passed:
  165,148/165,267 (`99.927995%`) forks reachable overall, with train,
  validation, and synthetic-test each above `99.92%`; only 119 pair-head rows
  are excluded. The one frozen local CPU fit completed in `4521.04 s` over
  3,674,104 pair and 2,616,835 event examples. Model JSON/array hashes are
  `53b36d96…d21dd82` / `d979a308…cfff652`; strict reload passed. The first
  truth-blind serial fit117 fragment projected beyond the four-hour cap and was
  stopped at `296.26 s` with zero shards/GT. Execution amendment 3 freezes four
  independent per-video workers plus exact serial/parallel byte equivalence;
  no scientific model, feature, candidate, threshold, or gate changes. Its
  first full attempt reached 91/117 atomic shards but no manifest in
  `14545.63 s`; the coordinator terminated only its over-cap processes after
  confirming the executor exception path left workers alive. No GT opened.
  Amendment 4 freezes one read-only cache of the identical physical-coordinate
  product plus bounded worker termination. The pre-change NPZ hash remains
  exact, 31 focused tests pass, and the formerly `296.26 s` fragment now seals
  the same 845,628-row NPZ/hash in at most 40 s. Generator freeze V4 opens one
  fresh four-worker fit117 attempt under the unchanged `14400 s` cap. That
  attempt completed in `1135.49 s`: 117/117 videos and 80,013,717 truth-blind
  candidates are frozen in manifest SHA `029061b1…036b4`, with `contains_labels=false`
  and `gt_archive_opened=false`. No cap extension was needed. The unchanged
  fit117 real-GT evaluation completed in `1509.17 s`: global beneficial-action
  AUROC `0.889300580471` passes the frozen `0.85` line, with 65 beneficial
  candidates and 65/83 exact-fork reachability. AP is only `0.000019093115`;
  44b6/6bba AUROC is `0.843482493424/0.895217431438`. More decisively, every
  one of the frozen 19×6 deployment policies selects zero actions: all scores
  lie below `0.0005`, so the complete frontier is `TP/FP/FN=0/0/83`, Jaccard
  `0.0` for all 114 rows. This invokes the preregistered zero-real-TP hard
  failure. Lower thresholds, score shifts, and post-GT recalibration are
  forbidden; the exact instantiation is PARK-C before graph integration.
  Before any payload read or model outcome, the lane froze 4.07%→0.26% loss
  reweighting, geometry/lineage/motion-only features, the published
  `1.86/+0.30/7.24` priors, a real beneficial-action AUROC `>=0.85` gate, and
  the full moderate-FP frontier. The decisive V32 integration dependency is
  moot for this instantiation because the hard failure stops first; APNCC is
  legacy control only. Codex task `019fc6b9-0154-7be1-8b37-7159cb6f5da8` is the sole
  authoritative watcher for the synthetic real-GT AUROC gate and the sole
  writer of the synthetic-on-V32 integration. The separate V32 continuation
  owns only base production/authentication and its contracted V32/G1 scoring;
  it has no synthetic-gate or synthetic-integration dispatch. CTC archives are
  registered restricted and excluded because official terms require written
  permission for non-CTC public scientific use; citation alone is insufficient
  for this Kaggle training use. The post-outcome split is precise: the
  parent-event head is random (AUROC/AP `0.521521/3.1367e-5`) under a
  geometry/lineage/motion-only contract that excluded raw appearance, while the
  conditional daughter-pair head is a working banked asset
  (`0.897081/0.368520`). Record “division event detection has no working
  mechanism; conditional daughter assignment is solved,” not “division is
  unreachable.” CTC real divisions are the leading appearance-event data
  candidate only after written organizer permission (`PARK-A-PERMISSION`).
  Until an authorized event detector clears 5 TP / <=17 FP in top 22 and
  AP-lift `>=8462x`, redirect mechanism effort to ordinary association
  (`+0.061628263275`). Real
  labels were used only by the sealed one-shot evaluation; no integrated graph,
  combined score, visualizer refresh, GPU, submission, or RunPod action exists.

- **OWNER-SUPERSEDED PUBLIC TEMPORALUNET3D BASELINE ADOPTION ACTIVE.** This is
  not a re-screen of the DEAD from-scratch TemporalUNet3D mechanism: the owner
  superseded that scope with the independently trained public pipeline. Kaggle
  notebook exact Version 32 (`scriptVersionId=337501010`, public `0.911`,
  successful `18m57s` T4x2) is pinned locally at SHA
  `3f995a97…b7847d`. Its primary public checkpoint is byte-identical to the
  existing support pack (`12f6881e…e771`), and the independently trained seed
  314159 checkpoint is authenticated at `9bac2fa0…e305f`. Frozen held-fit and
  CV57 reproduction has not yet run, so APNCC `0.881522069960` remains the
  base/legacy control and FINAL slot 1; no machinery has been re-pointed and no
  new submission is authorized yet. The public pipeline is substrate/ensemble
  only: weight refinement is forbidden below exact local `0.916`. Exact
  one-video V32 parity-smoke kernel
  `onebitbrain/g02-public-temporalunet-v32-smoke-v1` Version 1 is terminal
  **PASS**: exact public/checkpoint/runtime bindings, 50,304 nodes, 48,724
  edges, 99,028 rows, and strict graph validation with zero errors, warnings,
  merges, or outdegree violations. Prediction took `284.650 s`; the rounded
  debit was `0.11 h`, leaving `28.63/30.00 h`. This opens only the sharded
  immutable-weight 162-video reproduction. All 18 disjoint 27-video packages
  for V32, single-seed, and detection-only blend pass inline prelaunch. V32
  shard 0 fresh-slug Version 1 is terminal **PASS**: 27/27 exact datasets,
  541,109 nodes, 521,261 edges, 1,420 dividing nodes, and 1,062,370 rows;
  prediction took `3264.467 s`, strict graph/output validation passed, and the
  terminal debit is `1.14 h`. Frozen-official early scoring of those same 27
  videos gives native division TP/FP/FN `2/52/23`, Jaccard `0.025974`; the
  nine CV57 videos give `1/11/12`, Jaccard `0.041667`. Native branching is
  therefore neither absent nor usable as-is: 92% of GT divisions remain FN and
  52/54 scored predictions are FP, so the synthetic lane must add recall and
  aggressively refine precision. V32 shard 1 exact fresh Version 1 launched
  with code `a4e17b04…c4568`, shard manifest `5676398b…d752`, and a
  10,800-second whole-kernel kill, then completed in `3323.533 s`. Exact
  terminal validation passed 27/27 datasets, 468,656 nodes, 449,831 edges,
  1,039 dividing nodes, and 918,487 rows. Its reservation was released under
  the canonical lock after authoritative quota moved `2.76/27.24→3.88/26.12 h`
  used/remaining, a `1.12 h` debit. Shard 2's frozen package and explicit
  fresh-Version-1 predicate passed, but the bracketed launcher refused locally
  before `kernels push`: the newly hardened lifecycle guard treated Kaggle's
  normal fresh-slug denial as unparseable. No remote version was created and
  no GPU debit occurred; the temporary `3.0 h` reservation auto-released, so
  quota remains `3.88/26.12 h` used/remaining with zero active reservations.
  The exact-v1 lifecycle is repaired and passes 54 focused / 61 combined tests.
  In the next hourly window, the bracket's under-lock check was the sole target
  status read, proved fresh-slug absence, and push receipt
  `3d61d30d…0c7492` confirms shard 2 exact Version 1. Its immutable code is
  `97fa08fa…4c6fe1`, package manifest `288a877b…327847`, shard manifest
  `df876107…c8f1cf`, and whole-kernel kill 10,800 seconds. Immediate quota stayed
  `3.88/26.12 h` used/remaining; the canonical `3.0 h` reservation is active
  and will remain until terminal evidence plus authoritative quota
  reconciliation. Launch record `FULL_PUBLIC_V32_S02_LAUNCH.json`
  (`6b49ad89…5c6cb9`). The following hourly receipt found shard 2 COMPLETE, but
  strict graph validation rejected its exact Version-1 output: two parents
  have outdegree 3 and all other indegree/outdegree constraints pass. Shards 0
  and 1 revalidate PASS under the current validator. Authoritative quota moved
  `3.88/26.12→5.10/24.90 h` used/remaining (`1.22 h` debit); the reservation is
  released with zero active reservations. Terminal adjudication
  `FULL_PUBLIC_V32_S02_TERMINAL_ADJUDICATION.json` (`ee24311c…7aea97`). Shard 3
  is blocked pending a frozen packaging-only feasibility repair; no retry is
  authorized yet.
  The four-way scorer audit reproduces APNCC CV57 exactly and now
  separates detection, ordinary association, matched-lineage fragmentation,
  and division. Kaggle ref `55203439` is terminal COMPLETE at public `0.881`,
  tying APNCC's displayed score despite local `0.881635476737` versus
  `0.881522069960`. Control-anchoring the local `+0.000113406777` gain at
  APNCC's displayed `0.881` projects `0.881113407`, which still displays
  `0.881`. This interval-censored hidden-population observation does not justify
  changing frozen scorer arithmetic; local and public are recorded separately,
  and APNCC remains FINAL slot 1.

- **G-04 BELOW-MEDIAN REFRAME / LANE-A PUBLIC TIE / LANE-B KAGGLE-INFEASIBLE.** Public `0.881` is
  approximately rank `1100/1800`, below median; the prior conclusion that the
  achievable surface is exhausted is withdrawn.  G-04 optimizes exact frozen CV57
  combined score above APNCC `0.881522069960`.  Association continuity, 44b6-first,
  legacy `+0.005`, near-zero-FP division operation, and the local `0.89` submission
  floor do not gate these lanes.  Lane A completed its exact 57-video official
  sweep: baseline reproduced `0.881522069960`; `unfiltered_geometry_cap5` applied
  239 forks and scored `0.881635476737` (`+0.000113406777`) with division
  `0/11/42`, so the gain is edge-only.  Its observation-distinct four-test-video
  artifact validates cleanly (15 forks; SHA `da73c754…84509`).  After a guarded
  direct-file HTTP 400 with no entry, the byte-identical artifact was transported
  through authenticated CPU/no-network code kernel version 1 and submitted as
  Kaggle ref `55201829`.  It terminated unscored with Kaggle's generic
  incorrect-format error (`total_bytes=0`); exact graph parsing still passes,
  because that public-only copy kernel could not cover Kaggle's hidden rerun.
  A replacement exact-APNCC-plus-cap5 kernel passed raw-Zarr byte parity and 27
  tests, completed as exact version 1, and returned the byte-identical candidate.
  Terminal validation passed all package/runtime/output gates and created the
  one authorized code submission as ref `55203439`; it is COMPLETE at public
  `0.881`, a displayed tie with APNCC rather than a public improvement. The
  frozen scorer is unchanged: one three-decimal hidden-population result cannot
  separate display precision, population shift, and arithmetic bias. No retry
  is authorized.
  Lane B's authenticated `cyto3` one-frame CPU
  pilot produced 171 explicitly labelled computed masks and matched `234/243`
  detections at radius 15; radius 30 covers `243/243` and is preregistered with
  per-row match distance/mode.  The pilot is not production-manifest eligible;
  Its authenticated 223,211,151-byte CPython-3.12 runtime carrier is now READY
  as exact dataset version 1 and passed a 30/30-file byte-exact round trip.
  The timing package's declared CV57-cache dataset did not exist remotely; a
  safety review rejected exporting the 595 MB local derived cache.  The revised
  raw-only timing kernel instead authenticated organizer-hosted raw Zarr and all
  runtime files at Step 0, then failed closed after `75.597 s` because one
  computed region emitted a non-finite feature.  This is not a valid frame-time
  measurement and does not unlock production.  Exact-raw/config CPU reproduction
  identified planar labels `26/42/99` with finite native wrfeat and only legacy
  scikit-image solidity equal to `inf`.  The changed-signature v2 defines solidity
  as voxel-cell volume divided by the physical 3-D hull of half-voxel corners,
  with no sentinel; native Trackastra wrfeat12 is unchanged.  Its fresh-slug
  four-file package passes 37 tests plus static/inline guards and completed as
  exact fresh-slug version 1.  Its two label/feature payloads are byte-identical;
  repeats took `43.460/42.774 s` after `30.217 s` initialization, yielding
  CV57/held-fit105 steady projections of `67.734/124.767 T4-h` and combined
  worst `195.586 T4-h`.  Full Kaggle production is therefore closed against
  `28.74 h` remaining.  The official StarDist package supplies no generic
  pretrained 3-D model, and no alternative authenticated 3-D weights exist
  locally; RunPod is the remaining compute path and requires explicit owner
  approval.  Authoritative quota is now `1.26/30.00 h`; exact per-kernel
  attribution is deferred until all remote accounting settles.  Complete wrfeat coverage,
  held-fit105, and CV57 remain pending.
  Each observation-distinct lane winner with
  local `S>0.881522069960` is authorized for one submission under the standing
  serial/prelaunch/monitoring controls.  RunPod remains owner-gated.  Amendment:
  `../../../g04/screening/OWNER_REFRAME_20260803.md`.

- **G-03 OWNER-AMENDED ORTHOGONAL LANES / NO G-02 REOPENING.** The owner funded
  one fixed G-03 h005 held-fit36 slope read and CPU-only orthogonal screens; this
  does not reopen any archived G-02 mechanism. G-03's trained raw-intensity fork
  probe failed its first-fit cross-family 44b6 ceiling (precision/recall
  `0.454545/0.555556` at `0.5`; best precision at required recall `0.5`) and
  stopped before final 38 access, GPU, or graph integration. The classical dense
  deformable-field screen completed the unchanged frozen 44b6 B0/A1/A0 gate and
  returned `FAIL_PARK_C` for both DELETE and occupied SWAP; learned displacement
  is closed with zero GPU. H005's corrected private carrier and inline prelaunch guard are
  sealed; its single training-kernel version 1 passed remote STEP0, then failed
  before training because the launcher declared but did not install the canonical
  wheel set and the base image lacked `zarr`. There is no checkpoint, score, or
  slope; the rounded infrastructure debit is `0.01 h` and current quota is
  `1.02/28.98 h`. V1 is invalid rather than scientific. A private 14-wheel
  runtime carrier is READY and the mandatory CPU-only install/import/decode
  probe version 1 passed exact wheel, numeric-stack, and fixed 6bba raw+GEFF
  checks with zero GPU-quota movement. The unchanged-core sole V2 passed its
  release guards, but its exact four-source GPU worker failed at top-level
  STEP0 because the runtime manifest was absent or byte-divergent. The frozen
  check did not emit observed mount facts to distinguish those cases. No output
  file, dependency install, scientific import, CUDA allocation, data access,
  optimizer step, checkpoint, held-fit36/CV57 score, or slope exists. V2 debit
  is `0.00 h`; quota remains `1.02/28.98 h`. No V3 is authorized, so Lane 0 is
  infrastructure-terminal, h030 is not evaluable or funded, and its 25 h remain
  preserved.
  No G-02 gate, DEAD status, tier, CV57
  action, or submission changed.

- **G-03 PHASE-2 H000 RECONCILED / NO G-02 REOPENING.** The distinct G-03
  crop adapter now has exact-zero `Linear(48,12)` output at h000. Canonical
  re-decoding of the frozen native CUDA banks reproduced all 36 held-fit44b6
  zero-feature graphs byte-for-byte and scored `0.7630876414338513`. This does
  not conflict with the archived native `0.829194058080`, which is a CV57
  measurement on a different population. The G-03 Branch-B dossier fails, so
  h005/h030 remain unfunded with zero new GPU-hours. No G-02 candidate, DEAD
  mechanism, graph gate, tier, CV57 action, or submission reopened.

- **OWNER BARRIER DISPOSITION ACCEPTED / A7 SEPARATION FAIL / NO GATE CHANGE.**
  The owner accepted the gate-validity recommendation to retract the prior R9 and
  AE-P3/R0 evidence-park reclassifications and accepted all five D3 “no”
  recommendations; none of those historical mechanisms is reopened. After the G-03
  R4 scorer was killed, the exact frozen separation protocol evaluated A7's own
  45,961 occupied-swap candidates with 134 positives. Adding only the original
  phase/template `action_margin` produced A1 per-fold AUC
  `0.627866/0.828496/0.610208/0.659067/0.654961`, zero CG0 retention in all folds,
  and no passing AUC lift over sealed B0 or within-video/frame-permuted A0. It fails
  the occupied-swap `0.90/0.067910` requirement in every fold and is now parked on
  evidence rather than on its former threshold abstention. No FIT threshold, edit
  set, graph, 6bba, CV57, GPU, submission, or tier move opened. The requested
  `node >= -0.002` future-gate package is draft-only in G-03 and has not been applied;
  adjusted `>=+0.005` and raw `>=0` remain unchanged, and zero historical verdicts
  change.

- **DURABLE CEILING-FIRST / ATOMIC-RECEIPT MONITORING RULES ADOPTED / NO GATE
  CHANGE.** `tiered_surface/ALIGNMENT.md` now requires an oracle ceiling before
  any sweep, lattice, or search is funded; a ceiling below the promotion gate
  permits only an explicitly labelled, wall-clock-capped diagnostic. Long jobs
  with atomic receipts are waited on rather than interval-polled, with at most
  one status check per hour, no unchanged-state emission, and liveness checks
  only after a failed receipt/status check. The `+0.005` gate is unchanged.

- **44B6-FIRST GATE VALIDITY FAILURE / OWNER BARRIER / NO GATE CHANGE.** On the
  exact common 36-video held-44b6 set, banked APNCC-DIRECT versus its exact
  pre-APNCC `c=0.25` control is adjusted/raw/node
  **`+0.006949544395/+0.006411258660/-0.001307630206`**. It therefore would
  have **FAILED** the locked `>=+0.005/0/0` three-component gate on node recall,
  despite being the campaign's only confirmed full-CV57 win (`+0.008372`). The
  44b6-first vector gate is not a valid universal filter for CV57 improvement.
  This finding changes no gate or tier and opens no CV57 read; owner review is
  required. D2 exact action oracles find A7 `134/45,961` beneficial and R0
  `5/38,081` beneficial, so their near-empty selections were threshold misses,
  not evidence-negative candidate universes. Reverse literal consensus is
  genuinely evidence-negative at its fixed contract: all 268 literal passes
  and all 53 sealed actions have exact zero delta. The earlier terminal join's
  evidence reclassification of R9 and AE-P3/R0 is retracted; the cited park-audit
  census remains **`22/4/6/1/2`** evidence/dependency/control/policy/owner-override.
  No tier move follows. Report:
  `tiered_surface/diagnostics/gate_validity_abstention_oracle_20260802/REPORT.md`.
- **OWNER-OPENED NARROW CPU LANES TERMINAL / ZERO DEPLOYED EDITS / NO TIER
  MOVES.** A7 phase-template abstained on all 45,961 occupied-conflict swaps
  and is an exact APNCC no-op. Literal authentic reverse consensus sealed 53
  node-disjoint swaps but found `0/53` beneficial, so its graph gate stayed
  closed. The distinct R0 free/free shell no-link falsifier screened 38,081 of
  2,514,005 held-44b6 shell rows, then failed fixed-budget precision at `0/4`;
  it also deployed zero edits. Exact adjusted/raw/node deltas are `0/0/0` for
  all three scientific lanes. The parallel park-audit snapshot classified all
  35 blocked rows as evidence/dependency/control/policy/owner-override
  `22/4/6/1/2`. The later exact candidate-universe oracle diagnostic supersedes
  the terminal join's A7/R0 cause reclassifications; final remains
  `22/4/6/1/2`, with zero tier moves. All
  work was CPU-only; no Kaggle, GPU, RunPod, 6bba, CV57, or
  submission action occurred. Reports:
  `tiered_surface/branches/a7_phase_template_v1/TERMINAL_REPORT.md`,
  `tiered_surface/branches/reverse_endpoint_consensus_audit_v1/TERMINAL_REPORT.md`,
  `tiered_surface/branches/r0_shell_nolink_own_merits_v1/TERMINAL_REPORT.md`, and
  `tiered_surface/branches/cpu_narrow_lanes_20260801/lane4_park_audit/TERMINAL_REPORT.md`.
- **GLOBAL-REASSIGN ORACLE TERMINAL / COMPILER ESCALATION / PROPOSAL ONLY.**
  The full 36-video local-CPU GT-dependent node passed both frozen controls and
  measured T2∞ `+0.09641286660426185` [GT-DEPENDENT] versus T3
  `+0.10497364637634421` [GT-DEPENDENT]. Section 4 is
  `CONTINUE_PROPOSAL_ONLY`: precedent-adjusted `+0.01906337599412862`
  [GT-DEPENDENT] before raw T2∞−T0 `+0.0778414699637755` [GT-DEPENDENT].
  Terminal status remains `ESCALATE_COMPILER_INEXPRESSIBLE`: five retained
  division anchors are valid same-chain remove-and-readds that the frozen batch
  compiler cannot express, not under-merged chains. All T2 depths are identical;
  histogram `{1:453,2:6}` [GT-DEPENDENT] makes single-install endpoint legality,
  not chain depth, the binding constraint. Written proposal only; no campaign
  action is authorized. Report:
  `../../../dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_GLOBAL_REASSIGN_ORACLE.md`.
- **G-02 ARCHIVED / OWNER-SCOPED NATIVE-BASE EXCEPTION TERMINAL / NO
  IMPROVEMENT.** The frozen fit105 run scored all 108 names (76 distinct graphs)
  and irrevocably selected `r08_c08_a035_off` before CV57. The sole authorized
  CV57 measurement is complete on the already-downloaded 57-video bank: native
  combined/adjusted `0.829194058080`, raw `0.834460844608`, node recall
  `0.987196658416`, edge TP/FP/FN `32564/2756/3704`, and division
  TP/FP/FN/Jaccard `0/0/42/0`. This is `-0.052328011880` versus APNCC
  `0.881522069960`. Native predicted zero forks, so the preregistered hybrid is
  graph-identical to APNCC and ties it exactly. Independent pre/post audits pass
  all 57 hashes, one-time access markers, identities, graph parses, and ranking
  arithmetic. No alternative, tuning, repeat, GPU, Kaggle action, RunPod, or
  submission occurred. The result does not affect G-03 Phase 1. Terminal SHA
  `0add0c3f…60af95a`.

## Archived G-02 frontier (as of 2026-08-02)
- **OWNER-SCOPED NATIVE SCORER EXCEPTION TERMINAL / CV57 MEASURED ONCE / REJECT.** The
  unchanged frozen implementation completed all 105 APNCC control fragments; the
  preserved 66 are byte-identical to the recomputation. Exact fit105 control is
  combined/adjusted `0.860516437125`, raw `0.867176587961`, node recall
  `0.988149479570`, and division TP/FP/FN/Jaccard `0/0/76/0`. The frozen
  108-variant run exactly scored all 108 names (76 observation-distinct graphs).
  Frozen selection irrevocably chose `r08_c08_a035_off`: combined/adjusted
  `0.811024400045`, raw `0.817224994691`, recall `0.979235194777`, division
  `0/0/76/0`, and `-0.049492037080` versus identical-population APNCC. Its
  no-fork hybrid is exactly APNCC on fit105 (`0.860516437125`). The selection was
  sealed before CV57. Independent validation passes all 108 names and 105
  identities; fit results SHA is `36d6aa95…a31a3f9`. Remote selected-only v1
  and final infrastructure-only v2 both failed Step 0 before cache/model/GT/
  graph/scorer access, leaving the single measurement unused. The complete
  authenticated cache/checkpoint/source were already local; exact CPU inference
  first reproduced the frozen T4 graph on 3/3 fit-only parity cases. The one
  selected CV57 graph was then materialized and scored exactly once: native
  `0.829194058080` (`-0.052328011880` versus APNCC), with division `0/0/42/0`.
  Zero predicted forks make the hybrid an exact APNCC no-op at `0.881522069960`.
  No GPU was scientifically justified, and no alternative, tuning, repeat,
  submission, or G-03 change occurred. Terminal:
  `tiered_surface/branches/sa_native_base_wave_v1/CV57_LOCAL_TERMINAL_REPORT.json`.
- **NATIVE-BASE V1 TERMINAL / QUARANTINED UNSCORED BY G-03 SUPERSESSION.** The
  fresh-slug Trackastra fit105 v1 completed with exact version delta `0→1` in
  `1297.886 s`. Its downloaded bank passes independent SHA/schema validation:
  `105/105` video files (`36` 44b6 + `69` 6bba), `2,732,827` nodes,
  `12,523,336` candidate edges, `26,166` inference calls, peak `1,024` tokens,
  and zero reported errors. Authoritative quota moved **5.06/24.94 →
  5.43/24.57 h used/remaining**, a **0.37 GPU-h** debit; the shared launch lock
  is released. A supervisor false negative resulted only from checking the
  download root instead of the nested report path and is adjudicated in
  `tiered_surface/branches/sa_native_base_wave_v1/TERMINAL_OUTPUT_ADJUDICATION.json`.
  The newer owner's G-03 charter arrived before any native graph materialization
  or score read. The bank is therefore quarantined **UNSCORED**: no 108-variant
  sweep, selection, hybrid, CV57, dataset publication, or submission occurred.
  The local APNCC control was stopped at `66/105`; partial fragments are preserved
  and no partial score was consumed.
  Lane B is abandoned with zero new spend because no changed CELLECT checkpoint
  carrier exists after the terminal 25,055-file inventory found zero matches; it
  remains scientifically UNTESTED. Public submission still requires explicit human
  approval; RunPod is forbidden. Branch:
  `tiered_surface/branches/sa_native_base_wave_v1/`.
- **RAW3D-NOLINK-MVE STAGE 0 TERMINAL / PARK-C / CAMPAIGN CONSOLIDATE.** The
  owner-approved scoped plan amendment (§6.6) opened the zero-weight CPU probe,
  which completed the exact frozen DELETE
  population (`41` videos / `1,345,497` actions / `1,118` beneficial) in
  `254.704 s`. Raw-3D A1 mean/fold-floor AUC is
  `0.677967209/0.599374874`; mean/full-population CG0-budget retention is
  `0.006094647/0.004472272` (`5/1,118`). It improves DINOv2 mean AUC by
  `+0.007555465` and beats its A0 control by `+0.031393472`, so crude raw
  intensity contains some real pair information, but it fails both frozen
  `0.680411743` advance clauses and the measured `0.85/0.020394` surface.
  Stage 1 asset/provenance authentication therefore stays closed; no
  acquisition, weights, training, GPU, graph, 6bba, CV57, or submission is
  authorized. Trackastra fine-tuning remains unrecommended. Report:
  `tiered_surface/branches/raw3d_nolink_mve_v1/TERMINAL_REPORT.md`.
- **BANKED 0.881 CANDIDATE EXPLICITLY SELECTED AS FINAL.** Kaggle's
  authenticated Submissions state moved from `0/2` auto-selection candidates
  to `1/2` with `onebitbrain/bhap-submit` v1 checked. This binds submission
  ref `54942864` (public `0.881`, local `0.881522069960`, package
  `779aa8d3…606b`) as an explicit final rather than relying on the platform
  default. Competition deadline is **2026-09-29 23:59 UTC**. No new
  submission, slot, or GPU quota was consumed. Receipt:
  `KAGGLE_FINAL_SELECTION_20260731.md`.
- **G-02 TIERED-SURFACE CAMPAIGN TERMINAL / NO ACTIVE WAVE.**
  APNCC-DIRECT remains banked at local `0.881522069960` / public `0.881`.
  The measured selector requirements are DELETE AUC/retention
  `0.85/0.020394` and occupied SWAP `0.90/0.067910`. Independent DINOv2
  appearance and C2 assignment evidence converge on DELETE at
  `0.670412/0.670235` AUC; the distinct crude raw-3D probe raises the empirical
  best only to `0.677967`, still `0.172033` short of the measured AUC need and
  with a `0.599375` fold floor.
  Division's exact GT-dependent ceiling is `+0.092067842`, decomposed
  `(a)/(b)/(c)=1/37/0`: only `+0.002406490` is inferable now,
  `+0.089661334` is representation-blocked, and the GT-only tier is empty.
  CELLECT is permanently PARK-C / scientifically UNTESTED after six
  infrastructure failures. Machine census is LIVE/DEAD/PARK-A/PARK-B/PARK-C
  `0/1/14/18/13`. Reopen only on a new action-labeling formulation (for
  example batch/sequential valuation) or independent AUC `>=0.85` evidence on
  this exact discrimination task—not another appearance encoder. The sole
  owner-reopened §6.2 PEFT and RAW3D Stage-0 exceptions both completed and
  failed; there is again no active wave. Standing Kaggle authorization does not itself open a node;
  RunPod remains forbidden. Terminal report:
  `TERMINAL_CAMPAIGN_REPORT.md`.
- **PEFT TERMINAL AUTHORITY RECONCILED / 12:27 RECORD CURRENT.** The 12:04
  `BLOCKED_BY_TERMINAL_CAMPAIGN_AUTHORITY` receipt is preserved as a
  correct-at-time historical snapshot, but it is not current. The owner's
  pre-outcome 12:12 Amendment A explicitly overrode that closeout for
  `sa_peft_adapter_v1` only and selected the five nonempty frozen hash folds;
  the serial push followed at 12:19 and the sole v1 terminalized at 12:27.
  Amendment A (`4c4890ee…e63e01`) supplied owner authority; the byte-identical
  remote runtime amendment (`a5b72dbc…1093a`) supplied the executed split.
  The Amendment A `recorded_utc` suffix is malformed, but filesystem +08
  chronology and machine prelaunch/push/terminal receipts establish the
  pre-outcome order. Current quota recheck remains **5.06 h used / 24.94 h
  remaining**, confirming the banked `0.04 T4-h` debit. Seal:
  `tiered_surface/branches/sa_peft_adapter_v1/AUTHORITY_SUPERSESSION_20260731.md`.
- **PEFT §6.2 TERMINAL FAIL / CASE B BACKBONE STOP SIGNAL / PARK-C.** The sole
  T4 v1 completed the fixed 100:1 DELETE probe in `145.680 s` after inline Step
  0 authenticated 90 members in `1.500973 s` with zero competition visits.
  Version delta is exactly `0→1`; authoritative quota moved
  **5.02/24.98 → 5.06/24.94 h used/remaining** (`0.04` T4-h debit). Training
  used all `1,118` positives plus `111,800` fixed negatives; evaluation scored
  all `1,345,497` actions. Descriptive epoch AUC rose
  `0.448881→0.634749`, but AUC was forbidden as a selector. Nested graph-only
  selection chose epochs `1/2/2/2/2`, yielding selected AUC `0.496495`
  versus incumbent `0.502` and frozen case-B cutoff `0.552`. Retention is
  `3/1,118 = 0.002683` versus measured `0.020394`. Adjusted/raw/node graph
  deltas are `+0.000133827/+0.000133160/0`; the binding `+0.005` adjusted
  line fails. This exact PEFT realization is PARK-C, and more capacity on this
  frozen backbone is not indicated. Independent replay passes all 15 checks.
  6bba, CV57, transport packaging, submission, and any full fine-tune remain
  closed. Report:
  `tiered_surface/branches/sa_peft_adapter_v1/TERMINAL_REPORT.md`.
- **C2 ASSIGNMENT-EVIDENCE SEPARATION SCIENTIFIC FAIL BOTH / PARK-C /
  GRAPH BARRIER CLOSED.** The completed hash-pinned C0/K-best contract ran
  CPU-only at full frozen 44b6 coverage (1,345,497 delete + 45,961 occupied
  swaps; 40,894 components; max size 6 / 21 assignments). DELETE, the primary
  class, has mean A1 AUC `0.670235` versus binding `0.85`, with mean lifts only
  `+0.022964` over sealed B0 and `+0.021609` over within-frame A0; no fold
  clears the unchanged `+0.05` B0 lift. SWAP has mean A1 AUC `0.664616`
  versus `0.90` and regresses B0 in all five folds (mean `−0.016530`).
  Retention also fails both classes. Both supplier×class instantiations are
  PARK-C; no graph preregistration, graph run, 6bba, CV57, GPU, submission,
  tier move, or plan amendment is unlocked. Validation passes 10/10.
  Report:
  `tiered_surface/branches/sa_assignment_separation_v1/TERMINAL_REPORT.md`.
- **CELLECT FINAL CLOSEOUT TERMINAL INFRASTRUCTURE FAIL / PERMANENT PARK-C /
  SCIENTIFIC UNTESTED.** The one authorized private version completed the
  mandatory first-action inventory (25,055 files / 87,797,631,517 bytes /
  1,154.319 s) before role resolution, then failed at Step 1:
  `safe loader content resolution expected one match, found 0`. The inventory
  proved the mounted `g02-cellect-r7-published-assets-v1` payload was still the
  DINOv2 carrier: the canonical wheels, frozen population, and B0 were present,
  but both CELLECT safe-loader and checkpoint hashes had zero matches. No
  model, feature, A1, or A0 output exists, so both classes are NOT EVALUATED.
  Quota moved from **4.69/25.31 h** used/remaining to **5.02/24.98 h**
  (**0.33 GPU-h debit**). The precommitted hard rule permanently closes CELLECT
  at PARK-C as UNTESTED with no retry/further spend and no graph unlock.
  Diagnosis:
  `tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_closeout_final_v1/TERMINAL_DIAGNOSIS.json`.
- **DAG-4 DIVISION VALUE RECONCILED / DIAGNOSTIC HALT.** The already-complete
  38-bundle frozen-N1 replay remains a `+0.092067841519` GT-dependent ceiling
  (adjusted/raw/node `+0.001591651042/+0.001607524868/0`, division
  `38/0/4`, Jaccard `0.904761904762`). Information tiers are `(a)=1`,
  `(b)=37`, `(c)=0`; representation closure makes tier (b)
  **UNAVAILABLE**, leaving the honest available diagnostic at only
  `+0.002406489982`, below the `+0.005` amendment line. The banked 25-TP
  curve is `+0.059523810/+0.053191489/+0.048076923` at `0/5/10` FP.
  No amendment proposal, tier move, fitting action, or graph opening follows
  from this diagnostic. Reconciliation:
  `tiered_surface/branches/sa_division_value/REPRESENTATION_CLOSURE_RECONCILIATION.json`.
- **DIVISION FULL-38 FALSIFIED-BY-REALIZATION / PARK-C / D NONBINDING.**
  Plan §6.5
  makes the complete 38-event population (`9` 44b6 + `29` 6bba) and all 57
  authenticated fragments binding for division; per-family numbers are
  diagnostic and association retains 44b6-first. Target-free A/B/C are
  binding at TP/FP/FN `0/23/38`, `0/48/38`, and `0/22/38`, all Jaccard `0`
  and zero duplicates. Clean B's full-38 adjusted/raw/node/combined deltas are
  `+0.000029258/+0.000028813/0/+0.000029258`, but it fails the binding
  division-Jaccard line; A/C are `-0.000022335/-0.000022463/0/-0.000022335`
  each. Clean D's target-free 3-of-3 content is all-zero, but its clean
  preregistration is not pinned by either pre-outcome implementation freeze;
  D is therefore invalid/nonbinding with no retroactive freeze. C genuinely
  tested the biological prior using a
  476,307-byte summary-only cache over 385 candidates; all local arms used no
  Kaggle fallback. Across three independent target-free evidence families,
  realized beneficial division repairs are therefore `0/38`; the surface is
  FALSIFIED-BY-REALIZATION and moves PARK-A to PARK-C without a universal DEAD
  claim. The GT-dependent `+0.092067841519` oracle remains an exact ceiling,
  not a promise. No PASS or downstream unlock exists. Binding report:
  `tiered_surface/branches/sa_division_realizer_wave_v2/WAVE_TERMINAL_REPORT.md`;
  disposition:
  `tiered_surface/branches/sa_division_realizer_wave_v2/FALSIFIED_BY_REALIZATION_20260731.json`;
  D adjudication:
  `tiered_surface/branches/sa_division_realizer_wave_v2/arm_d_consensus_clean/PROVENANCE_ADJUDICATION.json`.
- **GATE RECALIBRATION APPROVED / NODETOPO GENERATOR DIAGNOSIS CORRECTED / HOLD AT BARRIER.**
  The supplier gate now uses measured future-supplier floors: delete AUC
  `>=0.85` / retention `>=0.020`; occupied-swap AUC `>=0.90` / retention
  `>=0.068`, retaining sealed B0/A0, all-five-fold, lift, and FP-budget
  clauses. This is non-retroactive: DINOv2 delete retention `0.001307` remains
  >15x short of `0.020`, so no failed supplier is resurrected and no tier moves.
  AE-P6 and R8 remain PARK-B; their future distinct-supplier gate records now
  cite the measured floors, while the failed Trackastra result remains terminal.
  `restore_topology`'s exact generated population has 50 actions / **0
  beneficial** / oracle ceiling `0.000`; the former `-0.000003436` result is a
  generator failure, not a selector failure. `NODETOPO-O0` remains PARK-A; the
  target-blind `NODETOPO-RECALL61-TARGETBLIND-GEN-V1` preregisters `>0`
  beneficial actions plus a positive oracle ceiling before any selector. Value
  rank: division `+0.092067842` combined ceiling, delete `+0.081897667` with
  AUC `0.85` target, occupied-swap `+0.018571397` deprioritized. APNCC v2
  build/guard checks are locally ready, but an authentic CPU-only Step-0 smoke
  failed closed in `0.010967 s` before heavy work: the local support copy has
  60 of 62 declared wheels (missing `llvmlite` and `polars_runtime_32`). A
  remote v2's former parity-FAIL is superseded by the authenticated
  public-archive adjudication below: transport is VALID, while graph, CV57,
  submission, and tier conclusions remain forbidden. Barrier:
  `selector_requirement_44b6/run_v6/RECALIBRATION_BARRIER_20260731.json`.
- **DIVISION REALIZER WAVE TERMINAL / NO PASS / HOLD AT 44B6.**
  All four arms were preregistered before their prediction outcomes and stopped
  at the frozen 21-fragment 44b6 barrier with zero duplicate survivors.
  Geometry/motion is a valid scientific FAIL: official division TP/FP/FN
  `0/1/11`, Jaccard `0`, adjusted/raw/node deltas
  `-0.000141523/-0.000140538/0`, combined `-0.000141523`.
  Persistence is non-promotable because a pre-prereg structural search exposed
  forbidden target/dossier lines; its sealed target-free predictor is retained
  only as a diagnostic (`0/49/11`, Jaccard `0`, combined `-0.009455519`).
  Appearance failed closed before prediction because no existing frozen cache
  contained adequate physical intensity fields. The frozen 3-of-3 consensus
  therefore abstained everywhere and is invalid due to missing/invalid
  component inputs; its diagnostic arithmetic is `0/0/11`, Jaccard `0`, and
  zero edge/node/combined deltas. No arm satisfies positive combined delta,
  division Jaccard `>=0.30`, and no edge regression; 6bba, CV57, graph, and
  external gates remain closed. Report:
  `tiered_surface/branches/sa_division_realizer_wave/WAVE_TERMINAL_REPORT.md`.
- **CPU BARRIER MERGE COMPLETE / NO TIER MOVE.** The authentic reverse
  Trackastra supplier is now locally merged and remains `FAIL_PARK_C` on both
  sealed classes (delete A1 retention `0–0.003268`; occupied-swap `0` in all
  five folds), so AE-P6/R8 stay PARK-B with no graph unlock. AE-P5's
  complete-N1 C0/C2 falsifier reconstructed 45,961 actions and 40,894
  components, compiled 361 decisive batches with zero rejections, and emitted
  zero edits; AE-P5 stays PARK-A pending a newly named non-inert realizer.
  AE-P7's three-sample raw-image interface smoke is `PASS_INTERFACE_ONLY`,
  not a selector/graph result, and remains PARK-A. DAG-2 local `run_v6` is
  `COMPLETE` with a `HUMAN_REVIEW_REQUIRED` barrier—not `INVALID_RESOURCE`—and
  its independent validation passes 8/8. Merge receipt:
  `tiered_surface/branches/barrier_merge_20260731/BARRIER_MERGE.json`.
- **APNCC-DIRECT PUBLIC-ARCHIVE PARITY PASS / TRANSPORT VALID / NO
  SUBMISSION.**
  The sole `onebitbrain/bhap-transport-smoke-v2/1` remote version passed Step 0
  in `2.715378342 s`: all 62 declared wheel byte hashes, expected manifest,
  and zero competition-path visits, then ran the banked runner (returncode 0).
  Against the authenticated public-0.881 T4 archive `031aa5f6…7219`, the
  263,282-row submission and all four fragments are byte-identical; non-timing
  `run.json` fields also match. The prior 1,096-row deficit compared APNCC to
  the wrong pre-appearance baseline (`bhsubmitout_v3`), so it is a
  **BENIGN_FP reference-selection error**, not numeric drift. The transport
  path is VALID; no CPU/T4 fallback or new launch is warranted. Authoritative quota moved from `4.53/25.47 h`
  used/remaining to `4.69/25.31 h` (**0.16 GPU-h debit**). This is a terminal
  transport result only: no submission, graph, CV57 conclusion, tier move, or
  retry. Receipt:
  `apncc_transport_smoke_v2/PUBLIC_ARCHIVE_PARITY_ADJUDICATION.json`.
- **LANE 1 DIVISION-EMISSION PACKAGE READY / NO SUBMISSION.** A hash-pinned
  post-graph kernel is ready to consume a passing 44b6 candidate receipt:
  frozen-D1 scorer-valid fork validation and one-to-one duplicate suppression
  feed frozen-N1 atomic division-fork compilation. Its BANKED APNCC dry run
  emitted a divisions-ON, empty `submission.csv` with the exact scorer-input
  graph preserved. It did not fit a candidate, rerun the scorer, push to
  Kaggle, or submit. The completed four-arm wave produced no qualifying
  receipt; a future route would require a new explicit human authorization.
- **CELLECT R7 CALIBRATED SWAP GUARD ADOPTED / PERMANENT PARK-C.** Three fresh
  60-second idle windows each measured `0.0 MiB` swapout growth, so the
  adopted threshold is `256 MiB` (`max(256, 5× idle)`), sampled
  continuously every 0.25 s with a 15-s sustained-growth abort and immediate
  abort at `384 MiB`. Historical `+3.5 MiB` (and `+12.875/+30.625 MiB`) stops
  are retired as noise-confounded; `+488.6875 MiB` exceeds the `384 MiB`
  immediate-abort line and remains a real high-pressure signal. CELLECT is
  therefore blocked in practice, and its separate permanent infrastructure
  closeout forbids further spend. No rerun, scientific verdict, or tier change.
  Receipt: `tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1/SWAP_GUARD_ADOPTION.json`.
- **CELLECT R7 PARK-C BLOCKED-ON-INFRASTRUCTURE / NO VERDICT / NO FURTHER SPEND.**
  The exact DINOv2 isolation contract is reproduced locally for CELLECT's
  authenticated 10,197,221-byte checkpoint: 77 declared request shards,
  one child at a time, fresh 45-second preflight, atomic checkpoints, strict
  cumulative zero-swap baseline, and no full-pool cache. Focused tests pass
  8/8. Five preflights each held zero growth, but every immediately following
  shard-0 start caused positive swapout (`+488.6875`, `+3.5`, `+174.6875`,
  `+30.625`, `+12.875 MiB`) under MPS/CPU reductions and was killed by the
  guard. There are zero feature files, zero checkpoints, and no B0/A1/A0 run:
  **CELLECT has no local scientific verdict.** The owner now parks R7
  administratively at PARK-C, blocked on both the consumed Kaggle transport
  stop rule and local memory, with no further spend. Checkpoint size is not a
  volumetric-inference memory proxy: activations, decoded tiles, framework
  workspaces, allocator behavior, and host pressure can dominate the
  10,197,221-byte weight file. Decision:
  `tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1/PARK_C_DECISION.json`.
- **EXTREME-PREVALENCE OPERATING-POINT FINDING BANKED / PROPOSAL ONLY.**
  At roughly `10^-4` beneficial-action prevalence, ROC-AUC is not diagnostic
  of usable campaign lift: the CG0 false-positive-budget tail is binding.
  DINOv2's real rank signal (`0.670/0.682` AUC) retained only
  `0.0013/0.000`; Trackastra and CG1/PU/CRC exhibit the same operating-point
  failure. Future supplier gates are proposed, for human approval, to lead
  with precision/recall-at-budget and report AUC secondarily. No frozen gate,
  threshold, preregistration, or verdict changed.
- **DIVISION PLAN AMENDMENT APPROVED / REALIZER WAVE NOW TERMINAL.**
  The 38/42 event-level bundle table now records required edges, motif,
  preferred late-fork equivalence, and duplicate risk. A fresh fail-closed
  receipt reverified the banked frozen-N1 oracle execution: TP/FP/FN `38/0/4`,
  adjusted-edge delta `+0.001591651042`, division Jaccard `0.904761904762`,
  and exact combined delta `+0.092067841519`. This is GT-dependent ceiling
  evidence only. Human decision §6.4 removed the association-continuity
  prerequisite for fit-only division realizers. That authority was exercised
  by the four separately preregistered CPU-first 44b6 arms, which are now
  terminal with no PASS; GT bundles remained targets, never features. The
  unchanged PASS line was combined delta `>0`, division
  Jaccard `>=0.30`, and nonnegative adjusted/raw/node deltas with zero
  duplicates. 6bba and CV57 remain behind this barrier.
- **DINOv2 FAIL BOTH / CELLECT TRANSPORT STOP / BARRIER HELD.**
  Attempt-6 wheel packaging is repaired with canonical PEP-427 names and
  byte-read manifest SHA checks; targeted step 0 hashes only the checkpoint,
  request/population manifests, and three wheels in `0.04–0.07 s`, never the
  24,968-file competition tree. CELLECT v7 accepts the expanded uncompressed-tar
  checkpoint and verifies member SHA `f3e8e730…89ce5`; genuine member drift
  fails closed. Focused transport/launch tests pass 24/24. Owner-authenticated
  remote audit acquired the exclusive lock, found zero in-flight G02 kernels
  and zero versions of the fresh CELLECT slug, and reconciled Kaggle's
  authoritative balance to **4.53 h used / 25.47 h remaining**; the prior
  3.6435 h named-debit total is audit-only and the **1.65 h** local drift is a
  correction entry. The quota-free DINOv2 v3 route completed all 77 sequential
  one-shard processes / 1,391,458 actions
  with atomic checkpoints and zero swap growth. The unchanged sealed
  B0→A1→A0 gate returned **`FAIL_PARK_C` for delete and occupied-swap**:
  A1 mean AUC/retention `0.670412/0.001307` and `0.682473/0`, respectively.
  DINOv2 graph preregistration stays closed. CELLECT's sole stop-rule-bound v7
  authorization was consumed by exactly one fresh version. It failed at step 0
  after `2.167 s`: the request manifest was absent at the declared
  `/kaggle/input/datasets/onebitbrain/...` mount. No model, feature, B0/A1/A0,
  or scientific verdict ran. The precommitted stop rule is binding:
  CELLECT was then administratively moved to **PARK-C
  blocked-on-infrastructure; GPU bake-off HALT; no seventh run; no further
  spend**, without inventing a scientific verdict.
  Authoritative pre/post quota remained **4.53 h used / 25.47 h remaining**
  (`0.0 h` debit). All downstream hard stops remain closed.
- **RECALL-SUPPLY N0/N1 GO → N2 DEAD / DIVISION FROZEN-CHECKPOINT N0 PARK.**
  The recall 44b6 full-population audit reproduced TP/FP/FN `10,341/1,064/673` and
  found pool-present/absent FNs `47/626`; pool-absent's zero-FP adjusted ceiling is
  `+0.052004594313` (GO at `>=+0.030`). N1 then selected the distinct
  `NDR-FBC-ONE-OFFER` label-free raw-volume registration supplier (three primary
  sources; 105,992 fixed fit117 offers; max `1.013459860492×` pool), but its
  frozen fit-only `NDR-FBC-local-translation-v1` N2 realization recovered only
  `9/290 = 3.103448%` versus the `<20%` DEAD line (inflation `1.002335090457×`).
  The recall route is terminal; N3 is closed and this realization may not be
  tuned or re-screened. Division N0 found `0/>=1` qualifying released frozen external
  checkpoints: no source revision, weight byte/SHA, weight-license receipt,
  corpus-exclusion proof, or source-published 16-GiB inference configuration was
  authenticated, so it is PARK and N1–N3 remain closed. Both lanes remain 0 GPU-h;
  no asset, kernel, push, submission, or tier move occurred; the root recorded the
  node rows in `LEDGER.md`. The external-checkpoint lane remains parked, but
  its former division-fitting dependency was superseded by owner decision
  §6.4 on 2026-07-31.
- **MCC CONTOUR-RETURN SOURCE AUDIT TERMINAL / NO KAGGLE COMPUTE.** The owner
  authorized the one public-source-only Reverse witness, but its pinned
  Apache-2.0 source `1219b669…9041` exposes no deterministic checkpoint and
  requires ordered-contour preprocessing plus source training. Its linked Zenodo
  record `7754589` reports CC-BY-4.0 and three checksummed files, but a faithful
  witness cannot fit the one-T4 <=2 GPU-h ceiling without replacing the
  mechanism or enlarging the budget. No data download, asset, kernel, GPU,
  campaign-data read, feature, or score occurred; this exact route is no-retry
  terminal and cannot reopen Reverse Trackastra. Receipt:
  `dag_prompts/2026-07-30-continuation/reverse/runs/N3_SOURCE_AUDIT_TERMINAL.md`.
- **DPRC-O0 DINOv2-REGISTER PREFLIGHT / NO KAGGLE ACTION.** The owner
  authorized exactly one fresh DINOv2 ViT-S/14-with-registers foreground
  patch-cycle route, never an R5 repair/retry. Official source
  `7764ea0f…25fc8` (Apache-2.0 license
  `600cc67c…935b7`) and the distinct 88,291,785-byte register checkpoint
  `f4331770…641fb` were authenticated by CMUX. No DPRC asset, kernel, model
  load, video decode, feature, outcome, or GPU debit exists. The candidate
  raw roots are available via the competition source, but a fresh
  action/materialization manifest, exact foreground/channel/crop contract, and
  local five-second preflight remain required; calibration must project the
  complete 44b6 delete+swap route to <=4.0 GPU-h or this exact route is terminal
  with no retry. Receipt: `dag_prompts/2026-07-30-continuation/dinov2/runs/N3_SOURCE_ASSET_RECEIPT.md`.
- **AUTHENTIC REVERSE TRACKASTRA TERMINAL / NO GRAPH UNLOCK.** Owner-cmux SDK
  retrieval authenticated the exact historical `model.pt` `24c290ce…981f`
  and `ctc.zip.bin` `2ede0848…a0e`. Genuine reversed-time inference with
  endpoint return `A_reverse[v,u]` is complete on frozen fit-only 44b6:
  36 videos / 3,539 transitions / 84,883 proposals / 121,534 rows, repeat
  witness array-identical, manifest `654cb499…6fe1`. A zero-swap guard stopped
  one partial attempt after an external swapout increase; authenticated resume
  verified 29 files and materialized only seven missing videos, then passed
  with 0 MiB swap growth. This supplier is **NON-GT-IDENTIFIABLE**. Under the
  unchanged sealed five-fold gate, delete A1 retention was `0–0.003268` and
  occupied-swap retention was `0` in every fold; both supplier×class units
  received the evaluator token `FAIL_PARK_C`. No tier move is applied or
  proposed, AE-P6/R8 remain unchanged, and graph, 6bba, CV57, tier, DEAD, plan,
  kernel, GPU, and submission remain closed. Raw result `bd1c3d55…f46e`.
- **DIVISION VALUE DIAGNOSTIC TERMINAL / FIT-ONLY DEPENDENCY UNLOCKED.** The 38 exact frozen
  Wave-3 repair bundles give a combined-score ceiling of `+0.092067842`
  **[GT-DEPENDENT]**, with division TP/FP/FN `38/0/4`, Jaccard `0.904762`,
  adjusted/raw/node deltas `+0.001591651/+0.001607525/0`. The deployment-time
  audit assigns only **1/38** events to current-pool tier (a), worth
  `+0.002406490` **[NON-GT-IDENTIFIABLE]** even at perfect selection, below
  the `+0.005` actionability target. The remaining 37/38 require an unbuilt
  stronger representation; their `+0.092067842` (a+b) value is a
  representation-conditioned upper bound **[NON-GT-IDENTIFIABLE]**, not a
  realized selector gain. DAG-2 division status was
  `NOT_EVALUABLE_NO_DIVISION_SELECTOR`; owner-approved §6.4 later opened the
  separately frozen 44b6-first realizer wave, which is now terminal with no
  PASS. Report:
  `tiered_surface/branches/sa_division_value/FINAL_REPORT.md`.
- **CELLECT SINGLE-BUNDLE INTEGRITY PASS / NO KERNEL LAUNCH.** The 77 frozen
  request shards (1,391,458 actions) and unchanged population manifest
  `4ab7bb23…76f0f` are now consolidated with the byte-exact v4 checkpoint,
  safe loader, and three wheels in the sole private input dataset. A first
  version-5 packaging attempt was rejected because the CLI skipped the nested
  request directory; corrected version 6 flattens the request filenames and is
  READY. Owner-cmux download plus the existing verifier authenticated all 87
  files byte-exact under bundle manifest `ae3bc9b2…7fecd3`; the expanded
  checkpoint remains `f3e8e730…89ce5`. Kernel metadata now names only
  `g02-cellect-r7-published-assets-v1`. Universal pure-stdlib step 0 authenticates
  the bundle, positive request count, population manifest, checkpoint, safe
  loader, request payloads, and wheels before NumPy/Torch/pip/model work; local
  valid preflight took 0.0545 s and focused tests pass 21/21, including
  missing/empty request manifests aborting under 5 s. No kernel, GPU debit,
  submission, RunPod, graph, full cache, threshold, gate, tier, or DEAD action.
- **SELECTOR-REQUIREMENT DIAGNOSTIC TERMINAL / HUMAN REVIEW REQUIRED.** Isolated
  delete and occupied-swap are valid and complete on frozen 44b6. Delete:
  1,345,497 / 1,118 beneficial, oracle-best adjusted delta `+0.081897667`,
  140/140 noise runs. Occupied swap: 45,961 / 134 beneficial, oracle-best
  `+0.018571397` at prefix 32,768 / applied support 29,574, 140/140 noise
  runs. Its repaired guard ran 2,221.571 s at 1.017 GiB RSS and 0 MiB swap.
  Restore is valid and complete at 50 actions / 0 beneficial: oracle-best is
  the zero-action point (`0` adjusted delta), so +0.005 is infeasible on the
  frozen population and AUC/retention is undefined rather than estimable.
  Its guard ran 349.107 s at 1.115 GiB RSS and 0 MiB swap.
  A concurrent transport-only source generalization was accepted under local
  repair 5 after a restore launch failed closed before its child process; the
  local root and frozen ZipStore remain unchanged and 22/22 tests pass.
  Mean +0.005 requires delete AUC `0.85` / retention `0.020394` at prefix
  16,384 and occupied-swap AUC `0.90` / retention `0.067910` at prefix 1,024.
  Robust p05 requires delete AUC `0.90` / retention `0.044946` and swap AUC
  `0.95` / retention `0.147388`. Frozen AUC `0.65` / retention `0.15` is
  **TOO LENIENT** for both feasible classes; restore is
  `NOT_APPLICABLE_POPULATION_LIMIT`. Amendment proposed, not applied;
  independent validation passes 8/8 at 0 MiB swap. No threshold or tier change.
- **SELECTOR-REQUIREMENT KAGGLE-CPU TERMINAL INVALID / NO RETRY.**
  The frozen DAG-2 population, folds, probe, scorer, 7×20 AUC/seed grid,
  supports, thresholds, and three action classes are unchanged. A pre-outcome
  transport amendment replaces only fixed macOS paths, the local ZIP store, and
  swap guard with a 16 GiB Kaggle CPU session (`enable_gpu:false`) reading the
  competition's extracted Zarrs. The private input archive contains 336 members,
  62 offline wheels, 1,391,458 frozen edge actions, and all exact baseline
  fragments; SHA-256 is `4bb1c4a6…e6da3`. Pure-stdlib step 0 pins that archive,
  population `4ab7bb23…76f0f`, positive action/wheel counts, and all 41 required
  competition videos before extraction, pip, NumPy, zarr, tracksdata, or scorer
  work. Focused scientific tests remain 22/22 PASS. The owner-cmux upload
  created READY assets version 7 as a temporary CPU carrier. The owner's manual
  download verified all 335 frozen inventory records; Kaggle expanded only
  `action_population.jsonl.gz`, whose downloaded plain bytes deterministically
  recompress to original SHA `6be1c77b…a5324`. Repaired pure-stdlib step 0
  validates every expanded member and this reversible transform in 0.592 s.
  Private kernel `g02-selector-requirement-cpu-run-v1` version 1 used CPU only
  and stopped at 0.858 s before step 0 because Kaggle packaged only the declared
  code file, not its sibling preflight module. No input, dependency, scorer, or
  scientific outcome was consumed. Its 0.0019269 CPU-h is recorded separately;
  GPU budget remains exactly 2.8847 used / 27.1153 remaining. Execution repair
  1 was locked before v2 and only inlined the unchanged preflight. V2 stopped
  inside step 0 at 197.23 s: Python 3.12 emitted a platform OS byte in the
  deterministic gzip header, and hashing the entire 595 MiB mounted expansion
  violated the first-seconds envelope. No pip, imports, inputs, or outcomes were
  consumed. V2 used 0.0565042 CPU-h; cumulative CPU is 0.0584311 h and GPU
  budget remains exactly 2.8847 used / 27.1153 remaining. Execution repair 2
  is locked before v3: normalize the gzip OS byte and use the manually
  byte-verified pinned inventory for exact presence/size checks plus manifest,
  inventory, population, and reversible-gzip hashes. Local step 0 is 0.0222 s;
  science and all downstream deep pin checks are unchanged. V3 stopped before
  heavy work because Kaggle lazy-mount traversal took 225.61 s and exceeded the
  five-second envelope. Across v1–v3, no dependency install, scorer, or outcome
  ran; **0.123161 CPU-h / 0 GPU-h** is recorded. The requested Kaggle rerun is
  terminal infrastructure-invalid with no scientific verdict and no retry.
  The unused v4 port is locally repaired and **WITHDRAWN**: its declared code
  file contains and time-asserts step 0 before 5 s; a twice-reproduced
  630,323,200-byte uncompressed tar accepts Kaggle-expanded directories and
  verifies 375 members by SHA-256 without any container round-trip. Genuine
  member drift fails closed; transport/existing tests pass 4/4 and 14/14. No
  CPU dataset upload or kernel v4 occurred. Receipt
  `selector_requirement_44b6/KAGGLE_CPU_PORT_WITHDRAWAL_V4.json`.
- **SELECTOR-REQUIREMENT LOCAL HISTORY / SUPERSEDED BY CPU ROUTE.** The
  original and streamed-lifetime processes remain invalid at +16.31 MiB and
  +1,567.69 MiB swap with no consumed output. Native-goal continuation has
  frozen execution repairs 3–4: delete, occupied-swap, restore-topology, and
  aggregation run as separate freshly guarded processes; the swap stage removes
  the heavyweight training-module import while preserving source-equivalent
  frozen matching/conflict semantics. Repair 4 changes only swap compiler
  receipts: each greedy conflict-disjoint set is compiled as one equivalent
  frozen N1 SWAP primitive union instead of an atomic batch that revalidated
  the full baseline per action. A direct fixture proves identical final graph
  hash. Scientific inputs, 7×20 AUC/seed grid,
  supports, compiler/scorer, thresholds, and tiers are unchanged. Focused tests
  pass 22/22. The isolated delete class is now valid and complete:
  1,345,497 actions / 1,118 beneficial, 140/140 noise runs, oracle-best
  adjusted delta `+0.081897667` at prefix 2,048, 286.858 s guard, 1.081 GiB
  peak RSS, and 0 MiB swap. The superseded swap receipt attempt was stopped
  after 10,922.910 s / 0.997 GiB / 0 MiB swap with no terminal artifact or
  exposed outcome. Swap and restore remain pending; no aggregate
  requirement/verdict exists yet.
- **DINOV2 R5 KAGGLE-CPU TRANSPORT TERMINAL / NO ACTIVE CPU RUN.** The unchanged
  source revision `7764ea0f…5fc8`, tree `2a27257b…f43`, checkpoint
  `b938bf1b…0cd9` (88,283,115 B), XY-beta contract, 77 requests,
  1,391,458 actions, and 1,417,863 unique endpoints are packaged under a
  pre-outcome CPU transport amendment. The 148,679,264-byte archive
  (`580499a2…ca40`) has 167 members, three offline wheels, the exact population
  `4ab7bb23…76f0f`, and a binary official-source checkout. Pure-stdlib step 0
  passes locally in 0.003668 s before pip/NumPy/Torch/zarr/model/inference.
  The owner-cmux upload created READY private assets version 8 as a temporary
  CPU carrier. The owner's manual download verified all 166 frozen inventory
  records byte-exact; Kaggle only expanded the inner bundle ZIP, with no
  missing, unexpected, or transformed payload. Receipt `2898e481…e487`.
  Private kernel `g02-dinov2-r5-cpu-run-v1` version 1 stopped before heavy work
  because recursive manifest discovery traversed the large lazy-mounted
  competition tree and breached the five-second step-0 envelope at 190.30 s.
  No pip, model load, inference, or outcome ran. Terminal usage is 0.0548687
  CPU-h / 0 GPU-h; cumulative CPU is 0.1780297 h and the GPU budget is
  unchanged. Execution repair 1 is locked before v2: use the exact private
  dataset manifest path, with all science and limits unchanged. The manually
  verified v8 expansion passes repaired step 0 locally in 0.0020515 s. V2 did
  not repeat the traversal failure; it stopped at 0.913 s because the repair
  used `/kaggle/input/<slug>` instead of Kaggle's evidenced private mount
  `/kaggle/input/datasets/<owner>/<slug>`. No heavy work or outcome ran. V2
  used 0.0022223 CPU-h / 0 GPU-h. The cross-campaign audit now records 17
  incidents in 14 root-cause families, including exact error/log hashes and a
  same-signature no-retry rule. A dedicated immutable dataset now owns this
  lane. Remote Kaggle preflight authenticated the owner-qualified mount,
  1,391,458 requests, 77 request and 77 population files, three wheels,
  checkpoint/source/input/population hashes, and GPU-disabled metadata in
  0.054277 s before heavy work. The shared launch guard and 12/12 regression
  fixtures pass; the real DINO step 0 passes 2/2 missing/empty-manifest
  fail-closed tests under five seconds. CPU-only scientific kernel v3 was
  pushed once, then failed closed inside step 0 at 0.992640 s because it found
  no exact private-dataset manifest. No dependency install, model load,
  inference, or outcome ran. V3 used 0.0022130 CPU-h; cumulative recorded CPU
  use is 0.1847631 h and GPU use remains 2.9035 h. Receipt
  `sa_dinov2/KAGGLE_CPU_V3_TERMINAL_INVALID.json`. A causal lineage-promotion
  push then exposed title/ID slug drift: Kaggle accepted it under actual ID
  `g02-dinov2-r5-cpu-promoted-preflight/2`. Owner-authenticated reconciliation
  now confirms that run is terminal ERROR. Its step 0 passed in 0.053483 s for
  all 77 requests / 1,391,458 actions, then the authenticated local projection
  was 38.662 h and failed closed before a result manifest or scientific
  outcome. It debits 0.1163353 CPU-h / 0 GPU-h; cumulative recorded CPU use is
  0.3010984 h and no CPU session remains active. KFG-15/16 bind preflight to
  kernel ID and reject title/ID drift; shared tests pass 14/14 plus real
  missing/empty-manifest tests 2/2. No additional CPU launch is permitted.
- **VIDEO-SUBSAMPLE FIDELITY UNSAFE / ROOT BARRIER COMPLETE.** A
  pre-outcome common-universe amendment standardized APNCC, C1, H1-ZWRFEAT,
  SA-DELETE, SA-SWAP, and NODETOPO on the already-opened 36-video held-44b6
  gate and common CG0 reference. All fragment hashes and eight aggregate
  replays matched within `1e-12`; 12 cells each used 10,000 fixed-seed
  finite-population draws. **0/12 were usable.** At `k=12`, all schemes reached
  median rho `0.828571`, but sign agreement was only `0.813733–0.820683` and
  maximum 95% width was `0.023017–0.023308`, failing the `0.9` and `<0.005`
  gates. No `SUBSAMPLE_CONTRACT` exists; full frozen 44b6 remains mandatory for
  every promotion. Table `67cf33ed…9a3b9`; barrier
  `tiered_surface/subsample_fidelity/BARRIER_REPORT.json`. No tier move, GPU,
  Kaggle, budget debit, RunPod, submission, DEAD, plan, N1, folds, or P0 action.
- **TWO-SUPPLIER BAKE-OFF ACTIVE / CELLECT INVALID / GRAPH BARRIER CLOSED.**
  The prior DINOv2 and CELLECT attempts remain infrastructure-INVALID, not
  scientific failures. Their shared frozen gate `9fb030bd…3310e`, sealed B0
  `a9ce8c0b…b4ff2`, population `4ab7bb23…76f0`, folds, probe, and thresholds
  are unchanged. The stale DAG-2-null reference is superseded by run_v6
  requirement surface `6e62a29c…6c71b`: mean/robust-p05 AUC requirements are
  delete **0.85/0.90** and occupied swap **0.90/0.95**. SWAP-O0 stays PARK-A,
  but its compiler-valid 44b6 ceiling basis is corrected from legacy CG0
  `+0.075828` to run_v6 **`+0.018571397`**. CELLECT N2 now passes on the
  owner-downloaded v6 payload: repaired shared step 0 resolves the bundle,
  request manifest, and all 77 request shards uniquely by pinned content,
  authenticating 1,391,458 frozen actions in 0.101 s before heavy work.
  Opaque names/parents and fail-closed fixtures pass 24/24 shared plus 7/7
  independent tests; the self-contained kernel compiles. The sole v6 T4 run
  then failed closed in step 0 at 3.956 s because the exact private-dataset
  mount exposed zero matches for the pinned bundle manifest. No heavy import,
  checkpoint load, 64-tile projection, feature, B0/A1/A0, or outcome occurred;
  CELLECT is infrastructure-INVALID with no verdict and no retry. Conservative
  debit is 0.0188 GPU-h, leaving 27.0965 h and no active authorization. DINO
  local class-isolated MPS delete passed a fresh 45.209 s zero-swap preflight
  and authenticated a 4.36294 h full-class projection after two released
  videos, but its guard terminated at 7,877.258 s after 22/41 videos when swap
  grew 140.875 MiB. There is no delete class manifest; partial shards are
  quarantined, occupied swap and aggregation stayed closed, and no outcome was
  consumed. DINOv2 is resource-INVALID with no verdict/no retry. No graph
  contract/run, training, full cache, 6bba, CV57, submission, RunPod, tier
  move, PARK-C, or threshold change has occurred. The superseded v2 cross-read
  recorded zero PASS, zero FAIL, zero PARK-C moves, and zero graph unlocks.
  The human-prioritized DINOv2 Kaggle GPU transport is now
  **TERMINAL INFRASTRUCTURE-INVALID / NO RETRY**. Its generated
  kernel is one self-contained file; step 0's first call recursively prints
  every `/kaggle/input` file with byte size and SHA-256 before decoding or
  resolving any embedded contract. An opaque expanded-mount fixture resolves
  the 77 requests, pinned checkpoint `b938bf1b…0cd9`, official source tar, and
  three wheels uniquely by content hash; missing and duplicate hashes fail
  closed, and embedded A1/A0 feature math is array-identical to the frozen
  implementation. The dataset contains no external bundle manifest and no
  partial features. All 724,634 quarantined MPS rows are pre-outcome discarded
  because MPS→CUDA numerical equivalence is unproven and no class manifest
  exists. Both classes will be recomputed; no held-fold partial can leak.
  Private assets `onebitbrain/g02-dinov2-r5-gpu-assets-v1` are READY at version
  1 after 82 successful payload receipts. Serial prelaunch found both prior
  campaign kernels terminal ERROR and no new GPU kernel. A concurrent push race
  violated the one-run serial authorization: version 1 was already RUNNING when
  the root launch call created byte-identical version 2. The exact latest-v2
  terminal evidence completed the 24,968-file / 87.726 GB inventory in
  1,242.976 s, resolved all 77 requests, weight, source, wheels, and competition
  root, and emitted step-0 PASS. It then failed closed because pip rejected the
  content-hash-prefixed wheel basename, before dependency completion,
  source/model/worker work, features, B0/A1/A0, or outcome. Version-addressed
  private status/output is unavailable; because v1 has the same invalid pip
  basename it cannot reach model or feature work. The full owner-authenticated
  3.79→4.53 h quota delta, **0.74 GPU-h**, is debited for both remote runs
  rather than only v2's independently timed 0.3989 h. The earlier CPU-promoted
  run is also now terminal:
  owner-authenticated status is ERROR; step 0 passed in 0.053483 s, then its
  authenticated projection was 38.662 h and failed closed with no result
  manifest or outcome. It debits 0.1163353 CPU-h. No external session or GPU
  authorization remains active. The repo GPU-debit subtotal is 3.6435 h used /
  26.3565 h remaining, while launch capacity uses the lower owner-authenticated
  postrun balance, **25.47 GPU-h**. Exact pushed code, inventory, step-0
  receipt, terminal payload, and logs are banked under
  `kaggle_gpu_bakeoff_v1/terminal_evidence/`.
  CELLECT bleed-stop v1 is locally
  **MOUNT-PASS / NOT LAUNCHED**: the generated kernel embeds all 77 request
  semantics and required content pins, resolves an opaque 83-file
  Kaggle-expanded fixture with no external bundle or request manifest, and
  fails closed on missing or duplicate content. Its post-step0 science suffix
  is unchanged; independent/shared tests pass 7/7 and 24/24. Neither supplier
  has a scientific verdict; CELLECT has no launch authorization. The canonical
  cross-read is
  `two_supplier_comparison_v3_gpu_terminal/TWO_SUPPLIER_COMPARISON.json`
  (`454bc4e0…acf8f`): all four cells remain NOT EVALUATED, both suppliers remain
  PARK-B, and no graph preregistration is unlocked.
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
  `29fa828f27e390034570e953ffa0dd14cbd7cd754ebbc879237905f28a8f3f5f`
  records **LIVE 0 / PARK-A 14 / PARK-B 18 / PARK-C 12 / DEAD 1** after
  the owner-approved R7 CELLECT PARK-B→PARK-C move and the terminal full-38
  DIV-F1 PARK-A→PARK-C disposition.
  Wave-3 accepted three cited moves: AE-P5 and AE-P7 PARK-B→PARK-A after
  exact contracts plus positive occupied-swap P0 headroom, and G2 PARK-B→PARK-C
  for the exact `SA-SWAP-W1-EXACT-REGRET` instantiation only. AE-P1/AE-P2 and
  the broader decision-focused family remain unchanged. Decisions
  `202cc2a5…bec9`; **accepted/uncited moves 3/0**.
- **CPU-ONLY WAVE-3 BARRIER COMPLETE.** The authorized independent NODETOPO
  replay confirmed `BLOCKPEAK-RESTORE-V1` **KILL_44B6_STOP** at
  adjusted/raw/node `−0.00000343615 / 0 / 0`, with 50/50 actions, 24/24 affected
  batches, 36/36 videos, unchanged edge TP/FP/FN `10,341/1,064/673`, and no
  DET-F0/F2/F3 or NODERESTORE unblock. That exact instantiation is PARK-C;
  broader NODETOPO-O0 remains PARK-A and NODERESTORE-O0 PARK-B.
  DINOv2 ViT-S/14 is now authenticated at official source revision
  `7764ea0f…5fc8`, weight SHA `b938bf1b…0cd9`, 22,056,576 parameters, and a
  strict safe CPU load; R4's provenance block is cleared and R5's XY-beta
  contract is staged as a second supplier beside CELLECT, but no feature or
  outcome ran. Reverse Trackastra now has an exact endpoint-return contract but
  stops on a missing exact checkpoint/reverse cache with a bounded
  **0.758 CPU-h / 0 GPU-h** request. Division continuity now has exact bundles
  for all 38 fragmented events (39 adds, 27 net deletes; 36 late equivalents);
  its then-current fitting lock is superseded by §6.4. No GPU, Kaggle, budget
  debit, RunPod, submission, DEAD, plan, or CELLECT action occurred in that wave.
- **CPU-ONLY WAVE-2 BARRIER COMPLETE / NODETOPO 44b6 GRAPH KILL.**
  `SA-NODETOPO-GEN` authenticated all 36 fit105 44b6 Zarrs and 3,600 frames,
  froze 107,370 label-free proposals plus an exact 50-action
  `restore_topology` population across 24 videos, and passed frozen N1 for
  50/50 singles and 55/55 nonempty budget batches. The released graph gate
  replayed all 50 actions and 24/24 affected-video batches exactly, but stopped
  at 44b6 with adjusted/raw/node deltas
  **`−0.00000343615 / 0 / 0`** against the unchanged
  `+0.005 / 0 / 0` floors. Edge TP/FP/FN stayed
  `10,341 / 1,064 / 673`; 6bba and CV57 were not opened.
  `SA-DIVISION-DEEP` found
  **38/42 CV57 divisions present-but-fragmented**, one role-present
  component-spanning event without a local fork, and three role-missing events;
  39/39 diagnostic actions compiled; its then-current fitting dependency is
  superseded by the 2026-07-31 §6.4 amendment. `SA-PARKB-SWEEP` audited 14
  CPU-testable PARK-B nodes, skipped eight weight/GPU-gated representation
  nodes, and proposed no move. Optional `SA-ASSOC-CYCLE` authenticated 36/36
  forward-cache files but found no immutable reverse supplier, so separation
  and graph stayed closed. Barrier `c60799d9…c63f3`; validation
  `6a467695…5c0f8` (**48/48 PASS**). Accepted and uncited tier moves are 0/0;
  census stays 0/13/22/9/1. No additional graph beyond the NODETOPO 44b6
  barrier, GPU, Kaggle launch/debit, CELLECT write, RunPod, submission, or DEAD
  action occurred.
- **CONCURRENT CPU WAVE / QUALIFIED BARRIER; ALL LANES HELD.** PARK-B support/identity sweep
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
  moves 0, census unchanged. Posthoc independent review quarantined the PARK-B
  fragment because it referenced an unmerged NODETOPO artifact; the reference
  drove no accepted move. It also found that the root evidence chain is not
  self-contained at clean checkout and that NODETOPO's original receipt omitted
  its top-level verdict; the hash-pinned terminal report supplies
  `PARK_RESOURCE_GUARD`. Review `27d9a87e…8db9e10`.
- **OWNER COMPUTE GOVERNANCE:** Kaggle GPU is authorized as needed within
  **30 h/week cumulative**, with hours banked every run. **RunPod is forbidden.**
  Public submission remains a separate explicit human gate. Local runs retain the
  45-second idle preflight and zero-swap rule. **Standing authentication rule:**
  every authenticated Kaggle CLI list, push, or submit must run through
  computer-use cmux, where the owner's real session lives; the agent-shell token
  must never be used for those operations.
- **CELLECT R7 SEPARATION / V4 TERMINAL INVALID.** The authenticated-weight
  per-class amendment is frozen at `9fb030bd…3310e`; exact fit-only 44b6 populations completed under
  guard with delete **1,345,497 / 1,118 beneficial** and occupied-swap
  **45,961 / 134 beneficial** actions. The official checkpoint matches pinned
  commit `35860709…c63236` at SHA-256 `f3e8e730…89ce5`, is GPL-2.0-only,
  passed guarded tensor-only local loading, and was registered in private Kaggle
  asset v3. Both-class B0 is sealed at `a9ce8c0b…b4ff2`. The one authorized v3
  kernel still observed zero mounted files under the exact checkpoint name and
  stopped before checkpoint load or supplier features. No A1/A0 metric or
  scientific verdict exists across any attempt; total CELLECT invalid-run debit
  is **0.4591 kernel-h**. The v3 manifest staged **19,467,606 bytes**, including
  **9,270,385** non-checkpoint bytes; the remote total of **18,508,409 bytes**
  therefore implies a **9,238,024-byte** checkpoint, **959,197 bytes smaller**
  than the authenticated **10,197,221-byte** file. Direct `.pth` transport is
  invalid. A deterministic local v4 package now wraps safe-ASCII
  `cellect_uext_x3rd_149_0_4p6540.pth` inside an uncompressed tar and recovers
  SHA-256 `f3e8e730…89ce5` at kernel step 0 before dependency installation.
  The earlier rules/entitlement diagnosis is retracted: this is the user's own
  private dataset, and the failure is in the CLI/OAuth token-scope path. After
  owner reauthentication in cmux, status and upload succeeded and private asset
  **v4 is READY**. The legacy `ListDatasetFiles` and metadata-preflight endpoints
  still return 403, but an authenticated SDK download in that same cmux owner
  session recovered all seven assets. Kaggle expanded the uncompressed tar during
  materialization; the safe-ASCII checkpoint round-tripped at exactly
  **10,197,221 B** and SHA-256 `f3e8e730…89ce5`, with all other payload hashes
  exact. Focused tests are **17/17 PASS**. The subsequently authorized v4
  kernel authenticated that checkpoint at step 0 but discovered zero frozen
  request files and stopped before feature computation. Its 0.3086 kernel-h
  debit makes the cumulative CELLECT invalid-run debit **0.7677 kernel-h**. No
  A1/A0 or scientific verdict exists; R7's historical PARK-B state here is
  superseded by the 2026-07-31 administrative PARK-C decision. Infrastructure-fix
  contract `a1e6180a…3398`; remote receipt `4aa00106…ecd0`; v4 terminal receipt
  `phase2_separation_gate/run_v4/TERMINAL.json`.
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

## Terminal lever state

There is **no live lever**. The supplier-agnostic
`R-SEPARATION-FALSIFIER-44B6` contract, its exact 1,391,458-action population,
sealed B0, folds, alignment control, probe, and measured requirements remain
banked as evidence. DINOv2 completed the full population and failed both
classes; C2 assignment evidence independently completed the same population
and failed both classes; CELLECT is permanently PARK-C / UNTESTED after six
infrastructure failures. No supplier-specific graph contract is open.

The representation and division findings close the current pointwise
beneficial-action formulation: independent evidence families converge near
DELETE AUC `0.670`, below `0.85`, while 37/38 division events worth
`+0.089661334` require a stronger representation and none are GT-only. This
surface can reopen only for a new action-labeling estimand or independent
exact-task evidence of AUC `>=0.85`. Another appearance encoder does not
qualify. `TERMINAL_CAMPAIGN_REPORT.md` is authoritative for the closeout.

## Final census / closed fallback registry

This registry is descriptive, not an executable queue. The terminal reopening
condition above governs every future proposal.

| Queue | Surface | Tier | 44b6 evidence | Headroom / next condition |
|---:|---|---|---|---|
| closed | `AE-P5` | PARK-A parent / PARK-C C2 instantiation | full assignment-evidence separation failed both classes | new action-labeling formulation or independent exact-task AUC `>=0.85` evidence |
| 1 | `AE-P7` | PARK-A | no graph; exact phase-template contract committed | `+0.018571397`; fit105 identity/runtime smoke |
| 1 | `AE-P1` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.018571397`; newly named realizer only |
| 2 | `AE-P2` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.018571397`; newly named realizer only |
| gated | `SWAP-O0` | PARK-A | exact-regret instantiation `−0.000310288/−0.000296592/0` (PARK-C breadcrumb) | compiler-valid run_v6 ceiling `+0.018571397` (`6e62a29c…6c71b`); representation falsifier PASS |
| closed | `R5` DINOv2 XY | PARK-C | local 77-shard supplier gate returned `FAIL_PARK_C` for delete and occupied-swap | no graph preregistration; new named supplier only |
| closed | `R7` CELLECT | PARK-C | six infrastructure failures; no scientific verdict | permanent no-further-spend / scientifically UNTESTED |
| closed | `G2` exact regret | PARK-C | `−0.000310288/−0.000296592/0` | new named decision-focused mechanism only |
| closed | `DELETE-O0` | PARK-C | exact retention beta `−0.000127119/−0.000131807/0` | materially new named deletion mechanism |

## Key numbers
- **Public-frontier calibration (2026-08-06, `public_solution_landscape_survey_v1.md`).**
  Live LB snapshot 2026-08-05T19:22:39, 2,019 teams: max `0.948`, **zero teams ≥0.95**,
  4 teams ≥0.94, 95 teams above our CV57 `0.914358` (rank ~96). The public 0.952–0.969
  notebook badges are stale pre-patch metric-exploit artifacts (host patch `aa65e90`,
  2026-07-17) and must not be used as targets. Best clean public author: pilkwang `0.921`
  (+0.007 over us). Our CV57 division term: `1 TP / 11 FP / 12 FN`, J=`0.0417`, 368 forks
  emitted against 13 GT divisions → ≈`0.096` of the 0.1-weighted division term unclaimed.
- Division-parent capacity: Kaggle `24.73 h` remaining minus V32 reservation
  `7.67 h` leaves `17.06 T4-h`; post-taxonomy model projection is not yet
  measured. Current committed split: Kaggle `0.0`, Colab `{}` by exact GPU
  model, RunPod `0.0 h`. RunPod SGD 20 planning envelope is not a launch
  approval. Colab durability cadence is `600 s`, manual session target `4--6 h`.
- V32 unsharded replication (local PASS; hidden Version 1 running): 162 frozen identities in one immutable
  V32 package; authenticated projection `6.8 h < 8 h`; ledger reservation
  `7.67 h`; authoritative launch bracket `5.27/24.73→5.27/24.73 h`
  used/remaining. The three authenticated 27-video V32 precedents total
  `10,028.512203 s`, a linear prediction-only `5.571396 h` for 162; the
  owner-amended whole-run projection carries the remaining materialization and
  terminal-output allowance. Exact Version 1 output is authenticated at
  162 datasets / 6,261,255 rows / 3,191,037 nodes / 3,070,218 edges; two
  degree-3 parents are preserved and fully logged. Quota is `11.70/18.30 h`,
  exact debit `6.43 h`, and zero reservations remain. Exact frozen CV57 is
  `0.914358473021` versus public `0.908--0.911` and APNCC `0.881522069960`
  (`+0.032836403061`); score receipt `8827cd15…93c2`. Taxonomy is PASS 57/57
  with zero replay mismatches, V32 ordinary-association events `1931`,
  fragmentation FN `1053`, and division `3/69/39`; comparison
  `8b72d8d3…ef07`. Fresh hidden package `4435e834…a03` / source
  `5193d060…b619d` discovers exactly four mounted test Zarrs dynamically and
  cannot copy public output; Version 1 is launched under a `0.50 h` active
  reservation and `0.30 h` projection, with no launch-time quota debit.
  Terminal output predicate/submission and the separate cap-to-two experiment
  remain pending; no V32+G1 result exists. The immutable active
  source has no persisted JSONL run-log binding, so that irrecoverable logging
  gap will be reported independently of replication validity.
- G04 hidden-rerun identity audit: no additional live code-submission path
  hard-codes mounted test identities. The only match is the registry-quarantined
  historical copy transport; APNCC submit mode and the G04 postprocessor both
  enumerate the mounted `test/*.zarr` population and require exact inventory
  agreement. Ref `55203439` already supplies the one end-to-end blind-rerun
  canary: it terminalized COMPLETE at public `0.881`. A duplicate identical
  launch is forbidden by the frozen one-shot/no-retry contract. Focused
  dynamic-discovery/guard/lifecycle suite: `21/21` PASS. Report:
  `../../g04/lanes/division_permissive/SUBMISSION_PATH_IDENTITY_AUDIT_20260803.md`.
- Data pathology: all-archive frozen/jump `967/227`; CV57 frozen/jump
  `282/75`; G1 exact `0.886764420260`, delta `+0.005242350300`, edge
  `34389/2308/1879`; flagged-pair reachable GT continuations
  `209/311→259/311`; GA-R12REACH flagged shell rescue `90/93`. F1 and exact
  stride parity both delta zero. Crop-wrfeat v1 `231/243` finite,
  `199/243` boundary-contact, `766.233` projected CV57 CPU-h: terminal.
- `VELOCITY-AWARE-CANDIDATE-WINDOW-v1` is now integrity-repaired and exact-
  scored on the G1 base. The actual fit117 G1-selected graph conserves
  `456−90=366` with predecessor vectors, G1 shifts, graph-endpoint presence,
  and cap-eight ranks sealed. Fixed-r10/cap8 recentering clears its ceiling
  (67 new true continuations, 98/366 residual-shell retention, `0.993272×`
  background burden) but loses 61 other true continuations. Exact CV57
  adjusted/raw/node is `0.880534473728/0.885184801739/0.990377056057`, delta
  `−0.006229946531/−0.006276210289/0` versus G1, edge TP/FP/FN
  `34200/2368/2068`. The realization is PARK-C / no submission; G1 remains the
  positive. Report:
  `../../g04/lanes/velocity_aware_candidate_window_20260804/REPORT.md`.
- **Division requirement surface measured / daughter head banked / event head
  PARK-C.** Exact sealed
  re-analysis finds a material `+0.005` division contribution needs at least
  5 TP / <=17 FP in the first 22 actions (precision@22 `>=0.2273`) and a
  conservative deployment AP-lift floor `8462x` versus the pair denominator;
  the frozen model has only `18.41x`. Its event head is AUROC/AP
  `0.521521/3.1367e-5` over 2,858,003 parents, whereas its conditional pair
  head is `0.897081/0.368520` over 2,292 pairs. Of 18 unreachable divisions,
  11 are detection-universe absent, 4 nearest-8 absent, 3 pair-universe
  absent; synthetic has no r10 gate. Division itself is not unreachable:
  conditional daughter assignment is solved, while division-event detection
  has no working mechanism under the appearance-free contract. CTC real
  divisions are the leading appearance-event dependency but remain
  `PARK-A-PERMISSION` because official terms require written permission for
  this non-CTC public scientific use; citation alone is insufficient. Until an
  authorized event detector reaches the measured deployment operating point,
  redirect work to ordinary association (`+0.061628263275`). Receipt:
  `../../g04/lanes/synth_div_sim2real_v1/REQUIREMENT_SURFACE_V1.json`.
- CV57 residual taxonomy (exclusive 57/57 matcher replay): edge
  `FP/FN=2459/1948` partitions into ordinary association `2459 FP`, detection
  `466 FN`, and matched-lineage fragmentation `1482 FN`; division is `42 FN`.
  Exact Shapley impacts are `+0.061628263275/+0.017446063106/+0.039403603659/
  +0.100000000000` respectively. Division is the largest total-score ceiling,
  ordinary association the largest edge surface; the all-categories theoretical
  ceiling is `1.100000000000` (`+0.218477930040`), not a deployable projection.
  Report: `../../g04/lanes/cv57_residual_error_taxonomy_20260804/REPORT.md`.
- Active-path P1 seam: `pipeline.py` now delegates only its existing
  validation/input normalization to `pipeline_inputs.py`, preserving public API
  and a deterministic `predict_dataset_window` graph-SHA fixture. Focused
  validation/routing/solver/import/compile checks pass; no generated, archived,
  or PARK-COST module changed. The active final-output audit also confirms both
  `scoreall --submission` and `--frags` are strict before scoring; six malformed
  edge-contract regressions cover both loaders. Receipts:
  `../../../code_quality/lane4_pipeline_p1_20260804/RECEIPT.md` and
  `../../../code_quality/lane4_pipeline_p1_20260804/FINAL_ARTIFACT_STRICTNESS.md`.
- G-03-only Phase-2 h000 reconciliation: held-fit36 zero-feature
  `0.7630876414338513`, `36/36` graphs byte-identical, versus archived G-02
  native CV57 `0.829194058080` on a different population; no G-02 gate change,
  allocation, or reopening.
- Narrow CPU lanes: A7 support/applied `0/0` from 45,961 occupied swaps;
  reverse literal consensus sealed/applied `53/0` with `0/53` beneficial; R0
  shell sealed/applied `4/0` with precision `0/4` over 38,081 free/free rows
  inside 2,514,005 held-44b6 shell rows. Every deployed candidate is the exact
  APNCC no-op at adjusted/raw/node
  `0.8665345260361912/0.8625960574674240/0.9926180002844743`, delta `0/0/0`.
  Exact own-universe oracles are A7 `134/45,961`, reverse literal-pass
  `0/268` (all neutral), and R0 `5/38,081`; therefore A7/R0 were
  selector-abstention misses while reverse consensus is evidence-negative.
  Park audit remains 35/35 classified at `22/4/6/1/2`; the prior terminal
  cross-lane reclassification to `25/3/6/1/0` is retracted. Zero tier moves.
- Global-reassign ceiling: T2∞ adjusted `+0.09641286660426185`
  [GT-DEPENDENT], T3 `+0.10497364637634421` [GT-DEPENDENT], same 465
  correct-edge installs [GT-DEPENDENT], and only `+0.00856077977208236`
  [GT-DEPENDENT] from T3's extra 245 FP deletions [GT-DEPENDENT]. T2 reaches
  `91.84482956666014%` [GT-DEPENDENT] of T3. Histogram `{1:453,2:6}`
  [GT-DEPENDENT] shows the unlock is single-install endpoint legality, not
  chain depth. Section 4 is proposal-only, with terminal frozen-compiler
  escalation.
- Training feasibility: PEFT descriptive AUC plateau `0.633477→0.634749`
  over epochs 6–8 versus required `0.85`; graph-selected epochs
  `1/2/2/2/2`, selected AUC `0.496495`, and adjusted/raw/node deltas
  `+0.000133827/+0.000133160/0`. Trackastra safe end-to-end MVE is
  `4.05–5.96 T4-h`, while an honest one-seed five-fold selection run is
  `~7–20 h`. Compact warm-started raw-3D is provisionally `2–6 T4-h` after a
  zero-GPU compatible-asset gate and `<=0.25 h` matched calibration. The
  banked 30,057-parameter crop linker is CPU-cheap but DEAD and cannot be
  retried. Exactly two decision options remain; recommendation is Option 1's
  asset gate only, but the plan is not amended and no fit is authorized.
- Kaggle standing authorization: authenticated quota `5.06/24.94 h`
  used/remaining of `30.00 h`, refresh `2026-08-01T00:00:00Z`; recent
  campaign kernels all terminal. Latest scored submission is public `0.881`;
  the only ready CSV is byte-identical (`6944ed58…bea3`) and was not
  resubmitted. Productive launches/submissions no longer need a repeated
  access prompt, but still require an approved node/candidate and all serial,
  quota, integrity, kill, and monitoring gates.
- Terminal standing: APNCC-DIRECT local `0.881522069960` / public `0.881`.
  The action-selector requirements are DELETE `0.85/0.020394` and occupied
  SWAP `0.90/0.067910` AUC/retention. DINOv2 and assignment DELETE AUCs are
  `0.670412/0.670235`; their `0.000176` separation supports the campaign's
  empirical pointwise-label ceiling near `0.670`.
- Terminal division decomposition: GT-dependent ceiling `+0.092067842`;
  tier (a) `1/38`, `+0.002406490`; tier (b) `37/38`, `+0.089661334`,
  representation-blocked; tier (c) `0/38`, `0`, EMPTY.
- Terminal census: **LIVE/DEAD/PARK-A/PARK-B/PARK-C `0/1/14/18/12`** over
  45 parent nodes. CELLECT is permanent PARK-C / UNTESTED after six
  infrastructure failures. No active scientific wave is approved.
- Division-realizer 44b6 wave: A official `0/1/11`, Jaccard `0`, combined
  `-0.000141523` (valid FAIL); B diagnostic `0/49/11`, Jaccard `0`, combined
  `-0.009455519` (information-firewall INVALID); C not scored (no adequate
  frozen intensity cache); D diagnostic `0/0/11`, Jaccard `0`, combined `0`
  (invalid/missing component inputs). Duplicate survivors: `0/0/0/0`.
  Realized-FP sensitivity remains: 25 TP with 0/5/10 FP contributes
  `+0.0595/+0.0532/+0.0481`.
- APNCC-direct transport smoke v1: manifest `779aa8d3…606b`; one non-submitting
  GPU push reported v1 (no second push recorded). Remote Step 0 **FAIL_CLOSED** at `5.061967561 s`
  (limit `5.0 s`, overage `0.061967561 s`), `heavy_work_started=false`, and no
  transport receipt/CSV/fragments. Quota pre/post **4.53/25.47 h** with `0.0 h`
  debit; fix is the Step-0 full-input SHA timing path.
- APNCC-direct transport smoke v2: hardened Step 0 **PASS** at `2.715378342 s`
  (manifest `779aa8d3…606b`, 62 wheel byte hashes, positive `264,378` request
  count, zero competition-path visits). The `263,282`-row CSV and all four
  fragments are byte-identical to the authenticated public-`0.881` APNCC T4
  archive; the former mismatch used the wrong pre-APNCC reference. Quota
  `4.53/25.47 -> 4.69/25.31 h` used/remaining (`0.16 GPU-h`). Transport is
  valid; no duplicate submission or retry.
- CELLECT local: **5/5 fresh 45-second preflights passed at 0 MiB swap
  growth; 5/5 shard-0 starts failed closed on positive system swapout**.
  Shard outputs/checkpoints/gate verdict: `0/0/none`; Kaggle/GPU debit `0`.
  Exact request plan: 77 shards / 1,391,458 actions / 217,595 tile jobs.
- Division amendment diagnostic: banked local `0.881522`, planning range
  `0.9255–0.9422`, gap `+0.044–+0.061`; 12/20/25 zero-FP division TP add
  `+0.0286/+0.0476/+0.0595`, versus edge-only cap near `0.896`. Frozen-N1
  oracle 38-bundle delta is exactly `+0.092067841519`; 25 TP with 5/10 FP
  falls to `+0.0532/+0.0481`. Human amendment approved; the four 44b6-first
  fit-only realizer arms are now terminal with no PASS, and 6bba/CV57 remain
  closed.
- Lane 1 division package identity
  `4673a59079b7b9e72bbe261f6a62e2dcf004854e0d0657838bb56ddd144facf7`:
  banked APNCC 36-fragment empty-divisions-on round trip wrote 2,415,478 rows,
  canonical graph `e3ada186…f8c5`, and CSV SHA `26ee9c44…edbd9`; zero score
  change is by exact scorer-input graph identity (no scorer rerun).
- Selector-requirement diagnostic: **2/2 fresh preflights PASS at zero growth;
  2/2 scientific executions INVALID_RESOURCE**, at 722.620 s / 1.904 GiB /
  +16.31 MiB swap and 1,034.259 s / 1.453 GiB / +1,567.69 MiB swap. **0 result
  files from those attempts**. Repair-3 delete is separately **COMPLETE** at
  1,345,497 / 1,118, oracle-best `+0.081897667`, 140/140 seeds, 1.081 GiB RSS,
  and 0 MiB swap. A superseded swap receipt attempt produced no artifact/outcome
  before an intentional stop at 10,922.910 s, 0.997 GiB RSS, and 0 MiB swap;
  repair 4's primitive-union equivalence test passes and the full suite is
  22/22. Repaired occupied swap is **COMPLETE** at 45,961 / 134,
  oracle-best `+0.018571397`, 140/140 seeds, 1.017 GiB RSS, and 0 MiB swap.
  Restore is **COMPLETE / CLASS-DEGENERATE** at 50 / 0 beneficial with oracle
  ceiling `0`; AUC/retention is undefined. Final mean requirements are delete
  AUC 0.85 / retention 0.020394 and swap AUC 0.90 / retention 0.067910; both
  frozen-gate verdicts are **TOO LENIENT**. Barrier is
  `HUMAN_REVIEW_REQUIRED`; no threshold or tier change.
- Retrospective screening-fidelity node: **6 candidates / 36 common held-44b6
  videos / 12 cells / 10,000 draws per cell / 0 usable**. Best-rank `k=12`
  simple random had rho `0.828571`, sign agreement `0.820683`, and maximum 95%
  width `0.023057`; the narrowest `k=12` width was still `0.023017` under
  residual-top10. Thresholds remain `|rho|>=0.8`, sign `>=0.9`, width `<0.005`.
- Baseline local `0.873150` → public `0.877`. Banked APNCC local `0.881522` → **public `0.881`**.
- Recall N0 full 44b6: baseline TP/FP/FN **10,341/1,064/673**; pool-present/absent
  FNs **47/626**; zero-FP adjusted/raw/node deltas for absent
  **`+0.052004594313/+0.051829773141/0`** (GO threshold `>=+0.030`); 10–12 µm
  shell overlap **18/626**; 45.738 s preflight and 52.741 s audit, both 0 MiB
  swap growth.
- Recall N1 `NDR-FBC-ONE-OFFER`: **3** primary sources; fixed label-free fit117
  offers **105,992** (`49,553` 44b6 / `56,439` 6bba) against pool `7,874,673`,
  maximum multiplier **`1.013459860492×`**; 45.489 s preflight and 29.096 s count,
  both 0 MiB swap growth.
- Recall N2 `NDR-FBC-local-translation-v1`: **9/290 = 3.103448%** recovery
  (DEAD `<20%`; GO `>=50%` at `<=3×`), **1,173** new offers / **502,336** pool =
  **`1.002335090457×`**; all three fixtures failed closed, repeat SHA
  `0eecfb95…1713f`, 45.950 s preflight and 315.815 s run at 0 MiB swap. N3 did
  not open; a future recall route requires a new named observable/mechanism.
- Frozen external division-checkpoint N0: **0/>=1** qualifying releases; authenticated
  source revisions/weight SHA objects/license SHA receipts/corpus-exclusion proofs/
  source-published 16-GiB inference configurations all **0**. Verdict **PARK**;
  division fitting and B1 remain locked with **0** qualifying continuity arms.
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
- Tier census is **0/14/18/12/1** over 45 nodes; machine state
  `tiered_surface/tier_census.json` is now SHA-256
  `29fa828f27e390034570e953ffa0dd14cbd7cd754ebbc879237905f28a8f3f5f`
  after the R7 CELLECT and full-38 DIV-F1 terminal moves. Weekly Kaggle budget artifact
  `tiered_surface/kaggle_budget.json` at SHA-256
  `6fd4df91…e3c0` mirrors the authoritative Kaggle balance **5.06/30.0
  kernel-h used / 24.94 h remaining** and retains the named debits only as the
  named-debit audit subtotal. The **1.65 h** drift from the earlier 2.88 h
  tracker is recorded as a correction. Cumulative CPU use is **0.3010984 h**.
  No GPU authorization or CPU session remains active.
  RunPod is forbidden; productive Kaggle action has standing owner authority.
- Final supplier separation: common frozen population is **1,345,497 delete /
  1,118 beneficial** and **45,961 occupied-swap / 134 beneficial**, across five
  identical OOF folds. CELLECT is permanent PARK-C / UNTESTED after six
  infrastructure failures and has no scientific verdict. DINOv2 completed all
  77 request shards locally with atomic resume and zero swap growth. Its
  unchanged sealed B0→A1→A0 gate returned
  `FAIL_PARK_C` for delete (A1 mean AUC/retention `0.670412/0.001307`) and
  occupied-swap (`0.682473/0`). C2 assignment evidence independently returned
  `FAIL_PARK_C` for delete (`0.670235/0.001851`) and occupied swap
  (`0.664616/0.005882`). No graph preregistration is unlocked; DINO result
  `c58b1a58…0d41f2`, assignment result `264f3182…994e`, CELLECT terminal
  diagnosis `4a9e2088…2c855`.
- CPU-only Wave-2 NODETOPO prerequisite: **36/36 videos, 3,600/3,600 frames,
  107,370 proposals, 106,668 false proposals, 100,983 topology-joinable,
  50 restoration actions across 24 videos**, with 0 MiB swap growth. Its graph
  gate is terminal at 44b6: adjusted/raw/node
  **`−0.00000343615 / 0 / 0`**, 50/50 actions and 24/24 affected batches exact,
  36/36 videos scored, 111.186 s committed-lane guard, 0.667 GiB peak RSS,
  and 0 MiB swap growth; 6bba/CV57 unrun. Committed lane `189c43b`,
  terminal `540adb67…b31ac`, validation `6f87905b…579b3`, and post-run
  provenance closure `f2c7b12a…2c9d4` bind the full 87.4 GB archive SHA-256,
  preregistered central directory, evaluator, wrapper, and imported scorer.
  A separate evaluator independently reproduced the exact scores, deltas, and
  verdict.
- CPU-only Wave-2 division diagnostic: **38/42 present-but-fragmented,
  1/42 present/component-spanning without a local fork, 3/42 role-missing**;
  39/39 diagnostic N1 actions compiled, 36 persistent actions selected, three
  missing-persistence actions suppressed, and zero duplicate survivors.
- CPU-only Wave-3: DINOv2 ViT-S/14 authenticates at **88,283,115 bytes,
  175 keys, 22,056,576 parameters**, strict safe CPU load and a deterministic
  768-D XY interface. The reverse supplier request is **0.758 CPU-h**, 1.5 h
  wall, 4 GiB RAM, 512 MiB scratch, and 0 GPU-h. Division continuity freezes
  **38 direct bundles / 36 one-frame-late equivalents / 39 adds / 27 net
  deletes**. NODETOPO independent repeat guard was 111.186 s, peak RSS
  0.667 GiB, swap growth 0 MiB.
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
`NDR-FBC-local-translation-v1` is DEAD at recall N2 (`9/290 = 3.103448%` recovery);
do not tune or re-screen it. A future recall attempt must name a different observable /
mechanism before any node opens.
Conditional-PARK (revisitable): detection front-end · divisions · GT-conflict gates.
Frozen Trackastra zero-wrfeat H1 is DEAD at the 44b6 graph gate. Only the distinct,
separately authorized authentic-mask `H3-MASK-SOURCE` provenance question remains eligible.
Frozen C1 conflict-set attention (`C1-TA-HOCT-LITE`) is DEAD at its 44b6 graph gate;
do not run 6bba or re-screen it without a new named mechanism.

## Log (newest first)

- 2026-08-06 — **BRANCH CONSOLIDATION: 13/14 LOCAL CODEX BRANCHES MERGED TO MAIN / NO
  CODE OR SCIENTIFIC-STATE CHANGE / NOT YET PUSHED.** Human-directed repo hygiene task,
  no new screening action. All 14 local `codex/*` branches (none previously pushed to
  `origin`) were additive-only against disjoint report/script paths; none touched
  `src/`; none named a DEAD mechanism (DAG §6.1) — this merges already-completed
  evidence into history, it does not re-open or re-screen anything. 13 branches merged
  cleanly (one trivial `.gitignore` conflict in duplicate `PHASE1_*` patterns, resolved
  by keeping main's broader pattern; `codex/g03-phase1-resume-proof` was a strict
  ancestor of `codex/g03-phase1-singlefile-retry` and is subsumed by it).
  `codex/sa-reverse-authentic-44b6` was **not** full-merged: its tip modified
  `LEDGER.md`, `LIVE_STATE.md`, and `.cruise/{FLIGHT,ORIGIN}.md` with content that
  predates and would regress the current authoritative state (`.cruise/` is
  deliberately untracked on main since `085b591c`). Only its two genuinely new report
  dirs were pulled: `tiered_surface/branches/sa_division_value/**` (23 files, new) and
  `tiered_surface/branches/sa_reverse_authentic/**` (146 files — already present on
  main byte-identical via commit `bcc9a6d9`, so no-op). That branch is kept
  un-deleted, un-merged for provenance; the other 13 were deleted after merge.
  Several merged dirs carry their own `PROPOSED_LEDGER_ROW(S)` files
  (`sa_division`, `sa_nodetopo`, `cellect_bleed_stop_v1`, `sa_division_value`) —
  **unreconciled**, left for the root coordinator per the LEDGER.md-sole-writer rule.
  Full `tests/` suite: **1231 passed, 10 failed, 39 xfailed, 273 subtests passed**
  (143s). All 10 failures reproduce identically on pre-merge main (`8d02aecb`, verified
  in an isolated worktree) — pre-existing, unrelated to this consolidation, not fixed
  here. Secret-pattern scan across all 14 branch diffs and the two most recent main
  commits found no credential values, only existing redaction infrastructure. Untracked
  `.playwright-cli/` (9 ephemeral browser-automation snapshots, unrelated to any
  branch here — pre-existing on main) and added it to `.gitignore`; files kept on
  disk. No Kaggle/RunPod/GPU action; no submission; no DEAD promotion; no LEDGER.md
  edit. Awaiting explicit human confirmation before `git push`. (Claude)

- 2026-08-06 — **N0b FIX PREPARED AS UNAPPLIED DIFF / 9 LIVE SITES NOT 3 / SECOND
  INDEPENDENT FAILURE MODE (H2) CONFIRMED SEVERE / NOTHING APPLIED, NO KAGGLE ACTION.**
  Verified untouched: `scripts/`, `src/`, `kaggle_kernel/` and the prelaunch guard are
  byte-unchanged (target-file mtimes `03:32`/`03:36`/`22:00`/`20:52` all predate the session;
  zero occurrences of any new symbol; the `EXPECTED_TEST_COUNT` gate is still intact at
  `:100,182`). The patch was exercised on a **scratchpad copy** of the generator, the kernel
  regenerated, diffed, and STEP0 executed against synthetic hidden mounts.
  **N0's fix is correct but incomplete — 9 live sites, not 3.** `REQUEST_COUNT` /
  `REQUEST_FILE_COUNT` / `POPULATION_FILE_COUNT` (`:101-103`) are **dead code** (assigned,
  never read); only `EXPECTED_TEST_COUNT` is live — one gate plus three telemetry labels.
  **H2 upgraded from ~0.10 to severe and independent.** `COMPETITION_TEST` (`:112`) is a
  single hard-coded root; `:228` unconditionally sets `BIOHUB_TEST_DIR` from it; `:311`
  prefers that env var over the body's own two-candidate `COMP_DIR_CANDIDATES` (`:306-308`).
  **The kernel body's mount fallback is therefore dead code in this build — there is no
  second chance anywhere.** If Kaggle mounts the hidden test at the non-`/competitions/`
  root, STEP0 dies regardless of cardinality. Both must be fixed together.
  **Scoring behaviour CONFIRMED UNTOUCHED:** the regenerated-kernel diff is 7 hunks, all
  between generated lines `101-240` of a ~3000-line file, with **zero changed bytes at or
  below the pinned V32 notebook body**; CV57 `0.914358` is a string literal carried into the
  receipt, not recomputed. **Expectation-setting caveat:** if the hidden set really is
  larger, a successful resubmission scores videos we have never scored locally, so the
  returned number is a **first observation, not a confirmation of 0.914358**.
  **Zero-submission validation already executed:** unpatched STEP0 rejects
  `N ∈ {0,1,3,5,17}` and dies on the alternate root; patched STEP0 passes `N ∈ {1,3,4,5,17}`
  and the alternate root, still rejects `N=0`, and `N=4` resolves to the same root as before.
  **KFG-26:** a name ban is rename-defeatable and a blanket `len(x)==<int>` rule yields **13
  false positives** in the pristine V32 body; a scoped taint check was built and validated
  instead — FAILs the shipped kernel and bisect stage 2, PASSes the fixed kernel and all
  three scored-lineage sources, zero false positives, and the 17 existing guard tests pass
  unmodified. **Tests:** baseline 36 passed; exactly one assertion breaks
  (`tests/test_public_temporalunet_baseline.py:98`, `assertIn('COMPETITION_TEST = Path(')`),
  broken by the *mount-root* half, not the cardinality half — and **no test anywhere asserts
  the count is 4**. **Two operational blockers:** `scripts/public_temporalunet_hidden_submit.py:19`
  hard-codes slug `onebitbrain/g02-v32-submit` and refuses any other, so a fresh slug needs
  that constant edited; and the current generator **no longer reproduces the shipped kernel
  byte-for-byte** (regenerated `1823973455ce…` vs shipped `5193d0609e40…`, three inert
  deltas) — the rebuild is not "55272130 plus one line" and must be disclosed.
  **Cheapest next step, costs nothing:** retrieve the rerun traceback for `55272130`, which
  under H1 literally prints `STEP0 hidden inventory drift: N != 4`, revealing the true hidden
  test count before any submission slot is spent. Report:
  `dag_prompts/2026-08-06-alpha-hunt/runs/N0b_FIX_PATCH.md`. (Claude)

- 2026-08-06 — **N1 DIVISION ALPHA GO / METRIC OPERATING POINT DERIVED / NO SPEND.**
  Research-only read of the frozen scorer source. `tp+fn ≡ G` in `score_divisions`, so
  **`division_jaccard = tp/(G+fp)`** exactly (verified: cv57 `1/(13+11)=0.0416667`, shard-0
  `2/(25+52)=0.0259740`, both matching the banked file). The exact fork-acceptance threshold
  is **`p* = J/(1+J)`** (Jaccard form of the Lipton/Elkan/Narayanaswamy F1 result) =
  **4.00% on cv57, 2.53% on shard-0**; one division TP is worth **24 FPs** (cv57) / 38.5
  (shard-0). Every FP channel except `malformed_forks` requires proximity to the sparse GT:
  `tp+fp=12` of **368** forks, so **96.7% of emitted forks are never evaluated**. Derived
  attribution: **≥7 of the 11 cv57 FPs sit in the two videos that contain ZERO GT
  divisions** (216/368 forks, no reachable TP) and are therefore `evaluable_forks` /
  `cross_component_forks`, not division-window rejects. The rule costing the 12 FNs is
  `local_forks` (local-parent-anchor) acting as a **recall** constraint — ≤5 forks were ever
  `considered`. Ceiling: banked GT oracle `38/0/4`, J `0.904761904762`, combined
  `+0.092067841519` ⇒ detection is not binding. Code-only extension: the out-degree cap
  lives **only in the edge path and only after FP counting** (`metrics.py:140-153`), so a
  fork's **3rd branch is free** on the edge term while still counting for divisions.
  Named alpha **`DIV-EVOP`** (metric-derived EV-optimal division operating policy on V32
  edge-prob, with `DIV-K3` / `DIV-TDILATE` / `DIV-CONSERVE`); `DIV-APPEARANCE-EVENT` labelled
  research bet (raw appearance is the one observable class never actually screened — the
  falsified parent-event head contractually **excluded** it). Standing correction: the former
  "near-zero-FP division operation" constraint and 3-of-3 consensus combination are provably
  suboptimal under `J = tp/(G+fp)`; use union, not consensus. All follow-up probes (P0–P3) are
  CPU-only on banked artifacts. No code, GPU, Kaggle, RunPod, submission, tier, DEAD, or
  LEDGER action. Report:
  `../../../dag_prompts/2026-08-06-alpha-hunt/runs/N1_DIVISION_ALPHA.md`.

- 2026-08-06 — **CLEAN-PORT PLAN GO (CONDITIONAL) / THREE BACKLOG ITEMS RE-SCOPED / NOTHING
  PORTED.** N2 verified all 15 patch anchors by AST harness against the *real* support-pack
  source (`external/Tracking_Support/repo/scripts/predict_unet_transformer.py`, 677 lines,
  md5 `a2bbd2b3…` — note two other local copies exist and are invalid: a pre-rename checkout
  and an already-patched 933-line output). All 15 match at count=1 and compile.
  **Corrections to the 2026-08-06 survey backlog:** (1) bidirectional harmonic fusion and
  4-view JS edge TTA are **one atomic patch**, not two — the body hard-raises unless
  `EDGE_TTA_MODE=js_reliability_log_pool` *and* `VIEWS=4`; (2) forward-acceleration lookahead
  is **not free-standing** — its call site nests inside
  `LOCAL_ASSOCIATION_RANKER_MODE == 'full_motion_assignment'` and its bonus rides a
  ranker-blended `evidence` term, so a standalone port is an unmeasured variant; (3) the
  0.916 author **disables DeepCenter veto and adaptive short-track rescue exactly as we do**
  — they are not part of the 0.916 recipe, and the earlier "config-disabled upside" framing
  came from comparing against `biohub-11` and `mige551`, which both score *below* him.
  **Blocker B1 (compile-clean, runtime-fatal):** the `_bi_old` anchor matches our source and
  passes `compile()`, but `_bi_new` references `_edge_tta_imgs`, which our chain never binds
  (their dual-seed replacement ends `_edge_tta_imgs = imgs`; ours ends `del imgs`; entry `[4]`
  also differs, the other 4 of 6 are byte-identical). Porting means editing shipped code, and
  `compile()` does not catch it — a name-binding assertion is recommended.
  Recommended order **M6 → M4-standalone → M3 → M4-faithful → M1+M2, M5 dropped**, organised
  so every CPU-falsifiable item precedes the only GPU-priced one (M1+M2, ~2.4-3.0x
  end-to-end — *reasoned from call counts, not measured*; confirm against a real V32 profile
  before budgeting). Reverse-direction pass is shape-correct but semantically
  out-of-distribution (model trained forward-only; author clamps the rescale ratio to
  `[0.5, 2.0]`) — instrument how often that clamp binds on our seeds before trusting `w=0.20`.
  No conflict with outdegree-cap-2 (ILP-enforced) or det threshold 0.96875 (identical in
  both); only delta is `SECONDARY_DETECTION_WEIGHT` 0.5 (ours) vs 0.475 — free CPU A/B, but
  check the ledger for the provenance of our 0.5 first. **Neither public notebook pins a test
  count** — the `EXPECTED_TEST_COUNT=4` gate is ours alone
  (`scripts/public_temporalunet_baseline.py:533`, `scripts/v32_submission_bisect.py:115`), so
  this plan is cardinality-neutral but the N0 remediation remains a **hard prerequisite** or
  every mechanism here scores zero on the hidden rerun. Report:
  `dag_prompts/2026-08-06-alpha-hunt/runs/N2_CLEAN_PORT_PLAN.md`. (Claude)

- 2026-08-06 — **UNSCORED-SUBMISSION ROOT CAUSE IDENTIFIED (H1, ~0.75) / BISECT LADDER
  MUST NOT LAUNCH AS BUILT / NO KAGGLE ACTION TAKEN.** Our verified public score is
  **`0.881`, rank `1111/2019`** (team `onebitbrain`); local CV57 `0.914358` has **never**
  been publicly confirmed. Six of thirteen refs are `COMPLETE` with blank public *and*
  private scores: `55272130, 55251996, 55212065, 55201829, 54893629, 54890576, 54717551`.
  Forensics rule out both transport (all 13 are code submissions; direct CSV upload returns
  HTTP 400 for this competition) and CSV schema (the retained V32 output matches
  `sample_submission` on header, 10 columns, int dtypes, `-1` sentinels, contiguous ids;
  graph is clean — 0 dangling edges, 0 `Δt≠1`, max outdegree 2, 0 merges).
  **The discriminator is a hidden-rerun inventory-cardinality pin.**
  `kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:100,182-184` sets
  `EXPECTED_TEST_COUNT = 4` and raises `RuntimeError("STEP0 hidden inventory drift")` at
  import when `len(selected) != 4`. It discovers dynamically — so it truthfully passes the
  lifecycle guard's `dynamic_test_discovery` / `public_identity_pinned` checks — then
  hard-rejects any hidden inventory that is not exactly the public four. Kaggle's own
  `error_description` for ref `54890576` reads: *"Your notebook hit an unhandled error while
  rerunning your code. Note that the hidden dataset can be larger/smaller/different than the
  public dataset"* (`reports/g02/kaggle_submission_failure_54890576_v1.json`,
  `total_bytes=0`). Controlled pair: `bhv45-submit/1` frozen-registry → `54890576` blank;
  same package with discovery → `bhv45-submit/3` → `54910497` **0.877**. `grep
  EXPECTED_TEST_COUNT` matches only `scripts/public_temporalunet_baseline.py` and
  `scripts/v32_submission_bisect.py` — **no scored kernel contains it**.
  **URGENT: do not launch the eight-stage bisect ladder.**
  `scripts/v32_submission_bisect.py:115,163-164` templates the identical
  `EXPECTED_TEST_COUNT` gate into *every* stage, so under H1 all stages fail identically and
  burn up to seven submission slots proving nothing. The full-recursive-hashing hypothesis
  is quantitatively weak (~0.04): 408 files / 1.906 GB / 37 s at four videos, so even 10×
  costs ~6 min. Minimal fix (rerun contract only — weights, thresholds, graph policy and
  CV57 `0.914358` untouched): replace the `!= EXPECTED_TEST_COUNT` gate with the
  `discover_submit_jobs` contract (`if not selected or len(set(selected)) != len(selected)`),
  derive the four `*_COUNT` constants from `len(selected)`, resolve `COMPETITION_TEST`
  against both mount roots, mirror into the bisect script, add a cardinality-pin refusal to
  the lifecycle/prelaunch guards, and register **KFG-26** distinct from KFG-25. Open lead:
  the rerun creates its own `scriptVersionId` (proved by `54890576` →
  `scriptVersionId=336990437`), so the raw REST `competitions/submissions/list` endpoint
  (not the CLI, which drops the fields) should carry `errorDescription` / `totalBytes` for
  `55272130` and would reveal the true hidden test count for free. Unexplained: four refs
  (`54717551, 54893629, 55212065, 55251996`) have zero local artifacts; `55212065` landed
  8 min after a kernel push, consistent with notebook auto-submission silently spending
  daily slots — unproven. Report:
  `dag_prompts/2026-08-06-alpha-hunt/runs/N0_SUBMISSION_FORENSICS.md`. (Claude)

- 2026-08-06 — **V32 SUBMISSION REPAIR EVIDENCE RETRIEVED / HISTORICAL ARTIFACT
  ACCESS BLOCKED / NO NEW KAGGLE ACTION.** Competition ref `55272130` now lists
  `COMPLETE` with blank public/private scores, not an explanatory scoring
  result. Exact `g02-v32-submit/1`, `/2`, and `/3` each report `COMPLETE`, yet
  their output downloads and version-qualified log requests are byte-identical
  retained-success artifacts (CSV `11ff6284…e88b1`, JSONL `86becd3c…f4e6d`, log
  `3fea029c…354454`); all historical source pulls return HTTP 403. Thus neither
  required failed traceback nor rejected CSV was recoverable through
  authenticated CMUX, and no cause is asserted. The
  retained successful log did full recursive input hashing without an error.
  A source-forward eight-stage builder is ready: byte-exact pristine control,
  Step-0 no-hash, no-hash telemetry, four cumulative patches, then hashing
  last; all stages pass static/inline guards and 21 focused tests. No package
  was pushed, no submission was made, and ALIGNMENT is unchanged until the
  final hashing-only stage actually proves causation. Evidence:
  `tiered_surface/branches/public_temporalunet_baseline_v1/submission_repair_v32/EVIDENCE_RETRIEVAL_20260806.md`.
  (Codex root)

- 2026-08-06 — **PUBLIC 0.95–0.97 TIER IS A PATCHED METRIC EXPLOIT / RESEARCH ONLY,
  NOTHING IMPLEMENTED.** Surveyed 18 public kernels against the official scorer
  (`075fc5f`) and a live LB snapshot (2,019 teams, 2026-08-05T19:22:39). The entire
  0.952–0.969 tier is one appended cell that fabricates a hub node at `t=-1000`,
  `(z,y,x)=(-10000,-10000,-10000)` wired to every track root plus synthetic negative-time
  forks; `xiaoleilian` labels it `METRIC EXPLOIT` in-source and measures `VAL 0.8388 ->
  0.9203`. It satisfied the pre-patch division rules "single connected component" +
  "contains a predicted fork", both global graph properties. Host commit `aa65e90`
  (2026-07-17, "updating metric to patch weakly connected component exploit") replaced
  those with local directed topology and added an out-degree≤2 cap plus a
  `t_target-t_source==1` filter. Kaggle rescored: **0 of 2,019 teams ≥0.95, max 0.948**;
  every hack author is now below our CV57 0.914358 (xiaoleilian 0.889, kaiwalyaatulraut
  0.901, boristown 0.911) while clean authors are above (pilkwang 0.921, yusuketogashi
  0.916, mige551 0.915). Our 0.914358 ranks ~96/2019. Rules §3.8.d ("unfair playing
  practices or abuses") and §3.4.d ("altered … submissions are void") cover it; **not
  implemented, not recommended**, and the campaign's existing anti-node-pruning principle
  stands. **Correction (2026-08-06): "ranks ~96/2019" above is a HYPOTHETICAL placement of
  local CV57 onto the public LB, NOT our standing. Our verified public score is `0.881`,
  team `onebitbrain`, rank `1111/2019` — see the submission-scoring entry below.**
  `praxel` 0.902 division calibration is **already ours verbatim** (all five
  `SAFE_DIV_*` constants bit-identical) — no public notebook has a learned division
  classifier, and our CV57 division term is `1 TP / 11 FP / 12 FN` (J=0.0417, 368 forks
  emitted for 13 GT divisions), leaving ≈0.096 of headroom there. Backlog of legitimate
  techniques absent from us: bidirectional harmonic edge fusion, 4-view JS-reliability
  edge TTA, 22-feature local association ranker, forward-acceleration lookahead (all
  new); DeepCenter veto and adaptive short-track rescue (present but config-disabled).
  Report: `public_solution_landscape_survey_v1.md`. (Claude)

- 2026-08-05 — **V32 CAP2 UNSHARDED VERSION 1 LAUNCHED / RUNNING / 7.67 H
  RESERVED.** Separately preregistered immutable package `1ac9e1c2…bbd86`
  launched as fresh slug `onebitbrain/g02-public-temporalunet-v32-cap2-unsharded-v1`
  exact Version 1. CMUX proved slug absence, rehashed the eligible inline
  Step-0 guard, and bracketed the push at unchanged quota `11.93/18.07 h`;
  projection `6.8 h` is auto-approved below 8 h, with no wall-clock kill or
  sharding. Initial exact-slug status `3c598cbe…6538` is `RUNNING`. The separate
  launch-phase diagnostic was inapplicable/FAIL for missing remote-preflight
  attestation and was not used as mutation authority. Score once only after
  terminal authentication; no cap submission is authorized. (Codex root)

- 2026-08-05 — **V32 PROOF SUBMISSION REF 55272130 ACCEPTED / PENDING / NO
  RETRY.** The first predicate build failed closed because fresh validation
  canonicalized paths while its truthful receipt retained relative spelling;
  hashes and every substantive field matched. The lifecycle now proves the
  recorded paths resolve to the exact predicate inputs while preserving their
  receipt spelling; 73 focused tests pass. Fresh predicate
  `1409662b…02ea` binds exact Version 1 output `11ff6284…e88b1` and was
  consumed once under the serial lock. Receipt `b2bcb154…12b38` identifies
  exactly one new ref, `55272130`; authoritative list `108742f6…e34cf`
  reports `PENDING`. Retry is disabled. (Codex root)

- 2026-08-05 — **V32 HIDDEN VERSION 1 COMPLETE + AUTHENTICATED / 0.23 H DEBIT /
  ONE-SHOT PREDICATE OPEN.** Exact-slug CMUX status `6e4f08d5…a9bd` reports
  `COMPLETE`; exact Version 1 download `8d4b5abb…bdde` validates four dynamically
  discovered hidden datasets, 235,198 rows, no identity pin or public-output
  copy, max outdegree two, and persisted JSONL `86becd3c…f4e6d`. Submission SHA
  is `11ff6284…e88b1`. Fresh quota is `11.93/18.07 h` used/remaining, a `0.23 h`
  debit; the `0.50 h` reservation is released and zero active reservations
  remain. Direct upload stays forbidden; only a freshly regenerated exact
  Version 1 one-shot predicate may submit. (Codex root)

- 2026-08-05 — **V32 CAP2 EXPERIMENT PREREGISTERED / UNSHARDED PACKAGE READY /
  NOT LAUNCHED.** The faithful baseline discarded raw GEFFs and edge
  probabilities, so its exact highest-score ranking cannot be reconstructed.
  A separate 162-video immutable replay now implements only the requested
  postlink cap (`outdegree>2`, retain top two by pristine `edge_sort_key`),
  leaves `add_safe_divisions_postlink` unchanged, preserves edge order, and
  logs every affected parent/ranking. Package `1ac9e1c2…bbd86`, source
  `d27e246e…e6a81`, static/inline guards and 88 focused tests pass. It remains
  unlaunched until hidden Version 1 terminates and its reservation is
  reconciled. (Codex root)

- 2026-08-05 — **V32 HIDDEN VERSION 1 RUNNING / HEARTBEAT UPDATE API TIMED
  OUT.** Exact-slug CMUX receipt `d031c6eb…eaaf` authenticates
  `onebitbrain/g02-v32-submit` as `RUNNING`; the `0.50 h` reservation remains
  active and no submission occurred. Three app-level attempts to replace the
  now-stale heartbeat prompt did not return and made no on-disk change. Do not
  bypass that control plane by editing automation files; subsequent work must
  follow current LIVE_STATE and monitor only the fresh hidden slug. (Codex root)

- 2026-08-05 — **V32 FRESH HIDDEN VERSION 1 LAUNCHED / 0.50 H RESERVED / NO
  SUBMISSION YET.** Local `0.914358473021` clears the explicit `0.91` gate;
  `0.916` remains refinement-only and no tuning opened. The fresh immutable
  `onebitbrain/g02-v32-submit` package uses dynamic four-video test discovery,
  contains no public/test identities or copied output, retains pinned public
  V32 source/weights, and persists comprehensive JSONL logs. Static and inline
  guards pass; exact Version 1 push succeeded under the canonical lock with a
  `0.30 h` projection and `0.50 h` reservation at unchanged quota
  `11.70/18.30 h`. Competition submission remains fail-closed until terminal
  output authentication and a fresh one-shot predicate. Launch:
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_HIDDEN_SUBMIT_LAUNCH.json`.
  (Codex root)

- 2026-08-05 — **V32 UNSHARDED CV57 0.914358 PASS / TAXONOMY PUBLISHED / NO
  SCORE RERUN.** The sole frozen CV57 execution published score
  `0.914358473021`, beating the public `0.908--0.911` band and APNCC
  `0.881522069960` by `+0.032836403061`; it preserves and logs the two
  degree-3 parents. Its post-score taxonomy launcher failed after receipt
  publication on a missing import, so the scorer was not rerun. The recovered
  deterministic taxonomy passes 57/57 with zero authoritative-row mismatches
  and the explicit competition-faithful graph policy. V32 residuals are
  detection FN `538`, ordinary association `1911 FP + 20 FN`, fragmentation
  `1053 FN`, and division `3/69/39` TP/FP/FN. Next is a fresh
  hidden-test-compatible immutable submission predicate; no public output copy,
  tuning, cap mutation, or additive G1 claim occurred. Receipts:
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_UNSHARDED_V2_CV57_SCORE.json`,
  `../../g04/lanes/cv57_residual_error_taxonomy_20260804/V32_RESIDUAL_ERROR_TAXONOMY.json`,
  and `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_UNSHARDED_V2_CV57_TAXONOMY_RECOVERY.json`.
  (Codex root)

- 2026-08-05 — **V32 UNSHARDED VERSION 1 COMPLETE + AUTHENTICATED / TWO
  DEGREE-3 PARENTS PRESERVED / RESERVATION RELEASED.** Exact-slug CMUX status
  is `COMPLETE`; exact Version 1 output downloaded and package-bound validation
  PASS covers 162 datasets, 6,261,255 rows, 3,191,037 nodes, and 3,070,218
  edges. Submission SHA is `c61784c3…ab446`. The faithful public graph contains
  exactly two degree-3 parents (max 3), each logged with daughters; nothing was
  capped or rewritten. Downloaded Kaggle log SHA is `b6cabe09…40491`; the
  immutable package's missing persisted JSONL is still disclosed. Fresh quota
  `11.70/18.30 h` gives a `6.43 h` debit; the `7.67 h` reservation is released
  with zero active reservations. Next is the one frozen CV57 score. Receipt:
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_UNSHARDED_V2_TERMINAL_AUTHENTICATION.json`.
  (Codex root)

- 2026-08-05 — **V32 STATUS PROBE BLOCKED BEFORE EXECUTION / NO NEW RECEIPT /
  RESERVATION RETAINED.** At `08:11Z`, the sandbox approval service disconnected
  and rejected process creation before the canonical CMUX status command ran.
  No Kaggle read occurred and no new receipt was created; the control will not
  be bypassed. The last authenticated state remains `RUNNING` at `20:01Z`, the
  later `00:00Z` Kaggle response remains access-denied/nonterminal, and the
  `7.67 h` reservation remains active. No download, reconciliation, score,
  retry, or submission occurred. (Codex root)

- 2026-08-05 — **V32 STATUS READ ACCESS-DENIED / LAST AUTHENTICATED STATE
  RUNNING / RESERVATION RETAINED.** The single permitted `00:00Z` CMUX probe
  of exact slug `onebitbrain/g02-public-temporalunet-v32-unsharded-v2`
  returned Kaggle `kernels.get` permission denied and no worker state. The same
  authenticated slug was `RUNNING` at `20:01Z`; permission denial is not
  terminal evidence, so no download, validation, quota reconciliation, score,
  retry, or submission occurred and the `7.67 h` reservation stays active.
  Next read is no earlier than `01:00Z`. Receipt:
  `tiered_surface/branches/public_temporalunet_baseline_v1/cmux_receipts/kernel_status_2026-08-05T000055614105Z.json`.
  (Codex root)

- 2026-08-05 — **COLAB THREE-CELL RESUME + DEVICE LEDGER PASS / CAPACITY GAP
  NOT ESTABLISHED / PARKED-COMPUTE AUDIT COMPLETE.** The sanctioned notebook is
  exactly three manual cells and contains no free-tier session automation. Cell
  1 records the assigned GPU, pulls the immutable checkpoint by manifest hash,
  recomputes its semantic state digest on CPU, and fails closed on absence,
  corruption, or provenance drift. Exact-runtime resumes retain bit-identical
  status; cross-GPU resumes are explicitly logged without that claim. Cell 2 uses the
  600-second callback; Cell 3 publishes complete state plus full JSONL and a
  per-device ledger receipt. Fifteen focused tests pass. With `17.06` Kaggle
  T4-h unreserved after V32 and no frozen/timed model yet, Colab is not presently
  capacity-justified; allocation stays `0/0/0`. The >2-frame architecture is a
  new model outside the weight-refinement rule, so no waiver is required.
  DPRC-O0 preparation is capacity-eligible but scientifically launch-blocked;
  crop-wrfeat, full organizer, and identical DINO R5 remain parked/dead/failed.
  RunPod SGD 20 is planning-only and every run still needs explicit approval.
  No V32/Kaggle/Colab/RunPod action occurred. Evidence:
  `tiered_surface/branches/division_temporal_colab_v1/`. (Codex root)

- 2026-08-05 — **COLAB DIVISION PREPROCESS + DURABILITY HARNESS PASS / RULES
  PASS / SCIENTIFIC TRAINING BLOCKED.** Actual Kaggle rules permit private
  Colab-trained weight import with privacy/reproducibility conditions. The exact
  stride `(1,4,4)` private training carrier completed and reverified at 199
  videos, 19,900 frames, 151 parent events, 302 daughter edges, and manifest
  `6048f8d7…e6a121`; 199/199 datasets and frames were modified, daughter
  assignments 0. The G-03-derived checkpoint/resume, semantic state digest,
  swap guard, full logger, and separate Colab ledger pass seven tests. Colab
  usage is 0.0 h. All 199 datasets and 19,900 frames were modified; daughter
  assignments modified remained 0. At this earlier checkpoint the objective was
  pending the absent V32 taxonomy and an owner amendment; the newer entry above
  supersedes the amendment need with the new-architecture classification. No V32/Kaggle/GPU action. Evidence:
  `tiered_surface/branches/division_temporal_colab_v1/`. (Codex root)

- 2026-08-04 — **V32 V2 IMMUTABLE RUN-LOG GAP RECORDED / NO REMOTE ACTION.**
  Direct hash-bound source inspection shows that active Version 1 kernel source
  `932e0b12…f8e03` predates the persisted `RUN_LOG.jsonl` logging amendment:
  neither its source nor terminal receipt contains that binding. The kernel is
  already running and must not be repushed or mutated. Terminal handling will
  retain all available downloaded output/log evidence, authenticate the exact
  V32 graph and receipts, and report `PERSISTED_STRUCTURED_RUN_LOG_UNAVAILABLE_IN_IMMUTABLE_V2` truthfully. This is not an outdegree/model defect.
  Record: `tiered_surface/branches/public_temporalunet_baseline_v1/V32_UNSHARDED_V2_LOGGING_GAP_20260804.json`.
  (Codex root)

- 2026-08-04 — **V32 OUTDEGREE PROVENANCE RESOLVED / BASELINE POLICY
  CORRECTED / NO REMOTE ACTION.** Immutable public notebook
  `3f995a97…b7847d` itself contains `add_safe_divisions_postlink` (code cell
  11) and no `outdegree` reference. Thus the former local strict outdegree-two
  validator exceeded the competition invariant; the active unsharded Version 1
  remains untouched, and terminal V32 validation/scoring will preserve it while
  logging every degree-three-or-greater parent. A cap-to-two arm is explicitly
  separate, must retain the two highest-scoring daughters, and may not be
  invented if scores were not persisted. Evidence:
  `tiered_surface/branches/public_temporalunet_baseline_v1/V32_OUTDEGREE_PROVENANCE_20260804.json`.
  (Codex root)

- 2026-08-04 — **V32 UNSHARDED VERSION 1 RUNNING.** The first permitted
  hourly, read-only status check at `14:44Z` reports
  `KernelWorkerStatus.RUNNING`; no output exists to authenticate and the `7.67 h`
  reservation remains active. No retry, score, submission, taxonomy, or G1
  action occurred. Receipt: `cmux_receipts/kernel_status_2026-08-04T144451846671Z.json`.
  (Codex root)

- 2026-08-04 — **V32 UNSHARDED REPLICATION VERSION 1 LAUNCHED / 6.8 H
  AUTO-APPROVED / 7.67 H RESERVED.** Fresh V32 package V2 covers all 162 frozen
  identities in one end-to-end run, with immutable public notebook/checkpoint
  hashes, inline guard PASS, no merge/reassembly, and no wall-clock kill. Under
  the canonical lock, fresh-slug absence and quota `5.27/24.73 h` authenticated;
  `onebitbrain/g02-public-temporalunet-v32-unsharded-v2` Version 1 pushed
  successfully with a `7.67 h` active reservation. Focused guard/baseline/CMUX
  tests: `79 passed, 3 subtests passed`. Next is one hourly-bounded terminal
  check, exact output authentication, frozen CV57, then conditional submission
  and APNCC-vs-V32 taxonomy. G1 remains separate. Receipts:
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_UNSHARDED_V2_LAUNCH.json`.
  (Codex root)

- 2026-08-04 — **OWNER GPU-GOVERNANCE + COMPREHENSIVE-LOGGING AMENDMENT
  LANDED / NO EXTERNAL ACTION.** `ALIGNMENT.md` supersedes the former 5 h
  per-launch kill: authenticated projections `<8 h` are auto-approved without a
  wall-clock kill; `>=8 h` needs an explicit literature-cited human-approval
  pitch. The shared 30 h/week ledger and serial reconciliation remain binding;
  runs may not be sharded to evade the gate. Full input inventory/SHA, lookup,
  stage/RSS/count, preprocess examined/modified, config, traceback/state, and
  persisted/downloaded JSONL logs are now mandatory. (Owner + Codex)

- 2026-08-04 — **GLOBAL-JUMP PUBLIC TRANSFER COMPLETE AT 0.881 / LOCAL +0.005242
  DID NOT TRANSFER / ANCHOR 2 SEALED.** Ref `55221497` is COMPLETE at displayed
  public `0.881`, tying APNCC ref `54942864` despite exact CV57
  `0.886764420260` versus `0.881522069960`. The output and one-shot lifecycle
  remain authenticated; terminal receipt `2f5e33e3…1f70`. Local corrections
  must no longer be treated as evidence for a public `0.89` gate without a
  substrate-level transfer result. The scoped monitor is stopped; V32 was not
  read or disturbed. (Codex root)

- 2026-08-04 — **GLOBAL-JUMP APNCC EXACT V1 AUTHENTICATED / REF 55221497
  PENDING / ONE-SHOT CONSUMED.** Fresh slug `onebitbrain/g1jump-submit` Version
  1 completed at the first permitted G1-only hourly check. Exact output
  validation passed dynamic test discovery and reproduced SHA-256
  `366a12d6…8c8` / 263,260 rows. The `6.0 h` reservation is released against
  authoritative quota `5.27/24.73 h` used/remaining (`0.17 h` debit). Exactly
  one code-output submission was accepted as ref `55221497`; its immediate
  state is PENDING and retry is prohibited. A 15-minute task heartbeat now
  monitors only that ref and will seal the second transfer anchor at terminal;
  V32 was not read or disturbed.
  Receipts: `../../g04/lanes/data_pathology_repair/global_jump_public_transfer_v1/`.

- 2026-08-04 — **VELOCITY-AWARE-CANDIDATE-WINDOW-v1 EXACT NEGATIVE /
  PARK-C / G1 RETAINED.** After the repaired fit117 ceiling passed every gate,
  the hash-frozen exact scorer ran on G1 CV57. All 57 frame-0 no-motion controls
  were bit-exact with the stock scorer. Recentered adjusted/raw/node is
  `0.880534473728/0.885184801739/0.990377056057`, delta
  `−0.006229946531/−0.006276210289/0`; edge TP/FP/FN is
  `34200/2368/2068`. Exact 57/57 coverage passed. This is a valid scientific
  negative, not a plumbing block; no submission or external action follows.
  (Codex root)

- 2026-08-04 — **DIVISION ASSET CORRECTED / PAIR HEAD BANKED / EVENT HEAD
  PARKED / CTC CITATION-ONLY READING REJECTED.** Sealed re-analysis separates a
  working conditional daughter-pair head (AUROC/AP `0.897081/0.368520`) from a
  random parent-event head (`0.521521/3.1367e-5`) whose contract excluded raw
  appearance. Correct state: event detection has no working mechanism;
  daughter assignment is solved. Official CTC terms require explicit organizer
  permission for public non-CTC scientific use, so citation alone does not
  authorize Biohub Kaggle training. CTC real divisions are
  `PARK-A-PERMISSION`; with written scope-covering permission they become the
  leading appearance-event data dependency. No CTC data was opened. (Codex root)

- 2026-08-04 — **VELOCITY WINDOW FIT117 INTEGRITY REPAIRED / STRICT 366
  CONSERVATION / MATERIAL SCORE GATE PASS.** Rebuilt the actual 117-video
  fit117 G1-selected graph (2,878,849 nodes, 2,758,880 edges) from its exact
  fragment/cache/example inventories; no CV57 predecessor substitution or GT
  read occurred. The reachability replay then conserved `456−90=366` without
  relaxing the guard and sealed predecessor vectors, prior G1 shifts, and
  recentered cap-eight ranks for all residual rows. Fixed r10/cap8 recentering
  adds 67 reachable true continuations, retains 98/366 residual edges, and has
  `0.993272×` background burden; every prescorer gate passes. Exact frozen
  scoring on the G1 CV57 base is therefore authorized and active. (Codex root)

- 2026-08-04 — **GLOBAL-JUMP-CORRECTED APNCC PUBLIC TRANSFER LAUNCHED /
  EXACT VERSION 1 / ONE-SHOT + QUOTA-BRACKETED.** Dynamic discovery over the
  four public raw-Zarr identities reproduced an observation-distinct output
  (`134,409` nodes, `128,851` edges, SHA `366a12d6…a8c8`) from the exact APNCC
  parent SHA `6944ed58…bea3`; labels were not read and same-run caches/fragments
  are mandatory. Inline prelaunch passed exact package identity
  `95f40b68…3479`. The fresh-slug push created only Version 1 under the shared
  lock and a conservative `6.0 h` reservation; authoritative pre/post quota
  remained `5.10/24.90 h` used/remaining. Submission remains closed until exact
  Version-1 terminal output re-authenticates; no V32 status read or cadence
  change occurred. (Codex root)

- 2026-08-04 — **DIVISION REQUIREMENT SURFACE / EVENT-PAIR SEPARATION /
  PARK-C REAFFIRMED.** The sealed 80,013,717-action deployment re-analysis
  required no refit, threshold search, graph integration, V32 read, or
  external compute. Material `+0.005` requires 5 TP / <=17 FP in top 22
  (precision@22 >=22.73%, AP-lift >=8462x); observed pair-action AP lift is
  18.41x despite AUROC 0.8893. Event-head AUROC/AP is 0.521521/3.1367e-5;
  conditional pair head is 0.897081/0.368520. The 18 unreachable divisions
  partition 11 detection-universe / 4 nearest-8 / 3 pair-universe; r10 does
  not exist in this frozen candidate generator. Future rare-action gates are
  precision@k/AP-lift on the deployment denominator, never AUROC. (Codex root)

- 2026-08-04 — **PUBLIC V32 SHARD 2/6 COMPLETE BUT GRAPH-INVALID / TWO
  OUTDEGREE-3 PARENTS / RESERVATION RELEASED / SHARD 3 BLOCKED.** The one
  scheduled hourly status read returned COMPLETE; exact Version-1 output
  downloaded and matched its run receipt (27 videos, 625,430 nodes, 602,054
  edges, 1,227,484 rows). Strict validation failed only at
  `44b6_a2bb48bb:27740` and `6bba_ebdf3b34:24468`, each with three children;
  no merge exists and max outdegree is 3. Current-code revalidation passes
  shards 0 and 1, excluding validator drift. Fresh quota moved
  `3.88/26.12→5.10/24.90 h` used/remaining, a `1.22 h` debit; the canonical
  `3.0 h` reservation was released with zero active reservations. Validation
  `373d47c1…2d9aa3`; terminal adjudication `ee24311c…7aea97`; shared ledger
  `62f4865c…a04b2a`. Fail closed: shard 3 and scoring remain blocked pending a
  frozen packaging-only deterministic feasibility repair; no GPU retry,
  submission, weight refinement, RunPod, h030, or G-03 spend. (Codex root)

- 2026-08-04 — **PUBLIC V32 SHARD 2/6 RUNNING / RESERVATION RETAINED / NO
  SECOND POLL.** The one scheduled hourly status read returned RUNNING for
  exact Version 1. Receipt `22b37f2f…c142a5`; status record
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S02_STATUS_20260803T1717Z.json`
  (`93f01208…67b29`). The canonical `3.0 h` reservation remains active under
  shared ledger `c825d878…5bbdeb`; terminal output validation, quota
  reconciliation, reservation release, and shard 3 all remain closed. No
  further Kaggle call this hour; no submission, weight refinement, RunPod,
  h030, G-03 spend, or score. (Codex root)

- 2026-08-04 — **VELOCITY-AWARE-CANDIDATE-WINDOW-v1 CHARACTERIZED /
  INTEGRITY-BLOCKED NO_SCORE.** This new fixed-radius per-detection recentering
  explicitly does not re-screen DEAD radius/motion sweeps. Canonical GA-R12REACH
  reproduces the residual `366=456−90` (44b6/6bba `37/329`), corrected-distance
  median/IQR `10.5312/10.2774–11.3458 µm`, and r10 source/target density
  medians `2/2`. 53 no-history sources are definitely outside r10; 313
  prior-bearing cases lack the sealed predecessor vector, prior G1 shift, and
  recentered cap-eight rank. Thus the prescorer gate is unevaluable and no
  graph/scorer/external action ran. (Codex root)

- 2026-08-04 — **P1 FINAL-ARTIFACT STRICTNESS AUDITED / REGRESSION-SEALED.**
  Both active `scoreall` input gateways already validate `SubmissionGraph`
  before any permissive diagnostic conversion; no production refactor was
  needed. New six-subtest coverage rejects missing endpoints, duplicate edges,
  and self edges through both aggregate and fragment loaders; the full
  `test_scoreall.py` run passed `17` tests / `6` subtests. (Codex root)

- 2026-08-04 — **P1 ACTIVE-PIPELINE VALIDATION SEAM COMPLETE.** Extracted the
  existing window/offset validation and normalization from `pipeline.py` into
  `pipeline_inputs.py` behind an API-preserving deterministic 14-node/7-edge
  `predict_dataset_window` graph-SHA fixture. Pipeline `33` tests / `20`
  subtests, routing+solver `41` tests / `19` subtests, public import, compile,
  and diff checks pass. No archived, generated/hash-coupled, or PARK-COST
  surface changed. (Codex root)

- 2026-08-04 — **CV57 RESIDUAL TAXONOMY COMPLETE / EXCLUSIVE FOUR-WAY
  PARTITION / NO GATE CHANGE.** Exact 57-video scorer replay reproduces
  `2459/1948` edge FP/FN with detection `466 FN`, ordinary association `2459
  FP`, matched-lineage fragmentation `1482 FN`, and division `42 FN`. Shapley
  score impacts are `+0.017446063106`, `+0.061628263275`, `+0.039403603659`,
  and `+0.100000000000`; their `1.100000000000` fully-fixed ceiling is
  theoretical, not an approved intervention. Per-video visualizer index is
  complete and worst-first; validation passed 57/57 exact rows, partition and
  Shapley conservation, plus `26` tests / `23` subtests. (Codex root)

- 2026-08-04 — **SYNTH-DIV REAL RANKING PASS / ZERO-TP DEPLOYMENT HARD FAIL /
  PARK-C:** authenticated fit117 evaluation completed in `1509.17 s`. Global
  beneficial-action AUROC `0.889300580471` passes `0.85`; 65/83 real divisions
  have an exact candidate. But all 80,013,717 scores fall below the frozen
  minimum `0.0005`, so all 114 threshold/cap rows select zero forks and return
  `TP/FP/FN=0/0/83`, Jaccard `0.0`. The maximum-J, within-95%-recall, and
  both-family-TP selection rules have no nonzero feasible policy. Adjudication
  `879db6b1…d8221` invokes the frozen zero-real-TP hard failure; no threshold
  relaxation, V32 integration, combined scoring, or visualizer refresh occurred.

- 2026-08-04 — **FULL-CATALOG VISUAL REVIEW RENDERED / DIAGNOSTIC ONLY / NO
  GATE CHANGE.** The requested uncapped `--scope all` APNCC CV57 bundle is
  complete: 57/57 validated H.264 clips, default `15.000 s` pacing and HUD,
  `90,888,617` bytes, in `292.797 s`; input SHA
  `ff4e2372…929f08`. The low-priority request was attempted, but managed macOS
  denied `setpriority`; BLAS caps applied and the renderer used its documented
  four-worker fallback. Receipt:
  `../../g04/lanes/full_catalog_visual_render_20260804/RECEIPT.md`. (Codex root)

- 2026-08-04 — **PUBLIC V32 SHARD 2/6 EXACT VERSION 1 LAUNCHED / SHARED
  3.0-H RESERVATION ACTIVE / HOURLY MONITORING ONLY.** The repaired explicit
  fresh-v1 lifecycle passed 61 combined tests. Under the canonical cross-thread
  lock, the bracket authenticated quota at `3.88/26.12 h` used/remaining,
  reserved `3.0 h`, and used its own target status as this hour's sole Kaggle
  kernel-status read. That receipt proved the private slug absent; the one push
  then returned “Kernel version 1 successfully pushed.” Immutable code/package/
  shard hashes are `97fa08fa…4c6fe1` / `288a877b…327847` /
  `df876107…c8f1cf`; hard whole-kernel kill is 10,800 seconds. Immediate debit
  is `0.00 h`; the reservation remains active until terminal evidence and fresh
  quota reconciliation. Launch record
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S02_LAUNCH.json`
  (`6b49ad89…5c6cb9`), push receipt `3d61d30d…0c7492`, shared ledger
  `c825d878…5bbdeb`. No further poll this hour; no submission, weight
  refinement, RunPod, h030, G-03 spend, or score. (Codex root)

- 2026-08-04 — **SYNTH-DIV FIT117 SCORE MANIFEST FROZEN / REAL GATE OPEN:**
  cache-repaired four-worker generation completed all 117 videos in
  `1135.49 s`, sealing 80,013,717 candidates (44b6 38,911,877; 6bba
  41,101,840). Manifest SHA is `029061b1…036b4`; all 117 records pin Amendments
  3/4, contain no labels, and attest GT unopened. The existing four-hour cap
  was ample, so the owner-authorized eight-hour extension was not used. The
  preregistered beneficial-action AUROC gate remains unchanged and precedes
  every integration action.

- 2026-08-03 — **SYNTH-DIV WATCHER OWNERSHIP CONSOLIDATED:** app-wide task
  inspection found exactly one active `cell_tracking` task: authoritative task
  `019fc6b9-0154-7be1-8b37-7159cb6f5da8`, renamed “Authoritative
  synthetic-division AUROC + V32 integration.” It alone owns the real-GT AUROC
  evaluation and synthetic-on-V32 integration write. The idle V32 continuation
  owns baseline shards/authentication and V32/G1 scoring only; its dispatch
  contains no synthetic gate/integration action. The recently updated untitled
  chat is unrelated finance. All three internal synthetic workers are complete;
  no second watcher remains active and none required interruption or archival.

- 2026-08-03 — **PUBLIC V32 SHARD 2 PRELAUNCH REFUSED CLOSED / NO REMOTE
  VERSION / FRESH-V1 GUARD REPAIRED / NEXT HOUR.** The frozen S02 package and
  inline predicate passed, and fresh authenticated quota was
  `3.88/26.12 h` used/remaining. The bracketed launch reserved `3.0 h` but
  refused before `kernels push` because its hardened status guard did not yet
  recognize Kaggle's exact inaccessible response for a genuinely fresh private
  slug. It auto-released the reservation as ERROR; no kernel version was
  created, no GPU debit occurred, and the canonical ledger has zero active
  reservations (`21a3dd58…fac2429`). This hour contains two disclosed status
  reads: the scheduled fresh-slug check plus the bracket's newly discovered
  mandatory under-lock check. No retry or further Kaggle call was made. The
  repair requires an explicit `expected_initial_kernel_version=1`, accepts only
  the exact fresh-slug denial, and refuses every existing status including
  COMPLETE; 54 focused and 61 combined tests pass. Next hour, the bracketed
  launch alone supplies the one allowed target-status read. Refusal adjudication
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S02_PRELAUNCH_REFUSAL_ADJUDICATION.json`
  (`9e3f8547…de535`); fresh predicate `d238152c…68834`. No submission, weight
  refinement, RunPod, h030, G-03 spend, or score occurred. (Codex root)

- 2026-08-03 — **TEST-SUITE TRIAGE COMPLETE / DEFAULT REVIEWED-GREEN:** the
  earlier 25 failures were a capped prefix; uncapped reproduction found 43.
  Four active failures were fixed (two process-pool sandbox probes, organizer
  shared-child division semantics, and scorer parity-v2 migration). The other
  39 exact retired/package/fixture nodes now carry reviewed `strict=True` XFAIL
  reasons; zero tests were deleted. Canonical result: 1,198 PASS / 39 XFAIL /
  267 subtests / zero FAIL, with 1,237 collected. No external action. Report:
  `../../../code_quality/TEST_SUITE_TRIAGE_20260803.md`. (Codex root)

- 2026-08-03 — **G04 HIDDEN-RERUN IDENTITY AUDIT PASS / NO DUPLICATE
  CANARY:** all live code-submission generation paths dynamically discover
  mounted test identities (APNCC submit runner plus G04 parent/mount inventory
  equality); the sole fixed four-public-ID transport is the already
  `TERMINAL_INVALIDATED_NO_RETRY` KFG-25 artifact. Ref `55203439` is the
  existing one-shot end-to-end Kaggle blind-rerun canary: terminal COMPLETE at
  public `0.881`, with no blank/format failure. A fresh identical external
  submission would violate the frozen one-shot/no-retry contract, so none was
  launched. Focused dynamic-discovery, guard, and lifecycle tests passed
  `21/21`; full audit:
  `../../g04/lanes/division_permissive/SUBMISSION_PATH_IDENTITY_AUDIT_20260803.md`.
  (Codex root)

- 2026-08-03 — **SYNTH-DIV FIT117 ATTEMPT FAIL-CLOSED / CACHE-ONLY REPAIR
  FROZEN:** the truth-blind four-worker attempt reached 91/117 atomic shards
  and 1,047,476,788 bytes but published no manifest before the `14400 s` cap.
  The pool's exception cleanup did not terminate running children, so the
  coordinator enforced SIGTERM at `14545.63 s`; all exact processes exited and
  no GT opened. Execution Amendment 4 (`5f87a284…9dbf3`) freezes a read-only
  cache of the bit-identical physical-coordinate product and explicit bounded
  child termination, with no scientific change. All 31 focused tests pass;
  the pre-change golden NPZ remains exact, and the prior `296.26 s` fragment
  now seals the identical 845,628-candidate NPZ in at most 40 s. Freeze V4
  opens one fresh four-worker fit117 score attempt under the same cap.

- 2026-08-03 — **CROP-WRFEAT PARK-COST / REPOSITORY DEBUG COMPLETE:** banked
  repaired crop geometry/coverage and prohibited a fourth throughput attempt
  without a new architecture. Archived orphaned pseudo-SSL/Ultrack and invalid
  G04 public-copy paths; added canonical pytest roots. Repaired zero-edge scorer
  parity, partial auto-routing validation, mixed-cost sparse assignment, and
  central Kaggle mutation/submission lifecycle. Registered KFG-25 and all live
  ownership slugs. Replaced stale-file deletion with an OS-held serial lock and
  bound active reservation release to exact per-kernel terminal evidence.
  Focused adversarial suite 100 PASS + 7 subtests; active collection 1,235. One
  banked equal-cost gap tie and complete package-tree/ref attribution remain
  documented P1 debt. No external action.

- 2026-08-03 — **PUBLIC V32 SHARD 1/6 EXACT VERSION 1 PASS / RESERVATION
  RELEASED / SHARD 2 DEFERRED TO NEXT HOURLY WINDOW.** The single scheduled
  status read found COMPLETE. Download receipt `81a4c639…f89f`; strict output
  validation passed 27/27 datasets, 468,656 nodes, 449,831 edges, 1,039
  dividing nodes, and 918,487 rows after `3323.533 s` prediction. Validation
  SHA `32bf52a3…354a0`. The focused harness regression was repaired by using
  direct official `DistanceMatching` for node-only fragmentation rather than a
  nonexistent private scorer import; all 7 module tests pass. Authoritative quota moved
  `2.76/27.24→3.88/26.12 h` used/remaining, so the exact debit is `1.12 h` and
  the canonical shared `3.0 h` ceiling is released with zero reservations left.
  Shard 2 was not queried or launched: its mandatory target-version check would
  be a second kernel-status read this hour. Terminal adjudication
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S01_TERMINAL_ADJUDICATION.json`
  (`155eee24…884ad1`). One sandbox DNS envelope failed before Kaggle returned
  any status; the one successful scheduled read is separately receipted and no
  second kernel status was obtained. No score, submission, weight refinement, RunPod, h030,
  or G-03 action occurred. (Codex root)

- 2026-08-03 — **CROP-WRFEAT RECORDED PARK-COST / COVERAGE BANKED / FOURTH
  CONFIGURATION FORBIDDEN.** Three throughput realizations (`195.59 T4-h`,
  `766.2 CPU-h`, `659.856 T4-h`) are operationally infeasible against the
  `27.24 h` balance. The repaired `914/916` coverage result is retained as
  sound reusable geometry evidence. This is neither PARK-EVIDENCE nor DEAD;
  reopening requires a newly named architecture that eliminates redundant
  per-detection tri-orientation fusion, with its own preregistration and bounded
  cost gate. Decision:
  `../../g04/lanes/global_jump_crop_wrfeat_v2/PARK_COST_DECISION.md`.
  (Codex root)

- 2026-08-03 — **PUBLIC V32 SHARD-0 NATIVE DIVISION DIAGNOSTIC COMPLETE /
  G04 REF 55203439 PUBLIC TIE.** The exact frozen official evaluator scored all
  27 authenticated shard-0 videos: division TP/FP/FN `2/52/23`, Jaccard
  `0.025974`; its nine CV57 videos are `1/11/12`, Jaccard `0.041667`, and its
  18 held-fit105 videos are `1/41/11`, Jaccard `0.018868`. The 1,420 raw
  branching nodes contain a nonzero native signal, but 92% GT recall deficit
  and 52/54 scored false predictions mean Prompt C must add missing events and
  refine native precision. Artifact
  `tiered_surface/branches/public_temporalunet_baseline_v1/FULL_PUBLIC_V32_S00_NATIVE_DIVISION_SCORE.json`
  (`a0cd099d…de022`). The authenticated submission-history read found ref
  `55203439` COMPLETE at public `0.881`, tying APNCC's displayed `0.881` after
  local `0.881635476737` versus `0.881522069960`. The APNCC-anchored projection
  is `0.881113407`, also displayed `0.881`. Frozen scorer arithmetic is
  unchanged; this is recorded as an interval-censored transfer calibration,
  and APNCC stays FINAL slot 1. Receipt `9d13c899…f15f`; calibration artifact
  `G04_CAP5_PUBLIC_CALIBRATION_20260803.json`. No V32 kernel-status read, GPU,
  launch, retry, weight refinement, RunPod, h030, or G-03 action occurred.
  (Codex root)

- 2026-08-03 — **PUBLIC V32 SHARD 1/6 EXACT VERSION 1 LAUNCHED / SHARED 3.0-H
  RESERVATION ACTIVE.** Local package/inline hashes revalidated: code
  `a4e17b04…c4568`, package `20c418f…a6bd07`, predicate
  `d36064e8…6fe1bf`. The one hourly target-version query found no accessible
  pre-existing slug; CMUX then pushed once and Kaggle assigned Version 1.
  Push receipt `5c4e68cc…da1c37`. The same cross-thread bracket reserved `3.0 h`
  and held authoritative quota at `2.76/27.24 h` used/remaining with zero
  immediate debit. Shard 1 is monitor-only until the next hourly heartbeat.
  No score, submission, weight refinement, RunPod, h030, or G-03 GPU action
  occurred. (Codex root)

- 2026-08-03 — **G04 CROP-WRFEAT BOUNDED T4 PILOT TERMINAL / COVERAGE PASS /
  BUDGET FAIL / FULL RUN CLOSED.** Authenticated Version 1 completed in
  `918.830 s` (`902.346 s` worker) with exact serial/batch-one label parity.
  The preregistered independent gate passed at `673/673` valid masks with zero
  boundary or empty masks; all 916 detections yielded 914 valid, 1 boundary,
  and 1 empty mask, with the enlarged fallback recovering 50/52 cases.
  Despite genuine CUDA batching, throughput was `0.938601 s/detection`; the
  conservative CV57 projection is `659.856 T4-h` and held-fit105+CV57 is
  `1875.361 T4-h`, versus `27.24 h` remaining. The `0.6 h` reservation was
  released after a `0.25 h` debit. Coverage is repaired, but no full feature
  extraction or crop-wrfeat score run is authorized. V32+G1 remains blocked on
  all-six-shard immutable base authentication and will be measured directly,
  never inferred from the APNCC delta. Terminal adjudication:
  `../../g04/lanes/global_jump_crop_wrfeat_v2/CROP_GPU_PILOT_TERMINAL_ADJUDICATION.json`.
  (Codex root)

- 2026-08-03 — **SYNTH-DIV-SIM2REAL SERIAL RUNTIME STOP / FOUR-WORKER
  EXECUTION REPAIR FROZEN.** Truth-blind fit117 generation spent `296.26 s` on
  the first 59,650-row fragment without sealing a shard, projecting beyond the
  14,400-second contract. It was stopped with zero candidate files and before
  GT access. Execution amendment 3 (`3e82d991…b1a7f8`) changes only topology to
  four independent per-video processes and requires serial/parallel candidate
  byte equivalence. Model and all scientific gates are unchanged. (Owner +
  Codex root)

- 2026-08-03 — **REVISED DEFAULT POST-SCORING VISUAL REVIEW READY / NO GATE
  CHANGE.** Implemented the exact four-category curated selector, 15-second
  default pacing with direct FPS override, registry-driven HUD/corner controls,
  cumulative edge FP/FN, default/all scopes, content-hash reuse, stable latest
  index, and the post-publication `scoreall.py` hook. Real APNCC delivery is
  9 deduplicated 100-frame clips / 9 category-complete cards / `16,558,395`
  bytes; all encode to `15.000 s`, 9/9 validation passes, and a repeat build is
  9/9 cache hits. Prompt-3 census events render at source `t` → target `t+1`.
  Browser checks passed range-backed seeking (frame-5 marker reads frame 5),
  390-pixel containment, and zero console errors. Tests: 47 passed plus 27
  parameterized subtests. Entry:
  `../../g04/post_scoring_visual_review/latest/index.html`.

- 2026-08-03 — **SYNTH-DIV-SIM2REAL-V2 MODEL FROZEN BEFORE REAL LABELS.**
  The sole full local fit completed in `4521.04 s` with 1,726 train sequences,
  3,674,104 pair examples, and 2,616,835 event examples / 131,311 positives.
  All event positives remained; only the preregistered 93 unreachable train
  forks left the conditional pair loss. Strict model reload authenticated JSON
  `53b36d96…d21dd82`, arrays `d979a308…cfff652`, and freeze report
  `065d1308…c9c2fc`. Fit117 labels remain unopened; truth-blind score generation
  is now allowed. (Owner + Codex root)

- 2026-08-03 — **G04 CROP-WRFEAT REPAIR PREREGISTERED / BOUNDED T4 PILOT
  VERSION 1 LAUNCHED.** A real computed-object audit measured p99 Z/Y/X
  extents `23.3/39.3/34.9` voxels, proving the old `16/32/32` crop undersized.
  Amendment 1 freezes intensity recentering capped at 4 um, primary
  `32/64/64`, and a single `48/96/96` fallback. The revised lane batches
  detections on CUDA and computes only checkpoint-native wrfeat12, excluding
  descriptive/Qhull CPU work. Its exact 3-frame / 916-detection package passed
  inline prelaunch and launched as
  `onebitbrain/g04-crop-wrfeat-gpu-pilot-v1` Version 1 under a `0.6 h`
  reservation and 30-minute worker kill; quota bracket stayed
  `2.51/27.49 h` used/remaining at push. The remote independent gate is
  finite-nonzero-nonboundary `>=98%`, boundary `<=10%`, empty `<=2%`, followed
  by exact CV57/held-fit105 T4 projections. Full production and scoring remain
  unauthorized. (Codex root)

- 2026-08-03 — **PUBLIC V32 SHARD 0/6 TERMINAL PASS / RESERVATION RELEASED /
  S01 DEFERRED TO HOURLY CADENCE.** The sole hourly status read returned
  COMPLETE. Download and strict validation authenticated 27/27 exact datasets,
  541,109 nodes, 521,261 edges, 1,420 dividing nodes, and 1,062,370 rows; all
  public/checkpoint/population/runtime/no-refinement bindings pass. Prediction
  took `3264.467 s`; validation SHA `f411b7d4…897d21`. Authoritative quota moved
  `1.37/28.63→2.51/27.49 h` used/remaining, a `1.14 h` debit; the shared `3.0 h`
  ceiling was released and active reservations are empty. S01 is not launched
  this cycle because CMUX's mandatory target-version check would be a second
  Kaggle status call inside the hour. No score, submission, G-04 re-point,
  weight refinement, RunPod action, or G-03 GPU action occurred. (Codex root)

- 2026-08-03 — **V32 CONTINUATION/INTEGRATION CONTRACT FROZEN / SHARED GPU
  RESERVATION LEDGER ACTIVE.** Hourly polling is unchanged and no extra Kaggle
  read occurred. The completion path now binds exact V32 CV57 scoring,
  four-way error diagnosis, independent V32+G1 measurement, and gated G-04
  machinery re-pointing. `scripts/cmux_ops.py` now requires a positive
  worst-case reservation for every bracketed GPU kernel push and capacity-tests
  all live reservations against fresh authoritative remaining quota under one
  canonical cross-thread lock. The already-running V32 shard 0 holds `3.0 h`;
  latest recorded quota remains `1.37/28.63 h` used/remaining. Contract SHA
  `439e9779…c7ef8c`; 40 CMUX tests pass. No score, submission, new launch,
  polling action, or weight refinement occurred. (Owner + Codex root)

- 2026-08-03 — **G04 DATA-PATHOLOGY G1 PROMOTE LOCALLY / F1+STRIDE NO-OP /
  CROP-WRFEAT KILL.** Complete 203-video census found 967 byte-confirmed frozen
  transitions and 227 shifts over 10 um. Exact CV57 G1 scored
  `0.8867644202595942` versus APNCC `0.8815220699597315`, with edge
  `34389/2308/1879`; candidate reachability rose `209/311→259/311` and the
  fit-only 10–12-um shell cross-check rescued 90/93 flagged positives. F1 and
  stride parity scored delta zero. The fixed all-243 crop-wrfeat pilot failed
  on 12 empty/unmatched and 199 boundary-truncated masks and projected 766.233
  CPU-h for CV57, so no feature production/score followed. Thirty-four tests
  pass; CPU only, no Kaggle/GPU/submission/RunPod action. (Codex root)

- 2026-08-03 — **OWNER LOCK: NO H030 OR G-03 GPU SPEND PENDING ALIGNMENT
  PATCH.** Direct inspection of `tiered_surface/ALIGNMENT.md` SHA
  `aff9e656…abaff` does not show the pending patch, so confirmation is negative
  and all G-03 GPU actions fail closed. The separately authorized immutable
  public-TemporalUNet substrate/ensembling run remains in scope. (Owner + Codex
  root)

- 2026-08-03 — **PUBLIC TEMPORALUNET SERIAL CONTINUATION HEARTBEAT ACTIVE.**
  An hourly same-task heartbeat now continues the long Kaggle reproduction
  under the existing standing authorization: at most one status read per hour,
  atomic output validation and quota reconciliation before each next serial
  shard, no RunPod, and automatic removal at a terminal outcome. This is an
  operational monitor only; no score, gate, promotion, submission, or weight
  refinement changed.

- 2026-08-03 — **SYNTH-DIV-SIM2REAL-V2 REACHABILITY PASS / ONE FIT OPEN.**
  Exact all-2,174-sequence audit SHA `1008e6ea…4565eb2` found
  165,148/165,267 reachable labelled forks (`99.927995%`): train
  131,218/131,311, validation 16,679/16,692, synthetic-test 17,251/17,264.
  Every frozen `>=95%` gate passes. The report is bound to every label hash;
  nearest-8 and all downstream gates remain unchanged. The 27-test combined
  suite is green, opening exactly one local CPU fit. (Owner + Codex root)

- 2026-08-03 — **PUBLIC TEMPORALUNET COMPARISON PACKAGES READY / FOUR-WAY
  DIAGNOSTIC CONTROL PASS.** Six single-seed and six detection-only 50/50
  blend packages joined the six V32 packages at inline-prelaunch PASS; none of
  the 12 comparison shards has launched. Detection-only skips the secondary
  association head while preserving aligned 50/50 detection logits. Exact
  18-package set SHA is `c1cbfee6…4273e`. The
  expanded frozen scorer reproduced APNCC CV57 exactly at
  `0.8815220699597315` and separately sealed detection, ordinary association,
  matched-lineage fragmentation, and division counts (SHA
  `60a1f283…1b5d4`). No weight refinement, score promotion, machinery re-point,
  or submission occurred.

- 2026-08-03 — **SYNTH-DIV-SIM2REAL-V1 STRUCTURAL FAIL / V2-REACHABLE FROZEN.**
  The first full fit stopped before any optimizer/model artifact when
  `seq_0019` parent 1834's exact daughters were not both in the preregistered
  nearest-8 inference universe. V1 is PARK-C. Before aggregate coverage or any
  model/real outcome, amendment 2 froze the named v2 correction: keep nearest-8,
  exclude only unreachable divisions from pair-head loss, retain every eligible
  parent in event-head loss, and require `>=95%` reachability overall and in each
  SHA split. No real label, graph, score, GPU, submission, or RunPod action
  occurred. (Owner + Codex root)

- 2026-08-03 — **SYNTH-DIV-SIM2REAL-V1 PAYLOAD AUTHENTICATED / LOCAL TRAINING
  OPEN.** The public generator-v3 output completed at exactly 3,716 files and
  19,868,008,592 bytes. Streamed SHA-256/CRC/schema authentication returned
  `AUTHENTICATED`, tree SHA `9a87684a…029865ec`, and exact aggregates
  1,539/2,174/4,056,226/165,267 (static/sequences/nodes/divisions), while never
  materializing image arrays. The 24-test model/authenticator/adapter suite is
  green; full CPU training may now run under the frozen four-hour kill. CTC
  remains permission-blocked and unused; real labels, graphs, scores, GPUs,
  submissions, and RunPod remain untouched. (Owner + Codex root)

- 2026-08-03 — **PUBLIC TEMPORALUNET V32 FULL REPRODUCTION SHARD 0/6
  LAUNCHED.** All six disjoint 27-video packages compile and pass the shared
  inline prelaunch guard. Fresh slug
  `onebitbrain/g02-public-temporalunet-v32-full-s00-v1` Version 1 launched with
  exact code SHA `a5467ae7…37f`, shard manifest `eeaf0ed5…fee5`, post-receipt
  scratch cleanup, and a 10,800 s whole-kernel hard kill. The authoritative
  bracket stayed `1.37/30.00 h` at push. Shards remain serial; no score,
  promotion, machinery re-point, submission, or weight refinement exists.

- 2026-08-03 — **PUBLIC TEMPORALUNET V32 ONE-VIDEO PARITY SMOKE PASS / FULL
  FROZEN REPRODUCTION OPEN.** Exact fresh-slug Version 1 completed with pinned
  V32, primary, secondary, population, and runtime hashes; emitted 50,304 nodes
  and 48,724 edges for `6bba_edf14583` in `284.650 s`. Downloaded CSV SHA
  `09c63601…3325` matches its atomic receipt and passes the strict validator
  with zero errors, warnings, merges, or outdegree violations. Quota moved
  `1.26→1.37/30.00 h` (`0.11 h` debit). This opens the sharded 105+57
  immutable-weight run only; there is no local score, promotion, machinery
  re-point, submission, or weight refinement. Terminal adjudication:
  `tiered_surface/branches/public_temporalunet_baseline_v1/SMOKE_TERMINAL_ADJUDICATION.json`.

- 2026-08-03 — **PUBLIC TEMPORALUNET V32 ONE-VIDEO PARITY SMOKE LAUNCHED.**
  Fresh slug Version 1 passed exact artifact/population binding, no-training
  static checks, focused scorer tests, 17 shared prelaunch tests, and inline
  guard (`d0c494a4…222e6` code; 3,300 s hard kill). All queried prior GPU
  kernels are terminal; the stale 2026-07-31 lock was preserved and archived
  after its kernel and PID were proven dead. Bracketed quota remained
  `1.26/30.00 h` at push. No score, promotion, re-point, weight refinement, or
  submission yet.

- 2026-08-03 — **G-04 TRACK VISUALISER V1 READY.** Rendered and validated
  57/57 seekable H.264 animations (117 MiB bundle) plus a worst-first HTML
  index with exact score/edge/division counts, sparse-GT semantics, error
  timelines, and explicit Prompt-3-unavailable state. Browser acceptance found
  57/57 valid media elements, top-10 seeking, no failed videos, no page overflow,
  and no console errors. Observed raw SHA-256 matched the registered 87 GB archive;
  media/fragment hashes and render contracts now fail closed against stale reuse.
  Diagnostic only; no model, score, gate, GPU, submission, or external action
  changed. (Owner + Codex root)

- 2026-08-03 — Owner opened `SYNTH-DIV-SIM2REAL-v1` in parallel with Prompt 1.
  Explorer v6 and generator v3 sources were acquired and hashed; the CC0
  payload download started. The pre-outcome contract freezes real-rate loss
  weighting, geometry/lineage/motion features, real AUROC `>=0.85`, and a full
  TP/FP/J/S frontier on the eventual authenticated Prompt-1 parent. CTC remains
  permission-blocked and unused. No model, graph, score, GPU, submission, or
  RunPod action occurred. (Owner + Codex root)

- 2026-08-03 — **PUBLIC TEMPORALUNET3D OWNER SUPERSESSION OPENED / IMMUTABLE
  V32 PINNED.** Recorded as adoption of an independently trained public
  pipeline, not a DEAD-mechanism re-screen. Exact public Version 32 is local
  (`3f995a97…b7847d`); primary and seed-314159 checkpoint hashes match their
  public manifests (`12f6881e…e771`, `9bac2fa0…e305f`). No local score, base
  promotion, machinery re-point, GPU launch, submission, or weight refinement
  has occurred. Owner gate: substrate/ensembling only below local `0.916`.

- 2026-08-03 — **G-04 LANE-A HIDDEN-COMPATIBLE OUTPUT PASS / SUBMISSION
  PENDING.** Exact kernel version 1 completed.  Downloaded `submission.csv` is
  byte-identical to the promoted public artifact (`13,713,344` bytes,
  `263,297` data rows, SHA `da73c754…84509`); package identity, prelaunch, and
  runtime receipts all pass.  The sole authorized code submission was created
  as ref `55203439` and is PENDING.  Submission receipt `90d2c8a7…0bd88`;
  no retry is authorized.

- 2026-08-03 — **G-04 LANE-B V2 TIMING PASS / FULL KAGGLE PRODUCTION CLOSED.**
  Exact version 1 returned byte-identical computed labels and per-mask features
  across repeats (`43.460/42.774 s`; initialization `30.217 s`), confirmed
  planar labels `26/42/99`, and populated finite native wrfeat plus voxel-cell
  morphology.  CV57 projects `67.734 h` steady (`68.820 h` worst), held-fit105
  `124.767 h` steady (`126.766 h` worst), and combined worst `195.586 h`, all
  beyond the remaining `28.74 h` Kaggle quota.  Kaggle production is closed;
  RunPod remains explicit-owner-gated.  Terminal adjudication
  `0af02b7e…b966f`.

- 2026-08-03 — **G-04 LANE-B CHANGED-SIGNATURE TIMING V2 LAUNCHED.** The
  fresh private slug `onebitbrain/g04-computed-segmentation-timing-v2` passed
  an independently rerun 37-test suite and fresh launch-eligible inline guard
  (`7ecd6778…b3d07`).  Exact version 1 launched under the shared serial/quota
  bracket with code `b33942da…c0f43`; initial status is RUNNING and quota was
  `1.21→1.21/30 h`.  It is timing/diagnostic-only and cannot access detections,
  GT, metrics, or submissions.  Production remains blocked until terminal PASS.

- 2026-08-03 — **G-04 LANE-B V1 DIAGNOSED / CHANGED-SIGNATURE TIMING V2
  SEALED LOCALLY.** Exact raw/config CPU reproduction found planar computed
  labels `26/42/99` (`47/18/42` voxels) with finite checkpoint-native diameter,
  intensity, and inertia; only legacy scikit-image 3-D center-hull solidity was
  `inf`.  V2 replaces that descriptive field with explicitly named voxel-cell
  3-D convex-hull solidity, uses no sentinel, and records affine-dimension plus
  legacy diagnostics.  Native wrfeat12 is unchanged.  The new private slug has
  exactly four files, 37 tests pass, and the shared static guard reports PASS
  with zero failures.  Production remains closed pending terminal v2 timing.

- 2026-08-03 — **G-04 POST-FAILURE QUOTA RECONCILED.** Authenticated Kaggle
  quota is `1.21/30.00 h` used (`28.79 h` remaining; refresh 2026-08-08),
  receipt `18e707b5…80b191`.  This is a rounded `+0.19 h` since the pre-Lane-B
  launch baseline; Lane A is still running, so exact per-kernel attribution is
  deferred.

- 2026-08-03 — **G-04 LANE-B RAW-ONLY TIMING V1 FAIL-CLOSED.** Authenticated
  Step 0 passed in `3.516 s`, including all 30 runtime files and the exact three
  organizer raw-Zarr files with no GT, detection, or metric access.  The worker
  then failed after `75.597 s` at the finite-feature guard
  (`ComputedSegmentationError: computed region features are non-finite`); the
  supervisor terminated at `90.598 s`.  No valid frame time or production gate
  exists.  The exact offending field/object and a provenance-explicit handling
  rule are required before any changed-signature v2; same-signature retry is
  forbidden.  Lane A remains in flight.

- 2026-08-03 — **G-04 LANE-A HIDDEN-COMPATIBLE V1 LAUNCHED.** The exact scored
  APNCC monolith now runs against mounted hidden test Zarrs before the frozen
  label-blind cap5 transform.  Public raw-Zarr parity is byte-exact to
  `da73c754…84509`, an unseen-ID test passes, and all 27 Lane-A tests pass.
  Exact kernel version 1 launched under the shared lock; submission is blocked
  until terminal output authentication.  Quota was `1.05→1.05/30 h`; the
  in-flight timing lane accounts for the earlier `1.02→1.05 h` movement.

- 2026-08-03 — **G-04 LANE-B RAW-ONLY TIMING V1 LAUNCHED.** Final code
  `12fe418f…f355` passed shared static/inline guards and 25 tests, attaches only
  the organizer competition raw source plus authenticated runtime dataset v1,
  and reads no GEFF, learned detections, targets, or metrics.  Exact kernel
  version 1 launched under the shared lock; quota bracket was
  `1.02→1.02/30 h`.  Terminal timing and supervisor receipts are pending.

- 2026-08-03 — **G-04 LANE-B RUNTIME READY / SAFER RAW-INPUT REFACTOR.** The
  corrected private Cellpose runtime carrier is READY at exact version 1 and
  round-trips all 30 files / 223,211,151 bytes byte-for-byte.  The separately
  declared CV57-cache source was absent remotely; exporting the 595 MB local
  derived cache was safety-rejected and will not be retried.  The timing node
  is being changed to authenticate and read the organizer-hosted competition
  Zarr directly.  Zero Lane-B GPU kernels launched; quota remains
  `1.02/28.98 h`.

- 2026-08-03 — **G-04 LANE-A TERMINAL UNSCORED / FORMAT DIAGNOSIS.** Kaggle ref
  `55201829` reached `COMPLETE` with a blank score, `total_bytes=0`, and the
  generic notebook-output incorrect-format diagnostic.  The downloaded output
  remains byte-identical and organizer graph parsing passes, so transport versus
  format is under diagnosis; no blind retry.

- 2026-08-03 — **G-04 LANE-A SUBMISSION CREATED / SCORE PENDING.** The private
  dataset and CPU/no-network code kernel both reached exact version 1, terminal
  output reproduced candidate SHA `da73c754…84509`, and the serial-locked code
  submission created Kaggle ref `55201829`.  Authenticated reconciliation shows
  `PENDING` with no public score yet; GPU quota was unchanged.  Lane B has not
  launched while its timing-kernel prelaunch contract is being corrected.

- 2026-08-03 — **G-04 LANE-A DIRECT FILE TRANSPORT REJECTED / NO SUBMISSION
  CREATED.** Exact candidate SHA `da73c754…84509` passed the remote CSV validator
  with four datasets, 134,409 nodes, 128,888 edges, 15 forks, zero merges, and no
  outdegree above two.  The shared serial guard uploaded the file, but Kaggle
  returned HTTP 400; authenticated submission history remained unchanged with
  latest ref `54942864`.  No slot or score was created.  The likely code-competition
  transport is being packaged and will be validated before any distinct retry.
  (Codex root)

- 2026-08-03 — **G-04 LANE-A LOCAL CANDIDATE / LANE-B REAL 3-D PILOT PASS.**
  The official 57-video APNCC baseline reproduced exactly `0.881522069960`.
  Permissive `unfiltered_geometry_cap5` scored `0.881635476737`
  (`+0.000113406777`), division `0/11/42`, after 239 fork edits; it qualifies
  under the owner rule and is being packaged, but no submission has occurred.
  Authenticated Cellpose3 `cyto3` SHA `2dc3087a…b93c` segmented one real
  `64x256x256` frame in `881.161 s` CPU: 171 masks, 224,324 foreground voxels,
  and `234/243` matches at radius 15.  Radius 30 covers all 243 with honest
  nearest-region provenance; the one-frame pilot remains partial and cannot
  enter the production manifest.  (Codex root)

- 2026-08-03 — **G-04 OWNER REFRAME RECORDED / PARALLEL EXECUTION OPEN.** Banked
  local/public `0.881522069960/0.881` is below median at approximately rank
  `1100/1800`; the exhausted-surface conclusion and legacy association-continuity,
  44b6-first, `+0.005`, near-zero-FP, and local-0.89 gates are withdrawn for G-04.
  Opened independent permissive-division and computed-segmentation/wrfeat lanes.
  Any observation-distinct lane winner strictly above the incumbent locally is a
  submission candidate.  No score, GPU launch, or submission has occurred yet.
  (Owner + Codex)

- 2026-08-03 — G-03 h005 V2 failed top-level STEP0 after `3.5865 s` because
  its four-source GPU-worker runtime manifest was absent or byte-divergent; no
  output, install, scientific import, CUDA/data/optimizer/checkpoint, held-fit36,
  CV57, score, or slope exists. V2 cost `0.00 h`; quota is `1.02/28.98 h`.
  Frozen no-V3 closes Lane 0 infrastructure-terminal and preserves h030's 25 h.
  No G-02 gate, tier, CV57, submission, hybrid, or RunPod action changed.
  (Codex root)

- 2026-08-03 — G-03's sole dependency-repaired h005 version 2 launched after
  exact CPU-probe, static, inline, serial, and quota guards; Kaggle accepted it
  at `2026-08-02T21:32:20Z` and quota stayed `1.02/28.98 h` immediately after
  push. It is the only live GPU job and is monitored no more than hourly. No
  G-02 gate/tier/CV57/submission/hybrid/RunPod action changed. (Codex root)

- 2026-08-03 — G-03's CPU-only h005 runtime probe passed exact 14-wheel,
  8-wheel-install, numeric-stack, and fixed 6bba raw+GEFF checks in `9.223 s`;
  GPU/optimizer/checkpoint/held/CV57/metric access and quota debit were zero.
  The unchanged-core sole V2 passed static release audit but still awaits fresh
  serial/quota/inline guards. G-02 remains unchanged. (Codex root)

- 2026-08-03 — G-03's frozen CPU-only h005 runtime probe version 1 launched
  after fresh inline and quota guards; GPU quota stayed `1.02/28.98 h`. Its
  terminal result is pending and the sole GPU repair remains closed. No G-02
  gate, tier, CV57, submission, hybrid, or RunPod action changed. (Codex root)

- 2026-08-03 — G-03 published and authenticated a private hash-pinned h005
  runtime carrier with no quota movement and froze a CPU-only release-path
  probe as the mandatory gate before any one dependency-repaired version 2.
  No GPU repair, checkpoint, held-fit36 score, slope, h030, CV57, submission,
  hybrid, RunPod, or G-02 reopening occurred. (Codex root)

- 2026-08-03 — G-03 h005 v1 passed authenticated STEP0 but failed importing its
  frozen training core because `zarr` was not installed. Training, checkpoint,
  held-fit36 score, and slope are absent; quota moved `1.01→1.02 h` used. V1 is
  infrastructure-invalid, with evaluation/h030 closed and no G-02 reopening.
  (Codex root)

- 2026-08-03 — G-03 dense deformable motion completed all 41 frozen 44b6
  identities and returned `FAIL_PARK_C` for DELETE and occupied SWAP under the
  unchanged B0→A1→A0 consumer. No learned displacement, graph, GPU, 6bba,
  CV57, hybrid, RunPod, or submission followed; G-02 remains archived and
  unchanged. (Codex root)

- 2026-08-03 — G-03 fixed h005 training-kernel version 1 launched under its
  fresh inline STEP0 and serial/quota guards (`1.01/28.99 T4-h` used/remaining
  before and immediately after push). This owner-amended G-03 action does not
  reopen any G-02 mechanism or change any G-02 gate, tier, CV57 action, or
  submission. (Codex root)

- 2026-08-03 — G-03 owner-amended orthogonal lanes recorded without reopening
  G-02. The trained raw-intensity fork probe failed cross-family 44b6 and stopped
  before final 38/GPU/graph access; dense deformable registration remains a
  CPU-only frozen-gate run; fixed h005 has a corrected version-2 carrier and
  eligible inline prelaunch receipt but has not launched. No archived G-02
  verdict, gate, tier, CV57 action, or submission changed. (Codex root)

- 2026-08-03 — **G-03 ZERO-INIT H000 RECONCILED / G-02 UNCHANGED.** Frozen
  native CUDA banks give 36/36 byte-identical held-fit graphs and corrected
  G-03 h000 `0.7630876414338513`; the archived `0.829194058080` is CV57 and not
  its same-population baseline. G-03's Branch-B dossier fails and h005/h030 stay
  unfunded. No G-02 mechanism, tier, gate, CV57, GPU, or submission reopened.
  (Codex root)

- 2026-08-03 — **OWNER BARRIER DECISIONS RECORDED / A7 FROZEN SEPARATION FAIL /
  EVIDENCE PARK.** Owner accepted the R9 and AE-P3/R0 park retractions and all five
  D3 “no” recommendations. After G-03 R4 terminated, A7's exact 45,961/134 occupied-
  swap universe ran through the inherited B0-first/A1/A0 five-fold gate. A1 AUC was
  `0.627866/0.828496/0.610208/0.659067/0.654961`, retention `0` throughout, and all
  absolute/lift clauses failed. A7 is now evidence-parked; no threshold or graph was
  opened. Result `../../g03/screening/a7_separation_measurement_v1/run_v1/SEPARATION_RESULT.json`
  (`3143a31d…f96d`). The proposed node `>=-0.002` amendment remains owner-only draft,
  unapplied, with no historical verdict change. (Codex root)

- 2026-08-03 — Adopted the durable **CEILING BEFORE MACHINERY** and
  atomic-receipt monitoring rules in `tiered_surface/ALIGNMENT.md`: below-gate
  ceilings do not fund promotion machinery; receipt jobs are event-waited with
  at most hourly status checks and no unchanged narration. No scientific gate,
  tier, experiment, or external action changed. (Codex root)

- 2026-08-02 — **44B6 GATE VALIDITY FAIL / ABSTENTION ORACLES COMPLETE / OWNER BARRIER.** Existing authenticated caches show APNCC-DIRECT versus the exact pre-APNCC `c=0.25` control at held-44b6 adjusted/raw/node `+0.006949544395/+0.006411258660/-0.001307630206`; the known full-CV57 winner would have failed the locked `>=+0.005/0/0` gate on node recall. This is recorded as a methodological false-negative finding, with no autonomous gate change or reopening. Exact own-candidate oracles are A7 `134/45,961`, reverse literal passes `0/268` beneficial (all neutral; selected `0/53`), and R0 `5/38,081`; thus A7 and R0 were threshold misses and cannot justify evidence-parks, while authentic literal reverse consensus is evidence-negative. The earlier R9/AE-P3/R0 cause reclassifications are retracted and the park audit remains `22/4/6/1/2`; zero tier moves. The valid near-zero 44b6 review contains SWAP, NODETOPO, PEFT, DELETE, and historical DIV-A; none merits CV57 under current scope, and no CV57 object, GPU, Kaggle, RunPod, 6bba, submission, or gate was opened. Report `tiered_surface/diagnostics/gate_validity_abstention_oracle_20260802/REPORT.md` (`71591478…b969e`); machine result `e51a9db9…6ba9`. (Codex root)

- 2026-08-02 — **FOUR NARROW CPU LANES TERMINAL / ALL SCIENTIFIC CANDIDATES KILLED / ZERO TIER MOVES.** A7 phase-template selected `0/45,961` occupied swaps and is byte-identical to APNCC on 36/36 videos. Authentic reverse endpoint consensus sealed 53 swaps but found `0/53` beneficial, closing the graph and leaving AE-P6/R8/C3 blocked. The distinct R0 topology-0 shell no-link screen evaluated 38,081/2,514,005 held-44b6 shell rows and failed fixed B=4 precision at `0/4`; its perfect-B4 adjusted ceiling was only `+0.000367524`, so the graph remained closed. All exact deployed adjusted/raw/node deltas are `0/0/0`; the four requested properties pass for A7 and R0, while reverse is not appearance-grounded because historical `wrfeat` is zero-filled. The parallel 35-row park-audit snapshot was evidence/dependency/control/policy/owner-override `22/4/6/1/2`; joining A7 and R0 reclassifies R9 plus AE-P3/R0 as evidence, making the final cause census `25/3/6/1/0`, with only DIV-F2, R2, and R3 dependency-parked and no tier move. Strict swap guards and resumable label-blind shards allowed completion alongside G-03. No GPU, Kaggle, RunPod, 6bba, CV57, submission, DEAD, or plan action. (Codex root)

- 2026-08-01 — **GLOBAL-REASSIGN ORACLE TERMINAL / COMPILER ESCALATION / PROPOSAL ONLY.** The full 36-video local-CPU GT-dependent node passed T0/T1 controls and measured T2∞ `+0.09641286660426185` [GT-DEPENDENT] versus T3 `+0.10497364637634421` [GT-DEPENDENT]. Section 4 is `CONTINUE_PROPOSAL_ONLY`: precedent-adjusted `+0.01906337599412862` [GT-DEPENDENT] before raw T2∞−T0 `+0.0778414699637755` [GT-DEPENDENT]. Terminal status remains `ESCALATE_COMPILER_INEXPRESSIBLE`: the frozen batch compiler rejects same-action retained division anchors, not under-merged chains. T2 tiers are identical with histogram `{1:453,2:6}` [GT-DEPENDENT], correcting the banked chain-depth diagnosis; T2/T3 install the same 465 correct edges [GT-DEPENDENT]. No selector, training, GPU/Kaggle, 6bba, CV57, submission, tier, DEAD, plan, or LEDGER action. Report: `../../../dag_prompts/2026-08-01-global-reassign-oracle/runs/N0_GLOBAL_REASSIGN_ORACLE.md`. (Codex root)

- 2026-08-01 — **ARCHIVED PARK-A/UNEXPLORED AUDIT / NO G-02 REOPENING.** All 14 machine PARK-A entries and 18 PARK-B entries were reconciled against the newer terminal closeout under the owner's corrected G-03 continuation gates (`>=+0.010` at 30 T4-h; `>=+0.035` at 90). Governance entries remain controls; no archived scientific parent is a standalone 90-hour route, `NODETOPO-O0`'s `+0.009812493` oracle is below even the 30-hour line, and stale DINOv2/PEFT supplier rows remain superseded by terminal failures. AE-P7 crop-interface evidence may inform G-03 Phase-2 preparation only. No G-02 node, graph, CV57, GPU, submission, tier, or DEAD mechanism reopened. Kaggle failure coverage is now 28 incidents / 22 families after a pre-GPU G-03 KFG-08 carrier-layout incident exposed and corrected archive-prefix stripping. Audit: `../../g03/screening/UNEXPLORED_PARKA_DIRECTION_AUDIT_20260801.md`. (Codex root)

- 2026-08-01 — **KAGGLE KERNEL REVIEW / NO G-02 RETRY OR SUBMISSION.** Live status reconciliation confirms native-fit v1 and PEFT v1 are COMPLETE; their valid outputs remain respectively native-base REJECT and PEFT PARK-C. All screenshot G-02 error lineages are terminal, superseded by valid local outcomes, or infrastructure-only artifacts with closed authority. APNCC ref `54942864` remains the distinct public `0.881` final; no completed cache/preflight/proof kernel is submission-eligible. The failure registry now covers 27 incidents / 22 families, including nested-output supervisor drift, and `cmux_ops.py` structurally refuses a kernel push without a fresh eligible guard receipt whose metadata/code hashes still match. No Kaggle mutation or submission occurred. (Codex root)
- 2026-08-01 — **LOCAL CV57 ONE-SHOT COMPLETE / NATIVE REJECT / HYBRID NO-OP.**
  The sole sealed `r08_c08_a035_off` measurement scored `0.829194058080`
  adjusted/combined, `0.834460844608` raw, recall `0.987196658416`, edge
  TP/FP/FN `32564/2756/3704`, and division `0/0/42/0`, trailing APNCC by
  `-0.052328011880`. All 57 selected graphs had zero forks, so the preregistered
  hybrid is graph-identical to APNCC at `0.881522069960`. Independent pre/post
  audits pass; there was no alternative, tuning, repeat, GPU/Kaggle action,
  RunPod, or submission. Frozen-result execution-field mislabel is adjudicated
  without rewrite or rerun. Terminal SHA `0add0c3f…60af95a`. (Codex root)
- 2026-08-01 — **LOCAL CV57 ONE-SHOT READY PREACCESS / REMOTE V2-LAST
  UNCHANGED.** The complete 57-video cache, exact checkpoint/config, and source
  are already local. CPU fit-only parity passed 3/3 across both families and
  threshold-stress cases with zero selected-edge differences and zero threshold
  flips versus the T4 bank. Runner/scorer/source/population are sealed in
  `CV57_LOCAL_ONE_SHOT_CONTRACT.json` SHA `2ca77e53…a1c6e`; preflight loaded all
  27,456,880 parameters without a CV57 cache-file or GT read. No v3, tuning,
  alternative, submission, RunPod, or Kaggle action. (Codex root)
- 2026-08-01 — **CV57 V2 TERMINAL PREACCESS INFRASTRUCTURE FAIL / EXCEPTION
  CLOSED.** V2 version 1 found zero of its three exact kernel-source cache roots
  and failed Step 0 in `0.000216127 s`. Heavy work=false; cache/model/GT/graph/
  scorer access and measurements are all zero. Native and hybrid CV57 remain
  NOT EVALUATED. Terminal global quota is `7.20/22.80 h`; its `0.03 h` movement
  is only an upper bound because G-03 A v2 is concurrent. V2 was the last repair:
  no v3, submission, or RunPod. Terminal SHA `2abce8f4…46e928`. (Codex root)
- 2026-08-01 — **CV57 V2 VERSION 1 LAUNCHED / RUNNING.** The fresh final-repair
  slug accepted exactly one version after the preaccess freeze and both guards.
  First authenticated status is RUNNING; bracketed quota remained
  `7.17/22.83 h` used/remaining with zero immediate debit. G-03 A v2 is concurrent,
  so later quota movement requires attribution caution. No v3, submission, or
  RunPod. Launch receipt `CV57_V2_LAUNCH_RECEIPT.json`. (Codex root)
- 2026-08-01 — **CV57 V2 PREACCESS INFRASTRUCTURE REPAIR READY / MEASUREMENT
  COUNT STILL ZERO.** Downloaded v1 evidence proves no cache, GT, model, graph,
  or score access, so it did not consume the owner's single authorized CV57
  measurement. Final v2 changes only exact allowlisted kernel-source mount
  resolution and requires the complete 57-video/114-file hash manifest before
  heavy work. Seven scientific functions are AST-identical; static+inline guards,
  manifest validation, and 29/29 cmux tests pass. V2 is unpushed; no v3,
  alternative, tuning, post-access repeat, submission, or RunPod. Repair SHA
  `58277d7d…134a68`. (Codex root)
- 2026-08-01 — **SELECTED-ONLY CV57 V1 TERMINAL PRE-HEAVY-WORK FAIL / NO
  MEASUREMENT / NO RELAUNCH.** Version 1 failed Step 0 in `0.000177395 s` with
  `exact asset or kernel-source mount is absent`; heavy work=false, competition
  visits=0, fragments=0, and native/hybrid CV57 remain NOT EVALUATED. The push
  bracket was unchanged at `7.00/23.00 h`; terminal global quota `7.03/22.97 h`
  includes concurrent G-03 A v2 accrual and is not attributable to this failure.
  The precommitted no-relaunch decision closes the exception; no submission.
  Terminal report SHA `d661b43e…0136c1`. (Codex root)
- 2026-08-01 — **SEALED SELECTED-ONLY CV57 V1 LAUNCHED / RUNNING.** Fresh slug
  `onebitbrain/g02-trackastra-native-cv57-v1` accepted exactly version 1 under
  the shared serial lock. Push-bracket quota stayed `7.00/23.00 h`
  used/remaining (`0.0` immediate debit). It runs as the owner-preferred second
  session and can emit only the sealed graph. Post-push asset status confirmed
  READY version 4 versus the launched descriptive version-1 label; exact payload
  hashes remain the Step-0 gate, the bookkeeping defect is recorded, and no
  relaunch is permitted. No submission. (Codex root)
- 2026-08-01 — **NATIVE CV57 OUTPUT SURFACE HARDENED TO SELECTED-ONLY.** Before
  launch, the package was narrowed from returning a reusable candidate-edge bank
  to returning only the 57 graph fragments for sealed `r08_c08_a035_off`. The
  original fit/hybrid evaluator remains byte-pinned; the new exact one-shot local
  evaluator, package, contract, and freeze are hash-bound, and current static+
  inline guards pass code SHA `6a4bf65b…2734d1`. The asset dataset's remote
  version declaration was reconciled to authenticated current version 4 before
  launch; exact payload hashes remain fail-closed in Step 0. No remote action or CV57
  outcome occurred; the shared lane remains occupied by G-03 A v2. (Codex root)
- 2026-08-01 — **NATIVE ONE-SHOT CV57 PACKAGE READY / WAITING FOR SERIAL GPU
  LANE.** The independently validated 108-way fit result and irrevocable
  `r08_c08_a035_off` selection are bound into a fresh private 2×T4 kernel. Its
  exact 57-cache source authentication, immutable cache manifest, code/metadata/
  contract hashes, internal `17,100 s` deadline, and static+inline prelaunch
  guards all pass. No kernel push, GPU debit, CV57 measurement, tuning, repeat,
  or submission has occurred; G-03 A v2 still owns the shared GPU lane. (Codex root)
- 2026-08-01 — **NATIVE FIT105 EXACT SWEEP COMPLETE / SELECTION SEALED BEFORE
  CV57.** All 108 frozen names (76 distinct aggregate graphs) were officially
  scored. The frozen rule selected `r08_c08_a035_off`, score `0.811024400045`
  (`-0.049492037080` vs fit APNCC), with divisions OFF and division Jaccard `0`.
  The preregistered hybrid inserts zero forks and is byte-equivalent in score to
  fit APNCC. Independent ranking replay passed; seal SHA `0c1b3ed…9836bcc`.
  Exactly one selected native+hybrid CV57 measurement is now open, still subject
  to the shared no-concurrent-launch controls. (Codex root)
- 2026-08-01 — **NATIVE FIT105 MATERIALIZATION COMPLETE / PRE-SCORE INFRASTRUCTURE
  RESUME REPAIR 2.** All `108 × 105` frozen variant fragments and manifests were
  written before macOS denied `SC_SEM_NSEMS_MAX` at exact-score worker-pool creation;
  zero candidate scores existed. A hash-pinned adapter now attests and reuses the
  complete materialization, then invokes the frozen evaluator's exact scorer,
  selection, hybrid, and report assembly unchanged outside that sandbox. This is an
  infrastructure-only repair: no scientific parameter, population, rule, fit outcome,
  or CV57 access changed. (Codex root)
- 2026-08-01 — The G-03 owner-authorized self-contained A v2 transport repair
  was accepted and remains RUNNING on two authenticated status probes; B v2
  remains contingent on a verified A handoff. G-02 scope is unchanged. (Codex root)
- 2026-08-01 — G-03 state corrected after owner amendment: A v1 remains a
  terminal pre-GPU packaging failure, while exactly one self-contained A v2 and
  contingent B v2 transport repair has been preregistered. G-02's narrowly
  scoped native-base exception is unchanged. (Codex root)
- 2026-07-31 — **NATIVE SWEEP PRE-OUTCOME SCORER-MANIFEST REPAIR 1.** The first
  invocation failed before output creation, native-bank load, graph materialization,
  or score because the old split manifest attested superseded upstream scorer hashes.
  A metadata-only execution manifest now supplies the three hashes already pinned in
  the outcome-blind fit scorer freeze. Code, scorer, population, 108-grid, thresholds,
  ranking, and selection are unchanged; one repaired invocation is authorized. (Codex root)
- 2026-07-31 — **APNCC FIT105 CONTROL COMPLETE / 66-FRAGMENT RESUME PARITY PASS.**
  The unchanged frozen CPU implementation completed 105/105; every preserved
  fragment is byte-identical to its recomputation. Exact control combined/adjusted
  is `0.860516437125`, raw `0.867176587961`, recall `0.988149479570`, and
  division `0/0/76`, Jaccard `0`. No GPU, Kaggle, CV57, or submission action.
  The frozen native 108-variant scorer is now active. (Codex root)
- 2026-07-31 — G-03 Phase 1's sole infrastructure A kernel terminalized at 0.70 s
  before Torch/CUDA/data work (`ModuleNotFoundError` from Kaggle's code-file-only
  packaging); terminal quota is unchanged at `5.43/24.57 h` (0.0 debit). G-03 is
  stopped by its own locked no-retry gate. This changes neither the narrow G-02
  native-scorer exception nor the locked final slot. (Codex root)
- 2026-07-31 — **OWNER-SCOPED NATIVE SCORER + ONE-SHOT CV57 EXCEPTION OPEN.**
  The owner expressly authorized completing the preserved fit105 APNCC control,
  scoring the frozen 108 native variants and preregistered hybrids from the existing
  bank, sealing exactly one selection, and then measuring only that native variant
  and hybrid once on CV57. G-03 Phase 1 is unchanged. GPU is conditional on the
  selected CV57 bank being absent and remains subject to all standing launch gates;
  no quota soak, CELLECT, RunPod, tuning, repeat CV57, or public submission. Authority:
  `tiered_surface/branches/sa_native_base_wave_v1/OWNER_SCOPE_AMENDMENT_20260731.json`.
  (Owner + Codex root)
- 2026-07-31 — **LEGACY TRACKASTRA V1 TERMINAL RECONCILED / QUARANTINED UNSCORED.**
  Remote v1 is COMPLETE with exact version delta `0→1`; the downloaded 105-video
  bank passes SHA/schema validation with 2,732,827 nodes and 12,523,336 candidate
  edges. Quota is `5.43/24.57 h` used/remaining (`0.37 GPU-h` debit), and the
  serial lock is released. Under the newer G-03 owner charter, no graph was
  materialized or scored, no variant/CV57/submission/dataset action occurred, and
  the APNCC control was stopped at 66/105 with partials preserved. Receipt:
  `tiered_surface/branches/sa_native_base_wave_v1/TERMINAL_OUTPUT_ADJUDICATION.json`.
  (Codex root)
- 2026-07-31 — Owner's G-03 end-to-end association-training charter supersedes G-02. G-02 is archived: retain its evidence, requirement surface, and byte-exact transport, but do not continue its native Trackastra wave. G-03 Phase 0 architecture dossier selected a new crop-adapter mechanism with no GPU work. (Codex root)
- 2026-07-31 — **FIT105 CONTROL + 108-VARIANT SCORER FROZEN OUTCOME-BLIND.**
  APNCC identical-population control code `a8f6ed7…37e` and native materialize /
  exact-score / hybrid code `458e958…63e` were pinned while Trackastra v1 was
  still RUNNING and before either score outcome. All variants emit 105 fragments;
  score reuse is allowed only for an all-105 SHA-256-identical graph class. A
  pre-outcome amendment added only the fit105-control delta field; selection logic
  did not change.
  APNCC CPU control is advancing; no CV57 or submission access. Freeze:
  `tiered_surface/branches/sa_native_base_wave_v1/FIT_SCORING_IMPLEMENTATION_FREEZE.json`.
  (Codex)
- 2026-07-31 — **TRACKASTRA NATIVE FIT V1 LAUNCHED / RUNNING.** Fresh remote
  slug `onebitbrain/g02-trackastra-native-fit-v1` moved exactly `0→1` at
  `07:36:41Z`; first poll is RUNNING. The owner-authorized supervisor holds the
  shared serial lock through terminal download, with prelaunch quota
  `5.06/24.94 h` used/remaining and `18,000 s` server hard timeout. A first
  attempt stopped before push because Kaggle's empty-search literal `Not found`
  was not parsed; the target-aware repair consumed no version or GPU and is
  durably recorded. No competition source, public submission, or RunPod. (Codex)
- 2026-07-31 — **NATIVE-BASE WINDOW OPEN / TRACKASTRA FIT PREREGISTERED / CELLECT
  NO-SPEND.** Owner-authorized Q1/Q2 asks whether a pretrained tracker's native graph
  and forks beat APNCC `0.881522069960`. Fit105 selection, one-shot CV57 exposure,
  hybrid overlay, exact scorer vector, `17,100 s` internal / `18,000 s` external kill,
  and immediate above-control halt are frozen in preregistration
  `f1ab1f5…9c0` / `8fec69e…1ef`. Trackastra fit code is `57117bf…b731a`; both guards
  PASS. CELLECT B is abandoned at `0.0` new GPU-h under the same-signature stop rule;
  its prior full inventory already found zero checkpoint/loader bytes and no carrier
  changed. No public submission or RunPod action. (Codex)
- 2026-07-31 — **RAW3D STAGE 0 PARK-C / CAMPAIGN CONSOLIDATED; BANKED FINAL
  EXPLICITLY SELECTED.** Pre-outcome preregistration `5b76da90…8588dd` fixed a
  `+0.010` margin over DINOv2 on both mean and fold-floor AUC. The CPU-only,
  label-blind raw-Zarr stream completed all 41 DELETE videos in `254.704 s`:
  A1 mean/floor AUC `0.677967209/0.599374874`, mean retention `0.006094647`,
  full-pop retention `5/1,118`; A0 mean `0.646573736`. Both advance clauses
  and measured `0.85/0.020394` fail, so RAW3D-NOLINK-MVE is PARK-C, Stage 1
  remains closed, and the campaign consolidates. Separately, Kaggle final
  selection changed `0/2→1/2` with banked `bhap-submit` v1 / ref `54942864`
  checked; deadline `2026-09-29 23:59 UTC`. No new submission, training,
  acquisition, GPU, full cache, RunPod, or DEAD action. The owner-approved
  RAW3D exception is now recorded as scoped plan amendment §6.6; it changes no
  authority outside this terminal node.
  (Codex)
- 2026-07-31 — **CODEX AUTONOMOUS CMUX PACK AUTHORED / NO EXTERNAL CALL.** Added
  `dag_prompts/2026-07-31-gpu-window/codex/` (`STATUS`, `C0`–`C4`) so an approved
  node can run unattended under the `AGENTS.md` standing authorization. Doctrine
  follows the existing `run_cellect_cmux_roundtrip.py`: **bounded `command_plan()`
  argv whitelist, never free-form shell** in the owner's authenticated session.
  `C0` specifies shared `kaggle_prelaunch/cmux_ops.py` — nine enumerated ops,
  `O_CREAT|O_EXCL` serial lockfile, credential-scrubbed receipts carrying argv and
  output hashes, **no retry logic**, `kernel_push` refusing while a version is
  `RUNNING`/`QUEUED` (direct fix for the `0.74` GPU-h race), and
  `competition_submit` refusing without an `eligible: true` predicate receipt.
  `C0` remains **unbuilt** and is still the open `KFG-18` fix: the v1 run again met
  the known owner `403` on first dataset status and correctly did not retry, which
  is exactly the endpoint-scope behavior `C1` encodes (verify by
  **download-and-hash**, never `datasets files`, never "accept the rules").
  `C4` encodes submission predicate `P1`–`P7`. Pack reconciled to the terminal PEFT
  FAIL and is now **lane-parameterized**; `C1`–`C3` have **no authorized target**
  until the owner resolves the `training_feasibility_v1/DECISION_DOSSIER.md`
  Option 1 / Option 2 barrier. No cmux invocation, Kaggle asset, kernel, push, GPU
  debit, or submission occurred. (Claude)
- 2026-07-31 — **PEFT TERMINAL AUTHORITY SUPERSESSION SEALED / DOSSIER
  RECALIBRATED / HUMAN BARRIER.** The 12:04 no-amendment reconciliation is now
  explicitly historical; owner Amendment A at 12:12 superseded it for the
  PEFT node only, before the 12:19 push and 12:27 terminal result. Fresh
  authoritative quota is `5.06/24.94 h` used/remaining, so the `0.04 T4-h`
  debit is fully reflected. The training dossier now separates PEFT's
  workload-specific `118.6×` miss from Trackastra's relevant `2.72×`
  full-cache projection miss, banks the measured CPU-cheap but DEAD 30k crop
  model, and recommends only a zero-GPU warm-start asset gate before any
  proposed raw-3D fit. No plan amendment, training, GPU launch, or submission.
  (Codex)
- 2026-07-31 — **OWNER STANDING KAGGLE AUTHORIZATION / NO DUPLICATE
  SUBMISSION.** Productive authenticated Kaggle kernel launches, competition
  submissions, and monitoring no longer require a repeated permission prompt;
  scientific/plan gates, serial lock, authoritative quota, prelaunch integrity,
  hard kill, and terminal monitoring remain binding. Fresh reads show
  `5.06/24.94 h` used/remaining, refresh `2026-08-01T00:00:00Z`, competition
  deadline `2026-09-29T23:59:00`, and no in-flight recent G-02 kernel. The sole
  ready CSV is exactly the already-scored public-`0.881` artifact
  (`6944ed58…bea3`), so no submission or GPU launch occurred. (Owner + Codex)
- 2026-07-31 — **TRAINING FEASIBILITY DOSSIER / HUMAN AMENDMENT BARRIER.**
  Banked PEFT's `0.634749` AUC plateau, graph-selection paradox
  (`1/2/2/2/2`, selected AUC `0.496495`, graph effects order `1e-4`), and
  `4.8 h→145.680 s` projection miss. Re-costed Trackastra useful-checkpoint
  fine-tuning at `4.05–5.96 T4-h` and a compact raw-3D/no-link MVE at
  provisional `2–6 T4-h` after matched calibration. An exactly two-option
  primary-source dossier recommends Option 1 (`RAW3D-NOLINK-MVE`) as the only
  new-information falsifier. No plan amendment, training, GPU kernel,
  submission, DEAD change, or RunPod action occurred. (Codex root)
- 2026-07-31 — **§6.2 PEFT V1 TERMINAL FAIL / CASE B / PARK-C.** Sole v1
  completed; exact version delta `+1`, Step 0 `1.500973 s`, zero competition
  visits, training `145.680 s`, and authoritative debit `0.04` T4-h
  (`24.98→24.94 h` remaining). The frozen 100:1 sample and full 1,345,497-action
  evaluation authenticate. Descriptive AUC reached `0.634749` at epoch 8, but
  graph-only nested selection (epochs `1/2/2/2/2`) yields AUC `0.496495`,
  retention `0.002683`, and adjusted/raw/node deltas
  `+0.000133827/+0.000133160/0`; adjusted misses `+0.005`. Frozen case B is
  binding: no larger adapter/full-finetune proposal on this backbone. Local
  replay passes 15/15. 6bba/CV57/package/submission did not run. (Codex)
- 2026-07-31 — **§6.2 PEFT V1 LAUNCHED / EXACT VERSION DELTA ONE / RUNNING.**
  The exclusive audit found zero in-flight campaign GPU kernels, zero target
  versions, and authoritative quota `5.02/24.98 h` used/remaining. The fresh
  private asset create returned zero; the first status query returned the known
  owner 403 and was not retried. Same-kernel inline Step 0 therefore remains
  mandatory before CUDA imports. One push produced server-assigned v1 and
  post-count delta `+1`; hard timeout is 18,000 s and the serial supervisor
  retains the lock. No scientific result, debit, downstream gate, or
  submission yet. (Codex)
- 2026-07-31 — **§6.2 PEFT SURVIVABLE ENVELOPE FROZEN / ONE SERIAL VERSION
  AUTHORIZED / PRELAUNCH READY.** Before any adapter score, fixed a 100:1
  DELETE training sample (all `1,118` positives + `111,800` deterministic
  negatives, digest `a1e1f638…51571`) and complete-population evaluation.
  The sealed five-fold correction resolves the prior empty grouped fold under
  explicit owner authority. The 2,665-parameter adapter remains `<1%`, with
  3,744 optimizer steps, eight epochs, 25-step atomic resume, a conservative
  `2.25` T4-h projection, at-step-250 `>3 h` abort, and hard `5 h` kill.
  Targeted Step 0 passes tar/expanded at `0.084/0.068 s` with zero competition
  visits; focused tests pass 6/6 and guard tests 16/16. Fresh authoritative
  quota and in-flight/version checks remain mandatory immediately before the
  sole push. No remote mutation, fitting outcome, or debit yet. (Codex)
- 2026-07-31 — **OWNER OVERRIDE: §6.2 PEFT REOPENED / H0 OPTION A / STILL NOT
  LAUNCHED.** Shown the adjudication against it, the owner overrode the
  `TERMINAL_CAMPAIGN_REPORT.md` closure **for `sa_peft_adapter_v1` only** — no
  general reopening of the tiered surface, DEAD register, PARK-C suppliers, `6bba`,
  CV57, submission, or RunPod — and chose **H0 Option A**: five frozen hash folds
  over the unchanged `1,345,497`-action DELETE population, five-outer-fold kill
  clause, all other preregistered fields unchanged, **`8.0` T4-h** projected and
  authorized above the `5` h auto-ceiling. Receipt
  `sa_peft_adapter_v1/PREREGISTRATION_AMENDMENT_A.json` is pre-outcome and leaves
  `PREREGISTRATION.json` (`0d96b336…1ad8`) byte-unchanged; every fold has nonzero
  support (`306/99/167/195/351` beneficial). **Permanent caveat: those five hash
  folds are not naive** — the supplier separation gate already used them, including
  for the DINOv2 and C2 `FAIL_PARK_C` results on this same population; any PEFT
  result must carry this and may not be presented as a clean-room nested-OOF
  estimate. K0 done: the authenticated Trackastra backbone is rescued to
  `assets/trackastra-authentic-v1/` with `24c290ce…981f` / `2ede0848…a0e` verified
  post-copy and gitignored. No fitting, projection, Kaggle asset, kernel, push, GPU
  debit, or submission; quota unchanged at `5.02/24.98 h`. Next node is K1
  (implementation, CPU-only). (Claude + owner)
- 2026-07-31 — **GPU-WINDOW ALLOCATION ADJUDICATED / NO LAUNCH / RECOMMEND EXPIRY.**
  Cross-worktree survey of `main` plus all nine `codex/*` branches found **no
  unreconciled science**: their artifacts are already in `main`'s tree and every
  lane is terminal, PARK-C, or needs a newly named mechanism. Under the freshly
  frozen `TERMINAL_CAMPAIGN_REPORT.md`, **`0` of the `24.98` remaining T4-h is
  spendable**; the §6.2 PEFT node is closed both by its own
  `PRELAUNCH_HARD_STOP_INVALID_EXPERIMENT` fold envelope and as an
  "architectural substitution" under the reopening condition. The `0.670412` /
  `0.670235` convergence across two information-disjoint suppliers is read here as
  label-side, not feature-side, and the `+0.075828 → +0.018571397` (`24.49%`)
  oracle-to-realized collapse is offered as independent support that the pointwise
  estimand is the bottleneck. Recommendation: let the quota expire at the
  `2026-08-01T00:00:00` refresh and spend the window on reopening-clause-1 design
  (batch/sequential action valuation), leaving next week's full `30` h to fund it.
  Also found: the authenticated Trackastra checkpoint `24c290ce…981f`
  (`109,906,058` B, `27,456,880` params) and `ctc.zip` `2ede0848…a0e` exist **only**
  under `/private/tmp/trackastra-owner-roundtrip.20260729/resolved/`, with the repo's
  `trackastra-6bba-assets-v1/` empty — a reboot forces another owner-cmux retrieval.
  No Kaggle asset, kernel, push, GPU debit, or submission occurred; quota unchanged
  at `5.02/24.98 h`. Prompt pack: `dag_prompts/2026-07-31-gpu-window/`. (Claude)
- 2026-07-31 — **G-02 TIERED-SURFACE CAMPAIGN TERMINAL / APNCC BANKED /
  NO FURTHER WAVES.** Frozen `TERMINAL_CAMPAIGN_REPORT.md`
  (`9b0a7819…1a961`): APNCC-DIRECT remains local `0.881522069960` / public
  `0.881`; run_v6 requires DELETE `0.85/0.020394` and occupied SWAP
  `0.90/0.067910`; DINOv2 and assignment evidence converge at DELETE AUC
  `0.670412/0.670235`, supporting an empirical ceiling near `0.670` for the
  current pointwise action label. Division decomposes to GT ceiling
  `+0.092067842`, tier (a) `1/38` / `+0.002406490`, tier (b) `37/38` /
  `+0.089661334` representation-blocked, tier (c) EMPTY. CELLECT is permanent
  PARK-C / UNTESTED after six infrastructure failures. Final parent-node
  census is LIVE/DEAD/PARK-A/PARK-B/PARK-C `0/1/14/18/12`. Reopen only on a
  new action-labeling formulation or independent exact-task AUC `>=0.85`
  evidence—not another appearance encoder. No experiment, graph, GPU,
  submission, RunPod, plan amendment, or external action occurred.
- 2026-07-31 — **PEFT PRELAUNCH INVALID / NO LAUNCH; DIVISION SURFACE
  PARK-C.** The exact DELETE population has no grouped-fold-0 actions, so it
  cannot satisfy the frozen three-outer-fold result and missing-fold
  fail-closed clauses. No projection, Kaggle mutation, GPU debit, or
  submission occurred; quota re-authenticated at `5.02/24.98 h`
  used/remaining. Division is recorded FALSIFIED-BY-REALIZATION after valid
  A/B/C each realized 0/38 TP; the `+0.092067841519` GT oracle remains a
  ceiling. (Codex root)
- 2026-07-31 — **C2 ASSIGNMENT-EVIDENCE FULL-POPULATION FAIL BOTH / PARK-C /
  BARRIER HELD.** Frozen C0 MAP/min-marginal/K-best features covered all
  1,391,458 actions without sampling or oracle-field access. DELETE A1 mean AUC
  `0.670235` (mean lifts `+0.022964` B0 / `+0.021609` A0) misses `0.85` and
  every-fold lift/retention conjunction; SWAP A1 mean `0.664616` misses `0.90`
  and regresses B0 in 5/5 folds. Both units are `FAIL_PARK_C`; validation
  10/10; no graph or downstream gate opened. (Codex root)
- 2026-07-31 — **FULL-38 INDEPENDENT REVIEW PASS / D FAIL-CLOSED /
  BARRIER HELD.** An independent recount reproduced binding A/B/C at
  `0/23/38`, `0/48/38`, and `0/22/38`, all Jaccard `0`; no arm passes.
  Clean D's empty target-free intersection is `INVALID_NONBINDING` because its
  preregistration was absent from both pre-outcome freezes. The PEFT-only
  preregistration now pins the actual grouped three-fold artifact exactly
  (`ddb313cb…8769`, declared/actual count `3`) and remains unlaunched. APNCC
  transport and correct public-T4 archive identity independently pass; no
  submission, fallback, downstream unlock, or other hard-stop change occurred.
  Review `9ec635c…d1fe5`. (Independent reviewer + Codex root)
- 2026-07-31 — **DIVISION CLEAN B/D TERMINAL / NO PASS / BARRIER HELD.**
  A fresh forbidden-read worker sealed 48 target-free persistence proposals
  across all 57 graphs with zero duplicates and zero swapout growth; the
  target-side full-38 evaluation returned B `0/48/38`, Jaccard `0`, despite a
  tiny `+0.000029258` combined graph delta. Clean 3-of-3 D content abstained
  at `0/0/38`, but a postrun audit fail-closed it as nonbinding because its
  clean preregistration was absent from the pre-outcome freeze; no retroactive
  freeze was made. A/C remain `0/23/38` and `0/22/38`. No valid arm passes,
  no downstream unlock exists, and PEFT remains preregistered but unlaunched.
  (Codex root)
- 2026-07-31 — **APNCC PUBLIC-T4 ARCHIVE IDENTITY PASS / TRANSPORT VALID.**
  The v2 submission and all four fragments are byte-identical to authenticated
  `results-bhap.zip` (`031aa5f6…7219`), the T4 artifact that scored public
  `0.881`; non-timing run fields and proposal/cap counts match exactly. The
  apparent 1,096-row deficit came from the wrong pre-APNCC baseline reference,
  not threshold drift, and is classified `BENIGN_FP`. No CPU/T4 fallback,
  relaunch, or submission. (Codex root)
- 2026-07-31 — **SWAP GUARD CALIBRATION ADOPTED / CELLECT NO-SPEND.** Adopted
  `256 MiB`, 0.25-s monitoring, 15-s sustained abort, and 384-MiB immediate
  abort. The `+3.5 MiB` stops are retired as noise-confounded; `+488.6875 MiB`
  remains a genuine immediate abort, so CELLECT stays permanently PARK-C with
  no further spend or tier/scientific action. (Human owner + Codex root)
- 2026-07-31 — **DIVISION B/D FIREWALL CORRECTION / CLEAN RERUN ACTIVE.**
  Root discovered after scoring that its fresh B design followed prior access
  to the target schema. The authentic `0/53/38` arithmetic and derived empty D
  consensus are quarantined as diagnostic-only, not terminal evidence. A and C
  remain valid. A fresh no-history worker is now preregistering B under an
  explicit forbidden-read firewall; the binding wave report is pending. No
  external or downstream action opened. (Codex root)
- 2026-07-31 — **DIVISION FULL-38 WAVE TERMINAL / NO PASS / BARRIER HELD.**
  Target-isolated A/B/C/D completed on all 38 repair events and all 57 graphs:
  `0/23/38`, `0/53/38`, `0/22/38`, and `0/0/38`, with Jaccard `0` and zero
  duplicate survivors throughout. C's first real appearance screen used only a
  476,307-byte bounded summary cache and all local arms had zero swap growth.
  No arm passes; PEFT §6.2 remains preregistered but unlaunched, with its
  unchanged 4.8-hour projection reconciled to the current 24.98-hour balance.
  No downstream or external action opened. (Codex root)
- 2026-07-31 — **APNCC VALID CSV TRANSPORT PASS / CV57 BYTE-PARITY FAIL.**
  The immutable wrapper ERROR is retained: all four T4 fragments differ from
  the local bank. A separate raw-test-derived validator nevertheless proves
  that the downloaded 263,282-row four-dataset `submission.csv` is structurally
  valid with zero graph/schema errors. This satisfies the end-to-end artifact
  transport requirement but permits no CV57 scientific claim, scored
  submission, or retry. The APNCC debit remains 0.16 GPU-h. (Codex root)
- 2026-07-31 — **CELLECT FINAL CLOSEOUT TERMINAL / PERMANENT PARK-C /
  SCIENTIFIC UNTESTED.** The sole authorized version completed the required
  25,055-file / 87.798-GB full SHA inventory before resolution, then found zero
  copies of the CELLECT safe loader (and the inventory also contains zero
  checkpoint matches). The mounted payload was the DINOv2 carrier. No model,
  feature, A1, A0, or per-class number exists. Authoritative debit is 0.33
  GPU-h; balance is 5.02 h used / 24.98 h remaining. The one-run
  infrastructure-failure rule permanently forbids CELLECT retry/further spend
  and unlocks no graph contract. (Codex root)
- 2026-07-31 — **CELLECT FINAL CLOSEOUT V1 LAUNCHED / ONE VERSION / RUNNING.**
  The fresh self-contained kernel passed local expanded-mount discovery,
  canonical-wheel validation, inline shared guard, zero-existing-version,
  zero-in-flight, and authoritative quota checks, then created exactly remote
  v1 under the 5 h timeout. A post-push list lag interrupted the first local
  supervisor; direct status recovery attached to the same running v1 with no
  second push. Authorization is consumed; prelaunch quota is 4.69 h used /
  25.31 h remaining. No outcome or postrun debit yet. (Codex root)
- 2026-07-31 — **DAG-4 DIVISION VALUE RECONCILIATION COMPLETE / NO AMENDMENT.**
  The 38-bundle frozen-N1 replay and information decomposition were executed
  and independently validated. GT ceiling is
  `+0.092067841519`; tier (a) is 1 event / `+0.002406489982`; tier (b) is 37
  events and now explicitly UNAVAILABLE after representation closure; tier
  (c) is empty. The exact 25-TP sensitivity at 0/5/10 FP is
  `+0.059523810/+0.053191489/+0.048076923`. Tier (a) misses the `+0.005`
  materiality line, so no amendment proposal or tier move. (Codex root)
- 2026-07-31 — **DIVISION-ONLY FULL-38 SCOPE CORRECTION / PEFT §6.2 OPEN,
  NOT LAUNCHED.** Human decision §6.5 makes the combined 9+29-event population
  binding for division while leaving association 44b6-first unchanged; old Arm
  A 0/9 is now INCONCLUSIVE-UNDERPOWERED. Fresh target-isolated A/B/C/D
  preregistrations were sealed. The plan's frozen-failure precondition is
  satisfied, so a frozen-backbone `<1%` adapter node was preregistered at
  `4.8` T4-h with nested-OOF graph-delta selection and no full fine-tune. No
  outcome, GPU launch, submission, RunPod, full-pool cache, or downstream
  unlock occurred. (Human owner + Codex root)
- 2026-07-31 — **APNCC V2 INDEPENDENT REVIEW PASS / BARRIER HELD.** Read-only
  review confirms hardened Step 0, the exact all-four-fragment parity failure,
  no leaderboard submission/retry, authoritative `0.16 GPU-h` debit, and the
  proposal-only CELLECT `256 MiB` continuous/sustained guard. The runner's
  local `mode=submit` only wrote the unattested local CSV; it did not submit to
  Kaggle. No scientific, tier, or barrier change. (Independent review + Codex root)
- 2026-07-31 — **APNCC V2 TERMINAL TRANSPORT FAIL / NO RETRY.** The sole
  remote v2 run passed full Step 0 in `2.715378342 s` (62 wheels, zero
  competition-path visits) and ran the banked runner, but the first 44b6 CSV
  fragment failed exact parity (`50,545` vs `50,711` rows). Quota reconciled
  `4.53→4.69 h` used / `25.47→25.31 h` remaining: **0.16 GPU-h debit**. No
  submission, graph, CV57 result, tier action, or retry; barrier remains held.
  (Codex root)
- 2026-07-31 — **APNCC V2 CPU STEP-0 SMOKE FAIL-CLOSED / REMOTE RUNNING
  RECONCILED.** The authentic local support copy has 60/62 declared wheels,
  missing exact `llvmlite-0.47.0…manylinux_2_17_x86_64.whl` and
  `polars_runtime_32-1.42.0…manylinux2014_x86_64.whl`; the generated kernel
  stopped in `0.010967 s` with `heavy_work_started=false`. A read-only owner
  status query independently confirmed remote `bhap-transport-smoke-v2/1` is
  RUNNING, so the barrier is now held for terminal transport/output/quota
  evidence—not authorization. No submission, graph, CV57 conclusion, or tier
  action. (Codex root)
- 2026-07-31 — **CELLECT SWAP-GUARD CALIBRATION / PROPOSAL ONLY.** Three
  fresh 60-second idle windows each showed 0.0 MiB system swapout growth.
  Proposed future-only guard is `max(256 MiB, 5× observed idle) = 256 MiB`,
  0.25-s monitoring, 15-s sustained-growth hard abort, and immediate abort at
  1.5× (`384 MiB`). The historical +3.5 MiB stop was below the banked
  10.06–54.87 MiB idle drift and is noise-confounded; +488.6875 MiB remains a
  real high-pressure signal. CELLECT stays PARK-C with no rerun, tier move,
  spend, or scientific verdict. (Codex root)
- 2026-07-31 — **APNCC-DIRECT TRANSPORT SMOKE V2 LAUNCHED / NO SUBMISSION.**
  Root rebuilt the requested `779aa8d3…606b` carrier with full declared-input
  byte-SHA verification, one support dataset, offline install, serial launch
  fence, and zero Step-0 competition-path visits. Step-0 elapsed time is now a
  60-second warning, while missing/mismatched input, non-positive request
  count, manifest drift, and a competition-path visit remain hard failures.
  Kaggle accepted fresh v1 of the v2 slug after a 4.53 h used / 25.47 h
  remaining prelaunch reconciliation. The run is active; no output acceptance
  or competition submission has occurred. (Codex root)
- 2026-07-31 — **APNCC V2 LOCAL ATTESTATION REPAIRED / BARRIER STILL HELD.**
  Independent review found stale internal BUILD_RECEIPT/INLINE_LAUNCH_GUARD
  hashes in the local-readiness record. Both were refreshed from a clean local
  rebuild/guard run; the barrier validator now checks the nested pins and
  passes, and focused v2 tests explicitly isolate v2 imports from sibling v1
  modules. No Kaggle request, GPU action, budget debit, submission, graph, or
  tier action occurred. (Codex root)
- 2026-07-31 — **ACTIVE SUPPLIER-GATE RECORD UPDATED / NO RESURRECTION.**
  AE-P6 and R8 remain PARK-B. Their future distinct-supplier records now cite
  the approved delete `0.85/0.020` and occupied-swap `0.90/0.068` all-fold
  gate, while the failed authentic Trackastra result remains terminal and is
  not re-evaluated. No graph, tier, DEAD, or external action. (Codex root)
- 2026-07-31 — **APPROVED MEASURED-GATE RECALIBRATION / NODETOPO CAUSAL
  CORRECTION / HOLD.** Human-approved future supplier floors are delete AUC
  `0.85` / retention `0.020` and occupied-swap AUC `0.90` / retention `0.068`;
  historic supplier verdicts stay immutable (DINOv2 delete retention `0.001307`
  is >15x short), no tiers move. `run_v6` establishes the 50-action
  `restore_topology` population had 0 beneficial actions and oracle `0.000`, so
  `NODETOPO-O0`'s old tiny negative graph result is generator—not selector—
  failure; it stays PARK-A with a target-blind 61-target recovery-generator
  preregistration and no selector. Measured surface order is division
  `+0.092067842`, delete `+0.081897667`, then occupied-swap `+0.018571397`.
  APNCC v2 local build/guard/tests PASS; remote GPU launch was not started
  pending an unambiguous exception to the prior CPU-only/no-Kaggle constraint.
  Receipt: `selector_requirement_44b6/run_v6/RECALIBRATION_BARRIER_20260731.json`.
  (Codex root)
- 2026-07-31 — **DIVISION REALIZER WAVE TERMINAL / NO PASS / HOLD 44B6.**
  Four separately preregistered CPU-first arms stopped at 44b6 with zero
  duplicate survivors. A geometry/motion validly failed (`0/1/11`, Jaccard
  `0`, combined `-0.000141523`). B persistence is fail-closed and
  non-promotable because of pre-prereg forbidden operator exposure; its sealed
  target-free output remains diagnostic only (`0/49/11`, combined
  `-0.009455519`). C appearance stopped before prediction because no adequate
  frozen intensity cache exists. D's frozen 3-of-3 consensus abstained
  everywhere and is invalid due to B/C component status; its raw zero-delta
  metrics are diagnostic only. Independent review verdict:
  `FAIL_CLOSED_PARTIAL_WAVE_INTEGRITY / HOLD_44B6`. No 6bba, CV57, graph,
  submission, GPU, RunPod, full-pool cache, DEAD, or further plan action.
  Report `tiered_surface/branches/sa_division_realizer_wave/WAVE_TERMINAL_REPORT.md`.
  (Codex root)
- 2026-07-31 — **CPU BARRIER MERGE COMPLETE / NO TIER MOVE.** Imported the
  byte-pinned authentic reverse terminal evidence (both frozen classes fail);
  reran AE-P5 with complete N1 ownership (45,961 actions, 361 decisive batch
  compiles, zero emitted edits); validated AE-P7's bounded raw-image smoke;
  and confirmed DAG-2 local `run_v6` is a valid complete requirement surface
  behind human review, distinct from its stale remote transport invalidity.
  AE-P5/P6/P7/R8 tiers remain PARK-A/PARK-B/PARK-A/PARK-B. Receipt:
  `tiered_surface/branches/barrier_merge_20260731/BARRIER_MERGE.json`.
- 2026-07-31 — **APNCC-DIRECT TRANSPORT SMOKE FAIL / NO SUBMISSION.** Root
  rebuilt the requested banked `779aa8d3…606b` package with one consolidated
  support dataset, canonical PEP-427 wheel names and byte-SHA checks, offline
  `--no-index --find-links`, exact test-store cardinality, an inline pre-heavy
  Step 0, and a shared serialized launch fence. Kaggle accepted exactly v1 of
  `onebitbrain/bhap-transport-smoke-v1`; Kaggle accepted a push reported as v1
  and no second push is recorded. It failed closed at Step 0 in `5.061967561 s`
  (hard limit `5.0 s`),
  with `heavy_work_started=false`. The
  banked runner never started and neither a submission CSV nor CV57-parity
  fragments exists; no competition submission or retry occurred. Authoritative
  quota before/after was 4.53 h used / 25.47 h remaining (0.0 h debit).
  Failing layer: complete input-verification timing budget. Receipt:
  `apncc_transport_smoke_v1/TERMINAL_DIAGNOSIS.json`. (Codex root)
- 2026-07-31 — **LANE 1 DIVISION-EMISSION PACKAGE READY / NO SUBMISSION.**
  Built hash-pinned division-capable post-graph kernel
  `4673a59079b7b9e72bbe261f6a62e2dcf004854e0d0657838bb56ddd144facf7`
  (manifest `tiered_surface/branches/sa_division/LANE1_DIVISION_PACKAGE_MANIFEST.json`).
  Reused byte-pinned frozen D1 evidence (14/14 official fixtures, 51/51
  invariant counterfactuals, zero duplicate survivors) without scorer rerun;
  D1 persistence/one-to-one suppression feeds atomic frozen-N1 `division_fork`.
  Banked APNCC 44b6's 36-fragment dry run emitted divisions-on empty
  `kaggle_kernel/g02_division_lane1/dry_run/submission.csv`
  (SHA `26ee9c44…edbd9`, 2,415,478 rows), preserving canonical scorer-input
  graph `e3ada186…f8c5` and score change `0.0`. No candidate fitting/new
  science, Kaggle push, or submission occurred. A non-empty receipt must
  attest 44b6 combined `>0`, division Jaccard `>=0.30`, nonnegative
  adjusted/raw/node deltas, and zero duplicates. (Codex)
- 2026-07-31 — **OWNER PLAN AMENDMENT APPROVED / DIVISION FIT-ONLY
  DEPENDENCY UNLOCKED.** Amended
  `campaign_narrowing_and_hypothesis_dag_v1.md` §6.4 using the frozen-N1
  `38/0/4`, Jaccard `0.904761904762`, combined `+0.092067841519` ceiling,
  edge-only `~0.896` cap, six near-zero association mechanisms, and four
  operating-point representation failures. Only the association-continuity
  prerequisite is removed. Four CPU-first arms are authorized 44b6-first with
  GT targets excluded from features, frozen N1/scorer, zero duplicates, and
  the unchanged positive-combined/`>=0.30`/nonnegative-edge PASS line. 6bba,
  CV57, submission, GPU, RunPod, full-pool cache, and DEAD action remain
  closed. (Human owner + Codex root)
- 2026-07-31 — **CELLECT R7 PARK-C BLOCKED-ON-INFRASTRUCTURE / NO FURTHER
  SPEND.** The owner combined the consumed Kaggle transport stop rule with
  five zero-preflight/positive-shard-start local swap-guard failures. No
  feature, checkpoint, B0/A1/A0, or scientific verdict exists. Recorded that
  a 10,197,221-byte checkpoint does not predict volumetric-inference memory,
  where activations/tiles/workspaces/allocators dominate. Census moves
  PARK-B→PARK-C administratively; no DEAD, GPU, Kaggle, RunPod, submission, or
  graph action. (Human owner + Codex root)
- 2026-07-30 — **CELLECT local resource block authenticated; no scientific
  verdict.** Reproduced the DINOv2 two-stage, 45-second-preflight,
  one-shard-process, atomic-checkpoint, strict zero-swap harness for the pinned
  CELLECT checkpoint. Eight focused tests pass. Five zero-growth preflights
  were followed by five shard-0 guard trips (`+3.5` to `+488.6875 MiB`);
  zero outputs exist and B0/A1/A0 never ran. No Kaggle, GPU debit, graph,
  full-pool cache, tier move, or DEAD action; resume is restricted to a stable
  zero-swap host window without guard weakening. (Codex)
- 2026-07-30 — **Operating-point methodology banked; existing gates
  unchanged.** Recorded that ROC-AUC is non-diagnostic at roughly `10^-4`
  prevalence when the CG0 FP budget binds, and proposed precision/recall at
  the frozen budget as the lead statistic for future supplier gates, subject
  to human approval. This is methodology only, not a gate amendment. (Codex)
- 2026-07-30 — **Division amendment dossier emitted; fitting remains
  locked.** Authenticated 38 per-event repair bundles and freshly reverified
  the banked zero-swap frozen-N1 oracle receipt (`38/0/4`,
  `+0.092067841519` combined), documented target-gap/FP arithmetic, B1
  staleness, duplicate risks, and the already preregistered 44b6-first kill
  line. No fit, plan amendment, graph, 6bba, CV57, or external action occurred;
  owner decision is required. (Codex)
- 2026-07-30 — **Reconciliation: recall pool-absence vs tensor reachability; stop-loss
  doctrine; Kaggle reliability runbook.** Traced `recall/N0`'s
  `+0.052004594313` pool-absent ceiling against the banked fit117 tensor-reachability
  result (99.04% coverage) via `scripts/analyze_action_class_oracle.py` and
  `achievable_headroom_v1.json`: the two are not contradictory — pool-absence is gated
  by the deployed action vocabulary (both-ends-free `add`; node-disjoint single-hop
  `swap`), not by candidate-tensor supply, mirroring the campaign's own prior
  `SWAP-O0` legacy-CG0-to-`run_v6` collapse (`+0.075828 → +0.018571397`, `24.49%`
  realized). Verdict extended to the **mechanism class**: no candidate-generation
  supplier can clear this wall; do not fund a second generator without a new compiler
  action-kind (compound/chained conflict-swap), which is a distinct, separately-gated
  proposal against the frozen `GO`-status N1 compiler. Receipt:
  `dag_prompts/2026-07-30-recall-and-division/recall/runs/N0B_RECONCILIATION.md`.
  Generalized this into a standing pre-compute check,
  `reports/g02/screening/MARGINAL_VALUE_GATE.md`: before any repeat-compute request,
  state the *realistic* (precedent-adjusted) ceiling, not the raw/optimistic one, and
  STOP at the mechanism-class level when the limiting factor is structural.
  Separately, audited why recent Kaggle launches keep failing
  infrastructure-only: the existing 16-family guard
  (`tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`) is real and its 14+2
  regression fixtures pass, but it has **zero external call sites** in the repo (no
  git hook, no wrapper, confirmed by direct grep) and its `dataset_ownership` map was
  stale — `onebitbrain/g02-cellect-r7-separation-v1` and
  `onebitbrain/g02-dinov2-r5-gpu-assets-v1` (the exact datasets behind the most recent
  GPU bake-off pushes) had zero coverage. Added `KFG-17`/`KFG-18` to
  `KAGGLE_FAILURE_REGISTRY.json`, the two missing `dataset_ownership` entries, a new
  `check_registry_freshness.py` (real run: `FAIL_REGISTRY_STALE` before the fix,
  `PASS` after), and `tiered_surface/kaggle_prelaunch/RELIABILITY_RUNBOOK.md` mandating
  a logged guard receipt before every future push. Re-ran the existing 14-fixture
  suite after the registry edit: 14/14 still pass. While this was in progress, CELLECT
  v7 failed with the same shape of problem live (`LIVE_STATE`'s own "DINOv2 FAIL BOTH /
  CELLECT TRANSPORT STOP" entry, above) — flagged as a candidate `KFG-19` for root to
  formalize with exact log evidence in its own terminal report. No GPU, Kaggle asset,
  push, submission, tier move, or `LEDGER.md` write occurred; nothing committed pending
  owner confirmation on commit/cross-pollination scope. (Claude)
- 2026-07-30 — Recall supply N2 **DEAD / LANE TERMINAL**: frozen fit-only 12-video
  full-frame slice (290 pool-absent targets; 502,336 frozen candidates) yielded
  `9/290 = 3.103448%` recovery, below the preregistered `<20%` kill line, despite
  `1,173` new offers / `1.002335090457×` inflation. All required fixtures failed
  closed and the byte repeat matched; the 45.950 s preflight and 315.815 s run
  had 0 MiB swap growth. `NDR-FBC-local-translation-v1` is no-retry DEAD; N3,
  Kaggle, GPU, asset, push, submission, and graph score remained closed. Receipt
  `dag_prompts/2026-07-30-recall-and-division/recall/runs/N2_SUPPLY_FALSIFIER.md`.
  (Codex root)
- 2026-07-30 — Recall supply N1 **GO → N2**: distinct `NDR-FBC-ONE-OFFER`
  raw-volume forward/backward-consistent registration is label-free, non-DEAD and
  non-selector, with three primary sources and a named crowding rationale. Its
  frozen label-free opportunity count is `105,992` fit117 offers (`49,553` 44b6,
  `56,439` 6bba), a max `1.013459860492×` pool multiplier; its 45.489 s
  preflight and 29.096 s count had 0 MiB swap growth. N2 now runs once on a
  frozen fit-only slice; no GPU/Kaggle/push/submission/ledger action. Receipt
  `dag_prompts/2026-07-30-recall-and-division/recall/runs/N1_MECHANISM_AND_CROWDING.md`.
  (Codex root)
- 2026-07-30 — Division frozen-external checkpoint N0 **PARK / NO PILOT**:
  `0/>=1` releases passed the five preregistered provenance and 16-GiB conditions;
  each verified category (source revision, weight byte/SHA, weight-license SHA,
  corpus exclusion, and source-published 16-GiB inference configuration) is `0`.
  The owner cmux retrieval pane was unavailable and was not retried; this cannot
  support the universal DEAD claim. N1–N3 are closed, division fitting/B1 remain
  locked (`0` qualifying arms), and no campaign/remote/Kaggle/GPU action occurred.
  Receipt `dag_prompts/2026-07-30-recall-and-division/division_ckpt/runs/N0_CHECKPOINT_PROVENANCE.md`.
  (Codex root)
- 2026-07-30 — Recall supply N0 **GO → N1**: frozen full 44b6 reproduces
  TP/FP/FN `10,341/1,064/673`; present/absent FNs are `47/626`, and absent's
  zero-FP adjusted/raw/node deltas are `+0.052004594313/+0.051829773141/0`, above
  its fixed `>=+0.030` GO bar. 10–12 µm absent overlap is `18`; fresh preflight
  `45.738 s` and audit `52.741 s` both had 0 MiB swap growth. No GPU/Kaggle/push/
  submission/ledger action; N1 proceeds under the same stop rules. Receipt
  `dag_prompts/2026-07-30-recall-and-division/recall/runs/N0_POOL_RECALL_CEILING.md`.
  (Codex root)
- 2026-07-30 — CELLECT v7 **TRANSPORT FAILURE / STOP RULE TRIGGERED / PARK-B
  BLOCKED-ON-INFRASTRUCTURE / NO SEVENTH RUN**: the exclusive prelaunch audit
  authenticated 25.47 h remaining and zero target versions/in-flight G02 runs;
  one push created exactly version 1. A transient non-JSON immediate list
  response interrupted only the post-push receipt, so supervision recovered
  the existing v1 without a second push and proved delta `+1`. Kernel step 0
  failed at `2.167 s` because
  `/kaggle/input/datasets/onebitbrain/g02-cellect-r7-published-assets-v1/action_request_manifest.json`
  was absent. No model, feature, B0/A1/A0, or scientific verdict ran.
  Authoritative pre/post quota stayed `4.53/25.47 h`; debit `0.0 h`. GPU
  bake-off halted; human decision required on further representation spend.
  No graph, 6bba, CV57, submission, RunPod, tier move, DEAD, or plan
  amendment. Terminal `c0beda9b…5f706f7`. (Codex root)
- 2026-07-30 — CELLECT v7 **SOLE SERIAL AUTHORIZATION OPENED / NOT YET
  LAUNCHED** after DINOv2's local verdict: active authorization
  `CELLECT-R7-SEPARATION-V7-ONE-SHOT` permits exactly one fresh-version GPU
  run under the exclusive lock, authenticated projection `<5 h` after 64 tile
  jobs, hard 5 h, and pre/post authoritative quota reconciliation from
  25.47 h remaining. Transport failure consumes the authorization, halts the
  GPU bake-off, parks CELLECT PARK-B blocked-on-infrastructure, and permits no
  seventh run. No push or debit yet. (Codex root)
- 2026-07-30 — DINOv2 R5 local separation **SCIENTIFIC COMPLETE / FAIL BOTH /
  PARK-C**: 77 isolated MPS children materialized exactly the frozen 1,391,458
  endpoint requests with atomic checkpoints, no full-pool cache, and 0 MiB
  swap growth. A fresh 45.07 s evaluation preflight passed; the unchanged
  sealed B0→A1→A0 evaluator then returned `FAIL_PARK_C` for delete
  (A1 mean/p05 AUC `0.670412/0.611373`, mean/p05 retention
  `0.001307/0`) and occupied-swap (`0.682473/0.628505`, `0/0`).
  Verifier PASS; result `c58b1a58…0d41f2`. No Kaggle quota, graph,
  6bba, CV57, submission, RunPod, tier move, DEAD, or plan amendment.
  CELLECT's sole stop-rule-bound v7 run remains unlaunched. (Codex root)
- 2026-07-30 — Attempt-6 packaging/serialization repair **LOCAL + REMOTE
  PREFLIGHT PASS / NO PUSH**: both DINOv2 and CELLECT now preserve canonical
  PEP-427 wheel names, verify wheel bytes against their manifests before
  no-index/find-links installation, and complete declared-input-only step 0 in
  `0.04–0.07 s` without walking the competition dataset. CELLECT accepts
  Kaggle's expanded uncompressed-tar checkpoint by member SHA and fails closed
  on a corrupted member. The exclusive launch-lock audit found no running G02
  kernel and zero versions of fresh slug
  `g02-cellect-r7-separation-v7-one-shot`. Direct quota is **4.53 h used /
  25.47 h remaining**; the local file is now derived and records the **1.65 h**
  correction. The 77-process resumable local DINO route is staged and tested;
  no feature, B0/A1/A0, kernel push, GPU debit, tier move, PARK-C, or graph
  unlock occurred. (Codex root)
- 2026-07-30 — **OWNER DECISIONS: recall lane reopened; division frozen-checkpoint
  pilot approved; MAIN AE-P7 and NODETOPO-neighbor pruned by the adversarial
  review's ranking.** Rationale recorded: banked `0.881522` local needs
  `+0.044–0.061` for public `0.910`, the in-pool re-ranking oracle is only
  `~0.896`, and free-add/node-add oracles on the frozen population are exactly
  `0` — so selection cannot cover the gap and candidate **supply** is the untested
  channel. `H2` was orphaned by the `H1` 44b6 kill and never replaced. Two DAGs
  written at `dag_prompts/2026-07-30-recall-and-division/` with pasteable dispatch
  in its `TRIGGER.md`; recall `N0` is a free CPU measurement that partitions the
  44b6 held edge FNs (`673`) into pool-present and pool-absent and computes each
  partition's zero-FP oracle ceiling, with GO at `>= +0.030`. Division continuation
  gate preregistered from `p0_parity.md`'s `0.1 * division_jaccard` and the 42
  CV57 divisions: `>=4` zero-FP TP. Rejected readings of the `+10%` rule are
  recorded in the node so the threshold cannot be retrofitted. No node has run;
  no asset, kernel, GPU debit, amendment, tier move, or LEDGER write. (Claude)
- 2026-07-30 — MCC contour-return witness **OWNER-APPROVED SOURCE AUDIT /
  TERMINAL BEFORE COMPUTE**: CMUX authenticated source `1219b669…9041`,
  Apache-2.0 license `cfc7749b…23d30`, and the public Zenodo-7754589 CC-BY-4.0
  metadata/file manifest. The source provides no checkpoint and mandates
  ordered-contour preprocessing, TFRecord conversion, and training in its
  Python-3.6.9/TensorFlow-2.4.3/CUDA-11 environment; it cannot yield a
  deterministic output under the owner-approved one-T4 <=2-GPU-hour ceiling.
  No data download, asset, kernel, compute, campaign-data access, feature, or
  outcome occurred. The exact MCC source/record/slice/budget is terminal with
  no retry, and Reverse Trackastra remains closed. Receipt
  `dag_prompts/2026-07-30-continuation/reverse/runs/N3_SOURCE_AUDIT_TERMINAL.md`.
  (Codex root)
- 2026-07-30 — DPRC-O0 DINOv2-register **OWNER AUTHORIZATION RECORDED /
  PROVENANCE PREFLIGHT ONLY**: the owner authorized the one separately bounded
  fresh-supplier route. CMUX authenticated the official source
  `7764ea0f…25fc8`, Apache-2.0 license `600cc67c…935b7`, and distinct
  official ViT-S/14-register weight `f4331770…641fb` (88,291,785 B). The
  two timing-only calibration roots and array geometry were read from the
  existing competition source; no archive/data upload or copy occurred. No
  DPRC Kaggle asset/kernel/run, model load, decode, feature, scientific
  outcome, or GPU debit exists. A fresh materialization manifest,
  foreground/channel/crop contract, and five-second local preflight are still
  launch blockers; any failure ends DPRC-O0 without retry. Receipt
  `dag_prompts/2026-07-30-continuation/dinov2/runs/N3_SOURCE_ASSET_RECEIPT.md`.
  (Codex root)
- 2026-07-30 — DINOv2 GPU launch race **RECONCILED / SERIAL AUTHORIZATION
  VIOLATION / NO RETRY**: a concurrent actor created authorized v1 at
  17:55:29Z before the root launch call completed; the root's stale absence
  check then created byte-identical v2 at remote last-run time
  17:57:50.703Z. Two remote runs were triggered under a one-run authorization.
  Exact latest-v2 evidence is terminal ERROR after step-0 PASS and before
  model/features/B0/A1/A0; the shared invalid wheel basename also prevents v1
  from reaching scientific work. Private version-addressed APIs return 404/403,
  so no exact individual v1 runtime is asserted. Owner quota moved
  3.79→4.53 h used; the full **0.74 GPU-h** delta is debited, yielding repo
  3.6435/30.0 h and direct owner capacity **25.47 h**. All four supplier×class
  cells remain NOT EVALUATED, both suppliers PARK-B, and zero PARK-C/graph
  unlock. Reconciliation
  `sa_dinov2/kaggle_gpu_bakeoff_v1/LAUNCH_RACE_RECONCILIATION.json`.
  (Codex root)
- 2026-07-30 — AE-P5 **STRUCTURALLY INERT ON FROZEN 44B6 / NO GRAPH SPEND**:
  the frozen C0/C2 rule is implemented under human implementation authorization
  and rank-1 is emitted. Over 36 shards / 45,961 occupied-conflict actions /
  44,161 components, **0 components apply and 44,161 fall back to APNCC**:
  44,144 have fewer than the eight feasible assignments C2 requires, and the 17
  that reach eight all fail the 7/8-introduce / 1/8-remove conjunction. 96.4% of
  components are singletons admitting exactly two assignments; feasible counts on
  a conflict path are Fibonacci-like `2,3,5,8,13`, so eight needs a component of
  at least four chained actions. The result is robust to the contract's ambiguous
  ownership definition — under the looser target-incoming-only reading **no**
  component anywhere reaches eight. Rank-1 therefore equals APNCC byte-semantically
  and 44b6 adjusted/raw/node are exactly `0/0/0` by construction, below `+0.005`,
  with **no graph run needed or performed**. This does not falsify the broader
  C2/appearance family; **no tier move is proposed** and a newly named realizer
  with larger conflict components, or a `k` matched to observed component sizes,
  stays eligible. Result `4fc8eb98…c2be`; tests 20/20. Report
  `sa_contracts/ae_p5_rank1/TERMINAL_REPORT.md`. (Claude)
- 2026-07-30 — AE-P7 fit105 smoke **BLOCKED ON MATERIALIZATION / NOT STARTED**:
  authorized, but the raw fit105 44b6 Zarrs it must read are unextracted inside
  the 87.4 GB archive (45.5 GB / 10,710 files per
  `raw_fit105_directory_manifest.json`); only 3 of 36 videos exist extracted under
  `data/derived/t03_fold0_fit10_images_v1`. A 45 GB extraction was not started on a
  host that had just thrown a transient `ENOSPC`. No preflight, guard, crop, read,
  or outcome. (Claude)
- 2026-07-30 — Separation gate **MEASURED REQUIREMENTS APPLIED / VERDICT LAYER
  PRE-REGISTERED**: under human authority the run_v6 `+0.005` mean requirements
  become the binding absolute floors of the per-class separation gate. Amendment
  `3526b54b…951d` was frozen while **zero** supplier×class cells were evaluated
  (verified: no `cellect-separation-result.v1` artifact anywhere in the tree);
  the subsequent DINO GPU and CPU terminal-INVALID receipts leave all four cells
  NOT EVALUATED, so the bar remains strictly pre-outcome. AUC tightens
  `0.65→0.85` (delete) and `0.65→0.90` (occupied swap). **Retention relaxes**
  `0.15→0.020394` and `0.15→0.067910`: both retentions were verified to be the
  same CG0-per-video-budget quantity, and the measured requirement sits below the
  preregistered guess. B0/A0 lift conditions are unchanged. A supplier-agnostic
  verdict layer over the **unmodified** evaluator emits per supplier×class
  verdicts against the binding bar while reporting the legacy and robust-p05 bars
  and flagging divergence; 20/20 fail-closed tests pass, and a CLI smoke confirms
  an AUC-0.70 delete arm that would have passed the legacy bar now FAILs.
  Ceiling propagation corrected AE-P2/AE-P5/AE-P7 **and SWAP-O0** in
  `tier_census.json` from legacy CG0 `+0.075828` to compiler-valid run_v6
  `+0.018571397`; SWAP-O0's correction had existed only in this doc's prose and
  never reached the machine state, while AE-P1 was already migrated. Census is
  now `ff551d8e…bc22` with no tier changed. Artifacts under
  `sa_repr/phase2_separation_gate/measured_gate_v4/`. No tier, DEAD, graph, 6bba,
  CV57, GPU, Kaggle launch, budget debit, RunPod, or submission action. (Claude)
- 2026-07-30 — Two-supplier accounting **RECONCILED / BARRIER CLOSED**:
  owner-authenticated status confirms the earlier DINO CPU-promoted run is
  ERROR. Its step 0 passed in 0.053483 s before a 38.662 h projection failed
  closed; no result manifest or outcome, 0.1163353 CPU-h / 0 GPU-h. No external
  session remains active. This accounting snapshot was later superseded by the
  launch-race reconciliation at the top of this log; its 0.3989 h/25.8111 h
  GPU figures are not current. Four supplier×class cells remain NOT EVALUATED;
  zero PARK-C or graph unlock. (Codex root)
- 2026-07-30 — Two-supplier separation cross-read v3 **TERMINAL / GRAPH
  CLOSED**: DINO GPU and the prior CELLECT launch are infrastructure-INVALID;
  CELLECT's zero-manifest expanded-mount repair is local-only PASS. No complete
  class manifest exists, so all four supplier×class cells are NOT EVALUATED
  against the unchanged frozen gate or run_v6 requirements. DELETE remains
  primary at ~4.41× swap value. Zero PASS, FAIL, PARK-C, or graph unlock. The
  later launch-race reconciliation supersedes the original debit snapshot:
  aggregate DINO debit is 0.74 GPU-h and owner capacity is 25.47 h. Comparison
  `two_supplier_comparison_v3_gpu_terminal/TWO_SUPPLIER_COMPARISON.json`
  (`454bc4e0…acf8f`). (Codex root)
- 2026-07-30 — DINOv2 Kaggle GPU latest run **TERMINAL
  INFRASTRUCTURE-INVALID / NO RETRY**: this evidence was initially attributed
  to v1 and is now attributed to latest v2 after launch-race reconciliation.
  First-action inventory authenticated 24,968 files / 87.726 GB in
  1,242.976 s and step 0 resolved all 77 requests, weight, source, three wheels,
  and competition root with zero partial reuse. Pip then rejected the
  content-hash-prefixed wheel basename before dependency completion,
  source/model/worker work, features, B0/A1/A0, or outcome. V2's timed
  conservative runtime is 0.3989 h; the aggregate two-run quota debit is
  0.74 h and owner capacity is 25.47 h with no GPU authorization. Both classes
  are INVALID with no
  PASS/FAIL, PARK-C, or graph unlock. Receipt
  `kaggle_gpu_bakeoff_v1/TERMINAL_INVALID_INFRASTRUCTURE.json`. (Codex root)
- 2026-07-30 — CELLECT bleed-stop v1 **LOCAL MOUNT PASS / NO LAUNCH**:
  the single-file kernel embeds the 77 request records and every required
  content pin. An opaque 83-file expanded-tar mount with the raw checkpoint
  member and zero manifest files passes; downloaded layout also passes and
  missing/duplicate objects fail closed. Post-step0 science is unchanged;
  independent/shared tests pass 7/7 and 24/24. No upload, authorization,
  kernel, debit, outcome, or verdict; CELLECT stays launch-closed until DINO
  Lane A reports. Receipt `cellect_bleed_stop_v1/TERMINAL_LOCAL_PASS.json`.
  (Codex root)
- 2026-07-30 — DINOv2 promoted Kaggle CPU run **RUNNING >379 s / PRIOR
  STEP-0 FAMILY NOT REPEATED**: v3 first exposed cross-kernel preflight scope
  at 0.992640 s (0.002213 CPU-h / 0 GPU-h). The causal promotion was accepted
  under Kaggle's title-derived ID; KFG-16 now rejects that drift before future
  pushes. The sole active CPU run is past every prior startup failure and
  therefore passed its own step 0 before heavy work. Registry is 20 incidents /
  16 families; shared tests 14/14 plus real missing/empty-manifest tests 2/2.
  No further CPU launch, CPU terminal debit, submission, RunPod, graph, tier,
  threshold, DEAD, or plan action.
- 2026-07-30 — DINOv2 Kaggle GPU v1 **LAUNCHED ONCE / RUNNING**:
  the owner client confirmed kernel version 1 successfully pushed; the first
  authenticated status is `KernelWorkerStatus.RUNNING`. Authorization
  `DINOV2-R5-SEPARATION-GPU-V1` is now consumed, a second push is forbidden,
  and 27.0965 h remains before terminal debit. No feature, outcome, verdict,
  graph, training, full cache, 6bba, CV57, submission, or RunPod action yet;
  CELLECT stays launch-closed. (Codex root)
- 2026-07-30 — DINOv2 Kaggle GPU **SERIAL AUTHORIZED / NOT LAUNCHED**:
  self-contained assets version 1 is READY with 82/82 successful uploads;
  local expanded-mount hash resolution and the 4.957520 h conservative
  projection pass. Both prior campaign kernels are terminal ERROR, the new
  slug is absent from the owner run list, and the budget opens exactly one
  authorization against 27.0965 h remaining. No launch, debit, feature,
  outcome, or verdict yet; CELLECT remains launch-closed. (Codex root)
- 2026-07-30 — DINOv2 Kaggle CPU v3 **TERMINAL INFRASTRUCTURE-INVALID /
  CPU LANE CLOSED**: live status is ERROR; step 0 failed at 0.992640 s with
  `exact DINOv2 private-dataset manifest missing`. No dependency install,
  model load, inference, outcome, or GPU use occurred. V3 used 0.0022130 CPU-h;
  no retry is allowed and Kaggle GPU remains reserved for the 44b6 supplier
  bake-off. (Codex root)
- 2026-07-30 — Selector Kaggle CPU v4 port **LOCALLY REPAIRED / WITHDRAWN**:
  live v3 status is ERROR and its terminal log confirms the 225.608 s step-0
  envelope failure before heavy work. The single code file now inlines and
  time-asserts step 0; the repeated uncompressed-tar build
  `bdd63c43…9801` verifies expanded members only and fails closed on genuine
  member mismatch. Tests pass 4/4 transport plus 14/14 existing. The valid
  zero-swap local DAG-2 result remains authoritative; no selector CPU upload,
  v4 launch, or new debit occurred. (Codex root)
- 2026-07-30 — DINOv2 Kaggle CPU v3 **LAUNCHED after remote preflight PASS**:
  dedicated lane-owned dataset mount authenticated the exact checkpoint,
  source, population, 1,391,458 requests, 77+77 shards, and three wheels in
  0.054277 s with no heavy work and GPU disabled. Shared launch guard and 12/12
  failure-family fixtures plus 2/2 real missing/empty-manifest tests pass; v3
  was pushed exactly once. This launch-state entry is superseded by the
  terminal v3 entry above. Cross-campaign
  audit records 17 incidents / 14 root causes with exact messages/log hashes
  and forbids unchanged-signature retries. CPU preflight use is 0.0022981 h;
  GPU budget remains 2.9035 used / 27.0965 remaining. No submission, RunPod,
  graph, tier, threshold, DEAD, or plan action.
- 2026-07-30 — DINOv2 Kaggle GPU package **LOCAL MOUNT PASS / NO LAUNCH**:
  the single-file kernel's first parent call is a full recursive mounted-input
  size/SHA listing. Under opaque expanded paths, content hashes uniquely
  resolved 77 requests, the exact weight, source tar, and three wheels; missing
  and duplicate hashes fail closed. Frozen feature math and A0 permutation are
  array-identical. The versioned dataset payload has no external bundle
  manifest and reuses 0/724,634 quarantined MPS rows. No remote action,
  authorization, debit, outcome, or verdict; Lane B remains launch-closed.
  (Codex root)
- 2026-07-30 — DINOv2 CPU v2 **INVALID before heavy work** at 0.913 s:
  exact manifest lookup used the wrong private-dataset mount convention.
  No pip/model/inference/outcome; 0.0022223 CPU-h / 0 GPU-h. Human authorized
  a new version only after aggregating all Kaggle errors into shared prevention.
- 2026-07-29 — Authentic reverse Trackastra **N3 FAIL BOTH / TERMINAL**:
  unchanged sealed five-fold separation gave delete A1 retention
  `0–0.003268` and occupied-swap retention `0` in all folds; both evaluator
  tokens are `FAIL_PARK_C`, with no tier move applied or proposed and no graph
  unlock. Raw result `bd1c3d55…f46e`. (Codex root)
- 2026-07-30 — Two-supplier cross-read v2 **TERMINAL / GRAPH BARRIER CLOSED**:
  identical frozen population/folds/probe were preserved, but neither supplier
  produced a complete feature manifest. All four supplier×class cells are
  INVALID/NOT EVALUATED against both the frozen gate and run_v6 requirements.
  Zero PASS, zero FAIL, zero PARK-C moves, and zero graph unlocks; R5/R7 stay
  PARK-B. Comparison `006b88ac…410c`. (Codex root)
- 2026-07-29 — Authentic reverse Trackastra **N1–N2 PASS / N3 OPEN**:
  genuine CPU reverse inference completed exact frozen 44b6 coverage
  `36/3539/84883/121534`; repeat arrays are identical and manifest is
  `654cb499…6fe1`. Authenticated resume after a swap-growth stop reused only
  29 verified files and materialized seven missing videos; final guard had
  0 MiB swap growth. Supplier is **NON-GT-IDENTIFIABLE**; no graph or tier move.
- 2026-07-29 — DINOv2 local MPS **TERMINAL RESOURCE-INVALID / NO RETRY**:
  after a valid 4.362940 h projection, the delete class guard stopped at
  7,877.258 s / 2.483 GiB peak RSS when swap grew 140.875 MiB. Only 22/41
  feature shards existed; no class manifest, occupied-swap run, aggregation,
  B0/A1/A0, metric, or verdict opened. Partial shards are quarantined; R5 stays
  PARK-B and there is no PARK-C or graph unlock. Guard `11e215f2…595938`.
  (Codex root)
- 2026-07-29 — Authentic reverse Trackastra **N0 PASS**: owner-cmux SDK
  download authenticated both historical assets by content SHA and confirmed
  the archive member equals the standalone model. No listing gate, mutation, or
  kernel action; N1–N3 opened under the existing frozen contracts.
- 2026-07-29 — DINOv2 local MPS delete **PROJECTION PASS / RUNNING**: fresh
  host preflight ran 45.209 s at zero swap growth; two complete per-video
  releases authenticated a 4.362940 h full-class projection from 49,643 /
  1,399,860 exact unique endpoints. No outcome or aggregation opened; the
  class-isolated worker continues with swapouts unchanged. (Codex root)
- 2026-07-29 — CELLECT N3 v6 **TERMINAL INFRASTRUCTURE-INVALID / NO RETRY**:
  step 0 found zero pinned bundle-manifest matches beneath the exact private
  dataset mount and aborted at kernel-log time 3.956 s. No heavy import,
  checkpoint, projection, feature, B0/A1/A0, or scientific outcome. Conservative
  debit 0.0188 GPU-h; weekly state is 2.9035 used / 27.0965 remaining with no
  active authorization. CELLECT has no verdict and no PARK-C move. (Codex root)
- 2026-07-29 — CELLECT N3 v6 **RUNNING**: the single permitted T4 run was
  created at 13:55:13Z after serial budget and hash preflight passed. Scope is
  only the 1,391,458 frozen 44b6 endpoint requests; it must project `<5 h`
  after 64 tile jobs or abort and self-stops before the 5 h hard limit. No
  debit or scientific result yet. (Codex root)
- 2026-07-29 — CELLECT N3 **SERIAL AUTHORIZATION RECORDED** after N2 PASS:
  exactly one v6 T4 run may compute only frozen 44b6 endpoint features. Budget
  precheck is 2.8847/30.0 GPU-h used and 27.1153 remaining; the run must
  authenticate a `<5 h` projection after 64 tile jobs or abort and has a 5 h
  hard limit. No launch or debit yet. (Codex root)
- 2026-07-29 — CELLECT two-supplier N2 **PASS / N3 STILL CLOSED**: the shared
  step-0/kernel transport repair resolves the exact bundle, request manifest,
  and 77 request payloads by pinned content hash, with structured frozen class
  and identity records. The real owner-downloaded v6 payload yielded 1,391,458
  actions in 0.101 s before heavy work; opaque-name/parent and fail-closed tests
  pass 24/24 + 7/7 and the self-contained kernel compiles. No authenticated
  `<5 h` end-to-end projection exists yet, so no authorization, launch, debit,
  feature, scientific verdict, graph, tier, or threshold action. (Codex root)
- 2026-07-29 — Authentic reverse Trackastra **STOP_N0**: the exact historical
  checkpoint was absent under the frozen local authentication audit (265 archive
  indexes plus exact name/size/cache searches). N1–N3 stayed closed; zero
  reverse calls/rows, no fabricated reverse evidence, no separation result, and
  AE-P6/R8 remain PARK-B.
- 2026-07-29 — Two-supplier bake-off **N0 RECONCILED / ACTIVE**: the historical
  comparison's DAG-2-null references now cite valid run_v6 requirement surface
  `6e62a29c…6c71b`. Mean/robust-p05 AUC requirements are delete 0.85/0.90 and
  occupied swap 0.90/0.95; frozen separation thresholds remain unchanged.
  SWAP-O0 remains PARK-A, while its 44b6 headroom basis is corrected from the
  legacy CG0 `+0.075828` diagnostic to the compiler-valid run_v6 ceiling
  `+0.018571397`. No tier, DEAD, graph, threshold, new GPU/Kaggle
  authorization, budget, submission, RunPod, training, full-cache, 6bba, or
  CV57 action. (Codex root)
- 2026-07-29 — Launched the single DINOv2 CPU v2 retry under locked exact-path
  execution repair 1. Status RUNNING; frozen science unchanged, GPU/TPU/internet
  disabled, and no GPU debit.
- 2026-07-29 — DINOv2 CPU v1 **INVALID before heavy work**: recursive manifest
  discovery traversed the lazy-mounted competition tree and step 0 exceeded
  five seconds at 190.30 s. No pip/model/inference/outcome; 0.0548687 CPU-h,
  zero GPU debit. Locked one exact-path-only v2 repair; local step 0 0.0020515 s.
- 2026-07-29 — Human emphasized schedule pressure and directed aggressive use
  of eligible Kaggle compute: quota conservation is no longer an optimization
  goal. Scientific gates, serial authorization, per-run hard kills, submission,
  RunPod, full-cache, and plan-amendment hard stops remain unchanged.
- 2026-07-29 — Launched DINOv2 R5 private Kaggle kernel v1 after the manual v8
  byte-exact pass. Status RUNNING; CPU only, GPU/TPU/internet disabled, 11.5 h
  soft stop. No GPU debit, submission, or RunPod action.
- 2026-07-29 — Manual owner download of private assets v8 passed byte-exact
  DINOv2 CPU-bundle verification: 166/166 frozen inventory records, exact
  checkpoint/source/population/77 requests/three wheels, zero extra or missing
  payloads. Kaggle only expanded the inner ZIP. No kernel launch or budget use.
- 2026-07-29 — Private assets version 8 is READY as the temporary DINOv2 R5
  CPU carrier. It contains the 148,679,264-byte archive `580499a2…ca40`,
  pinned source/weight/population/77 requests and three wheels; optimized local
  step 0 passes in 0.003668 s. Awaiting manual v8 round-trip; no kernel launch.
- 2026-07-29 — Selector Kaggle CPU rerun **TERMINAL INVALID / NO RETRY**:
  v3 failed closed before heavy work when lazy-mount traversal exceeded the
  five-second step-0 envelope. V1–v3 total 0.123161 CPU-h / 0 GPU-h; no
  dependency install, scorer, aggregate, threshold, tier, or scientific result.
- 2026-07-29 — Locked and locally verified the DINOv2 R5 Kaggle CPU transport
  before outcomes: 167-member archive `580499a2…ca40`, exact weight/source/
  population/77 requests, step 0 at 0.226 s, 9/9 focused tests, GPU disabled.
  No DINO dataset version or kernel yet.
- 2026-07-29 — Selector Kaggle CPU v3 confirmed RUNNING under execution repair
  2. The push client hung after server acceptance and was released only after a
  separate owner-shell status confirmed version 3; no duplicate launch.
- 2026-07-29 — Selector Kaggle CPU v2 **INVALID inside step 0** at 197.23 s:
  Python 3.12 gzip header OS-byte drift plus an overbroad 595 MiB full rehash.
  No heavy work/outcome; 0.0565042 CPU-h recorded, zero GPU debit. V3 repair is
  limited to cross-version gzip normalization and first-seconds inventory use.
- 2026-07-29 — Launched selector Kaggle CPU v2 under locked single-file
  execution repair 1; `enable_gpu:false`, science unchanged, no GPU debit.
- 2026-07-29 — Selector Kaggle CPU v1 **INVALID before step 0**:
  `ModuleNotFoundError` for the sibling preflight at 0.858 s because Kaggle
  packaged only `code_file`. No inputs or outcomes consumed; 0.0019269 CPU-h
  recorded separately, zero GPU debit. Repair is single-file inlining only.
- 2026-07-29 — Launched private selector-requirement CPU kernel v1 after manual
  v7 round-trip and 0.592 s expanded-input preflight verification;
  `enable_gpu:false`, no GPU debit, no submission, and no RunPod action.
- 2026-07-29 — Manual owner download of private assets v7 passed logical
  byte-exact verification for all 335 inventory records. Kaggle's sole
  transform expanded one deterministic gzip; `gzip.compress(level=9,mtime=0)`
  reproduces frozen SHA `6be1c77b…a5324`. Expanded-input step 0 passes in
  0.592 s before heavy work. No kernel or GPU debit.
- 2026-07-29 — Division-value DAG **TERMINAL / DIAGNOSTIC HALT**: exact
  38-event ceiling `+0.092067842` [GT-DEPENDENT], but current-pool tier (a) is
  1/38 and only `+0.002406490` [NON-GT-IDENTIFIABLE] at perfect selection.
  Tier (b) requires an unbuilt stronger representation and has no measured
  selector/AUC/retention surface. The +0.005 condition is therefore infeasible
  for (a) and achievability is unestablished for (a+b). No amendment proposal,
  fitting unlock, tier/plan/DEAD action, GPU, Kaggle, budget debit, RunPod,
  submission, N1, fold, or P0 change. (Codex root)
- 2026-07-29 — Owner-cmux upload created READY private assets v7 containing
  only the selector CPU carrier. Status JSON confirms version 7 READY, but the
  CLI download stopped at the known private `GetDatasetMetadata` 403 before any
  bytes; repeated download attempts are halted for manual owner verification.
  No kernel launched. CELLECT v6 must be restored as latest before its GPU run.
- 2026-07-29 — Locked the DAG-2 Kaggle CPU transport amendment before any
  remote outcome and built its 394,376,773-byte private archive
  (`4bb1c4a6…e6da3`; 336 members; 62 wheels; 41 frozen 44b6 videos);
  `enable_gpu:false`, 16 GiB, pure-stdlib step 0, and 22/22 tests verified. No
  remote launch or GPU debit yet.
- 2026-07-29 — CELLECT single-bundle Task 1 **PASS / NO LAUNCH**: corrected
  private dataset v6 is READY and owner-cmux round-trip verified 87/87 files
  byte-exact under `ae3bc9b2…7fecd3`, including 77 frozen request shards,
  1,391,458 actions, population `4ab7bb23…76f0f`, checkpoint
  `f3e8e730…89ce5`, and all wheels. The incomplete v5 directory-skip attempt is
  quarantined. Kernel metadata has one dataset source; universal step 0 passes
  locally in 0.0545 s and 21/21 focused tests include <5 s missing/empty
  request aborts. No kernel, GPU debit, submission, RunPod, graph, full cache,
  tier, threshold, or DEAD action.
- 2026-07-29 — Selector-requirement diagnostic **TERMINAL /
  HUMAN_REVIEW_REQUIRED**: mean +0.005 requirements are delete AUC 0.85 /
  retention 0.020394 / prefix 16,384 and occupied-swap AUC 0.90 / retention
  0.067910 / prefix 1,024; robust p05 requires 0.90 / 0.044946 and 0.95 /
  0.147388 respectively. Frozen 0.65 / 0.15 is TOO LENIENT for both. Restore
  is population-infeasible (50 / 0 beneficial). Independent validation passes
  8/8 with all three direct official oracle-point matches and 0 MiB swap.
  Amendment proposed but not applied; no tier, threshold, plan, LEDGER, DEAD,
  GPU/Kaggle/RunPod/submission, 6bba, CV57, or external action occurred.
  Terminal `selector_requirement_44b6/run_v6/`; review `reviews_v1/final/`.
  (Codex root)
- 2026-07-29 — Selector-requirement **EXECUTION REPAIR 6 FROZEN AFTER
  CLASSES / BEFORE AGGREGATE**: the aggregate loader now accepts zero noise
  runs only for an explicitly one-class degenerate population and continues to
  require exact 7×20 coverage for mixed-label classes. True-degenerate
  acceptance and false-degenerate rejection pass; full suite 24/24. No score,
  requirement rule, threshold, tier, plan, LEDGER, DEAD, GPU/Kaggle/RunPod/
  submission, 6bba, CV57, or external action changed. (Codex root)
- 2026-07-29 — Selector-requirement `restore_topology` stage **COMPLETE /
  VALID CLASS-DEGENERATE**: the exact frozen population has 50 actions / 0
  beneficial, and the oracle-best is the zero-action point with adjusted/raw/
  node delta `0/0/0`; +0.005 is therefore infeasible and controlled AUC/
  retention is undefined, so no synthetic noise runs were emitted. The fresh
  guard ran 349.107 s at 1.115 GiB RSS and 0 MiB swap; terminal class SHA is
  `03443bb2…281f22a`. Aggregate requirements remain pending. No tier,
  threshold, plan, LEDGER, DEAD, GPU/Kaggle/RunPod/submission, 6bba, CV57, or
  external action occurred. (Codex root)
- 2026-07-29 — Selector-requirement local **EXECUTION REPAIR 5 FROZEN AFTER
  SWAP / BEFORE RESTORE**: a restore launch failed closed before its child
  process when concurrent transport work generalized shared source paths.
  The accepted source transition defaults to the same local root and frozen
  ZipStore, preserves repair 4, and passes the full 22/22 suite; a separate
  local manifest pins it without modifying the concurrent transport manifest.
  No restore outcome, tier, threshold, plan, LEDGER, DEAD,
  GPU/Kaggle/RunPod/submission, 6bba, CV57, or external action occurred.
  Repair `EXECUTION_REPAIR_5.json`; manifest
  `EXECUTION_MANIFEST_LOCAL_V5.json`. (Codex root)
- 2026-07-29 — Selector-requirement repair-4 `occupied_swap` stage **COMPLETE /
  VALID**: exact frozen population 45,961 actions / 134 beneficial, 140/140
  deterministic AUC runs, and oracle-best adjusted/raw/node delta
  `+0.018571397/+0.018473552/0` at prefix 32,768 / applied support 29,574.
  The fresh guarded process ran 2,221.571 s at 1.017 GiB peak RSS and 0 MiB
  swap; terminal class SHA is `c37b8e0b…1eddfb03`. Restore and aggregate
  requirements remain pending. No tier, threshold, plan, LEDGER, DEAD,
  GPU/Kaggle/RunPod/submission, 6bba, CV57, or external action occurred.
  (Codex root)
- 2026-07-29 — Selector-requirement **EXECUTION REPAIR 4 FROZEN BEFORE
  OCCUPIED-SWAP OUTCOMES**: the atomic swap receipt path was stopped with no
  terminal artifact after 10,922.910 s because full-baseline validation per
  selected action made the 140-run stage non-viable inside the five-hour guard.
  Peak RSS was 0.997 GiB and swap growth 0 MiB. The repair compiles each
  already conflict-disjoint selected set as one exact primitive-union SWAP
  through the unchanged frozen N1 compiler; a direct test matches the atomic
  batch final graph hash, and the full focused suite passes 22/22. Rankings,
  populations, support/AUC/seed grids, scorer, thresholds, tiers, and the valid
  delete result are unchanged. No tier, threshold, plan, LEDGER, DEAD,
  GPU/Kaggle/RunPod/submission, 6bba, CV57, or external action occurred.
  Repair `selector_requirement_44b6/EXECUTION_REPAIR_4.json`. (Codex root)
- 2026-07-29 — Selector-requirement repair-3 `delete` stage **COMPLETE /
  VALID**: exact frozen population 1,345,497 actions / 1,118 beneficial,
  140/140 deterministic AUC runs, and oracle-best adjusted/raw/node delta
  `+0.081897667/+0.081489580/0` at prefix/support 2,048. The fresh guarded
  process ran 286.858 s at 1.081 GiB peak RSS and 0 MiB swap; terminal class
  SHA is `edae5bfd…a69dda3`. Swap/restore and aggregate requirements remain
  pending. No tier, threshold, plan, LEDGER, DEAD, GPU/Kaggle/RunPod/submission,
  6bba, CV57, or external action occurred. (Codex root)
- 2026-07-29 — Selector-requirement diagnostic **RESUMED / EXECUTION REPAIR 3
  FROZEN PRE-OUTCOME**: after two invalid zero-result processes, the active
  native goal realigned to one fresh zero-swap process per action class plus a
  guarded aggregate. The lightweight swap path removes Torch/training imports
  and reimplements only frozen input authentication, official matching,
  topology, edge classification, and conflict ownership; all scientific
  populations, AUC targets, 20 seeds, supports, compiler/scorer rules, targets,
  and frozen gates are unchanged. Focused tests pass 21/21. No resumed outcome,
  tier/threshold/plan/LEDGER/DEAD move, GPU/Kaggle/RunPod/submission, 6bba,
  CV57, or external action exists. Repair
  `selector_requirement_44b6/EXECUTION_REPAIR_3.json`. (Codex root)
- 2026-07-29 — Two-supplier separation **TERMINAL INVALID / GRAPH BARRIER
  CLOSED**. DINOv2 R5 closed `INVALID_RESOURCE_NO_VERDICT` after both guarded
  local paths stopped on swap growth before a feature shard. The one authorized
  CELLECT v4 kernel authenticated checkpoint SHA `f3e8e730…89ce5` at step 0,
  then stopped before feature computation with `FeatureError: no frozen feature
  requests found`. Its 1,110.781-second run debited 0.3086 kernel-h; weekly
  budget is now 2.8847 used / 27.1153 remaining, SHA `4dfecb95…b5b2`. Neither
  supplier has A1/A0 metrics or a per-class scientific verdict; both remain
  PARK-B, not PARK-C. No retry, graph preregistration/run, full cache, training,
  6bba, CV57, submission, RunPod, DEAD, plan, tier, or LEDGER change occurred.
  (Codex root)
- 2026-07-29 — `SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC` is terminal
  **INVALID_RESOURCE / NO VERDICT**. A preregistered exact dose/noise runner
  passed 17/17 focused tests and two fresh zero-swap preflights. Attempt 1 grew
  swap +16.31 MiB at 1.904 GiB peak RSS before any result file; one documented
  class-lifetime streaming repair preserved all scientific inputs/rules but
  attempt 2 grew +1,567.69 MiB at 1.453 GiB peak RSS. Both were killed, their
  empty outputs quarantined, and no partial curve consumed. Delete,
  occupied-swap, and restore-topology requirements and AUC 0.65 / retention
  0.15 sufficiency remain unmeasured. No tier, threshold, plan, LEDGER, DEAD,
  GPU/Kaggle/RunPod/submission, 6bba, CV57, or external action occurred; any
  retry needs human review. Barrier
  `selector_requirement_44b6/RESOURCE_BARRIER.json`. (Codex root)
- 2026-07-29 — Video-subsample fidelity **SUBSAMPLING_UNSAFE / ROOT BARRIER
  PASS**: preregistration `c0d2de2a…ac5e` and pre-outcome common36 amendment
  `3c497948…673fc` preceded every draw. Six standardized banked candidates
  replayed exactly on held 44b6; all fragment hashes and eight aggregates
  matched within `1e-12`. Across `k={3,5,8,12}`, three schemes, and 10,000
  fixed-seed draws/cell, 0/12 passed. `k=12` cleared rank only
  (`rho=0.828571`); sign was `0.813733–0.820683` and max width
  `0.023017–0.023308`. No contract, tier move, GPU, Kaggle, budget debit,
  RunPod, submission, DEAD, plan, N1, folds, or P0 action. Table
  `67cf33ed…9a3b9`; independent replay `12dcdcb2…bb6f`. (Codex root + three
  isolated lanes)
- 2026-07-29 — CELLECT v4 **REMOTE RUN LAUNCHED / IN FLIGHT**: after the
  zero-remote-effect filename repair, computer-use cmux successfully pushed
  `onebitbrain/g02-cellect-r7-separation-v1` version 4 at
  2026-07-28T19:06:40.866162Z. First authenticated status is RUNNING. This is the
  one authorized run; no retry remains. The monitor enforces the frozen kernel's
  `<5 h` projection and 4 h 55 m self-stop, with a 5 h outer limit. No feature
  result, debit, graph contract/run, full cache, training, 6bba, CV57,
  submission, or RunPod action exists yet. (Codex root)
- 2026-07-29 — CELLECT v4 cmux launch **LOCAL CLI REJECTION / ZERO REMOTE
  EFFECT**: serial prelaunch passed at 27.4239 h remaining, but the first push
  command exited in 0.691 s before network dispatch because Kaggle CLI requires
  the canonical filename `kernel-metadata.json`. No kernel version/run or debit
  exists. A byte-exact canonical metadata copy at SHA `4d214527…7264` is the
  execution-only staging repair; kernel `1c8a6961…e58`, gate, B0, population,
  labels, and folds are unchanged. The one authorized remote run remains unused;
  no retry is allowed after a remote run is created. (Codex root)
- 2026-07-29 — DINOv2 R5 local materialization **TERMINAL
  INVALID_RESOURCE / NO VERDICT**: the integrated retry's 45.108-second
  preflight passed at zero growth, then model initialization caused
  +254.3125 MiB system swap growth in 9.765 s at 0.436 GiB child RSS. The guard
  terminated it before any endpoint shard and quarantined an empty partial
  directory. Together with attempt 1's pre-load +21.3125 MiB handoff growth,
  this closes the authorized local path. R5 remains PARK-B; it is not PARK-C
  because no A1/A0 or scientific gate outcome exists. No DINO Kaggle fallback,
  graph, full cache, training, 6bba, CV57, submission, budget debit, or RunPod
  action occurred. (Codex root)
- 2026-07-29 — DINOv2 R5 local attempt 1 **FAIL-CLOSED / RESOURCE GUARD**:
  exact MPS forward benchmark projected 0.62 h for 1,417,863 endpoints and the
  45.019-second idle preflight passed at zero growth, but system swapouts rose
  21.3125 MiB in the handoff before model load; the guard terminated the child
  after 0.271 s at 0.069 GiB peak RSS. No output directory, endpoint feature,
  A1/A0, verdict, graph, full cache, training, 6bba, CV57, submission, Kaggle
  debit, or RunPod action exists. One integrated preflight/launch execution
  repair will fail closed on any further growth. (Codex root)
- 2026-07-29 — Two-supplier separation **AUTHORIZED / ACTIVE**: owner opened
  local CPU/MPS DINOv2 R5 endpoint-feature materialization first and exactly one
  serial CELLECT v4 Kaggle feature run against unchanged gate
  `9fb030bd…3310e` and sealed B0 `a9ce8c0b…b4ff2`. CELLECT prelaunch budget is
  27.4239 kernel-h with `<5 h` authenticated projection and hard 5 h kill;
  scope is the exact 44b6 separation population only. No launch, feature,
  outcome, debit, graph contract/run, full cache, training, 6bba, CV57,
  submission, or RunPod action has occurred. (Codex root)
- 2026-07-29 — CELLECT R7 **PRIVATE ASSET V4 READY / AUTHENTICATED BYTE-EXACT
  ROUND-TRIP PASS / NO LAUNCH**: owner reauthentication in computer-use cmux
  restored status/upload access and created version 4. `ListDatasetFiles` and
  the CLI metadata preflight remain endpoint-specific 403s, not entitlement
  failures; a direct authenticated SDK download in the same cmux session
  recovered all seven assets. Kaggle expanded the uncompressed tar, and its
  safe-ASCII member returned at exactly 10,197,221 B with SHA
  `f3e8e730…89ce5`; every other payload hash matched. Focused tests pass 17/17.
  No kernel, GPU, submission, RunPod, DEAD, or plan action occurred. Contract
  `a1e6180a…3398`; receipt `4aa00106…ecd0`. (Human auth + Codex root)
- 2026-07-29 — CPU-only Wave-3 **ROOT BARRIER PASS**: merged five disjoint
  lane fragments. AE-P5 and AE-P7 move PARK-B→PARK-A on exact committed
  contracts plus positive banked occupied-swap P0 headroom; G2 moves
  PARK-B→PARK-C for `SA-SWAP-W1-EXACT-REGRET` only. DINOv2 authenticates and
  stages R5 as a second independent supplier; reverse Trackastra stops at a
  bounded 0.758 CPU-h request; all 38 fragmented division events receive exact
  continuity targets. The independent NODETOPO replay confirms
  `−0.00000343615/0/0` and unblocks nothing. Census becomes 0/15/19/10/1;
  accepted/uncited moves 3/0. No GPU, Kaggle, budget debit, RunPod, submission,
  DEAD, plan, or CELLECT action. Decisions `202cc2a5…bec9`. (Codex root +
  five isolated lanes)
- 2026-07-29 — CELLECT R7 **CMUX ROUND-TRIP WORKFLOW FROZEN / AUTH HOLD
  UNCHANGED**: inspected the installed CLI without an API call and pinned its
  `status/current_version_number` and `name/size` JSON receipt schemas. The
  fail-closed orchestrator dry-runs only private-dataset version, status, files,
  and download commands; it contains no kernel or submission command. Synthetic
  exact/mismatched receipt tests pass, bringing the focused suite to 16/16.
  Contract `e4b51091…a6bc`. No credential refresh, upload, download, launch, or
  other external action occurred. (Codex root)
- 2026-07-29 — NODETOPO exact 50-action graph gate **KILL_44B6_STOP**:
  adjusted/raw/node deltas were
  `−0.000003436150368 / 0 / 0`; raw and node passed at equality, but adjusted
  missed the preregistered `+0.005` floor. All 50 frozen restorations and 24/24
  affected-video N1 batches replayed exactly; edge TP/FP/FN remained
  `10,341/1,064/673`. The authoritative committed lane `189c43b` used a
  111.186 s guard, 0.667 GiB peak RSS, and 0 MiB swap growth; a separate
  evaluator reproduced its exact scores, deltas, and verdict. The run stopped
  before 6bba/CV57 with no GPU, Kaggle, external action, tier move, DEAD
  promotion, or plan amendment. Terminal `540adb67…b31ac`; validation
  `6f87905b…579b3`; provenance closure `f2c7b12a…2c9d4`.
  (Codex root + independent lanes)
- 2026-07-29 — CELLECT R7 **AUTH/BYTE DIAGNOSIS CORRECTED / LOCAL V4 READY**:
  v3 arithmetic proves the remotely stored direct checkpoint is 9,238,024 B,
  959,197 B below the authenticated 10,197,221 B pin. The rules/entitlement
  diagnosis is retracted; the owner-private dataset's 403 is a CLI token/scope
  failure. A deterministic uncompressed tar (`a8110e0d…f559`) with safe-ASCII
  member recovers exact SHA `f3e8e730…89ce5` at step 0 in 0.027 s, and 16/16
  focused tests pass. Cmux list/download still return 403 under its saved
  credential, so upload and remote round-trip await credential refresh. No
  dataset version or kernel launch occurred. Infrastructure-fix contract
  `e4b51091…a6bc`. (Human correction + Codex root)
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
- 2026-07-28 — CELLECT R7 **RETRACTED 403 DIAGNOSIS / HISTORICAL**:
  the then-recorded competition-entitlement explanation is superseded by the
  2026-07-29 owner correction and byte arithmetic above. The dataset is
  owner-private; the denial is a CLI token/scope failure. No external action
  occurred under the retracted diagnosis. (Superseded human clarification +
  Codex root)
- 2026-07-28 — CELLECT R7 **SUPERSEDED V3 PACKAGING DIAGNOSIS / HISTORICAL**:
  `kaggle datasets status` confirmed the private published-assets dataset is
  ready at current version 3, while `kaggle datasets files` and read-only
  download both returned HTTP 403 under the configured owner OAuth session.
  Its filename-mangling hypothesis is superseded by the exact 2026-07-29 byte
  mismatch proof above. No re-upload, kernel push/launch, GPU use, debit,
  training, cache, graph, 6bba, CV57, submission, or RunPod action occurred.
  (Codex root)
- 2026-07-28 — Concurrent CPU wave **POSTHOC QUALIFIED / FAIL CLOSED**:
  independent review confirmed all four exact commits, scopes, fragment hashes,
  and terminal worktrees, but found that SA-PARKB referenced a NODETOPO artifact
  absent from its commit, violating the no-live-sibling rule. PARKB is
  quarantined; its three proposed moves remain rejected, accepted/uncited moves
  remain 0/0, and census remains 0/13/22/9/1. The root chain is not
  clean-checkout self-contained because several shared authority artifacts are
  untracked, and NODETOPO's root receipt had a null top-level verdict despite
  terminal `PARK_RESOURCE_GUARD`. No rerun, GPU/Kaggle action, debit, submission,
  RunPod, DEAD promotion, or plan amendment occurred. Review
  `27d9a87e…8db9e10`. (Codex root + independent review)
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
