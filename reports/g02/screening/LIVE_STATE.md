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
- Stop-loss doctrine, mandatory before any repeat-compute request: `MARGINAL_VALUE_GATE.md`.
- Kaggle kernel launch reliability, mandatory before any push: `tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_AUDIT.md` and `tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`.

## Current frontier (as of 2026-07-30)
- **CELLECT LOCAL HARNESS VERIFIED / ZERO-SWAP RESOURCE BLOCK / NO VERDICT.**
  The exact DINOv2 isolation contract is reproduced locally for CELLECT's
  authenticated 10,197,221-byte checkpoint: 77 declared request shards,
  one child at a time, fresh 45-second preflight, atomic checkpoints, strict
  cumulative zero-swap baseline, and no full-pool cache. Focused tests pass
  8/8. Five preflights each held zero growth, but every immediately following
  shard-0 start caused positive swapout (`+488.6875`, `+3.5`, `+174.6875`,
  `+30.625`, `+12.875 MiB`) under MPS/CPU reductions and was killed by the
  guard. There are zero feature files, zero checkpoints, and no B0/A1/A0 run:
  **CELLECT has no local scientific verdict.** Resume from shard 0 only during
  a stable zero-swap host window; do not weaken the guard. The GPU stop rule is
  honored and Kaggle debit is zero. Receipt:
  `tiered_surface/branches/sa_repr/phase2_separation_gate/cellect_local_bakeoff_v1/BLOCKED_REPORT.json`.
- **EXTREME-PREVALENCE OPERATING-POINT FINDING BANKED / PROPOSAL ONLY.**
  At roughly `10^-4` beneficial-action prevalence, ROC-AUC is not diagnostic
  of usable campaign lift: the CG0 false-positive-budget tail is binding.
  DINOv2's real rank signal (`0.670/0.682` AUC) retained only
  `0.0013/0.000`; Trackastra and CG1/PU/CRC exhibit the same operating-point
  failure. Future supplier gates are proposed, for human approval, to lead
  with precision/recall-at-budget and report AUC secondarily. No frozen gate,
  threshold, preregistration, or verdict changed.
- **DIVISION PLAN-AMENDMENT DOSSIER READY / HUMAN DECISION / FITTING LOCKED.**
  The 38/42 event-level bundle table now records required edges, motif,
  preferred late-fork equivalence, and duplicate risk. A fresh fail-closed
  receipt reverified the banked frozen-N1 oracle execution: TP/FP/FN `38/0/4`,
  adjusted-edge delta `+0.001591651042`, division Jaccard `0.904761904762`,
  and exact combined delta `+0.092067841519`. This is GT-dependent ceiling
  evidence only. The dossier proposes a 44b6-first arm for owner review but
  performs no amendment, fit, graph run, 6bba, or CV57. **FITTING REMAINS
  LOCKED.** Report:
  `tiered_surface/branches/sa_division_value/PLAN_AMENDMENT_DOSSIER.md`.
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
  **CELLECT PARK-B blocked-on-infrastructure; GPU bake-off HALT; no seventh
  run; human decision required before any further representation spend.**
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
  node rows in `LEDGER.md`. Division fitting and the B1 association-continuity
  prerequisite remain locked.
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
- **DIVISION VALUE DIAGNOSTIC TERMINAL / FITTING LOCKED.** The 38 exact frozen
  Wave-3 repair bundles give a combined-score ceiling of `+0.092067842`
  **[GT-DEPENDENT]**, with division TP/FP/FN `38/0/4`, Jaccard `0.904762`,
  adjusted/raw/node deltas `+0.001591651/+0.001607525/0`. The deployment-time
  audit assigns only **1/38** events to current-pool tier (a), worth
  `+0.002406490` **[NON-GT-IDENTIFIABLE]** even at perfect selection, below
  the `+0.005` actionability target. The remaining 37/38 require an unbuilt
  stronger representation; their `+0.092067842` (a+b) value is a
  representation-conditioned upper bound **[NON-GT-IDENTIFIABLE]**, not a
  realized selector gain. DAG-2 division status is
  `NOT_EVALUABLE_NO_DIVISION_SELECTOR`; no amendment proposal or tier move.
  Division fitting remains locked. Report:
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
  `ff551d8ee56503c13e769e03d7313cf1c95b26f8c779524d1095515aa40bbc22`
  (was `2d8299ee…da22f` before the 2026-07-30 run_v6 ceiling propagation)
  records **LIVE 0 / PARK-A 15 / PARK-B 19 / PARK-C 10 / DEAD 1**.
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
  fitting stays locked. No GPU, Kaggle, budget debit, RunPod, submission, DEAD,
  plan, or CELLECT action occurred.
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
  39/39 diagnostic actions compiled, while fitting remains locked behind a
  both-family association-continuity pass. `SA-PARKB-SWEEP` audited 14
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
  A1/A0 or scientific verdict exists; R7 remains PARK-B. Infrastructure-fix
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

## The one live lever
Phase 1 is closed. The supplier-agnostic **R-SEPARATION-FALSIFIER-44B6**
per-class amendment is frozen at `tiered_surface/branches/sa_repr/phase2_separation_gate/`
(`9fb030bd…3310e`). Its exact 44b6 populations and both-class B0 are banked, but
the feature run remains **INVALID** before supplier outcomes. The direct v3
checkpoint transport is proven byte-changing rather than absent. Private asset
v4 is READY and its authenticated cmux download proves a byte-exact safe-ASCII
checkpoint after Kaggle expands the uncompressed tar. The one authorized v4
launch is now terminal INVALID before feature computation because it found zero
frozen feature requests; the authorization is closed and debited 0.3086
kernel-h. The independent DINOv2 R5 local path is also terminal
INVALID_RESOURCE. Neither supplier has A1/A0 metrics or a scientific verdict.
On the exact fit-only Phase-1
N1-valid delete and occupied-swap populations, every class/fold must clear the
**measured binding floors** (amendment v4, 2026-07-30): A1 AUC `≥0.85` (delete) /
`≥0.90` (occupied swap) and CG0-budget retention `≥0.020394` (delete) /
`≥0.067910` (occupied swap), plus the unchanged `≥B0+0.05` / `≥A0+0.03` lifts on
both metrics, with exact per-video budgets and full integrity. This supersedes the
frozen `0.65`/`0.15` absolute floors: AUC is **tightened**, retention is
**relaxed**, because run_v6 measured what the `+0.005` target actually requires.
Trackastra's prior AUC `0.918491` still led to a failed graph, so PASS only
unlocks a supplier-class-specific 44b6 graph contract; it is not a promotion.
No further retry, A1/A0 evaluation, full cache, graph, 6bba, or CV57 run is authorized. Division still
requires one association-continuity arm passing both families. C2 remains
unscored, but its newly exact AE-P5 contract is PARK-A pending implementation
and rank-1 output. DINOv2 remains an authenticated second supplier at PARK-B;
its authorized local materialization path closed INVALID_RESOURCE.

## Census / fallback queue

| Queue | Surface | Tier | 44b6 evidence | Headroom / next condition |
|---:|---|---|---|---|
| 1 | `AE-P5` | PARK-A | no graph; exact C0/C2 contract committed | `+0.018571397`; implement frozen rule and emit rank-1 |
| 1 | `AE-P7` | PARK-A | no graph; exact phase-template contract committed | `+0.018571397`; fit105 identity/runtime smoke |
| 1 | `AE-P1` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.018571397`; newly named realizer only |
| 2 | `AE-P2` | PARK-A | broader mechanism unscored; failed exact realizers do not demote it | `+0.018571397`; newly named realizer only |
| gated | `SWAP-O0` | PARK-A | exact-regret instantiation `−0.000310288/−0.000296592/0` (PARK-C breadcrumb) | compiler-valid run_v6 ceiling `+0.018571397` (`6e62a29c…6c71b`); representation falsifier PASS |
| closed | `R5` DINOv2 XY | PARK-C | local 77-shard supplier gate returned `FAIL_PARK_C` for delete and occupied-swap | no graph preregistration; new named supplier only |
| blocked | `R7` CELLECT | PARK-B | sole v7 stopped at step 0 on missing declared Kaggle mount; no scientific verdict | infrastructure stop rule: GPU bake-off HALT, no seventh run, human decision required |
| closed | `G2` exact regret | PARK-C | `−0.000310288/−0.000296592/0` | new named decision-focused mechanism only |
| closed | `DELETE-O0` | PARK-C | exact retention beta `−0.000127119/−0.000131807/0` | materially new named deletion mechanism |

## Key numbers
- CELLECT local: **5/5 fresh 45-second preflights passed at 0 MiB swap
  growth; 5/5 shard-0 starts failed closed on positive system swapout**.
  Shard outputs/checkpoints/gate verdict: `0/0/none`; Kaggle/GPU debit `0`.
  Exact request plan: 77 shards / 1,391,458 actions / 217,595 tile jobs.
- Division amendment diagnostic: banked local `0.881522`, planning range
  `0.9255–0.9422`, gap `+0.044–+0.061`; 12/20/25 zero-FP division TP add
  `+0.0286/+0.0476/+0.0595`, versus edge-only cap near `0.896`. Frozen-N1
  oracle 38-bundle delta is exactly `+0.092067841519`; 25 TP with 5/10 FP
  falls to `+0.0532/+0.0481`. Human decision pending; fitting locked.
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
- Tier census is **0/15/19/10/1** over 45 nodes; machine state
  `tiered_surface/tier_census.json` is now SHA-256
  `ff551d8ee56503c13e769e03d7313cf1c95b26f8c779524d1095515aa40bbc22`
  after the run_v6 ceiling propagation (no tier changed). Weekly Kaggle budget artifact
  `tiered_surface/kaggle_budget.json` at SHA-256
  `1e9efa08…f7426` mirrors the authoritative Kaggle balance **4.53/30.0
  kernel-h used / 25.47 h remaining** and retains **3.6435 h** only as the
  named-debit audit subtotal. The **1.65 h** drift from the earlier 2.88 h
  tracker is recorded as a correction. Cumulative CPU use is **0.3010984 h**.
  No GPU authorization or CPU session remains active.
  RunPod is forbidden and submission human-gated.
- Two-supplier separation: common frozen population is **1,345,497 delete /
  1,118 beneficial** and **45,961 occupied-swap / 134 beneficial**, across five
  identical OOF folds. CELLECT N2 authenticated 77 request shards / 1,391,458
  actions locally. Its sole v7 run created exactly one fresh version, then
  failed at step 0 after 2.167 s because the declared owner-qualified Kaggle
  request-manifest path was absent. The stop rule parks CELLECT PARK-B
  blocked-on-infrastructure and permits no seventh run; it has no scientific
  verdict. DINOv2 instead completed all 77 request shards locally with atomic
  resume and zero swap growth. Its unchanged sealed B0→A1→A0 gate returned
  `FAIL_PARK_C` for delete (A1 mean AUC/retention `0.670412/0.001307`) and
  occupied-swap (`0.682473/0`). No graph preregistration is unlocked for either
  supplier; result `c58b1a58…0d41f2`, CELLECT terminal
  `c0beda9b…5f706f7`.
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
