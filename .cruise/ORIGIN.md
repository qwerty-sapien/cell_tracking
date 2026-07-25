ROLE: You are the ROOT COORDINATOR driving the human-locked G-02 cell-tracking DAG.
Advance it one node at a time under its own falsification order and gates. Do not
improvise scope.

READ FIRST (in order), then restate the current frontier before acting:
1. reports/g02/screening/LIVE_STATE.md
2. reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md
   (human-locked 2026-07-25 plan; §3.1, §4, §5, §6)
3. reports/g02/screening/DAG.md
4. reports/g02/screening/LEDGER.md
   (bet history; the root coordinator is its SOLE writer)
5. reports/g02/screening/p0_parity.md
Also obey AGENTS.md.

CURRENT FRONTIER (verify against LIVE_STATE; follow LIVE_STATE if it differs):
- H0 DONE: proposal-only screening BANNED. Every candidate routes directly to the
  44b6-first held-fit GRAPH gate. Proposal metrics may be reported but never gate.
- H1-PREP DONE: the two-family Option-C frozen-Trackastra feature package is ready.
- NEXT node: H1 execution.

EXPLICIT HUMAN AUTHORIZATION FOR THIS RUN:
This prompt is the human approval required by DAG §4/§6.3 for the following narrowly
scoped external actions:
1. Upload/update the H1 Kaggle timing-smoke and feature-cache kernel/package.
2. Run the two-family CUDA timing smoke on Kaggle T4.
3. If and only if the smoke and all integrity gates pass, launch the frozen-feature
   cache generation for 44b6 and 6bba.
4. If and only if H1 subsequently passes every 44b6, 6bba, and full-CV57 gate below,
   prepare/upload/run the observation-distinct H1 inference/submission kernel and make
   exactly ONE public Kaggle submission.
5. Once Kaggle confirms that the one submission has been created, record its identity
   and END THE GOAL. Do not launch another probe or continue to H2.

Do not pause or ask for additional human approval for the authorized actions above.
This authorization does NOT cover RunPod, backbone training/fine-tuning, PEFT training,
extra submissions, or unrelated Kaggle jobs.

EXECUTION ORDER — DO NOT SKIP OR REORDER:

A. Record the authorization
- Re-read LIVE_STATE immediately before execution.
- Record that the narrowly scoped H1 Kaggle GPU/cache/public-probe gate has been
  explicitly authorized.
- Preserve all existing banked artifacts and histories.

B. Final launch verification and timing smoke
- Finalize and verify the launch-ready package, remote dataset versions, checkpoint
  identity, source/environment witnesses, explicit cuda:0 paths, and authenticated H3
  mask/wrfeat inputs.
- Run the exact two-family CUDA timing smoke.
- Require verified T4 witnesses and a defensible projection of <=7 hours for the complete
  two-family cache run, including packaging margin.
- Establish T4-to-T4 repeatability.
- Freeze a conservative, evidence-derived M5<->T4 tolerance before any graph scoring.
  The human authorizes adoption of that frozen tolerance without another approval,
  provided it is derived only from the preregistered parity evidence and is not tuned
  after graph results are seen.
- If runtime cannot be proven <=7 hours, remote datasets/witnesses cannot be authenticated,
  repeatability fails, tolerance cannot be defensibly frozen, or H3 mask/wrfeat cannot be
  authenticated: RESERVE quota and STOP. Do not launch a guillotine-risk job.

C. Generate H1 frozen feature caches
- Launch only after every step-B gate passes.
- Backbone remains strictly FROZEN and inference-only.
- Generate complete authenticated caches for both families: 44b6 and 6bba.
- Record exact kernel version, dataset versions, accelerator, elapsed time, cache counts,
  manifests, and SHA-256 identities.
- Do not access validation12, CV57, holdout, audit, test, or submission inputs during
  cache generation.

D. Run H1 held-fit GRAPH gate
- Use the implemented H1 runner immediately after caches authenticate.
- H3 is an ablation INSIDE the 44b6 gate: zeroed mask/wrfeat versus authenticated real
  mask/wrfeat.
- Preserve nested grouped held-fit separation.
- Run 44b6 FIRST.
- Preregistered H1 success:
  adjusted edge Jaccard delta >= +0.005 on BOTH families,
  with no raw edge-score regression and no node-recall regression.
- KILL H1 immediately if the 44b6 graph delta <= 0.
- On any 44b6 failure, do not run 6bba, CV57, or a public probe.
- If 44b6 is positive but below +0.005, H1 has not passed; report/PARK according to the
  locked DAG and do not promote it.
- Only after 44b6 passes all guards may 6bba run.
- Only after both families pass may full CV57 run.

E. Full CV57 and conditional public probe
- Do not read or run CV57 before both held-fit family gates pass.
- The public candidate must be observation-distinct and achieve local CV57 >=0.89.
  This is a submission-worthiness floor, not a prediction of public >=0.89.
- If local CV57 is below 0.89, do not upload or submit the public candidate.
- If local CV57 is >=0.89 and all integrity checks pass:
  1. prepare and hash-pin the inference/submission kernel;
  2. verify package-path parity and output identity;
  3. upload and run it on Kaggle;
  4. verify successful output;
  5. create exactly ONE public submission;
  6. record the kernel version, output archive, submission ID, status, and all SHA-256s;
  7. END THE GOAL immediately after submission creation is confirmed.
- Do not wait for or use the public score to tune this identity.
- Do not launch H2 in this goal. H2 remains eligible only if H1 demonstrates genuine
  graph-level signal and must be handled as a later node.

FALSIFICATION ORDER:
manifest/literature -> data/leakage -> 44b6 held-fit GRAPH delta
-> 6bba held-fit GRAPH delta -> full CV57 -> one authorized public probe

HARD INVARIANTS:
- Frozen scorer only.
- No validation12/CV57/holdout access or path reads before its named gate.
  Fail closed if a forbidden split is referenced.
- Trackastra backbone stays FROZEN and inference-only.
- No training or fine-tuning.
- A PEFT adapter (<1% parameters) remains a possible future path only if frozen features
  fail 44b6, but it is NOT authorized in this run.
- Do not re-screen any DEAD mechanism listed in LIVE_STATE/DAG §6.1.
- No RunPod launch. RunPod remains locked until accepted public >=0.89 and all DAG §3.1
  conditions are met.
- Zero-swap CPU discipline; perform a 45-second clean-idle preflight before every local
  run and never co-load V45 with Trackastra.
- Do not modify banked *_v1 reports or the bhap-submit package.
- Do not exceed the Kaggle submission cap. This authorization permits exactly one new
  H1 public submission and no others.

NODE ACCOUNTING:
- H1 is the active node.
- If H1 stops or fails, complete H1 accounting and end.
- If H1 passes through CV57, complete H1 accounting before beginning the public-probe node.
- The single public probe is a separate node and receives its own accounting.
- For each completed node:
  1. update LIVE_STATE Current frontier and Key numbers;
  2. append a dated newest-first LIVE_STATE Log entry;
  3. append EXACTLY ONE coordinator-owned row to LEDGER.md;
  4. commit locally on a codex/ branch;
  5. do not push or open a PR.

FINAL REPORT:
Report each completed node separately:
- node ID and verdict;
- exact adjusted/raw graph deltas and node recall per family;
- H3 zeroed-versus-real ablation results;
- CV57 score if authorized by prior gates;
- timing, GPU-hour accounting, kernel/dataset/submission identities;
- artifacts and complete SHA-256s;
- commit identity;
- any STOP/KILL reason;
- public submission ID/status if created.

Begin by restating the verified LIVE_STATE frontier and naming H1 as the single node
being advanced. Then execute continuously without waiting for further approval, stopping
only on a preregistered failure/STOP condition or after the one authorized public
submission is confirmed.
