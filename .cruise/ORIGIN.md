TASK: Act as ROOT COORDINATOR to initiate the campaign cascade defined in
reports/g02/screening/campaign_narrowing_and_hypothesis_dag_v1.md (the human-locked
2026-07-25 DAG). Follow its falsification order and gates EXACTLY. Do NOT create a
competing DAG doc. Do NOT re-screen any Section-1 DEAD mechanism. Do NOT launch any
Kaggle GPU / submission / RunPod job (human-gated). Frozen scorer only; no holdout
(validation12/CV57) access; zero-swap CPU discipline; commit locally per node. You are
the sole writer of LEDGER.md; workers own disjoint outputs.

READ FIRST: campaign_narrowing_and_hypothesis_dag_v1.md, DAG.md, _shared_context.md,
P1_framework_and_rubric.md, p0_parity.md, and LEDGER.md rows GA-TRACKASTRA-* (64-70).

INITIATE TWO NODES IN PARALLEL (per §5 ranking):

== NODE H0 — proxy-validity audit (CPU, no training; run to completion now) ==
Purpose: settle whether proposal-level screening should be formally banned.
- Assemble every available (proposal metric, held-fit GRAPH delta) pair from the ledger
  (~15: OOF precision, utility gain, AUC, support vs the measured graph adjusted/raw
  delta) — e.g. CTRL-OOF-GRAPH, GA-TEMPHYBRID, GA-POINTCTX-OOF, GA-CROPLINK-6BBA-OOF,
  GA-SCENCAL, GA-TRACKASTRA-QUAL, etc. Use only values already banked in reports; do NOT
  rerun any model.
- Fit graph-delta (and its sign) on each proxy; report Pearson/Spearman/Kendall and a
  sign-agreement rate with bootstrap CIs.
- PREREGISTERED DECISION: if no proxy predicts graph-delta sign at |rho| >= 0.5, emit a
  formal "proposal-only screening BANNED; route candidates straight to the 44b6-first
  held-fit graph gate" verdict. If one clears 0.5, recover it as a fast-fail WITH its
  measured error bar.
- Output reports/g02/screening/h0_proxy_validity_v1.md + .json. Add ONE LEDGER row
  (id H0-PROXY-VALIDITY, class DIAGNOSTIC/CONTROL). No holdout, no training, no GPU.

== NODE H1-PREP — prepare (not launch) the frozen Trackastra feature caches ==
Purpose: unblock H1 by making the 44b6+6bba frozen-inference kernel launch-ready, under
the corrected parity contract. This is PREP ONLY; the GPU launch is human-executed.
- Adopt the option-C contract decided this session: canonical features are generated
  ENTIRELY on T4 (one pinned env); parity becomes (1) T4<->T4 array_equal repeatability
  - (2) a one-time per-column M5<->T4 tolerance classification (benign-FP vs bug). REMOVE
  the cross-device numpy.array_equal byte-identity requirement from the launch path;
  replace with per-column tolerance (report proposed atol/rtol per float column; exact on
  integer/geometry columns). Any tolerance as FINAL contract is flagged for human approval,
  not self-enacted.
- Reuse the corrected Stage-A-once/Stage-B-twice kernel and the two-stage isolation
  (GA-TRACKASTRA-2STAGE) that reached Stage A 59/69; scope to BOTH families 6bba AND 44b6
  (H1 requires 44b6 — a 6bba-only cache replays the GA-TEMPHYBRID regression and cannot
  test the surviving question).
- Package: frozen 27.46M checkpoint as a Kaggle dataset + scripts + data access; pin
  seeds/env/library versions; verify a 45-second clean zero-swap idle preflight; emit
  kernel/script/checkpoint/manifest SHA-256s and a projected T4 wall-time + GPU-hour
  estimate (must fit inside 7 h with packaging margin; frozen inference does NOT consume
  the 30-h training cap).
- STOP at the launch boundary: write a launch-readiness report
  reports/g02/screening/h1_feature_cache_prep_v1.md with the exact human-run command,
  the quota estimate, and the §6.3 fallback ("reserve quota; do not launch if the kernel
  cannot be verified clean before reset"). Add ONE LEDGER row (id H1-PREP, class
  CONTROL/IMPLEMENT). Do NOT launch.

FOLLOW-ON (document as the next-node contract; do NOT execute yet):
H1 graph gate runs 44b6-FIRST: held-fit adjusted edge Jaccard >= +0.005 on BOTH families,
no raw/node-recall regression; kill immediately if 44b6 graph delta <= 0. Fold in H3
(zeroed vs real mask-feature ablation) inside the 44b6 gate. H2 (shell background
abstention) only after H1 shows real graph-level signal.

CONSTRAINTS
- Deterministic; record seeds. If a required artifact is missing, STOP and name it;
  do not fabricate. Fail closed on any holdout/CV57 path.
- Add unit tests for the H0 correlation/decision logic and the H1-PREP tolerance-contract
  comparator (benign-FP vs structural classification on a synthetic M5/T4 pair).
- New branch. Do not touch bhap-submit or any banked *_v1 report. Append LEDGER rows as
  coordinator; do not rewrite existing rows.
