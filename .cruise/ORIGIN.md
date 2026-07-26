ROLE: ROOT COORDINATOR resuming H1 on the human-locked G-02 DAG, under a NEW human
decision (2026-07-26): DECOUPLE the H1 feature arms and run ZERO-WRFEAT H1 FIRST. The
real-mask (H3) arm is data-blocked (no authentic per-detection masks exist in the pipeline;
pseudo-masks forbidden) and is now a CONDITIONAL follow-on, not a launch prerequisite.

READ FIRST, then restate the frontier: reports/g02/screening/LIVE_STATE.md;
campaign_narrowing_and_hypothesis_dag_v1.md (§3, §3.1, §4, §5, §6); DAG.md; LEDGER.md
(you are its sole writer); p0_parity.md; reports/g02/screening/h1_execution.md. Obey AGENTS.md.

CONTEXT (verify against LIVE_STATE):
- H0 done: proposal-only screening BANNED.
- H1 PARKed at the launch-integrity gate because the contract required the real-mask arm
  present before the timing smoke, which also blocked the cheaper zero-wrfeat run. The raw
  fit-train105 (105 videos = 36 44b6 + 69 6bba) contains NO masks; the H1 remote datasets/
  kernel do not exist yet (trackastra-h1-option-c-assets-v1, trackastra-h1-fit105-cache-v1,
  kernel trackastra-h1-feature-cache all returned Not found; trackastra-6bba-assets-v1 is ready).
- Human has PRE-AUTHORIZED the narrowly scoped H1 timing smoke, conditional frozen-cache
  launch, and one conditional public probe (all gates still mandatory).

TASK — advance ZERO-WRFEAT H1 to the next human gate:
1. CONTRACT REVISION: amend the H1 launch contract to remove "authenticated real H3
   masks/wrfeat" from the pre-smoke prerequisites. The primary H1 arm is POINTS-ONLY,
   zero-filled 12-channel wrfeat (the exact interface QUAL/FULLPOOL used; needs no masks).
   Record the decouple decision + rationale in the contract and LIVE_STATE.
2. PACKAGE (both families, zero-wrfeat): finalize the H1 frozen-Trackastra feature-cache
   kernel + datasets for all 105 fit-train videos; pin seeds/env/library/source SHA-256s;
   reuse the frozen 27.46M checkpoint and the two-stage isolation. Emit the EXACT human-run
   commands to create + version-verify the remote datasets and kernel, and a read-only
   verification step to confirm they register (guard against the prior slug-registration
   failure). Uploads/creation are HUMAN-EXECUTED external mutations — STOP at that boundary
   with the commands; do not push assets yourself.
3. SMOKE CRITERIA (for the human-run two-family T4 timing smoke): <=7 h two-family
   projection, T4<->T4 repeatability, a proposed frozen M5<->T4 tolerance (exact on
   integer/geometry columns; per-column tolerance on float/neural, flagged for approval).
   No real-mask requirement.
4. GRAPH-GATE RUNNER: implement the H1 44b6-FIRST held-fit graph gate on the zero-wrfeat
   features so it runs the instant caches exist. Preregistered success: held-fit adjusted
   edge Jaccard >= +0.005 on BOTH families, no raw or node-recall regression. KILL
   immediately if 44b6 delta <= 0 (never spend 6bba/CV57 after a 44b6 failure).
5. CONDITIONAL H3 BRANCH (do NOT start now): only if zero-wrfeat H1 FAILS 44b6, open a
   separate scoped sub-node "H3-MASK-SOURCE" to first establish whether AUTHENTIC per-
   detection masks are obtainable (competition data / detector re-run) with provenance;
   pseudo-masks forbidden. The zero-vs-real ablation runs only after authentic masks are
   authenticated. Document this as the fallback; do not execute it.

HARD INVARIANTS:
- Frozen scorer only; NO validation12/CV57/holdout access; fail closed if referenced.
- Backbone FROZEN (inference only); no training/fine-tune; PEFT adapter only if frozen
  fails 44b6 and only selected on the 44b6 graph gate (§6.2).
- Do NOT re-screen any DEAD mechanism (LIVE_STATE DEAD list / DAG §6.1).
- No Kaggle dataset/kernel/GPU/submission or RunPod action executed by you; prepare exact
  commands and STOP at the human boundary. RunPod locked until accepted public >=0.89.
- Zero-swap discipline; 45-second clean idle preflight before any local run; never co-load
  V45 + Trackastra. Do not modify banked *_v1 reports or bhap-submit.

ON COMPLETION (required):
- UPDATE LIVE_STATE.md: record the decouple decision, refresh Current frontier / NEXT, and
  append a dated Log line (newest first).
- Append EXACTLY ONE coordinator LEDGER row (id H1-ZWRFEAT, do not rewrite existing rows).
- Commit locally per node on a codex/ branch; no push/PR unless asked.
- Report: what is packaged, the exact human-run create/verify + smoke commands, the next
  human gate, and any STOP reason.

OUTPUT: restate the frontier from LIVE_STATE, then execute steps 1-4 to the human upload
boundary and report. If LIVE_STATE disagrees with this summary, follow LIVE_STATE.
