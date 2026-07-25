---
version: 2
status: complete
outcome: Execute the human-authorized H1 frozen Trackastra campaign node through the locked two-family T4 timing and cache gates, the 44b6-first held-fit graph gate with H3 ablation, gated 6bba and CV57, and—only if every gate passes—one public Kaggle probe; otherwise stop at the first preregistered failure with complete node accounting.
verify:
  - exact package, checkpoint, dataset-version, source, environment, CUDA-device, and split-scope authentication
  - two-family T4 timing smoke with exact repeatability witnesses and a conservative full-run projection at or below seven hours
  - evidence-derived M5-to-T4 tolerance frozen before graph scoring
  - authenticated real H3 mask or aligned wrfeat inputs for both fit-train families
  - complete both-family cache manifests and hashes
  - 44b6-first nested held-fit graph metrics, followed only on pass by 6bba and then CV57
  - conditional observation-distinct local score at least 0.89 and exactly one confirmed public submission
  - exactly one coordinator ledger row and one local commit per completed node
scope:
  touchpoints:
    - .cruise/ORIGIN.md
    - .cruise/FLIGHT.md
    - reports/g02/screening/LIVE_STATE.md
    - reports/g02/screening/LEDGER.md
    - reports/g02/screening/h1_execution.md
    - reports/g02/screening/h1_execution.json
    - reports/g02/screening/h1_public_probe.md
    - reports/g02/screening/h1_public_probe.json
    - scripts
    - tests
    - kaggle_kernel/trackastra-h1-feature-cache-prep
    - kaggle_kernel/trackastra-h1-public
    - private temporary H1 staging and downloaded-output directories
    - authorized H1 Kaggle datasets kernels and one competition submission
  non_goals:
    - any RunPod action
    - backbone training fine-tuning or PEFT training
    - any Section-6.1 DEAD mechanism
    - H2 execution
    - more than one new public submission
    - calibration-only submission or public-score-driven retuning
    - modification of banked *_v1 reports or the bhap-submit package
invariants:
  - The human-locked DAG and manifest-to-44b6-to-6bba-to-CV57-to-public falsification order remain authoritative.
  - The Trackastra backbone remains frozen and inference-only and the scorer identity remains frozen.
  - Forbidden validation12 CV57 holdout audit test and submission paths are not read before their named gates.
  - The H1 cache scope is exactly fit-train105 with 36 44b6 and 69 6bba videos.
  - H3 uses authenticated real instance-mask or aligned wrfeat inputs and never substitutes pseudo-masks.
  - The full cache launch occurs only after remote datasets witnesses repeatability tolerance runtime and H3 authentication all pass.
  - The 44b6 graph arm runs first and a delta at or below zero kills H1 before any 6bba or CV57 run.
  - H1 success requires adjusted edge-Jaccard delta at least 0.005 on both families with no raw-score or node-recall regression.
  - The public probe requires observation distinction and local CV57 at least 0.89.
  - Exactly one new public submission is permitted; RunPod remains forbidden.
  - Local runs observe a fresh 45-second clean-idle zero-swap preflight and never co-load V45 with Trackastra.
  - LIVE_STATE is updated after each gate change or result and the root coordinator alone appends exactly one LEDGER row per completed node.
  - Existing user work and protected banked artifacts are preserved.
quality:
  risk: high
  profiles:
    - core
    - backend-data
  gates:
    -
      id: core.intent
      dimension: intent
      criterion: Execution and any stop exactly match the authorized H1 outcome, locked order, scope fences, and protected invariants.
      oracle: Review ORIGIN, Flight, LIVE_STATE, reports, ledger rows, commits, and external action history.
      severity: high
      state: passed
      evidence:
        summary: H1 followed the exact authorized order and stopped at the mandatory Step-B prerequisite gate before every prohibited downstream action.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: review
    -
      id: core.functional
      dimension: functional-correctness
      criterion: Package verification, timing, cache generation, graph gating, CV57, and conditional submission behave exactly as specified at every reached stage.
      oracle: Run focused tests and inspect authenticated runtime outputs and scorer reports.
      severity: high
      state: passed
      evidence:
        summary: All twelve focused fail-closed tests passed and the reached launch-integrity stage correctly reserved quota without upload or execution.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: command
    -
      id: core.regression
      dimension: regression
      criterion: Existing Trackastra, scorer, graph-gate, and submission contracts remain intact and protected artifacts are unchanged.
      oracle: Focused regression tests, source hashes, and final diff/name-status review.
      severity: high
      state: passed
      evidence:
        summary: Frozen package hashes match H1-PREP, focused tests pass, and the H1 node commit changes only assurance, live-state, one ledger row, and new H1 reports.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: command
    -
      id: core.failure-paths
      dimension: failure-paths
      criterion: Missing or mismatched assets, forbidden paths, timing overflow, nondeterminism, tolerance breach, graph failure, low CV57, duplicate submission, and partial external failures fail closed.
      oracle: Failure-path tests plus inspection of guards and actual stop evidence.
      severity: high
      state: passed
      evidence:
        summary: Missing authenticated H3 inputs and absent remote package deterministically triggered STOP before upload, timing smoke, graph, CV57, or submission.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: command
    -
      id: core.completion-review
      dimension: completion-review
      criterion: A skeptical final evidence audit finds no missing requirement, fabricated result, unauthorized action, or unsupported completion claim.
      oracle: Fresh post-mutation review against every ORIGIN requirement.
      severity: high
      state: passed
      evidence:
        summary: Post-commit requirement audit found the explicit STOP condition satisfied, exact node accounting complete, and no unsupported result or remaining in-scope action.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: review
    -
      id: core.independent-review
      dimension: independent-review
      criterion: A genuinely separate skeptical review pass challenges package provenance, leakage boundaries, gate ordering, metrics, hashes, accounting, and external side effects.
      oracle: Fresh independent review receipt after all reached-stage mutations.
      severity: high
      state: passed
      evidence:
        summary: A separate post-commit skeptical pass re-queried Kaggle, reproduced absent H1 datasets/kernel, reviewed commit scope, and found no gate-order or side-effect defect.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: review
    -
      id: backend.data-integrity
      dimension: data-integrity
      criterion: Remote dataset versions, cache manifests, output archives, atomic publication, hashes, and retry behavior preserve exact identities without overwriting unrelated data.
      oracle: Inspect Kaggle metadata/status, local and remote manifests, hashes, counts, and retry/idempotency behavior.
      severity: high
      state: passed
      evidence:
        summary: Fit-only manifests authenticate 10710 raw and 210 cache files; no remote mutation occurred because required H3 and remote identities were absent.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: command
    -
      id: backend.security
      dimension: security
      criterion: Kaggle authentication is used without exposing secrets and all mutations are least-privilege and restricted to the authorized H1 slugs and one submission.
      oracle: Inspect commands and outputs for scope and secret hygiene.
      severity: high
      state: passed
      evidence:
        summary: Only owner-scoped read-only Kaggle status/list calls were made; no credential material was printed and no write privilege was exercised.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: review
    -
      id: backend.concurrency
      dimension: concurrency
      criterion: Repeated status checks or retries cannot create duplicate dataset versions, kernel launches, downloads, ledger rows, commits, or submissions.
      oracle: Verify resolved version IDs and idempotent status-driven control before each mutation.
      severity: high
      state: passed
      evidence:
        summary: Status was resolved before mutation; no remote write or retry occurred, and exactly one H1 ledger row and one node commit exist.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: review
    -
      id: backend.observability
      dimension: observability
      criterion: Timing, device, environment, dataset, cache, graph, CV57, and submission failures are diagnosable from durable non-secret reports and exact identifiers.
      oracle: Inspect local reports, Kaggle statuses, logs, manifests, hashes, and node accounting.
      severity: high
      state: passed
      evidence:
        summary: Durable Markdown and JSON reports record the exact missing prerequisites, remote state, hashes, zero usage, metrics not run, resume condition, and commit.
        context_fingerprint: 104461e75831fcdc500e951a2b6b203c064855b446a84cb4c2d779d5325c6e5b
        observed_at: 1785007676
        source: command
stall:
  quiet_turns: 3
  same_failure: 3
---
# Flight plan

First authenticate the current package, available local H3 inputs, Kaggle account, and
remote dataset/kernel state. Record the authorization gate in LIVE_STATE. Run only the
two-family timing/repeatability smoke. Continue to full caches only if every launch gate,
including authenticated H3 inputs and the seven-hour projection, passes. Run the nested
held-fit graph gate 44b6 first, then 6bba, then CV57. Create one public submission only
after an observation-distinct local score of at least 0.89. At the first failed gate,
reserve quota, write the H1 STOP/KILL evidence, append exactly one ledger row, commit the
node locally, and end.
