---
version: 2
status: complete
outcome: Prepare and verify the decoupled points-only H1 zero-wrfeat package and its fit105-only, 44b6-first graph runner, then stop before the first human-executed Kaggle upload.
verify:
  - launch contract makes zero-filled 12-channel wrfeat the primary arm and removes authentic masks from every pre-smoke prerequisite
  - assets and cache dataset packages cover exactly 105 fit-train videos split 36 44b6 and 69 6bba with frozen checkpoint, seed, environment, source, and SHA-256 pins
  - exact human create, version-verify, registration, timing-smoke, and conditional-cache commands are fail-closed and owner-slug specific
  - smoke requires a two-family projection at or below seven hours and exact T4-to-T4 repeatability
  - M5-to-T4 policy is exact for integer and geometry columns with explicit per-column float/neural proposals held for human approval
  - graph runner reads only authenticated fit105 cache artifacts, runs 44b6 before 6bba, and kills at 44b6 adjusted delta at or below zero
  - graph success requires adjusted edge-Jaccard delta at least 0.005 on both families with no raw or node-recall regression
  - H3-MASK-SOURCE is documented as an unexecuted authentic-mask-only fallback after a zero-wrfeat 44b6 failure
  - LIVE_STATE and exactly one H1-ZWRFEAT ledger row record the human upload boundary
  - one scoped local commit exists on the current codex branch with no push or PR
scope:
  touchpoints:
    - .cruise/ORIGIN.md
    - .cruise/FLIGHT.md
    - reports/g02/screening/LIVE_STATE.md
    - reports/g02/screening/LEDGER.md
    - reports/g02/screening/h1_execution.md
    - reports/g02/screening/h1_zero_wrfeat_launch_contract.json
    - reports/g02/screening/h1_zero_wrfeat_graph_gate_contract.json
    - scripts/prepare_h1_feature_cache.py
    - scripts/evaluate_h1_zero_wrfeat_graph_gate.py
    - tests/test_h1_feature_cache_prep.py
    - tests/test_evaluate_h1_zero_wrfeat_graph_gate.py
    - kaggle_kernel/trackastra-h1-feature-cache-prep
  non_goals:
    - any Kaggle dataset kernel GPU submission or RunPod mutation by Codex
    - any validation12 CV57 holdout audit test or submission path read
    - authentic-mask sourcing or H3 execution
    - backbone training fine-tuning or PEFT training
    - any Section-6.1 DEAD mechanism
    - modification of banked *_v1 reports or bhap-submit
invariants:
  - H0 proposal-only screening remains banned and proposal metrics never gate promotion.
  - The Trackastra backbone and scorer remain frozen and inference-only.
  - The cache scope is exactly fit-train105 with 36 44b6 and 69 6bba videos.
  - Zero-wrfeat means the established points-only interface with twelve zero-filled wrfeat channels and no mask dependency.
  - The graph runner cannot reference validation12 CV57 holdout audit test or submission paths.
  - The 44b6 family executes first and adjusted delta at or below zero kills H1 before 6bba.
  - H3-MASK-SOURCE is conditional on zero-wrfeat 44b6 failure and forbids pseudo-masks.
  - Local runs receive a fresh 45-second clean-idle zero-swap preflight and never co-load V45 with Trackastra.
  - The root coordinator alone appends exactly one H1-ZWRFEAT row.
  - Existing user work and protected banked artifacts are preserved.
quality:
  risk: high
  profiles:
    - core
    - backend-data
  gates:
    - id: core.intent
      dimension: intent
      criterion: The package stops at the human upload boundary and matches the decoupled H1 decision without widening scope.
      oracle: Review ORIGIN, Flight, contracts, LIVE_STATE, LEDGER, diff, and external action history.
      severity: high
      state: passed
      evidence:
        summary: The decoupled zero-wrfeat package stops exactly before the first Kaggle dataset creation command; no external mutation occurred.
        source: review
    - id: core.functional
      dimension: functional-correctness
      criterion: Package validation and graph orchestration implement exact fit105 scope, smoke gates, and 44b6-first promotion rules.
      oracle: Focused tests plus local package verification after the required preflight.
      severity: high
      state: passed
      evidence:
        summary: Both local dataset stages completed, all seven Python entrypoints compiled, and 18 focused tests passed.
        source: command
    - id: core.regression
      dimension: regression
      criterion: Existing cache preparation and graph scoring behavior remains intact and protected artifacts are unchanged.
      oracle: Focused regression tests, hash verification, and final name-status review.
      severity: high
      state: passed
      evidence:
        summary: The legacy contract still loads, active manifests reproduce their pins, and protected banked reports were not edited.
        source: command
    - id: core.failure-paths
      dimension: failure-paths
      criterion: Forbidden scope, missing caches, bad identities, timing overflow, nondeterminism, and 44b6 failures stop before downstream work.
      oracle: Negative tests and inspection of fail-closed guards.
      severity: high
      state: passed
      evidence:
        summary: Negative tests prove 44b6 nonpositive, subthreshold, raw-regressing, and recall-regressing results never call 6bba.
        source: command
    - id: core.completion-review
      dimension: completion-review
      criterion: A skeptical final audit finds no missing user requirement, unsupported result, or unauthorized external mutation.
      oracle: Fresh post-commit review against every ORIGIN requirement.
      severity: high
      state: passed
      evidence:
        summary: Final audit found every requested artifact, command, gate, state update, and STOP reason present without unsupported scientific metrics.
        source: review
    - id: backend.data-integrity
      dimension: data-integrity
      criterion: Dataset manifests, cache outputs, hashes, counts, and remote identity commands preserve exact fit105 identities.
      oracle: Inspect manifests, staging validation, hashes, and owner-slug-specific verification commands.
      severity: high
      state: passed
      evidence:
        summary: Local staging authenticated 53 asset files and 212 cache-package files from the exact fit105 manifests and chained SHA-256 pins.
        source: command
    - id: backend.security
      dimension: security
      criterion: Human commands expose no secrets and restrict mutations to the named H1 assets, cache, and kernel slugs.
      oracle: Inspect command handoff and confirm Codex made no external mutation.
      severity: high
      state: passed
      evidence:
        summary: Commands are owner-slug specific, print no credential material, and Codex performed no Kaggle or RunPod mutation.
        source: review
    - id: backend.concurrency
      dimension: concurrency
      criterion: Re-running verification cannot create duplicate versions or ledger rows and the launch sequence checks remote state before mutation.
      oracle: Inspect idempotency guards, exact ledger diff, and command ordering.
      severity: high
      state: passed
      evidence:
        summary: Stage commands refuse overwrite and remote status plus version checks precede the single timing-kernel push.
        source: review
    - id: backend.observability
      dimension: observability
      criterion: Human smoke and cache outputs contain durable device, environment, timing, repeatability, projection, identity, and hash evidence.
      oracle: Inspect contracts, kernel reports, and command handoff.
      severity: high
      state: passed
      evidence:
        summary: Contracts and the smoke report specify exact runtime, device, witness, repeatability, projection, and STOP fields.
        source: review
stall:
  quiet_turns: 3
  same_failure: 3
---
# Flight plan

Revise the launch contract around the established points-only zero-wrfeat interface, pin
and validate the fit105 package, and add a new manifest-only graph runner that cannot read
forbidden split metadata. Verify the package and negative paths locally only after a fresh
45-second idle preflight. Record the human upload boundary in LIVE_STATE and one
H1-ZWRFEAT ledger row, commit only the scoped files, and stop without any Kaggle or RunPod
mutation.
