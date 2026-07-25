---
version: 2
status: active
outcome: Complete H0 from banked report evidence and prepare, without launching, the option-C 44b6+6bba frozen Trackastra feature-cache package; append exactly one coordinator-owned ledger row per node and commit locally per node.
verify:
  - focused H0 correlation and decision unit tests
  - focused H1 tolerance comparator and package fail-closed unit tests
  - deterministic report and manifest validation
  - git diff review proving no banked v1 report, bhap-submit, holdout, or CV57 mutation/access
scope:
  touchpoints:
    - .cruise/ORIGIN.md
    - .cruise/FLIGHT.md
    - reports/g02/screening/h0_proxy_validity_v1.md
    - reports/g02/screening/h0_proxy_validity_v1.json
    - reports/g02/screening/h1_feature_cache_prep_v1.md
    - reports/g02/screening/h1_feature_cache_prep_v1.json
    - reports/g02/screening/LEDGER.md
    - scripts
    - tests
    - kaggle_kernel/trackastra-h1-feature-cache-prep
  non_goals:
    - competing DAG documentation
    - rescreening any Section-1 DEAD mechanism
    - Kaggle GPU launch submission RunPod launch or any external mutation
    - training or backbone fine-tuning
    - validation12 CV57 holdout audit or test access
    - final human approval of proposed cross-device tolerances
    - execution of H1 H2 or H3 graph gates
invariants:
  - The human-locked 2026-07-25 DAG and its falsification order remain authoritative.
  - H0 uses only already banked report values and never reruns a model.
  - H1-PREP generates no feature cache and stops at the exact human launch boundary.
  - Canonical Trackastra features are T4-generated in one pinned environment with exact T4-to-T4 repeatability.
  - M5-to-T4 tolerance values remain proposals requiring human approval; integer and geometry columns remain exact.
  - H1-PREP covers both 44b6 and 6bba and packages the frozen 27.46M checkpoint.
  - No forbidden validation12 or CV57 path is read.
  - Local CPU work observes zero-swap discipline and a 45-second clean idle preflight.
  - Only the root coordinator edits LEDGER.md; workers own disjoint outputs.
  - Exactly one ledger row is appended for H0 and exactly one for H1-PREP.
  - Banked *_v1 reports and bhap-submit are not modified.
quality:
  risk: high
  profiles:
    - core
  gates:
    - id: core.intent
      dimension: intent
      criterion: H0 and H1-PREP match the locked DAG, stop boundaries, ownership, and exact forbidden-action constraints.
      oracle: Review reports, manifests, ledger append, commits, and prohibited-path evidence.
      severity: high
      state: pending
      evidence: null
    - id: core.functional
      dimension: functional-correctness
      criterion: H0 deterministically computes all declared statistics and preregistered decision; H1 package validates both-family scope, option-C parity, assets, hashes, runtime, and human-run command without launch.
      oracle: Run focused unit tests and deterministic report/package validators.
      severity: high
      state: pending
      evidence: null
    - id: core.regression
      dimension: regression
      criterion: Existing Trackastra parity/cache tests and relevant repository integrity checks remain green and no protected artifact changes.
      oracle: Run focused legacy tests and inspect git diff/name-status.
      severity: high
      state: pending
      evidence: null
    - id: core.failure-paths
      dimension: failure-paths
      criterion: Missing assets, structural drift, tolerance breaches, holdout/CV57 paths, swap growth, dirty/unverified package state, and runtime-budget overflow fail closed.
      oracle: Unit-test synthetic drift and inspect package guards and report stop conditions.
      severity: high
      state: pending
      evidence: null
    - id: core.completion-review
      dimension: completion-review
      criterion: A skeptical final review finds no fabricated pair, unbanked model result, unauthorized tolerance, external launch, leakage path, ownership violation, or missing follow-on contract.
      oracle: Independent diff-and-evidence review after final mutation.
      severity: high
      state: pending
      evidence: null
    - id: core.independent-review
      dimension: independent-review
      criterion: An independent agent adversarially checks H0 evidence provenance/statistics and H1 package/launch boundary against the locked DAG.
      oracle: Independent review of final artifacts and tests with credible findings resolved.
      severity: high
      state: pending
      evidence: null
stall:
  quiet_turns: 3
  same_failure: 3
---
# Flight plan

Run H0 and H1-PREP concurrently with disjoint worker ownership. H0 owns its analysis
script, tests, and two reports; H1-PREP owns its comparator/package scripts, tests,
kernel package, and readiness reports. The root coordinator alone integrates the two
ledger rows and serializes one local commit per node. No external launch is in scope.
