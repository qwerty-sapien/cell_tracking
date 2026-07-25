---
version: 2
status: blocked
outcome: Retry CGL D-14 fit-only on CPU and either produce the preregistered held-fit verdict or fail closed under a 12 GiB RSS ceiling and the existing 1.5 GiB swap ceiling.
verify:
  - .venv313/bin/python scripts/validate_model_candidate_gate.py --candidate reports/g02/model_candidate_cgl_tail_rank_head_v1.json --policy reports/g02/compute_governance_v1.json
  - .venv313/bin/python -m unittest -v tests.test_model_candidate_gate
  - jq consistency checks over the guard and final CGL reports
scope:
  touchpoints:
    - .cruise/ORIGIN.md
    - .cruise/FLIGHT.md
    - reports/g02/model_candidate_cgl_tail_rank_head_v1.json
    - reports/g02/screening/candgen/CGL_D14_learned_representation_fastfail.md
    - reports/g02/screening/candgen/cgl_d14_runtime_guard_amendment_preregistered_v3.json
    - reports/g02/screening/candgen/cgl_d14_fastfail_guard_amended_v3.json
    - reports/g02/screening/candgen/cgl_d14_fastfail_v1.json
    - reports/g02/screening/candgen/cgl_d14_fastfail_v1.md
    - reports/g02/screening/LEDGER.md
    - scripts/run_local_training_guard.py
    - scripts/run_cgl_d14_fastfail.py
    - src/celltrack_alpha/model_candidate_gate.py
    - tests/test_model_candidate_gate.py
  non_goals:
    - GPU or full T-03 training
    - Kaggle RunPod or competition submission
    - validation12 CV57 holdout audit or test access
    - graph replay or a scientific verdict when training and held-fit evaluation do not complete
invariants:
  - Process-tree RSS must remain at or below 12 GiB.
  - Cumulative swap growth must remain at or below 1.5 GiB.
  - System free memory must remain at or above 25 percent.
  - Wall time must remain at or below 360 minutes.
  - The frozen core compact head objective folds and preregistered retention gates remain unchanged.
  - Forbidden holdout and CV57 paths are rejected before read.
  - Unrelated dirty-worktree changes are preserved.
quality:
  risk: medium
  profiles:
    - core
  gates:
    -
      id: core.intent
      dimension: intent
      criterion: The retry uses no more than 12 GiB RSS and preserves every existing scientific and external-action restriction.
      oracle: Review the amended manifest preregistration guard telemetry and final report.
      severity: high
      state: pending
      evidence: null
    -
      id: core.functional
      dimension: functional-correctness
      criterion: The guarded retry either completes full-pool held-fit evaluation or records the exact fail-closed boundary without fabricating retention.
      oracle: Validate the guard and final JSON reports with jq assertions.
      severity: high
      state: pending
      evidence: null
    -
      id: core.regression
      dimension: regression
      criterion: Candidate validation legacy zero-swap behavior and the CGL-specific bounded amendment remain tested.
      oracle: Run tests.test_model_candidate_gate and Python compilation.
      severity: high
      state: pending
      evidence: null
    -
      id: core.failure-paths
      dimension: failure-paths
      criterion: RSS swap free-memory and timeout breaches all remain fail-closed at their exact boundaries.
      oracle: Review strict comparisons unit tests and observed guard stop behavior.
      severity: medium
      state: pending
      evidence: null
    -
      id: core.completion-review
      dimension: completion-review
      criterion: A skeptical final review finds the contract hash telemetry metrics cascade and prohibited-action statements mutually consistent.
      oracle: Final diff artifact and hash review.
      severity: high
      state: pending
      evidence: null
stall:
  quiet_turns: 3
  same_failure: 3
---
# Flight plan

Retry the deterministic CGL D-14 CPU fast-fail under a newly hashed contract whose
process-tree RSS ceiling is 12 GiB. Preserve the already authorized 1.5 GiB cumulative
swap-growth ceiling and every scientific leakage and external-compute restriction.
Produce the preregistered GO/PARK/DEAD result only if feature assembly, compact-head
training, and full-pool held-fit evaluation complete. Otherwise fail closed and record
the exact runtime boundary without cascading scientifically.

## Completion evidence and assurance blocker

The guarded computation completed with verdict DEAD. Focused validation, eight unit
tests, Python compilation, JSON invariants, full-pool counts, artifact hashes, guard
ceilings, and prohibited-action assertions all passed. The final and preserved
post-training head artifacts are byte-identical.

Cruise assurance attestation cannot produce a context fingerprint because its workspace
fingerprinter refuses more than 64 MiB of untracked files; this pre-existing research
workspace has approximately 4.0 GiB untracked. Unrelated artifacts were not deleted,
staged, or ignored to bypass that protection. This blocks only Cruise's completion
receipt, not the completed and independently hash-verified CGL report.
