---
version: 2
status: complete
outcome: Correct CELLECT packaging and authentication handling, prove a byte-exact Kaggle round-trip via cmux without launching, and execute the frozen NODETOPO 44b6 gate
verify:
  - python -m pytest focused CELLECT packaging and NODETOPO tests
  - python focused artifact validators
  - cmux Kaggle list and byte-exact download verification
  - git diff and campaign-state review
scope:
  touchpoints:
    - reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/**
    - reports/g02/screening/tiered_surface/branches/sa_nodetopo/**
    - reports/g02/screening/LIVE_STATE.md
    - reports/g02/screening/LEDGER.md
    - tests/**
    - .cruise/ORIGIN.md
    - .cruise/FLIGHT.md
  non_goals:
    - No Kaggle kernel launch or submission
    - No DEAD promotion or plan amendment
    - No GPU run of five hours or more
    - No RunPod action
    - No authenticated Kaggle CLI from the agent shell
invariants:
  - Checkpoint discovery is content-addressed and fails closed before dependency installation
  - The pinned checkpoint SHA-256 is f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5
  - Kaggle stores an uncompressed tar or bin wrapper that round-trips the checkpoint bytes exactly
  - NODETOPO uses only the frozen 50 N1-valid 44b6 restoration actions and the preregistered adjusted/raw/node gate
  - Only the root coordinator writes LEDGER.md
quality:
  risk: high
  profiles:
    - core
    - backend-data
  gates:
    -
      id: core.intent
      dimension: intent
      criterion: Delivered changes and evidence satisfy all three CELLECT corrections and the frozen NODETOPO gate without crossing a hard stop
      oracle: Review the exact origin request against diffs, receipts, external actions, and final campaign state
      severity: high
      state: pass
      evidence: All three CELLECT corrections are implemented; private dataset v4 is READY and byte-exact; NODETOPO stopped at 44b6; no kernel, submission, GPU, DEAD, plan, or RunPod action occurred
    -
      id: core.functional
      dimension: functional-correctness
      criterion: The kernel finds only SHA-matching checkpoint bytes from pth, bin, or uncompressed tar and the NODETOPO evaluator produces the preregistered 44b6 metrics
      oracle: Run focused tests, artifact validators, and the CPU-only graph gate
      severity: high
      state: pass
      evidence: CELLECT 17/17 PASS; committed NODETOPO validator 14/14 and gate/scorer tests 12/12 PASS; KILL_44B6_STOP at adjusted/raw/node -0.000003436150368/0/0
    -
      id: core.regression
      dimension: regression
      criterion: Existing CELLECT package and frozen campaign contracts remain intact outside the requested corrections
      oracle: Run focused regression tests and review scoped diffs against existing manifests and pins
      severity: high
      state: pass
      evidence: Focused suites pass; scientific populations, labels, folds, representation, and B0/A1/A0 contract are unchanged
    -
      id: core.failure-paths
      dimension: failure-paths
      criterion: Missing, mismatched, ambiguous, unsafe, or malformed checkpoint containers fail closed before dependency installation
      oracle: Exercise negative fixtures for no match, duplicate matches, corrupt tar, unsafe member, and hash mismatch
      severity: high
      state: pass
      evidence: Resolver negative fixtures pass for missing, duplicate, compressed/corrupt, unsafe, and nonmatching candidates
    -
      id: core.completion-review
      dimension: completion-review
      criterion: A skeptical final review finds no unresolved in-scope gap or unauthorized external action
      oracle: Fresh final diff, artifact, command, and hard-stop review
      severity: high
      state: pass
      evidence: cmux v4 receipt verifies seven downloaded assets, 19,467,606 materialized bytes, and exact 10,197,221-byte checkpoint SHA f3e8e730...89ce5
    -
      id: core.independent-review
      dimension: independent-review
      criterion: An independent reviewer finds no unresolved defect in content-addressed discovery, round-trip proof, NODETOPO gating, or campaign-state updates
      oracle: Independent agent reviews the final scoped changes and evidence adversarially
      severity: high
      state: pass
      evidence: Independent review recomputed all CELLECT pins, reran 17/17 tests, confirmed hard stops and unchanged budget, and required the authoritative NODETOPO closure in this scoped final commit
    -
      id: backend.data-integrity
      dimension: data-integrity
      criterion: The staged and remotely downloaded checkpoint unwrap to the exact pinned bytes with preserved size and SHA-256
      oracle: Compare local tar member and cmux-downloaded dataset object extraction size and SHA-256
      severity: high
      state: pass
      evidence: Local USTAR is 10,199,040 B at a8110e0d...f559; authenticated cmux download recovered the exact 10,197,221-byte member at f3e8e730...89ce5
    -
      id: backend.security
      dimension: security
      criterion: Authenticated Kaggle operations use only the real cmux session and no credentials or private payload details leak into reports
      oracle: Review command provenance and output redaction boundaries
      severity: high
      state: pass
      evidence: Status, upload, list attempt, and authenticated SDK download ran only in cmux; no credential value was read or recorded; version 4 is READY
    -
      id: backend.concurrency
      dimension: concurrency
      criterion: Parallel NODETOPO work and CELLECT packaging do not race campaign records or overwrite unrelated dirty-worktree changes
      oracle: Root-only state reconciliation and final git diff ownership review
      severity: medium
      state: pass
      evidence: NODETOPO workers did not edit root records; concurrent committed lane 189c43b was preserved and reconciled by root into LIVE_STATE and LEDGER
    -
      id: backend.observability
      dimension: observability
      criterion: Content misses and round-trip mismatches report exact candidate provenance, sizes, and hashes without allowing a launch
      oracle: Inspect negative-test diagnostics and cmux verification receipt
      severity: medium
      state: pass
      evidence: Step-0 receipts expose candidate source/bytes/hash and fail closed; local verifier cannot claim authenticated remote success without cmux receipts
stall:
  quiet_turns: 3
  same_failure: 3
---
# Flight plan

First freeze repository state and identify the exact package/kernel and frozen NODETOPO evaluator inputs. In parallel, execute the CPU-only NODETOPO 44b6 gate while the root repairs and tests content-addressed checkpoint handling. Then package the authenticated local checkpoint inside an uncompressed tar with the safe ASCII member name, use only the authenticated cmux session to list, push, download, and verify the stored object, and stop before any kernel launch. Finally reconcile the gate outcome and verified packaging state into the root-owned campaign records and close all assurance gates with fresh evidence.

## Completion boundary

The owner refreshed the cmux Kaggle credential. Private asset version 4 is
READY and an authenticated cmux download proves the pinned checkpoint bytes
round-trip exactly after Kaggle expands the uncompressed tar. The CLI
`ListDatasetFiles` and metadata-preflight endpoints remain 403 under the fresh
OAuth credential, but the direct authenticated SDK download in the same cmux
session succeeded. No kernel launch, submission, GPU debit, RunPod action,
DEAD promotion, or plan amendment occurred.
