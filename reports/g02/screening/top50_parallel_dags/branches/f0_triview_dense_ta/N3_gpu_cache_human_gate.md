# N3 — bounded dense-feature cache

> Load `DAG.md`, `runs/<run-id>/N0_literature_contract.md`,
> `runs/<run-id>/N1_fitonly_runtime_smoke.md`, `runs/<run-id>/N2_wave12_ceiling.md`,
> and `../../../LIVE_STATE.md`. **HUMAN-GATE: GPU/Kaggle/external launch.**

## Objective / gate

Can the full frozen feature cache be generated within the approved envelope and
authenticated for local graph work? Without a separate approval naming F0, the
platform, and the maximum resource, the only verdict is **HUMAN-DECISION**. After an
approved run, **GO** requires all 105 videos/rows present, exact identities and atomic
manifests, no GT fields, resource use within approval, and exact same-T4 repeats on one
44b6 and one 6bba witness.

## Tasks

1. Before launch, verify the explicit approval record, immutable source/data/weights
   hashes, projected bound, abort rule, output schema, and non-submittable packaging.
2. Generate only frozen embeddings; do not train or tune any arm and do not create a
   submission.
3. Authenticate completeness, identity, finite values, ordering, hashes, wall/GPU/RAM
   use, and two-family same-device repeatability.
4. On interruption or drift, stop; do not silently resume from unauthenticated parts.

## Output

Before approval write `runs/<run-id>/N3_gpu_approval_request.md`. After an approved
run write `runs/<run-id>/N3_cache_attestation.md`, atomic manifests, hashes, and a
proposed root ledger row.

## Verdict

- **HUMAN-DECISION** — readiness passes but explicit GPU/platform/resource approval is
  absent; halt before launch.
- **GO** — approval existed and the completed cache passes every attestation.
- **DEAD** — identity, leakage, repeatability, completeness, or resource cap fails.
- **PARK** — approved work stops cleanly before a complete cache, with no corrupted
  artifact promoted.

## Cascade

- On approved **GO** → proceed directly to `N4_44b6_graph_kill.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

