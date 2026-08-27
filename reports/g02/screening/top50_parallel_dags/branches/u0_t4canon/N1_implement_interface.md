# U0-N1 — Implement the fail-closed T4-native interface

> Load `DAG.md`, `results/U0_N0_contract_preregistration.md`, the frozen draft
> contract, and the existing H1 graph-runner tests. Do not run a graph score.

## Objective / gate

Does the implementation enforce all 12 U0 invariants and reject every forbidden
domain/split mutation? **GO** requires `100%` focused-test pass, no validation12/CV57/
holdout path in the executable interface, and a negative test proving an M5 decision
input cannot be supplied.

## Tasks

1. Implement a new U0 contract/loader namespace; do not modify the old N1 contract,
   evidence, status, or runner in place.
2. Require hashes for T4 arrays, atomic manifests, fit105 identities, folds, APNCC,
   code, checkpoint, graph reconstruction, and scorer.
3. Make M5 data an optional diagnostic report channel with no return path into fit,
   calibration, routing, graph construction, or verdict code.
4. Add positive fixtures for 36+69 coverage and exact T4 identities; add negative
   fixtures for a changed row, ID, fold, schema, hash, scorer, M5 decision argument,
   and forbidden split path.
5. Run focused unit/static tests only. Record commands, environment, hashes, runtime,
   RSS, and swap delta; do not evaluate held graphs.

## Output

Write `results/U0_N1_interface_verification.md` plus a hashed implementation manifest.
Include a proposed `U0-T4CANON-IMPLEMENT` ledger row (`IMPLEMENT / CONTROL`, `NOT
SCORED`, zero external usage) for the root coordinator.

## Verdict

- **GO** only on a `100%` focused-test pass, exact scope/hash pins, zero forbidden
  path exposure, and a passing M5-isolation negative test.
- PARK on a correctable implementation/test failure.
- DEAD if M5 values or forbidden splits are structurally required by the new runner.
- ESCALATE if compliance requires changing scorer, folds, data, or old evidence.

## Cascade

- On **GO** → proceed directly to `N2_exact_replay_rehearsal.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
