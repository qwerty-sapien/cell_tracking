# U0-N2 — Exact T4-native replay rehearsal

> Load `DAG.md`, U0-N0/N1 outputs, the authenticated T4 cache, and the new
> implementation manifest. This is a local replay test, not a graph-score experiment.

## Objective / gate

Are two clean, isolated CPU fits and graph reconstructions from the same T4 inputs
byte-identical? **GO** requires equality of every fitted parameter, calibration
decision, selected row/edge ID, reconstructed graph payload, and manifest hash, with
APNCC reproduced exactly when the challenger is disabled.

## Tasks

1. Run the 45-second clean-idle/zero-swap preflight and fail closed on any swap growth.
2. Execute two fresh-process U0 replays with the frozen seed and environment. Do not
   call the scorer or reveal any adjusted/raw/node metric.
3. Compare arrays bytewise and manifests/hash sets exactly; compare no floats under a
   new tolerance. Confirm family counts remain 36/69 and no forbidden identity opened.
4. Run the challenger-disabled control and require exact APNCC graph payload identity.
5. Record wall time and peak RSS as feasibility diagnostics, never promotion metrics.

## Output

Write `results/U0_N2_exact_replay.md` and an exact replay-attestation JSON. Include a
proposed `U0-T4CANON-REPLAY` ledger row (`CONFIRM / INTEGRITY`, `NOT SCORED`, zero
external usage) for the root coordinator.

## Verdict

- **GO** only for exact two-run equality across all named payloads and exact disabled-
  challenger APNCC identity.
- PARK on resource/preflight failure before a complete comparison.
- DEAD on authenticated T4 nondeterminism, contamination, identity drift, or control
  mismatch.
- ESCALATE if exact replay would require changing the frozen environment or contract.

## Cascade

- On **GO** → proceed directly to `N3_human_lock.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
