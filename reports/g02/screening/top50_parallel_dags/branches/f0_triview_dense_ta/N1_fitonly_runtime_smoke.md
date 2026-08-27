# N1 — fit-only identity and runtime smoke

> Load `DAG.md`, `runs/<run-id>/N0_literature_contract.md`, the authenticated fit-only
> registry, and local model weights. Do not use held labels or launch GPU/external work.

## Objective / gate

Can the frozen feature path preserve row identity and fit the hard resource envelope
before any full extraction? **GO** requires low/median/high density fit-only witnesses
to preserve 100% point/row identity, produce finite fixed-width embeddings, repeat
byte-identically on the same device, and project the 105-video cache to `<=30 T4-h`
with peak device memory `<=16 GiB`.

## Tasks

1. Select one fit-only video at each preregistered density quantile without outcomes;
   record selection and hashes.
2. Exercise crop/resample/projection/encoder/compression/fusion on a fixed point subset
   using local CPU or an already available local device only.
3. Repeat each witness; verify IDs, times, coordinates, view orientation, finite values,
   shapes, stable ordering, no GT fields, and exact same-device payload hashes.
4. Project time and memory conservatively from measured p95 per-point/per-frame cost.
   Report all failures; do not shrink crops or switch encoders after seeing a score.

## Output

Write `runs/<run-id>/N1_fitonly_runtime_smoke.md`, witness manifests and hashes,
runtime/RAM projection, test output, and a proposed root ledger row.

## Verdict

- **GO** — all three witnesses pass identity/repeat/finite/shape checks and projected
  full extraction is `<=30 T4-h`, `<=16 GiB`.
- **PARK** — identity is sound but runtime exceeds the cap; retain evidence and stop.
- **DEAD** — identity, leakage, orientation, deterministic repeat, or finite-value
  checks fail for the frozen contract.

## Cascade

- On **GO** → proceed directly to `N2_wave12_ceiling_gate.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

