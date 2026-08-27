# N2 — Wave-1+2 ceiling unlock

> Load `DAG.md`, `runs/<run-id>/N1_fitonly_runtime_smoke.md`, `../../../LIVE_STATE.md`,
> and authenticated terminal reports for all Wave-1 association and Wave-2 division
> arms.

## Objective / gate

Is expensive representation work justified because cheaper Wave 1+2 cannot plausibly
reach local `0.925`? **GO** requires completed Wave-1+2 evidence and a frozen ceiling
audit with `maximum_credible_local_score <0.925`, where only graph-observed,
observation-distinct, non-double-counted gains are included and no unresolved cheaper
arm has a preregistered upper bound reaching `0.925`.

## Tasks

1. Authenticate all included scorer artifacts and enumerate completed, PARKed, DEAD,
   and unresolved cheaper arms; do not use proposal extrapolations.
2. Compute the ceiling with an explicit combination rule that forbids summing
   overlapping gains or oracle per-video arm selection.
3. Record best validated score, admissible remaining graph-observed headroom, final
   ceiling, uncertainty, and every exclusion. Freeze this audit before GPU approval.

## Output

Write `runs/<run-id>/N2_wave12_ceiling.md`, machine-readable arithmetic and source
hashes, and a proposed root-coordinator ledger row.

## Verdict

- **GO** — complete, reproducible evidence gives maximum credible local score `<0.925`
  and no cheaper unresolved arm can reach it.
- **PARK** — Wave 1+2 is incomplete, reaches `>=0.925`, or still has a cheaper
  preregistered path to that bar; F0 spend remains closed.
- **ESCALATE** — a ceiling cannot be computed without changing the locked objective or
  combining non-comparable scorers.

## Cascade

- On **GO** → proceed directly to `N3_gpu_cache_human_gate.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

