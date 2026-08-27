# R0 N1 — freeze population, risk semantics, and cache boundary

> Load `DAG.md`, GO-qualified `outputs/r0_n0_dependency_novelty.{md,json}`, the CG0
> budget artifact, exact fit105 candidate manifests, the shell reachability reports,
> and the nnPU/CRC/BY primary papers cited in `DAG.md`.

## Objective / gate

Can R0 be frozen as a statistically honest, fit-only test over the **complete**
unlabeled 10–12 µm occupied-endpoint deployment population? **GO** requires exact row
coverage, a fixed PU/class-prior protocol, valid groupwise calibration assumptions, an
explicit no-link decision, exact FP budgets, one architecture, and no external compute
need. Missing representation coverage returns HUMAN-DECISION before enrichment.

## Tasks

1. Hash/count every fit105 positive and unlabeled candidate by family, video, time,
   endpoint occupancy, radius band, and stable candidate ID. Prove unlabeled rows were
   never relabeled negative and no validation12/CV57 identity was accessed.
2. Audit frozen upstream representation coverage for every shell row. If incomplete,
   write an exact enrichment manifest, runtime/memory estimate, hashes, and commands,
   but launch nothing.
3. Freeze five video-grouped cross-fit folds; nnPU class-prior estimator and clipping
   rule; one explicit link/no-link head; one seed (`20260727`); one schedule; and fixed
   groups `family × radius-band × preregistered density stratum`, with a fixed
   minimum-sample backoff hierarchy chosen before outcomes.
4. Define valid calibration evidence. Conformal null p-values must be demonstrably
   super-uniform within each fixed group under stated exchangeability; BY controls FDR
   across dependent candidate tests. If PU data cannot support that null, forbid an FDR
   claim and PARK—do not call a heuristic threshold “conformal.”
5. Freeze the admission rule, exact CG0 family FP budgets, per-video cap, scorer null
   `epsilon_null`, and graph gates: total adjusted delta vs APNCC `>=+0.005`, incremental
   adjusted delta vs the upstream no-shell arm `>epsilon_null`, raw/node deltas vs that
   arm `>=0`, and no budget breach.

## Output

Write self-hashed `outputs/r0_n1_population_preregistration.{md,json}` with coverage,
assumptions, folds, groups, budgets, full fixed method, permission needs, and proposed
LEDGER row `R0-TA-PU-SHELL-N1 | PREREGISTRATION/COVERAGE | result | full-pool identity |
0 external usage | verdict | reason`. Do not edit LEDGER; update LIVE_STATE after the
node verdict.

## Verdict

- **GO** — complete representation coverage already exists and the statistical/model
  contract is valid and fully fixed.
- **HUMAN-DECISION** — coverage is incomplete and the exact prepared enrichment would
  require GPU, upload, paid compute, push, or another outward mutation; halt before it.
- **PARK** — nnPU prior or super-uniform/exchangeability evidence is not supportable, or
  a required fit-only field is absent without a justified enrichment path.
- **DEAD** — full-pool burden or identifiability makes the fixed mechanism impossible
  even before training.
- **ESCALATE** — the scorer, population, budget, or upstream contract must change.

## Cascade

- On **GO** → proceed directly to `N2_build_dry_validate.md` in the same run; do not wait
  for a human turn.
- On HUMAN-DECISION → halt with the exact enrichment manifest; approval must be explicit
  and N1 rerun must authenticate the completed cache before advancing.
- On PARK / DEAD / ESCALATE → halt and report; do not advance.
