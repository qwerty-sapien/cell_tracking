# Proposed LEDGER row — Phase-2 representation separation gate

Root coordinator only may copy or adapt this row after human review. It has not been appended
to `reports/g02/screening/LEDGER.md`.

| Bet | Mechanism | Stage | Frozen gate | Result | Evidence / scope | GPU h | Submissions | Verdict | Next |
|---|---|---|---|---|---|---:|---:|---|---|
| `R-SEPARATION-FALSIFIER-44B6` | supplier-agnostic fit-only representation falsifier over exact Phase-1 N1-valid delete + occupied-swap actions | Phase-2 preregistration barrier | For both classes in every grouped OOF fold: A1 AUC ≥0.65 and ≥B0+0.05 and ≥A0+0.03; A1 retention at exact per-video CG0 FP budget ≥0.15 and ≥B0+0.05 and ≥A0+0.03; 100% coverage, zero FP-budget violations, zero invalid/duplicate/nonfinite rows | **FROZEN / NOT RUN** — historical `~0.502` AUC is a planning anchor only; B0 must be recomputed and sealed on the identical population before A1/A0 opens | `PREREGISTRATION_v1.json`; P0 `8ad548c1…9c06`; folds `ddb313cb…8769`; N1 `2a72950c…f865`; fit-only; no graph/6bba/CV57/cache/GPU/external action | 0 | 0 | HOLD FOR HUMAN REVIEW; no tier move | PASS unlocks only a supplier-specific 44b6 graph contract. FAIL parks that supplier instantiation PARK-C and forbids its full-pool cache spend. |

Trackastra's fit AUC `0.918491` did not prevent its later 44b6 graph failure, so this gate is
explicitly necessary and never sufficient. `uncited_tier_moves = 0`.
