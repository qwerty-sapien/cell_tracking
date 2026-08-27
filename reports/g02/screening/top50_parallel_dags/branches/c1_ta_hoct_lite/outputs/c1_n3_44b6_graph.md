# C1 N3 — 44b6 graph falsification

**Verdict: DEAD** — adjusted edge-Jaccard delta is at or below zero.

## Exact graph vector

| Comparison | Adjusted Δ | Raw Δ | Node-recall Δ |
|---|---:|---:|---:|
| C1 − APNCC | -0.007602934502 | -0.007591069107 | -0.000185393073 |
| C1 − C0 | -0.007602934502 | — | — |

`epsilon_null=1e-12`; alpha floor `0.001`.

## Counts

| Arm | TP | FP | FN | adjusted | raw | node recall |
|---|---:|---:|---:|---:|---:|---:|
| APNCC | 10327 | 958 | 687 | 0.866534526036 | 0.862596057467 | 0.992618000284 |
| C0 | 10327 | 958 | 687 | 0.866534526036 | 0.862596057467 | 0.992618000284 |
| C1 | 10284 | 1014 | 730 | 0.858931591534 | 0.855004988360 | 0.992432607211 |

## Proposal diagnostics (non-gating)

All-fit105 held accepted `1211` / `1217`; precision `0.975227`. On 44b6
specifically, the frozen fold thresholds accepted `294` / `295` labeled proposals
with precision `0.952381` (`280` correct / `14` harmful); held-fold precision was
below `0.95` in folds 2 and 3. Full held 44b6 score AUC was
`0.5020335536349771`. These metrics are diagnostic only; the machine-readable
supplement is `c1_n3_44b6_proposal_diagnostic.json`.

## Integrity

- APNCC and C0 were scored from the same authenticated fragments; exact-repeat null: `True`.
- Observation-distinct C1 fragments: `36` of `36`.
- One configuration, one fixed seed, CPU only; no adaptive retry.
- No validation12, CV57, GPU, Kaggle, RunPod, paid compute, or submission access.

## Proposed LEDGER row

`C1-TA-HOCT-LITE-44B6 | SCREEN | >=+0.005 and >=+0.001 over beta | exact score vector | identity/observation class | 0 external usage | DEAD | adjusted edge-Jaccard delta is at or below zero`
