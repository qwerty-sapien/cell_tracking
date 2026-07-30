# Selector requirement on frozen Phase-1 44b6 actions

**Barrier:** `HUMAN_REVIEW_REQUIRED`. This is diagnostic-only: no tier, threshold, plan, ledger, DEAD, GPU, Kaggle, RunPod, submission, 6bba, or CV57 action occurred.

## Result

| Action class | Actions | Beneficial | Oracle ceiling Δ adjusted | Mean +0.005 requirement | Frozen gate verdict |
|---|---:|---:|---:|---|---|
| `delete` | 1,345,497 | 1,118 | +0.081897667 | AUC 0.85, retention 0.0204 | **TOO_LENIENT** |
| `occupied_swap` | 45,961 | 134 | +0.018571397 | AUC 0.90, retention 0.0679 | **TOO_LENIENT** |
| `restore_topology` | 50 | 0 | +0.000000000 | INFEASIBLE_ON_FROZEN_POPULATION | **NOT_APPLICABLE_POPULATION_LIMIT** |

Requirements are discrete over the preregistered AUC and power-of-two support grids; no interpolation was used. The detailed JSON contains all oracle points, all 20-seed curves, mean/spread summaries, per-fold AUC/retention, breakeven requirements, compiler receipts, and exact support counts.

## Governance consequence

At least one feasible action class needs a stricter selector than the frozen AUC 0.65 / retention 0.15 gate. An amendment is proposed, not applied. Execution halts here for human review.
