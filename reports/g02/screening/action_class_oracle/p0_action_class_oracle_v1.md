# P0 action-class oracle headroom

**COMPLETE — DIAGNOSTIC ONLY / NO TRAINING**

Control: the exact fit117 c=0.25 fragment graphs used by CG0. These are
not the banked APNCC incumbent, so the table ranks action surfaces but does
not claim additive headroom over local 0.881522.

## 44b6

Control score: **0.864116** over 41 videos.

| action | oracle score | Δ score | Δ adjusted | Δ raw | Δ node | Δ division J | positive actions | raw background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| add | 0.864116 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 4,022,839 |
| delete | 0.952890 | 0.088774 | 0.081016 | 0.080613 | 0.000000 | 0.077576 | 1,334,426 | 11,071 |
| swap | 0.939944 | 0.075828 | 0.075828 | 0.075448 | 0.000000 | 0.000000 | 393 | 4,022,839 |
| gap_bridge | 0.867755 | 0.003639 | 0.003639 | 0.003642 | 0.001180 | 0.000000 | 14 | 10,659,021 |
| division_fork | 0.878839 | 0.014723 | 0.001390 | 0.001376 | 0.000000 | 0.133333 | 10 | 5,159,369 |
| node_add | 0.864116 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 61 | N/A |
| node_restore | 0.873928 | 0.009812 | 0.009794 | 0.009766 | 0.005540 | 0.000180 | 61 | N/A |

### Edge-FP-equivalent budgets

| action | target 0.895 | target 0.9255 | target 0.9422 |
|---|---:|---:|---:|
| add | 0 | 0 | 0 |
| delete | 770 | 352 | 135 |
| swap | 617 | 191 | 0 |
| gap_bridge | 0 | 0 | 0 |
| division_fork | 0 | 0 | 0 |
| node_add | 0 | 0 | 0 |
| node_restore | 0 | 0 | 0 |

Budgets are extra edge-FP equivalents with TP/FN/node/division held
fixed. They are not interchangeable with false forks or false nodes.

## 6bba

Control score: **0.854532** over 76 videos.

| action | oracle score | Δ score | Δ adjusted | Δ raw | Δ node | Δ division J | positive actions | raw background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| add | 0.854532 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 3,780,767 |
| delete | 0.929372 | 0.074840 | 0.072067 | 0.072914 | 0.000000 | 0.027729 | 1,356,863 | 60,039 |
| swap | 0.914267 | 0.059735 | 0.059730 | 0.060826 | 0.000000 | 0.000049 | 1,737 | 3,780,767 |
| gap_bridge | 0.858956 | 0.004424 | 0.004424 | 0.004562 | 0.001351 | 0.000000 | 94 | 10,575,460 |
| division_fork | 0.875937 | 0.021405 | 0.001271 | 0.001287 | 0.000000 | 0.201340 | 62 | 4,568,879 |
| node_add | 0.854532 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 878 | N/A |
| node_restore | 0.878572 | 0.024040 | 0.020539 | 0.021277 | 0.013616 | 0.035014 | 878 | N/A |

### Edge-FP-equivalent budgets

| action | target 0.895 | target 0.9255 | target 0.9422 |
|---|---:|---:|---:|
| add | 0 | 0 | 0 |
| delete | 2,474 | 269 | 0 |
| swap | 1,441 | 0 | 0 |
| gap_bridge | 0 | 0 | 0 |
| division_fork | 0 | 0 | 0 |
| node_add | 0 | 0 | 0 |
| node_restore | 0 | 0 | 0 |

Budgets are extra edge-FP equivalents with TP/FN/node/division held
fixed. They are not interchangeable with false forks or false nodes.

## Interpretation constraints

- Every transformed graph passes the submission graph validity contract.
- Labels are used deliberately for an upper-bound diagnostic; no result is
  eligible for model selection, CV57, or submission.
- `node_add` and `node_restore` have no honest raw-background budget because
  the banked record contains no node proposal generator.
- P1 feasibility compilation and P2 division fitting remain unopened.
