# C2 assignment-evidence separation screen — terminal

Date: 2026-07-31

Population: frozen fit-only 44b6, 1,345,497 delete actions and 45,961 occupied
swaps, coverage 1.0

Verdict: **FAIL_PARK_C for both action classes; graph barrier closed**

## What was tested

The completed C0 contract uses the APNCC graph as the no-op assignment and the
frozen occupied-swap Trackastra challenger-minus-incumbent margin as the
label-blind action cost. Candidate swaps are decomposed into complete ownership
components and enumerated exactly. Each frozen action receives eight global
assignment diagnostics: MAP membership, exact force-in/forbid regrets, signed
min-marginal/reduced cost, K-best membership and rank-weighted stability, and
component/local competition sizes.

For a swap row, the binary event is selecting that swap. For a delete row, the
event is removal of its incumbent edge by a selected competing swap. This lets
DELETE be evaluated without inventing a label-tuned no-link cost. An edge that no
frozen candidate can remove receives the preregistered structural-infeasibility
sentinel.

Full population was feasible without the sampling amendment: the pinned
label-blind census contains 40,894 independent components, maximum size six and
maximum 21 feasible assignments. Feature generation took 27.7 CPU-seconds. It
read neither `oracle_label` nor `oracle_delta_S`.

A0 cyclically permuted complete A1 rows within each video, action class, and
source frame using the frozen SHA-256 order. It moved 1,391,313 rows; 145
single-row strata were necessarily fixed. The unchanged evaluator consumed the
already sealed B0 first, then A1 and A0.

## Per-fold results

DELETE is the primary class because its measured prize is approximately 4.41
times occupied swap's. Values below are AUC; lifts are absolute.

| Fold | B0 | A1 | A0 | AUC A1−B0 | AUC A1−A0 | A1 retention | Ret. A1−B0 | Ret. A1−A0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6650 | 0.6724 | 0.6661 | +0.0073 | +0.0063 | 0.00327 | 0 | 0 |
| 1 | 0.6957 | 0.7117 | 0.6943 | +0.0160 | +0.0174 | 0 | 0 | 0 |
| 2 | 0.5969 | 0.6387 | 0.5930 | +0.0418 | +0.0456 | 0.00599 | +0.00599 | +0.00599 |
| 3 | 0.6656 | 0.6786 | 0.6744 | +0.0130 | +0.0042 | 0 | 0 | 0 |
| 4 | 0.6130 | 0.6498 | 0.6153 | +0.0368 | +0.0345 | 0 | −0.00285 | 0 |
| **Mean** | **0.6473** | **0.6702** | **0.6486** | **+0.0230** | **+0.0216** | **0.00185** | **+0.00063** | **+0.00120** |

DELETE misses the binding AUC floor 0.85 in every fold, never reaches the
unchanged +0.05 lift over B0, and clears the +0.03 A0 lift only in folds 2 and 4.
Its retention is below 0.020394 in every fold and its retention lifts are also
below the unchanged +0.05 clauses. This is a genuine but insufficient
assignment-alignment signal, not an infrastructure failure.

| Fold | B0 | A1 | A0 | AUC A1−B0 | AUC A1−A0 | A1 retention | Ret. A1−B0 | Ret. A1−A0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6255 | 0.6166 | 0.6241 | −0.0090 | −0.0076 | 0.02941 | +0.02941 | +0.02941 |
| 1 | 0.8371 | 0.8253 | 0.8303 | −0.0117 | −0.0049 | 0 | 0 | 0 |
| 2 | 0.6223 | 0.6153 | 0.6298 | −0.0070 | −0.0145 | 0 | 0 | 0 |
| 3 | 0.6648 | 0.6335 | 0.6512 | −0.0313 | −0.0178 | 0 | 0 | 0 |
| 4 | 0.6560 | 0.6325 | 0.6101 | −0.0236 | +0.0223 | 0 | 0 | 0 |
| **Mean** | **0.6811** | **0.6646** | **0.6691** | **−0.0165** | **−0.0045** | **0.00588** | **+0.00588** | **+0.00588** |

Occupied swap misses the binding AUC floor 0.90 in every fold and regresses B0
in all five folds. It beats A0 only in fold 4 and misses the 0.067910 retention
floor in every fold.

All per-video CG0 false-positive budgets passed. Both classes also fail the
legacy and robust-p05 reference bars, so no threshold interpretation changes
the terminal outcome.

## Literature and transfer boundary

[Murty (1968)](https://doi.org/10.1287/opre.16.3.682) supports ranking
assignments in increasing cost; [Lawler
(1972)](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.401) supports the
general K-best partitioning transfer. [Wainwright, Jaakkola, and Willsky
(2005)](https://doi.org/10.1109/TIT.2005.856938) is primary MAP/LP literature
for reasoning about constrained optimal configurations. Here, exact
force-in/forbid cost differences are used only as binary-event MAP sensitivity
features. None of these sources validates the G-02 action costs, predicts
beneficial cell-tracking edits, or supplies evidence of graph gain.

## Barrier

`C2-KBEST-STABILITY-ASSIGNMENT-EVIDENCE-V1 × delete` and
`× occupied_swap` are **PARK-C**. No supplier-specific graph-gate
preregistration is unlocked. No graph run, 6bba, CV57, GPU, Kaggle, RunPod,
submission, tier move, or plan amendment occurred.
