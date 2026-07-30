# C2-KBEST-STABILITY terminal report

Date: 2026-07-27  
Terminal node: `C2-N0`  
Verdict: **PARK_DEPENDENCY**

## Cascade trace

`C2-N0 → PARK`; `C2-N1`, `C2-N2`, `C2-N3`, and `C2-N4` were not dispatched.

The K=8 membership/min-marginal rule, beta controls, causal hypothesis, null,
mechanism ablation, grouped split, graph endpoints, budgets, and kill thresholds
were frozen in:

- `../../../branches/c2_kbest_stability/results/C2_N0_preregistration.md`
- `../../../branches/c2_kbest_stability/results/C2_N0_contract.json`

The branch stopped because neither the separate owner decision
`LOCK U0-T4CANON` nor a frozen C0 edge/no-link assignment-cost contract exists.
The completed T4 cache and old H1 direct-swap contract are non-activating evidence;
the old H1 N1 remains DEAD.

## Usage and outcomes

- C2 implementation: not started
- Held-fit/graph/CV57 scores: none
- 6bba access: none
- GPU/Kaggle/RunPod/submission/push/external mutation: none
- Proposed ledger row: contained in `C2_N0_preregistration.md`; root coordinator only

## Resume condition

Resume at C2-N0—not N1—only after the owner records `LOCK U0-T4CANON` and the C0
contract pins the complete edge/no-link cost matrix, component/feasibility rules,
and same-cost rank-1/MAP output. The scientific rule and thresholds may not change.
