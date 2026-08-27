# C2-N0 — Freeze one operations-research transfer

> Load `DAG.md`, the owner-locked U0 contract, the frozen C0 cost-source contract,
> Murty (1968), Lawler (1972), and the seeking-alpha novelty/rigor references.

## Objective / gate

Can one label-blind, graph-feasible stability rule be frozen before held data opens?
**GO** requires one cost source with unchanged coefficients, `K=8`, one exact rule,
two beta controls, and no hyperparameter or graph-cost sweep.

## Tasks

1. Verify U0 is owner-locked and hash the T4 arrays, C0 costs, component builder,
   feasibility constraints, APNCC control, folds, graph builder, and scorer.
2. Freeze component-local enumeration of the eight lowest-cost distinct feasible
   assignments. Components with fewer than eight solutions fall back to APNCC.
3. Freeze a whole-component change only when every introduced edge occurs in at least
   `7/8` assignments, every removed APNCC edge occurs in at most `1/8`, and exact
   constrained min-marginals favor both decisions beyond numerical tolerance
   `100 * eps * max(1, |best component cost|)`. Otherwise use APNCC unchanged.
4. Freeze controls: APNCC and C0's rank-1/MAP assignment under the identical cost
   source. State the null: stability adds no adjusted graph gain over the stronger
   control. Proposal AUC/precision/support never gate.
5. Preregister gates: 44b6 `<=0` DEAD; `0..+0.003` PARK; `>=+0.003` may test 6bba;
   only `>=+0.005` on both families opens CV57. Raw/node deltas must be `>=0`.

## Output

Write `results/C2_N0_preregistration.md` and a hashed C2 contract JSON. Include a
proposed `C2-KBEST-PREREG` ledger row (`LITERATURE / ALPHA`, `NOT SCORED`, zero
external usage) with citations, crowding verdict, thresholds, and exact verdict.

## Verdict

- **GO** only when all fixed choices and hashes above are present and neither labels
  nor a sweep can affect them.
- PARK if U0/C0 is not locked or the fixed costs are unavailable.
- DEAD if feasible assignments cannot be enumerated without changing graph costs.
- ESCALATE if the proposal requires a new scorer, split, cost sweep, or permission.

## Cascade

- On **GO** → proceed directly to `N1_implement_verify.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
