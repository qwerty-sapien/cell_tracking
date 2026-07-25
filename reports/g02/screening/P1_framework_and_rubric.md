# P1 — Controller, rubric, and ledger

> Load `_shared_context.md` and `DAG.md`. P1/CTRL runs first and remains the sole
> writer of shared ledger and gate state.

## Controller responsibilities

1. Reconcile active G-02 authority and effect permissions.
2. Freeze exact source, scorer, split, prediction, package, model, and score identities.
3. Own `RUBRIC.md`, `LEDGER.md`, DAG state, submission accounting, and integrated
   calibration outputs.
4. Assign workers disjoint output paths; workers return evidence and do not edit shared
   controller files.
5. Keep scorer semantic parity (`P0A`) separate from transfer calibration (`P0B/CAL`).

## Evidence tiers

0. **LITERATURE** — before implementation or training, validate the candidate's primary
   evidence, measured-bottleneck mapping, expected >=3.5% improvement, M5 feasibility,
   falsification test, and stop/promotion criteria with
   `scripts/validate_model_candidate_gate.py`.
1. **SCREEN** — cheapest leakage-safe proxy capable of falsifying the mechanism. Prefer
   cached replay or a preregistered worst-slice screen. For learned candidates, use the
   guarded fit-only M5 lane before external compute.
2. **CONFIRM** — the appropriate full frozen local evaluation: grouped CV and, only when
   allowed by the active Plan, the untouched audit.
3. **BANK / PARK / HUMAN-DECISION**:

   - `>= +0.005`: BANK when graph, generalization, and runtime evidence also pass.
   - `+0.002 .. +0.005`: HUMAN-DECISION; preserve evidence and evaluate stacking value.
   - `< +0.002`: PARK by default.
   - DEAD only when compelling mechanism-level negative evidence rules out the direction.

The thresholds are repository decision heuristics, not literature-derived constants.

## Independent GPU gates

### GA — association

Requires:

- association-bound headroom on matched nodes;
- classical/cached association screens completed;
- a preregistered CPU/proxy improvement supporting the proposed compute;
- graph/runtime feasibility and a frozen training/evaluation identity.

### GD — detection

Requires:

- detection-bound evidence on the affected slice;
- a detection/front-end proxy showing recoverable node/edge benefit;
- runtime and downstream graph feasibility;
- a frozen training/evaluation identity.

Calibration family dependence is neither necessary nor sufficient for GA or GD.

Initial training may use at most 30 cumulative Kaggle T4 GPU-hours per model. Archive a
model that cannot become viable inside that ceiling. RunPod is unavailable before an
accepted score reaches `0.89`; any later higher-quality-GPU-dependent optimization needs
literature/theory-backed expected improvement greater than `5%`, must fit the `$10` total
cap, and requires explicit approval for the exact logged job. The ordinary hard
improvement gate is `3.5%`.

The local fast-fail lane is independent and may run alongside an eligible external job,
but only one local training process may own it. It requires a passing candidate manifest,
fit-only data, no holdout access, <=18 GiB training memory, >=6 GiB background reserve,
zero swap growth, and a preregistered timeout no longer than seven hours. Use
`scripts/run_local_training_guard.py`; a memory, swap, or time violation fails the local
runtime gate rather than triggering external escalation.

## Submission policy

- Maximum five Kaggle submissions per competition day; maximum two final selections.
- Calibration probes count against the cap.
- While transfer calibration is unresolved, approve at most one genuinely new,
  observation-distinct probe at a time.
- Record its prediction/package hash and predicted information gain before submission.
- Update the interval-aware calibration and remaining probe order after each returned
  score.
- Keep both final selections uncommitted until S1.

## LEDGER columns

`id | mechanism | tier | predicted delta | actual delta | local identity | observation class | Kaggle submissions | Kaggle GPU-hours | RunPod GPU-hours/$ | decision | reason | commit`

Ordinary commits and pushes to this private repository are permitted. No Kaggle
submission or RunPod launch occurs without explicit human approval.
