# R0 N2 — build and leakage-safe dry validation

> Load `DAG.md` and self-hashed GO-qualified N0/N1 outputs. Authenticate complete
> representation coverage. Do not open held graph outcomes.

## Objective / gate

Does the literal R0 implementation reproduce the frozen population, folds, nnPU risk,
groupwise p-values/FDR decisions, explicit no-link semantics, and admission budgets
deterministically? **GO** requires all tests twice with byte-identical decisions.

## Tasks

1. Implement only the single frozen no-link head, cross-fit nnPU estimator, fixed
   groupwise calibration, BY correction, and immutable admission rule.
2. Unit-test nnPU against analytic toy risks, non-negative clipping, class-prior bounds,
   no-link polarity, stable group/backoff assignment, conformal p-value orientation,
   BY monotonicity, per-video and family caps, and zero unlabeled-as-negative labels.
3. Prove train/held video isolation and reject any validation12/CV57/test path in all
   fitting, prior, calibration, and threshold interfaces.
4. Replay the upstream association graph byte-for-byte when shell admission is disabled.
   Prove exact candidate identity and deterministic one-to-one conflict handling.
5. Run two no-scorer dry passes; record artifact hashes, runtime, RSS/swap, total and
   per-group admitted support. Support is diagnostic until N3.

## Output

Write `outputs/r0_n2_build_validation.{md,json}`, tests, hashes, and proposed LEDGER row
`R0-TA-PU-SHELL-N2 | IMPLEMENT | tests/repeats | no graph score | 0 external usage |
verdict | reason`. Do not edit LEDGER; update LIVE_STATE after the node verdict.

## Verdict

- **GO** — all tests pass, repeated outputs are byte-identical, disabled-shell replay is
  exact, and every admission satisfies the frozen caps.
- **PARK** — a bounded implementation defect exists without mechanism-level evidence.
- **DEAD** — nnPU/control semantics or deterministic full-pool execution are
  intrinsically infeasible.
- **ESCALATE** — implementation would require changing N1, population identity, or
  scorer/governance.

## Cascade

- On **GO** → proceed directly to `N3_44b6_graph_falsification.md` in the same run; do
  not wait for a human turn.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
