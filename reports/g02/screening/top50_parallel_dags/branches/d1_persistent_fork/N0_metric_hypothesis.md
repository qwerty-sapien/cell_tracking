# N0 — freeze the division hypothesis

> Load `DAG.md`, `../../SHARED_CONTRACT.md`,
> `../../../../division_sandbox_and_holdout_decomposition_v1.md`, the pinned official
> scorer source, and the Trackastra ECCV paper.

## Objective / gate

Can D1 be stated as one literature-grounded, scorer-faithful, locally killable
hypothesis without using held outcomes to choose the mechanism? **GO** requires a
complete shared-contract hypothesis card, exactly four frozen arms, and all graph
thresholds below recorded before any held graph is opened.

## Tasks

1. Pin scorer commit `075fc5f…`, scorer-file SHA-256 `0635c386…`, frozen Trackastra
   cache/contract hashes, official graph/scorer hashes, grouped folds, and seeds.
2. Freeze four arms: unchanged graph; D0 parental-softmax/greedy fork beta; D1
   one-frame-late daughter/grandchild-persistent alpha with one-to-one suppression;
   D1 with persistence evidence removed.
3. Define one fixed motif grammar: a parent-side node, exactly two distinct child
   branches, direct or one-frame-late timing, branch evidence through at most the
   grandchild stage, and one emitted fork per scorer-equivalence class.
4. State the causal hypothesis, null, crowding defense, leakage rules, budget, and
   required graph vector. Mark proposal/motif metrics diagnostic only.

## Output

Write `runs/<run-id>/N0_metric_hypothesis.md` plus a machine-readable preregistration
and a proposed `LEDGER.md` row for the root coordinator; do not edit the ledger.

## Verdict

- **GO** — all sources, hashes, arms, splits, endpoints, thresholds, and stops are
  frozen and internally consistent.
- **ESCALATE** — the scorer semantics or cache identity cannot be pinned.
- **DEAD** — the proposal materially collapses to an already-DEAD hand-tuned graph
  cost or from-scratch ranker.

## Cascade

- On **GO** → proceed directly to `N1_build_sandbox_validate.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

