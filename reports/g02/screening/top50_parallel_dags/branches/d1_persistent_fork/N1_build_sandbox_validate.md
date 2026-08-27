# N1 — build and dry-validate the fixed decoder

> Load `DAG.md`, `runs/<run-id>/N0_metric_hypothesis.md`, the official scorer fixtures,
> and the frozen Trackastra cache schema. Do not open held-fit graph outcomes.

## Objective / gate

Does the fixed decoder reproduce the scorer's local fork semantics, preserve graph
invariants, and eliminate equivalent duplicate forks deterministically? **GO** requires
all 14 official sandbox classifications to match the pinned scorer, all 51 local
counterfactuals to preserve declared invariants, zero duplicate survivors after
suppression, and byte-identical repeated dry-run outputs.

## Tasks

1. Implement only the frozen D0, D1, and persistence-ablation contracts; train at most
   a compact head on frozen pretrained features with five grouped fit-only folds.
2. Add tests for direct, one-frame-late, grandchild-supported, spurious, duplicate,
   and successive forks plus in-degree `<=1`, out-degree `<=2`, DAG, and time direction.
3. Prove one-to-one suppression is applied before graph emission using a stable
   score/key ordering and record every suppressed alternative.
4. Run a fit-only leakage audit and deterministic repeat; do not graph-score 44b6,
   6bba, or CV57.

## Output

Write `runs/<run-id>/N1_decoder_validation.md`, full test output, arm manifests, hashes,
and a proposed root-coordinator `LEDGER.md` row.

## Verdict

- **GO** — 14/14 scorer fixtures agree, 51/51 counterfactual invariants pass, no
  duplicate survives, repeats are identical, and leakage audit passes.
- **PARK** — the implementation is valid but the fixed pretrained feature schema lacks
  a required motif field and a bounded non-held repair is named.
- **DEAD** — scorer equivalence, graph invariants, duplicate suppression, determinism,
  or leakage fails.

## Cascade

- On **GO** → proceed directly to `N2_association_dependency.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

