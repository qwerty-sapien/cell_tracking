# Division-ckpt N1 — free local falsifier for the frozen checkpoint

> Load `DAG.md`, `runs/N0_CHECKPOINT_PROVENANCE.md`, and the two fixed analytic
> split / non-split fixtures plus the `E_syn` contract in
> `../../2026-07-30-continuation/division/runs/N2_FALSIFIER.md`.
> CPU only, 0 GPU-h, fit-only, 45-second idle preflight, zero-swap rule.
> No Kaggle asset, no held-fold read, no campaign score.

## Objective / gate

Does the authenticated frozen checkpoint, loaded safely on CPU and run **inference-only**,
satisfy the preregistered mechanical contract `E_syn` on the two fixed analytic fixtures —
and does it do so deterministically, under the five-second step-0 envelope, with no swap
growth?

`E_syn` is unchanged from the N2 contract and is the conjunction of all four clauses:

```
E_syn = [Q(split) > Q(non-split)]
      ∧ [A(forward) > A(reverse)]
      ∧ [|Q − Q_swap-reflect| <= 1e-6]
      ∧ [forbidden-field schema rejection]
```

## Tasks

1. Strict safe load of the pinned weight on CPU: tensors only, no arbitrary code
   execution, and re-verify the SHA-256 at load time. Record parameter count and key count
   and compare them to N0's recorded values.
2. Implement the append-only evidence schema `A, S, U, Q = A x S x U` with its order
   checksum. The schema must **reject** any record carrying an action ID, current-pool
   rank or score, continuity result, oracle-bundle ID, GT field, threshold, or graph edit.
   Prove the rejection with a fixture, not a comment.
3. Evaluate all four `E_syn` clauses on both fixed analytic fixtures. Report each clause
   separately; a conjunction reported as a single boolean is not admissible.
4. Add the degeneracy checks a global-pooling failure would pass: a uniform-intensity
   input must tie, a spatially reversed input must not score as a split, and an empty
   foreground must fail closed rather than return a value.
5. Run a byte-identical repeat. Report guard runtime, peak RSS, and swap growth.
6. Write the step-0 preflight this lane will reuse remotely: pure stdlib, authenticating
   weight hash, source pin, fixture hashes, and fold-manifest disjointness **before** any
   dependency install or model load, completing inside five seconds.
7. Do not touch competition raw frames, held folds, the 38 repair bundles, the action
   population, or the scorer.

## Output

`runs/N1_FROZEN_INFERENCE_FALSIFIER.md` — per-clause `E_syn` results, degeneracy-check
results, schema-rejection fixture result, repeat determinism, step-0 timing, resource
guard, and a proposed (not written) LEDGER row.

## Verdict (preregistered)

- **GO**: all four `E_syn` clauses true on both fixtures, all three degeneracy checks
  behave as specified, the schema rejects every forbidden field, the repeat is
  byte-identical, step 0 completes `< 5 s`, and swap growth is `0 MiB`.
- **PARK**: the contract is intact but the environment blocks a clean safe load or the
  fixtures cannot be exercised as published. Do not improvise a replacement loader.
- **DEAD**: with valid pinned inputs any `E_syn` clause fails, a degeneracy check passes
  when it should not, the schema admits a forbidden field, or the repeat is not identical.
  This tested checkpoint is dead as a division representation; a different weight or
  observable is a new named mechanism.
- **ESCALATE**: the contract cannot be evaluated without a plan change.

## Cascade

- On **GO** → proceed directly to `N2_human_kaggle_pilot.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
