# Recall N2 — cheap local falsifier for candidate supply

> Load `DAG.md`, `runs/N0_POOL_RECALL_CEILING.md`, `runs/N1_MECHANISM_AND_CROWDING.md`,
> and LIVE_STATE. CPU only, fit-only data, zero-swap rule, 45-second idle preflight.
> No GPU, Kaggle, asset, training, held-fold read, or graph run.

## Objective / gate

On a frozen **fit-only** slice, does the N1 mechanism actually nominate N0's pool-absent
true edges **without labels**, at a candidate-set inflation the frozen scorer could
survive? This is the free test that must precede any compute request.

## Tasks

1. Freeze the slice before running: named fit-only videos, frame range, and the exact
   pool-absent target subset restricted to that slice. No held fold is touched. Write the
   slice manifest and its hash first.
2. Implement the minimum viable form of the N1 mechanism — the smallest thing that can
   nominate candidates. Do not train, do not tune, do not fit a threshold on the target
   set. Every free parameter must be either source-published, fixed a priori and recorded,
   or set from fit-side statistics that never see the target labels.
3. Measure and report:
   - **recovery** — fraction of the slice's pool-absent true edges nominated;
   - **inflation** — nominated candidates as a multiple of the frozen pool's candidate
     count on the same slice;
   - **shell discrimination** — recovery against the 10–12 µm background admitted, since
     separating `456` correct edges from `5.34M` background is the actual difficulty;
   - determinism: a byte-identical repeat run.
4. Run the negative fixtures that must fail closed: a permuted-frame input, a
   label-shuffled target, and an empty-foreground input. A mechanism that "recovers" edges
   under a permuted frame is measuring geometry, not correspondence.
5. Report guard runtime, peak RSS, and swap growth. Any swap growth invalidates the run.
6. Do not compute a graph delta, open a held fold, or estimate a score gain. Recovery is
   not score; the campaign has already proven proposal metrics do not predict the graph.

## Output

`runs/N2_SUPPLY_FALSIFIER.md` — slice manifest hash, recovery, inflation, shell
discrimination, negative-fixture results, repeat determinism, resource guard, and a
proposed (not written) LEDGER row.

## Verdict (preregistered)

- **GO**: recovery `>= 50%` of the slice's pool-absent true edges at `<= 3x` candidate-set
  inflation, all three negative fixtures fail closed, repeat byte-identical, `0 MiB` swap
  growth.
- **PARK**: recovery `>= 20%` but either below `50%` or above `3x` inflation. The
  mechanism has signal but not at a usable operating point; halt rather than tune.
- **DEAD**: recovery `< 20%`, any negative fixture passes when it should fail, or the
  repeat is not byte-identical. The tested realization is dead; a different observable
  would be a new named mechanism.
- **ESCALATE**: the mechanism cannot be exercised at all without held data, GT at decision
  time, or a plan change.

## Cascade

- On **GO** → proceed directly to `N3_human_kaggle_gate.md` in the same run.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report; do not advance.
