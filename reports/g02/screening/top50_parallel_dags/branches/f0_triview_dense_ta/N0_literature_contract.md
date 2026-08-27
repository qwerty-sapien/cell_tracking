# N0 — freeze the literature-to-representation contract

> Load `DAG.md`, `../../SHARED_CONTRACT.md`, the MIDL dense-embedding paper, the ASCENT
> ICCV paper, and the Trackastra ECCV paper.

## Objective / gate

Can one exact, licensed, point-compatible tri-view representation be derived without
claiming that the literature already validated it? **GO** requires a complete
hypothesis card, claim-to-source table, four frozen arms, and one encoder selected only
by license, input compatibility, availability, and runtime—not scorer performance.

## Tasks

1. Pin physical XY/XZ/YZ crop extents, anisotropy resampling, projection/compression,
   intensity normalization, coordinate channels, frozen encoder/version/weights,
   per-view linear compression, fusion point, compact-head size, folds, and seeds.
2. Separate supported claims (dense features can aid Trackastra; learned projection can
   compress 3-D fluorescence) from the unvalidated alpha claim (three orthogonal views
   preserve complementary cell identity here).
3. Freeze four arms: Trackastra baseline, XY-only dense beta, tri-view alpha, and
   deterministic non-XY permutation ablation. Cap trainable fusion parameters at
   `<=100k`; no backbone or encoder tuning.
4. Record licenses, model/data provenance, expected sign, null, 44b6 mechanism,
   deployment abstention behavior, graph thresholds, runtime stop, and human boundary.

## Output

Write `runs/<run-id>/N0_literature_contract.md`, a machine-readable feature contract,
weights/license manifest, and a proposed root-coordinator `LEDGER.md` row.

## Verdict

- **GO** — every claim and artifact is traceable, all four arms are fixed, and the
  proposed model is legally and technically usable under the campaign constraints.
- **PARK** — the hypothesis is grounded but no compatible frozen encoder/weights are
  locally available; name the exact evidence needed without downloading it.
- **DEAD** — the design requires full fine-tuning, segmentation retraining, held-score
  model selection, or another already-DEAD mechanism.

## Cascade

- On **GO** → proceed directly to `N1_fitonly_runtime_smoke.md`.
- On PARK / DEAD / ESCALATE / HUMAN-DECISION → halt and report.

