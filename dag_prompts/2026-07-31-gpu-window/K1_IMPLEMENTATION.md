# K1 — PEFT adapter implementation and local verification (CPU-only, zero GPU)

**Precondition: `H0` decided and its amendment/contract receipt written. `K0` complete.**
If `H0` is unresolved, stop and say so. Do not pick a fold contract.

## What you are building

A LoRA (or bottleneck) adapter over the **frozen** Trackastra association
transformer/head, fit against the frozen DELETE action population, selected on the
held-fit frozen-scorer **graph adjusted-edge delta** under nested OOF.

Frozen, immutable, must not be touched:

| Object | Identity |
|---|---|
| Trackastra backbone | `24c290ce…981f`, `27,456,880` params — **all base weights frozen** |
| DELETE population | `4ab7bb23…76f0c`, `1,345,497` actions / `1,118` beneficial / `1,344,379` harmful |
| Sealed B0 | `a9ce8c0b…b4ff2` |
| Frozen gate | `9fb030bd…3310e` |
| N1 feasibility compiler | `457f6607…24f5` (implementation), `2a72950c…f865` (interface) |
| Frozen scorer | unchanged; no reimplementation, no re-derivation |
| Outer split | per `H0` |

## Hard constraints from the preregistration

```
imaging_backbone_trainable        false
full_finetune                     false
trainable_parameter_fraction      strictly < 0.01   (hard abort if ratio >= 0.01)
optimizer                         AdamW
adapter_learning_rate             3e-4
backbone_learning_rate            0.0
schedule                          10% linear warmup then cosine decay
weight_decay                      0.01
dropout                           0.1
gradient_clip_norm                1.0
max_epochs                        24
early_stopping_patience_epochs    4
layer_wise_learning_rates         true
ensemble                          2 fixed seeds per fold; average fold/seed predictions
selection_metric                  outer held-fit graph adjusted-edge delta — NEVER proposal loss
information_leakage_exact         0
```

The trainable/base ratio check must run **before** the optimizer is constructed and abort
the process on `>= 0.01`. `27,456,880 × 0.01 = 274,568` — the adapter must have strictly
fewer trainable parameters than that. Print the exact count and ratio into the receipt.

## Method

Write tests first. This lane has no room for a bug discovered on Kaggle.

1. **Backbone load.** Reuse `scripts/trackastra_checkpoint_loader.py` if its contract fits;
   read it before writing anything new. Strict, safe, tensor-only load. Assert the loaded
   parameter count is exactly `27,456,880` and the file hash matches.
2. **Freeze proof.** A test that asserts `requires_grad is False` for **every** base
   parameter after adapter injection, and that the set of trainable tensors is exactly the
   adapter's. Not a spot check — enumerate.
3. **Adapter.** LoRA on the association transformer/head projections, or a bottleneck
   adapter. Keep it small enough to clear the 1% wall with margin.
4. **Leakage firewall.** A test proving the training path for outer fold *k* reads no
   action, video, or feature belonging to fold *k*'s held side, and never reads GT repair
   bundles, the scorer's outcome fields, or B0/A1/A0 outputs. `information_leakage_exact: 0`
   is a preregistered kill clause — make it mechanically checkable, and emit a firewall
   receipt counting exactly what was read.
5. **Selection.** Inner loop may use whatever proxy it likes for early stopping *within* a
   fold, but the **outer** selection and the reported result are the frozen-scorer graph
   adjusted-edge delta. Compile edits through the frozen N1 compiler; never hand-build a graph.
6. **Determinism.** Fixed seeds, deterministic dataloader ordering, recorded torch/numpy
   versions. Two runs of the same config must produce identical adapter hashes on CPU.
7. **Local dry run.** One fold, tiny epoch cap, CPU only, under the 45-second idle preflight
   and the zero-swap rule. Prove the whole loop closes: load → adapt → fit → score actions →
   compile through N1 → frozen scorer → adjusted/raw/node delta. Quarantine and discard the
   dry-run outputs; they are timing/plumbing evidence only, never a result.

## Do not

- Do not co-load V45 and Trackastra in one process. That is the original swap deadlock.
- Do not tune any threshold, graph cost, or the CG0 FP budget (`261` for delete).
- Do not touch `6bba` or CV57.
- Do not use proposal loss, AUC, or retention as the outer selection metric. They are
  reportable, never selective — that is the `H0`-era proposal-screening ban.
- Do not "fix" a disappointing dry-run number by changing a frozen field.

## Done when

Focused tests pass with real output pasted into the report; the freeze proof, ratio check,
and leakage firewall each have a receipt; the CPU dry run closed the full loop at `0 MiB`
swap growth; and no frozen artifact changed hash. Report the exact trainable count and ratio.
