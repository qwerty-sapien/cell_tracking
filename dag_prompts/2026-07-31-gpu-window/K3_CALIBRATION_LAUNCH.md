# K3 — calibration launch and the projection gate (~0.3–0.6 GPU-h)

**Precondition: `K2` complete with a `PASS` static guard receipt.**
This is the first node in the window that debits GPU. It is deliberately small.

## Purpose

Prove the kernel reaches scientific code on real hardware, and produce an **authenticated
runtime projection** before committing the full run. No campaign kernel this week reached
scientific code at all — treat that as the base rate, not a formality.

## Preregistered abort

```
calibration_abort: "After two epochs on two folds, abort unless authenticated total
                    projection is <= 4.8 T4-h."
```

Under `H0` Option A this becomes `<= 8.0 T4-h` (5 folds × 2 seeds); the amendment receipt
must state the substituted number explicitly, pre-outcome. Under Option B it stays `4.8`.

The projection must be **authenticated** — measured wall-clock on the real T4, extrapolated
by the actual fold/seed/epoch counts — not an estimate. The precedent for why: a DINOv2 CPU
run passed step 0 in `0.053483 s`, then projected `38.662 h` and correctly failed closed.

## Sequence

1. **Pre-launch, via cmux:**
   - `kaggle quota --format json` — record `used_h` / `remaining_h` and the UTC timestamp.
   - Confirm **zero in-flight G02 kernels**. One authorized run at a time.
   - Re-run the launch-phase guard bound to the exact kernel ID:
     ```bash
     python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py \
       --kernel-dir <kernel_dir> --contract <contract.json> \
       --registry reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json \
       --phase launch --kernel-id <owner/exact-slug> \
       --output <GUARD_RECEIPT_LAUNCH.json>
     ```
     `KFG-15`/`KFG-16` exist because Kaggle silently accepted a run under a drifted slug.
     After the push, read back the **actual** kernel ID Kaggle assigned and assert it matches.

2. **Push exactly one version, via cmux.** `enable_gpu: true`. Then stop touching it. A
   second push while the first is `RUNNING` is the `0.74` GPU-h launch race.

3. **Watch for the step-0 receipt.** If step 0 fails, the run is terminal for that signature.
   Check `KAGGLE_FAILURE_REGISTRY.json`: a **same-signature retry is forbidden**. A new
   signature requires a new `KFG-N` family plus a regression fixture in the same change that
   fixes it, before any retry.

4. **Run 2 epochs × 2 folds.** Emit a timing manifest: per-epoch wall time, per-fold setup,
   parameter counts, the trainable/base ratio, device, and the extrapolated total.

5. **Apply the gate.** Projection over the ceiling → **terminal before score-bearing
   computation**. Quarantine every partial artifact. Do not retry with a smaller schedule to
   sneak under the line; that is a post-hoc design change.

6. **Post-launch, via cmux:** re-query quota. The **pre/post delta is the debit** — not your
   own timing. Update `tiered_surface/kaggle_budget.json` and append the debit entry.

## Discard rule

Calibration outputs are **timing evidence only**. No score, ranking, adapter checkpoint, or
partial prediction from this node may be carried into `K4` or reported as a result. The
campaign has already pre-emptively discarded `724,634` quarantined rows for exactly this
reason (MPS→CUDA equivalence unproven). Hold that line.

## Done when

Either a recorded terminal abort with the projection and its debit, or: step 0 `PASS`,
2×2 completed, authenticated projection **at or under the ceiling**, both guard receipts
hashed and cited, quota reconciled, calibration outputs quarantined. Then `K4` may open.
