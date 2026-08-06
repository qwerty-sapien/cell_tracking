# C2 — launch (autonomous, spends the owner's real GPU quota)

**Precondition: `C1` verified round-trip. `C0` lock available.**
Launch is **pre-authorized** for this node per `AGENTS.md` — do not stop to ask. The gates
below are what that authorization explicitly told you to preserve.

## Budget for this lane

| Field | Value |
|---|---:|
| Approved projection (H0 Option A) | `8.0` T4-h |
| Calibration abort ceiling | `<= 8.0` T4-h authenticated projection |
| Quota at last authenticated query | `5.02` used / `24.98` remaining of `30.0` |
| Quota `refresh_at` | `2026-08-01T00:00:00` |

`8.0` h exceeds the `<5` h auto-ceiling and is covered by the recorded owner decision in
`PREREGISTRATION_AMENDMENT_A.json`. **Re-query anyway** — the balance moved twice on
2026-07-31 alone, and the authoritative number is whatever cmux returns now, not the mirror.

**Quota expiry is explicitly not a reason to launch** (`AGENTS.md`). If a gate fails, let the
hours expire.

## Two launches, in this order

### C2a — calibration

2 epochs × 2 folds. Purpose: prove the kernel reaches scientific code on real hardware and
produce an **authenticated** runtime projection (measured wall clock extrapolated by real
fold/seed/epoch counts — not an estimate).

Abort unless the projection is `<= 8.0` T4-h. **Do not shrink the schedule to fit under the
line** — that is a post-hoc design change. Calibration outputs are timing evidence only:
quarantine them, and never carry a score, ranking, adapter checkpoint, or partial prediction
into `C2b`.

### C2b — full run

Only after C2a's projection clears. Five outer folds × two fixed seeds under the amendment.

## The launch sequence, every time

1. **Guard chain.** All three, fresh, for this exact push:
   ```bash
   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py

   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py \
     --kernel-dir <kernel_dir> --contract <contract.json> \
     --registry reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json \
     --phase static --output <GUARD_RECEIPT_STATIC.json>

   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py \
     --kernel-dir <kernel_dir> --contract <contract.json> \
     --registry reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json \
     --phase launch --kernel-id <owner/exact-slug> --output <GUARD_RECEIPT_LAUNCH.json>
   ```
   Exit `0` and `"status": "PASS"` on each. Record path **and SHA-256**. A missing or stale
   receipt is itself grounds to refuse the launch (`MARGINAL_VALUE_GATE.md`).

2. **Acquire the serial lock** (`C0`). If held, **stop** — do not wait-loop into a race.

3. **Confirm zero in-flight.** `kernel_status` must not report `RUNNING`/`QUEUED` for this
   kernel. `cmux_ops` refuses by default; do not bypass.

4. **Quota pre-query**, owner-authenticated. Record `used_h`, `remaining_h`, UTC.

5. **Push exactly one version.** `enable_gpu: true`. Then **stop touching it.** A second push
   while the first is `RUNNING` is precisely the `0.74` GPU-h race.

6. **Exact version accounting.** Read back the kernel ID **and version number Kaggle actually
   assigned**. Assert both match what you pushed and what the launch guard was bound to.
   Kaggle has silently accepted a run under a drifted slug
   (`g02-dinov2-r5-cpu-promoted-preflight/2`) — `KFG-15`/`KFG-16` exist for this.

7. **Hand off to `C3`** with the lock still held. `C3` releases it.

## Step-0 failure handling

If step 0 fails, the run is terminal for that signature. Check
`KAGGLE_FAILURE_REGISTRY.json`:

- **Signature already present → stop. No retry.** This is absolute.
- **New signature → add the `KFG-N` family and a regression fixture in the same change that
  fixes it**, then one retry is permitted.

Known step-0 killers, all already paid for: recursive `/kaggle/input` traversal
(`190.30 s` / `197.23 s` / `225.61 s` / `1,242.976 s` vs a `5 s` envelope); wrong private
mount root (`/kaggle/input/<slug>` instead of `/kaggle/input/datasets/<owner>/<slug>`);
sibling module not packaged; hash-prefixed wheel basename; payload absent at run time.

## Do not

- Do not launch `6bba`, CV57, or any second family. Closed at this barrier.
- Do not launch DPRC-O0 or any other lane from this node.
- Do not submit anything. That is `C4`, and it has its own predicate.
- Do not modify the sealed preregistration, the frozen scorer, N1, B0, or the population.

## Done when

A single version is running under a verified ID/version, both guard receipts are `PASS` and
hashed, the quota pre-query is recorded, and the lock is held for `C3`.
