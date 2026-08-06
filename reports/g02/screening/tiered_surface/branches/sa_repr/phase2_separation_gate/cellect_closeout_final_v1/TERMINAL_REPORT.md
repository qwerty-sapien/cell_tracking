# CELLECT final closeout — terminal report

Status: **PERMANENT PARK-C / SCIENTIFIC UNTESTED**

The one authorized final kernel created exactly one evidenced remote version
(`onebitbrain/g02-cellect-r7-closeout-final-v1`, server-assigned version 1)
and terminated with an infrastructure input-mount failure. No second push was
issued and no further CELLECT spend is authorized.

## What completed

Step 0 ran first and printed the full recursive `/kaggle/input` inventory with
path, byte size, and SHA-256 before any role resolution. It covered 25,055
files / 87,797,631,517 bytes and completed at 1,154.318675 seconds. The
inventory artifact hash printed by the kernel is
`aacff97ee5e8720432373084d1db5d7e74f85f219ed5e25cfa7e65b6b4cb2409`.

## Failure

The first content-resolution check then failed:

`Step0InputError: safe loader content resolution expected one match, found 0`

The inventory makes this self-diagnosing. The mounted
`g02-cellect-r7-published-assets-v1` payload was the `dinov2_cpu_bundle`
carrier. It contained all three canonical wheel files, the frozen population
files, and sealed B0, but contained neither the pinned CELLECT safe-loader
hash nor the pinned checkpoint hash. This is the previously registered
KFG-09 wrong-carrier family, not a supplier measurement.

The model was not loaded, no feature file was produced, and neither A1 nor A0
was evaluated. Therefore DELETE and occupied-SWAP are both **NOT EVALUATED**;
the supplier has no scientific PASS/FAIL result.

## Accounting and disposition

Authoritative Kaggle quota moved from 4.69 h used / 25.31 h remaining to
5.02 h used / 24.98 h remaining: **0.33 GPU-h debited**.

The precommitted hard rule now controls:

- CELLECT returns permanently to **PARK-C**.
- Scientific status is **UNTESTED**, not FAIL.
- No retry or further spend is authorized.
- No supplier-specific graph-gate preregistration is unlocked.
- The human-review barrier remains closed; no graph, training, full cache,
  6bba, CV57, submission, or RunPod action follows.

Evidence: `run_final/TERMINAL.json` SHA-256
`2d1f5c863ce2bcb0633d63001da54122594b22fb91b099508d410f72b509037d`;
`run_final/kernel.log` SHA-256
`1a5bf8ef08e8e4df330d4fa778a53564d1cd92f884314570e881fe29adeebb33`.

Version-count caveat: Kaggle's status endpoint normalizes version suffixes, so
it is not cited as independent uniqueness evidence. The credible evidence is
the fresh-slug zero-version preflight, one local push call, push stdout
assigning version 1, the held serial lock, and the recovery path's explicit
zero-push behavior.
