# K2 — Kaggle packaging, dataset registration, and mandatory guard receipts (zero GPU)

**Precondition: `K0` and `K1` complete.** This node creates the private dataset and the
kernel file. It does **not** push a kernel that runs science — that is `K3`.

Read `kaggle_prelaunch/RELIABILITY_RUNBOOK.md` in full before starting. Every failure this
campaign suffered on Kaggle happened **before reaching any model, checkpoint, or scoring
code** — 100% infrastructure, 0% scientific signal.

## Known-fatal patterns — each has already killed a run here

| Pattern | Evidence | Prevention |
|---|---|---|
| Recursive walk of `/kaggle/input` | 3 kernels dead at `190.30 s`, `197.23 s`, `225.61 s` vs a 5 s envelope; one full inventory took `1,242.976 s` | Address the exact pinned mount path. Never `rglob` the competition tree. |
| Wrong private mount root | `DINOV2-R5-CPU-V2` dead at `0.913 s` | Private datasets are at `/kaggle/input/datasets/<owner>/<slug>`, **not** `/kaggle/input/<slug>` |
| Sibling module not packaged | `SELECTOR-REQUIREMENT-CPU-V1` dead at `0.858 s` | One self-contained file. Inline step 0. Do not import a sibling. |
| Content-hash-prefixed wheel basename | GPU bake-off v1+v2, `0.74` GPU-h burned | Canonical PEP-427 wheel filenames. pip rejects the prefixed form. |
| Dataset content absent at run time despite passing local guard | CELLECT v7, `2.167 s` | Verify the **mounted** payload from inside the kernel, by pinned hash, and fail closed |
| Title/ID slug drift | accepted under an unexpected kernel ID | `KFG-15`: bind the launch guard to the exact kernel ID |
| Concurrent push race | two byte-identical versions, `0.74` GPU-h | Serial only; confirm zero in-flight G02 kernels first |
| Python 3.12 gzip OS byte | `SELECTOR-REQUIREMENT-CPU-V2`, `0.0565` CPU-h | Normalize the header or avoid recompression identity checks |

## Task

1. **Register the dataset before creating it.** Add the new slug to `dataset_ownership` in
   `KAGGLE_FAILURE_REGISTRY.json` **in the same change** that creates it, then run:
   ```bash
   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
   ```
   A `FAIL_REGISTRY_STALE` must be fixed, never pushed around.

2. **Build the private dataset.** Contents: the authenticated `model.pt` (from `K0`, exact
   hash `24c290ce…981f`), the frozen population shards, the sealed B0, the outer-split
   artifact per `H0`, the frozen N1 compiler and scorer sources, and offline wheels with
   **canonical** PEP-427 filenames. Emit a manifest with every member's byte size and
   SHA-256.
   - If any object must be wrapped for transport, use the **uncompressed tar** pattern that
     already round-tripped correctly through Kaggle expansion (CELLECT v4 recovered
     `f3e8e730…89ce5` exactly). Direct `.pth` transport is proven byte-changing.
   - Expect Kaggle to expand archives during materialization. The step-0 check must accept
     both the archive and the expanded-directory layout, resolving members **uniquely by
     pinned content hash**.

3. **Write the kernel as one self-contained file.** Step 0, first call, pure stdlib, under
   5 s, before any pip / import / model load:
   - resolve the bundle at the exact pinned mount path;
   - verify manifest, population manifest `4ab7bb23…76f0c`, checkpoint `24c290ce…981f`, and
     every wheel hash;
   - assert a positive action count;
   - fail closed on any missing **or duplicate** content match;
   - assert its own elapsed time is `< 5.0 s` and print it.
   Then the science suffix from `K1`, unchanged.

4. **Upload via cmux.** The owner's authenticated session only. Confirm the dataset reaches
   `READY`. Then have the owner download it and verify **every** member hash against the
   manifest — this campaign has twice discovered transport corruption only at this step.

5. **Run the shared guard.** Do not write a lane-local validator; that is `KFG-18`, and
   `sa_dinov2/.../validate_package.py` is the cautionary example already in the tree.
   ```bash
   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py \
     --kernel-dir <kernel_dir> \
     --contract   <contract.json> \
     --registry   reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json \
     --phase static \
     --output     <GUARD_RECEIPT_STATIC.json>
   ```
   Exit `0` and `"status": "PASS"` required. If the guard lacks a check this lane needs,
   **extend the shared module and its registry**. Never fork it.

6. **Local mount fixture.** Build an opaque fixture that mimics the Kaggle-expanded layout
   with no external bundle manifest, and prove step 0 resolves everything by content alone.
   Include negative fixtures: missing hash, duplicate hash, empty manifest — each must fail
   closed in under 5 s.

## Done when

Dataset `READY` and owner-verified member-by-member; kernel is one self-contained file;
static guard receipt is `PASS` with its **path and SHA-256** recorded; registry freshness
passes; local opaque fixture and all negative fixtures pass. **No kernel has been pushed.**
