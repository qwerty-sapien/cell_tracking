# Kaggle kernel reliability runbook — why launches keep failing, and the fix

**Status: standing process rule, effective 2026-07-30. Mandatory reading before any
Kaggle kernel push, for every lane, in every worktree/branch.**

## The finding

The technical knowledge to prevent every observed Kaggle failure already exists:
The shared failure registry now contains **25** root-cause families
(`KFG-01`..`KFG-25`) and the launch path is structurally enforced by
`scripts/cmux_ops.py`: every kernel push requires a fresh eligible receipt bound
to the exact metadata and code bytes. The G-02 guard suite has 17 fixtures; CMUX
and the G-03 protocol guard add their own focused fixtures. The original finding
below explains why structural enforcement was added.

1. **`KFG-18` — the guard has zero external call sites.** `grep -rln
   "kaggle_prelaunch_guard"` across the whole repo returns only its own directory and
   one prose mention in `LIVE_STATE.md`. No git hook, wrapper script, or CI step invokes
   it. `.git/hooks/` contains only `*.sample` files, and so does every vendored
   `external/*/.git/hooks/`. Any lane can push a kernel without ever running the guard,
   and evidently has: `reports/g02/screening/tiered_surface/branches/sa_dinov2/kaggle_gpu_bakeoff_v1/validate_package.py`
   re-implements its own bespoke, partial preflight (a single ad hoc heavy-import check)
   instead of importing the shared module — so that lane's push was protected only
   against the one failure mode its author happened to think of, not the accumulated 18.
2. **`KFG-17` — the dataset-ownership registry goes stale as new lanes are created.**
   `check_registry_freshness.py` (new, 2026-07-30) parses every `onebitbrain/*` slug
   referenced in `LIVE_STATE.md` and diffs it against `dataset_ownership`. Before the fix
   applied in this same change, it failed:

   ```
   FAIL_REGISTRY_STALE
   registry_uncovered_slugs: [
     "onebitbrain/g02-cellect-r7-separation-v1",
     "onebitbrain/g02-dinov2-r5-gpu-assets-v1"
   ]
   ```

   These are **the exact two datasets used by last week's GPU bake-off pushes** — the
   ones that failed. `dataset_ownership` now has 4 entries (was 2); the checker passes
   (`status: PASS`). But the underlying process gap remains: nothing forces a new dataset
   to be registered before it is used, so this will recur for the *next* new dataset
   unless the mandate below is followed.

Neither gap is a science problem. Every failure in the registry (`KFG-01`..`KFG-16`)
happened before reaching any model, checkpoint, or scoring code — 100% infrastructure,
0% scientific signal, exactly matching the observation that no kernel this week produced
a usable result.

**Live confirmation, same day.** While this runbook was being written, CELLECT's v7
attempt ran and failed with the same shape of problem: local static/regression checks
passed (24/24), a remote attestation had passed, yet the actual kernel run failed at
step 0 after `2.167 s` because "the request manifest was absent at the declared
`/kaggle/input/datasets/onebitbrain/...` mount" (`LIVE_STATE.md`, "DINOv2 FAIL BOTH /
CELLECT TRANSPORT STOP" entry). This is a **candidate `KFG-19`**
("remote dataset content absent at kernel-run time despite passing local guard and a
prior remote attestation" — a time-of-check/time-of-use gap between whatever was
attested and what the actual run mounted). It is left as a *candidate* rather than a
formalized family here because the exact log path/hash belongs in root's own terminal
report for that run, per the registry's evidentiary standard — add it there with the
same rigor as `KFG-01`..`18`, do not skip that step because the pattern looks familiar.
The campaign's own response — "GPU bake-off HALT; no seventh run; human decision
required" — independently reaches the same conclusion as this runbook's mandate below,
by a different route, in real time.

## The mandate (applies to every lane, every worktree, starting now)

**No kernel push is valid without a fresh, logged guard receipt for that exact
`kernel_dir` and `contract`.** Concretely, immediately before every `kaggle kernels push`:

1. Run the registry freshness check:
   ```bash
   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py
   ```
   If it fails, add the missing `dataset_ownership` entry (and, if the failure reveals a
   genuinely new symptom rather than plain staleness, a new `KFG-N` family) **before**
   proceeding. Do not push around a `FAIL_REGISTRY_STALE`.

2. Run the shared static guard against the kernel directory about to be pushed:
   ```bash
   python3 reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py \
     --kernel-dir <path/to/kernel_dir> \
     --contract   <path/to/contract.json> \
     --registry   reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json \
     --phase static \
     --output <path/to/GUARD_RECEIPT.json>
   ```
   Exit code `0` and `"status": "PASS"` in the receipt are required. **Do not write a new
   lane-local preflight script that duplicates these checks.** If a protocol is
   structurally incompatible with the G-02 single-dataset/step-0 contract (for
   example G-03's two-source atomic checkpoint handoff), add one repository-level,
   tested protocol validator that emits the same CMUX predicate schema, registers
   every asset/failure family, and verifies the protocol-specific bytes. It is a
   companion to the shared guard, not a weaker receipt or a CMUX bypass.

3. After a remote/launch-phase attestation, re-run the guard with `--phase launch` and
   bind it to the exact kernel ID (`KFG-15`) before any scientific kernel is trusted.

4. Cite the receipt's file path and SHA-256 in the pushing node's report. A report that
   says "the guard exists" without a receipt hash for *this* push is not evidence the
   guard ran.

5. If any push fails with a signature not already in `KAGGLE_FAILURE_REGISTRY.json`,
   add a new `KFG-N` family entry (root cause, exact symptom, enforced prevention) and a
   regression fixture in the same change that fixes it — before retrying. A same-signature
   retry remains forbidden per the existing policy.

6. **Dataset directory transport is a two-boundary contract.** Kaggle CLI defaults
   to `--dir-mode skip`, may print `Dataset creation error: ...` while exiting zero,
   and strips the one directory used as the archive root. Therefore:

   - CMUX dataset create/version always uses explicit zip mode and treats mutation
     error text as a failed operation even when the process status is zero;
   - to expose runtime path `<dataset-root>/handoff/...`, stage
     `<upload-root>/<wrapper>/handoff/...` (one wrapper deeper); and
   - after READY, enumerate remote member paths, download the exact version to a
     fresh directory, and verify the full declared tree before any consumer push.

   A READY status alone is not evidence that nested paths or bytes survived.

7. **Code submissions are hidden reruns, not uploads of public output.** A
   submission kernel must discover the mounted test population dynamically and
   generate a fresh output for those identities. A terminal predicate must bind
   the downloaded output to the exact `owner/slug/version` CMUX receipt and
   attest `public_only_output_copy:false`. The shared lifecycle atomically
   consumes that predicate before the external mutation, submits once, and
   requires exactly one new competition submission reference. CLI exit status
   zero or kernel `COMPLETE` without a score is not success.

## Interaction with the marginal-value gate

Per `reports/g02/screening/MARGINAL_VALUE_GATE.md`, a missing or stale guard receipt is
itself grounds to refuse a compute request, exactly like a failed HUMAN-GATE precondition
— it means the realistic probability of the launch reaching scientific code is unknown
and, on this week's evidence, low. Do not treat "we already have a guard" as sufficient;
treat "here is this push's fresh PASS receipt" as the bar.

## Root-cause family reference (updated 2026-07-30)

See `KAGGLE_FAILURE_AUDIT.md` for the full `KFG-01`..`16` table and worked incidents.
Added this change:

| Family | Root cause | Enforced prevention |
|---|---|---|
| `KFG-17` | Dataset-ownership registry goes stale as new lanes create new dataset slugs | `check_registry_freshness.py`, mandatory before push; add the entry in the same change that creates the dataset |
| `KFG-18` | The shared guard has no structural enforcement — no hook, no wrapper, no required call site — so lanes write bespoke, incomplete validators instead of reusing it | This runbook's mandate: a logged, receipted guard run is a precondition of every push; extend the shared module rather than forking it |

## Evidence

- `reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`,
  `KAGGLE_FAILURE_REGISTRY.json`, `KAGGLE_FAILURE_AUDIT.md` (pre-existing; 16 families).
- `reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py`
  (new; real run output above).
- `reports/g02/screening/tiered_surface/branches/sa_dinov2/kaggle_gpu_bakeoff_v1/validate_package.py`
  (confirmed bespoke, does not import the shared guard).
- `grep -rln "kaggle_prelaunch_guard" .` (confirmed zero external call sites, 2026-07-30).
- `.venv/bin/python -m unittest test_kaggle_prelaunch_guard` — 14/14 OK, run 2026-07-30,
  confirming the registry edits above did not break the existing suite.
