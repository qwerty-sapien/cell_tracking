# Kaggle kernel reliability runbook — why launches keep failing, and the fix

**Status: standing process rule, effective 2026-07-30. Mandatory reading before any
Kaggle kernel push, for every lane, in every worktree/branch.**

## The finding

The technical knowledge to prevent every observed Kaggle failure already exists:
`kaggle_prelaunch_guard.py` encodes **18** root-cause families (`KFG-01`..`KFG-18`,
`KAGGLE_FAILURE_REGISTRY.json`) with a 14+2-fixture regression suite, all passing
(`test_kaggle_prelaunch_guard.py`, `.venv/bin/python -m unittest`, 14/14 OK, verified
2026-07-30). **The problem is not missing knowledge. It is that this guard is never
structurally invoked.** Two confirmed gaps, found by direct inspection, not inference:

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
   lane-local preflight script that duplicates these checks.** If the shared guard is
   missing a check a new lane needs, extend the shared module and its registry — do not
   fork it. A bespoke per-lane validator is exactly the pattern that let `KFG-01`..`16`'s
   already-solved failure modes recur in a new lane.

3. After a remote/launch-phase attestation, re-run the guard with `--phase launch` and
   bind it to the exact kernel ID (`KFG-15`) before any scientific kernel is trusted.

4. Cite the receipt's file path and SHA-256 in the pushing node's report. A report that
   says "the guard exists" without a receipt hash for *this* push is not evidence the
   guard ran.

5. If any push fails with a signature not already in `KAGGLE_FAILURE_REGISTRY.json`,
   add a new `KFG-N` family entry (root cause, exact symptom, enforced prevention) and a
   regression fixture in the same change that fixes it — before retrying. A same-signature
   retry remains forbidden per the existing policy.

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
