# N0b — Reviewable fix patch for the hidden-rerun cardinality pin (UNAPPLIED)

Date: 2026-08-06 · Node: N0b · Upstream: `dag_prompts/2026-08-06-alpha-hunt/runs/N0_SUBMISSION_FORENSICS.md`

**Nothing in this document has been applied.** No file under `scripts/`, `src/`, or
`kaggle_kernel/` was modified. No Kaggle command was run. No kernel was pushed and no
submission was made. Every diff below is text awaiting owner approval. All prototyping,
generation and probing was done on throwaway copies in the session scratchpad.

**Everything below was executed against real prototypes, not reasoned about on paper.**
The generator patch was applied to a scratchpad copy, the kernel was regenerated, the
regenerated kernel was diffed against the unpatched one, the STEP0 block was executed
against synthetic 0/1/3/4/5/17-video hidden mounts and against the alternate mount root,
and the new static guard was run over the defective kernel, the fixed kernel and three
scored-lineage sources. Results are quoted verbatim.

---

## 0. Verdict

| question | answer |
|---|---|
| Is the N0 fix correct? | **Yes**, and it is *necessary but not sufficient* — see A3 below. |
| Is the N0 fix complete? | **No.** N0 lists 3 sites; there are **9** live sites across 2 files plus 1 dead-code class it misses, and 1 operational blocker for its own D5 plan. |
| Does the fix perturb scoring? | **No.** Proven by regenerating the kernel: the diff is confined to lines 101–240 of a 3000-line file, and every symbol it touches is dead or telemetry-only in the shipped kernel (verified by grep). |
| Can it be proven without a submission? | **Yes**, fully, for both H1 and H2. Concrete harness in §(d), executed, results below. |
| Do any tests assert count == 4? | **One assertion breaks**, and it is not the one you would guess — `tests/test_public_temporalunet_baseline.py:98` asserts `'COMPETITION_TEST = Path('`, which the *mount-root* half of the fix removes. No test asserts the cardinality gate itself. |

---

## (a) Complete inventory of pinned-cardinality / public-inventory assumptions

### A. Fatal on the hidden rerun — the V32 submission path

| # | site | what is pinned | effect on the rerun |
|---|---|---|---|
| **A1** | `scripts/public_temporalunet_baseline.py:615-618` → shipped at `kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:182-185` | `len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT` | `RuntimeError` at import (`kernel:227 STEP0 = run_step0()`), before `submission.csv` exists. This is the N0 root cause. |
| **A2** | `scripts/public_temporalunet_baseline.py:533-536` → `kernel:100-103` | `EXPECTED_TEST_COUNT`, `REQUEST_COUNT`, `REQUEST_FILE_COUNT`, `POPULATION_FILE_COUNT`, all `= 4` | **New finding:** the last three are **dead**. `grep -n 'REQUEST_COUNT\|REQUEST_FILE_COUNT\|POPULATION_FILE_COUNT' kernel` returns *only* the assignments at 101–103 — never a read. `EXPECTED_TEST_COUNT` is read at 157, 182, 184, 222: one gate, three telemetry labels. Deleting all four is therefore behaviour-preserving except for the gate. |
| **A3** | `scripts/public_temporalunet_baseline.py:56` and `:545` → `kernel:112` | `COMPETITION_TEST` is one hard-coded literal root | `FileNotFoundError: STEP0 hidden test root is unavailable` if the rerun mounts at `/kaggle/input/<comp>/test`. **New finding, raises H2's severity above N0's 0.10 estimate:** `kernel:228` does `os.environ["BIOHUB_TEST_DIR"] = str(COMPETITION_TEST)`, and `kernel:310-311` reads `BIOHUB_TEST_DIR` *in preference to* its own two-candidate `COMP_DIR_CANDIDATES` resolution (`kernel:305-309`). **The kernel body's fallback is dead code in this build.** STEP0's single literal is authoritative for the entire run, so there is no second chance anywhere in the kernel. |
| **A4** | `scripts/public_temporalunet_baseline.py:590` and `:655` → `kernel:157, 222` | telemetry labels `{"expected_test_datasets": EXPECTED_TEST_COUNT}` | Cosmetic, but it is the last live reference keeping the pin alive as a concept. Rename with the rest. |

### B. Same class, and the prepared repair ladder inherits all of it

| # | site | note |
|---|---|---|
| **B1** | `scripts/v32_submission_bisect.py:115, 163-164` | Identical gate templated into the `step0_no_input_hash` stage. **Verified by generating it:** the pin lands at line 55 of the emitted stage-2 script. |
| **B2** | `scripts/v32_submission_bisect.py:118` | Single hard-coded `baseline.COMPETITION_TEST` — same H2 exposure as A3. |
| **B3** | `scripts/v32_submission_bisect.py:286-291` | Stages 4–8 call `baseline.build_hidden_script(...)` and inherit A1–A4 verbatim. |
| **B4** | `scripts/v32_submission_bisect.py:321-323, 331` | Contract fields pinned to `--expected-test-count` (default 4). Local metadata only. |

**Consequence, quantified.** Stage 1 (`pristine_control`) ships the raw notebook and carries
no pin, so under H1 it would score. Stages 2–8 all carry the pin, so they would all blank
identically. The ladder would "localise" the defect to the first non-pristine stage — a true
but useless statement — after spending up to **7 submission slots**. N0's E7 is confirmed
by construction, not by inference.

### C. Local-only pins — real, but they cannot cause a blank score

| # | site | why it is not the bug | recommendation |
|---|---|---|---|
| **C1** | `scripts/public_temporalunet_baseline.py:1828-1838` `validate_hidden_output` | `len(selected) != expected_count` against the package's `expected_test_count`. Runs **locally**, over the *public* run output we download before submitting. Kaggle rerun outputs are not downloadable, so this never sees a hidden inventory. | Leave unchanged. Add a comment. Changing it would cascade into the predicate and its receipts for zero benefit. |
| **C2** | `scripts/public_temporalunet_baseline.py:1133-1135, 1143, 1172` | Prelaunch-contract / package fields = `expected_test_count`. Consumed by `kaggle_prelaunch_guard.py:320-334` as "positive, and the literal appears somewhere in the source" (a weak check the digit `4` satisfies incidentally) and, **only in `launch` phase**, compared against a remote attestation. The V32 path uses `phase="inline"` (`scripts/public_temporalunet_hidden_submit.py:151`), so the attestation branch never executes. | Leave the numbers; add the new `hidden_inventory_cardinality_pinned: false` declaration. |
| **C3** | `scripts/public_temporalunet_baseline.py:457` (`step0_source`) | `len(SELECTED_DATASETS) != REQUEST_COUNT`. This is the **local CV kernel**, which mounts `train/` and is legitimately identity-pinned to the cv57 population. Not a competition code submission. | Out of scope. Do not touch. |
| **C4** | `scripts/pkgbh.py:186, 232`; `scripts/mkab45.py:155` | `compact_registry(expected_test_count=4)` — packaging-time check on the frozen `bhreg.json`, i.e. the original KFG-25 artefact. **Verified not consulted at rerun time:** `scripts/run_bh.py:3592-3599` calls `discover_submit_jobs(args.input)` in `submit` mode and never reaches `load_registry`. | Out of scope. |
| **C5** | `scripts/public_temporalunet_hidden_submit.py:19` | `KERNEL_ID = "onebitbrain/g02-v32-submit"` is hard-coded, and the module refuses any other slug (`predicate_payload` compares `package["kernel_id"] != KERNEL_ID`). | **Operational blocker for N0's D5** ("rebuild as a fresh slug"). Either edit this constant in the same change or parameterise it. Flagged, not patched — it is a naming decision for the owner. |

### D. Same class, different lineage — and it *survives* the rerun (new supporting evidence)

**D1** — `scripts/run_bh.py:839-842` `submit_input_contract_report(jobs, expected_frames=FRAME_COUNT)`
plus the `IMAGE_SUPPORT_SCALE_ZYX` voxel-scale comparison. These are per-video fail-closed
gates on the hidden mount, raising
`RunnerError("hidden submission input violated locked V45 assumptions")` — the same *class*
of frozen public assumption as A1, but on **geometry** rather than **cardinality**.

Refs `54910497` (0.877), `54942864` (0.881), `55203439` (0.881) and `55221497` (0.881) all
scored **through** this gate. Therefore the hidden test videos really do have 100 frames and
the locked voxel scale. The scored lineage constrains frame count, voxel scale, metadata
presence, and identity uniqueness — **everything except the count.** That is an independent
line of support for H1 that does not rest on E1/E2.

### E. Checked and cleared — no public-inventory assumption

- `kernel:1143-1153` `list_test_stems()` — rejects only an empty listing.
- `kernel:1313` `worker_count = min(2, available_gpu_count, len(test_stems))` — scales with N.
- `kernel:1250` `expected = set(test_stems[shard_index::worker_count])` — derived from N.
- `kernel:2952` `sorted(SELECTED_DATASETS) != sorted(test_stems)` — both sides dynamic.
- `kernel:242` cover image and `kernel:409-419, 1547-1550` DeepCenter paths — hard-coded
  dataset ids, but both are guarded: `if cover_image_path.exists()`, and `kernel:274-275`
  sets `BIOHUB_USE_DEEPCENTER_VETO='0'` / `BIOHUB_REQUIRE_DEEPCENTER_VETO='0'`. Absent
  identically in the public and hidden runs; not a discriminator.
- No frame-count, shape, node-count, edge-count or row-count pin anywhere in the V32 kernel.
- **Runtime headroom:** `…/g02_v32_submit_output_v1/PUBLIC_TEMPORALUNET_RUN_RECEIPT.json`
  records `prediction_seconds = 524.086` for 4 videos on 2×T4 ⇒ ~131 s/video. A 40-video
  hidden set is ≈87 min. Removing the cardinality pin creates **no timeout exposure** at
  any plausible N.
- **STEP0 hashing cost:** 408 files / 1.906 GB / ≈37 s at N=4 ⇒ ≈477 MB and ≈9 s per test
  video. At N=40 that is ≈6 min. Real, worth bounding (optional hunk 5), not fatal.

### F. Generator drift — disclose before rebuilding

`scripts/public_temporalunet_baseline.py` **no longer reproduces the shipped kernel byte for
byte.** Regenerating with the exact shipped gate SHA
(`5596270269a29bd3…ef2a`) and `expected_test_count=4` yields sha256
`1823973455ce946d…17c7`; the shipped file is `5193d0609e406f6d…b619d`. The three deltas are
all post-submission generator refactors, all inert:

1. `HASH_INPUT_CONTENTS = True` plus the `content_hashing=` inventory refactor — telemetry only.
2. `PUBLIC_TEMPORALUNET_CAP_OUTDEGREE_TWO = False` at generated line 1060 — assigned once,
   **never read** in a `public_v32` build (verified by grep on the regenerated source).
3. The STEP0 block itself is **byte-identical** between generator and shipped kernel.

So the resubmitted kernel will not be "55272130 plus one line". It will be 55272130 plus one
line plus two inert generator refactors. That is fine, but it must be stated in the receipt
rather than discovered later.

---

## (b) Proposed diffs — UNAPPLIED

Five hunk groups. Groups 1–2 are the fix. Groups 3–4 are the KFG-26 guard. Group 5 is
optional (N0's D3).

### Patch 1 of 5 — `scripts/public_temporalunet_baseline.py` (the fix)

```diff
--- scripts/public_temporalunet_baseline.py
+++ scripts/public_temporalunet_baseline.py
@@ -53,7 +53,11 @@
 PRIMARY_ROOT = f"/kaggle/input/datasets/{PRIMARY_SOURCE}"
 SECONDARY_ROOT = f"/kaggle/input/datasets/{SECONDARY_SOURCE}"
 COMPETITION_TRAIN = f"/kaggle/input/competitions/{COMPETITION}/train"
-COMPETITION_TEST = f"/kaggle/input/competitions/{COMPETITION}/test"
+COMPETITION_TEST_CANDIDATES = (
+    f"/kaggle/input/competitions/{COMPETITION}/test",
+    f"/kaggle/input/{COMPETITION}/test",
+)
+COMPETITION_TEST = COMPETITION_TEST_CANDIDATES[0]
 POPULATION_SCHEMA = "celltrack-alpha.public-temporalunet-populations.v1"
 MERGE_SCHEMA = "celltrack-alpha.public-temporalunet-merge.v1"
 SCORE_SCHEMA = "celltrack-alpha.public-temporalunet-frozen-score.v1"
@@ -530,10 +534,10 @@
 
 RUN_SCHEMA = {RUN_SCHEMA!r}
 VARIANT = {variant!r}
-EXPECTED_TEST_COUNT = {expected_test_count}
-REQUEST_COUNT = {expected_test_count}
-REQUEST_FILE_COUNT = {expected_test_count}
-POPULATION_FILE_COUNT = {expected_test_count}
+# Public-run reference only.  Kaggle reruns this kernel against a hidden test
+# set that "can be larger/smaller/different than the public dataset", so this
+# number is telemetry and must never gate the rerun (KFG-26).
+PUBLIC_REFERENCE_TEST_COUNT = {expected_test_count}
 WHEEL_COUNT = 60
 POPULATION_MANIFEST_SHA256 = {local_gate_sha256!r}
 GLOBAL_POPULATION_MANIFEST_SHA256 = {local_gate_sha256!r}
@@ -542,7 +546,11 @@
 PUBLIC_SCRIPT_VERSION_ID = 337501010
 PRIMARY_ROOT = Path({PRIMARY_ROOT!r})
 SECONDARY_ROOT = Path({SECONDARY_ROOT!r})
-COMPETITION_TEST = Path({COMPETITION_TEST!r})
+COMPETITION_TEST_CANDIDATES = [Path(_root) for _root in {list(COMPETITION_TEST_CANDIDATES)!r}]
+COMPETITION_TEST = next(
+    (_root for _root in COMPETITION_TEST_CANDIDATES if _root.is_dir()),
+    COMPETITION_TEST_CANDIDATES[0],
+)
 PRIMARY_MANIFEST_SHA256 = {PRIMARY_MANIFEST_SHA256!r}
 PRIMARY_WEIGHT_SHA256 = {PRIMARY_WEIGHT_SHA256!r}
 SECONDARY_MANIFEST_SHA256 = {SECONDARY_MANIFEST_SHA256!r}
@@ -587,7 +595,7 @@
     inventory_input("primary_support_pack", PRIMARY_ROOT)
     inventory_input("secondary_support_pack", SECONDARY_ROOT)
     inventory_input("competition_test", COMPETITION_TEST)
-    stage_start("step0", {{"expected_test_datasets": EXPECTED_TEST_COUNT}})
+    stage_start("step0", {{"public_reference_test_datasets": PUBLIC_REFERENCE_TEST_COUNT}})
     exact_files = {{
         PRIMARY_ROOT / "ARTIFACT_MANIFEST.json": PRIMARY_MANIFEST_SHA256,
         PRIMARY_ROOT / "weights/unet_transformer/split_0/edge_predictor_best.pth": PRIMARY_WEIGHT_SHA256,
@@ -604,17 +612,24 @@
         if actual != expected:
             raise RuntimeError(f"STEP0 hash mismatch for {{path}}: {{actual}} != {{expected}}")
         observed[str(path)] = actual
+    for _root in COMPETITION_TEST_CANDIDATES:
+        log_lookup(_root, _root if _root.is_dir() else None)
     if not COMPETITION_TEST.is_dir():
-        log_lookup(COMPETITION_TEST, None)
-        raise FileNotFoundError(f"STEP0 hidden test root is unavailable: {{COMPETITION_TEST}}")
+        raise FileNotFoundError(
+            "STEP0 hidden test root is unavailable in every known Kaggle mount: "
+            f"{{[str(_root) for _root in COMPETITION_TEST_CANDIDATES]}}"
+        )
     selected = sorted(
         path.name[:-5]
         for path in COMPETITION_TEST.iterdir()
         if path.is_dir() and path.name.endswith(".zarr")
     )
-    if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT:
+    # The scored lineage's contract (scripts/run_bh.py:629 discover_submit_jobs):
+    # reject only an empty or duplicated inventory, never a specific count.
+    if not selected or len(set(selected)) != len(selected):
         raise RuntimeError(
-            f"STEP0 hidden inventory drift: {{len(selected)}} != {{EXPECTED_TEST_COUNT}}"
+            f"STEP0 hidden inventory invalid: {{len(selected)}} discovered, "
+            f"{{len(set(selected))}} unique"
         )
     for dataset in selected:
         source = COMPETITION_TEST / f"{{dataset}}.zarr"
@@ -644,6 +659,9 @@
         "competition_path_visits": len(selected),
         "artifacts": observed,
         "selected_datasets": selected,
+        "competition_test_root": str(COMPETITION_TEST),
+        "public_reference_test_count": PUBLIC_REFERENCE_TEST_COUNT,
+        "hidden_inventory_cardinality_pinned": False,
         "hidden_test_compatible": True,
         "dynamic_test_discovery": True,
         "public_identity_pinned": False,
@@ -652,7 +670,7 @@
         "elapsed_seconds": time.time() - started,
     }}
     _atomic_json(Path("/kaggle/working/PUBLIC_TEMPORALUNET_STEP0.json"), value)
-    stage_end("step0", {{"expected_test_datasets": EXPECTED_TEST_COUNT}},
+    stage_end("step0", {{"public_reference_test_datasets": PUBLIC_REFERENCE_TEST_COUNT}},
               {{"verified_artifacts": len(observed), "discovered_datasets": len(selected)}})
     return value
```

Notes on design choices:

- The `expected_test_count` **keyword argument is deliberately left alone** everywhere
  (`hidden_step0_source`, `build_hidden_script`, `--expected-test-count`, the contract, the
  package manifest). Renaming it would cascade through `build_hidden_package`,
  `validate_hidden_output`, `v32_submission_bisect`, the prelaunch contract and four tests
  for no behavioural gain. What changes is what the generator *emits*: a single telemetry
  constant instead of four pinned ones, and no gate.
- Keeping one `PUBLIC_REFERENCE_TEST_COUNT = 4` line preserves `kaggle_prelaunch_guard.py:329-332`
  ("the exact request count must appear in the source") **honestly** rather than by accident
  via a hex digit.
- The candidate probe loop is added so that if H2 is the real cause, the rerun log now
  records both roots and which one was found, instead of a bare `FileNotFoundError`.
- `_root` leaks into the kernel's module globals. Harmless; no name collision exists
  (verified against the regenerated 3000-line kernel).

### Patch 2 of 5 — `scripts/v32_submission_bisect.py` (mirror, so the ladder is not self-defeating)

```diff
--- scripts/v32_submission_bisect.py
+++ scripts/v32_submission_bisect.py
@@ -112,10 +112,14 @@
 from pathlib import Path
 import time
 
-EXPECTED_TEST_COUNT = {expected_test_count}
+PUBLIC_REFERENCE_TEST_COUNT = {expected_test_count}
 PRIMARY_ROOT = Path({baseline.PRIMARY_ROOT!r})
 SECONDARY_ROOT = Path({baseline.SECONDARY_ROOT!r})
-COMPETITION_TEST = Path({baseline.COMPETITION_TEST!r})
+COMPETITION_TEST_CANDIDATES = [Path(_root) for _root in {list(baseline.COMPETITION_TEST_CANDIDATES)!r}]
+COMPETITION_TEST = next(
+    (_root for _root in COMPETITION_TEST_CANDIDATES if _root.is_dir()),
+    COMPETITION_TEST_CANDIDATES[0],
+)
 PRIMARY_MANIFEST_SHA256 = {baseline.PRIMARY_MANIFEST_SHA256!r}
 PRIMARY_WEIGHT_SHA256 = {baseline.PRIMARY_WEIGHT_SHA256!r}
 SECONDARY_MANIFEST_SHA256 = {baseline.SECONDARY_MANIFEST_SHA256!r}
@@ -155,13 +159,19 @@
             raise RuntimeError(f"STEP0 hash mismatch for {{path}}: {{actual}} != {{expected}}")
         observed[str(path)] = actual
     if not COMPETITION_TEST.is_dir():
-        raise FileNotFoundError(f"STEP0 hidden test root is unavailable: {{COMPETITION_TEST}}")
+        raise FileNotFoundError(
+            "STEP0 hidden test root is unavailable in every known Kaggle mount: "
+            f"{{[str(_root) for _root in COMPETITION_TEST_CANDIDATES]}}"
+        )
     selected = sorted(
         path.name[:-5] for path in COMPETITION_TEST.iterdir()
         if path.is_dir() and path.name.endswith(".zarr")
     )
-    if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT:
-        raise RuntimeError(f"STEP0 hidden inventory drift: {{len(selected)}} != {{EXPECTED_TEST_COUNT}}")
+    if not selected or len(set(selected)) != len(selected):
+        raise RuntimeError(
+            f"STEP0 hidden inventory invalid: {{len(selected)}} discovered, "
+            f"{{len(set(selected))}} unique"
+        )
     for dataset in selected:
         source = COMPETITION_TEST / f"{{dataset}}.zarr"
         if not (source / "0").exists():
@@ -188,6 +198,9 @@
         "competition_path_visits": len(selected),
         "artifacts": observed,
         "selected_datasets": selected,
+        "competition_test_root": str(COMPETITION_TEST),
+        "public_reference_test_count": PUBLIC_REFERENCE_TEST_COUNT,
+        "hidden_inventory_cardinality_pinned": False,
         "hidden_test_compatible": True,
         "dynamic_test_discovery": True,
         "public_identity_pinned": False,
```

### Patch 3 of 5 — `reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py` (KFG-26)

A name-based ban on the string `EXPECTED_TEST_COUNT` would be defeated by a rename, and a
blanket "no `len(x) == <int>`" rule produces **13 false positives** in the pristine V32
notebook body (measured: `len(matches) != 1`, `len(targets) == 1`, `len(neighbourhood) < 3`, …).
The check below is a scoped taint analysis: it taints only names assigned from a *directory
listing of a competition mount*, within the same name scope, and refuses a length comparison
of a tainted name against anything that is not another `len(...)` or the constant `0`.

Measured behaviour (§(d) for the run): **FAIL** on the shipped defective kernel and on the
generated bisect stage-2 script; **PASS** on the fixed kernel, on `kaggle_kernel/g04divcap5-v1/main.py`,
`kaggle_kernel/g1jump-submit/main.py`, `scripts/run_bh.py` and `scripts/public_temporalunet_baseline.py`.
Zero false positives.

```diff
--- reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py
+++ reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py
@@ -25,6 +25,16 @@
     ".glob('**/",
     ".rglob(",
 )
+# KFG-26 - runtime test-inventory cardinality pin on the hidden rerun.
+# Kaggle states the hidden dataset "can be larger/smaller/different than the
+# public dataset".  A submission kernel may reject an empty or duplicated
+# discovered inventory and nothing else.
+INVENTORY_LISTING_CALLS = {"iterdir", "glob", "rglob", "listdir", "scandir"}
+COMPETITION_MOUNT_MARKERS = (
+    "/kaggle/input/competitions/",
+    "/kaggle/input/biohub-cell-tracking",
+)
+_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
 PRISTINE_PUBLIC_V32_NOTEBOOK_SHA256 = (
     "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d"
 )
@@ -34,6 +44,98 @@
     pass
 
 
+def _scope_nodes(scope):
+    """Walk a scope without descending into nested function/class scopes."""
+    stack = list(ast.iter_child_nodes(scope))
+    while stack:
+        node = stack.pop()
+        yield node
+        if not isinstance(node, _NESTED_SCOPES):
+            stack.extend(ast.iter_child_nodes(node))
+
+
+def _scopes(tree: ast.Module):
+    yield tree
+    for node in ast.walk(tree):
+        if isinstance(node, _NESTED_SCOPES) and not isinstance(node, ast.Lambda):
+            yield node
+
+
+def _names(node) -> set[str]:
+    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
+
+
+def _strings(node) -> list[str]:
+    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
+
+
+def _targets(node) -> list[str]:
+    tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
+    return [n.id for t in tgts for n in ast.walk(t) if isinstance(n, ast.Name)]
+
+
+def _is_len(node) -> bool:
+    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
+
+
+def _len_arg_names(node) -> set[str]:
+    return _names(node.args[0]) if _is_len(node) and node.args else set()
+
+
+def competition_root_names(tree: ast.Module) -> set[str]:
+    roots: set[str] = set()
+    for _ in range(3):                                   # fixpoint over aliases
+        for node in ast.walk(tree):
+            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
+                continue
+            literal = any(m in s for s in _strings(node.value) for m in COMPETITION_MOUNT_MARKERS)
+            if literal or (_names(node.value) & roots):
+                roots.update(_targets(node))
+    return roots
+
+
+def _tainted(scope, roots: set[str]) -> set[str]:
+    tainted: set[str] = set()
+    for node in _scope_nodes(scope):
+        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
+            continue
+        lists = any(
+            isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
+            and c.func.attr in INVENTORY_LISTING_CALLS
+            for c in ast.walk(node.value)
+        )
+        if lists and (_names(node.value) & roots):
+            tainted.update(_targets(node))
+    return tainted
+
+
+def cardinality_pin_failures(tree: ast.Module) -> list[str]:
+    roots = competition_root_names(tree)
+    failures: list[str] = []
+    for scope in _scopes(tree):
+        tainted = _tainted(scope, roots)
+        if not tainted:
+            continue
+        for node in _scope_nodes(scope):
+            if not isinstance(node, ast.Compare):
+                continue
+            sides = [node.left, *node.comparators]
+            pinned = [s for s in sides if _is_len(s) and (_len_arg_names(s) & tainted)]
+            if not pinned:
+                continue
+            for other in sides:
+                if any(other is p for p in pinned) or _is_len(other):
+                    continue                # len(x) vs len(set(x)) is the legal dedup check
+                if isinstance(other, ast.Constant) and other.value == 0:
+                    continue                # rejecting an empty inventory is required
+                failures.append(
+                    f"KFG-26 runtime test-inventory cardinality pin at line "
+                    f"{node.lineno}: {ast.unparse(node)}"
+                )
+                break
+    return sorted(set(failures))
+
+
 def sha256(path: Path) -> str:
     digest = hashlib.sha256()
     with path.open("rb") as handle:
@@ -239,6 +341,15 @@
     if expected_root not in source:
         failures.append("code does not contain the exact owner-qualified mount root")
 
+    failures.extend(cardinality_pin_failures(tree))
+    if (
+        contract.get("hidden_test_compatible") is True
+        and contract.get("hidden_inventory_cardinality_pinned") is not False
+    ):
+        failures.append(
+            "contract must declare hidden_inventory_cardinality_pinned: false (KFG-26)"
+        )
+
     allow_pristine_discovery = contract.get("allow_pristine_broad_discovery") is True
     if allow_pristine_discovery and contract.get("execution_profile") not in {
         "bisection_step0_only",
```

The contract check is deliberately gated on `hidden_test_compatible is True`. The 17 existing
guard tests use a contract fixture that omits that key, so they are unaffected — **verified:
17/17 still pass against the patched guard**.

### Patch 4 of 5 — lifecycle guard, package contract, and the registry

Three small additions, described rather than diffed because they depend on the slug decision
in C5 and on whether the owner wants the field name above:

1. `scripts/public_temporalunet_baseline.py:1108-1162` `build_hidden_package` — add
   `"hidden_inventory_cardinality_pinned": False` to the `contract` dict (so patch 3's
   contract check has something to read) and to the `package` dict at 1167-1187.
2. `scripts/public_temporalunet_baseline.py:1949-1954` `validate_hidden_output` — add
   `"hidden_inventory_cardinality_pinned": step0.get("hidden_inventory_cardinality_pinned") is not False`
   to the `dataset_discovery` block (fail-closed: a STEP0 receipt from an unpatched kernel
   reports `True`).
3. `scripts/public_temporalunet_hidden_submit.py:164-179` — add
   `or validation.get("dataset_discovery", {}).get("hidden_inventory_cardinality_pinned") is not False`
   to the "terminal hidden validation is ineligible" condition. This makes it structurally
   impossible to submit a predicate built from an unpatched STEP0 receipt.
   Its three existing tests exercise `revalidate_hidden` / `exact_status_complete` /
   `exact_output_download` only and are unaffected.
4. `reports/…/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json` — append to `families`:

```json
{
  "id": "KFG-26",
  "name": "RUNTIME_TEST_INVENTORY_CARDINALITY_PIN_ON_HIDDEN_RERUN",
  "signature": "A code-submission kernel discovers the mounted test population dynamically and truthfully reports dynamic_test_discovery=true and public_identity_pinned=false, but then compares the discovered inventory's length against a frozen public constant and raises before writing submission.csv. The rerun terminalises COMPLETE with blank scores and total_bytes=0 - outwardly identical to KFG-25.",
  "prevention": "A submission kernel may reject only an empty or duplicated runtime inventory (the scripts/run_bh.py:629 discover_submit_jobs contract). Any comparison of a discovered competition-mount listing's length against a constant is refused statically by kaggle_prelaunch_guard.cardinality_pin_failures, and every hidden-test-compatible contract must declare hidden_inventory_cardinality_pinned: false. Resolve the competition mount against every known Kaggle root, never one literal.",
  "incidents": [
    {
      "id": "G02-V32-SUBMIT-V1-REF-55272130",
      "error": "EXPECTED_TEST_COUNT = 4 gate at kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:182-185 raises at import (line 227) whenever the hidden inventory is not exactly the public four; submission completed with blank public and private scores.",
      "evidence": "dag_prompts/2026-08-06-alpha-hunt/runs/N0_SUBMISSION_FORENSICS.md"
    }
  ]
}
```

KFG-26 is **distinct from KFG-25**: KFG-25 is a *membership* pin (the kernel selects or
copies a frozen identity set); KFG-26 is a *cardinality* pin on an otherwise correct dynamic
discovery. The V32 kernel passes every KFG-25 attestation and still fails.

### Patch 5 of 5 — OPTIONAL (N0's D3): bound the competition-mount inventory

Not the fix. Include only if the owner wants the ~9 s/video STEP0 hashing cost bounded in
the same change. Validated with the same probe (all cases pass, KFG-26 clean).

```diff
--- scripts/public_temporalunet_baseline.py   (on top of patch 1)
+++ scripts/public_temporalunet_baseline.py
@@ -326,22 +326,25 @@
     _run_log("config", config=config)
 
 
-def inventory_input(label, root):
+def inventory_input(label, root, hash_contents=None):
     root = Path(root)
+    hashing = HASH_INPUT_CONTENTS if hash_contents is None else bool(hash_contents)
     if not root.exists():
         _run_log("input_inventory", label=label, root=str(root), status="missing", files=[])
         return []
     files = []
+    total = 0
     for directory, _dirs, names in os.walk(root):
         for name in sorted(names):
             path = Path(directory) / name
             if path.is_file():
                 entry = {"path": str(path), "bytes": path.stat().st_size}
+                total += entry["bytes"]
-                if HASH_INPUT_CONTENTS:
+                if hashing:
                     entry["sha256"] = _run_sha256(path)
                 files.append(entry)
     _run_log("input_inventory", label=label, root=str(root), status="present",
-             content_hashing=HASH_INPUT_CONTENTS, files=files)
+             content_hashing=hashing, file_count=len(files), total_bytes=total, files=files)
     return files
 
 
@@ -594,7 +597,10 @@
     }})
     inventory_input("primary_support_pack", PRIMARY_ROOT)
     inventory_input("secondary_support_pack", SECONDARY_ROOT)
-    inventory_input("competition_test", COMPETITION_TEST)
+    # The competition mount scales with the hidden inventory; record paths,
+    # sizes and counts only.  Content attestation stays on the two pinned
+    # support packs, which is what actually needs it.
+    inventory_input("competition_test", COMPETITION_TEST, hash_contents=False)
     stage_start("step0", {{"public_reference_test_datasets": PUBLIC_REFERENCE_TEST_COUNT}})
```

---

## (c) Scoring behaviour: CONFIRMED UNTOUCHED

Three independent lines of proof.

**1. The regenerated kernel diff is spatially confined.** Patch 1 was applied to a scratchpad
copy of the generator and `build_hidden_script(expected_test_count=4, local_gate=<real gate>)`
was run on both. The complete diff between the unpatched and patched *generated kernels* is
**7 hunks, all between generated lines 101 and 240**, of a 3000-line file. There is not one
changed byte at or below line 240 — which is where the pinned public V32 notebook body
begins (`# ---- PUBLIC V32 CODE CELL 0 ----`).

**2. Every symbol the patch removes is provably unread.** On the shipped kernel:

```
$ grep -n "REQUEST_COUNT\|REQUEST_FILE_COUNT\|POPULATION_FILE_COUNT\|WHEEL_COUNT\|EXPECTED_TEST_COUNT" \
      kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py
100:EXPECTED_TEST_COUNT = 4
101:REQUEST_COUNT = 4                 <- assigned, never read
102:REQUEST_FILE_COUNT = 4            <- assigned, never read
103:POPULATION_FILE_COUNT = 4         <- assigned, never read
104:WHEEL_COUNT = 60
157:    stage_start("step0", {"expected_test_datasets": EXPECTED_TEST_COUNT})   <- telemetry
182:    if len(selected) != EXPECTED_TEST_COUNT or ...                          <- THE GATE
184:            f"STEP0 hidden inventory drift: ..."                            <- its message
200:        "wheel_count": WHEEL_COUNT,                                         <- telemetry
222:    stage_end("step0", {"expected_test_datasets": EXPECTED_TEST_COUNT}, ...) <- telemetry
```

Nothing in the inference, graph, threshold or CSV path reads any of them.

**3. Nothing scoring-relevant is in the changed region.** Unchanged and verified present,
byte-identical, in the regenerated patched kernel:

- `PRIMARY_WEIGHT_SHA256 = '12f6881e…fe771'`, `SECONDARY_WEIGHT_SHA256 = '9bac2fa0…e305f'`
  and the four exact-file hash assertions.
- Every `BIOHUB_*` threshold/radius/window env assignment (`kernel:264-283`, `kernel:840-870`),
  including `BIOHUB_DET_THRESHOLD`, `BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48`,
  `BIOHUB_SECONDARY_EDGE_WEIGHT = 0.15`, `BIOHUB_SECONDARY_LINK_MODE = low_margin_consensus`.
- `graph_policy = "competition_faithful_no_outdegree_cap"`; no outdegree cap is introduced.
- The `predict_unet_transformer.py` runtime patching, the ILP config, the CSV writer, the
  sharding logic, and the `sorted(SELECTED_DATASETS) != sorted(test_stems)` consistency check.
- `local_cv57_score = 0.9143584730207287`. **The CV57 number is not recomputed by this
  kernel at all** — it is a string literal carried from the local submission gate into the
  terminal receipt. It cannot move.

**Two honest caveats, stated loudly:**

- **The fix changes which inventories the kernel will run on, by design.** If the hidden set
  is genuinely larger than four, the resubmission will produce rows for videos we have never
  scored locally. The returned public score will therefore be a **first observation**, not a
  confirmation of 0.914358. Do not present it as a replication of CV57.
- **The mount-root change cannot alter the public run.** The public root is first in the
  candidate order, so `next(...)` selects it whenever it exists. Confirmed empirically in
  §(d): the `primary-root N=4` case resolves to
  `/kaggle/input/competitions/biohub-cell-tracking-during-development/test`, exactly as
  before the patch.

---

## (d) Concrete zero-submission local validation — ALREADY EXECUTED

### The harness

Split the generated kernel at `# ---- PUBLIC V32 CODE CELL 0 ----` and again at
`STEP0 = run_step0()`, `exec` the telemetry + STEP0 prelude into a fresh namespace, then
override five globals in that namespace before calling `run_step0()` by hand:
`RUN_LOG_PATH`, `PRIMARY_ROOT`, `SECONDARY_ROOT`, `COMPETITION_TEST(_CANDIDATES)`, plus stubs
for `_sha256` (returns the pinned constants, so the artifact gate is still exercised as a real
comparison) and `_atomic_json` (captures the receipt instead of writing to `/kaggle/working`).
Synthetic mounts are `N` directories named `hid_NNNN.zarr` each containing a `0/` child —
exactly what STEP0's discovery and malformed-dataset checks look for.

```python
"""Zero-submission exercise of the generated hidden STEP0 against synthetic mounts."""
import json, pathlib, tempfile

REAL = {  # the four pinned exact-file hashes, keyed by their Kaggle mount paths
    ".../biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json": "bc20f1f0...8103",
    ".../biohub-tracking-support-pack-50ep-v1/weights/.../edge_predictor_best.pth": "12f6881e...e771",
    ".../biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json": "7f01c0b2...95eb",
    ".../biohub-temporal-unet3d-seed314159-v1/weights/.../edge_predictor_best.pth": "9bac2fa0...305f",
}

def prelude(path):
    head = pathlib.Path(path).read_text().split("\n# ---- PUBLIC V32 CODE CELL 0 ----")[0]
    return head.split("STEP0 = run_step0()")[0]

def make_test_root(root, count):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"hid_{index:04d}.zarr" / "0").mkdir(parents=True)
    return root

def run_case(gen_path, *, count, alt_root=False, label=""):
    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        primary, secondary = base / "primary", base / "secondary"
        hashes = fake_support(base, primary, secondary)   # materialises the 4 pinned files
        c0 = base / "kaggle/input/competitions/biohub-cell-tracking-during-development/test"
        c1 = base / "kaggle/input/biohub-cell-tracking-during-development/test"
        make_test_root(c1 if alt_root else c0, count)
        ns = {}
        exec(compile(prelude(gen_path), "step0_prelude.py", "exec"), ns)
        Path = ns["Path"]
        ns["RUN_LOG_PATH"] = base / "RUN_LOG.jsonl"
        ns["PRIMARY_ROOT"], ns["SECONDARY_ROOT"] = Path(primary), Path(secondary)
        ns["COMPETITION_TEST_CANDIDATES"] = [Path(c0), Path(c1)]
        ns["COMPETITION_TEST"] = next((p for p in (Path(c0), Path(c1)) if p.is_dir()), Path(c0))
        ns["_sha256"] = lambda path: hashes[str(path)]
        captured = {}
        ns["_atomic_json"] = lambda path, value: captured.update(value)
        try:
            ns["run_step0"]()
            return {"case": label, "outcome": "PASS",
                    "request_count": captured.get("request_count"),
                    "root": captured.get("competition_test_root", "<n/a>")}
        except Exception as exc:
            return {"case": label, "outcome": type(exc).__name__, "error": str(exc)[:140]}

if __name__ == "__main__":
    import sys
    for count in (1, 3, 4, 5, 17, 0):
        print(json.dumps(run_case(sys.argv[1], count=count, label=f"primary-root N={count}")))
    print(json.dumps(run_case(sys.argv[1], count=6, alt_root=True, label="alternate-root N=6")))
```

### Measured results

**Unpatched generator** (what is on Kaggle as `g02-v32-submit` v1):

```
{"case": "primary-root N=1",  "outcome": "RuntimeError", "error": "STEP0 hidden inventory drift: 1 != 4"}
{"case": "primary-root N=3",  "outcome": "RuntimeError", "error": "STEP0 hidden inventory drift: 3 != 4"}
{"case": "primary-root N=4",  "outcome": "PASS", "request_count": 4}
{"case": "primary-root N=5",  "outcome": "RuntimeError", "error": "STEP0 hidden inventory drift: 5 != 4"}
{"case": "primary-root N=17", "outcome": "RuntimeError", "error": "STEP0 hidden inventory drift: 17 != 4"}
{"case": "primary-root N=0",  "outcome": "RuntimeError", "error": "STEP0 hidden inventory drift: 0 != 4"}
{"case": "UNFIXED alt-root only, N=4", "outcome": "FileNotFoundError",
 "error": "STEP0 hidden test root is unavailable: .../kaggle/input/competitions/..."}
```

**Patched generator:**

```
{"case": "primary-root N=1",  "outcome": "PASS", "request_count": 1,  "root": ".../input/competitions/biohub-cell-tracking-during-development/test"}
{"case": "primary-root N=3",  "outcome": "PASS", "request_count": 3,  "root": ".../input/competitions/biohub-cell-tracking-during-development/test"}
{"case": "primary-root N=4",  "outcome": "PASS", "request_count": 4,  "root": ".../input/competitions/biohub-cell-tracking-during-development/test"}
{"case": "primary-root N=5",  "outcome": "PASS", "request_count": 5,  "root": ".../input/competitions/biohub-cell-tracking-during-development/test"}
{"case": "primary-root N=17", "outcome": "PASS", "request_count": 17, "root": ".../input/competitions/biohub-cell-tracking-during-development/test"}
{"case": "primary-root N=0",  "outcome": "RuntimeError", "error": "STEP0 hidden inventory invalid: 0 discovered, 0 unique"}
{"case": "alternate-root N=6","outcome": "PASS", "request_count": 6,  "root": ".../input/biohub-cell-tracking-during-development/test"}
```

This is a complete, zero-cost proof of both H1 and H2 and of the regression case (N=4 is
unchanged, still resolves to the same root, still reports `request_count: 4`). It also proves
the empty-inventory refusal is retained — the one thing we must not lose.

### Second zero-submission check — the static guard

```
$ python3 kfg26.py <targets>
kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py: FAIL
    KFG-26 runtime test-inventory cardinality pin at line 182: len(selected) != EXPECTED_TEST_COUNT
    KFG-26 runtime test-inventory cardinality pin at line 182: len(set(selected)) != EXPECTED_TEST_COUNT
<generated bisect stage-2 script>:                              FAIL   (line 55, same pin)
<patched generated kernel>:                                     PASS
<patched generated bisect stage-2 script>:                      PASS
kaggle_kernel/g04divcap5-v1/main.py:                            PASS
kaggle_kernel/g1jump-submit/main.py:                            PASS
scripts/run_bh.py:                                              PASS
scripts/public_temporalunet_baseline.py:                        PASS
```

### Ordered pre-approval checklist (all zero-submission)

1. Apply patches 1–2. Regenerate a package to a *scratch* directory and diff the emitted
   kernel against `kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py`.
   Expect exactly the 7 STEP0 hunks plus the 3 known generator-drift deltas of §F.
2. Run the probe above on the newly emitted kernel. Expect the "Patched generator" table.
3. Apply patch 3. Run `cardinality_pin_failures` over the emitted kernel (PASS) and over the
   shipped one (FAIL, 2 findings).
4. `.venv313/bin/python -m pytest tests/test_public_temporalunet_baseline.py tests/test_v32_submission_bisect.py tests/test_public_temporalunet_hidden_submit.py reports/g02/screening/tiered_surface/kaggle_prelaunch -q`
5. Only then, with owner approval, push a fresh slug and spend **one** submission.

---

## (e) Test files needing updates

**Current baseline, measured:** `36 passed, 3 subtests passed in 8.51s` for
`tests/test_public_temporalunet_baseline.py`, `tests/test_v32_submission_bisect.py`,
`tests/test_public_temporalunet_hidden_submit.py` and
`reports/g02/screening/tiered_surface/kaggle_prelaunch/` (all four are collected by
`pyproject.toml [tool.pytest.ini_options] testpaths`).

Every string assertion in those suites was re-evaluated against the regenerated patched
kernel. **Exactly one breaks.**

| file:line | assertion | status |
|---|---|---|
| **`tests/test_public_temporalunet_baseline.py:98`** | `self.assertIn('COMPETITION_TEST = Path(', source)` | **BREAKS.** The mount-root hunk replaces this with `COMPETITION_TEST_CANDIDATES = [Path(...)]` + `COMPETITION_TEST = next(`. Replace with `self.assertIn('COMPETITION_TEST_CANDIDATES = [', source)` and `self.assertIn('COMPETITION_TEST = next(', source)`. |
| `tests/test_public_temporalunet_baseline.py:99-104` | `DYNAMIC_TEST_DISCOVERY = True`, `PUBLIC_OUTPUT_COPIED = False`, `RUN_LOG_PATH = Path(...)`, `assertNotIn 44b6_ / 6bba_ / SELECTED_TRAIN` | all still hold |
| `tests/test_public_temporalunet_baseline.py:95` | `build_hidden_script(expected_test_count=4, …)` | call argument only — the kwarg is retained by design; unaffected |
| `tests/test_public_temporalunet_baseline.py:134, 196, 211` | `expected_test_count: 1`, `request_count: 1` | these are the *local validator* fixtures (C1); unaffected |
| `tests/test_public_temporalunet_baseline.py:696` | `contract["request_count"] == 1` | `build_package` (local CV path, C3); unaffected |
| `tests/test_v32_submission_bisect.py:61, 75` | `expected_test_count=4` | call arguments only; unaffected |
| `tests/test_v32_submission_bisect.py:63-64` | `assertIn("STEP0 = run_step0()")`, `assertIn('recursive_input_content_hashing": False')` | both survive the bisect patch — **verified on the regenerated stage-2 script** |
| `tests/test_public_temporalunet_hidden_submit.py` (3 tests) | `revalidate_hidden` / `exact_status_complete` / `exact_output_download` | do not reach `predicate_payload`; unaffected by patch 4.3 |
| `reports/…/kaggle_prelaunch/test_kaggle_prelaunch_guard.py` (17 tests) | fixture source has no competition mount and no listing call; fixture contract omits `hidden_test_compatible` | **17/17 verified passing against the patched guard** |
| `tests/test_g04_segmentation_production_package.py:248` | `prelaunch["request_count"] == 2` | unrelated G-04 package; unaffected |

**No test anywhere asserts that the hidden test count is 4.** `grep -rn "EXPECTED_TEST_COUNT"
tests/` returns nothing; the literal `4` appears only as a call argument.

### New tests the patch should add

1. `tests/test_public_temporalunet_baseline.py` — `test_hidden_step0_admits_any_positive_inventory`:
   the §(d) harness, parameterised over N ∈ {1, 3, 4, 5, 17} (PASS) and N = 0 (raises), plus
   the alternate-root case. This is the regression test that would have caught 55272130.
2. `tests/test_public_temporalunet_baseline.py` — `test_hidden_script_has_no_cardinality_pin`:
   assert `guard.cardinality_pin_failures(ast.parse(build_hidden_script(...))) == []`.
3. `tests/test_v32_submission_bisect.py` — the same assertion on `build_step0_only_script`.
4. `reports/…/test_kaggle_prelaunch_guard.py` — three cases: a source with
   `len(<discovered>) != <const>` is rejected; a source with the `not selected or
   len(set(selected)) != len(selected)` contract is accepted; a contract declaring
   `hidden_test_compatible: true` without `hidden_inventory_cardinality_pinned: false` is
   rejected.

---

## Residual risks and open items

1. **C5 blocks N0's D5.** `scripts/public_temporalunet_hidden_submit.py:19` hard-codes
   `onebitbrain/g02-v32-submit`. A fresh slug requires editing that constant. Owner decision.
2. **§F generator drift** must be disclosed in the rebuild receipt. The resubmission is not
   byte-equal to 55272130 plus one line.
3. **H3 is untouched by this patch.** The two-dataset-source composition
   (`pilkwang/biohub-tracking-support-pack-50ep-v1` and
   `pilkwang/biohub-temporal-unet3d-seed314159-v1`, both third-party) with exact SHA-256
   assertions on four files remains a fail-closed dependency. Nothing here reduces that
   exposure. If the resubmission still blanks, H3 is the next suspect and the STEP0 lookup
   log (now including the candidate-root probe) will say so.
4. **Not attempted here, and still the cheapest evidence:** N0's D4.1 — retrieve the rerun's
   own `scriptVersionId`, `errorDescription` and logs for ref 55272130 through the
   authenticated CMUX/Chrome session. If H1 is right, the rerun traceback literally prints
   `STEP0 hidden inventory drift: N != 4` where **N is the true hidden test count**. That
   costs nothing and would convert this patch from "correct fix for a 0.75-confidence
   hypothesis" into "correct fix for a measured fact". It requires no code change and should
   be done before any submission is spent.
