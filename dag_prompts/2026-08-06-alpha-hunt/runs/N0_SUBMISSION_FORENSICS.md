# N0 — Kaggle submission forensics: scored vs. unscored

Date: 2026-08-06 · Node: N0 (read-only investigation) · Competition:
`biohub-cell-tracking-during-development` · Team `onebitbrain`

**No Kaggle mutation occurred.** The local `.venv313/bin/kaggle` CLI is
unauthenticated in this sandbox (`Authentication required to call the Kaggle
API`), so every finding below is derived from repo artifacts already captured by
CMUX, from the shipped kernel packages, and from the organizer's public
competition download. No live query was issued.

---

## 0. Headline

**All 13 submissions used the same transport (Kaggle code/kernel-output
submission) and the same CSV schema. Transport and schema are NOT the
discriminator.** The discriminator is the *runtime test-inventory contract* of
the submitted kernel — i.e. what the kernel does when Kaggle re-runs it against
the hidden test mount.

Every scored submission comes from a kernel that enumerates the mounted
`test/*.zarr` with **no cardinality and no identity constraint**. Every
diagnosed unscored submission comes from a kernel that imposes some *frozen
public-test assumption* on the rerun.

For the flagship `55272130`, the frozen assumption is a single line:

```python
# kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:100
EXPECTED_TEST_COUNT = 4
...
# lines 182–185, inside run_step0(), executed at import time (line 227)
if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT:
    raise RuntimeError(
        f"STEP0 hidden inventory drift: {len(selected)} != {EXPECTED_TEST_COUNT}"
    )
```

The V32 kernel discovers the test set dynamically (so it truthfully reports
`dynamic_test_discovery: true` and `public_identity_pinned: false`, which is all
the lifecycle guard checks) and then **hard-rejects any inventory whose size is
not exactly the public 4**. This is KFG-25 in a new disguise: a *cardinality*
pin instead of an *identity* pin, and it is the only membership constraint the
V32 kernel adds over the pinned public notebook, which is itself
cardinality-agnostic (`list_test_stems()` at kernel line 1143 accepts any
non-empty count).

---

## (a) Scored vs. unscored comparison table

Authoritative listing:
`reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/submission_repair_v32/evidence/cmux_receipts/competition_submissions_2026-08-05T191913532266Z.json`
(stdout SHA-256 `ef3bfe2f…d044`).

| ref | date (UTC) | public | kernel / version | transport | runtime test-inventory contract | local evidence |
|---|---|---|---|---|---|---|
| 54717551 | 07-15 08:02 | **blank** | *unknown* | *unrecorded* | *unknown* | **none in repo** |
| 54890576 | 07-22 01:01 | **blank** | `onebitbrain/bhv45-submit` **v1** (rerun scriptVersionId `336990437`) | code | **frozen 4-ID registry** `bhreg.json`, selected before inspecting the mount | `reports/g02/kaggle_submission_failure_54890576_v1.json` — Kaggle: *"Notebook Threw Exception … hidden dataset can be larger/smaller/different"*, `total_bytes=0` |
| 54893629 | 07-22 04:13 | **blank** | *unknown* (falls inside the broken-registry window; plausibly `bhv45-submit/2`, **unconfirmed**) | *unrecorded* | *unknown* | **none in repo** |
| 54910497 | 07-22 18:42 | 0.877 | `onebitbrain/bhv45-submit` **v3** | code | `run_bh.discover_submit_jobs` — every mounted `test/*.zarr`, **no count/ID pin** | `reports/g02/scorer_parity_v1.json` |
| 54921344 | 07-23 06:44 | 0.860 | `onebitbrain/bhp0-c0` v1 | code | same corrected bh45 lineage | `reports/g02/scorer_parity_v1.json` |
| 54923074 | 07-23 08:16 | 0.878 | `onebitbrain/bhp0-c2` v2 | code | same corrected bh45 lineage | `reports/g02/scorer_parity_v1.json` |
| 54942864 | 07-24 04:26 | 0.881 | `onebitbrain/bhap-submit` (scriptVersionId `337554797`) | code | same corrected bh45 lineage | `…/hidden_transport_v1/BUILD_RECEIPT.json` |
| 55201829 | 08-03 04:04 | **blank** | `onebitbrain/g04-division-permissive-submit-v1` v1 (scriptVersionId `339815258`) | code | **copies a frozen 4-public-dataset CSV; `competition_input_visits=0`** | `…/code_transport_v1/TRANSPORT_INVALIDATION.json` — Kaggle: *"Your notebook submission file is in an incorrect format"*, `total_bytes=0` |
| 55203439 | 08-03 05:39 | 0.881 | `onebitbrain/g04divcap5-v1` v1 | code | wraps bh45 → `run_bh.py:564 discover_submit_jobs`, **no count/ID pin** | `…/hidden_transport_v1/CODE_SUBMISSION_PREDICATE.json` |
| 55212065 | 08-03 12:51 | **blank** | *unknown* — 8 min after `kernel_push_2026-08-03T124323791747Z` (`g02-public-temporalunet-v32-full-s01-v1`) | *unrecorded* | *unknown* | **none in repo** |
| 55221497 | 08-03 21:00 | 0.881 | `onebitbrain/g1jump-submit` v1 | code | same bh45 dynamic discovery | `…/global_jump_public_transfer_v1/PUBLIC_TRANSFER_RESULT.json` |
| 55251996 | 08-04 23:27 | **blank** | *unknown* | *unrecorded* | *unknown* | **none in repo** |
| **55272130** | **08-05 14:02** | **blank** | `onebitbrain/g02-v32-submit` **v1** | code | **dynamic discovery + `EXPECTED_TEST_COUNT = 4` hard gate** | `…/submission_receipts/competition_code_submit_2026-08-05T140233835944Z.json`; predicate `1409662b…02ea` |

Notes on the table:

- `privateScore` is empty for **every** submission including the scored ones —
  that is the normal hidden private leaderboard, not a defect.
- Four refs (`54717551`, `54893629`, `55212065`, `55251996`) have **no artifact
  anywhere in the repo** beyond appearing in submission listings. They were not
  created through the receipted CMUX lifecycle. `55212065` is temporally
  adjacent to a kernel push (8 minutes), which is consistent with a Kaggle
  notebook-editor "Submit to Competition" auto-attachment, but this is
  **unproven** and must not be treated as established.

### Direct-upload transport is not an available alternative

`reports/g02/kaggle_submission_failure_54890576_v1.json` records a direct CSV
upload attempt returning **HTTP 400**, "competition requires a code submission
identified by kernel, version, and output filename"; LIVE_STATE repeats this for
the G-04 candidate (`da73c754…84509`). So *all* 13 refs are code submissions and
transport cannot discriminate them.

---

## (b) Root cause, ranked

### H1 — `EXPECTED_TEST_COUNT = 4` fails closed on the hidden rerun mount — **confidence ~0.75**

STEP0 runs at module import (kernel line 227), before any heavy work. If the
rerun's `/kaggle/input/competitions/biohub-cell-tracking-during-development/test`
contains any number of `.zarr` directories other than exactly 4, the kernel
raises `RuntimeError` before writing `/kaggle/working/submission.csv`. The
submission then terminalises `COMPLETE` with blank scores and `total_bytes=0` —
the exact observed signature, and the same signature Kaggle produced for
`54890576` (which threw for the identity-pin equivalent).

### H2 — STEP0's single hard-coded competition mount root — **confidence ~0.10**

`COMPETITION_TEST` (kernel line 112) is a single literal
`/kaggle/input/competitions/<comp>/test`. The kernel *body* tries two candidate
roots (`COMP_DIR_CANDIDATES`, lines 305–311) and the scored bh45 lineage tries
"the two known Kaggle mount paths" (decoded `bh45.py:405–407`), but STEP0 runs
first and raises `FileNotFoundError: STEP0 hidden test root is unavailable` if
the rerun mounts at the alternate root. Same failure class, same fix location.

### H3 — Secondary weight dataset drift — **confidence ~0.08**

`g02-v32-submit` is the **only** submitted kernel with two dataset sources
(`pilkwang/biohub-tracking-support-pack-50ep-v1` *and*
`pilkwang/biohub-temporal-unet3d-seed314159-v1`); every scored kernel mounts one.
STEP0 asserts exact SHA-256 for four files across both. A third-party dataset
version bump or unavailability at rerun time raises `FileNotFoundError` /
`RuntimeError` in STEP0 with the identical outward signature.

### H4 — Full recursive input hashing (the currently prepared hypothesis) — **confidence ~0.04**

Measured from the retained `RUN_LOG.jsonl`:
`inventory_input("competition_test", …)` walked **408 files / 1,906,332,008
bytes**, and total STEP0 elapsed was **37.006 s** covering ≈2.62 GB (≈70 MB/s).
That is ~102 files and ~477 MB per test video. Even a 10× larger hidden set
costs ≈6 minutes of hashing and ≈1.3 MB of `RUN_LOG.jsonl` — far inside any
rerun budget. Hashing is a real cost and worth bounding, but on this arithmetic
it is **not a plausible fail-closed cause**, and the eight-stage ladder in
`scripts/v32_submission_bisect.py` is currently built to test it.

### H5 — CSV schema / dialect defect — **REJECTED, confidence <0.01**

Measured directly (see (c)). The V32 CSV is schema-identical to two scored CSVs
and to the organizer `sample_submission.csv`, and it is a clean graph.

---

## (c) Discriminating evidence

### E1 — The already-executed controlled experiment: same kernel, only the discovery contract changed

`onebitbrain/bhv45-submit` **v1** → ref `54890576` → **blank**, Kaggle "Notebook
Threw Exception … the hidden dataset can be larger/smaller/different than the
public dataset" (`reports/g02/kaggle_submission_failure_54890576_v1.json`).
The only change made was replacing the frozen `bhreg.json` selection with
`discover_submit_jobs` (`reports/g02/scorer_submission_path_findings_v1.md`
items 18–21). `onebitbrain/bhv45-submit` **v3** → ref `54910497` → **0.877**.
Same package family, same scorer, same CSV writer. The rerun test-inventory
contract is the sole varied factor and it flipped the outcome.

### E2 — A byte-identical CSV both scored and did not score

SHA-256 `da73c7540dc0f1052b1f6f0d1399a9a9867f24de9c8463180fa6eef2bb484509`
(263,297 data rows, 13,713,344 bytes) is present at **both**

- `reports/g04/lanes/division_permissive/code_transport_v1/remote_output_v1/submission.csv` (ref `55201829`, **blank**, "incorrect format"), and
- `reports/g04/lanes/division_permissive/hidden_transport_v1/remote_output_v1_terminal/submission.csv` (ref `55203439`, **0.881**).

The failing kernel emitted those exact bytes at rerun time (it copies a frozen
file and never reads competition input). The scoring kernel emitted those bytes
only during its *public* run and **regenerated** its output at rerun time.
Therefore the bytes cannot be the defect, and — critically — **a submission
covering exactly the four public dataset ids is rejected at rerun time**, which
is direct evidence that the rerun's expected test membership is not
`{44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1}`.

### E3 — Schema is identical across the scored/unscored boundary

Measured on the four available artifacts plus the organizer template:

| artifact | header | cols | dtypes | node sentinels | edge sentinels | ids | line ending | bytes | rows |
|---|---|---|---|---|---|---|---|---|---|
| sample_submission (organizer, from the 87 GB competition zip) | exact | 10 | all int | `source_id=-1,target_id=-1` | `node_id..x = -1` | contiguous from 0 | LF | 890 | 20 |
| **55272130 (unscored, V32)** | exact | 10 | all int | same | same | contiguous from 0 | **CRLF** | 12,238,394 | 235,198 |
| 55201829 (unscored, frozen copy) | exact | 10 | all int | same | same | contiguous from 0 | CRLF | 13,713,344 | 263,297 |
| **55203439 (scored 0.881)** | exact | 10 | all int | same | same | contiguous from 0 | **CRLF** | 13,713,344 | 263,297 |
| **55221497 (scored 0.881)** | exact | 10 | all int | same | same | contiguous from 0 | **LF** | 13,448,137 | 263,260 |

Header is `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id` in all five.
No quoting anywhere; `-1` sentinels are integers, never floats; there is no extra
index column. CRLF appears on both a scored and an unscored file, so line
endings are excluded. All four submissions name the same four datasets.

Graph validation of the V32 CSV: 119,616 nodes / 115,582 edges, **0 dangling
endpoints, 0 edges with `t_target − t_source ≠ 1`, max out-degree 2, max
in-degree 1, 0 merges, 285 forks**. It satisfies the post-`aa65e90` organizer
rules described in `external/kaggle-cell-tracking-competition/metrics.md`. The
artifact is clean; the failure is not in the file.

### E4 — The scored kernels contain no cardinality gate; the V32 kernel does

- `run_bh.discover_submit_jobs` (decoded from `kaggle_kernel/bhap-submit/bh45.py`
  → `bh45src.zip` → `scripts/run_bh.py:564`) raises only on **zero** jobs or
  **duplicate** identities: `if not jobs or len(set(identities)) != len(identities)`.
  Its docstring states the reason outright: *"Kaggle code submissions are rerun
  against hidden test identities, so a packaged public registry cannot define
  submission membership."*
- `grep EXPECTED_TEST_COUNT` over `kaggle_kernel/g04divcap5-v1/main.py`,
  `kaggle_kernel/g1jump-submit/main.py`, `kaggle_kernel/bhap-submit/bh45.py` →
  **no matches**.
- `kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:100,182–185`
  → the gate quoted at the top of this report.

### E5 — The audit that would have caught it predates the V32 submit kernel

`reports/g04/lanes/division_permissive/SUBMISSION_PATH_IDENTITY_AUDIT_20260803.md`
(2026-08-03) audited **every** submission path for hard-coded runtime test
membership and returned PASS for all live paths. `kaggle_kernel/g02-v32-submit/`
was created **2026-08-05** and was never covered by that audit. The audit's own
closing rule applies verbatim: *"If a future code-output canary ever errors, the
KFG-25 diagnosis must be reopened rather than patched and resubmitted blindly."*

### E6 — The lifecycle guard cannot see this defect

`scripts/public_temporalunet_hidden_submit.py:140–142, 169–170` gate on exactly
three booleans: `hidden_test_compatible is True`, `dynamic_test_discovery is
True`, `public_identity_pinned is False`. The V32 kernel satisfies all three
**and still pins the count**. There is no cardinality check in
`scripts/public_temporalunet_hidden_submit.py` or in
`reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`.

### E7 — The prepared repair ladder inherits the bug in every stage

`scripts/v32_submission_bisect.py:115` and `:163–164` emit the same
`EXPECTED_TEST_COUNT` gate into the `step0_no_input_hash` template, and every
later cumulative stage builds on it. If H1 is correct, **stages 2–8 of the
prepared ladder would all fail identically**, spending up to seven further
submissions without localising anything, and the ladder would misattribute the
failure to the last stage it happens to reach.

### What the evidence does **not** establish

The exact size and membership of the rerun test mount is **not** directly
observed anywhere in the repo. The public competition download
(`data/kaggle/raw/biohub-cell-tracking-during-development.zip`, verified) contains
**4 test zarrs, 398 train entries, and a `sample_submission.csv` naming exactly
those 4 test datasets**. The retained V32 log line
`{"data":"Found 4 test videos\n"}` is from the *public* run, not the rerun.
H1 therefore rests on E1+E2 (the rerun membership demonstrably differs from the
public four) plus E4/E5/E6 (the V32 kernel is the only submitted kernel that can
reject such a difference), not on a direct count observation.

---

## (d) Minimal fix

### D1 — Remove the cardinality gate (the actual fix)

In `scripts/public_temporalunet_baseline.py` (the generator; the shipped kernel
is derived from it — do not hand-edit `kaggle_kernel/g02-v32-submit/`):

- **line 615–618**, replace

  ```python
  if len(selected) != EXPECTED_TEST_COUNT or len(set(selected)) != EXPECTED_TEST_COUNT:
      raise RuntimeError(f"STEP0 hidden inventory drift: {len(selected)} != {EXPECTED_TEST_COUNT}")
  ```

  with the `discover_submit_jobs` contract — reject only empty or duplicated
  inventories:

  ```python
  if not selected or len(set(selected)) != len(selected):
      raise RuntimeError(f"STEP0 hidden inventory invalid: {len(selected)} discovered, {len(set(selected))} unique")
  ```

- **lines 533–536**: demote `EXPECTED_TEST_COUNT` / `REQUEST_COUNT` /
  `REQUEST_FILE_COUNT` / `POPULATION_FILE_COUNT` from pinned constants to
  telemetry derived from `len(selected)` (the emitted receipt already reports
  `len(selected)`; only the module-level constants are frozen at 4).
- **line 545** (`COMPETITION_TEST`): resolve against both mount roots, mirroring
  `COMP_DIR_CANDIDATES` in the kernel body and `bh45.py:405–407` (closes H2).

Mirror the identical edit in `scripts/v32_submission_bisect.py:115, 163–164`
and its `step0_without_telemetry_source` template, otherwise every bisection
stage keeps the defect (E7).

Everything else — weights, thresholds, graph policy, the CSV writer, the 0.914358
CV57 result — stays byte-identical. This is a rerun-contract change only, so V32
immutability and the replication claim are untouched.

### D2 — Close the guard gap so this cannot recur

Add to `scripts/public_temporalunet_hidden_submit.py` (next to the
`public_identity_pinned` check) and to
`reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py`
a static refusal for any submission kernel that compares a runtime-discovered
test inventory against a constant — i.e. reject sources containing
`EXPECTED_TEST_COUNT`, or any `len(<discovered>) != <literal>` / `!= EXPECTED_*`
comparison on the competition mount. Register the new family as **KFG-26 —
"runtime test-inventory cardinality pin on hidden rerun"** in
`reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json`
and `KAGGLE_FAILURE_AUDIT.md`, distinct from KFG-25 (identity pin).

### D3 — Bound the input inventory (cheap, not the root cause)

Change `inventory_input` to record path, size and count for the competition
mount and drop per-file SHA-256 there (keep content hashing for the two pinned
support-pack weight files, which is what actually needs attestation). Measured
justification: 408 files / 1.906 GB / ≈37 s at 4 videos, scaling linearly with
the hidden inventory. Do this in the same change, but do **not** present it as
the fix.

### D4 — Evidence still missing, and the exact commands that produce it

1. **The rerun's own scriptVersionId, logs, and error text for `55272130`.**
   The prior retrieval attempt queried kernel versions `/1`, `/2`, `/3` and got
   the retained *public-run* artifacts plus HTTP 403. That is expected: **the
   submission rerun creates its own script version, distinct from the versions
   you pushed.** The precedent is in
   `reports/g02/kaggle_submission_failure_54890576_v1.json`, whose `url` field is
   `/code/onebitbrain/bhv45-submit?scriptVersionId=336990437` — a version id that
   is not 1, 2 or 3. Obtain the equivalent for `55272130`:
   - `kaggle competitions submissions … --format json` **drops** `errorDescription`,
     `totalBytes` and `url`; the raw authenticated REST response
     `GET /api/v1/competitions/submissions/list/biohub-cell-tracking-during-development`
     carries them. Fetch it through the authenticated CMUX/Chrome session (per
     the CLI-auth memory: the local token is not authorised — a 403 is a
     token/scope issue, never a rules prompt).
   - Then, with the rerun `scriptVersionId` in hand:
     `.venv313/bin/kaggle kernels output onebitbrain/g02-v32-submit -v <rerunVersionId>`
     and the UI log panel for that version. The single decisive line to look for
     is `Found N test videos` — or, if H1 holds, the STEP0 traceback
     `RuntimeError: STEP0 hidden inventory drift: N != 4`, which reports **N, the
     true hidden test count**, directly.
2. **Origin of `54717551`, `54893629`, `55212065`, `55251996`.** No repo artifact
   exists. The same raw submissions REST response gives each row's `url` /
   scriptVersionId, which identifies the originating kernel. Worth doing before
   the next submission: if notebook auto-submission is enabled on any
   competition-attached kernel, every push silently spends a daily submission
   slot and pollutes the ledger.
3. **Direct confirmation of the hidden test cardinality.** There is no
   zero-submission way to observe it — a rerun only happens on submission, and
   rerun outputs are not downloadable. Item 1 is the cheapest path (it costs
   nothing). Failing that, the D1 resubmission *is* the discriminating
   experiment: it changes exactly one named mechanism, so it is a mechanism
   change and not a blind retry under the no-retry doctrine.

### D5 — Order of operations

Do **not** launch the eight-stage ladder as currently built (E7). Apply D1+D2,
rebuild `g02-v32-submit` as a fresh slug, and submit that once. If it scores,
H1 is confirmed, the flagship CV57 0.914358 finally gets a public number, and
stages 2–8 are unnecessary. If it still returns blank, H2/H3 remain live and the
rerun log retrieved via D4.1 will separate them.

---

## Artifact index

| what | path |
|---|---|
| authoritative 13-ref listing | `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/submission_repair_v32/evidence/cmux_receipts/competition_submissions_2026-08-05T191913532266Z.json` |
| V32 submit receipt (ref 55272130) | `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/submission_receipts/competition_code_submit_2026-08-05T140233835944Z.json` |
| V32 kernel source (the defect) | `kaggle_kernel/g02-v32-submit/public_temporalunet_hidden_kernel.py:100,112,182–185,227` |
| V32 kernel generator (fix here) | `scripts/public_temporalunet_baseline.py:533–536,545,615–618` |
| bisection ladder (inherits defect) | `scripts/v32_submission_bisect.py:115,163–164` |
| lifecycle guard (gap) | `scripts/public_temporalunet_hidden_submit.py:140–142,169–170` |
| prelaunch guard (gap) | `reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py` |
| unscored V32 CSV | `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/g02_v32_submit_output_v1/submission.csv` (`11ff6284…e88b1`) |
| V32 run/step0 receipts + log | `reports/g02/screening/tiered_surface/branches/public_temporalunet_baseline_v1/submission_repair_v32/evidence/latest_retained_output/` |
| scored CSV (55203439) | `reports/g04/lanes/division_permissive/hidden_transport_v1/remote_output_v1_terminal/submission.csv` (`da73c754…84509`) |
| unscored CSV, byte-identical (55201829) | `reports/g04/lanes/division_permissive/code_transport_v1/remote_output_v1/submission.csv` (`da73c754…84509`) |
| scored CSV (55221497) | `reports/g04/lanes/data_pathology_repair/global_jump_public_transfer_v1/remote_output_v1_terminal/submission.csv` (`366a12d6…a8c8`) |
| 55201829 diagnosis | `reports/g04/lanes/division_permissive/code_transport_v1/TRANSPORT_INVALIDATION.json` |
| 54890576 diagnosis (Kaggle error text) | `reports/g02/kaggle_submission_failure_54890576_v1.json` |
| the fix that made scoring work in July | `reports/g02/scorer_submission_path_findings_v1.md` items 18–21 |
| dynamic-discovery reference implementation | `kaggle_kernel/bhap-submit/bh45.py` → embedded `bh45src.zip` → `scripts/run_bh.py:564 discover_submit_jobs` |
| submission-path audit that predates V32 | `reports/g04/lanes/division_permissive/SUBMISSION_PATH_IDENTITY_AUDIT_20260803.md` |
| organizer template + test inventory | `data/kaggle/raw/biohub-cell-tracking-during-development.zip` → `sample_submission.csv`, `test/` (4 zarrs), `train/` (398 entries) |
| organizer metric | `external/kaggle-cell-tracking-competition/metrics.md` |
