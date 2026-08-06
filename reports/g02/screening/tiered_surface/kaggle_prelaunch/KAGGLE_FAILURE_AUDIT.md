# G-02 Kaggle failure audit and non-recurrence controls

Date: 2026-08-03

Scope: every evidenced G-02/G-03/G-04 Kaggle dataset, kernel-startup, input-mount,
transport, code-submission, output-contract, and exact-parity failure through this date. The machine-readable
authority is `KAGGLE_FAILURE_REGISTRY.json`; terminal excerpts and exact
downloaded-log hashes are in `KAGGLE_TERMINAL_LOG_EXCERPTS.json`.

## Result

Thirty-two incidents collapse to twenty-five root-cause families. A retry with
an unchanged error signature is forbidden. A scientific kernel is launchable
only when the shared guard emits a fresh, push-eligible receipt for the exact
package bytes. `cmux_ops.py` now refuses every kernel push without that receipt;
same-kernel inline or remote step 0 must attest the Kaggle-visible mount before
heavy work. Timing is diagnostic after complete category verification, not a
standalone scientific failure.

| Family | Root cause | Exact observed error or symptom | Enforced prevention |
|---|---|---|---|
| KFG-01 | Remote source-registration drift | `missing or ambiguous dataset onebitbrain/trackastra-h1-fit105-cache-v1` | Exact local/remote `dataset_sources` equality |
| KFG-02 | Working-path artifact omission | `cache_dataset_manifest.json` absent from `/kaggle/working` | Authenticate every source-to-working copy |
| KFG-03 | Unused optional dependency on launch path | `tracksdata` imported through an unused sweep module | Isolated import smoke; self-contained code file |
| KFG-04 | Cross-platform exact-parity drift | `row_features` exact parity failed | One-video remote parity diagnostic before population work |
| KFG-05 | Heavy import before authenticated step 0 | `ModuleNotFoundError: No module named 'zarr'` | Reject heavy imports before top-level `STEP0` |
| KFG-06 | Required checkpoint absent | `expected one U-ext+-x3rd-149.0-4.6540.pth, found 0` | Resolve exactly one checkpoint by SHA and size before pip/model work |
| KFG-07 | Split bundle omitted requests | `FeatureError: no frozen feature requests found` | One content-addressed consumer bundle; positive exact request count |
| KFG-08 | Dataset CLI omitted or rebased nested files | READY dataset lacked nested request files; a one-level archive stripped the required `handoff/` prefix | Explicit archive wrapper depth plus owner round-trip path/hash inventory |
| KFG-09 | Mutable shared dataset exposed another lane's carrier | `bundle manifest content resolution expected one match, found 0` | One immutable dataset slug per lane |
| KFG-10 | Sibling Python module not packaged | `ModuleNotFoundError: No module named 'kaggle_selector_step0'` | Reject imports matching sibling `.py` files |
| KFG-11 | Platform-specific gzip OS byte | `expanded gzip transport is not byte-reversible` | Normalize gzip OS byte and regression-test runtimes |
| KFG-12 | Recursive lazy-mount traversal | `step 0 exceeded the five-second ... envelope` | Reject `rglob`/`**`; exact roots and small manifests only |
| KFG-13 | Wrong private-dataset mount template | `exact DINOv2 private-dataset manifest missing` | Require `/kaggle/input/datasets/<owner>/<slug>` plus remote attestation |
| KFG-14 | Repeated private-target 403 | authenticated status/download returned `403` | Stop after the first 403 unless target or authorization changes |
| KFG-15 | Preflight attestation transferred across kernel IDs | preflight passed, but scientific kernel mounted no manifest | Bind remote attestation to the exact scientific kernel ID |
| KFG-16 | Kernel title does not resolve to declared ID | CLI launched under the title-derived slug | Require normalized title slug to equal declared ID before push |
| KFG-17 | Ownership registry stale | active lane slug absent from the registry | Register every dataset and kernel slug in the same change that first uses it |
| KFG-18 | Guard not structurally enforced | pushes could bypass the existing guard | `cmux_ops.py` requires a fresh eligible guard receipt and re-hashes metadata/code before any status query or push |
| KFG-19 | Full-byte step-0 timing overrun | complete valid verification took `5.061960 s` | Category failures remain hard; elapsed time is a warning after verification |
| KFG-20 | Output parity used the wrong reference | valid APNCC output compared to a pre-APNCC baseline | Bind parity oracle, mechanism contract, submission, and fragments to the same successful run |
| KFG-21 | Kernel-source mount layout mismatch | registered cache absent at all assumed roots | Same-kernel mount attestation plus an immutable exact-root allowlist |
| KFG-22 | Supervisor output-path drift | valid nested output falsely classified missing | Declare exact output roots/files and verify their recursive inventory after download |
| KFG-23 | Declared offline wheels not installed | authenticated wheel bundle existed but `zarr` was not installed/importable | Install exact ABI-selected wheels and execute an isolated release-path import/decode probe |
| KFG-24 | Exact source-composition mount divergence | single-source probe passed while exact multi-source worker failed the carrier manifest | Probe the exact scientific source composition and record per-manifest existence, bytes, and hash before comparison |
| KFG-25 | Public-only code output on hidden rerun | refs `54890576` and `55201829` used frozen public identities and completed blank/zero-byte | Require dynamic mounted-test discovery, fresh output generation, exact-version download binding, and one-shot predicate consumption |

The recurring `mistune.py` and `nbconvert` warnings occur after a kernel has
already failed and are recorded as post-failure noise, not root causes.

## Shared launch gate

`kaggle_prelaunch_guard.py` enforces:

- exact dataset sources and CPU/GPU/TPU/internet properties;
- lane-exclusive dataset ownership;
- owner-qualified private-dataset mount roots;
- no recursive `/kaggle/input` discovery;
- a self-contained declared code file;
- no heavy imports before step 0;
- exact code, manifest, request-count, and wheel-count pins;
- a same-kernel inline or remote PASS attestation before heavy work;
- a push-eligible receipt containing the exact metadata, code, contract, and
  registry hashes; and
- structural enforcement in `cmux_ops.py`, which re-hashes the package and has
  no override for a missing or stale receipt.

The focused guard regression suite contains seventeen fixtures covering a valid kernel,
recursive traversal, wrong mount, heavy pre-step-0 import, sibling import,
source drift, cross-lane ownership, missing remote attestation, and slow remote
preflight, plus zero request count, missing checkpoint pin, and preflight heavy
work, cross-kernel attestation, title/ID slug drift, receipt eligibility, and
fresh package hashes. The `cmux_ops.py` suite adds forty-eight fixtures, including
missing/stale/cross-kernel push-predicate refusal, one-shot direct predicates,
latest-status enforcement, trusted credential routing, lock ownership, and
finite GPU accounting. The code-submission lifecycle
adds exact-version output binding, hidden-rerun invariants, atomic one-shot
predicate consumption, and explicit new-reference reconciliation. Two additional tests
execute the real DINOv2 step-0 code and prove that a
missing or empty request manifest fails closed in under five seconds.

## DINOv2 v3 release evidence

The dedicated CPU-only dataset is
`onebitbrain/g02-dinov2-r5-cpu-input-v1`, mounted at
`/kaggle/input/datasets/onebitbrain/g02-dinov2-r5-cpu-input-v1`.
Remote preflight version 1 completed with PASS in `0.054277081 s`, before heavy
work, and authenticated 1,391,458 requests, 77 request files, 77 population
files, three wheels, the pinned checkpoint, official-source archive, input
manifest, and frozen population manifest. The downloaded attestation SHA-256 is
`6553a6358fe6a57462a6206b792d95142f1a440812c19d21b82cba3946a174bc`;
the downloaded log SHA-256 is
`36de1af00ddafe43e3bb036aec6c2901a65fe4bd37c04ded91d48a5f11372973`.

DINOv2 CPU kernel version 3 was released only after this remote attestation and
the launch-phase shared guard both passed, but failed at `0.992640098 s` with
the same missing-manifest message. The metadata and path repair were present;
the invalid assumption was that a mount observed by one kernel ID attested a
different kernel ID. This is KFG-15 and v3 is terminal-invalid, with no heavy
work or scientific result. The strengthened post-launch replay
of the same gate also validates the checkpoint and population hashes, positive
request/request-file/wheel counts, and `heavy_work_started:false`. It uses
`enable_gpu:false`; it cannot
debit the 30-hour GPU budget.

The causal repair promoted the passing preflight lineage, but the CLI warned
that the changed title did not resolve to the declared ID and accepted the run
under `onebitbrain/g02-dinov2-r5-cpu-promoted-preflight`. This is KFG-16. The
guard now rejects title/ID slug drift, and local metadata has been restored so
the title resolves exactly to the declared ID.

The promoted CPU-only lineage later passed Step 0 in `0.053483 s`, then its
authenticated runtime projection was `38.662 h` and it failed closed before a
scientific result. This is a correct resource stop, not a transport retry case.
Its terminal debit is `0.1163353 CPU-h / 0 GPU-h`; the later complete local
DINOv2 evaluation failed both scientific gates, so no further R5 launch is
justified.

## Future package contract

Every new kernel package should be built from one immutable contract and should
ship one self-contained declared code file, exact source IDs/versions, exact
mount roots, a pure-stdlib input-authentication boundary before heavy imports,
an isolated cold-start self-test, and declared output roots/files. Preflight and
science must use the same kernel lineage. After completion, the supervisor must
verify the declared output inventory and semantic receipt before deciding
whether the artifact is eligible for competition submission; `COMPLETE` alone
is never eligibility.
