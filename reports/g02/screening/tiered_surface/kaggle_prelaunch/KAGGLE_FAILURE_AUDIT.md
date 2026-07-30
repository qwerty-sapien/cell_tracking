# G-02 Kaggle failure audit and non-recurrence controls

Date: 2026-07-30

Scope: every evidenced G-02 Kaggle dataset, kernel-startup, input-mount,
transport, and exact-parity failure through this date. The machine-readable
authority is `KAGGLE_FAILURE_REGISTRY.json`; terminal excerpts and exact
downloaded-log hashes are in `KAGGLE_TERMINAL_LOG_EXCERPTS.json`.

## Result

Twenty incidents collapse to sixteen root-cause families. A retry with an
unchanged error signature is forbidden. A scientific kernel is launchable only
when the shared static guard passes and a separate remote, pure-stdlib step-0
kernel attests the exact Kaggle-visible mount in under five seconds.

| Family | Root cause | Exact observed error or symptom | Enforced prevention |
|---|---|---|---|
| KFG-01 | Remote source-registration drift | `missing or ambiguous dataset onebitbrain/trackastra-h1-fit105-cache-v1` | Exact local/remote `dataset_sources` equality |
| KFG-02 | Working-path artifact omission | `cache_dataset_manifest.json` absent from `/kaggle/working` | Authenticate every source-to-working copy |
| KFG-03 | Unused optional dependency on launch path | `tracksdata` imported through an unused sweep module | Isolated import smoke; self-contained code file |
| KFG-04 | Cross-platform exact-parity drift | `row_features` exact parity failed | One-video remote parity diagnostic before population work |
| KFG-05 | Heavy import before authenticated step 0 | `ModuleNotFoundError: No module named 'zarr'` | Reject heavy imports before top-level `STEP0` |
| KFG-06 | Required checkpoint absent | `expected one U-ext+-x3rd-149.0-4.6540.pth, found 0` | Resolve exactly one checkpoint by SHA and size before pip/model work |
| KFG-07 | Split bundle omitted requests | `FeatureError: no frozen feature requests found` | One content-addressed consumer bundle; positive exact request count |
| KFG-08 | Dataset CLI omitted nested files | READY dataset lacked nested request files | Flat/archive transport plus owner round-trip inventory |
| KFG-09 | Mutable shared dataset exposed another lane's carrier | `bundle manifest content resolution expected one match, found 0` | One immutable dataset slug per lane |
| KFG-10 | Sibling Python module not packaged | `ModuleNotFoundError: No module named 'kaggle_selector_step0'` | Reject imports matching sibling `.py` files |
| KFG-11 | Platform-specific gzip OS byte | `expanded gzip transport is not byte-reversible` | Normalize gzip OS byte and regression-test runtimes |
| KFG-12 | Recursive lazy-mount traversal | `step 0 exceeded the five-second ... envelope` | Reject `rglob`/`**`; exact roots and small manifests only |
| KFG-13 | Wrong private-dataset mount template | `exact DINOv2 private-dataset manifest missing` | Require `/kaggle/input/datasets/<owner>/<slug>` plus remote attestation |
| KFG-14 | Repeated private-target 403 | authenticated status/download returned `403` | Stop after the first 403 unless target or authorization changes |
| KFG-15 | Preflight attestation transferred across kernel IDs | preflight passed, but scientific kernel mounted no manifest | Bind remote attestation to the exact scientific kernel ID |
| KFG-16 | Kernel title does not resolve to declared ID | CLI launched under the title-derived slug | Require normalized title slug to equal declared ID before push |

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
- a remote PASS attestation with step 0 below five seconds.

The focused regression suite contains fourteen fixtures covering a valid kernel,
recursive traversal, wrong mount, heavy pre-step-0 import, sibling import,
source drift, cross-lane ownership, missing remote attestation, and slow remote
preflight, plus zero request count, missing checkpoint pin, and preflight heavy
work, cross-kernel attestation, and title/ID slug drift. Two additional tests
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

The already-accepted CPU-only run remains active. It was still `RUNNING`
379.003 seconds after the push receipt, past every prior step-0 failure time.
Because the code cannot begin pip/model work until step 0 returns, the
missing-manifest and slow-traversal signatures did not recur. CPU usage is not
booked until its terminal receipt; GPU debit is zero.
