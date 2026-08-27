# Trackastra 6bba feature kernel — Kaggle parity stop

Node: `GA-TRACKASTRA-6BBA-FEAT-20260725`

Finalized: `2026-07-25 15:59 +08`

Decision: **STOP — Stage B exact first-video parity failed; no cache banked**

## Authorization boundary

The human authorized one private Kaggle GPU feature-generation workflow for frozen
Trackastra inference over exactly the 69 `train:6bba_` fit-train105 videos, and
subsequently directed it to proceed despite the original quota-reset timing risk.
The authorization did not cover training, 44b6 processing, a competition
submission, or RunPod. None of those actions occurred.

## Authenticated inputs

- Contract:
  `e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac`.
- Input manifest:
  `reports/g02/trackastra_6bba_feature_kernel_manifest_v1.json`,
  SHA-256
  `53789b467b529ef9d10be1bc9cf5d8391a1ec65733810f72de46f59a20cbb0c4`.
- Raw directory manifest:
  `reports/g02/trackastra_6bba_raw_directory_manifest_v1.json`,
  7,038 files covering exactly 69 contract identities, SHA-256
  `1a87473654dcfcc970daa06db4f8b2012af7a831720055980720f0449541c5ff`.
- Raw source archive: 87,393,127,165 bytes, SHA-256
  `538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19`.
- First-video parity identity: `train:6bba_edf14583`.
- Authenticated local v1 parity file SHA-256:
  `fed771aaf78df907460e8833a66d698067766e942258c088e31219e998115016`.
- Private assets dataset:
  `onebitbrain/trackastra-6bba-assets-v1`, version 4, ready.
- Private sparse-cache dataset:
  `onebitbrain/trackastra-6bba-cache-v1`, version 1, ready.
- Existing public support dataset:
  `pilkwang/biohub-tracking-support-pack-50ep-v1`.

The approved 87,393,127,165-byte raw archive upload completed, but Kaggle never
registered `onebitbrain/trackastra-6bba-raw-v1`: authenticated status, files, and
metadata calls returned 403 and the owner dataset listing omitted the slug. The
kernel therefore used the declared competition directory mount. Before Stage A it
authenticated every exact 6bba Zarr file against the raw directory manifest. It did
not read 44b6 member contents.

## Kernel

- ID: `onebitbrain/trackastra-6bba-feat`.
- URL: `https://www.kaggle.com/code/onebitbrain/trackastra-6bba-feat`.
- Private; internet disabled; `NvidiaTeslaT4`.
- Script SHA-256:
  `caf3e02e20f45dc688c16e3018269cae11bffee3a92c2930f675d1dca35613f8`.
- Metadata SHA-256:
  `b4d3f96915d84479cbbeb97c860e8c91d49d82d527857e7b43dcb8a7a0537f28`.

Versions 1 and 2 stopped before inference in approximately 31 and 27 seconds,
respectively, because the packaged source still imported the optional
`tracksdata` dependency through an unused sweep module. The dependency was removed
from the launch path, the focused suite passed, and an isolated import smoke test
then passed with `tracksdata` explicitly blocked.

Version 3 passed dependency startup and raw authentication, then completed Stage A
for every contract video:

- 69/69 videos;
- 110,471 proposals;
- 158,561 conflict rows;
- 5,824 transitions;
- 46,415,917 bytes of atomic Stage A output;
- first video: 3,605 proposals / 99 calls;
- Stage A completion at 2,854.53 seconds.

Stage B restored the pinned Trackastra source and checkpoint, enriched the first
video, and invoked the mandatory exact v1 parity assertion. At 2,932.35 seconds it
failed closed with:

`v2 numerical parity with v1 failed for row_features`

The scalar identity, feature names, selected-edge count, pool counts, inference
metadata, challenger/conflict arrays, offsets, transition targets, and geometry
arrays all passed before the failing `row_features` comparison. The contract uses
`numpy.array_equal`, so no tolerance or device-drift waiver is allowed.

Version 3 reached terminal `KernelWorkerStatus.ERROR` after approximately
2,938.18 seconds (0.816 T4-hours). Including the two short startup failures, total
consumption was approximately 0.83 T4-hours.

## Local gates

- Same-code-path local Stage A parity: 3,605 proposals / 99 calls / 5,409 rows.
- Same-code-path local Stage B exact parity: PASS.
- Focused tests: `6 passed`.
- Launch scripts: Python compilation PASS.
- Isolated dependency test: launch imports PASS with `tracksdata` blocked.

The local gate therefore passed, but the Kaggle Linux/Python/PyTorch execution did
not reproduce the authenticated local v1 `row_features` byte-for-byte.

## Retrieval and banking

Kaggle listed no downloadable output files for the errored version. No Stage A
artifact, 49-D cache, run archive, or final manifest was retrieved. The cache is
unusable under the contract, and the required independent 69/69 plus first-video
verification cannot be performed.

No `GA-TRACKASTRA-6BBA-FEAT-20260725` LEDGER row was added. This node is not BANK:
the exact parity stop condition fired, and the task also requires stopping after a
Kaggle run error. No fourth kernel was launched.

## Resume condition

A new authorized node must first isolate which of the 49 `row_features` columns
drifts on Kaggle and by how much, using a parity-only diagnostic that preserves its
failed payload or prints columnwise mismatch statistics. Any proposed tolerance,
canonicalization, or replacement of the authenticated v1 reference changes the
contract and requires explicit approval. Do not consume another full 69-video run
before that diagnostic passes.
