# Trackastra two-stage feature cache v2

Decision: **BANK implementation and 59/69 authenticated Stage-A checkpoints;
resume only after a clean host-paging window.**

## Why the implementation changed

The conventional combined builder restored V45/APNCC and Trackastra in one
process. Isolated diagnostics showed each stack could restore safely, while
co-residence repeatedly caused system swap growth. Contract v2 therefore
requires:

1. V45 plus raw images produce a lossless 48-D APNCC base cache.
2. A new process restores Trackastra with mmap/meta assignment, reads only
   authenticated detections and the base cache, and appends one aligned margin.

The legacy combined entry point now fails closed under the v2 contract.

## Verification

- Focused builder/cache tests: **24 passed**.
- Mmap/meta Trackastra-only diagnostic: peak RSS `0.4488 GiB`, zero swap
  growth.
- V45-only diagnostic: peak RSS `0.4230 GiB`, zero swap growth.
- First Stage-A video: 3,605 proposals, 5,409 conflict rows, 99 calls.
- First-video base parity: every identity, offset, geometry array, pool count,
  selected-edge count, and all 48 feature columns exactly match the
  authenticated v1 reference.

Implementation identities at the first Stage-A run:

| Artifact | SHA-256 |
|---|---|
| Stage-A builder | `61b6be888ddef7baf1787ff54411dbf6a9442a66d9d5a9a36a698995aecfd1a8` |
| Shared flatten/enrich implementation | `9d7ea7394c153ec52fb1d3e16f9d369ae36ee655fce437b684435b6620e504e3` |
| Stage-B enricher | `016e8c8412be4969bb474c28708887b903a59f6b094fabad6482846e6c1a5eb7` |
| Contract v2 | `e060b284da623a373a83c8dd690985503af945399518f4d77eb517fddb8128ac` |

## Current execution state

Stage A published 59 of 69 atomic video pairs (`.pt` plus checkpoint JSON),
covering the exact first 59 eligible 6bba identities. The initial guarded run
lasted 1,399 seconds, peaked at `0.839 GiB` process-tree RSS, kept at least 32%
system memory free, and stopped on `218 MiB` of system swap growth.

Two exact resumes failed closed during V45 restoration before processing a new
video:

- guard v2: `20.5 MiB` swap growth at `0.481 GiB` RSS;
- guard v3: `144.62 MiB` swap growth at `0.472 GiB` RSS.

A no-work 45-second control between them had zero swap growth and 36–40% free
memory. No workspace-owned training process remained. This points to
host-wide paging triggered during model restoration, not proposal workload or
model-stack co-residence.

Do not bypass the zero-swap policy. Preserve the 59 checkpoints and exact
Stage-A implementation hash; retry the remaining ten only after a stable
window. Stage B and the nested OOF/graph gates remain unopened.

No validation12, CV57, Kaggle submission, Kaggle GPU, RunPod, or paid action
occurred.
