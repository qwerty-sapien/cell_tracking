# OCC-SLICED-W1 Version-1 failure and repair plan

Date: 2026-08-09 SGT  
Failed kernel: `onebitbrain/g02-occ-sliced-w1-v1`, Version 1  
Classification: **KFG-09 INFRASTRUCTURE FAILURE / SCIENTIFIC UNMEASURED**  
Repair state: **PRIVATE DATASET CREATED / FRESH KERNEL VERSION 1 SENT ONCE**

## Evidence and cause

The owner-supplied terminal log records 169 files, 187,738,899 bytes, and 168
unique hashes below the mounted private dataset.  Step 0 then failed in 4.39 s
because the expected v6 consolidated bundle-manifest hash had zero matches;
`heavy_work_started` remained false.

Those file and byte totals exactly equal the prior authenticated inventory of
the same slug's DINOv2 v8 carrier.  That inventory contains the frozen
population manifest, all 36 occupied-swap request files, and the three required
wheels, but not the v6 consolidated bundle manifest.  Kaggle dataset metadata
attaches the slug's latest revision rather than the contract's documentary
`dataset_version: 6` field.  This is the already registered KFG-09 mutable
shared-dataset/wrong-carrier signature, not a numerical or scientific result.

Read-only `dataset_files` and `dataset_download` calls both returned HTTP 403;
no further private-dataset CLI read is permitted under KFG-14.  The supplied
kernel log plus the banked full inventory are sufficient to identify the cause.

## Causal repair

The repair follows KFG-09's registered prevention instead of adapting science
to the wrong carrier:

- new private lane-exclusive dataset:
  `onebitbrain/g02-occ-sliced-w1-input-v1`;
- 43 flat files / 9,881,374 bytes: dedicated manifest, original request and
  population manifests, 36 occupied-swap NPZ requests, three offline wheels,
  and Kaggle metadata;
- dedicated manifest SHA-256:
  `806c7fd4198792b651b2c76536b26f0ae53195fa6ad5cc20252c636b465d3a94`;
- fresh one-version kernel slug:
  `onebitbrain/g02-occ-sliced-w1-repair1-v1`;
- Step 0 now reads one exact dedicated-manifest path, logs existence/bytes/hash
  before resolution, and verifies every declared file by exact flat path, size,
  and SHA-256; there is no recursive discovery.

No crop, normalization, projection, histogram, A0, runtime, output, population,
or downstream gate changed.

## Local verification

- exact dedicated Step 0: PASS in approximately 0.0235 s;
- legacy mutable-carrier fixture: fail closed before heavy work;
- tampered request fixture: fail closed on byte drift;
- focused scientific kernel parity/safety: 11 PASS;
- shared prelaunch failure-regression suite: 17 PASS;
- static guard: PASS; inline guard: PASS / eligible;
- code SHA-256:
  `3b7990897dcf829435cc4731377b90dd65ce3e57522c9f47513b6f040193b10b`;
- metadata SHA-256:
  `aa2e9acec1c6cc34d7964b75da2b36657336dd719c48cace929564492ff3421d`.

The owner explicitly approved the private upload.  CMUX created the fresh
dataset once, verified it appeared in the owner's listing, and pushed
`onebitbrain/g02-occ-sliced-w1-repair1-v1` Version 1 exactly once.  Kaggle
returned `Kernel version 1 successfully pushed.`  There was no retry and, per
owner instruction, no kernel poll.  Terminal launch evidence is in
`REPAIR1_LAUNCH_RECEIPT.md`.
