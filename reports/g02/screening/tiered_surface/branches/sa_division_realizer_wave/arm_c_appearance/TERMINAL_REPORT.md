# Arm C — appearance/intensity redistribution

## Terminal status

`FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE`

Arm C did not reach prediction or scientific evaluation. No pre-existing
frozen cache both covers the complete held-44b6 universe and exposes physical
intensity summaries that can support the preregistered parent-to-daughters
redistribution calculation. Falling back to the raw archive, creating a new
cache, or treating learned V45 latents as additive intensity would change the
mechanism, so the arm stopped.

## Frozen scope and preregistration

- Corrected frozen inventory: 21 `44b6` plus 36 `6bba` fragments, not 36
  `44b6`. This arm selected exactly the 21 `train__44b6_*.csv` identities.
- Shared contract:
  `08f3e875e77a5c7f93c56ce779df93f4cbbed1bb8a9a7f183369452c9e29e647`.
- Preregistration SHA-256:
  `77805969b5b25e62f778c4fa5960d5027da283b51f083940c6559abd415d000b`.
- Standalone preregistration commit: `ff46bad`.
- Prediction code accepts no target/oracle/bundle path. Its closed gate emitted
  `FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE` and wrote no
  `PREDICTIONS.json`; prediction SHA and prediction commit are therefore
  not applicable.

## Cache authentication

The audit authenticated only declared, pre-existing cache paths:

1. `reports/g02/trackastra_h1_feature_cache/videos` is a genuine frozen
   appearance feature cache, pinned by manifest
   `d52a4a42cf16c4e43d146d1cebfde68192db095c7cf6dfa3cfd9b7e4d820c0cf`,
   but its 36 44b6 files are fit105 identities and overlap the 21 held identities
   at `0/21`.
2. `kaggle_kernel/bhcv57out_v3/cache` covers `21/21` and authenticates
   253,265,784 bytes under record-set SHA
   `9357342bed7ba13584bc71f71e715759775466d1f1c478c5415c3833079728a2`.
   Its schema exposes node IDs, coordinates, detection confidence, grid
   coordinates, and learned 32-D `source_feat`/`target_feat` association
   latents. It exposes no raw/excess intensity mass, moment, voxel patch, or
   equivalent physical-intensity field. The preregistration explicitly forbids
   relabeling those latent vectors as additive intensity.
3. The archived parity3 cache covers only `3/21` held identities.

Cache-audit SHA-256:
`d8ce829d26bd380eb1573e60e4aa6a84cda677a331d791243719a6fb07a71fd0`.

## Resource and firewall receipts

- Fresh idle preflight: 45.490 s, 0 MiB swap growth; receipt
  `e9ea4e0d8ef7b4aa98a3a15b824eeeda505d11fd72dcbf047de4c76b9b884514`.
- Guarded cache audit: 3.479 s, 554.094 MiB peak process RSS, 0 MiB swap
  growth; receipt
  `172d35741ca480bbafcc93449f5934c069226c70d1b1882cd3f3ee28c3f2b367`.
- Focused tests: 3/3 pass; receipt
  `b6eb7477e8944831de045741d6fb5fe432ec2ba006b1893f2d9c7d323c3779a9`.
- `6bba` paths read: 0. Raw archive/image reads: 0. Cache files
  created/extended: 0. Target/oracle/repair-bundle/scorer reads: 0.
  N1 calls: 0. Kaggle/GPU/RunPod/submission/DEAD actions: 0.

## Required outcome fields

Because prediction was blocked before a candidate could be sealed, scientific
metrics are not defined:

| Field | Result |
|---|---:|
| Division TP / FP / FN | not evaluated |
| Division Jaccard | not evaluated |
| Adjusted edge delta | not evaluated |
| Raw edge delta | not evaluated |
| Node-recall delta | not evaluated |
| Exact combined delta | not evaluated |
| Duplicate survivors | 0 |
| PASS | no — infrastructure/input fail-closed, no scientific verdict |

The unchanged PASS line remains combined delta `>0`, division Jaccard `>=0.30`,
nonnegative adjusted/raw/node deltas, and zero duplicates.

The required realized-FP sensitivity remains explicit: at 25 true-positive
forks, 0/5/10 false-positive forks yield division-only combined additions
`+0.0595 / +0.0532 / +0.0481`.

Machine result SHA-256:
`4e04cc3a8a182f5113d0a0f1414eed4e45b03d885613ad7d4a2145e79de365d9`.

## Barrier

Arm C contributes no consensus input to Arm D because it has no sealed
prediction file. Reopening the exact arm would require a separately authorized,
pre-existing full-coverage held-44b6 physical-intensity cache or a new
cache-building decision. This run did not authorize either.
