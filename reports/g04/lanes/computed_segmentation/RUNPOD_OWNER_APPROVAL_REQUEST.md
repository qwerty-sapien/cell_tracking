# Lane B RunPod owner-approval request

Status: **PREPARED, UNAPPROVED, NO EXTERNAL ACTION TAKEN**

This request authorizes a benchmark-first decision tree, not an immediate
production launch. No RunPod sign-in, upload, pod creation, charge, competition
submission, or LIVE_STATE/LEDGER change was made while preparing it.

## Decision requested

The proposed authorization has two stages:

1. Run exactly one raw frame (train:44b6_24264f12, timepoint 0) on every GPU
   class the owner authorizes. Each class gets one model initialization and
   exactly two post-initialization repeats, with a 1,800 billed-second hard
   limit. Total benchmark spend is capped at **$2.00**.
2. Permit atomic held-fit105+CV57 segmentation production only if the offline
   gate verifies all authorized classes have terminal benchmark receipts,
   selects an exact-output-parity class from measured timing, refreshes its
   on-demand price within one hour, and proves both the **$95 production compute
   cap** and **$100 total owner cap** pass. The $100 cap includes up to $2 of
   benchmark spend and a $3 noncompute reserve.

Recommended benchmark set: **A40, RTX 4090, L40S, and A100 PCIe**. Authorizing
all four does not assert that any is faster than the T4; it measures each before
selection. The approval template may instead name a smaller sorted subset.

## Dated planning prices

These are on-demand Pod planning rates retrieved 2026-08-03 from RunPod's
[official GPU pricing page](https://www.runpod.io/pricing), which says it was
updated 2026-07-27. They are **time-sensitive planning references**, not a
promise of price or availability:

| GPU class | VRAM | Planning USD/GPU-hour |
|---|---:|---:|
| A40 | 48 GB | $0.44 |
| RTX 4090 | 24 GB | $0.69 |
| L40S | 48 GB | $0.99 |
| A100 PCIe | 80 GB | $1.39 |

Production is on-demand only; spot/preemptible execution is excluded. RunPod
documents the candidate types in its [GPU-type reference](https://docs.runpod.io/references/gpu-types).
Storage is covered by the $3 reserve and must still be receipted; RunPod lists
its current storage rates separately on the
[official pricing page](https://www.runpod.io/pricing). A fresh no-purchase
price receipt is mandatory immediately before the local production gate.

## Scientific identity and parity gate

The benchmark reuses the authenticated Cellpose cyto3 model and license:

- model SHA-256:
  2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c
- BSD-3-Clause license SHA-256:
  fa908286fef07e1e8acd05adcdcd7bbb8c7ca4c02f563a4722dad28b94ce0b49
- CPython-3.12 runtime manifest SHA-256:
  0ccf485df9f9962b897037b9ea2a218992b220d587193954c115cab93f7e01e0
- raw logical-frame SHA-256:
  0afa1821c7f1b97845b5409f247c2457771270fba196799c753513653f53d641
- exact three-file raw contract SHA-256:
  0e0b2a696f44a3dcbecf18f1ebb13958b5a8ef63d4c30acf0d1d9efcea47b753
- voxel-cell-solidity-v2 feature contract SHA-256:
  910177ae086013bfdff04b1e9528526ff4961f170fbdba49af6f578a8a3436b7

Both repeats must be byte-identical to one another and to the authenticated T4
reference:

| Output | Required value |
|---|---|
| computed masks | 207 |
| labels SHA-256 | 97416a30f4358e59d01986b4875b06ec4928a1293733f0df51fae78c4649c6d0 |
| native wrfeat12 SHA-256 | b793c15963352178c047c50d8649f534a88ee28efafb14c6468fa02d7e4e41be |
| region payload SHA-256 | 803e13ff41a1b025f6035ca635c2ad22aee5e63e2d6c14f082e8815f566e2247 |

Any architecture-dependent output change fails closed. It is not silently
treated as an acceptable speed measurement; using it would require a separately
reviewed amended signature.

All outputs remain explicitly labelled **COMPUTED_SEGMENTATIONS**, with
authentic_masks=false, organizer_provided=false, and h3_authentic_masks=false.
The remote benchmark and production workers do not read detections, GEFF, GT,
graph scores, or submission data.

## Measured projection and cap

The gate makes no cross-GPU speed assumption. For each exact-parity class it
uses:

~~~
frame hard bound = ceil(1.25 * max(two measured repeat seconds))
init hard bound  = ceil(1.25 * measured initialization seconds)
max frames/shard = floor((36000 - init hard bound - 600) / frame hard bound)

projected GPU seconds =
  16200 * max(two measured repeat seconds)
  + total_shards * (init hard bound + 600)

risk-adjusted production cost =
  1.05 * projected GPU seconds / 3600 * fresh USD/GPU-hour
~~~

Held-fit105 and CV57 are sharded separately. The gate then adds all benchmark
spend and the $3 reserve. The class must pass both the $95 production-compute
cap and $100 total cap.

As a planning sanity check only, substituting the measured T4 timing
(30.217 s initialization, 43.460 s worst repeat) gives 26 shards and
210.186 risk-adjusted aggregate GPU-hours. At the dated A40 rate this is
$92.482 production compute and $95.482 before benchmark spend, so A40 fits even
with no assumed speedup. The other planning-rate classes do not fit at T4 speed
and would need their own benchmark to demonstrate enough speed. These numbers
do not unlock production.

## Atomic production handoff

After a passing local gate receipt, production must use exactly one selected
GPU class and the measured hard bounds. The existing production semantics are
reused:

- per-frame schema:
  celltrack-alpha.g04-computed-object-frame.v2
- complete-shard receipt:
  celltrack-alpha.g04-computed-object-production-shard.v2
- each frame NPZ/JSON is written to a temporary path and promoted with
  os.replace;
- the complete shard receipt is written only after every assigned frame is
  validated and hash-inventoried;
- partial, timed-out, duplicate, missing, and extra outputs are never
  manifest-eligible;
- the existing local assembler requires exact frozen-population coverage before
  producing detection-aligned wrfeat12.

The RunPod adapter may change filesystem and supervisor transport only.
Cellpose configuration, feature definitions, provenance, raw hashes, voxel-cell
solidity v2, artifact schemas, hard kills, and the assembler remain hash-bound
to the package.

## Approval mechanics

The exact package is:

- RUNPOD_OWNER_APPROVAL_PACKAGE.json
- package SHA-256:
  0ad49f21f3a7d9af51e593ded857940400368abf6490619f0f2e1ff61122c49f

RUNPOD_OWNER_APPROVAL_TEMPLATE.json is intentionally **UNAPPROVED**. To grant
authority, the owner/root coordinator must create a separate approval receipt
bound to the exact package hash, set status to OWNER_APPROVED, set
owner_approved=true, record identity/time, and list a sorted nonempty subset of
the four GPU classes. The budget values must remain exact.

The offline command below must fail with the checked-in template. With a valid
owner receipt it emits benchmark contracts only and still performs no external
action:

~~~
PYTHONDONTWRITEBYTECODE=1 .venv313/bin/python \
  scripts/g04_runpod_gate.py benchmark-plan \
  --package reports/g04/lanes/computed_segmentation/RUNPOD_OWNER_APPROVAL_PACKAGE.json \
  --approval OWNER_SIGNED_RUNPOD_APPROVAL.json \
  --output RUNPOD_BENCHMARK_PLAN.json
~~~

Production cannot be unlocked until all authorized benchmark results and a
fresh production-price receipt pass production-gate. That command emits only a
local receipt; it contains no RunPod client, credentials, upload, launch, or
submission capability.
