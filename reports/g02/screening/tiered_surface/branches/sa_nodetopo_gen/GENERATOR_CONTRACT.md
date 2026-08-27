# BLOCKPEAK-RESTORE-V1 generator contract

Status: **FROZEN BEFORE GENERATOR OUTCOMES**  
Frozen: 2026-07-28T09:56:53Z

This lane reads every frame of the 36 frozen fit105 `44b6` raw Zarr arrays. For
each nonoverlapping `4×16×16`-voxel block, it takes the first C-order maximum and
keeps the peak only when its raw intensity is at least the dataset's frozen
Zarr-metadata 0.999 quantile. Peaks are ordered by descending intensity and then
ascending `(z,y,x)`; at most 64 are retained per frame. Neither fit labels nor
masks enter this raw-proposal step.

The raw bank is frozen by canonical gzip and SHA-256 before a second, fit-only
identity audit joins proposals to same-time GT. The audit defines a false
proposal as one not selected by a one-to-one match within 7 µm. It is diagnostic
and cannot alter raw proposal coordinates, inclusion, order, or identity.

Only `restore_topology` actions may be emitted. A false proposal at an interior
frame joins to the nearest baseline node at `t-1` and `t+1` when both are within
16 µm. Each action declares the complete outgoing/incoming incident-conflict
boundary required by frozen N1. The deterministic population is capped at 50,
and every action and every requested per-video budget batch must compile through
frozen interface `2a72950c…f865`.

The fixed false-node budgets are `0,1,2,5,10,20,50`. Their curve is structural:
affected videos and N1-confirmed added/removed graph primitives only. No scorer,
graph metric, threshold selection, family opening, or tier move is authorized.
Passing this lane can only satisfy the authentic-generator prerequisite for a
later human-reviewed NODETOPO graph gate.
