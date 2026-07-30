# SA-REPR Phase-1 terminal report

Date: 2026-07-28

Branch: `codex/sa-repr`

Verdict: **GO — ONE SUPPLIER CONTRACT FROZEN FOR THE PHASE-2 ROOT BARRIER**

## Outcome

The lane authenticated exactly one local microscopy-specific supplier:
`F2-CELLECT-PUBLISHED-64D-V1`, the published frozen CELLECT 3-D U-Net
64-channel per-voxel contrastive embedding sampled at existing released-detector
points. The immutable contract is `FROZEN_SUPPLIER_CONTRACT_v1.json`,
SHA-256 `fffb09c2865fd70010f5d93d5db9eb4cca48bac851c9abe2f77bbc9b27e83e70`.

This is supplier readiness only. It is not a graph, proposal, oracle, CV57, or
leaderboard result, and R7 remains PARK-B until a later consumer establishes
fit-only support/headroom and the unchanged graph gates.

## Frozen-input and provenance evidence

All required root inputs matched before action:

- N1 interface `2a72950c…f865`;
- grouped folds `ddb313cb…8769`;
- P0 `8ad548c1…9c06`;
- fit provenance `6a9139bf…f14f`;
- locked plan `a322724e…5116`.

The broad preregistration is `PREREGISTRATION_v1.json` at
`5dcd9801…52b06`. It scanned three declared local roots: 5,082 directories and
13,511 files. It classified 181 representation-named artifacts: 173 frozen
Trackastra association payloads (excluded from new-supplier use) and 8
CELLECT-related checkpoints. Two competition-fit CPU-smoke derivatives were
excluded; the six published upstream files were resolved to one three-part
CELLECT model family. No DINOv2/SAM target was authenticated.

The selected upstream checkpoint and source are exact:

- repository `https://github.com/zzz333za/CELLECT.git`;
- clean pinned commit `3586070926f7f1fd5d8df37456861d22bdc63236`;
- tracked U-Net weight `f3e8e730…c89ce5`;
- upstream model source `3b44f7c6…c17d3ba`;
- top-level GPL-2.0-only license text `8177f975…80643`.

The locally distributed weight is tracked by the pinned repository under that
top-level license; no separate weight-license file is asserted. Runtime used a
hash-pinned clean-room numeric-checkpoint loader and did not import the GPL
reference source.

## Support, runtime, and controls

The supplier-specific preregistration is
`CELLECT_SUPPORT_PREREGISTRATION_v1.json` at `65a06697…fdb8b2`.
The fresh synthetic-only support audit passed:

- strict load of exactly 2,541,972 parameters;
- input `1×2×32×32×8` -> finite embedding `1×64×32×32×8`;
- two fresh CPU loads byte-identical at `d4cf08e6…0ec4dd`;
- three exact point identities -> `3×64` vectors at `a9a10dc9…aa6a97`;
- load+forward `0.102941 s` and `0.067885 s` on the small synthetic witness;
- nonfinite input failed closed;
- no-feature output exactly replayed baseline bytes;
- within-frame embedding permutation is frozen as the Phase-2 mechanism control;
- guard peak RSS `0.279 GiB`, swap growth `0 MiB`.

The support result is `cellect_support_audit.json` at
`86dd3205…b479b`; provenance is `cellect_provenance_manifest.json` at
`2a4e358b…79d8b`.

No full-fit runtime projection is claimed from the small synthetic witness.
Therefore no GPU request is credible or emitted.

The focused mechanical and artifact-consistency suite passed 9/9. Its guard is
`guard_tests.json` at `913c9d96…2bd45`; the test preflight and guarded execution
both recorded 0 MiB swap growth.

## Authentic-mask audit

The known absence remains:

- 10,710 fit105 raw-manifest paths: 0 mask/wrfeat/regionprops candidates;
- 24,886 source-archive entries: 0 candidates;
- 351 sparse-cache files: 0 candidates;
- authentic complete mask/wrfeat suppliers: 0;
- pseudo-masks created: 0.

Raw intensity Zarr data and point coordinates were not relabeled as masks.
R2 remains PARK-B for an owner-supplied provenance-complete artifact.

## Stops and quarantine

Produced no feature cache, graph score, CV57, second-family run, swap
integration, GPU job, external download/service, submission, RunPod job, or
shared-state mutation. Archived CELLECT performance files encountered during
provenance discovery were quarantined and are not decision evidence here.
C2/K-best and Branch-B memo numbers were not used.

## Next gate

At the Phase-1 barrier, root may hash-freeze the supplier contract. Only then may
a Phase-2 consumer preregister actual fit-only raw-volume identity/runtime
support and the three fixed arms: exact no-feature baseline, CELLECT 64-D
supplier, and within-frame embedding permutation. A credible full-cache
projection must exist before any GPU request. Graph evaluation remains
`44b6 -> 6bba -> CV57`.

The final git/path allowlist audit covers 21 branch-local paths with 0
violations and confirms no protected shared-state file was modified.

## Strict-barrier schema amendment

After root's strict barrier rejection, `state_fragment.json` was amended
schema-only at SHA-256
`ead93c50a7c41f17fa88aabfcb5a598aa85da7dcd96b1da18a77f2fd1890fc3c`.
It now records `cv57_used=false`, `submission_action=false`, and
`runpod_action=false`, plus explicit `44b6_ran=false`, `6bba_ran=false`,
`cv57_ran=false`, and `graph_run_count=0`. No node, tier, evidence hash,
scientific result, or frozen supplier contract changed.
