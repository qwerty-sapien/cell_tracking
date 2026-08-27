# 6bba point-context full-pool held-graph gate

Status: **IMPLEMENTED; MODEL VERDICT PENDING A ZERO-SWAP IDLE PREFLIGHT**

This is not a model-quality verdict. APNCC remains the banked `0.8815220699597315`
candidate, and validation12, CV57, Kaggle, GPU, and RunPod remain closed.

## Locked design

- Candidate: `compact-point-context-6bba-oof-v1`.
- Scope: the 69 `6bba` videos in fit-train105 only.
- Five deterministic outer folds:
  - held fold = outer fold;
  - threshold-calibration fold = `(held + 1) mod 5`;
  - model fit = the other three folds.
- Each video is held exactly once. Within every outer model, fit,
  threshold-calibration, and held identities are mutually disjoint.
- Proposal prerequisite: at least 25 aggregate held labeled proposals at
  precision `>= 0.95`.
- Exact graph prerequisite: adjusted edge-Jaccard delta `>= +0.005`, raw
  edge-Jaccard delta `>= 0`, node-recall delta `>= 0`, and at least one
  observation-distinct held fragment versus APNCC.
- A pass authorizes only validation12 confirmation. It does not authorize
  CV57, a Kaggle launch/submission, or external compute.

The immutable contract is
`reports/g02/point_context_oof_graph_gate_contract_v1.json`, SHA-256
`a374a9e04488e28c911385c7af7873a10f45dced630844af816a39e97ba46f53`.

The rotated contract additionally preregisters atomic fold checkpoints. A fold
becomes authoritative only after its fitted model, calibration result, and
held-proposal metrics are cross-bound and published by one same-filesystem
directory rename. Resume accepts a fold only when contract, implementation,
partition, model, artifact, and metric identities all match. Unpublished stage
directories are non-authoritative. Graph replay is deliberately not resumable.

## Deployment identity repair

The pilot scored labeled plus sampled-background tables and discarded proposal
identities. The new runtime instead:

1. enumerates every selected-graph conflict from
   `appearance_swap.proposal_feature_groups`;
2. applies the banked APNCC geometry prefilter;
3. constructs the exact 48-feature context token from the raw image volume;
4. retains the challenger pair and every displaced conflict pair beside the
   score;
5. applies parental probability thresholding, the same node-disjoint greedy
   policy, and the banked APNCC per-video cap.

No sampled-background proposal is used as a substitute for the inference pool.

A real dry-run on `train:6bba_edf14583` reconciled 52,323 selected raw edges
and produced:

- 198,861 sparse scorer candidate pairs;
- 144,677 raw selected-graph conflict proposals;
- 266,053 conflict comparison rows;
- 3,605 geometry-admissible transformer tokens;
- 99 transition windows, maximum 154 tokens/window;
- 17,425 extracted node visits;
- zero sampled-background inference proposals.

Fourteen focused tests pass, including score-to-identity alignment, limiting-row
selection, empty-pool behavior, nested partition disjointness, threshold-aware
metric aggregation, and fail-closed contract hashing.

## Guarded launch record

Six CPU-only launches stopped before completing the first fit because the unchanged
zero-swap guard observed global macOS swap growth:

| Attempt | Peak process RSS | Global swap growth | Result |
|---|---:|---:|---|
| v1 | 1.03 GiB | 4.69 MiB | stopped |
| v2 | 1.15 GiB | 131.56 MiB | stopped |
| v3 | 1.18 GiB | 46.07 MiB | stopped |
| v4, partition-at-a-time loader | 0.87 GiB | 99.00 MiB | stopped |
| v5, informed low-pressure retry | 0.845 GiB | 38.26 MiB | stopped |
| v6, detection-only cache reader | 0.568 GiB | 183.07 MiB | stopped |

System free memory remained above the locked 25% floor. The v4 refactor and v5
retry lowered the candidate process working set, but global paging continued.
This is a host runtime block, not evidence for or against the model.

## Memory-path repair and controls

The repeated full sparse-cache decoder was unnecessary during training-window
construction: that path needs only the authenticated node-to-frame registry.
`bh_model.load_sparse_cache_node_times` now streams the complete cache digest,
checks the exact model/source/batching identity, and validates only the canonical
detection registry. The fit loader uses that mapping and does not deserialize
pair-context tensors.

The largest 43,983,316-byte cache and a complete 41-video outer-fit partition
both passed the unchanged guard:

| Control | Peak process RSS | Global swap growth | Result |
|---|---:|---:|---|
| largest cache, full decoder | 0.669 GiB | 0 MiB | completed |
| largest cache, detection-only reader | 0.446 GiB | 0 MiB | completed |
| complete 41-video fit partition | 0.464 GiB | 0 MiB | completed |

The module import also completed at 0.382 GiB and zero swap. The first v6
screen was nevertheless stopped at 0.568 GiB. A narrower one-fold diagnostic
then stopped after 5.8 seconds on 6.19 MiB of global swap growth, before
reaching the already-cleared partition-load peak.

The decisive attribution controls are idle v2 and v3. `/bin/sleep 45` was
stopped after 8.7 seconds when system-wide swap grew 27.87 MiB, and again after
9.7 seconds when it grew 10.06 MiB; the guarded process used only 0.001 GiB in
both cases. Therefore the present counter is observing unrelated host paging.
It is not valid to label the point-context candidate as memory-unsafe, but it
is also not valid to waive the zero-swap policy while the host is in that
state.

## Completed resumable screen

Idle controls v4 and v5 each completed the required 45 seconds with exactly
zero swap growth. The unchanged v7 screen then trained all five outer folds in
two guarded stages. The first stage published folds 0 and 1 before unrelated
host paging stopped it; the second stage authenticated and resumed those exact
checkpoints, completed folds 2–4, and exited with zero swap growth.

| held fold | threshold | correct / harmful | held precision |
|---:|---:|---:|---:|
| 0 | 0.419340 | 505 / 7 | 0.986328 |
| 1 | 0.942237 | 440 / 17 | 0.962801 |
| 2 | 0.625046 | 484 / 53 | 0.901304 |
| 3 | 0.932032 | 162 / 0 | 1.000000 |
| 4 | 0.273221 | 359 / 41 | 0.897500 |
| **OOF aggregate** | — | **1,950 / 118** | **0.942940** |

The aggregate retained 2,068 labeled proposals and therefore cleared the
minimum support of 25, but it missed the locked 0.95 precision prerequisite.
The final guarded stage used 0.639 GiB peak process-tree RSS, completed with
zero swap growth, and kept validation12 and CV57 closed.

## Decision

**BANK the runtime/checkpoint gate; PARK the compact point-context candidate
before graph reconstruction.** The failure is cross-video precision transfer,
not support or execution feasibility. The fail-closed prerequisite correctly
prevented exact graph replay and any CV57 consideration. Do not retune the
thresholds against these held folds or reinterpret the earlier validation
pilot as confirmation.

Machine report:
`reports/g02/point_context_oof_graph_gate_v7/report.json`, SHA-256
`488687568ff5a72aa7ca81647421bf395a4d49f9f947c60845fa8f9f4a98ec36`.
