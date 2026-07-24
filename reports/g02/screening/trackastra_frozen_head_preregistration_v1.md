# Frozen Trackastra linear-head preregistration

## Decision

**BANK DESIGN / EXECUTION PENDING.**

The official frozen Trackastra checkpoint has already passed the provenance,
CPU-feasibility, and five-video signal screens. The complete deployed proposal
pool also fits the checkpoint's 1,024-token limit. This phase locks the next
test before any labels are joined to Trackastra features:

- materialize one label-free 49-feature record for every one of the 110,471
  geometry-admissible 6bba proposals;
- fit a fixed linear head in five nested video-grouped outer folds;
- choose each fold's operating threshold on its disjoint calibration videos;
- require high held-proposal precision and support before reconstructing any
  graph;
- compare one held-fit candidate graph per video with exact APNCC on the same
  69 fit-train105 videos;
- keep validation12 and CV57 closed unless the graph gate passes.

No Kaggle, GPU, RunPod, validation12, CV57, or public-submission action is
authorized by this design.

## Locked identities

| Item | Identity |
|---|---|
| Candidate manifest | `model_candidate_trackastra_frozen_head_6bba_oof_v1.json`, SHA-256 `66c5bc23…396e` |
| Label-free cache contract | `trackastra_full_pool_feature_cache_contract_v1.json`, SHA-256 `58016f20…d66` |
| Nested OOF contract | `trackastra_linear_oof_graph_gate_contract_v1.json`, SHA-256 `89350403…da6` |
| APNCC runtime control | `appearance_direct_swap_contract_v1.json`, SHA-256 `cee547ed…cf7` |
| Prepared fit-only labels | `selected_graph_temporal_background_prepare_v1/prepare.json`, SHA-256 `c41bc216…1d89` |
| Frozen Trackastra source | commit `fb265888c8fce2ac16a362ebf182ccd5ea48e936` |
| Frozen checkpoint archive | SHA-256 `2ede0848…a0e` |

The cache builder additionally authenticates the exact 87,393,127,165-byte raw
archive, source files, checkpoint member, V45 scorer inputs, selected graph,
per-video proposal count, and transition count. Each video publishes an atomic
feature file plus a hash-bound checkpoint. Resume accepts only checkpoints
created by the same contract and implementation.

## Feature and label boundary

The feature cache is generated before label access. Every conflict row contains
the exact 48 APNCC/context features plus one frozen Trackastra
challenger-minus-incumbent association margin. The proposal token is fixed as
the row with the smallest APNCC pair-NCC margin at feature index 44; row order
breaks ties.

The later OOF runner joins only prepared fit-train105 labels by exact
`(source_id, target_id)` challenger identity. Unmatched full-pool proposals are
inference-only: they cannot affect normalization, head weights, threshold
selection, or proposal-precision metrics.

## Nested OOF design

For each outer held fold:

1. Assign videos by `sha256(identity)[0:8] mod 5`.
2. Use `(held + 1) mod 5` as calibration.
3. Fit on the remaining three folds only.
4. Standardize 49 features using fit-fold mean and scale.
5. Fit the single preregistered ridge least-squares binary head:
   `lambda = 0.01`, float64 normal equations, unpenalized intercept, no search.
6. Choose the maximum-support calibration threshold with at least 100 accepted
   labeled proposals and precision at least 0.95.
7. Score the held fold exactly once.

The proposal prerequisite is:

- at least 1,000 accepted held proposals overall;
- overall held precision at least 0.95;
- every fold accepts at least 100 proposals;
- every fold precision is at least 0.93.

Failure PARKs the candidate before graph reconstruction. Passing authorizes
only the exact held-fit graph screen.

## Exact graph gate

For every held video, the runner scores the complete authenticated feature
pool with that video's outer-fold head and threshold, then applies
node-disjoint greedy ordering and the banked APNCC cap. It reconstructs APNCC
and candidate fragments from the same base graph and requires:

- adjusted edge Jaccard delta at least `+0.005`;
- nonnegative raw edge Jaccard delta;
- nonnegative node-recall delta;
- at least one observation-distinct held fragment.

A pass opens validation12-only confirmation. CV57 remains a separate gate.

## Resource-gate evidence

The first 45-second idle preflight passed, but the first feature-cache launch
was stopped before publishing a checkpoint when system swap grew 33.88 MiB at
only 1.01 GiB process RSS. A following sleep-only control independently grew
54.87 MiB at about 1 MiB RSS, proving host paging rather than model pressure.
The zero-growth policy was not weakened. A later 61.6-second idle control
passed with 36% free memory and exactly zero swap growth, reopening the same
sealed resumable build. That run atomically published the first video (3,605
proposals and 99 calls) before another 11.50 MiB system-swap increase stopped
it at 1.06 GiB peak process RSS and 32% minimum free memory. The checkpoint
then passed an independent contract/code/file-hash resume authentication.
Execution remains PARKed on host state, not candidate evidence.

## Verification

- 19 focused Trackastra linear/cache/checkpoint tests plus five model-gate
  subtests pass.
- Both new scripts compile and expose their CLI contracts.
- Fold checkpoints bind the OOF contract, implementation hashes, exact
  fit/calibration/held identities, threshold, metrics, and artifact SHA-256.
- The candidate manifest passes the locked literature and local-resource gate.
