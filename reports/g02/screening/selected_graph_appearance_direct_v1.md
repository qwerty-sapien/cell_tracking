# Selected-graph appearance direct swap — full CV57 confirmation

**Decision: BANK as the new local high. Public status: HUMAN-DECISION.**

The frozen 57-video scorer gives the appearance-aware direct-swap candidate
`0.8815220699597315`, an absolute gain of `+0.008372476430831166` over the exact
backward-motion `c=0.25` control at `0.8731495935289003`. This is the first
confirmed G-02 candidate above the local `0.88` barrier.

No Kaggle submission, Kaggle GPU job, RunPod launch, or paid-compute action was
performed.

## Frozen comparison

| output | adjusted edge Jaccard | edge TP | edge FP | edge FN | node recall |
|---|---:|---:|---:|---:|---:|
| exact `c=0.25` control | 0.873149594 | 34,205 | 2,685 | 2,063 | 0.990448275 |
| appearance direct swap | **0.881522070** | **34,320** | **2,459** | **1,948** | 0.990377056 |
| change | **+0.008372476** | **+115** | **−226** | **−115** | −0.000071219 |

The scorer reconciliation in
`obs_diff_selected_graph_appearance_direct_cv57_v1.json` reproduces both scores
and both confusion tuples exactly. The baseline tuple is the corrected
`34,205 / 2,685 / 2,063` identity, not the older
`34,174 / 2,736 / 2,094` run.

## Mechanism and leakage boundary

This is an association/post-solver mechanism change, not a detector-family or
learned-architecture claim:

1. Reconstruct the exact `c=0.25` legacy adjacent graph using the frozen edge
   scorer.
2. Run the unchanged one-frame gap closure.
3. Verify the post-gap selected identities against the frozen parent screen.
4. Propose node-disjoint alternative links only when both frozen gates pass:
   geometry probability `>= 0.605` and full 3-D NCC margin
   `>= -0.04768145084381103`.
5. Apply at most `max(16, ceil(0.005 * selected_raw_edges))` swaps per video,
   ordered by NCC margin and challenger identity; geometry is the frozen
   prefilter, not a secondary reranking score.
6. Remove every pre-prune outgoing/incoming edge incident to the displaced
   source/target endpoint, insert the challenger, and run the unchanged
   pruning, smoothing, and image-support projection.

The appearance decision is zero-mean, unit-norm NCC over centroid-aligned
`16 × 32 × 32` raw-image patches. It does not rebuild the dead P4 attention
layer and does not introduce a GNN.

The thresholds were derived using the existing grouped fit boundary. The
appearance audit used 105 fit-training videos and a disjoint 12-video
fit-validation set; it did not use audit, test, or public labels:

| support check | accepted | correct | harmful | precision |
|---|---:|---:|---:|---:|
| NCC-only at train-frozen threshold on validation | 172 | 164 | 8 | 0.953488 |
| support-adjusted NCC threshold on validation | 107 | 105 | 2 | 0.981308 |
| geometry + NCC + node-disjoint cap on validation | 38 | 38 | 0 | 1.000000 |

The deployment-support sample does **not** establish natural full-pool
acceptance: its sampled acceptance rate at the support threshold is 20.45%.
Consequently the registered per-video runtime cap is part of the model
contract, not an optional tuning knob.

## Integration evidence

The first integration globally altered edge probabilities and reran the solver.
It improved the full CV57 score only to `0.877400878` (`+0.004251285`) because
only 2,732 of 5,047 proposed challengers survived while 3,116 declared conflicts
remained. That integration is PARK.

The direct-swap integration eliminates that mismatch:

| tier | control | candidate | delta | proposals | realized | retained declared conflicts |
|---|---:|---:|---:|---:|---:|---:|
| registered top 10 | 0.721796529 | 0.736482209 | **+0.014685681** | 1,878 | 1,808 | 0 |
| full CV57 | 0.873149594 | **0.881522070** | **+0.008372476** | 5,047 | 4,855 | 0 |

Across the full replay, 6,381 declared conflicts and 1,347 additional incident
pre-prune edges were removed. The full CPU replay took 1,007.6 seconds.
Earlier `_direct_screen10_v1` and `_v2` directories are failed/superseded
attempts; `_direct_screen10_v3` is the authoritative top-10 result.

## Observation-level distinctness

The full candidate is `DISTINCT` from the exact control:

| component | result |
|---|---:|
| node Jaccard | 0.992972344 |
| raw/scorer-eligible edge Jaccard | 0.984736037 |
| raw edge disagreement fraction | 0.007690677 |
| node unmatched fraction | 0.003526219 |
| fork agreement | 1.000000 |

Node, edge, and fork components remain separate; no arbitrary scalar distance is
used.

## Artifact identities

- Full result:
  `reports/g02/selected_graph_appearance_direct_cv57_v1/report.json`
  (`sha256 8da96a3da9e66869192f446fc6d59fedd6016bdefe4fb272b136df68258a0df4`)
- Full observation comparison:
  `reports/g02/screening/obs_diff_selected_graph_appearance_direct_cv57_v1.json`
  (`sha256 83d32d49f869fa85fd85be9adca12bded29f33cb377392a140d9500ee3305ca6`)
- Appearance audit:
  `reports/g02/selected_graph_appearance_audit_v1.json`
  (`sha256 0dd616423fa4d1577b6013750193916cb23c75dadb8062af7f7eb43b97007089`)
- Support audit:
  `reports/g02/selected_graph_appearance_support_v1.json`
  (`sha256 f83fd33d4836ac616a84baff9fa53fabe0a78667acda226779e1764af8609cb9`)
- Combined support audit:
  `reports/g02/selected_graph_appearance_combined_v1.json`
  (`sha256 a34d912a0011cae9aca9cb1c7ea238007f14f6fae59587897fcbc4d09948d0c0`)

## Gate result

The local `>= +0.005` confirmation gate passes. BANK the mechanism as the new
local baseline candidate and preregister it as the first high-range public probe.
Do not infer an exact public score from the three historical same-family points:
P0B transfer remains unresolved above `0.88`, and displayed Kaggle values are
rounded intervals. Packaging and a public submission remain separate,
fail-closed steps behind explicit human approval.
