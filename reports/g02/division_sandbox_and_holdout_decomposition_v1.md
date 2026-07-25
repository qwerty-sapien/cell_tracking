# Division sandbox and 57-video decomposition

## Bottom line

The official division scorer does **not** require exact edge recovery or an
exact-time fork. It rewards a local fork on the matched parent side whose two
child branches contain matches from two distinct GT daughter lineages (a GT
daughter or its child can supply the branch evidence). This makes a one-frame
late fork a TP and allows a division TP even when the ordinary edge metric is
wrong around it.

Our current `add_safe_divisions` does not reproduce any sandbox fork. On the 13
fixtures that satisfy our graph invariants, direct replay adds zero edges. The
three-child fixture fails closed because our graph permits out-degree at most
two. In 51 counterfactual trials made by removing one outgoing fork edge and
asking the helper to restore it, the helper adds zero edges and restores 0/51.

On the current division-OFF 57-video baseline, error is concentrated by video
but not by family in the same sense as the cited public four-dataset breakdown:
10/57 videos carry 51.19% of the weighted adjusted-edge residual. Family
`6bba` carries 82.75%, but it also carries 84.48% of all edge-event weight and
has the *better* family score. The discriminating scenarios are crowding and a
high fraction of detections unmatched at the tight 6 um motion gate, followed
by positive node-count error; depth does not distinguish these videos.

## A. The 14 division sandbox cases

The scorer replay uses checkout `075fc5f5a52d11077f9dc2b074644618f26939e2`
and `division_metrics.py` SHA-256
`0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9`.
`TP/FP/FN` below are division-event counts, not edge counts. A case can contain
more than one GT event or predicted fork, so the labels are not mutually
exclusive at file level.

| JSON | Exact topology presented | Fork classification | Official TP / FP / FN |
|---|---|---|---:|
| `3_division` | GT is the linear chain `1→2→3`. Prediction is `5→6→7` with two additional children `6→11` and `6→15`, i.e. one three-child fork at `6(t1)`. | Fork `6` is spurious and evaluable: FP. | **0 / 1 / 0** |
| `complex` | GT has fork `3(t2)→{4,6}(t3)→{5,7}(t4)` plus independent chain `24→25→26→27→28`. Prediction has forks `9(t1)→{10,23}`, `11(t3)→{12,13}`, and `20(t3)→{21,22}`. | Late fork `11` spans the two GT daughter lineages and is TP. The other two forks are not evaluable FPs under the scorer's annotation/matching rules. | **1 / 0 / 0** |
| `complex2` | Same three predicted forks and same GT fork as `complex`; the coordinates and edge insertion order differ (`9→10` is listed last). | Same result: fork `11` is TP, showing edge order is irrelevant. | **1 / 0 / 0** |
| `division` | GT has successive forks: `62(t1)→{63,64}` and later `65(t3)→{74,75}`. Prediction has successive forks `67(t0)→{68,73}` and `68(t1)→{69,70}`. | Fork `68` recovers the first GT event (TP); fork `67` is FP; the later GT event is not recovered (FN). | **1 / 1 / 1** |
| `division_at_end` | GT is `61→62→63`, then `63(t2)→{64,66}→{65,67}`. Prediction has the matching fork `70(t2)→{71,73}` and an extra terminal fork `78(t4)→{79,81}(t5)`. | Fork `70` is TP. The post-annotation terminal fork `78` is ignored, not FP. | **1 / 0 / 0** |
| `division_node_stage` | GT fork `45(t2)→{46,59}`. Prediction contains fork `49(t2)→{50,55}` on one track and nested/overlapping forks `52(t1)→{53,56}`, `53(t2)→{54,57}` on another. | Fork `49` is TP. Forks `52` and `53` are evaluable false forks: two FPs. | **1 / 2 / 0** |
| `division_test` | GT fork `7(t2)→{8,9}` with grandchildren `201` and `125`. Prediction fork `13(t2)→{14,205}` continues to those two grandchild positions. | Fork `13` is a direct local TP. | **1 / 0 / 0** |
| `duplicated_division` | One GT fork `161(t1)→{162,173}`. Prediction contains two alternative forks at the same time: `165→{166,167}` and `177→{178,179}`. | Bipartite pairing assigns fork `165` as the single TP; equally plausible leftover fork `177` is FP. | **1 / 1 / 0** |
| `edges_wrong_division_ok` | GT has fork `187(t1)→{188,195}` in one component and fork `192(t2)→{193,194}` in another. Prediction has one fork `181(t1)→{182,184}` with continuations `182→183` and `184→185`; surrounding ordinary edges do not reproduce the GT graph. | Fork `181` is TP for the first GT event despite wrong edge topology. The second GT event is FN. | **1 / 0 / 1** |
| `late_division` | GT fork `3(t2)→{4,6}(t3)→{5,7}(t4)`. Prediction stays linear through `10(t2)→11(t3)` and forks one frame late: `11→{12,13}(t4)`. | Fork `11` is TP because its two child branches match the two GT daughter lineages through the grandchildren. | **1 / 0 / 0** |
| `merge_delay` | GT fork `7(t2)→{8,9}` with `9→125`. Prediction fork `13(t2)→{14,15}` with `15→124`; one branch's evidence is delayed to the grandchild. | Fork `13` is TP; grandchild fallback supplies the delayed branch evidence. | **1 / 0 / 0** |
| `node_conflict` | GT fork `45(t2)→{46,59}` plus independent chain `101→102→103→104`. Prediction has the same three-fork topology as `division_node_stage`, but the extra GT component and 12-unit match radius change full-graph assignments. | Fork `49` is TP. Forks `52` and `53` are ignored rather than FP under the resolved matching. | **1 / 0 / 0** |
| `simple` | Two linear GT chains and two linear prediction chains; there is no fork in either graph. | No division event exists: neither TP, FP, nor FN. | **0 / 0 / 0** |
| `successive_div` | GT has successive divisions `20(t1)→{5,7}(t2)` then `7(t2)→{8,9}(t3)`. Prediction mirrors this as `11(t1)→{12,13}(t2)` then `13(t2)→{14,15}(t3)`, plus `12→16`. | First fork `11` is TP. Second fork `13` is FP and the second GT division is FN: superficially mirrored successive forks do not guarantee both pairings. | **1 / 1 / 1** |

### Shapes that earn a division TP

| Rewarded shape | Sandbox evidence | What must be emitted |
|---|---|---|
| Direct local binary fork at the matched GT divider | `division_test`, `division_at_end`, `division_node_stage`, `merge_delay` | One parent-side node with two distinct child branches, each carrying a match to a different GT daughter lineage. |
| Fork one frame late, on the GT daughter stage | `late_division`, `complex`, `complex2` | A connected fork can occur after the annotated divider if its branches reach the two daughter/granddaughter groups. Exact fork time is not the objective. |
| Correct fork embedded in otherwise wrong edge topology | `edges_wrong_division_ok` | Division topology is scored separately from exact edge correctness. The local branch pairing matters. |
| One of multiple duplicate candidate forks | `duplicated_division` | At most one predicted fork can pair to a GT division; extra valid duplicates become FP. |
| One event in a successive/multi-event graph | `division`, `successive_div` | Each GT event needs its own unambiguous local pairing. Recovering one fork does not confer credit on the next. |

### Shapes that do not earn a division TP

| Non-rewarded shape | Sandbox evidence | Official result |
|---|---|---|
| Fork where GT is linear | `3_division` | One FP, no FN. |
| No fork where GT has a division | The unrecovered events in `division`, `edges_wrong_division_ok`, and `successive_div` | One FN per unpaired GT division. |
| Duplicate or competing fork left after one-to-one pairing | `duplicated_division`, `successive_div` | FP. |
| Locally evaluable fork whose branches do not form the required GT daughter pairing | Forks `52/53` in `division_node_stage`; fork `67` in `division` | FP. |
| Fork beyond the annotated/evaluable window | Fork `78` in `division_at_end`; unmatched forks in `complex` | Ignored: no TP and no FP. |
| Entirely linear graph with linear GT | `simple` | No division count at all. |

### `add_safe_divisions` versus the sandbox shapes

The comparison interprets each sandbox y-coordinate in the same physical unit
used by its `max_distance`, then converts it to y voxels before calling the
production helper (fixed y scale 0.40625 um/voxel). The checked
`src/celltrack_alpha/bh_graph.py` SHA-256 is
`76061d1bf550d09d73f3a07ffdb890fe9c2c39caf4e1abd6fe257f431421be66`.

| Check | Result |
|---|---:|
| Direct fixtures accepted by our graph invariants | 13 / 14 |
| Direct fixtures on which a new division edge was added | **0 / 13** |
| Direct failure | `3_division`: `GraphInvariantError: outdegree exceeds two` |
| Counterfactual trials (remove one fork edge, make its child orphan, rerun) | 51 |
| Original fork edges exactly restored | **0 / 51** |
| Any division edge added in those trials | **0 / 51** |

This is the structural answer to “what to emit”: the scorer wants a connected
local two-branch lineage motif, including late forks and grandchild evidence.
Our helper instead searches only an immediate next-frame orphan around a
one-child parent and applies the inherited `existing child ≤7.65 um`,
`candidate child ≤4.66 um`, and `sister ≤8.5 um` geometry plus caps. None of the
official sandbox fork motifs passes the complete helper path. Global tuning of
those constants is not enough evidence for activation; candidate generation
must first be able to construct scorer-valid daughter-lineage connectivity.

## B. 57-video decomposition

### Baseline and family split

The decomposed artifact is the current post-patch, division-OFF baseline at
`0.8711885823490648`, edge `TP/FP/FN = 34,174 / 2,736 / 2,094`, and division
`TP/FP/FN = 0 / 0 / 42`. The adjusted-edge aggregate is weighted per video by
`edge_tp + edge_fp + edge_fn`; all contribution shares below use that official
weighting. The source score artifact SHA-256 is
`94d7b9823fbc9780ea7c0f07fd1d27079b5d58a2c4ef7d500ac6e2106ef6c075`.

| Family | Videos | Edge-event weight | Family adjusted-edge score | Weighted residual | Share of total residual | FP / FN | Median detections/frame | Median NN distance | Median 6 um unmatched | Median node ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `44b6` | 21 | 6,053 (15.52%) | 0.856779 | 0.143221 | 17.25% | 540 / 363 | 261.5 | 8.286 um | 9.25% | -7.45% |
| `6bba` | 36 | 32,951 (84.48%) | 0.873836 | 0.126164 | 82.75% | 2,196 / 1,731 | 76.75 | 10.017 um | 9.17% | +6.69% |
| **All** | **57** | **39,004** | **0.871189** | **0.128811** | **100%** | **2,736 / 2,094** | **129.0** | **9.335 um** | **9.18%** | **+5.35%** |

`6bba` does not have a worse score; its large residual share is slightly *less*
than its edge-event exposure. `44b6` is the harder family per weighted event.
Therefore a prefix-only family router is not the main opportunity.

### Videos carrying the error

| Rank | Video | Family | Score | Residual share | FP / FN | Detections/frame | NN distance | 6 um unmatched | Node ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `6bba_786893ac` | 6bba | 0.652638 | 12.15% | 362 / 229 | 727.5 | 7.267 um | 9.16% | +16.57% |
| 2 | `6bba_f17befbc` | 6bba | 0.727617 | 6.52% | 187 / 124 | 307.0 | 8.125 um | 11.15% | +18.41% |
| 3 | `6bba_7f87b3d8` | 6bba | 0.718789 | 6.17% | 169 / 143 | 97.0 | 9.614 um | 26.58% | -2.31% |
| 4 | `6bba_3abfe10a` | 6bba | 0.759050 | 5.18% | 146 / 98 | 753.0 | 7.447 um | 8.68% | +19.67% |
| 5 | `6bba_bb9f20c3` | 6bba | 0.890004 | 4.37% | 117 / 93 | 251.0 | 8.444 um | 5.75% | +5.35% |
| 6 | `6bba_7af54fde` | 6bba | 0.839853 | 3.80% | 99 / 66 | 114.5 | 9.335 um | 7.67% | +25.08% |
| 7 | `44b6_d5e7d891` | 44b6 | 0.814785 | 3.68% | 104 / 66 | 544.0 | 7.083 um | 5.86% | +17.73% |
| 8 | `6bba_0c7fa718` | 6bba | 0.759648 | 3.67% | 108 / 76 | 103.5 | 8.900 um | 23.29% | +0.60% |
| 9 | `6bba_ebdf3b34` | 6bba | 0.828781 | 3.02% | 82 / 61 | 654.0 | 7.961 um | 7.07% | +11.49% |
| 10 | `6bba_3a1849c2` | 6bba | 0.890237 | 2.65% | 69 / 52 | 82.0 | 9.750 um | 7.12% | +10.94% |

Concentration is meaningful: the worst contributor alone supplies 12.15% of
the residual; the top 3 supply 24.84%, top 5 34.39%, top 10 51.19%, and 24
videos supply 80%. The complete 57-row ranked table is in the companion CSV.

### What distinguishes them

The top-10 contributors have median 279 detections/frame versus 129 in the
remaining 47, median nearest-neighbor spacing 8.284 versus 9.475 um, and median
positive node ratio +14.03% versus +2.38%. They are not uniformly fast: the
list mixes very crowded videos with low-density/high-motion videos such as
`6bba_7f87b3d8` and `6bba_0c7fa718`.

| Observable split | Low-risk quartile weighted score | High-risk quartile weighted score | Direction |
|---|---:|---:|---|
| Density: 46–69 vs 290–753 detections/frame | 0.941004 | **0.794928** | Dense is much worse. |
| Crowding: NN distance 11.49–15.33 vs 7.08–8.29 um | 0.946905 | **0.784552** | Close neighbors are much worse. |
| Tight-motion failure: 4.62–7.07% vs 12.24–27.26% unmatched at 6 um | 0.924131 | **0.833979** | High motion/association demand is worse. |
| Absolute node ratio: 0.60–4.94% vs 24.03–64.46% | 0.900607 | **0.857005** | Large count mismatch is worse. |
| Signed node ratio: -55.74% to -4.59% vs +13.72% to +64.46% | 0.909026 | **0.783177** | Overprediction is especially harmful. |

Across all 57 videos, residual-error Spearman correlations are `+0.433` with
detections/frame, `-0.495` with nearest-neighbor distance, `+0.372` with the
6 um unmatched fraction, and `+0.343` with signed node ratio. These directions
survive family demeaning (Pearson magnitudes 0.429, 0.523, 0.443, and 0.349).

All 57 videos have 64 z planes, so nominal depth has no variance. The occupied
depth-edge fraction has essentially no rank relationship to error
(`rho=-0.054`, `p=0.691`). Depth is not a supported routing variable in this
holdout.

## Decision

The evidence supports two scenario handlers rather than another global sweep:

1. **Crowded/high-density association:** route on detections/frame or, better,
   nearest-neighbor spacing. This is the dominant and cleanest split.
2. **Low-density/high-motion association:** route on the fraction unmatched by
   the tight 6 um pass; it captures failures hidden by a density-only router.

Node ratio is a useful secondary guard, especially against overprediction.
Depth and dataset prefix should not be primary routes here. Division should
remain OFF until a fit-safe generator demonstrably emits the local two-daughter
connectivity that the sandbox scorer rewards; the present helper emits none of
the 14-case motifs.

## Reproduction artifacts

- Analysis: `scripts/analyze_division_sandbox_holdout.py`
- Full machine-readable evidence: `reports/g02/division_sandbox_and_holdout_decomposition_v1.json`
- Complete ranked 57-video table: `reports/g02/division_sandbox_and_holdout_decomposition_v1_per_video.csv`
- Command: `PYTHONPATH=src:external/kaggle-cell-tracking-competition/src .venv/bin/python scripts/analyze_division_sandbox_holdout.py`
