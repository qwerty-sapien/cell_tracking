# Division investigation: frozen fold-0 holdout prefix-52

Status: **verified structural objective mismatch; production division output must default OFF.**

## Post-patch revalidation (2026-07-22)

Status: **the connectivity conclusion survives, but the old component-based
classification is historical rather than a description of the patched rule.**

Verified against official checkout
`075fc5f5a52d11077f9dc2b074644618f26939e2` (`division_metrics.py` SHA-256
`0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9`):

- The focused diagnostic bridge `596 -> 800` (submission IDs `667 -> 876`)
  still changes GT divider 3 from FN to TP. The patched scorer accepts it as a
  local matched-parent fork whose two child branches contain matches from the
  two distinct GT daughter lineages. It is therefore not an instance of the
  arbitrary weak-component reachability exploit that the patch removes.
- The regression at `tests/test_bh_division_holdout.py` now freezes both sides:
  baseline score `0`, bridged score `1` for divider 3.
- The current division-off D4 57-video artifact was classified under the
  patched local rule in
  `reports/g02/division_failure_holdout57_postpatch_v1.json`: all 42 GT
  divisions remain FN; 39 have the strict matched parent/daughter roles but no
  local parent fork, and 3 lack one of the required matched roles. There are
  zero predicted forks, topology candidates, TPs, and FPs, as required by the
  production division-off setting.
- The inherited fit-only geometry envelope is unchanged at `19/83 =
  22.891566%`. Because the focused bridge passes the stricter pairing rule but
  fails the inherited child/sister geometry, geometry remains the demonstrated
  candidate-generation constraint. This does not prove a full-holdout causal
  allocation while production divisions are off.

The patched division arithmetic for a hypothetical `30 TP / 20 FP` outcome is
`FN = 12`, division Jaccard `30 / (30 + 20 + 12) = 0.4838709677`, and score
contribution `0.0483870968`. Holding the post-patch D4 adjusted-edge baseline
fixed at `0.8711885823` gives `0.9195756791`, not `0.9077`. The ladder remains
an arithmetic scenario only: the patched scorer requires all 30 TPs to satisfy
local daughter-branch pairing, and the current division-off output supplies no
such candidates.

Scope and provenance:

- Prediction/scoring artifact: `reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json`.
- Frozen split and scorer were read only. No Kaggle write, submission, public artifact, paid compute, or split/scorer change was made.
- The artifact reports `division_tp=0`, `division_fp=133`, `division_fn=37` across 52 videos.
- A focused known-GT replay at `tests/test_bh_division_holdout.py` replays
  `train:44b6_267148e4`, GT divider graph ID 3. It now asserts the patched
  baseline remains 0 and the diagnostic bridge changes it to 1.

## Verified findings

### 1. What the frozen metric rewards

- `external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py:60-65` extracts the predecessor, divider, two daughters, and their children.
- `division_metrics.py:191-223` requires a matched node in the one-node stage and matches covering at least two daughter lineages.
- `division_metrics.py:226-257` computes weakly connected components through both matched and unmatched prediction nodes.
- `division_metrics.py:331-353` awards a TP only when one such spanning component also contains a predicted node with out-degree at least two.
- Node matching is exact in time: `.venv/lib/python3.14/site-packages/tracksdata/metrics/_ctc_metrics.py:162-177` groups both graphs by `t`, and `:199-215` matches one time group at a time. The frozen division scorer contains no explicit `abs(predicted_fork_t - gt_divider_t) <= 1` check. Its effective event window comes from the extracted predecessor/divider/daughter/granddaughter subgraph; fork timing is accepted only indirectly when that fork lies in a component spanning the required matched stages.

### 2. All 37 GT events fail before the fork test

The exact frozen prefix-52 predictions were replayed for all 24 division-bearing videos and all 37 GT events:

- 35/37 events have full one-node plus two-daughter stage coverage, but those matched nodes are split across prediction components (`association_components_disconnected`).
- 2/37 events lack a required matched stage (`detection_or_localization_stage_missing`).
- 0/37 events have a scorer-valid spanning component.
- Consequently 0/37 events can present a fork candidate to the global pairing, and `division_tp=0` is reproduced.

As a timing/proximity diagnostic (not a scorer condition), 24/37 GT events have some emitted fork at `GT t +/- 1`, but 0/37 has such a fork within the node-matching radius of 7 um. Thus timing coincidence alone does not connect the scored lineages.

This is a structural mismatch: `add_safe_divisions` ranks locally close orphan nodes, while the metric rewards a fork that connects the detected pre-split chain to both detected daughter lineages in one component.

### 3. Focused known-GT failure: `train:44b6_267148e4`, divider 3 at t=3

Pinned matching finds all six required nodes:

- component A: prediction graph IDs `415(t2) -> 596(t3) -> 792(t4) -> 992(t5)`, matching GT predecessor/divider/daughter/granddaughter;
- component B: `800(t4) -> 1008(t5)`, matching the other GT daughter/granddaughter.

There is an emitted fork at the exact GT time t=3, but it is 17.698704 um from the GT divider and belongs to an unrelated component. The only emitted fork in the video that full-graph node matching associates with any GT node is also unrelated: it is at t=14 and matches GT node 31, not the division event. The scorer therefore reports `TP=0, FN=1, FP=1`.

An oracle diagnostic adding the single bridge `596 -> 800` (submission IDs `667 -> 876`) changes divider 3 from 0 to 1 and the counts from `TP=0,FN=1,FP=1` to `TP=1,FN=0,FP=1`. This proves the missing connected fork is sufficient for a true positive, but it uses held-out GT and is diagnostic only; it is not an eligible production rule.

### 4. Distance checks use the intended node pairs, but exclude the scoring bridge

`src/celltrack_alpha/bh_graph.py:1610-1633` applies:

- existing-child distance to `(parent, existing child)`;
- candidate-child distance to `(parent, candidate child)`;
- sister distance to `(existing child, candidate child)`.

For the focused scoring bridge, the prediction-space distances are:

- parent to existing child: 1.816805 um (passes 7.65 um);
- parent to candidate child: 9.951052 um (fails 4.66 um);
- sister to candidate child: 11.170028 um (fails 8.5 um).

The analogous GT distances are 5.525597 um, 7.094851 um, and 11.504839 um. Independent fit-only evidence in `reports/g02/fit_division_geometry_fold0_v2.json` shows the inherited geometry envelope covers only 19/83 = 22.891566% of fit GT divisions; fit medians are 7.175806 um for maximum parent-child distance and 10.908399 um for sister distance.

Thus the constants match the V45 reference, but matching the reference does not establish alignment with the metric. No distance-pair inversion was found.

### 5. Postprocessing does not remove added forks

- `bh_graph.py:1775-1782` explicitly retains short components containing an out-degree-two node when `preserve_division_components=True` (the default at `bh_graph.py:124`).
- `bh_graph.py:1790-1805` retains every edge whose endpoints remain.
- `bh_graph.py:1905-1917` then smooths positions without changing edges.
- `bh_graph.py:1957-1971` adds divisions before postprocessing.
- Existing regression `tests/test_bhg.py:671-682` passes and verifies a short division component survives while a non-division short component is pruned.
- The frozen focus fragment itself still contains its unrelated emitted fork, independently confirming that postprocessing did not erase it.

## Inference and decision

- **Verified:** current division output is net-negative on the division term: 133 false forks and 0 true forks. **Inferred, not directly labeled here:** a false added fork edge is expected to contribute an edge FP when its endpoints do not match a GT edge; this investigation did not prove a one-for-one 133-edge increase.
- **Verified:** a true-positive topology is possible, but the only demonstrated repair is the held-out oracle bridge above. The current heuristic does not contain a fit-safe signal that identifies that bridge rather than the 133 false forks.
- **Inferred:** widening geometry alone is unsafe. It would increase candidates substantially, and the fit-only envelope statistics do not identify which candidate connects real daughter lineages at inference time.
- **Decision:** do not widen geometry or add holdout-derived rewiring. The public runtime now defaults to `division_enabled=False` at `src/celltrack_alpha/bh_graph.py:1930`; explicit opt-in is retained for future fit-derived work. Re-enable only after a fit-safe end-to-end test demonstrates at least one official true positive without an unacceptable FP/edge-FP increase.

## Commands/evidence checks

- `.venv/bin/python -m unittest -v tests.test_bh_division_holdout` — passes;
  patched baseline `0`, diagnostic bridge `1`.
- `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_bhg.PostprocessingTests.test_short_division_component_kept_while_other_short_component_pruned` — passes.
- `PYTHONPATH=src .venv/bin/python -m unittest -v tests.test_official_scorer_parity` — all 7 frozen-scorer parity tests pass, including the active division-FP fixture.
- Read-only all-event replay through `tracking_cellmot.division_metrics`: 37 events; classification counts 35 disconnected, 2 stage missing, 0 spanning, 0 fork candidate, 0 TP.
