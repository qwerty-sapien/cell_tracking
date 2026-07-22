# Metric revalidation and graph-cost lever sweeps

Date: 2026-07-22

Branch: `codex/fragmentation-57-rollback`

Pre-edit checkpoint: `2fc4f3128710f9ee28510f3a2c5a8d90fedeae97`

## Outcome

The scorer is re-vendored and its seven parity fixtures are green. The patched
scorer leaves the 57-video division-off baseline exactly unchanged at
`0.8711885823490648`, so that value survives as the submittable baseline.

The first cheap lever has a real but small non-monotone signal: increasing
`learned_bonus` from `1` to `4` improves adjusted edge Jaccard by
`0.0009024638875084`. Bonuses `8` and `16` regress. The bounded joint sweep does
not beat `(learned_bonus=4, raw_distance_weight=0.05)`. Nothing was promoted;
production remains locked at `(1, 0.05)`, divisions remain off, and geometry is
unchanged.

## N0 — official scorer re-vendor

### Verified

- The parent repo now points its scorer checkout from
  `016845facc82e7f5f0c7a37f8da2a9432f51076b` to upstream main commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2`, `Merge pull request #2 from
  royerlab/metrics-fix`.
- The two working-tree files match that commit byte-for-byte:

  | File | Git blob | SHA-256 |
  |---|---|---|
  | `external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py` | `e536cdc9f0877542ab227ec701ef0fdbb667189a` | `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444` |
  | `external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py` | `9afa8630f3d7a294f9a25fca81cce1e0c7c7aeca` | `0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9` |

- Contrary to the task premise, `metrics.py` also changed upstream: it now drops
  nonconsecutive edges, collapses predicted edges that merge onto one matched GT
  edge, and caps predicted out-degree at two (`metrics.py:57-113`). The adjusted
  edge formula itself is unchanged and remains uncapped above one
  (`metrics.py:445-450`). Across both scorer files, the vendored-to-patched diff is
  359 insertions and 149 deletions.
- The division exploit closure replaces admission of every prediction fork
  reachable from a matched weak component with local parent-successor
  candidates. A candidate must connect a matched parent side to two distinct GT
  daughter lineages, must survive malformed/cross-component rejection, and is
  assigned one-to-one by bipartite maximum matching
  (`division_metrics.py:311-384`, `393-492`). Thus remote forks can no longer gain
  credit merely by being reachable through an over-connected component.
- The parity fixtures cover the new nonconsecutive-edge, duplicate-edge, and
  malformed-shared-child behavior (`src/celltrack_alpha/scorer_parity.py:61`,
  `107`, `176`). All seven pass. The frozen ledger records the checkout and both
  file hashes (`src/celltrack_alpha/scorer_parity.py:350-366`).
- Canonical parity report:
  `reports/g02/scorer_parity_v1.json`, SHA-256
  `bb52ceaffc7453bf7c2d607df7022c6be6c1289240db3ae69517bcd957535572`.

### Inferred

The upstream changes are broader exploit hardening than the original prompt
anticipated, but the invariant relevant to N1—the adjusted-edge formula—was
preserved. N1 empirically verifies that the new edge filtering does not alter
this cached holdout.

## N1 — patched re-score of 57 cached videos

`scripts/score_bh_cv57.py:63-88` now validates the green canonical parity ledger,
including the upstream checkout and both scorer hashes, before scoring.

| Metric | Pre-patch cached score | Patched score | Delta |
|---|---:|---:|---:|
| adjusted edge Jaccard | 0.8711885823490648 | 0.8711885823490648 | 0 |
| edge Jaccard | 0.8761665470208184 | 0.8761665470208184 | 0 |
| edge TP / FP / FN | not separately frozen | 34174 / 2736 / 2094 | — |
| division TP / FP / FN | 0 / 0 / 42 | 0 / 0 / 42 | 0 / 0 / 0 |
| node recall | 0.9903079300966847 | 0.9903079300966847 | 0 |
| final score | 0.8711885823490648 | 0.8711885823490648 | 0 |

The before/after files are now retained under
`_ARCHIVE/conflict_residual/reports/g02/run_bh_d4_conflict_edge_f0_holdout57_v1/`.
The patched result has SHA-256
`94d7b9823fbc9780ea7c0f07fd1d27079b5d58a2c4ef7d500ac6e2106ef6c075`.

**Decision:** `0.8711885823490648` survives as the submittable division-off
baseline.

## N2 — division connectivity re-test

### Verified

- The diagnostic now classifies candidates under the patched local-pairing rules
  (`scripts/diagnose_division_failure.py:104-106`). The 57-video report contains
  42 GT divisions: 39 `local_parent_fork_missing` and 3
  `detection_or_localization_role_missing`; the production prediction has zero
  fork nodes and scores `TP=0, FP=0, FN=42`.
- Report: `reports/g02/division_failure_holdout57_postpatch_v1.json`, SHA-256
  `10a8e480cbd0134f0fec18c1d302b388c4dd3ecbaf02f6eb3298f79eb40d2b44`.
- The held-out diagnostic bridge `596 -> 800` still changes the focused divider
  from FN to TP. It supplies a local matched-parent fork with evidence on two
  distinct daughter lineages, so it is not rejected by the patch
  (`tests/test_bh_division_holdout.py:39-60`; findings doc lines 14-29 and
  78-83).
- The inherited fit geometry envelope remains `19/83 = 22.891566%`. The focused
  bridge passes patched pairing but lies outside that envelope
  (`reports/g02/division_bug_investigation_holdout_prefix52_v1.md:29`).
- For a hypothetical `30 TP / 20 FP / 12 FN`, patched division Jaccard is
  `30 / 62 = 0.4838709677`; its weighted contribution is `0.0483870968`, and the
  total with fixed edge baseline is `0.9195756791`, not `0.9077` (findings doc
  lines 34-39).

### Inferred and bounded

The current empirical binding constraint is absence of any predicted local fork,
not a scorer pairing rejection: divisions are off. Geometry is still a
demonstrated feasibility constraint, while patched daughter pairing is an
additional eligibility condition. The division-off output cannot apportion all
42 failures between those two hypothetical downstream constraints. The 30/20
ladder is arithmetically possible but is not evidenced as reachable by the
current predictor.

## N3 — learned-bonus sweep

The sweep runner uses persisted `.bhc` detections and `PairContext` candidate
probabilities. It performs no detector forward pass; it reads original frames
only for the existing locked lazy-p20 refinement. The experimental unlock is
private and provenance-checked (`src/celltrack_alpha/bh_graph.py:238-265`,
`1053-1060`, `1967-2082`). Every sweep result is explicitly
`production_eligible=false`; submit preflight asserts `(1, 0.05)` and rejects
experimental provenance (`scripts/run_bh.py:697-702`, `2658-2659`).

| bonus | raw-distance weight | adjusted J | edge J | edge TP | edge FP | edge FN | node recall | components |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 0.8711885823 | 0.8761665470 | 34174 | 2736 | 2094 | 0.9903079301 | 48881 |
| 2 | 0.05 | 0.8714246055 | 0.8763877849 | 34180 | 2733 | 2088 | 0.9902194188 | 48843 |
| **4** | **0.05** | **0.8720910462** | **0.8770651616** | **34188** | **2712** | **2080** | **0.9897281047** | **48859** |
| 8 | 0.05 | 0.8670102112 | 0.8719586997 | 34118 | 2860 | 2150 | 0.9904868323 | 49172 |
| 16 | 0.05 | 0.8479018414 | 0.8528194255 | 33787 | 3350 | 2481 | 0.9890894216 | 50280 |

The bonus-1 replay matches N1 in every aggregate and has exact topology on all
57 videos. One synthesized coordinate differs by one voxel on this Mac versus
the original T4 fragment; the official aggregate is unchanged.

### Inferred

The peak at 4 supports the hypothesis that a learned-evidence cap of 1 was too
small, but only weakly: the gain is `+0.0009024639`. The failure mode above 4 is
not empirical over-linking. Component count rises, TP falls, and both FP and FN
rise, which is consistent with poorer link selection/pruning and more
fragmentation.

## N4 — joint cost sweep

N3 supplied the required signal. The bounded neighborhood is bonus
`{2,4,8}` by raw-distance weight `{0.025,0.05,0.1}`; the center column reuses N3.

| bonus | raw-distance weight | adjusted J | edge J | edge TP | edge FP | edge FN | node recall | components |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.025 | 0.8715940230 | 0.8765577722 | 34184 | 2730 | 2084 | 0.9902753033 | 48838 |
| 2 | 0.05 | 0.8714246055 | 0.8763877849 | 34180 | 2733 | 2088 | 0.9902194188 | 48843 |
| 2 | 0.1 | 0.8718223109 | 0.8767920802 | 34187 | 2723 | 2081 | 0.9901977439 | 48861 |
| 4 | 0.025 | 0.8720549393 | 0.8770233179 | 34189 | 2715 | 2079 | 0.9897312210 | 48860 |
| **4** | **0.05** | **0.8720910462** | **0.8770651616** | **34188** | **2712** | **2080** | **0.9897281047** | **48859** |
| 4 | 0.1 | 0.8717564285 | 0.8767313019 | 34182 | 2720 | 2086 | 0.9897004309 | 48829 |
| 8 | 0.025 | 0.8667882353 | 0.8717483518 | 34115 | 2866 | 2153 | 0.9904964450 | 49192 |
| 8 | 0.05 | 0.8670102112 | 0.8719586997 | 34118 | 2860 | 2150 | 0.9904868323 | 49172 |
| 8 | 0.1 | 0.8681098569 | 0.8730722999 | 34138 | 2833 | 2130 | 0.9905186546 | 49109 |

Machine-readable aggregate:
`reports/g02/graph_cost_sweeps_holdout57_v1.json`, SHA-256
`4b3e24412d1c791be92372d36502c67c2a435e68b787b79f1b2972f5401aec81`.
Per-setting reports and fragment hashes remain under
`reports/g02/learned_bonus_sweep_holdout57_workers/` and
`reports/g02/joint_cost_sweep_holdout57_workers/`.

**Decision:** the raw-distance interaction is shallow near bonus 4 and no joint
cell beats `(4, 0.05)`. Per the stop-before-promotion requirement, the candidate
was not promoted.

## N5 — division decision

Divisions remain off. No geometry was widened. The patched re-test preserves one
oracle proof of feasibility but does not establish a fit-safe rule that can
separate that bridge from false forks. There is therefore no basis to reopen the
production division path.

## N6 — archive sweep

`_ARCHIVE/README.md:1-17` tells agents to ignore archived material unless a user
explicitly requests it and records a one-line reason for every category.

| Category | Files | Bytes on disk | Reason |
|---|---:|---:|---|
| conflict residual | 174 | 215,527,424 | measured only about +0.0016 on 57 videos; line closed |
| pseudo SSL | 3 | 86,016 | never run; supervised-scaling gate answered it |
| compact head | 44 | 24,358,912 | no submission path; `f4cmp` rejected it |
| scored-partial | 4 | 2,994,176 | measured -0.0746 topology regression |
| unused Ultrack | 4 | 65,536 | never installed; wrong sparse-point fit |
| general-model | 922 | 55,472,128 | 921 superseded probes plus this run's parity-check output |
| failed BH9 | 232 | 617,312,256 | `bh9o6`/`bh9o7` failed-run output |
| Cruise staging | 52 | 18,366,464 | 49 drift blobs plus 3 stale script copies |

The active-tree residual counts are zero for `reports/general_model_*`,
`bh9o6`/`bh9o7`, `drift-*.bin`, and the three stale scripts. The protected
`f4gtr.json`, `division_failure_fold0.json`, nested `error_budget.md`, nested
`fragmentation.md`, and fixed-node oracle all remain outside `_ARCHIVE` and are
listed by exact path in `_ARCHIVE/README.md:13-17`.

Historical-only defaults in `scripts/sweep_bh_graph_weights.py:42-51`,
`scripts/diagnose_division_failure.py:40-45`, and `tests/test_bh.py:430-435` now
point explicitly to archived inputs. Production paths were not redirected to
retired artifacts.

## N7 — parity ledger trigger

The trigger has not fired. Locally there is only `bhv45-submit/1`; the sole `/2`
reference is the hypothetical `kernel_ref_if_uploaded` in
`reports/g02/kaggle_submission_failure_54890576_v1.json:6`. There is no v2 output,
score, or `bhdiag.json`. Consequently
`reports/g02/scorer_parity_v1.json:1` correctly retains
`calibration_pair_ledger.status="no_observed_kaggle_pairs"`; no local/site pair
or residual was invented. No Kaggle read, upload, submission, or compute action
was taken.

## Final verification and constraints

- `PYTHONPATH=src:scripts .venv313/bin/python -m unittest tests.test_bhg
  tests.test_bh tests.test_official_scorer_parity tests.test_score_bh_cv57
  tests.test_bh_division_holdout`: **90 tests passed**.
- `scripts/check_scorer_parity.py`: `overall_pass=true`.
- No detector inference, training, Kaggle action, geometry widening, division
  enablement, default promotion, push, or conflict-residual resumption occurred.

The completed work remains uncommitted after the required pre-edit checkpoint;
no push was performed.
