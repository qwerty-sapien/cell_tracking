# P0A scorer semantics and P0B transfer calibration

## Verdict

P0A: **PASS**

P0B: **STILL-UNRESOLVED**

The patched local implementation is byte-for-byte attested against the current
official `metrics-fix` checkout and passes its deterministic fixtures. This is
P0A and authorizes leakage-safe local scoring, decomposition, diagnostics, and
candidate development.

Three completed local-57/public pairs identify only a coarse relationship. The
two near-baseline configurations reverse order: `c=2` is lower locally by
0.00099965 and higher publicly by one displayed bin. This is a P0B
local-to-Kaggle transfer failure, not a scorer-code failure. It blocks fine
public forecasting, interpretation of new public probes, and S1 final
selection; it does not lock P1, B1/B2/E1 consolidation, or local model work.

## Available local/public pairs

I searched `reports/g02/scorer_parity_v1.json`, the G02 submission ledger and
failure record, the G01 calibration ledgers/manifests, and report text that
mentions public, site, Kaggle, submission, leaderboard, or calibration scores.
The baseline record and two approved calibration submissions have both a
numeric local 57-video score and a displayed Kaggle public score:

| Submission | Configuration | Local CV57 | Public | Public - local |
|---|---|---:|---:|---:|
| `onebitbrain/bhp0-c0/1` | divisions off; backward motion; `c=0` | 0.8483945156 | 0.8600000000 | +0.0116054844 |
| `onebitbrain/bhp0-c2/2` | divisions off; backward motion; `c=2` | 0.8721499475 | 0.8780000000 | +0.0058500525 |
| 54910497, `onebitbrain/bhv45-submit/3` | divisions off; backward motion; `c=0.25` | 0.8731495935 | 0.8770000000 | +0.0038504065 |

The local evidence is
`reports/g02/phase1_e3_backward_motion_coefficient_cv57_v1/report.json`.
The baseline submission identity and Kaggle API score are frozen in
`reports/g02/scorer_parity_v1.json`; all three fit inputs are frozen in
`p0_public_calibration_v1.json`.

Records excluded from the fit:

- Submission 54890576 has local score 0.8711885823 but no public score; its
  hidden-data rerun ended with `Notebook Threw Exception`.
- `reports/g02/kaggle_submission_ledger_v1.json` contains an unsubmitted
  reservation with both scores null.
- The five G01 calibration-probe entries have local scores but no Kaggle
  scores; their ledger status is `awaiting_kaggle_scores`.
- Submission 54633411 and the repeated 0.902/0.903 references in
  `reports/g02/v45ref.json`, `reports/g02/v45diag.json`, and related reports
  describe the pre-rescoring V45 public-notebook/account result. The frozen
  record explicitly says the scored CSV hash differs from the local replay
  CSV hash, and there is no exact patched 57-video score for that scored
  artifact. These values therefore cannot be mixed into the post-rescoring
  parity fit.
- `public_kaggle_score: 0.903` fields in older rung reports such as
  `reports/g02/s70.json` are marked `external_reference` and
  `used_in_local_comparison: false`; they are not submission pairs.
- Historical local experiments that do not identify an exact submitted
  package/public score are not pairs and were not imputed.

## Local versus public plot

```text
public score
0.879 |
0.878 |                                  ● c=2
0.877 |                                    ● c=0.25
0.876 |
  ... |
0.860 | ● c=0
      +----------------------------------------
       0.848             0.860          0.873
                     local CV57 score

       The two rightmost points reverse local/public rank.
```

## Fit, rank correlation, residual, and confidence band

The ordinary least-squares affine model is:

`public_hat = 0.2497831845 + 0.7193024335 * local`.

For forecasts relative to the known current baseline, use the anchored form:

`public_hat = 0.877 + 0.7193024335 * (local - 0.8731495935)`.

Thus a local delta maps coarsely to
`delta_public_hat = 0.7193024335 * delta_local`.

| Quantity | Result |
|---|---|
| Pair count | 3 |
| Affine slope | 0.7193024335 |
| Affine intercept | 0.2497831845 |
| Pearson correlation | 0.9963791653 |
| Spearman rank correlation | 0.5 |
| Kendall rank correlation | 0.3333333333 |
| R² | 0.9927714411 |
| Residual standard deviation | 0.0012163244 with only 1 residual degree of freedom |
| Rounding-only slope range | 0.6781267088 to 0.7604781582 |
| Baseline fitted public score | 0.8778418119 |
| Baseline 95% mean-response band | 0.8666863880 to 0.8889972358; very wide because `df=1` |

The high Pearson correlation and R² are dominated by the deliberately distant
`c=0` negative control. Rank statistics expose the near-baseline failure. The
full numeric calibration is frozen in `p0_public_calibration_v1.json`.

## Do local deltas track public deltas?

**Only at coarse scale.** Relative to the baseline:

- `c=0`: local delta -0.02475508, public delta -0.017; sign matches and the
  public magnitude is 0.687 times the local magnitude.
- `c=2`: local delta -0.00099965, public delta +0.001; sign flips.

Public scores are displayed to 0.001, so the exact size of the second reversal
is unresolved, but its displayed ordering is not monotone. No affine or
isotonic scalar calibration can make a locally lower score rank above a higher
one; isotonic calibration can only pool the two as an approximate 0.8775 tie.
The evidence therefore supports the affine formula for coarse absolute-score
forecasting, not fine candidate selection.

An operational heuristic is to treat `abs(delta_local) < 0.004` as unresolved:
approximately two fitted public residuals plus one display bin, transformed
through the fitted slope. This is not a statistical confidence guarantee. It
also exceeds the campaign's historical 0.002 effect ceiling, so it does not
rescue the current screening design.

Pure display precision alone corresponds to 0.00139024 local-score units per
0.001 public tick. The wider 0.004 abstention zone additionally reflects the
observed fit residuals. No affine line intersects all three public rounding
bins.

## Patched implementation versus the official scorer

### Provenance

- External checkout: `075fc5f5a52d11077f9dc2b074644618f26939e2`,
  `Merge pull request #2 ... metrics-fix`.
- Local `HEAD` and the locally recorded `origin/main` of the external checkout
  both resolve to that commit.
- The checkout is clean.
- Current SHA-256 values exactly match the frozen
  `reports/g02/scorer_parity_v1.json` attestation:

| Source | SHA-256 |
|---|---|
| official `metrics.py` | `cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444` |
| official `division_metrics.py` | `0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9` |
| local `official_eval.py` | `22f12e0eb7568c364a8503b8eeb9167147c152f182c2aa10dc10aef588bc01e6` |
| local `scorer_parity.py` | `bce55ba267397cd28d12cbbb2de22af8835232ce65b947a136c19cd097ab90a1` |

`src/celltrack_alpha/official_eval.py` does not carry a stale metric copy. It
adds the external `src` directory to `sys.path` and imports `evaluate`,
`per_sample_metrics`, and `summarise` directly from the official
`tracking_cellmot.metrics` module. `scripts/score_bh_cv57.py` validates the
frozen source hashes before scoring, and `scripts/scoreall.py` calls
`official_metric_row` per video followed by the official `summarise`.

### Requested semantic diff

| Area | Official patched behavior | Local full CV57 behavior | Divergence |
|---|---|---|---|
| Division credit | Directed local parent/daughter topology, distinct predicted branches, fork-to-division bipartite matching, cross-component and malformed-fork FP accounting | Calls the official `evaluate`, which calls the current official division implementation | None |
| Gap closing | Only edges with `t_target - t_source == 1` remain scoreable; backward and multi-frame edges are dropped | The official `_evaluate_matched_graph` performs the same filter after wrapper graph conversion | None |
| Edge matching | 7 µm time-aware node assignment; duplicate matched edges, merge duplicates, and out-degree >2 extras are capped/dropped as patched upstream | Calls the same official functions with per-video scale and 7 µm maximum distance | None |
| Node versus edge weighting | Edge Jaccard is multiplied by the `alpha=0.1` predicted-node-count adjustment; node recall is diagnostic, not an additive score term | Calls official `per_sample_metrics` unchanged | None |
| Normalization | Per-video adjusted edge Jaccard weighted by `W=TP+FP+FN`; division TP/FP/FN micro-averaged; final score adds `0.1 * division_jaccard` | Calls official `summarise` unchanged | None |

### Wrapper-only behavioral differences

These are concrete differences, but none changes the full CV57 score on valid,
non-empty competition samples:

1. `prediction_to_graph` silently drops dangling endpoints and self-edges and
   de-duplicates repeated source/target rows while constructing a valid graph.
   The official command-line scorer starts from already materialized GEFF
   graphs. Duplicate scoring is also guarded upstream.
2. Empty-GT diagnostic clips return a NaN row and are skipped locally; the
   current official implementation can fail on that degenerate Polars join.
   Full competition videos have GT nodes.
3. `official_edge_metric_row` intentionally sets division counts to zero for
   cheap edge-only probes. It is not the full scorer used by
   `scripts/score_bh_cv57.py`; final CV57 scoring uses
   `official_metric_row`.
4. The wrapper materializes per-row `division_jaccard` for diagnostics. The
   run-level official `summarise` recomputes the division Jaccard from summed
   TP/FP/FN, so this does not alter normalization or the final score.

No implementation fix is indicated by the current fixtures. The root-gate
failure is now an observed transfer/ranking mismatch between the 57-video
holdout and the public set, not a missing calibration series. If a strict
byte-equivalent CLI boundary is desired, the wrapper-only sanitation and
empty-GT behavior above should be reconciled or explicitly rejected before a
future PASS.

## Verification

- `.venv/bin/python scripts/check_scorer_parity.py --output /tmp/p0_scorer_parity.json`
  returned `overall_pass: true` across 7 full and 7 edge-only fixture cases,
  including the patched division and summary cases.
- `.venv/bin/python -m unittest -v tests.test_official_scorer_parity tests.test_score_bh_cv57`
  passed all 10 focused tests.
- Current source hashes were recomputed with `sha256sum` and match the frozen
  ledger.

## Gate action

Continue local work under P0A. Keep public forecasting and S1 unresolved under
P0B. The canonical comparison and six-rung dispersion ladder now exist:
all 15 full-CV57 rung pairs are observation-distinct, and APNCC-direct is the
only rung above local 0.88. The next action is therefore the already
preregistered single APNCC-direct probe. Charge it to the
five-per-competition-day cap and obtain explicit human approval before launch
or submission. Measure that one probe and update the interval-aware design
before choosing any second submission.

## Continuation audit

On 2026-07-23 the local artifacts were re-scanned after the initial verdict.
An authenticated, read-only Kaggle history query then confirmed four existing
records, but still only one scored post-rescoring local/public pair:
submission 54910497 at 0.877. The human explicitly approved at least two
additional, parameter-distinct calibration submissions.

Two inference-only, non-production P0 probes were prepared:

| Kernel | Motion coefficient | Frozen local CV57 | Package manifest SHA-256 |
|---|---:|---:|---|
| `onebitbrain/bhp0-c0` | 0.0 | 0.8483945155545333 | `340a3f67dc2c4c06da0bdd0341986bb0b4c68b96658515c522d901b8a83162d7` |
| `onebitbrain/bhp0-c2` | 2.0 | 0.8721499474601118 | `7fe8bd7273df246913c43b9f7d42452c159450b993683af6c5cca5c70a365e44` |

The packages use distinct slugs because the installed Kaggle CLI addresses
kernel status/output by the latest slug version even when a version suffix is
supplied. Each package is fail-closed to its approved P0 experiment ID and
weight, keeps divisions off and the detector/edge artifact fixed, and records
`production_eligible: false`, `submit_eligible: true`. All 13 package tests and
the 9 focused graph-sweep tests passed.

The human subsequently ran both kernels. The supplied Kaggle result screenshot
shows `bhp0-c0` version 1 succeeded at 0.860, `bhp0-c2` version 2 succeeded at
0.878, and the baseline version 3 at 0.877. These results complete the intended
three-point calibration and produce the final **FAIL / HALT** verdict above.
