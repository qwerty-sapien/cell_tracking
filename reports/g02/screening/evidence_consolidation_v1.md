# B1/B2/E1 evidence consolidation

Status: **COMPLETE — local evidence only**

This report consolidates existing artifacts under the revised DAG. It does not rerun
expensive analyses, use the untouched audit for selection, submit to Kaggle, or launch
GPU work. Quantities are kept tied to their exact prediction identity.

## Executive decision

| Node | Verdict | Decision | Reason |
|---|---|---|---|
| B1 division | Association prerequisite not met | **PARK** | The official scorer rewards connected two-branch lineage motifs, but `add_safe_divisions` restored 0/51 removed fork edges and emitted no new edge on 13 valid sandbox fixtures. |
| B2 decomposition | Crowding/high-motion screen is supported | **BANK** | On the banked `c=0.25` baseline, 10/57 videos carry 52.00% of weighted residual; dataset prefix alone is not the discriminator. |
| E1 taxonomy | **ASSOCIATION-BOUND** | **BANK** | A conservative 61.92% of audited FNs are verified occupied-endpoint/degree conflicts, and the fixed-node association oracle reaches 0.97499 on its exact source prediction. |
| GA | Evidence direction passes; compute gate not yet met | **PARK** | Run a CPU/cached full association-cost replacement screen first. GPU still requires a preregistered affected-slice proxy gain of at least +0.005, runtime/budget fit, and approval where applicable. |
| GD | Secondary candidate | **PARK** | Threshold 0.985 is fit-supported but provisional and requires fresh detector inference; association has stronger verified headroom. |

## Identity table

| Evidence | Exact identity and role |
|---|---|
| Banked decision baseline | `c=0.25`, score `0.8731495935289003`, edge TP/FP/FN `34,205 / 2,685 / 2,063`, W `38,953`; re-attested in `obs_diff_c0p25_vs_c2_v1.json`. |
| Older decomposition source | `bhcv57out_v3`, score `0.8711885823490648`, edge TP/FP/FN `34,174 / 2,736 / 2,094`, W `39,004`; used for covariates and earlier rank correlations, not substituted for the banked baseline. |
| Fragmentation taxonomy source | Source-softmax F4 complete-57 result, score `0.8495036625176058`, edge TP/FP/FN `34,107 / 3,230 / 2,161`; its mechanistic buckets are not silently relabeled as banked-baseline counts. |
| Fixed-node oracle | Same source-softmax prediction nodes with held-out-GT-supported edges; score `0.9749936253918458`. Diagnostic only and ineligible for selection/submission. |

## B1 — division evidence

The sandbox replay establishes the actual credited topology:

- a direct binary fork on the matched parent side can score;
- a one-frame-late fork can score when its two branches reach distinct GT daughter
  lineages;
- grandchild evidence can complete a branch;
- duplicate/evaluable competing forks become FP after one-to-one pairing;
- ordinary local edge correctness and division credit are related but not identical.

Production `add_safe_divisions` does not currently construct these motifs:

- 13/14 sandbox graphs satisfy production graph invariants;
- 0/13 receive a new division edge;
- 0/51 counterfactual removed fork edges are restored;
- inherited geometry covers only `19/83 = 22.89%` of fit divisions;
- on the audited holdout, 35/37 GT divisions have their matched stages present but split
  across prediction components.

Conclusion: division constants, wider geometry, or a new division head are downstream of
the observed failure. Keep divisions OFF until association continuity creates scorer-valid
two-branch motifs without FP leakage.

Source:
`reports/g02/division_sandbox_and_holdout_decomposition_v1.{md,json}` and
`reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.md`.

## B2 — current banked-baseline decomposition

The frozen score rows embedded in `obs_diff_c0p25_vs_c2_v1.json` reproduce the current
banked score and allow a fresh residual aggregation without new model execution.

### Family aggregation

| Family | Videos | Edge-event weight | Weighted adjusted score | Residual share | FP / FN |
|---|---:|---:|---:|---:|---:|
| `44b6` | 21 | 6,062 | 0.854173 | 17.89% | 549 / 371 |
| `6bba` | 36 | 32,891 | 0.876647 | 82.11% | 2,136 / 1,692 |

`6bba` carries most residual because it carries most event weight. `44b6` remains harder
per weighted event, so a prefix-only family router is not supported.

### Concentration

| Worst-k videos | Share of total weighted residual |
|---:|---:|
| 1 | 12.02% |
| 3 | 24.46% |
| 5 | 34.45% |
| 10 | **52.00%** |
| 24 | 80.48% |

The leading contributors remain dominated by crowded and/or high-motion `6bba` videos,
with `44b6_d5e7d891` also in the top ten. The earlier exact-covariate report found:

- residual Spearman `+0.433` with detections/frame;
- residual Spearman `-0.495` with nearest-neighbor distance;
- residual Spearman `+0.372` with the 6 µm unmatched fraction;
- occupied depth has essentially no relationship (`rho=-0.054`);
- event-level neighborhoods above cap 8 had a 5.50x edge-break risk ratio.

Those covariates came from the older exact prediction identity and are used as
mechanism-direction evidence, not as current-baseline score rows. The current residual
concentration and family direction agree closely enough to retain crowding and tight-pass
unmatched fraction as the next SCREEN selectors.

## E1 — association versus detection

The complete-57 fragmentation audit classifies all 2,161 FNs on its exact
source-softmax prediction:

| Observable class | FNs | Interpretation here |
|---|---:|---|
| GT source or target endpoint unmatched | 308 | Detection/localization lower bound |
| Synthesized gap endpoint | 99 | Ambiguous mixed detection/association |
| Final matched-pair geometry above 10 µm | 407 | Ambiguous candidate/continuity consequence |
| Correct pair absent with an endpoint occupied elsewhere | 1,319 | Verified association conflict |
| Correct sister already consumes source degree | 19 | Verified association/solver conflict |
| Both endpoints free with no observed conflict | 9 | Unresolved |

Therefore:

- verified association-bound lower bound: `(1,319 + 19) / 2,161 = 61.92%`;
- detection/localization lower bound: `308 / 2,161 = 14.25%`;
- unresolved/mixed remainder: `23.83%`.

The diagnostic fixed-node oracle increases its exact-source score from `0.8495037` to
`0.9749936` (`+0.1254900`) while preserving prediction nodes and the node-count penalty.
It uses held-out labels and is not a candidate, but it proves that association quality has
enough headroom to dominate the residual on that prediction identity. It also explains
why zero division credit is largely an upstream fragmentation symptom.

Verdict: **ASSOCIATION-BOUND**. This is sufficient to prioritize association work locally.
It is not sufficient by itself to authorize GPU use.

## Existing learned attempts and exclusions

- Small learned-bonus blends are observation-distinct, but their best banked result
  (`lb4`, `0.8720910`) remains below `c=0.25`. Additive bonuses have not exploited the
  oracle headroom.
- The P4 geometry-free edge-attention layer remains DEAD as built.
- `reports/g02/organizer_unet_transformer_f0/stage00_v3` records a fit-only monitor score
  of `0.9078951`, but its provenance explicitly says
  `random_scratch_no_released_or_foreign_checkpoint`. It violates the active G-02
  checkpoint-based model contract and has no holdout evaluation, so it is excluded from
  promotion regardless of its fit monitor.
- The fit-only node-threshold analysis supports 0.985 provisionally (`+0.00436` mean on
  14 fit videos) but used a mismatched execution path and needs fresh one-pass inference.

## Next ready local screen

Implement and evaluate a **full learned association-cost replacement** on existing
candidate tensors, not another additive learned bonus:

1. freeze detector/node identity where the cache permits;
2. replace the hand-tuned motion term with the learned association cost;
3. screen first on the preregistered crowded/high-motion subset (top residual videos,
   high density/low NN distance, or high tight-pass unmatched fraction);
4. require at least `+0.005` on that affected slice to open GA consideration;
5. confirm any survivor on the complete frozen CV57 scorer and observation-diff it
   against a same-score graph-cost comparator;
6. keep divisions OFF and report node/edge/fork vectors separately.

If the replacement cannot produce the proxy gain, PARK GA and run the fresh 0.970 versus
0.985 detector-front-end screen under GD. No public probe is needed to choose between
these local screens.
