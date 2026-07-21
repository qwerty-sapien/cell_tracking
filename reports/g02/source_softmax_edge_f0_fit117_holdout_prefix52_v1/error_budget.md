# Prefix-52 reachability and edge error budget

Status: verified from the frozen `scores.json` samples; no scorer, split, prediction,
or Kaggle state was changed.

## Metric and inputs

- Source rows: `scores.json:345` begins the 52-sample array;
  `scores.json:2368-2378` reports the aggregate summary. The run requested 58
  videos but selected 52 (`scores.json:2225-2227`, `scores.json:2289`).
- The frozen scorer defines per-video
  `w_i = TP_i + FP_i + FN_i`, `J_i = TP_i / w_i`, and
  `m_i = max(0, 1 - 0.1 r_i)`, where
  `r_i = (N_pred_i - N_total_i) / N_total_i`
  (`external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py:28-34`,
  `:353-363`). It aggregates adjusted edge Jaccard as
  `S = sum_i(w_i J_i m_i) / sum_i(w_i)` (`metrics.py:385-418`).

## Verified prefix-52 arithmetic

Summing all 52 samples gives:

| quantity | value |
|---|---:|
| `W = sum_i w_i` | 34,146 |
| edge TP | 29,676 |
| edge FP | 2,660 |
| edge FN | 1,810 |
| adjusted numerator `A = sum_i(w_i * adj_i) = sum_i(TP_i*m_i)` | 29,197.071175842644 |
| `TP / W` | 0.8690915480583378 |
| `A / W` | 0.8550656350917426 |

Both recomputed scores reproduce `scores.json:2369` and `scores.json:2374`
exactly at stored precision.

The exact additive decomposition of the loss from an ideal adjusted score of
1.0 is:

| source | exact formula | score loss | share of total loss | numerator-equivalent events |
|---|---|---:|---:|---:|
| edge FP | `sum FP_i / W` | 0.0779007790 | 53.7490% | 2,660 |
| edge FN | `sum FN_i / W` | 0.0530076729 | 36.5736% | 1,810 |
| node adjustment, net | `sum TP_i*(1-m_i) / W` | 0.0140259130 | 9.6774% | 478.928824 |
| total | `1 - S` | 0.1449343649 | 100% | 4,948.928824 |

This decomposition is exact because `w_i J_i = TP_i`, so
`1-S = FP/W + FN/W + sum(TP_i*(1-m_i))/W`.

The node term is signed, because under-prediction produces `m_i > 1`. Across
the 52 samples, 41 videos contribute a gross node penalty of 0.0158605830
(541.575466 numerator-equivalent events) while 11 contribute a node bonus of
0.0018346700 (62.646642 events), leaving the net loss above. Removing the
entire net node loss alone would raise the score only to 0.8690915481.

## Edge corrections required

The adjusted-numerator deficits are

- to 0.90: `0.90*W - A = 1,534.328824`, or 1,535 idealized unit-gain
  score-equivalent events;
- to 0.92: `0.92*W - A = 2,217.248824`, or 2,218 idealized unit-gain
  score-equivalent events.

Those unit-gain counts are normalization figures, not literal graph edits.
Under the actual frozen formula, an FN-to-TP recovery in video `i` adds `m_i`
to the numerator, while deleting an FP leaves the numerator fixed and reduces
the denominator by one. Holding every node ratio fixed and allocating repairs
optimally across the observed errors gives these optimistic lower bounds on
individual metric-error labels corrected:

| target | minimum corrected labels | one optimal composition | baseline error labels corrected |
|---|---:|---:|---:|
| 0.90 | 1,558 | 1,549 FN recoveries + 9 FP deletions | 34.85% of 4,470 |
| 0.92 | 2,292 | 1,780 FN recoveries + 512 FP deletions | 51.28% of 4,470 |

For reproducibility, FNs were sorted by descending observed `m_i`; for each
prefix of `n` FN recoveries with cumulative gain `G_n`, the required FP count
was computed as `f = ceil(W - (A + G_n)/target)`, then `n + f` was minimized
subject to the observed 1,810 FNs and 2,660 FPs.

For lever-isolation: FP deletion alone requires 1,705 deletions for 0.90 and
2,411 for 0.92. FN recovery alone reaches 0.90 after 1,558 optimally placed
recoveries, but even recovering all 1,810 FNs reaches only 0.9070215856, so it
cannot reach 0.92 without also reducing FPs or the node penalty. A graph rewire
can retire one FP and recover one FN in one graph operation, so the table counts
metric error labels, not implementation edits.

## Six-video ceiling check

The selected prefix contains 31 `6bba` samples with total weight 28,037, hence
verified mean weight `28,037 / 31 = 904.4193548`. Extrapolating that observed
mean to the six requested-but-unselected `6bba` videos gives inferred
`W6 = 5,426.516129` (13.7128% of the inferred 58-video combined weight).

Using `S52 = 0.8550656350917426`, `W52 = 34,146`, and
`S58 = (S52*W52 + c*W6)/(W52+W6)`:

- at missing-video score `c=1.00`, the inferred ceiling is 0.874940254;
- at assumed missing-video score `c=1.07`, it is 0.884539242.

Both are below 0.90. Solving the same formula for the weight needed to reach
0.90 gives `W6 >= 15,343.288242` at `c=1.00` and
`W6 >= 9,025.463672` at `c=1.07` (integer weights: at least 15,344 or 9,026).
Thus the user's rounded ceiling and reachability arithmetic is reproduced.

The six-video weights and resulting ceilings are inferred until the six videos
are actually scored. Also, 1.07 is an assumed scenario bound: the metric source
has no upper clamp on `m_i` (`metrics.py:361-363`); the largest multiplier
observed in these 52 samples is 1.040905064.

## Finding

Verified: the 52-video adjusted score is 0.8550656351, and under the observed
`6bba` weight extrapolation the six missing videos cannot lift the completed
run to 0.90 even if each scores 1.00 or the assumed 1.07. The `>=0.90` gate is
therefore closed for this run on reachability evidence; the full-58 evaluation
remains useful only for the complete citable score and behavior check.
