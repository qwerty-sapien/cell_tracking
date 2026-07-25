# CV57 metric-asymmetry and admission sweep

## Decision

The metric asymmetry is real, but cheap admission is not a large-gain strategy on the cached candidate graph. Baseline edges are 96.584% ignored, yet the best joint setting reaches only **0.871220267 adjusted edge Jaccard**, a gain of **+0.000031684** over 0.871188582. The formal “small ignored fraction” stop premise is false, but adjusted score still never reaches 0.880 and remains far below the 0.890 promotion threshold. This change should **not** be promoted.

The best setting is edge threshold 0.9, relaxed gate 10 um, neighbor cap 2, with baseline node admission for both families. It changes TP/FP/FN from 34,174/2,736/2,094 to 34,177/2,738/2,091. Raw Jaccard moves from 0.876166547 to 0.876198534; adjusted moves from 0.871188582 to 0.871220267.

All results below replay cached CV57 fragments and candidates on CPU. Detector inference, training, GPU use, divisions, pushes, and Kaggle actions were all disabled.

## Baseline edge classes

The patched scorer classification over all 1,080,440 retained predicted edges is:

| Class | Count | Notes |
|:--|--:|:--|
| TP | 34,174 | maps to a GT edge |
| FP case (a) | 1,383 | target maps to a GT node with another incoming source |
| FP case (b) | 1,369 | source maps to a GT node with another outgoing target |
| case (a) and (b) overlap | 16 | included in both case totals; counted once in FP union |
| FP union | 2,736 | 1,367 a-only + 1,353 b-only + 16 both |
| IGNORED | 1,043,530 | neither FP condition applies |

Ignored fraction = **1,043,530 / 1,080,440 = 0.965837992 (96.584%)**. No baseline edge was removed by missing-node, self-edge, duplicate, nonconsecutive, merge-duplicate, or outdegree filtering.

## Why the large ignored region does not pay

Added edges do land mostly in the ignored class, as predicted. The remaining scored subset is not selective enough: false positives grow much faster than true positives once admission is widened.

| Setting | Added retained | Added ignored | Marginal ignored | Delta TP | Delta FP |
|:--|--:|--:|--:|--:|--:|
| p0.9, r10, k2 | 33 | 28 | 84.85% | +3 | +2 |
| p0.8, r10, k2 | 1,365 | 1,274 | 93.33% | +25 | +66 |
| p0.6, r6, k2 | 679 | 629 | 92.64% | +7 | +43 |
| p0.5, r6, k2 | 2,854 | 2,704 | 94.74% | +16 | +134 |
| p0.3, r6, k2 | 25,868 | 24,834 | 96.00% | +80 | +954 |
| p0.1, r6, k2 | 108,974 | 105,210 | 96.55% | +275 | +3,489 |
| p0.02, r6, k2 | 140,543 | 135,805 | 96.63% | +406 | +4,332 |
| p0.5, r10, k2 | 54,488 | 51,890 | 95.23% | +189 | +2,409 |

## Complete edge-admission sweep

`total_node_ratio` is the event-weighted per-video value used by the official summary. Node admission is held at baseline in this table. Raw and adjusted edge Jaccard are reported separately.

| p | gate um | cap | TP | FP | FN | total node ratio | raw J | adjusted J | ignored % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 2 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 1 | 6 | 4 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 1 | 6 | 8 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 1 | 10 | 2 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 1 | 10 | 4 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 1 | 10 | 8 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 0.9 | 6 | 2 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 0.9 | 6 | 4 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 0.9 | 6 | 8 | 34,174 | 2,736 | 2,094 | +0.061036 | 0.876166547 | 0.871188582 | 96.584 |
| 0.9 | 10 | 2 | 34,177 | 2,738 | 2,091 | +0.061036 | 0.876198534 | 0.871220267 | 96.583 |
| 0.9 | 10 | 4 | 34,177 | 2,738 | 2,091 | +0.061036 | 0.876198534 | 0.871220267 | 96.583 |
| 0.9 | 10 | 8 | 34,177 | 2,738 | 2,091 | +0.061036 | 0.876198534 | 0.871220267 | 96.583 |
| 0.8 | 6 | 2 | 34,174 | 2,737 | 2,094 | +0.061033 | 0.876144084 | 0.871166247 | 96.584 |
| 0.8 | 6 | 4 | 34,174 | 2,737 | 2,094 | +0.061033 | 0.876144084 | 0.871166247 | 96.584 |
| 0.8 | 6 | 8 | 34,174 | 2,737 | 2,094 | +0.061033 | 0.876144084 | 0.871166247 | 96.584 |
| 0.8 | 10 | 2 | 34,199 | 2,802 | 2,069 | +0.061082 | 0.875326337 | 0.870356164 | 96.580 |
| 0.8 | 10 | 4 | 34,199 | 2,803 | 2,069 | +0.061085 | 0.875303934 | 0.870333888 | 96.580 |
| 0.8 | 10 | 8 | 34,199 | 2,803 | 2,069 | +0.061085 | 0.875303934 | 0.870333888 | 96.580 |
| 0.7 | 6 | 2 | 34,176 | 2,742 | 2,092 | +0.061029 | 0.876083056 | 0.871106194 | 96.583 |
| 0.7 | 6 | 4 | 34,176 | 2,742 | 2,092 | +0.061029 | 0.876083056 | 0.871106194 | 96.583 |
| 0.7 | 6 | 8 | 34,176 | 2,742 | 2,092 | +0.061029 | 0.876083056 | 0.871106194 | 96.583 |
| 0.7 | 10 | 2 | 34,241 | 3,074 | 2,027 | +0.061148 | 0.870342128 | 0.865401195 | 96.568 |
| 0.7 | 10 | 4 | 34,244 | 3,079 | 2,024 | +0.061151 | 0.870307774 | 0.865367389 | 96.568 |
| 0.7 | 10 | 8 | 34,244 | 3,079 | 2,024 | +0.061151 | 0.870307774 | 0.865367389 | 96.568 |
| 0.6 | 6 | 2 | 34,181 | 2,779 | 2,087 | +0.060985 | 0.875380951 | 0.870408010 | 96.581 |
| 0.6 | 6 | 4 | 34,181 | 2,779 | 2,087 | +0.060985 | 0.875380951 | 0.870408010 | 96.581 |
| 0.6 | 6 | 8 | 34,181 | 2,779 | 2,087 | +0.060985 | 0.875380951 | 0.870408010 | 96.581 |
| 0.6 | 10 | 2 | 34,305 | 3,674 | 1,963 | +0.061417 | 0.858870362 | 0.853994229 | 96.551 |
| 0.6 | 10 | 4 | 34,310 | 3,694 | 1,958 | +0.061456 | 0.858565637 | 0.853691768 | 96.550 |
| 0.6 | 10 | 8 | 34,310 | 3,694 | 1,958 | +0.061456 | 0.858565637 | 0.853691768 | 96.550 |
| 0.5 | 6 | 2 | 34,190 | 2,870 | 2,078 | +0.060955 | 0.873575553 | 0.868610426 | 96.579 |
| 0.5 | 6 | 4 | 34,190 | 2,871 | 2,078 | +0.060952 | 0.873553233 | 0.868588234 | 96.579 |
| 0.5 | 6 | 8 | 34,190 | 2,871 | 2,078 | +0.060952 | 0.873553233 | 0.868588234 | 96.579 |
| 0.5 | 10 | 2 | 34,363 | 5,145 | 1,905 | +0.061353 | 0.829763601 | 0.825046536 | 96.519 |
| 0.5 | 10 | 4 | 34,373 | 5,203 | 1,895 | +0.061431 | 0.828844253 | 0.824133137 | 96.519 |
| 0.5 | 10 | 8 | 34,373 | 5,203 | 1,895 | +0.061431 | 0.828844253 | 0.824133137 | 96.519 |
| 0.3 | 6 | 2 | 34,254 | 3,690 | 2,014 | +0.060559 | 0.857250113 | 0.852377921 | 96.570 |
| 0.3 | 6 | 4 | 34,254 | 3,717 | 2,014 | +0.060596 | 0.856671252 | 0.851802337 | 96.571 |
| 0.3 | 6 | 8 | 34,254 | 3,717 | 2,014 | +0.060596 | 0.856671252 | 0.851802337 | 96.571 |
| 0.3 | 10 | 2 | 34,596 | 13,508 | 1,672 | +0.061207 | 0.695033751 | 0.691082970 | 96.414 |
| 0.3 | 10 | 4 | 34,625 | 14,564 | 1,643 | +0.061065 | 0.681165408 | 0.677293445 | 96.413 |
| 0.3 | 10 | 8 | 34,626 | 14,567 | 1,642 | +0.061070 | 0.681144880 | 0.677272876 | 96.413 |
| 0.2 | 6 | 2 | 34,316 | 4,697 | 1,952 | +0.060446 | 0.837690712 | 0.832920934 | 96.575 |
| 0.2 | 6 | 4 | 34,316 | 4,801 | 1,952 | +0.060429 | 0.835569408 | 0.830811740 | 96.575 |
| 0.2 | 6 | 8 | 34,316 | 4,801 | 1,952 | +0.060429 | 0.835569408 | 0.830811740 | 96.575 |
| 0.2 | 10 | 2 | 34,752 | 20,996 | 1,516 | +0.061318 | 0.606873428 | 0.603411413 | 96.388 |
| 0.2 | 10 | 4 | 34,794 | 23,692 | 1,474 | +0.061336 | 0.580286858 | 0.576973068 | 96.384 |
| 0.2 | 10 | 8 | 34,795 | 23,726 | 1,473 | +0.061375 | 0.579974664 | 0.576662524 | 96.384 |
| 0.1 | 6 | 2 | 34,449 | 6,225 | 1,819 | +0.060221 | 0.810698233 | 0.806076856 | 96.580 |
| 0.1 | 6 | 4 | 34,449 | 6,354 | 1,819 | +0.060335 | 0.808244569 | 0.803637949 | 96.580 |
| 0.1 | 6 | 8 | 34,449 | 6,354 | 1,819 | +0.060335 | 0.808244569 | 0.803637949 | 96.580 |
| 0.1 | 10 | 2 | 34,950 | 29,244 | 1,318 | +0.061436 | 0.533490048 | 0.530438129 | 96.380 |
| 0.1 | 10 | 4 | 34,950 | 31,940 | 1,318 | +0.062029 | 0.512403237 | 0.509471140 | 96.375 |
| 0.1 | 10 | 8 | 34,952 | 32,038 | 1,316 | +0.062086 | 0.511697362 | 0.508768848 | 96.372 |
| 0.05 | 6 | 2 | 34,537 | 6,905 | 1,731 | +0.060318 | 0.799967572 | 0.795403803 | 96.587 |
| 0.05 | 6 | 4 | 34,531 | 6,993 | 1,737 | +0.060396 | 0.798201613 | 0.793648021 | 96.586 |
| 0.05 | 6 | 8 | 34,531 | 6,993 | 1,737 | +0.060396 | 0.798201613 | 0.793648021 | 96.586 |
| 0.05 | 10 | 2 | 35,046 | 31,694 | 1,222 | +0.061582 | 0.515670522 | 0.512716745 | 96.387 |
| 0.05 | 10 | 4 | 34,995 | 33,424 | 1,273 | +0.062321 | 0.502137979 | 0.499262838 | 96.376 |
| 0.05 | 10 | 8 | 34,998 | 33,472 | 1,270 | +0.062349 | 0.501835389 | 0.498961379 | 96.373 |
| 0.02 | 6 | 2 | 34,580 | 7,068 | 1,688 | +0.060399 | 0.797950895 | 0.793395240 | 96.589 |
| 0.02 | 6 | 4 | 34,573 | 7,143 | 1,695 | +0.060480 | 0.796411048 | 0.791864562 | 96.587 |
| 0.02 | 6 | 8 | 34,573 | 7,143 | 1,695 | +0.060480 | 0.796411048 | 0.791864562 | 96.587 |
| 0.02 | 10 | 2 | 35,085 | 32,053 | 1,183 | +0.061652 | 0.513531711 | 0.510587410 | 96.389 |
| 0.02 | 10 | 4 | 35,012 | 33,579 | 1,256 | +0.062361 | 0.501267055 | 0.498395677 | 96.377 |
| 0.02 | 10 | 8 | 35,015 | 33,623 | 1,253 | +0.062387 | 0.500994406 | 0.498123989 | 96.375 |

## Per-family node-admission sweep

Each family is scored independently with the best edge setting fixed at p0.9/r10/k2. A target is a desired per-video node-count ratio relative to the estimated count; baseline nodes are never removed, and extra nodes are restored in cached detector-confidence order. This independently tests node admission rather than coupling it to edge admission.

### 44b6

| target | TP | FP | FN | total node ratio | raw J | adjusted J | node recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5,150 | 540 | 363 | -0.060028 | 0.850817776 | 0.856779127 | 0.994254 |
| -0.100 | 5,130 | 555 | 383 | -0.036075 | 0.845418589 | 0.849066561 | 0.995444 |
| -0.075 | 5,130 | 555 | 383 | -0.035418 | 0.845418589 | 0.849010723 | 0.995444 |
| -0.050 | 5,125 | 558 | 388 | -0.032554 | 0.844177236 | 0.847507498 | 0.995898 |
| -0.025 | 5,123 | 560 | 390 | -0.030337 | 0.843569900 | 0.846701657 | 0.995898 |

### 6bba

| target | TP | FP | FN | total node ratio | raw J | adjusted J | node recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 29,027 | 2,198 | 1,728 | +0.083274 | 0.880860620 | 0.873872900 | 0.988006 |
| +0.100 | 28,877 | 2,286 | 1,878 | +0.114016 | 0.873974759 | 0.864301322 | 0.993567 |
| +0.125 | 28,849 | 2,300 | 1,906 | +0.121514 | 0.872757525 | 0.862408358 | 0.994217 |
| +0.150 | 28,831 | 2,311 | 1,924 | +0.128338 | 0.871922821 | 0.860958352 | 0.994376 |
| +0.175 | 28,815 | 2,304 | 1,940 | +0.133054 | 0.871623461 | 0.860241485 | 0.994580 |

Restoring nodes raises node recall in both families but worsens both raw and adjusted edge Jaccard. The family-specific joint optimum therefore keeps baseline node admission for both 44b6 and 6bba.

## Joint optimum

| edge setting | 44b6 target | 6bba target | TP | FP | FN | total node ratio | raw J | adjusted J | delta |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| p0.9/r10/k2 | baseline | baseline | 34,177 | 2,738 | 2,091 | +0.061036 | 0.876198534 | 0.871220267 | +0.000031684 |

## Verification and artifacts

- The manual classifier reproduces the frozen 57-video baseline exactly: TP 34,174, FP 2,736, FN 2,094, raw 0.8761665470208184, adjusted 0.8711885823490648.
- On a widened non-baseline replay (`train:44b6_3bb3690f`, p0.9/r10/k2), all nine manual edge/node fields match `official_edge_metric_row` exactly within 1e-12.
- Focused scorer tests pass: 10/10 in `tests.test_official_scorer_parity` and `tests.test_score_bh_cv57`.
- The machine-readable report contains all 66 edge settings, all per-video classifications, the complete 125-row joint finalist sweep, provenance hashes, and execution guards: `metric_asymmetry_sweep_cv57_v1.json`.
- Replay implementation: `scripts/sweep_metric_asymmetry_cv57.py`.
