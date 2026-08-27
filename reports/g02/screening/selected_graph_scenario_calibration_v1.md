# Fit-only per-video competition calibration

Decision: **PARK without CV57.**

This screen tested the cheapest remaining explanation for the unstable
cross-video operating threshold: whether the banked pair-NCC proposal score
could be made transferable using only observable per-video score
distributions, proposal competition, and the deployed cap. It did not use
family prefix, video identity, residual rank, CV57 labels, public scores, an
optimizer, GPU, or external compute.

The locked contract is
`reports/g02/selected_graph_scenario_calibration_contract_v1.json`, SHA-256
`037dccf60cbae4000d2ba67590d8bf42d84b117d9d79a7d24cc96e829b9269a1`.
It preregistered 11 configurations:

- exact raw pair-NCC control;
- robust `(q50, q90)` and `(q75, q95)` normalization;
- empirical-CDF normalization;
- rank relative to the deployed proposal cap;
- competition penalties `0`, `0.25`, and `0.5` where admissible.

Configuration and threshold selection used deterministic five-fold
video-grouped OOF over fit-train105. Validation12 was opened once only after
OOF selection. Promotion required aggregate OOF precision `>=0.95`, every
fold `>=0.93`, OOF utility gain `>=+50`, validation precision `>=0.95`,
validation utility gain `>=+5`, and no family utility regression.

## Result

The raw control reproduced `2,020` correct and `112` harmful OOF rescues:
precision `0.947467`, conservative utility `1,796`. No normalized
configuration was admissible.

| configuration | correct / harmful | precision | utility | delta vs raw | minimum fold precision |
|---|---:|---:|---:|---:|---:|
| raw control | 2,020 / 112 | 0.947467 | 1,796 | 0 | 0.925344 |
| robust50–90, penalty 0.25 | 1,801 / 100 | **0.947396** | 1,601 | **−195** | 0.906977 |
| robust50–90, penalty 0 | 1,792 / 105 | 0.944649 | 1,582 | −214 | 0.910596 |
| cap rank | 1,777 / 109 | 0.942206 | 1,559 | −237 | 0.879545 |
| empirical CDF, penalty 0 | 1,692 / 95 | 0.946838 | 1,502 | −294 | 0.925267 |

The fallback selection rule chose robust50–90 with penalty `0.25` because it
had the highest non-control OOF precision. It independently failed
validation:

| rule | correct / harmful | precision | utility |
|---|---:|---:|---:|
| raw control | 185 / 14 | 0.929648 | 157 |
| selected normalization | 117 / 8 | 0.936000 | 101 |

Validation utility changed by **−56**. On `6bba`, precision improved to
`0.978947` but the rule discarded 75 correct rescues and lost 55 utility. On
`44b6`, it gained some recall but precision fell to `0.80` and utility lost
one. Both family guards failed.

## Interpretation

Per-video distribution normalization mainly strengthens abstention; it does
not recover the correct crowded associations. This agrees with the earlier
background gate and rules out another scalar calibration or score-percentile
rule as the path to 0.90. Keep APNCC `0.881522070` banked. The next useful
evidence remains the sealed point-context held-graph screen, which represents
proposal competition directly rather than shifting the operating threshold.

Machine report:
`reports/g02/selected_graph_scenario_calibration_gate_v1/report.json`,
SHA-256
`956fdf649b76d1add35b7ef595deaaf509434622699d02833450d471eccb84b6`.
The run used fit-only prepared tables, took 80.06 CPU seconds, and performed no
Kaggle, GPU, RunPod, submission, or CV57 action.
