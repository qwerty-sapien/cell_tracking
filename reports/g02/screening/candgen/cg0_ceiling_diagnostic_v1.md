# CG0 candidate-generation ceiling diagnostic

**GO**

## Root-gate result

The literal fit117 perfect-selection oracle already exceeds 0.92 for the current r10/cap-8 proposal policy. The prior 0.895678 value is reproduced, but it is a 378-swap target allocation rather than a full-pool oracle ceiling. Candidate generation is therefore not the binding mathematical ceiling; selection/abstention remains the realization problem.

## Legacy controls

| control | score | interpretation |
|---|---:|---|
| current pool | 0.895677948 | reproduces 0.895678; target-limited projection |
| published shell control | 0.897745837 | reproduces 0.897746; actually the 344/456 motion-retained subset |
| full r12/k8 under the same legacy projection | 0.898436139 | 456/719 shell recall |

## Exclusion buckets

| family | cause | count | correct edges recovered | oracle ceiling |
|---|---|---:|---:|---:|
| overall | present-but-unrankable | 4,300 | 4,300 | 0.960918 |
| overall | out-of-radius-10-to-12 | 456 | 456 | 0.966754 |
| overall | out-of-radius-beyond-12 | 262 | 262 | 0.969940 |
| overall | sub-cap | 1 | 1 | 0.960931 |
| overall | gap-closing-multi-frame | 231 | 462 | 0.966728 |
| 44b6 | present-but-unrankable | 822 | 822 | 0.986249 |
| 44b6 | out-of-radius-10-to-12 | 49 | 49 | 0.990416 |
| 44b6 | out-of-radius-beyond-12 | 15 | 15 | 0.991766 |
| 44b6 | sub-cap | 1 | 1 | 0.986333 |
| 44b6 | gap-closing-multi-frame | 35 | 70 | 0.992194 |
| 6bba | present-but-unrankable | 3,478 | 3,478 | 0.956267 |
| 6bba | out-of-radius-10-to-12 | 407 | 407 | 0.962409 |
| 6bba | out-of-radius-beyond-12 | 247 | 247 | 0.965933 |
| 6bba | sub-cap | 0 | 0 | 0.956267 |
| 6bba | gap-closing-multi-frame | 196 | 392 | 0.962052 |

Counts are mutually exclusive by primary cause for adjacent edges: r10/cap-8 present-but-unselected; 10–12 µm; beyond 12 µm; within-10 cap exclusion. Gap rows are separate one-missed-frame bridge events and recover two correct edges per valid event.

## Deterministic policy ladder

| policy | exact ceiling | pool | correct | background | raw correct/background | max background FP at 0.92 | min rejection |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_r10_k08 | 0.960918 | 7,874,673 | 73,999 | 7,800,674 | 0.00948623 | 3,376 | 99.956722% |
| current_r10_k16 | 0.960931 | 7,877,697 | 74,000 | 7,803,697 | 0.00948268 | 3,377 | 99.956726% |
| shell_r12_k08 | 0.966754 | 13,216,744 | 74,455 | 13,142,289 | 0.00566530 | 3,858 | 99.970644% |
| current_r10_k08_plus_gap | 0.966728 | 29,725,469 | 74,230 | 29,651,239 | 0.00250344 | 3,855 | 99.986999% |
| shell_r12_k08_plus_gap | 0.972564 | 35,067,540 | 74,686 | 34,992,854 | 0.00213432 | 4,337 | 99.987606% |
| wide_r20_k32_oracle_filter | 0.969940 | 48,502,787 | 74,704 | 48,428,083 | 0.00154258 | 4,120 | 99.991493% |
| wide_r20_k32_oracle_filter_plus_gap | 0.975750 | 70,353,583 | 74,935 | 70,278,648 | 0.00106626 | 4,600 | 99.993455% |

The wider-radius `oracle_filter` rows are upper bounds: their raw background is reported before the perfect abstention oracle, so a real proposer must meet the stated false-positive budget.

## Per-family maxima

| family | maximum ceiling | policy | GO at 0.92 |
|---|---:|---|---|
| overall | 0.975750 | wide_r20_k32_oracle_filter_plus_gap | yes |
| 44b6 | 0.997711 | wide_r20_k32_oracle_filter_plus_gap | yes |
| 6bba | 0.971718 | wide_r20_k32_oracle_filter_plus_gap | yes |

## Verdict

**GO**

The least-burden policy already over 0.92 is `current_r10_k08` at 0.960918. It may admit at most 3,376 background FP edges on fit117 while retaining 0.92, equivalent to rejecting at least 99.956722% of its raw background proposals. This is plausible as an absolute budget because it is the existing pool and is comparable to the banked graph's error scale; no wider pool is required to clear the mathematical gate.

CG1 should target the present-but-unrankable/abstention realization inside the current pool first. Wider-radius and gap unions have higher maxima but substantially harsher background-rejection requirements.
