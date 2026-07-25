# H0 proxy-validity audit — v1

**Verdict: Proposal-only screening BANNED; route candidates straight to the 44b6-first held-fit graph gate.**

This deterministic audit used only numbers already banked in reports. It ran no model, training, scorer replay, GPU job, submission, or prediction-artifact read. The preregistered `rho` is **Spearman rank correlation** on one scored representative per mechanism/identity cluster; the recovery threshold is `|rho| >= 0.5`.

## Primary independent-cluster corpus

The honest primary corpus has **4 mechanism/identity clusters and 9 populated proxy/adjusted-graph pairs**, not an imputed ~15 candidates. Missing graph deltas are absent, never encoded as zero.

| candidate | graph adjusted delta | precision | utility gain | support | graph scope |
|---|---:|---:|---:|---:|---|
| GA-RGSWAP-20260724 | -0.001633610 | 0.9997352 | — | 3777 | registered CV57 top-10 screen (banked report value only) |
| GA-APNCC-DIRECT-20260724 | 0.008372476 | 1.0000000 | — | 38 | full CV57 confirmation (banked report value only) |
| GA-SGBG-20260724 | -0.000861814 | 1.0000000 | — | 10 | registered CV57 top-10 screen (banked report value only) |
| CTRL-OOF-GRAPH-20260724 | -0.000612091 | 0.9648497 | 313 | 1963 | 69-video five-fold held-fit reconstructed graph |

The RGSWAP precision and support are exact arithmetic from the banked `3,776 correct / 1 harmful` counts. All other displayed primary values are direct report fields. The primary chooses one prespecified representative per cluster: the final banked APNCC direct full realization and the objective-first temporal held-fit69 graph gate; later/top-10 realizations remain in the complete inventory. Source paths, fields, and this selection policy are frozen in the companion JSON.

## Complete measured-outcome inventory

The complete inventory contains every comparable banked outcome found in the rescan: **8 graph rows in 4 mechanism/identity clusters**. `GA-TEMPHYBRID` repeats the temporal proxy/model on CV57. `GA-APNCC-GLOBAL` and the APNCC top-10/direct outcomes share the same banked 38/38 proposal screen. These rows are reported, but their repeats do not inflate independent mechanism n.

| outcome | cluster | adjusted delta | raw delta | precision | utility gain | support |
|---|---|---:|---:|---:|---:|---:|
| GA-RGSWAP-20260724/top10 | relative_geometry_swap | -0.001633610 | -0.001721170 | 0.9997352 | — | 3777 |
| GA-APNCC-DIRECT-20260724/full-cv57 | apncc | 0.008372476 | 0.008093907 | 1.0000000 | — | 38 |
| GA-SGBG-20260724/top10 | selected_graph_background | -0.000861814 | -0.000887768 | 1.0000000 | — | 10 |
| CTRL-OOF-GRAPH-20260724/held-fit69 | temporal_6bba_ranker | -0.000612091 | -0.000640836 | 0.9648497 | 313 | 1963 |
| GA-APNCC-GLOBAL-20260724/top10 | apncc | 0.006597193 | 0.006712388 | 1.0000000 | — | 38 |
| GA-APNCC-GLOBAL-20260724/full-cv57 | apncc | 0.004251285 | 0.004238882 | 1.0000000 | — | 38 |
| GA-APNCC-DIRECT-20260724/top10 | apncc | 0.014685681 | 0.014619191 | 1.0000000 | — | 38 |
| GA-TEMPHYBRID-20260724/full-cv57 | temporal_6bba_ranker | -0.001408068 | -0.001497784 | 0.9648497 | 313 | 1963 |

Complete-row point estimates and **mechanism-cluster bootstrap** intervals are sensitivity statistics only:

| outcome / proxy | Pearson | Spearman ρ | Kendall τ-b | sign agreement |
|---|---|---|---|---|
| adjusted / precision | rows=8, clusters=4; 0.491209 [-0.670822, 1.000000] | rows=8, clusters=4; 0.659829 [-0.866025, 1.000000] | rows=8, clusters=4; 0.504184 [-0.816497, 1.000000] | rows=8, clusters=4; 0.500000 [0.000000, 0.923077] |
| adjusted / support | rows=8, clusters=4; -0.638795 [-1.000000, 0.733375] | rows=8, clusters=4; -0.536350 [-1.000000, 0.707107] | rows=8, clusters=4; -0.453632 [-1.000000, 0.632456] | rows=8, clusters=4; 0.500000 [0.000000, 0.923077] |
| raw / precision | rows=8, clusters=4; 0.492236 [-0.659930, 1.000000] | rows=8, clusters=4; 0.659829 [-0.866025, 1.000000] | rows=8, clusters=4; 0.504184 [-0.816497, 1.000000] | rows=8, clusters=4; 0.500000 [0.000000, 0.923077] |
| raw / support | rows=8, clusters=4; -0.641927 [-1.000000, 0.735008] | rows=8, clusters=4; -0.536350 [-1.000000, 0.707107] | rows=8, clusters=4; -0.453632 [-1.000000, 0.632456] | rows=8, clusters=4; 0.500000 [0.000000, 0.923077] |

Naïve complete-row Spearman magnitudes cross 0.5 because four APNCC rows repeat one positive mechanism and two temporal rows repeat one negative mechanism. The cluster bootstrap intervals remain broad. The locked decision therefore uses the one-representative-per-cluster primary corpus below, not these repeated rows.

## Independent-cluster results

| proxy | n | Pearson r (95% bootstrap CI) | Spearman ρ (95% bootstrap CI) | Kendall τ-b (95% bootstrap CI) | sign agreement (95% bootstrap CI) |
|---|---:|---|---|---|---|
| precision | 4 | 0.275942 [-1.000000, 1.000000] | 0.210819 [-1.000000, 1.000000] | 0.182574 [-1.000000, 1.000000] | 0.250000 [0.000000, 0.750000] |
| utility_gain | 1 | insufficient | insufficient | insufficient | insufficient |
| auc | 0 | insufficient | insufficient | insufficient | insufficient |
| support | 4 | -0.574715 [-1.000000, 1.000000] | -0.400000 [-1.000000, 1.000000] | -0.333333 [-1.000000, 1.000000] | 0.250000 [0.000000, 0.750000] |

Sign agreement asks whether a banked positive proposal-screen outcome correctly predicted `graph delta > 0`. Only one of the four cluster-level proposal passes had a positive graph delta. No primary proxy reaches `|Spearman rho| >= 0.5`; utility gain and AUC are untestable at the independent-cluster level because their paired n is below 3.

## Raw scorer-edge-Jaccard outcome

Adjusted graph delta above is the preregistered decision outcome. This secondary analysis uses scorer raw edge Jaccard only: `TP/(TP+FP+FN)`. For RGSWAP, APNCC-direct, and SGBG it is derived exactly from the banked scorer confusion tuples; CTRL-OOF-GRAPH reports it explicitly. Prediction-arm observation Jaccards are not used.

| candidate | raw edge-Jaccard delta | provenance |
|---|---:|---|
| GA-RGSWAP-20260724 | -0.001721170 | exactly derived from banked scorer TP/FP/FN |
| GA-APNCC-DIRECT-20260724 | 0.008093907 | exactly derived from banked scorer TP/FP/FN |
| GA-SGBG-20260724 | -0.000887768 | exactly derived from banked scorer TP/FP/FN |
| CTRL-OOF-GRAPH-20260724 | -0.000640836 | explicit report field |

| proxy | n | Pearson r (95% bootstrap CI) | Spearman ρ (95% bootstrap CI) | Kendall τ-b (95% bootstrap CI) | sign agreement (95% bootstrap CI) |
|---|---:|---|---|---|---|
| precision | 4 | 0.271568 [-1.000000, 1.000000] | 0.210819 [-1.000000, 1.000000] | 0.182574 [-1.000000, 1.000000] | 0.250000 [0.000000, 0.750000] |
| utility_gain | 1 | insufficient | insufficient | insufficient | insufficient |
| auc | 0 | insufficient | insufficient | insufficient | insufficient |
| support | 4 | -0.580263 [-1.000000, 1.000000] | -0.400000 [-1.000000, 1.000000] | -0.333333 [-1.000000, 1.000000] | 0.250000 [0.000000, 0.750000] |

The raw outcome has the same sign for all four candidates. No raw-outcome proxy reaches the preregistered `|Spearman rho| >= 0.5` threshold; the formal adjusted-outcome verdict therefore does not change.

## Temporal-fold sensitivity (not independent candidate evidence)

The temporal graph report is the only dense paired source. Its five folds supply 15 repeated proxy/graph pairs (precision, utility gain, and support for each fold). They are correlated slices of one mechanism and therefore cannot promote a campaign-wide fast-fail.

| proxy | sensitivity result |
|---|---|
| precision | adjusted: n=5; ρ=0.300000; 95% bootstrap CI [-1.000000, 1.000000]; raw: n=5; ρ=0.300000; 95% bootstrap CI [-1.000000, 1.000000] |
| utility_gain | adjusted: n=5; ρ=0.600000; 95% bootstrap CI [-1.000000, 1.000000]; raw: n=5; ρ=0.600000; 95% bootstrap CI [-1.000000, 1.000000] |
| support | adjusted: n=5; ρ=0.600000; 95% bootstrap CI [-1.000000, 1.000000]; raw: n=5; ρ=0.600000; 95% bootstrap CI [-1.000000, 1.000000] |

Utility gain and support each have point `rho = 0.6` inside this one temporal mechanism, but the bootstrap intervals are wide and the result does not replicate across independent candidates. The measured `GA-TEMPHYBRID` adjusted/raw deltas (`-0.0014080676 / -0.0014977843`) are included in the complete inventory under the same temporal cluster, not counted as another independent candidate.

## Exclusions and missingness

| ledger id(s) | graph-delta status | reason |
|---|---|---|
| GA-POINTCTX-PILOT-20260724; GA-POINTCTX-OOF-20260724; GA-POINTCTX-THRESH-20260724 | absent, not zero | Proposal metrics exist, but exact graph scoring was not entered. |
| GA-CROPLINK-PILOT-20260724; GA-CROPLINK-6BBA-OOF-20260724 | absent, not zero | Precision/utility/support exist, but no exact graph output was produced. |
| GA-SCENCAL-20260724 | absent, not zero | Proposal utility/precision exist; the report explicitly says no predictions were emitted and no graph replay occurred. |
| GA-TRACKASTRA-QUAL-20260724 | absent, not zero | ROC AUC 0.918491 (gain +0.077265) is banked, but the row is ranking-only with no graph or prediction CSV. |
| GA-SGCONFLICT-20260724 | absent, not zero | Proposal precision/support exist, but the support gate failed before score. |
| GA-SP-20260724; GA-B2-20260724; GD-T985-20260724 | measured; proxy side absent | Graph deltas exist, but no comparable proposal precision/utility-gain/AUC/support proxy is banked for the scored identity. |
| GA-APNCC-POLICY-20260724 | measured; comparable proxy side absent | Adjusted/raw graph deltas are measured, but the nine added proposals are an observation-difference count, not labeled proposal support, precision, utility gain, or AUC; treating nine as support would mix definitions. |

In particular, Trackastra AUC, POINTCTX precision, CROPLINK precision/utility/support, and SCENCAL utility cannot be paired with graph delta because those graph evaluations do not exist. They were not treated as failures or zeros.

## Formal decision

**Proposal-only screening BANNED; route candidates straight to the 44b6-first held-fit graph gate.**

No proposal proxy is recovered as a legitimate fast-fail. H1 and later candidates must use the locked objective-first order: data/leakage checks, then the 44b6-first held-fit graph gate.

## Reproducibility

- Seed: `20260725`.
- Paired percentile bootstrap replicates: `20000`.
- Correlations: Pearson, tie-aware Spearman, and Kendall tau-b.
- Primary unit: one scored representative per mechanism/identity cluster.
- Temporal folds: sensitivity only; excluded from the formal decision.
- GPU-hours / submissions / training runs: `0 / 0 / 0`.
