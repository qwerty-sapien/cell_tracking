# CG1 abstention-realization mechanism screen

**PARK**

No mechanism earned **plan** status. The CG1 cascade therefore halts; CG2 is not
authorized.

## Preregistration and scope

This screen was fixed before inspecting its outcomes:

- Five deterministic held-video **calibration** folds over fit-train105;
  **44b6 first**.
- Fit-validation12 was not used for selection. No holdout, audit, test, GPU, Kaggle, or
  graph replay was used.
- The frozen, hash-pinned temporal proposal scorer was reused; no ranker was trained.
- Fit117 deployment-background budgets inherited from CG0:
  **848 FP for 44b6**, **2,528 FP for 6bba**, and **3,376 FP overall**.
- The corresponding held-train105 burdens are 760.35, 2,404.70, and 3,165.05
  projected FP.
- Correct-edge retention floor: **20% in each family**. The legacy frozen-score
  arithmetic needs 779 ideal swaps to reach 0.92, or 18.12% of CG0's 4,300
  present-but-unrankable opportunities; 20% is the preregistered buffer.
- Kill line: projected held background FP above the family budget, or known-correct
  challenger retention below 20% in either family.

The parked scorer was previously fit on train105; it is frozen here, but it is not an
outer-fold OOF ranker. Only each mechanism's operating point is learned from other
videos. This makes the retention screen optimistic rather than pessimistic and means a
pass could not promote without CG2's nested graph gate; the observed failure remains a
valid cheap falsification.

The deployment cache contains 19,398 sampled unlabeled proposals, of which 19,398 are
never relabeled as negatives. The OOF screen uses 8,668 44b6 and 10,730 6bba samples.
For reporting, `projected FP` conservatively treats every admitted unlabeled proposal
as false and scales each video's sample rate to its exact r10/cap-8 raw background.
The 95% column is the one-sided binomial upper projection. Because the cache is a
hard-plus-uniform sample rather than an i.i.d. deployment census, a point estimate or
sampled zero is not promoted as a formal conformal guarantee.

## Alpha candidates and grounding

| mechanism | one-line defensibility | transfer condition | crowding verdict |
|---|---|---|---|
| Global conformal risk control (CRC) | Calibrate a monotone false-admission loss to the explicit CG0 risk budget rather than proxy precision. | Held videos must be exchangeable with calibration videos at the operating tail. | **Alpha**, structurally different from another ranker; finite-sample tail support is the red flag. |
| Family-conditional/Mondrian CRC | Condition calibration on 44b6/6bba so marginal coverage cannot hide a 44b6 collapse. | Videos must be exchangeable within family, with enough tail samples per family. | **Alpha**, but not a free cure for within-family shift. |
| Positive-unlabeled density-ratio calibration | Model background as an unlabeled mixture and estimate its positive prior instead of asserting clean negatives. | Known positives must represent deployment positives and the mixture prior must be identifiable/stable by family. | **Alpha**; the current challenger-biased positives make the prior assumption doubtful. |
| Per-video conformal tail rank | Replace an absolute probability threshold by a video's empirical unlabeled-background rank. | A common rank law must transfer and the target-video background ECDF must resolve the 0.02–0.07% tail. | Mechanism-specific version is new, but distribution normalization as a broad lever is already **DEAD** locally. |
| Hard per-video top-\(k\) budget | Allocate a deterministic count budget before graph optimization; FP count can never exceed \(k\). | Full-pool score ranks—not a 256-row sample—must retain at least 20% of correct challengers in both families. | **Alpha / first-principles**; it directly encodes the scorer's FP constraint. |
| Shared point-threshold control | Choose the frozen-score threshold that meets the projected background rate on other videos. | One absolute threshold must transfer. | **Beta and DEAD**: this is the scalar-threshold family already falsified by `SCENCAL`, `POINTCTX-THRESH`, and `APNCC-POLICY`. |

Primary grounding:

- Angelopoulos et al., [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/),
  control the expectation of a monotone loss with a calibration set. That maps naturally
  to an FP-admission loss, but its guarantee does not establish exchangeability across
  these video families.
- Bates et al., [Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703),
  give finite-sample expected-loss control for black-box predictors calibrated on held
  data.
- Barber et al., [Conformal prediction beyond exchangeability](https://doi.org/10.1214/23-AOS2276),
  explicitly identify exchangeability as the standard validity condition and introduce
  weighting for drift. The local IQR-0.237 operating-point dispersion is evidence that
  unweighted cross-video exchangeability cannot simply be assumed.
- Kiryo et al., [Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html),
  derive a non-negative PU risk from positive and unlabeled samples.
- du Plessis, Niu, and Sugiyama,
  [Class-prior Estimation for Learning from Positive and Unlabeled Data](https://arxiv.org/abs/1611.01586),
  ground positive-prior estimation, subject to mixture identifiability. The five-fold
  local estimates (44b6 3.73–4.94%; 6bba 3.85–4.41%) are far above the raw current-pool
  correct fraction (0.29% and 1.62%), showing that the prepared challenger/sample
  distributions are not a defensible deployment mixture without additional work.
- Geifman and El-Yaniv, [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html),
  show that jointly learning selection can improve risk/coverage. It is not built here:
  another from-scratch selective MLP would repeat the campaign's DEAD 44b6 failure mode.

## Held-fit falsification

`Proj FP` is the conservative point projection from the sampled unlabeled background.
`95% FP` is its one-sided upper projection. `Keep` is known-correct challenger
retention. The held budgets below are train105-equivalent portions of the locked fit117
budget.

| mechanism | family | projected FP / budget | 95% FP / budget | correct kept | known harmful kept | local result |
|---|---|---:|---:|---:|---:|---|
| Shared point threshold (control) | **44b6** | 733.89 / 760.35 | 3,225.22 / 760.35 | 7/766 = **0.91%** | 0 | DEAD |
|  | 6bba | 2,291.93 / 2,404.70 | 4,402.81 / 2,404.70 | 53/2,993 = **1.77%** | 0 | DEAD |
| Global CRC | **44b6** | 204.16 / 760.35 | 1,973.38 / 760.35 | 3/766 = **0.39%** | 0 | DEAD |
|  | 6bba | 621.70 / 2,404.70 | 2,108.41 / 2,404.70 | 27/2,993 = **0.90%** | 0 | DEAD |
| Family/Mondrian CRC | **44b6** | 0 / 760.35 | 1,246.23 / 760.35 | 0/766 = **0.00%** | 0 | DEAD |
|  | 6bba | 621.70 / 2,404.70 | 2,108.41 / 2,404.70 | 27/2,993 = **0.90%** | 0 | DEAD |
| PU family density ratio | **44b6** | 0 / 760.35 | 1,246.23 / 760.35 | 0/766 = **0.00%** | 0 | research-bet; this instantiation DEAD |
|  | 6bba | 0 / 2,404.70 | 1,003.31 / 2,404.70 | 0/2,993 = **0.00%** | 0 | research-bet; this instantiation DEAD |
| Per-video tail rank | **44b6** | 0 / 760.35 | 1,246.23 / 760.35 | 91/766 = **11.88%** | 1 | DEAD |
|  | 6bba | 0 / 2,404.70 | 1,003.31 / 2,404.70 | 331/2,993 = **11.06%** | 1 | DEAD |
| Hard per-video top-\(k\) | **44b6** | ≤760.35 / 760.35 by construction | count-safe | certified **0.00%**; optimistic sample 91/766 = **11.88%** | optimistic 1 | research-bet; fails current screen |
|  | 6bba | ≤2,404.70 / 2,404.70 by construction | count-safe | certified **0.00%**; optimistic sample 331/2,993 = **11.06%** | optimistic 1 | research-bet; fails current screen |

The global CRC used OOF thresholds 0.998844–0.999066. Family CRC could not certify
any finite 44b6 threshold: even zero admissions among 8,668 samples has a 95% upper
projection of 1,246 FP, above the 760 held budget. Its only certified 44b6 action was
complete abstention. This is a sample-size limit, not evidence that CRC theory is
wrong.

The per-video rank rule is the strongest observed-retention mechanism and attacks
threshold non-transfer directly. It still misses the preregistered floor in both
families, and its sampled zero does not certify the 44b6 tail. A hard top-\(k\) rule
does guarantee the count budget, but the plus-one rank bound from 256 samples cannot
certify any correct candidate as lying inside the deployment top-\(k\); even its
optimistic sample estimate stays near 11%.

## Classification and ranking

No **plan** survived. The remaining ideas are ranked by pertinence × feasibility ×
defensibility, with each component on a five-point scale:

| rank | mechanism | P × F × D | seeking-alpha label | disposition |
|---:|---|---:|---|---|
| 1 | Hard per-video top-\(k\) over a stronger frozen representation | 5 × 4 × 5 = **100** | **research-bet** | Count safety is exact, but current frozen scores and sample resolution cannot establish retention. |
| 2 | PU risk plus full-pool pretrained features | 5 × 3 × 4 = **60** | **research-bet** | Grounded, but a representative positive sample and stable class-prior estimate are missing. |
| 3 | Global CRC | 5 × 5 × 5 = **125** before falsification | not a survivor | Cheap and grounded; killed by <1% retention and 44b6 tail uncertainty. |
| 4 | Family/Mondrian CRC | 5 × 4 × 5 = **100** before falsification | not a survivor | Family conditioning exposes rather than repairs the 44b6 finite-sample limit. |
| 5 | Per-video tail rank | 4 × 5 × 3 = **60** before falsification | not a survivor | Best observed retention, but below floor in both families and crowded with a DEAD normalization direction. |
| 6 | Shared point threshold | 2 × 5 × 1 = **10** | **guess / beta** | Reproduces the known non-transferable scalar repair. |

The two research bets are not CG1 survivors: neither passed the registered local
test. Their named unlock is a complete both-family full-pool cache from a genuinely
strong frozen representation, followed by the same untouched 20%/budget gate. That
is an escalation/new evidence requirement, not authorization to build CG2 now.

## Verdict and cascade

**PARK**

Maximum observed transferable retention was 11.88% on 44b6 and 11.06% on 6bba under
the per-video tail-rank/top-\(k\) family, versus the required 20%. No mechanism is both
grounded, within the background budget, and transferable with sufficient retention
on both families. CG2 is not run; no held graph replay, validation12 selection, GPU,
CV57, or Kaggle action occurred.
