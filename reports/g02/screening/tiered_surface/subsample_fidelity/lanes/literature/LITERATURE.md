# Literature basis for retrospective video-subsample fidelity

Status: **terminal literature diagnostic; no outcome opened**

This lane supports the frozen estimator in
`reports/g02/screening/tiered_surface/subsample_fidelity/PREREGISTRATION.md`
without inspecting any candidate decomposition, sampled delta, fidelity table, or verdict.
It proposes no tier move.

## What the estimator is

The preregistered procedure is most accurately described as **repeated,
design-weighted finite-population subsampling without replacement**. It is not an
ordinary Efron bootstrap: each draw selects distinct videos, and the finite
authenticated video universe is known.

For simple random sampling, each video has inclusion probability `k/N`, so inverse
inclusion weighting gives `N/k`. Within a frozen stratum `h`, inclusion probability is
`n_h/N_h`, so the corresponding weight is `N_h/n_h`. These are the
Horvitz–Thompson expansion factors for the additive per-video sufficient statistics.
The expanded totals are then passed through the unchanged official nonlinear score.
That last plug-in step matters: inverse-probability weighting supports the additive
totals, but does **not** make a Jaccard ratio or the resulting graph delta unbiased.

The repeated 10,000 draws approximate the estimator's finite-population
sampling-design distribution. The 2.5th and 97.5th percentiles are therefore best
described as a **central 95% design-distribution interval**. They measure how much the
screening estimate changes across eligible video subsets; they are not, by themselves,
a 95% confidence interval for an unknown superpopulation parameter.

## Primary sources and the claim each supports

1. Bradley Efron (1979), “Bootstrap Methods: Another Look at the Jackknife,”
   *The Annals of Statistics* 7(1), 1–26.
   [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).
   This is the primary source for using resampling to approximate the sampling
   distribution of a statistic. Its ordinary bootstrap setting is not the exact
   without-replacement design used here.

2. Dimitris N. Politis and Joseph P. Romano (1994), “Large Sample Confidence Regions
   Based on Subsamples under Minimal Assumptions,” *The Annals of Statistics* 22(4),
   2031–2050.
   [doi:10.1214/aos/1176325770](https://doi.org/10.1214/aos/1176325770).
   This is the closer primary source for recomputing a statistic over subsamples to
   approximate a sampling distribution. Its results are asymptotic and do not certify
   fixed `k` values or this nonlinear graph statistic.

3. Peter J. Bickel and David A. Freedman (1984), “Asymptotic Normality and the
   Bootstrap in Stratified Sampling,” *The Annals of Statistics* 12(2), 470–482.
   [doi:10.1214/aos/1176346500](https://doi.org/10.1214/aos/1176346500).
   This establishes that stratified resampling has design-specific asymptotic theory.
   It does not validate the benchmark-specific residual strata or small-`k` behavior.

4. D. G. Horvitz and D. J. Thompson (1952), “A Generalization of Sampling Without
   Replacement from a Finite Universe,” *Journal of the American Statistical
   Association* 47(260), 663–685.
   [doi:10.1080/01621459.1952.10483446](https://doi.org/10.1080/01621459.1952.10483446).
   This is the primary basis for inverse-inclusion-probability expansion under
   finite-population sampling without replacement. Here it justifies `N/k` and
   `N_h/n_h` for additive sufficient statistics, not unbiasedness of the nonlinear
   score.

5. Jerzy Neyman (1934), “On the Two Different Aspects of the Representative Method:
   The Method of Stratified Sampling and the Method of Purposive Selection,”
   *Journal of the Royal Statistical Society* 97(4), 558–606.
   [doi:10.1111/j.2397-2335.1934.tb04184.x](https://doi.org/10.1111/j.2397-2335.1934.tb04184.x).
   This is the primary foundation for probability sampling within predefined strata.
   The current proportional largest-remainder allocation is not claimed to be Neyman
   variance-optimal allocation, and the paper does not establish that `TOP10` residual
   mass or family-by-residual cells are optimal strata.

6. Charles Spearman (1904), “The Proof and Measurement of Association between Two
   Things,” *The American Journal of Psychology* 15(1), 72–101.
   [doi:10.2307/1412159](https://doi.org/10.2307/1412159).
   This is the primary source for rank correlation. Spearman rho is suitable for
   checking whether candidate ordering is preserved, but it does not measure score
   calibration or sign preservation.

## Transfer limits for this campaign

- Survey-sampling theory transfers cleanly to the known finite set of videos and the
  design weights on additive sufficient statistics. It does not directly cover the
  nonlinear graph-score transform.
- Subsampling and bootstrap theory is largely asymptotic. With roughly 36 videos and
  `k` as small as 3, empirical retrospective fidelity against banked full results is
  indispensable.
- Candidate estimates share the same selected videos, so their errors and the
  draw-level rank/sign metrics are dependent.
- Spearman rho addresses ordinal fidelity only. Sign agreement and the maximum
  per-candidate 95% width are necessary project-specific complements.
- `|rho| >= 0.8`, sign agreement `>= 0.9`, width `< 0.005`, 10,000 draws, average-rank
  tie handling, and the tie-break order are preregistered campaign choices. They are
  not constants established by these sources.
- Residual stratification is outcome-independent only because it is frozen from the
  reference graph before candidate outcomes open. The literature does not prove that
  this covariate faithfully predicts candidate-specific graph deltas.
- A usable result can support screening speed only. It cannot move a tier or promote a
  mechanism; every promotion still requires the full frozen 44b6-first graph gate.

## Literature verdict

The preregistered estimator has a defensible methodological decomposition:
finite-population probability sampling, inverse-inclusion expansion of additive
sufficient statistics, repeated subsampling to expose design variability, and
rank-based fidelity assessment. The sources do **not** establish that any tested
`(k, scheme)` is faithful. That is solely an empirical N0/N1 question, and an
`UNSAFE` finding would be fully consistent with the literature.
