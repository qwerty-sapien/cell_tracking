# Video-subsample fidelity preregistration

Status: **FROZEN BEFORE ANY SUBSAMPLE OUTCOME**

Date: 2026-07-29

## Objective and authority

This retrospective node asks whether a small, CPU-only video subsample predicts the
already-banked full-family graph delta closely enough to be useful for future
**screening only**. It does not rescore a graph, open a new family or CV57 outcome,
alter a tier, revisit a DEAD mechanism, or authorize a promotion. Every promotion
continues to require the full frozen 44b6-first gate.

Only pre-existing per-video decompositions for the named banked candidates are
eligible: APNCC, C1, H1-ZWRFEAT, SA-DELETE, SA-SWAP, and NODETOPO. A candidate is
included only if its banked per-video sufficient statistics:

1. cover the same authenticated full-result video universe used by its cited banked
   full delta;
2. reconstruct that full delta to absolute tolerance `1e-12`; and
3. can be combined without reading predictions, labels, scores, or fragments from a
   new family or split.

No missing video is imputed. No candidate is silently dropped. If the common,
like-for-like candidate set has fewer than four candidates, or if any named candidate
lacks an eligible decomposition, the primary verdict is `UNSAFE_INCOMPLETE_EVIDENCE`;
coverage-limited diagnostics may still be emitted but cannot freeze a contract.

## Frozen estimand and estimator

The estimand is each candidate's banked full-family delta in the exact official
graph objective

`S(G) = weighted_adjusted_edge_Jaccard(G) + 0.1 * division_Jaccard(G)`.

For every selected video set, additive per-video sufficient statistics are expanded
with their design weights and then passed through the same score formula. The
subsample delta is the resulting candidate score minus its banked reference score.
Per-video delta averaging is forbidden because the graph objective is nonlinear.

Sampling is without replacement from lexicographically sorted authenticated video
IDs. The pseudorandom generator is NumPy `Generator(PCG64(20260729))`. Each
`(k, scheme)` uses an independently derived seed
`SHA256("g02-subsample-fidelity-v1|20260729|<k>|<scheme>")[:16]`, interpreted as an
unsigned 64-bit big-endian integer, so loop order cannot change the draws. Exactly
`10,000` repeated finite-population subsampling draws are used.

The tested sizes are `k in {3, 5, 8, 12}`.

### Sampling schemes

1. `simple_random`: simple random sampling without replacement. Every selected video
   receives the common expansion weight `N/k`.
2. `residual_top10`: two strata defined only from the frozen reference graph, before
   candidate outcomes are opened. Residual mass is the reference's additive
   adjusted-edge error mass (weighted false-positive plus weighted false-negative
   mass) for a video. The ten largest values, with video ID as the tie-break, form
   `TOP10`; all others form `REST`. Allocation is proportional to stratum size by
   largest remainder, with at least one video from each nonempty stratum. A selected
   video in stratum `h` receives weight `N_h/n_h`.
3. `family_residual`: the Cartesian strata `(family, TOP10-or-REST)`, using the same
   global top-ten definition. Allocation is proportional by largest remainder with
   at least one video from each nonempty stratum. If `k` is smaller than the number
   of nonempty strata, that cell is structurally non-estimable and cannot be usable.
   A selected video in stratum `h` receives weight `N_h/n_h`.

If only one family is present in the authenticated full-result universe,
`family_residual` is reported as degenerate with `residual_top10` and cannot win a
tie over it.

## Frozen fidelity metrics

For each draw and `(k, scheme)`, compute:

- Spearman rank correlation between the vector of subsample deltas and the vector of
  cited banked full deltas across eligible candidates, using average ranks for ties;
- sign agreement across candidates, where `sign(x)` is `-1`, `0`, or `+1` with no
  tolerance band.

For each `(k, scheme)`, the primary reported rank statistic is the median draw-level
Spearman `rho`; the primary sign statistic is the fraction of all
candidate-by-draw signs agreeing with the full delta. The 95% interval for a
candidate is the percentile interval from its 2.5th to 97.5th subsample-delta
quantiles. The primary width is the **maximum** such width across candidates. Also
report the 2.5th/97.5th percentiles of draw-level rho, the draw-level sign-agreement
distribution, every candidate interval, bias, RMSE, and median absolute error as
non-gating diagnostics.

## Acceptance rule and verdict

A `(k, scheme)` is `USABLE` if and only if all three preregistered conditions hold:

1. `abs(median Spearman rho) >= 0.8`;
2. pooled sign agreement `>= 0.9`; and
3. maximum candidate 95% interval width `< 0.005`.

Negative rank correlation is allowed by the literal preregistered `|rho|` rule but
must be flagged as direction-reversing; a reusable contract additionally requires
positive median rho because a reversed ranking is not operationally faithful.

The selected scheme is the smallest usable `k`. Ties at the same `k` are broken by:
smaller maximum interval width, then higher sign agreement, then higher positive
median rho, then lexical scheme name.

If a pair is selected, its exact reusable video list is draw zero from the frozen
generator and allocation rule. The list is outcome-independent: no later draw may be
chosen because it performed better. The contract records the list, estimator,
weights, source hashes, seed derivation, scope, and its own SHA-256. It is valid only
for screening. If no pair passes, or evidence coverage is incomplete, emit
`SUBSAMPLING_UNSAFE` and no contract.

## Hard stops

- No new graph scoring, CV57 access, second-family opening, fitting, GPU, Kaggle,
  RunPod, submission, or budget debit.
- No modification of N1, folds, P0, caches, banked candidate artifacts, the locked
  plan, or any DEAD mechanism.
- Diagnostics never gate a tier or promotion.
- Root is the sole writer of `LIVE_STATE.md`, `LEDGER.md`, and `tier_census.json`.
