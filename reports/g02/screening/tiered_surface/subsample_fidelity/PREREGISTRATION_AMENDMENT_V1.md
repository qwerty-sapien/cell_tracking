# Video-subsample fidelity preregistration — pre-outcome universe amendment

Status: **FROZEN BEFORE ANY SUBSAMPLE OUTCOME**

Date: 2026-07-29

This amendment supersedes only the source-universe eligibility paragraphs in
`PREREGISTRATION.md`. The estimator, schemes, seed derivation, 10,000 draw count,
metrics, thresholds, tie-breaking, screening-only scope, and hard stops are unchanged.

## Why an amendment was required

The pre-outcome source audit found that historical reports used three presentation
universes:

- APNCC's older full result is CV57 (57 videos);
- SA-DELETE reports 41 44b6 videos, consisting of the later held-gate 36 plus five
  additional videos;
- C1, H1-ZWRFEAT, SA-SWAP, and NODETOPO report the same 36-video held-fit 44b6 gate.

It also found two historical controls: CG0 and APNCC. Ranking deltas taken directly
from these heterogeneous reports would confound candidate effect with universe and
reference choice. No bootstrap/subsample draw, draw-zero list, rho, sign rate, or
interval was computed or inspected before this amendment.

## Amended primary estimand

The primary retrospective universe is the exact common 36-video 44b6 held-gate list
banked by C1, H1-ZWRFEAT, SA-SWAP, and NODETOPO. This is the campaign's required
first-family screening gate and therefore the relevant target for a reusable
screening shortcut.

Every candidate delta is standardized to:

`candidate score on the common 36 videos - CG0 score on the same 36 videos`.

The candidate set remains exactly:

1. APNCC;
2. C1;
3. H1-ZWRFEAT;
4. SA-DELETE;
5. SA-SWAP;
6. NODETOPO.

APNCC uses its already-banked 36-video control graph from the later held-fit gates.
SA-DELETE uses only its 36 rows whose IDs are in the common held-gate list; its five
additional banked rows are excluded. C1, H1-ZWRFEAT, SA-SWAP, and NODETOPO use their
banked 36-video candidate graphs. CG0 uses the already-banked common-36 control graph.
No new model, graph construction, family, split, prediction, or label is opened.

The historical 57-video APNCC and 41-video SA-DELETE deltas remain provenance
diagnostics and are not mixed into the cross-candidate rank metric.

## Permitted decomposition replay

Where a terminal report already contains per-video scorer sufficient statistics, use
those rows directly. Where it banks candidate/control fragment bytes plus complete
provenance but only aggregate scorer output, a CPU-only invocation of the frozen
official evaluator on each already-scored video is permitted solely to recover the
missing per-video sufficient statistics.

This is a decomposition replay of an opened result, not a new scientific outcome.
It must satisfy all of the following before any subsample draw:

1. exact candidate and control fragment SHA-256s match the banked report;
2. video IDs equal the amended common-36 universe;
3. summing/recombining the recovered sufficient statistics reproduces the report's
   candidate and control aggregate scores and deltas to absolute tolerance `1e-12`;
4. evaluator/scorer source hashes match the banked parity authority; and
5. no fit, model inference, graph construction, CV57, 6bba, audit/test split, GPU, or
   external action occurs.

Any failed aggregate replay, missing fragment, missing already-opened label authority,
or source-hash mismatch makes that candidate ineligible and forces
`UNSAFE_INCOMPLETE_EVIDENCE`. No candidate may be silently dropped.

## Stratification consequence

The amended primary universe contains one family, 44b6. Therefore
`family_residual` is a preregistered degenerate duplicate of `residual_top10`; it must
be reported but cannot win a tie over `residual_top10`. Residual mass is computed from
the common-36 CG0 reference only; the older CV57 B2 top-ten list is not reused.

## Interpretation

A passing contract predicts only the standardized 36-video 44b6 screening result. It
does not predict CV57, 6bba, a full promotion gate, or the public leaderboard. Every
promotion still requires the full frozen 44b6-first graph gate and all downstream
campaign gates.
