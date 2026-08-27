# P0B interval-aware calibration feasibility v2

Verdict: **STILL-UNRESOLVED for fine global transfer**

Operational high-range alignment: **PASS**

APNCC-direct scored `0.881522070` on the frozen local CV57 scorer and `0.881`
on the displayed Kaggle public leaderboard. The absolute difference is
`-0.000522070`, inside one `0.001` display tick. Relative to the `c=0.25`
baseline, the local improvement was `+0.008372476` and the displayed public
improvement was `+0.004`; the sign transfers, while the displayed magnitude is
about 47.8% of the local magnitude.

This is sufficient to trust the scorer for coarse high-range development and
to bank the first public score above 0.88. It is not sufficient to rank
near-ties or fit a mechanism-specific calibration law.

## Artifact authentication

`results-bhap.zip` was read and tested in place; it was not expanded.

| Identity | Value |
|---|---|
| Output archive SHA-256 | `031aa5f64fd6eaf918213fc072a098657e2a6bc3d3e5a2d2ae0642d4037b7219` |
| Streamed `submission.csv` SHA-256 | `6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3` |
| `run.json` submission SHA-256 | `6944ed58c5b1c6b636ebf2cd62f673d13e7edbd7ad3ae1eecaf99d5a1548bea3` |
| Run status | 4/4 videos, `ok=true`, no worker errors |
| Submission rows | 263,282: 134,409 nodes and 128,873 edges |
| Runtime | 534.968 seconds |
| APNCC contract SHA-256 | `cee547ed7a82d1bf7f6afe8cbbc3cc9d554181f9e19e87d319a5350863718cf7` |

The archive integrity test reported no compressed-data errors.

## Four-pair interval audit

Every displayed score is represented as
`[displayed - 0.0005, displayed + 0.0005)`.

| Configuration | Mechanism | Local CV57 | Public interval |
|---|---|---:|---:|
| `c=0` | graph cost/solver | 0.848394516 | `[0.8595, 0.8605)` |
| `c=2` | graph cost/solver | 0.872149947 | `[0.8775, 0.8785)` |
| `c=0.25` | graph cost/solver | 0.873149594 | `[0.8765, 0.8775)` |
| APNCC-direct | appearance association postprocessor | 0.881522070 | `[0.8805, 0.8815)` |

No unconstrained or nonnegative-slope affine line intersects all four bins. A
uniform expansion of `0.000970557` on each side is required for feasibility.
The four-point center OLS,

`public_hat = 0.3010866732 + 0.6594275648 * local`,

is descriptive only. Its residuals are `-0.000541`, `+0.001794`,
`+0.000134`, and `-0.001387`; the largest exceed one display tick.

With only one public APNCC anchor, neither a stable mechanism intercept nor an
APNCC slope is identifiable. Four pairs also do not support a multivariable
covariate regression. The appropriate verdict therefore remains
`STILL-UNRESOLVED` for fine global transfer rather than claiming universal
global adequacy or family specificity.

## Submission action

Do not spend another slot on a calibration-only near-tie. Under the
five-submissions-per-competition-day cap, the next public measurement should
also advance accuracy:

- require an observation-distinct candidate;
- target local CV57 at least `0.89`;
- submit one candidate at a time, only after explicit approval;
- feed its rounded result back into this audit;
- keep both final selections uncommitted.

A second score-separated APNCC/appearance anchor would be needed to estimate a
mechanism effect, but it should come from a materially stronger candidate, not
from a dedicated calibration probe.

Machine-readable evidence:
`p0_interval_feasibility_v2.json`
(`sha256 50e8af76b071030683d46824df4c5f11c98ea839fbbdd8d236b8ae2f9d128404`).

No Kaggle submission, GPU launch, RunPod action, archive extraction, or push
was performed while producing this audit.
