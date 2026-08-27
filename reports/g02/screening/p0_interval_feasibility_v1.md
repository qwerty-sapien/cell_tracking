# P0B interval-aware calibration feasibility v1

Verdict: **STILL-UNRESOLVED**

The three displayed public scores do not admit one affine mapping even when
each score is treated as its full rounding interval
`[displayed - 0.0005, displayed + 0.0005)`. This remains a transfer-design
failure, not a scorer-semantic failure.

No Kaggle submission, Kaggle GPU, RunPod, or external action occurred.

## What is identifiable now

The descriptive center-point OLS fit remains:

`public_hat = 0.2497831845 + 0.7193024335 * local`

It predicts `0.883864155` for APNCC-direct at local `0.881522070`, but that
number is not decision-grade. There are only three adaptively selected public
pairs, all from the graph-cost/solver mechanism.

| Interval test | Result |
|---|---:|
| One unconstrained affine line intersects all three bins | **No** |
| One nonnegative-slope affine line intersects all three bins | **No** |
| Additional uniform expansion needed on each side of every bin | 0.000343242 |
| Mechanism intercept identifiable | **No** |
| Per-mechanism slope identifiable | **No** |
| Covariate regression admissible | **No** |

The extra expansion is a feasibility diagnostic, not a confidence interval.
It would widen each existing 0.001 display bin by about 0.000686 total before
one affine line becomes possible.

## Conflicting slope constraints

| Pair | Public-bin-implied slope range |
|---|---:|
| `c=0` to `c=2` | 0.715626 to 0.799817 |
| `c=0` to `c=0.25` | 0.646332 to 0.727124 |
| `c=2` to `c=0.25` | -2.000708 to 0 |

The first two pairs require a positive slope, while the near-baseline reversal
requires a nonpositive slope. The three ranges have no common intersection.

## Leave-one-pair-out interval envelopes

For each row, every affine line consistent with the other two public bins is
projected to the held local score. “Miss” is the minimum distance from that
projection envelope to the held public bin.

| Held pair | Training-bin projection envelope | Held public bin | Minimum miss |
|---|---:|---:|---:|
| `c=0` | 0.877500 to 0.926028 | `[0.8595, 0.8605)` | **0.017000** |
| `c=2` | 0.875814 to 0.876814 | `[0.8775, 0.8785)` | **0.000686** |
| `c=0.25` | 0.878215 to 0.879300 | `[0.8765, 0.8775)` | **0.000715** |

The two near-baseline bins do not admit a nonnegative-slope fit at all. Their
unconstrained negative-slope fit catastrophically misses the distant `c=0`
anchor. Conversely, each distant-plus-near pair misses the other near pair by
more than half a display bin.

## Prospective APNCC-direct envelopes

Projecting local `0.881522070` from each two-pair subset gives:

| Fit subset | Unconstrained APNCC public envelope | Nonnegative-slope fit |
|---|---:|---|
| `c=0`, `c=2` | 0.884207 to 0.885996 | feasible |
| `c=0`, `c=0.25` | 0.881911 to 0.883588 | feasible |
| `c=2`, `c=0.25` | 0.859749 to 0.877500 | **infeasible** |

This is why the existing broad planning band is not a forecast. Depending on
which incompatible subset is trusted, extrapolation to APNCC-direct changes
materially.

## Gate action

The exact missing measurement is unchanged: **APNCC-direct**, because it is:

- the only observation-distinct, package-ready anchor above local 0.88;
- a new appearance-association mechanism relative to the three public pairs;
- both a performance candidate and the highest-information transfer probe.

If explicitly approved, measure that one configuration and immediately rerun
this audit before selecting another submission. It consumes one competition
submission and remains subject to the five-per-day cap; both final selections
stay reserved.

Machine-readable evidence:
`p0_interval_feasibility_v1.json`
(`sha256 056ee65c228724b68379b1e67c9d8f9f1d596d215a8ce580086d6cd2d0295d8f`).

Reproduction:

```bash
PYTHONPATH=src .venv313/bin/python \
  scripts/analyze_public_calibration_intervals.py
```
