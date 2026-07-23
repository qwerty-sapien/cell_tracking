# P0 public-calibration probe batch — 2026-07-23

Status: **SUPERSEDED — DO NOT SUBMIT AS A BATCH**

These packages remain candidate artifacts, but each must first pass P0R's canonical
observation classification. Any approved probe is measured one at a time, counts against
the five-per-competition-day submission cap, and is followed by a redesign of the
remaining order.

These three probes were selected to test whether the public score can be
calibrated from the local CV57 score alone, or whether the calibration depends
on the association-parameter family.

All three use the historical forward-motion configuration
(`motion_error_weight=1.0`) and vary only the learned-link bonus. Their exact
CV57 scores were freshly reproduced through the current graph and frozen
scorer in `p0_lb_replay_cv57_v1/report.json`.

| Submit order | Kernel | Experiment | Learned bonus | Exact local CV57 |
|---:|---|---|---:|---:|
| 1 | `onebitbrain/bhp0-lb16` | `p0-lb16` | 16 | 0.8479018413756242 |
| 2 | `onebitbrain/bhp0-lb4` | `p0-lb4` | 4 | 0.8720910462365732 |
| 3 | `onebitbrain/bhp0-lb8` | `p0-lb8` | 8 | 0.8670102112352311 |

The first two are the decisive matched-local comparisons:

- `p0-lb16` versus submitted `p0-c0`: local scores differ by only
  0.0004926742. A public-score difference of at least 0.002 is evidence that
  one global local-to-public calibration is invalid.
- `p0-lb4` versus submitted `p0-c2`: local scores differ by only
  0.0000589012. A public-score difference of at least 0.002 is likewise
  evidence of parameter-family dependence.
- `p0-lb8` supplies an interior point to test monotonicity and estimate a
  family-specific forward-motion calibration.

Because Kaggle displays only three decimal places, a difference of 0.001 is
ambiguous due to rounding. Treat at least 0.002 as decisive.

P0B transfer calibration remains unresolved until sufficient observation-distinct,
mechanism-diverse public evidence exists. This does not halt P0A-valid local work or the
independent GA/GD model-development gates.
