# Screening rubric

The controlling rubric is `P1_framework_and_rubric.md`, adopted on 2026-07-24.

Key rules:

- scorer semantic parity gates local scores;
- transfer calibration gates public inference and final selection only;
- association and detection have independent GPU gates;
- every learned candidate passes a primary-literature/mechanism manifest before
  implementation or training;
- one guarded fit-only M5 fast-fail lane may run alongside eligible external work, with
  an 18 GiB cap, 6 GiB background reserve, no holdout, no swap growth, and <=7 hours per
  unattended stage;
- calibration probes are sequential, observation-distinct, and charged to the
  five-per-competition-day submission cap;
- initial training is capped at 30 cumulative Kaggle T4 GPU-hours per model;
- RunPod is unavailable before an accepted 0.89 score and thereafter requires >5%
  literature/theory-backed expected improvement, a $10 total cap, and explicit approval
  per launch; the ordinary hard improvement gate is 3.5%;
- commits and pushes to the private repository are permitted;
- the performance milestones are >0.88, then robust 0.90s, while 0.92 remains the
  completion threshold.
