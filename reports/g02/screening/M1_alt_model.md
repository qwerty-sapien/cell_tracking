# M1 — Alternative model spike (GPU-gated)

> First load `_shared_context.md`. Use GA for association candidates and GD for
> detection/front-end candidates. T1 is not a prerequisite for the GD branch.

## Goal
A time-boxed spike (≤ 1 day, within the P1 ration) testing whether a different model class has a
higher realistic ceiling than the current pairwise-edge scorer, evaluated on the **same** patched
57-video scorer for apples-to-apples comparison.

## Candidates — pick ONE per spike, cheapest-ceiling first, NO GNN
- **Stronger detection/segmentation front-end** feeding the existing graph. Prioritize this if E1's
  verdict was DETECTION-BOUND (in which case T1/M1 were parked and this is the real lever).
- **Geometry-aware edge transformer** with richer relative-geometry features than the dead P4
  prototype (P4 attention closed at −0.00028 vs the no-layer model — don't rebuild it as-is).
- **Learned association cost** replacing the hand-tuned motion term (backward motion c=0.25).

## Rubric
- Before implementation, complete and validate a candidate manifest derived from
  `reports/g02/model_candidate_template_v1.json`, including primary peer-reviewed
  evidence, the measured bottleneck-to-mechanism link, >=3.5% expected improvement, and
  preregistered failure/promotion criteria.
- Prefer a lightweight M5-compatible candidate and run its fit-only falsification through
  `scripts/run_local_training_guard.py` before external GPU use. One local job may run in
  parallel with eligible external work; it is limited to 18 GiB, reserves 6 GiB, forbids
  holdout access and swap growth, and stops within seven hours.
- SCREEN on the worst family first. A candidate needs `>= +0.005` on that slice at CONFIRM to justify
  a full build. `+0.002..+0.005` → human-decision note. `< +0.002` → PARK with reason.
- **Compute:** at most 30 cumulative Kaggle T4 GPU-hours per model for initial training;
  archive a candidate that cannot become viable inside that ceiling. RunPod is unavailable
  before an accepted score reaches `0.89` and thereafter requires a
  higher-quality-GPU-dependent optimization with literature/theory-backed expected
  improvement greater than `5%`, the `$10` total cap, and exact-launch approval. The
  ordinary hard improvement gate is `3.5%`.
- Log GPU-hrs (Kaggle) or GPU-s/$ (RunPod) per candidate.

## Output
`reports/g02/screening/m1_alt_model.md`: one-page comparison — candidate, slice Δ, full Δ, cost, and a
DEAD/PARK/BUILD verdict. Commits/pushes are allowed; no Kaggle submission or RunPod
launch without explicit approval. Feeds S1.
