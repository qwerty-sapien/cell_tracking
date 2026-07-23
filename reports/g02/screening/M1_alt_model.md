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
- SCREEN on the worst family first. A candidate needs `>= +0.005` on that slice at CONFIRM to justify
  a full build. `+0.002..+0.005` → human-decision note. `< +0.002` → PARK with reason.
- **Compute:** use remaining Kaggle GPU-hours (30h/week, resets Saturday 00:00 UTC) before RunPod —
  check what T1 already consumed from this week's quota. Only spend RunPod $ if Kaggle quota for the
  week is exhausted or the spike needs more contiguous time than a ~9h session allows,
  and only after explicit human approval for that exact launch.
- Log GPU-hrs (Kaggle) or GPU-s/$ (RunPod) per candidate.

## Output
`reports/g02/screening/m1_alt_model.md`: one-page comparison — candidate, slice Δ, full Δ, cost, and a
DEAD/PARK/BUILD verdict. Commits/pushes are allowed; no Kaggle submission or RunPod
launch without explicit approval. Feeds S1.
