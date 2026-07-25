# T1 — Training optimization (GPU-gated)

> First load `_shared_context.md`. This is the GA association branch. Do not start
> unless GA in P1 is satisfied.

## Goal
Squeeze the existing learned edge scorer (`scripts/train_edge_scorer.py`) without changing
architecture, on the **worst family from B2** — not the global set.

## Design — small Bayesian search, not a grid
- Use a TPE / Bayesian loop (e.g. Optuna), ~20–30 trials.
- Search space (hypothesis: more-stochastic, higher-variance training escapes the plateau):
  - optimizer ∈ {AdamW, RMSprop, SGD+momentum}
  - lr log-uniform [3e-4, 3e-2] (bias high)
  - dropout [0.1, 0.5]
  - batch size ∈ {8, 16, 32} (bias small)
  - weight decay log-uniform [1e-6, 1e-3]
  - schedule ∈ {cosine, onecycle, constant}
- **Objective = patched 57-video CONFIRM score on the target family's held-out videos**, never
  training loss.

## Budget
- Optimization begins only after the existing model's literature/mechanism manifest and
  fit-only local falsification are valid. Run lightweight trials through the single
  guarded M5 lane first when compatible: <=18 GiB, >=6 GiB background reserve, no
  holdout, no swap growth, and <=7 hours per unattended stage.
- Initial training is capped at **30 cumulative Kaggle T4 GPU-hours for this model**,
  across all sessions. Project the full loop first and archive/cut it if it cannot become
  viable within that ceiling.
- Estimate cost/trial as ~3.6 GPU-s/pair × pairs/trial BEFORE launching. Every
  RunPod launch is prohibited before an accepted score reaches `0.89`; any later
  higher-quality-GPU-dependent optimization requires literature/theory-backed expected
  improvement greater than `5%`, must fit the `$10` total cap, and needs explicit approval
  for the exact logged job. The ordinary hard improvement gate is `3.5%`.
- Log every trial: params, score, GPU-hrs (Kaggle) or GPU-s/$ (RunPod).

## Output
`reports/g02/screening/t1_training_opt.md`: top-3 configs, the score-vs-cost curve, and the two-tier
gate decision on the best trial vs baseline. Commits/pushes are allowed; no Kaggle
submission without explicit approval.
