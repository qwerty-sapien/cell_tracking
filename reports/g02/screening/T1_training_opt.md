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
- **Check remaining Kaggle GPU-hours this week first** (30h/week, resets Saturday 00:00 UTC). A
  20-30 trial Optuna loop on small graph slices should fit comfortably inside one or two ~9h Kaggle
  sessions — run it there and spend $0. Only reach for RunPod if this week's Kaggle quota is already
  spent on a higher-priority node or the search needs more contiguous time than a session allows.
- Estimate cost/trial as ~3.6 GPU-s/pair × pairs/trial BEFORE launching. Every
  RunPod launch requires explicit human approval for the exact job after hours/$ and
  artifact identity are logged; if 20 trials exceed the cap, cut trials first.
- Log every trial: params, score, GPU-hrs (Kaggle) or GPU-s/$ (RunPod).

## Output
`reports/g02/screening/t1_training_opt.md`: top-3 configs, the score-vs-cost curve, and the two-tier
gate decision on the best trial vs baseline. Commits/pushes are allowed; no Kaggle
submission without explicit approval.
