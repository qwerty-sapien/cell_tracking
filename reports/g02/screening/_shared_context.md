# Shared context (read before any prompt in this folder)

**Repo:** `/Users/agent/Desktop/cell_tracking` (work on a branch off `main`).
Classical BH-graph cell-tracking pipeline for the Biohub "cell tracking during development"
competition.

## Standing
- Public: **0.877**, rank ~787. Frontier: **0.929** (post-rescoring).
- The rescoring dropped the former champion **0.985 → 0.913**: the new metric punishes whatever
  the old leaderboard rewarded. **Robust, generalizing solutions win now** — weight this in every
  keep/park decision.
- Local 57-video holdout baseline = **0.873150** (backward motion `c=0.25` banked, divisions off).
- Immediate performance milestone: **break 0.88**, then pursue robust scores in the
  competitive **0.90s**; G-02 completion remains 0.92 on grouped CV, audit, and Kaggle.

## The one true scorer
- Use the exact patched 57-video scorer behind `reports/g02/scorer_parity_v1.json`.
- Entry points: `scripts/score_bh_cv57.py`, `scripts/check_scorer_parity.py`,
  `src/celltrack_alpha/scorer_parity.py`. Do not invent a new scoring path.
- Banked `c=0.25` edge confusion: **TP/FP/FN = 34,205 / 2,685 / 2,063**,
  `W = 38,953`.
- Keep the older frozen submission-path artifact distinct:
  `kaggle_kernel/bhcv57out_v3/cv57.csv` scores **0.8711885823** with
  **34,174 / 2,736 / 2,094**, `W = 39,004`. Never assert one identity with the
  other's tuple.

## Budgets & rules
- **Submissions:** competition rules allow **5/day, 2 final selections**. Treat submission as cheap
  for screening, but final selections are precious and rescoring-sensitive.
- **Compute hierarchy — spend free quota before paid dollars:**
  1. **Kaggle GPU quota (free, use first):** **30 GPU-hours/week**, resets **Saturday 00:00 UTC**.
     This is a hard wall-clock quota, not money — it does not roll over and cannot be topped up.
     Any T1/M1 training that fits a Kaggle notebook session (P100/T4×2, ~9h/session max) should run
     there before touching RunPod. Track cumulative hours used this week in LEDGER.md so a node
     never gets started that can't finish before the reset guillotines it mid-run.
  2. **RunPod (paid, last resort):** **$20 total**, of which **$5–10 is rationed for training**,
     spent ONLY when a model is likely the only remaining way to be competitive AND the job doesn't
     fit Kaggle's weekly quota/session limits (e.g. needs >9h contiguous, or the week's 30h is
     already spent on higher-priority runs). Estimate cost from the ~3.6 GPU-s/pair budget × pair
     count BEFORE launching, log every dollar, and obtain explicit human approval for
     that exact launch.
- **No GNN.** HOCT adjusted homophily ≈ 0.01 on this graph class; a geometry-free edge transformer
  already beat GNN variants (0.922 vs 0.916/0.909).
- Ordinary commits and pushes to this private repository are permitted. Do not rewrite
  shared history or publish artifacts.
- **No Kaggle submission or RunPod launch without explicit human approval.**

## Historical effect sizes (calibrate expectations)
Every intervention measured on this pipeline landed **+0.00003 to +0.0020**. The two-tier gate in
P1 exists because a +0.010 gate would reject a genuinely good direction. Do not expect large deltas;
do expect to have to stack several small robust ones.

## Reporting contract for every node
Write results to `reports/g02/screening/LEDGER.md` (owned by the root controller): node id, tier reached,
**predicted vs actual** delta, GPU-s and $ if any, decision (BANK / PARK / DEAD / HUMAN-DECISION),
and a one-line reason. PARK is the default for weak results; DEAD only on unusually compelling
negative evidence.
