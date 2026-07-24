# Shared context (read before any prompt in this folder)

**Repo:** `/Users/agent/Desktop/cell_tracking` (work on a branch off `main`).
Classical BH-graph cell-tracking pipeline for the Biohub "cell tracking during development"
competition.

## Standing
- Public: **0.881** from APNCC-direct (`0.881522` local). Frontier: **0.929**
  (post-rescoring).
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
- Machine-readable authority: `reports/g02/compute_governance_v1.json`.
- **Preferred exploration funnel:** literature-validated lightweight candidates first.
  Before implementation or training, validate a candidate manifest derived from
  `reports/g02/model_candidate_template_v1.json`. Run the cheapest fit-only falsification
  first; only promote a candidate whose preregistered learning and runtime gates pass.
- **Standing local lane:** keep at most one guarded M5 training job available in parallel
  with eligible external work. Use `scripts/run_local_training_guard.py`: process-tree
  training memory <= **18 GiB**, background reserve >= **6 GiB**, no holdout access, no
  swap growth, and <= **7 hours** per unattended stage. A resource violation stops the
  stage and fails the runtime gate.
- **Submissions:** competition rules allow **5/day, 2 final selections**. Treat submission as cheap
  for screening, but final selections are precious and rescoring-sensitive.
- **Compute hierarchy — spend free quota before paid dollars:**
  1. **Kaggle GPU quota (free, initial training only):** at most **30 cumulative T4
     GPU-hours per model**, across sessions and quota resets. Archive a model when its
     minimum credible training state cannot be reached inside that ceiling. Track both
     per-model cumulative use and the live weekly quota in `LEDGER.md`.
  2. **RunPod (paid, gated optimization only):** **$10 total** and never used for initial
     training. It may be considered only after an artifact-linked accepted score reaches
     **0.89** and the exact higher-quality-GPU-dependent optimization has literature or
     theoretical support for an expected score improvement **greater than 5%**. The
     ordinary hard improvement gate is **3.5%**. Estimate cost before launch, log every
     dollar, and obtain explicit human approval for that exact launch.
- Passing the local fast-fail lane does not itself authorize Kaggle or RunPod and does
  not relax either platform's gates.
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
