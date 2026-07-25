# CG1 — Seeking-alpha search over abstention-realization mechanisms

> Load `.agents/skills/seeking-alpha/SKILL.md`, `.agents/skills/dag-prompts/SKILL.md`,
> `DAG.md`, and `cg0_ceiling_diagnostic_v1`. Runs on CG0 = **GO**. Seeking-alpha
> **Phases 3–5**: locate the alpha node, generate non-obvious mechanisms, ground and
> cheaply falsify each — before any model is built.

## The alpha node (corrected by CG0)

The correct edges are already in the current r10/cap-8 pool (perfect-selection oracle
0.960918). The unsolved problem is **realization**: reject **≥99.957 %** of the 7.8 M
*unlabeled* background (admit ≤ 3,376 FP on fit117) while keeping enough of the 74 k
correct edges to clear 0.92, and do it with a rule that **transfers across videos and
holds on 44b6**. Target the **present-but-unrankable** population first (CG0's directive);
wider-radius/gap unions are deprioritized (harsher FP budgets).

Run the crowding test: another from-scratch MLP ranker is beta *and* DEAD (§6.1) — it
died on exactly the three failure modes here (threshold non-transfer, unlabeled-background
abstention, 44b6). Force a mechanism that is *structurally* different.

## Generate mechanisms (Phase 4) — mine these transfers

Produce several candidates, each with a one-line defensibility reason. High-yield
transfers the failed rankers never used, aimed at the three failure modes:

- **Selective prediction / conformal risk control** (statistics/ML): abstain unless
  confident *at a guaranteed error rate*. "Admit ≤ 3,376 FP" is a conformal FP-budget by
  construction, and conformal calibration is designed to **transfer across exchangeable
  groups (videos)** — directly attacking the IQR-0.237 non-transfer.
- **PU-learning (positive-unlabeled)** (ML): the background is *unlabeled-uncertain*, not
  negative (the crop-linker gate said as much). Estimate the class prior and score under a
  PU risk, instead of treating background as clean negatives.
- **Per-video calibration transfer / domain adaptation:** replace the global threshold
  with a *transferable statistic* (per-video conformal quantile, rank-normalized score,
  distribution-matched calibrator) so one rule holds across 44b6 and 6bba.
- **First principles:** optimize the *held-fit graph* objective's own FP/FN trade at the
  frozen scorer, not a proxy precision — the proposal→graph gap is proven.

## Ground and falsify (Phase 5) — the rigor gate

Each mechanism must pass both, cheaply, before it earns a build:

- **Evidentiality:** a real primary source (conformal prediction / selective
  classification; PU-learning) *and* a stated reason it holds on this data (exchangeability
  across videos, plausible class prior). An un-producible citation is a red flag.
- **Local falsification against CG0's budget:** on fit-only folds, **44b6 first**, measure
  whether the mechanism keeps its FP admissions within the per-video budget *and*
  transfers its operating point to held videos without a family collapse. Preregister:
  "held FP ≤ budget with correct-edge retention ≥ [target], both families; else dead." A
  mechanism that only works with a per-video oracle threshold has not solved the transfer
  problem — that is the whole point.

Classify each survivor **plan** / **research-bet** / **guess** (seeking-alpha rigor doc).

## Output

`reports/g02/screening/candgen/cg1_abstention_screen_v1.md`: candidate mechanisms, each
with transfer + citation, crowding verdict, held FP/retention numbers per family, and a
label. Rank survivors by pertinence × feasibility × defensibility. LEDGER rows. No held
graph replay yet, no CV57, no Kaggle action.

## Verdict & cascade

- **GO** — ≥1 **plan** mechanism (grounded, within the FP budget, transfers to both
  families) → proceed directly to `CG2_build_selected_proposer.md` in the same run.
- **PARK / DEAD** — no mechanism transfers → halt and report; consider `CG_ESCALATE`.
