# CG0 — Candidate-generation ceiling diagnostic (ROOT GATE)

> Load `.agents/skills/seeking-alpha/SKILL.md` and `DAG.md`. This is seeking-alpha
> **Phase 1**: make the objective computable before searching. Deterministic, fit-only,
> CPU. It gates the entire sub-DAG.

## Why this is the root

`ACH-HEADROOM` and `GA-MGSHELL` proved that *selection* over the current and shell pools
caps at ~0.896 / ~0.898, while the full-association oracle (correct nodes) is 0.975. The
goal (0.92) sits in between. Before building any generator, we must know **which
candidate-generation policy — if any — has a perfect-selection oracle ≥ 0.92**. If none
does, no downstream model can reach the goal and we escalate instead of building.

## The objective, made computable

For a candidate-generation *policy* P (a rule that proposes the edge set fed to the
solver), define:

`ceiling(P) = frozen-scorer score of the oracle selection over P's proposals`,

computed fit-only, with the exact banked graph/solver, divisions off. This is the single
number every mechanism in CG1 will be falsified against.

## Tasks

1. **Decompose the 0.898 → 0.975 gap by *cause of exclusion*.** Take the correct
   associations that the current/shell pools miss and bucket each:
   - **out-of-radius** (beyond 10 µm / beyond 12 µm shell),
   - **sub-cap** (excluded by cap-8),
   - **gap-closing / multi-frame** (correct edge spans a frame gap or a missed
     detection, so no single frame-pair candidate exists),
   - **present-but-unrankable** (in-pool but no transferable threshold separates it from
     background — the IQR-0.237 population).
   Report counts and the *oracle ceiling recoverable* if each bucket were perfectly
   admitted, per family (44b6 vs 6bba). This tells you which generation change even
   *could* matter.

2. **Compute ceiling(P) for a ladder of deterministic policies**, cheapest first, each
   fit-only with the exact frozen scorer:
   - current pool (baseline ceiling 0.895678 — reproduce it as a control),
   - + 12 µm shell (reproduce 0.897746),
   - + multi-frame / gap-closing candidates (bridge one missed frame),
   - + wider radius with an *oracle* background filter (upper bound if abstention were
     perfect),
   - + any union that a mechanism in CG1 would target.
   For each, report the ceiling **and** the background burden (pool size, correct/background
   ratio) — the ceiling is only useful if the background is tractable for a real model.

3. **State the verdict.** Emit, per family and overall:
   - the **maximum achievable oracle** across policies, and the policy that reaches it;
   - whether **any** policy's oracle ≥ 0.92 (GO) or not (ESCALATE);
   - the **background budget** the eventual real model must meet to realize it.

## Output

`reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.{md,json}`:
the exclusion-bucket table, the policy→(ceiling, background) ladder, per-family maxima,
and a one-word verdict **GO** (→ CG1) or **ESCALATE** (→ CG-ESCALATE). Add a LEDGER row.
No model training, no holdout/CV57 access, no Kaggle action. Reject holdout/CV57 paths
before reading (mirror the `analyze_achievable_headroom` hygiene fix).

## Verdict & cascade

**Result: GO.** The current r10/cap-8 pool's perfect-selection oracle is **0.960918 ≥
0.92** — candidate generation is *not* the binding wall; **selection/abstention
realization** is (reject ≥99.957 % of 7.8 M unlabeled background, admit ≤ 3,376 FP,
transferably across videos and on 44b6). The present-but-unrankable population is the
first target; wider-radius/gap unions carry harsher FP budgets.

- **GO** → proceed directly to `CG1_alpha_candidate_generation.md` in the same run,
  reaimed at abstention realization (not candidate generation).
- **ESCALATE** (not taken) — had no policy reached 0.92, the goal would have been
  unreachable over this detection front-end → `CG_ESCALATE_goal_feasibility.md`.
