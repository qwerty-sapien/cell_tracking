# CG1b — Abstention re-screen on a stronger frozen representation

> Load `.agents/skills/seeking-alpha/SKILL.md`, `.agents/skills/dag-prompts/SKILL.md`,
> `DAG.md`, `cg1_abstention_screen_v1`, and `cg0_ceiling_diagnostic_v1`. Runs after CG1 =
> PARK. Seeking-alpha **Phase 5** re-run: the mechanisms (hard top-k, conformal, PU) were
> grounded and correct; CG1 proved the **representation** is the weak link. This node
> falsifies "a stronger frozen representation lifts tail retention" — cheapest first.

## Objective / gate

At CG0's per-video FP budget (admit ≤ 3,376 background FP overall; the 44b6 sub-budget is
≈0.021%), does a stronger frozen representation lift **known-correct retention to ≥ 20% on
both families**, with a **transferable** operating point? CG1's best was 11.9% / 11.1% on
the parked scorer's scores. This is necessary-not-sufficient for 0.92 (CG2's graph gate
still follows), so keep it cheap.

**Fix CG1's certification flaw:** evaluate FP on the **full unlabeled deployment pool**
per video, not the 8,668-row 44b6 sample — an 0.021% rate cannot be certified on a sample
that small. Use the actual per-video r10/k8 background.

## Stage 1 — existing representations, NO new cache (do this first)

Re-run the CG1 mechanisms using the strongest representations that **already exist for
both families**, alone and combined with the current scores:

- **APNCC 3-D NCC appearance** separation (the banked mechanism; exists for 44b6 and 6bba).
- Any other existing both-family signal (geometry residual, motion consistency) as a
  cheap complement.

Measure retention at the FP budget, per family, on the full pool, with a transferable
operating point (conformal quantile / rank-normalized), 44b6 first. **No new cache, no
Kaggle, no graph replay, no CV57.**

### Stage-1 gate

- **GO** (retention ≥ 20% both families, budget certified on the full pool) → cascade to
  `CG2_build_selected_proposer.md`.
- **PARK** (no material lift over CG1's ~12% — the appearance representation also fails) →
  strong evidence the frozen-representation route is dead → cascade to
  `CG_ESCALATE_goal_feasibility.md`.
- **PARTIAL** (a real lift toward 20% but short, i.e. the representation clearly helps) →
  Stage 2 is justified.

## Stage 2 — Trackastra both-family cache (only on Stage-1 PARTIAL)

Only if Stage 1 shows the representation is the lever: build the complete Trackastra +
APNCC 49-D cache for **both** families and re-screen.

- 6bba: complete the cache via `K_trackastra_6bba_feature_kernel_v1` (**HUMAN-GATE** —
  Kaggle push needs explicit approval) or local completion once host swap clears.
- 44b6: build the 44b6 feature pipeline (mirror the 6bba proposals; relax the fail-closed
  `train:6bba_` eligibility) — this does not exist yet and is real CPU work.
- Re-run the Stage-1 screen on the 49-D representation, same gate.

Preregister before building: "Trackastra representation must reach ≥ 20% retention both
families at the FP budget; else the frozen-representation route is dead → escalate." Do
not build the 44b6 pipeline unless Stage 1 justified it — that is the whole point of
cheap-first.

## Output

`reports/g02/screening/candgen/cg1b_representation_rescreen_v1.{md,json}` + LEDGER rows:
per-representation, per-family retention at budget, full-pool certification, and the
Stage-1 verdict (GO / PARK / PARTIAL). Fit-only; leakage guards; reject holdout/CV57 paths.

## Verdict & cascade

- **GO** → `CG2_build_selected_proposer.md` in the same run.
- **PARTIAL** → Stage 2 (its Trackastra cache step is a HUMAN-GATE — halt for Kaggle
  approval; the 44b6 pipeline build may proceed on CPU).
- **PARK** → `CG_ESCALATE_goal_feasibility.md`.
