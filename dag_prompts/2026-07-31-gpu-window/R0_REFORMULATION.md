# R0 — batch / sequential action valuation: the reopening condition, on its merits

**Zero GPU. CPU and design work. This is the recommended use of the next 20 hours.**

`TERMINAL_CAMPAIGN_REPORT.md` closed the tiered surface and named exactly two ways back in.
This node pursues clause 1:

> a **new action-labeling formulation** that changes the estimand — for example, batch or
> sequential action valuation that assigns value in the context of the global edit set

It does **not** need an override of the terminal report to begin *design and preregistration*.
It does need one — plus a fresh hash-pinned preregistration and human review — before any
graph run, GPU, `6bba`, CV57, or submission. Do not blur those.

## Why this is the right target

Every measured supplier converges near AUC `0.670` on the pointwise label:

| Source | Information type | DELETE A1 mean AUC |
|---|---|---:|
| DINOv2 ViT-S/14 | frozen endpoint appearance | `0.670412` |
| C2 assignment evidence | MAP / min-marginal / K-best combinatorics | `0.670235` |

Difference: `0.000176`, across two sources that share essentially no information. The
requirement is `0.85`. Convergence this tight across unrelated feature families is evidence
that the **label** is the bottleneck.

There is independent, quantitative support for that already in the campaign, and it predates
the terminal report — the marginal-value gate's own worked example:

- The legacy CG0 swap oracle valued actions **without** legality or degree constraints and
  scored `+0.075828`. The frozen, node-disjoint, legality-constrained population realized
  `+0.018571397` — a **`24.49%` realization, a `4.08×` collapse.**

That gap *is* the context-dependence the pointwise label throws away. An action's value
depends on which other actions are applied; scoring each one in isolation asks a question
whose answer does not determine the objective. That is a mis-specified estimand, and no
encoder fixes a mis-specified estimand.

## Task

1. **Write the diagnosis first**, with numbers, as `REFORMULATION_DIAGNOSIS.md`: the `0.670`
   convergence, the `24.49%` oracle-to-realized collapse, and the argument connecting them.
   If the argument does not survive being written down precisely, stop here and say so — that
   outcome is worth more than a preregistration built on a hunch.

2. **Specify candidate estimands.** At least: (a) **sequential/greedy valuation** — value of
   an action given the edits already committed, generating labels by rollout over the frozen
   population; (b) **batch/set valuation** — value of a conflict-disjoint edit set, scored
   once through the frozen scorer. For each, state exactly what is labeled, how labels are
   generated, and what the selector consumes at deployment.

3. **Prove tractability on frozen data before proposing anything.** The frozen population is
   `1,345,497` delete actions across `40,894` components (max size `6`) and `45,961` occupied
   swaps / `21` assignments. Sequential rollout labels are expensive; measure the cost on the
   real population rather than estimating it. Reuse the frozen N1 compiler
   (`457f6607…24f5`) — the repair-4 primitive-union work already proved a greedy
   conflict-disjoint set can be compiled as one equivalent primitive union with an identical
   final graph hash. That is directly reusable machinery here.

4. **Apply the marginal-value gate before requesting any compute.** State the realistic
   ceiling. The honest anchor is the `24.49%` collapse ratio: if context-aware valuation
   recovers even part of the `+0.075828 → +0.018571397` gap, quantify what "part" must mean to
   clear `+0.005`. If it cannot clear the floor even optimistically, **STOP and record that** —
   that is the gate working, and it costs nothing.

5. **Preregister, hash-pinned, pre-outcome**, if and only if step 4 clears: frozen population,
   folds, label-generation procedure, selector class, kill line, budget, and abort rules.
   Selection stays on the held-fit frozen-scorer graph delta. Then take it to the owner for
   review. **Do not launch anything off the back of this node.**

## Do not

- Do not re-screen any DEAD mechanism (`LIVE_STATE.md` DEAD list / DAG §6.1).
- Do not touch the frozen scorer, the N1 compiler, the sealed B0 (`a9ce8c0b…b4ff2`), the
  population manifest (`4ab7bb23…76f0c`), or any existing preregistration.
- Do not propose a new encoder. Clause 1 is about the estimand; clause 2 needs *independent*
  AUC `>=0.85` evidence, which no candidate has.
- Do not open `6bba` or CV57.
- Do not treat this node as authorization for anything. It produces a diagnosis and, at most,
  a preregistration awaiting human review.

## Done when

`REFORMULATION_DIAGNOSIS.md` exists with real numbers; candidate estimands are specified
precisely enough to implement; tractability is **measured** on the frozen population; the
marginal-value gate is written out with a verdict; and either a hash-pinned preregistration
is ready for owner review or a recorded STOP explains why the direction does not clear the
floor. Append the dated line to `LIVE_STATE.md` either way.
