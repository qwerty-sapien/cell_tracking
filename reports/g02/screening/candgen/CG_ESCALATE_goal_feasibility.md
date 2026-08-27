# CG-ESCALATE — Goal-feasibility escalation

> Load `DAG.md`, `../../../.cruise/goal/GOAL.md`, and `cg0_ceiling_diagnostic_v1`.
> Runs when CG0 finds no candidate-generation policy with oracle ≥ 0.92, or CG1 finds no
> plan mechanism. This node does not build; it surfaces a locked-decision conflict.

## Why escalate rather than keep building

The G-02 completion condition is conjunctive and includes **local CV ≥ 0.92**. If CG0
shows the *perfect-selection oracle* over every reachable candidate-generation policy is
below 0.92, then no selector, ranker, or proposer over this detection front-end can reach
the goal — the ceiling, not the model, is the wall. Continuing to build against an
impossible ceiling burns budget and violates the seeking-alpha rule against funding
un-falsifiable-or-capped directions. GOAL.md's interruption conditions explicitly include
"a locked decision must change" and "the only credible improvement is forbidden."

## What to assemble for the goal owner

1. **The ceiling evidence.** CG0's per-family policy→oracle table and the 0.898→0.975
   exclusion decomposition; the maximum achievable oracle and why it is < 0.92.
2. **The locus of the wall.** Whether the ceiling is set by candidate generation alone or
   by the **detection front-end** (are correct nodes even present for the excluded edges?).
   If detection-bound, the previously-deprioritized detection front-end (`GD`) becomes the
   only lever that can move the ceiling — a reversal of the E1 verdict's *priority*, not
   its correctness, and a decision only the goal owner can authorize (it reopens a locked
   scope).
3. **The options, with honest costs.** (a) A stronger detection/segmentation front-end to
   lift the ceiling (reopens `GD`, likely GPU); (b) accept a sub-0.92 local ceiling and
   renegotiate the goal or its stopping condition; (c) a different problem formulation.
   For each: expected ceiling, compute, and gate implications.

## Output

`reports/g02/screening/candgen/cg_escalation_v1.md` + LEDGER row. **Stop and hand to the
goal owner**; do not self-authorize a scope change, a detection-front-end reopen, or any
GPU/RunPod/Kaggle spend. Preserve all evidence; name the exact locked decision that would
have to change and the minimum prerequisite to resume.
