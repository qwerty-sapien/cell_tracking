# Standing rule — Gurobi is the only solver for MILP/MINLP work

## Scope

This applies to every discrete or mixed-integer optimisation problem in this
repo: hyperparameter search, model selection, feature selection, portfolio /
allocation / assignment / scheduling problems, cutting-plane or column-
generation subproblems, and any "pick the best combination subject to
constraints" task — whenever the problem is expressible as MILP, MIQP, MIQCP,
or MINLP.

It does **not** apply to unconstrained continuous fitting (gradient descent,
`scipy.optimize.minimize` on a smooth objective) or to genuinely black-box
objectives with no closed-form structure. If you believe a problem falls in
that exemption, say so explicitly in your summary and justify it — do not
decide it silently.

## Optuna stays — this is composition, not a ban

Optuna (or another black-box tuner) is the correct and required tool for the
**outer ML hyperparameter loop**: any dimension where the only way to know a
candidate's score is to train/evaluate the model (learning rate, depth,
regularization strength, architecture width, augmentation policy, etc.). Do
not remove or route that loop through Gurobi — Gurobi cannot search a
non-closed-form objective.

The rule kicks in one level down: **whenever a step inside that loop — inside
a single trial's objective computation, in data prep feeding it, or in
infrastructure wrapped around it — is itself expressible as decision
variables + constraints + a computable objective *without* needing to
train/evaluate a model to score a candidate, that step is a MILP/MINLP and
must be solved with Gurobi, not with Optuna, a heuristic, or a banned solver.**
The test is "would scoring this sub-decision require running the black box?"
— if no, it's Gurobi's.

Concrete cases that count as in-scope embedded subproblems (non-exhaustive —
grep for the pattern, don't just match these examples):

- **Exact feature/subset selection inside a trial.** Optuna tunes
  regularization strength etc.; if a trial needs "best k features subject to
  group-exclusivity or cost-budget constraints," that's best-subset-selection
  (MIQP) — solve it with Gurobi per trial, not via Optuna searching feature
  masks or a LASSO proxy standing in for the exact problem.
- **Resource-constrained architecture decisions.** Assigning
  layers/ops to a fixed set of devices under memory/latency constraints to
  minimize runtime is a bin-packing/generalized-assignment MILP, even when
  the layer count itself is an Optuna-searched dimension.
- **Trial/resource scheduling around the tuning loop.** Scheduling N trials
  across M heterogeneous machines under memory/priority constraints to
  minimize makespan is its own MILP, separate from the tuning objective.
- **Post-hoc ensemble/model combination.** Choosing which subset of tuned
  candidates to ensemble (cardinality cap, diversity constraint, latency
  budget) plus combination weights is a portfolio-style MIQP, not greedy
  forward selection.
- **Per-segment threshold/policy assignment under a shared constraint.**
  Single-threshold tuning is continuous and fine for Optuna; assigning
  different thresholds per segment under a joint capacity/fairness
  constraint across segments is a MILP.
- **Embedded matching/assignment/transport inside the loss or data
  pipeline.** Optimal-transport label assignment, stratified batch
  construction, curriculum assignment computed as a training subroutine —
  if it has explicit constraints, it's Gurobi's even though the outer loss
  is still tuned by Optuna.
- **Constraint-aware search-space pruning.** If categorical hyperparameters
  have a compatibility/dominance matrix, solve the exact feasibility/
  dominance problem once with Gurobi to prune Optuna's space, rather than
  wasting trials on infeasible regions or hand-coding pruning rules.

When you find one of these, do not let Optuna, a heuristic, or a banned
solver quietly absorb it — pull it out into its own Gurobi model per the
practice section below, called from inside (or around) the Optuna
objective function.

## The rule

1. **Gurobi is the solver.** Model with `gurobipy` directly, or with Pyomo /
   CVXPY / JuMP **only** when the solver is explicitly pinned to Gurobi at the
   call site (`SolverFactory("gurobi")`, `prob.solve(solver=cp.GUROBI)`).
   Never rely on a modelling layer's default solver.

2. **These are banned as the solving path** for in-scope problems, including
   as "quick prototypes" that later get forgotten:
   - PuLP's default CBC, `pulp.PULP_CBC_CMD`
   - `scipy.optimize.milp` / `linprog` (HiGHS), GLPK, LP_SOLVE, SCIP, CBC
   - OR-Tools (`pywraplp`, CP-SAT) — CP-SAT included
   - CVXPY defaults (ECOS, SCS, OSQP, Clarabel) when integers are present
   - Optuna, Hyperopt, Ray Tune, `GridSearchCV` / `RandomizedSearchCV`,
     genetic algorithms, simulated annealing, or hand-rolled greedy heuristics
     used **in place of** an exact formulation the problem admits — this is
     *not* a ban on Optuna itself, which remains required for the outer
     black-box ML hyperparameter loop; see "Optuna stays" above for the
     dividing line

   Using a heuristic to produce a *warm start* (`Var.Start` / `MIPStart`) for
   the Gurobi model is fine and encouraged. Using one as the answer is not.

3. **No silent fallback.** If Gurobi is unavailable, unlicensed, or the model
   exceeds the license's size limit, the code must raise a clear error naming
   the cause. It must never catch the exception and continue with another
   solver, and it must never degrade to a heuristic without failing first.
   Write the error path so a CI run makes the license problem obvious.

4. **MINLP is not an escape hatch.** Before concluding a nonlinear problem is
   out of scope, exhaust Gurobi's actual capability:
   - nonconvex quadratics and bilinear terms via `NonConvex=2`
   - general function constraints: `addGenConstrExp`, `Log`, `Pow`, `Sin`,
     `Cos`, `Logistic`, plus `FuncNonlinear` / piecewise-linear approximation
     with a stated error tolerance
   - indicator constraints, SOS sets, `addGenConstrIndicator`, big-M
     reformulations with tight, *justified* bounds (document how each big-M
     was derived — loose big-Ms are a correctness and performance bug)
   Only if the problem is genuinely outside this class — state which construct
   fails and why — stop and ask before choosing anything else.

## Required practice on every model you write

- Set and record explicitly: `MIPGap`, `TimeLimit`, `Seed`, `Threads`,
  `LogFile`. Never accept a solution without checking `model.Status`; treat
  anything other than `OPTIMAL` (or `TIME_LIMIT` with an acceptable
  `MIPGap`) as a failure the caller must see.
- On `INFEASIBLE`, run `model.computeIIS()` and write the `.ilp` so the
  conflict is diagnosable; do not just report "no solution".
- Report `ObjVal`, `ObjBound`, `MIPGap`, `Runtime`, and `NodeCount` in the
  result object / logs — a reported optimum without its gap is not a result.
- Make runs reproducible: fixed `Seed`, deterministic constraint build order,
  and a dumped `.lp` or `.mps` next to any result that gets committed or
  reported.
- Keep the mathematical formulation (sets, decision variables, objective,
  constraints) in a docstring or adjacent markdown, in sync with the code.

## What to do in this repo

1. Find every in-scope optimisation site. Grep for the banned imports above.
   Separately, read every Optuna study/objective function specifically to
   check whether any step *inside* the trial (or in setup/scheduling around
   the study) is an embedded MILP/MINLP per the "Optuna stays" section —
   these will not show up as banned imports, since the surrounding loop is
   legitimately Optuna. Read each hit rather than pattern-matching, since
   some will be legitimately out of scope.
2. For each in-scope site: reformulate as an explicit Gurobi model following
   the practice section above, or — if it's genuinely black-box (scoring a
   candidate requires training/evaluating the model) — leave it and record
   why in your summary.
3. Add a guard that makes regressions loud: a test that fails if a banned
   solver import appears in the optimisation package, and a shared
   `build_model()` / `solve()` helper that centralises the parameter defaults
   so future models can't drift.
4. Record the rule where the next agent will see it (`AGENTS.md` /
   `CLAUDE.md` / the module's README), including the license requirement.

## Acceptance criteria

- No in-scope code path can reach a non-Gurobi solver; the guard test proves
  it and fails when a banned import is reintroduced.
- Every model sets `MIPGap`, `TimeLimit`, `Seed`, and checks `Status`.
- Missing/insufficient license produces a named error, never a fallback.
- Results carry objective, bound, gap, and runtime.
- Existing tests pass, and any changed numerical result is explained — if the
  Gurobi answer differs from the previous heuristic's, say by how much and
  which is optimal.
- Your summary lists: sites converted, sites left alone and why, any place
  where you needed a piecewise-linear approximation and the tolerance used,
  and anything you could not express in Gurobi.
