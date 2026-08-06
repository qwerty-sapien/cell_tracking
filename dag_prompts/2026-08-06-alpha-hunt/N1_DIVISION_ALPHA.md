# N1 — Division alpha (the uncrowded lane)

> Load `.agents/skills/seeking-alpha/SKILL.md` + `references/novelty-techniques.md` +
> `references/rigor-and-validation.md`, `DAG.md`,
> `reports/g02/screening/public_solution_landscape_survey_v1.md`,
> `external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py`.
> Independent of N0 — pure local-CV57 work, no leaderboard needed.

## Objective / gate

`score = adj_edge_jaccard + 0.1 * division_jaccard`, frozen official scorer.

Our CV57 division term: **1 TP / 11 FP / 12 FN → J = 0.0417**, from **368 emitted forks**
against **13 GT divisions**. We bank `0.1 x 0.0417 = 0.0042` of a term worth `0.1`.
**~0.096 of headroom sits here** — the largest single unclaimed block in the objective.

Crowding verdict: **uncrowded**. No public notebook contains a learned division classifier;
the entire public tier runs one geometric heuristic (`add_safe_divisions_postlink`) and we
already run it at its exact tuned constants (4.66 / 8.5 / 7.65 / 0.0076 / 0.00375), so
constant-tuning is exhausted. Per the crowding test, the nameable reason this is defensible
is that the public frontier stopped at a geometric heuristic and the metric's division rules
are strict enough that geometry alone cannot satisfy them.

**Preregistered GO threshold:** at least one candidate passing BOTH rigor gates —
evidentiality (real primary source or first-principles derivation) AND cheap local
falsifiability (a CV57 test that could kill it before any GPU spend) — with a named
defensibility reason. Candidates that are grounded but unfalsifiable are labelled
research-bets, not plan.

## Tasks

1. From the scorer source, derive exactly WHICH rule kills our 368 forks (parent anchor /
   two-distinct-daughter bipartite matching / directed topology / cross-component branch
   evidence / unmerged branches). Quote the code.
2. Exploit the metric's asymmetry: *"unmatched, structurally valid forks with no local GT
   evidence are ignored."* Establish whether forks in unannotated regions are near-free
   while forks near annotated GT are costly — i.e. whether the metric rewards a
   precision-first or recall-first division strategy. This is the likely alpha.
3. Check the LIVE_STATE DEAD list before proposing anything.
4. Generate 4-6 candidates via cross-disciplinary transfer, literature-edge mining,
   first-principles re-derivation, constraint inversion, boundary reasoning.
5. Rank by pertinence x feasibility x defensibility; name the single alpha node.

## Output

`runs/N1_DIVISION_ALPHA.md` + a LEDGER row. Preregister each kill criterion before running.

## Verdict & cascade

- **GO** (>=1 candidate passes both gates) -> proceed to `N4_KERNEL_PACKAGE_SUBMIT.md`
  (HUMAN-GATE, halts).
- **PARK** (all candidates ungrounded or unfalsifiable) -> halt; report and keep searching.
