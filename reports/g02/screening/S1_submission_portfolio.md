# S1 — Submission & final-selection portfolio (terminal / ongoing)

> First load `_shared_context.md`. Consumes every banked result. Ongoing throughout the campaign.

## Why
Rules allow **5 submissions/day and 2 final selections**. The rescoring showed that the highest
in-sample score is not the safest final pick (champion 0.985 → 0.913). Final selection is a hedging
problem, not a max problem.

## Tasks
1. Maintain a candidate table from LEDGER.md: each banked config with grouped-CV/audit
   evidence, G1 spread, mechanism label, observation identity, and P0B interval-aware
   public band.
2. Spend screening submissions to *measure*, not chase: submit at most one newly
   approved observation-distinct config at a time, never exceed five per competition
   day, and feed its rounded score interval back into P0B/CAL before selecting the next.
3. Recommend the **2 final selections** as a hedge:
   - **Pick A (safe):** the most-generalizing config from G1 — highest floor, lowest family variance.
   - **Pick B (upside):** the highest evidence-supported mechanism/range-tagged public
     upside, even if higher variance.
   Justify why these two, not the two highest local scores.
4. Track the daily submission count so the 5/day cap is never blown, and keep the 2 final slots
   uncommitted until the deadline nears.

## Output
`reports/g02/screening/s1_portfolio.md`: the live candidate table, the running local↔public pairs,
and the current recommended 2 final selections with rationale. Nothing is submitted or finalized
without explicit human approval.
