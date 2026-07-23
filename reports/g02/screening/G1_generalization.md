# G1 — Generalization / leave-one-family-out

> First load `_shared_context.md`. Wave 2. Depends on B2 (family structure). Feeds S1.

## Why
The rescoring punished the champion by 0.072 (0.985 → 0.913): the hidden set differs from what people
optimized against. Our 57-video local score could be similarly optimistic. Estimate the true gap to
the hidden set so final selection (S1) can hedge instead of picking the highest local score blindly.

## Tasks
1. Using the family structure from B2, run **leave-one-family-out**: for each family, score the banked
   config on that family after any tuning done without it. Report the spread.
2. Estimate the optimism gap: how much of the local 0.873 is family-specific overfit vs robust signal?
   Combine with P0's local→public fit to project a hidden-set band.
3. Flag any banked change from B1/B3 that helps in-sample but generalizes poorly across families —
   those are the ones to keep OUT of a "safe" final selection.
4. Recommend a robustness ranking of candidate configs: most-generalizing first.

## Output
`reports/g02/screening/g1_generalization.md` with the leave-one-family-out spread, the projected
hidden-set band, and a robustness ranking. This is the primary input to S1's "safe" final pick.
