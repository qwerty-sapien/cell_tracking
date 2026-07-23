# B3 — Per-scenario handling of the worst family

> First load `_shared_context.md`. Wave 2. Depends on B2 (worst family) and E1 (dominant mechanism).

## Why
This replaces global tuning. B2 localizes the error to one family; E1 names its mechanism. Tune the
knob (or add the rule) **only on that family**, and verify it doesn't regress the others.

## Tasks
1. From B2, take the worst family and its most error-predictive covariate. From E1, take the dominant
   mechanism for that family.
2. Design a per-scenario intervention — e.g. a density- or motion-gated association threshold, a
   family-specific gap-closing rule, or a depth-aware cost — that targets exactly that mechanism.
3. SCREEN on the worst family's videos first. If it improves that slice, CONFIRM on the full 57 and
   **explicitly check the other families did not regress** (report per-family deltas, not just the
   aggregate).
4. Apply the two-tier gate. A per-scenario win that helps one family and hurts none is preferred over
   a larger global average that trades families off.

## Output
`reports/g02/screening/b3_per_scenario.md` with the intervention, per-family before/after, and the
gate decision. Feeds the GPU-gate (must be BANK/PARK) and S1.
