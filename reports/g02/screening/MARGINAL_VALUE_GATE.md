# Marginal-value gate — a standing stop-loss doctrine

**Status: standing process rule, effective 2026-07-30. Applies to every branch, lane, and
node in the G-02 campaign, for both Claude and Codex.** This does not replace any
per-node preregistered threshold; it is a mandatory check that must run **before**
requesting additional compute (Kaggle GPU/CPU, a local guarded pilot, or a further
mechanism iteration) on a direction that already has one negative, marginal, or
partially-negative signal.

## Why this exists

The campaign lost real time to two failure patterns that a cheap upfront check would
have caught for free:

1. **Banking an unconstrained oracle as if it were realized headroom.** The legacy CG0
   swap oracle (`+0.075828`, fit117, no legality/degree constraint) was originally
   treated as `SWAP-O0`'s ceiling; the actual frozen, node-disjoint, legality-constrained
   population realized only `+0.018571397` — a `24.49%` realization, a `4.08×`
   collapse. The same shape of mistake recurred on 2026-07-30: `recall/N0`'s
   `+0.052004594313` pool-absent ceiling looked like real headroom, but tracing its
   construction (`recall/runs/N0B_RECONCILIATION.md`) showed it is bounded by the same
   kind of unconstrained-oracle-to-legal-action gap, and the actual bottleneck (action
   vocabulary, not candidate supply) makes the whole mechanism *class* — not just the one
   tested realization — a dead end.
2. **Iterating a mechanism after its ceiling was already known to be too small.** Each
   fresh pilot should ask "even in the best case, does this clear the floor?" *before*
   spending the compute, not after a disappointing result invites "one more tweak."

## The gate

Before requesting compute for a direction (a Kaggle kernel, a local guarded pilot, or a
new mechanism realization) that is not the *first* attempt in its lane, answer these in
writing, in the requesting node or a companion note, before the request:

1. **State the realistic (not best-case) ceiling**, using any directly applicable
   precedent collapse ratio from this campaign if one exists (e.g., the `24.49%`
   raw-to-realized swap ratio). If no precedent exists, say so explicitly rather than
   defaulting to the raw/optimistic number.
2. **Compare the realistic ceiling to the actionability floor** (`+0.005` adjusted edge
   Jaccard per family, or the relevant task-specific floor, e.g. the division
   `>=4`-TP gate in `division_ckpt/N3_efficacy_gate.md`).
3. **If the realistic ceiling is at or below the floor, or the limiting resource is
   structural** (an action-vocabulary/compiler constraint, a locked dependency, a
   licensing/provenance gap) **rather than something the proposed mechanism can move,
   the verdict is a hard STOP** — `DEAD` or `PARK` as appropriate — regardless of how
   promising the raw/upstream number looks. Do not request the compute "to see."
4. **A STOP applies to the mechanism *class*, not only the tested instance**, whenever
   the limiting factor is shared across the class (as with recall/candidate-generation
   lanes above: any generator hits the same node-disjoint/both-ends-free wall,
   independent of generator quality).
5. **Escalate, don't iterate, when the only path past the wall requires touching a
   frozen/`GO` component** (e.g., the N1 feasibility compiler) or a locked dependency
   (e.g., `B1`'s association-continuity prerequisite). That is a new, separately gated
   mechanism proposal with its own novelty/crowding and feasibility review — never a
   quiet parameter change on the current request.

## What this does not do

- It does not relax any preregistered per-node threshold — those still bind exactly as
  written.
- It does not forbid a *first* attempt at a genuinely new mechanism; the gate applies to
  continued/incremental investment, not initial exploration.
- It does not authorize skipping a HUMAN-GATE. A direction that clears this gate still
  needs explicit owner approval for any Kaggle/GPU spend.

## Worked example (2026-07-30)

`recall/N2` measured `9/290 = 3.10%` recovery for `NDR-FBC-local-translation-v1` — a
correct per-mechanism `DEAD`. Before considering a second generator, this gate asks: is
the limiting factor the generator, or something structural? `recall/N0B_RECONCILIATION.md`
traced the actual action-population code and found the wall is the deployed action
vocabulary (both-ends-free `add`; node-disjoint single-hop `swap`), confirmed by the
campaign's own `SWAP-O0` precedent. Verdict: STOP at the class level. A better generator
would not have been worth funding, and this was determined for free, without a second
CPU pilot or Kaggle request.

## Where this is referenced

- `reports/g02/screening/LIVE_STATE.md` — authoritative artifacts list.
- `dag_prompts/2026-07-30-recall-and-division/README.md` — standing rules.
- Any future DAG's `README.md`/`DAG.md` standing-rules section should link here rather
  than restate it.
