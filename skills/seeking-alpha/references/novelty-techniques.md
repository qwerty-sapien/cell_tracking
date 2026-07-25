# Novelty techniques — the inquiry engine

Read this when populating an alpha node (Phase 4). The goal is to generate several
*candidate* non-obvious moves cheaply. Do not filter yet — grounding and falsification
happen in Phase 5. Breadth is cheap here; commitment is expensive.

Each technique below is a *generator*. Run several. For each candidate you keep, write
one line: the move, and the reason it might be defensible (feeds the crowding test).

## 1. Cross-disciplinary transfer (highest yield)

Import a mechanism, tool, invariant, or result from a *distant* field whose underlying
mathematical or physical structure matches yours. The alpha is exactly that the link
is non-obvious on the surface but apt underneath — which is why competitors in your
own field miss it.

How to run it deliberately:
- Strip your problem to its abstract structure ("a sparse assignment under a hard
  one-to-one constraint with noisy costs"; "a chain that must be 1-D and stable").
- Ask which *other* fields have a mature theory of that exact structure. Assignment →
  operations research, economics (auctions), ecology (mating). 1-D stable chains →
  organometallic chemistry, polymer physics, DNA nanotech.
- Borrow their sharpest tool or their known impossibility result.

Guard: transfer requires a reason the analogy holds *under your conditions*. Name the
structural correspondence, or derive it — otherwise it's decoration, not transfer.

## 2. Literature-edge mining

The defensible answer frequently lives where competitors don't look:
- **Old** papers (pre-digital, pre-current-hype). The metallic-chains answer sat in
  2–3 papers, one from the 1980s.
- **Obscure / low-citation** venues, theses, technical reports, patents.
- **Adjacent-field** literature that never uses your field's keywords.
- **Non-English** literature and primary sources behind popular secondary summaries.

Practical moves: follow citation chains *backward* to the origin of a method (the
original paper often states assumptions later work forgot); search the vocabulary of
the *adjacent* field, not your own; read the patent literature for engineering tricks
that never reached papers.

Why it's alpha: crowded approaches come from the crowded, recent, top-of-results
literature. An edge that requires reading a 40-year-old paper in another field is one
few will replicate.

## 3. First-principles re-derivation

Rebuild the approach from the governing theory (physics, information theory, the loss
landscape, the metric's own definition). The dominant approach is often an unexamined
convention that made sense under conditions that no longer hold. Re-deriving surfaces
where the convention is loose and where a better-grounded choice exists.

Ask: what does the *objective's own definition* reward? Optimizing a proxy that merely
correlates with it is a common, crowded mistake; optimizing the objective directly is
sometimes the whole edge.

## 4. Constraint inversion

List every assumption currently treated as fixed. For each, ask: *is it actually
fixed, or just conventionally fixed?* Alpha often comes from moving a constraint that
everyone else leaves alone (a fixed radius, a fixed threshold, a fixed representation,
a fixed data modality). Then check feasibility and grounding for the moved version.

## 5. Boundary / extreme-case reasoning

Push a parameter to its limit (zero, infinity, one, all). Extremes expose what becomes
*possible* (a term vanishes, a problem becomes convex) or *forbidden* (a hard wall).
The metallic-chains ligand choice fell out of an extreme: the *smallest* strong
π-acceptor maximizes the metal–metal bond, which points at CO.

## 6. Analogical reframing / representation change

Restate the problem in a different representation (graph ↔ matrix ↔ geometry ↔
sequence ↔ probability). A move that is invisible in one representation is obvious in
another. Ask what structure appears only after the reframe.

## 7. Inversion of the goal

Instead of "how do I achieve X", ask "how would I *guarantee failure* at X, and am I
accidentally doing any of it?" Failure-mode enumeration is a fast generator of
non-obvious protective or corrective moves.

## Using the generators together

A strong alpha node often stacks two: a *cross-disciplinary transfer* whose validity
you establish by *first-principles* derivation, or a *moved constraint* whose payoff
you confirm by *boundary* reasoning. Generate from at least two independent techniques
so you're not anchored on the first idea — then hand everything to the Phase-5 gates.
