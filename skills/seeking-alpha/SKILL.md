---
name: seeking-alpha
description: >-
  Use this to attack hard, open-ended problems where the goal is a genuine,
  defensible edge — not the first approach everyone else will also try. Trigger it
  whenever the user wants to WIN or DIFFERENTIATE on a competition, research
  direction, grant, or design problem; asks "how do we get an edge / unique
  advantage / alpha", "what will everyone else miss", or "is this just what
  everyone does?"; wants a complex problem decomposed into a rigorous, deterministic
  DAG of subtasks; wants novel but literature-grounded and locally-falsifiable
  ideas; or is about to commit to an obvious strong tool/approach and should first
  check whether it's actually an edge. Use it even when the user doesn't say the
  word "alpha" — any time the real need is a novel, feasible, well-substantiated,
  highly pertinent solution tree rather than a competent-but-crowded one.
---

# Seeking Alpha

## What this is for

**Alpha** is a defensible edge: a move that materially advances the objective *and*
that most equally-competent competitors will not easily find or replicate. Its
opposite is **beta** — the competent, obvious approach (the strongest well-known
tool, the first idea everyone lists). Beta is often necessary, but it is never an
edge: if a thousand capable teams would do it, it cannot be what separates you.

Most failure on hard problems is not a failure of effort or of tooling — it is
spending the whole budget on beta. This skill is a discipline for (a) decomposing a
problem *deterministically* so the search is reproducible and the answer defensible,
and (b) forcing at least one node of that decomposition to carry real, grounded,
falsifiable novelty.

The spirit is inquiry and curiosity held in tension with rigor: reach widely for
non-obvious ideas, then refuse to keep any that you cannot ground in the literature
*and* cheaply try to kill. Novelty without grounding is theater; grounding without
novelty is consensus.

## The loop

Work these six phases with the user. They are a DAG, not a straitjacket — revisit
earlier phases when a later one teaches you something. Ask sharp questions one at a
time when framing; do not bury the user in a questionnaire.

### 1. Define the objective deterministically

Before hunting for solutions, pin down three things precisely:

- **The equation to optimize.** State the exact quantity that decides winning, in a
  form *you can compute yourself*. "Better tracking" is not an objective; "adjusted
  edge Jaccard on the frozen 57-video scorer" is. The point of insisting on a
  computable objective is that it later lets you falsify ideas locally, for free,
  before spending the expensive resource.
- **The hard disqualifiers.** The constraints most candidates must *fail*. These do
  the heavy pruning. In the metallic-chains problem the disqualifier "the chain must
  be one-dimensional and linear — no crosslinking" eliminated almost every candidate
  compound in one stroke.
- **Operationalize any fuzzy criterion into a sieve.** If the target is vague
  ("covalency", "robustness", "quality", "novelty"), do not leave it to judgment —
  decompose it into a small set of *near-orthogonal, individually measurable*
  criteria and apply them as a sieve. This is what converts a hand-wavy claim into a
  deterministic, defensible verdict. (Metallic chains: covalency was made precise via
  a five-criterion sieve, mostly orthogonal.) Prefer orthogonality so the criteria
  don't secretly re-measure the same thing.

The deliverable of this phase is a one-paragraph objective + a disqualifier list + (if
needed) the sieve. If you cannot write the objective as something computable, that is
the first subtask, not a detail to skip.

### 2. Decompose into a DAG of subtasks

Break the problem into subtasks, each with **explicit inputs, an explicit output, and
a local validation** (how you'll know that node is done and correct). Draw the
dependencies; identify the critical path. Determinism here means every node has a
checkable done-condition, so you can verify the *plan* before investing in it.

Most nodes will be routine. That is fine and expected — do not manufacture novelty
everywhere. The value of the decomposition is that it isolates *where* an edge could
live.

### 3. Locate the alpha node(s)

Find the one or few subtasks where a non-obvious move yields disproportionate,
defensible edge. Usually it is one of:

- the **operationalization** of the fuzzy criterion (a sharper sieve than others use),
- the **search** step where everyone else stops at the obvious candidate, or
- a **constraint** everyone else treats as fixed but that can be moved.

Apply the crowding test (below) to each candidate node. **Force at least one node to
carry genuine novelty.** If no node does, the whole tree is beta and will be
replicated by every competent competitor — say so plainly and keep searching.

### 4. Generate novelty — the inquiry engine

Populate the alpha node(s) with *candidate* moves (don't commit yet). Use structured
creativity rather than waiting for inspiration. The full catalog with examples is in
`references/novelty-techniques.md`; the high-yield ones:

- **Cross-disciplinary transfer** — import a mechanism, tool, or result from a distant
  field whose *underlying structure* matches yours. The alpha is precisely that the
  link is non-obvious on the surface but apt underneath.
- **Literature-edge mining** — the defensible answer often lives in old, obscure,
  cross-field, non-English, or patent literature, not the top of this year's results.
  (The metallic-chains answer sat in 2–3 papers, one from the 1980s.)
- **First-principles re-derivation** — rebuild from the governing theory; the obvious
  approach is often an unexamined convention.
- **Constraint inversion** — ask which "fixed" assumption is actually movable.
- **Boundary / extreme-case reasoning** — push a parameter to its limit and see what
  becomes possible or forbidden.

Generate several candidates. Breadth here is cheap; commitment is expensive.

### 5. Ground and validate each candidate — the rigor gate

Every candidate must pass **two independent gates** before it earns resources. See
`references/rigor-and-validation.md` for how to run each well.

- **Evidentiality.** Substantiated by real literature — prefer primary sources, and
  actively value obscure/old/cross-field ones. A citation you cannot actually produce
  is a red flag, not a rounding error. Cross-disciplinary claims need a source (or a
  first-principles derivation) for *why* the transfer holds under your conditions.
- **Local falsifiability.** A cheap, self-contained test against the Phase-1 equation
  that *could kill the idea* before you spend the expensive resource. "Rigorous" here
  means concrete: you compute the objective yourself, locally, and watch it move. If a
  candidate cannot be cheaply falsified, your first job is to *make* it falsifiable —
  reframe it as a hypothesis with a preregistered local test — not to fund it on faith.

A candidate that is grounded but not locally falsifiable is a research bet, not a
plan; label it so. A candidate that is falsifiable but ungrounded is a lucky guess;
find the mechanism or drop it.

### 6. Score, assemble, preregister

Rank surviving candidates by **pertinence × feasibility × defensibility**:

- *Pertinence* — does it actually move the Phase-1 objective, and by enough to matter?
- *Feasibility* — within the real compute/time/data/skill constraints?
- *Defensibility* — does it pass the crowding test?

Assemble the survivors into the DAG from Phase 2. **Preregister the validation**: the
exact local test, the threshold, and the kill criterion — written *before* you run it,
so the verdict is honest and not retrofitted to whatever happened.

## The crowding test (the heart of alpha)

For any move, ask: **"If a thousand competent teams attacked this, how many would
arrive at this exact move?"**

- Many → it's **beta**. Maybe necessary, but not your edge. Don't mistake *strong* for
  *differentiating*: adopting the obvious best-known pretrained model is beta by
  construction — it's the first thing everyone identifies.
- Few → ask *why*. Alpha requires a nameable reason others won't easily replicate it:
  a hard-won operationalization, a non-obvious cross-disciplinary link, obscure
  literature, a moved constraint, or a genuinely creative synthesis. **If you cannot
  name the reason your edge is defensible, you have not found alpha yet.**

## Anti-patterns

- **Consensus-chasing.** Reaching for the obvious strong tool because it's strong.
  Strong ≠ edge. Interrogate the crowded default before adopting it.
- **Novelty theater.** Novelty for its own sake — ungrounded, unfalsifiable, or
  irrelevant to the objective. Reach wide, then cut hard.
- **Fuzzy objective.** Optimizing a target you never made computable. It renders every
  downstream judgment non-deterministic and every claim indefensible.
- **Skipping the free test.** Spending the expensive resource before running the cheap
  local falsification you could have run first.
- **One-big-leap thinking.** Waiting for a single genius idea. Alpha in practice is
  *correct deterministic decomposition* + *one well-placed, grounded, falsifiable
  novel node* — not a miracle.

## Reference files

- `references/novelty-techniques.md` — the full idea-generation catalog with worked
  transfers; read when populating an alpha node (Phase 4).
- `references/rigor-and-validation.md` — how to build the optimizable objective, the
  orthogonal sieve, the cheap local falsification, and preregistration (Phases 1, 5, 6).
- `references/worked-examples.md` — the metallic-chains problem decomposed through all
  six phases, plus a research-direction example; read to see the whole loop end-to-end.
