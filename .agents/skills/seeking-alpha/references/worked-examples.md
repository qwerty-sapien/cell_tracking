# Worked examples

Two end-to-end runs of the six-phase loop. The first is a clean, self-contained
chemistry problem; the second is a messy, ongoing research campaign. Read them to see
how the abstract phases feel in practice — especially how one well-placed novel node,
not a miracle, carries the edge.

## Example A — Metallic chains (competition problem)

**Prompt:** find the compound that forms the longest run of consecutive
metal–metal covalent bonds.

**Phase 1 — objective, deterministic.**
- *Objective:* maximize the number of consecutive M–M **covalent** bonds in a single
  chain.
- *Fuzzy criterion → sieve:* "covalent" is vague, so covalency was operationalized as a
  **five-criterion sieve** (mostly orthogonal; two criteria correlated, and that was
  stated openly rather than hidden). This turned "is this bond covalent?" from a
  judgment call into a deterministic verdict — and the sieve is itself much of the
  edge, because it is sharper than a hand-wave.
- *Hard disqualifier:* the chain must be **one-dimensional and linear — no
  crosslinking.** This single constraint eliminates the vast majority of extended-solid
  candidates in one stroke.

**Phase 2 — DAG.** (a) define covalency test → (b) enumerate compound classes with M–M
bonding → (c) apply the 1-D linear disqualifier → (d) within survivors, maximize chain
length and stability → (e) justify the winner from theory → (f) confirm it is real
(synthesized / characterized in the literature).

**Phase 3 — alpha nodes.** Two: node (a), the covalency sieve (sharper
operationalization than competitors will bother with), and node (b)+(c), the *search*
for a genuinely 1-D linear M–M chain — where the crowd stops at famous but crosslinked
or cluster compounds.

**Phase 4 — novelty generation.**
- *Boundary reasoning:* to maximize M–M bond strength, the ligand should be the
  *smallest* strong π-acceptor (push "ligand size" to its minimum) → points at CO.
- *First principles:* filled metal t₂g → empty CO π* back-donation strengthens the M–M
  bond; a staggered conformation minimizes steric repulsion so monomers connect *only*
  through M–M bonds — exactly the linear-chain requirement.
- *Literature-edge mining:* a true 1-D linear ruthenium carbonyl chain, {Ru(CO)₄}ₙ,
  appears in only 2–3 papers, one from the 1980s — precisely the obscure/old corner
  the crowd doesn't reach.

**Phase 5 — grounding + falsification.** Each candidate chain is run through the
covalency sieve (local, deterministic test) and checked against the 1-D disqualifier;
{Ru(CO)₄}ₙ is grounded in the primary organometallic literature and in MO theory (the
back-bonding argument is derivable, not asserted).

**Phase 6 — assemble.** {Ru(CO)₄}ₙ survives every gate; the write-up preregisters the
covalency criteria so the verdict is defensible rather than retrofitted.

**Why it's alpha, by the crowding test:** most teams would list famous metal clusters
or 2-D/3-D metal frameworks (beta). The winning move required (i) a hard-won
operationalization of covalency and (ii) reading obscure, decades-old literature for a
genuinely linear chain — few competitors replicate either. The edge is *nameable*.

## Example B — a research-direction campaign (messy, ongoing)

**Prompt:** improve a cell-tracking pipeline's score enough to place well; a large
budget of experiments has produced mostly flat results.

**Phase 1 — objective, deterministic.** The winning quantity is a specific frozen
graph-level scorer output. Crucial move: an intermediate "proposal precision" metric
was being optimized as a proxy — and was *empirically shown not to track* the true
graph objective (a candidate passed the proxy at 0.965 precision yet moved the graph
score −0.0006). Lesson: the objective must be the thing that decides winning, and
proxies are untrusted until proven to correlate.

**Phase 3 — alpha node.** The crowding test flags the obvious lever — "use the strong
pretrained tracking foundation model" — as **beta**: it is the first thing thousands of
teams identify, so even if it works it is not an edge. The alpha node is elsewhere: the
*operationalization* of where the residual error actually is (a diagnostic proved it
was association-bound, concentrated, and gated by cross-dataset generalization), which
points at a specific, less-crowded sub-problem.

**Phase 4–5 — generation + rigor.** Candidates (appearance-based association; learned
background abstention on an extended-radius shell; cross-dataset domain adaptation) are
each attached to a *cheap local graph-level falsification on the hardest data family
first* — so a candidate dies on the hard slice before consuming the full budget. Ideas
that only ever passed proposal-level proxies are downgraded to research bets, not
plans.

**Phase 6 — preregister + prune.** Effect sizes and kill lines are written before each
run; directions that failed the same way repeatedly (with a proven mechanism of
failure) are marked **dead** so the search stops re-drifting into them.

**Transferable lessons from B:**
- The strongest, most obvious tool is usually beta — interrogate it before adopting it.
- If you are optimizing a proxy, *prove* it tracks the objective, or you may be
  climbing the wrong hill with great discipline.
- One correct deterministic decomposition (where is the error *actually*?) plus one
  well-placed grounded move beats a large pile of undirected experiments.
- Preregistration + honest "dead" verdicts are what convert a sprawling campaign into a
  *narrowing* one.
