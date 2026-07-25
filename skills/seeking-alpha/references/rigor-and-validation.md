# Rigor and validation

Read this for Phases 1, 5, and 6. "Rigorous" is not a mood — it is a set of concrete
moves that make a claim computable, defensible, and killable before it costs you.

## Building the optimizable objective (Phase 1)

The objective must be something *you can compute yourself*, locally, cheaply. That
property is what makes everything downstream falsifiable.

Checklist:
- **One scalar (or a small vector) that decides winning.** Name it, and name the exact
  function that produces it from a candidate. If the true objective is hidden (a
  held-out leaderboard, a physical experiment), build the closest faithful local proxy
  *and* record how well the proxy tracks the true objective (rank correlation, residual
  after calibration). A proxy you haven't validated against reality is a liability.
- **Disqualifiers as boolean gates.** The hard constraints most candidates fail.
  Evaluate them *first* — they do the cheap pruning before any expensive scoring.
- **Units and direction.** Higher-is-better? What is the meaningful difference (the
  resolution below which two scores are indistinguishable)? Below that resolution, do
  not rank.

## The orthogonal sieve (operationalizing a fuzzy criterion)

When the target is fuzzy, replace judgment with a sieve of measurable criteria.

1. **Enumerate what the fuzzy word actually means** in this context. List candidate
   sub-criteria (for "covalency": electronegativity difference, orbital overlap,
   bond-length vs sum of covalent radii, computed bond order, spectroscopic
   signatures, …).
2. **Prune for near-orthogonality.** If two criteria measure the same underlying thing
   they double-count and give false confidence; keep a set that is *mostly*
   independent, and *note* the ones that remain correlated (in metallic chains, of five
   criteria, two were correlated — this was stated, not hidden).
3. **Make each criterion a computable test** with a threshold or a score.
4. **Define the combination rule** up front (all-must-pass sieve, weighted sum, or
   rank-then-cut) so the verdict is deterministic and reproducible.

The payoff: a claim like "this is the most covalent" becomes a defensible, checkable
result instead of an assertion — and *the sieve itself is often the alpha*, because a
sharper operationalization than competitors use is hard to replicate.

## The cheap local falsification (Phase 5)

Every candidate earns resources only after a test that *could kill it* for near-zero
cost. Design the test to be maximally *disconfirming*, not confirming.

- **Pick the cheapest slice that carries the signal.** The hardest sub-case, the worst
  family, the smallest faithful subset — not the full expensive run. If it dies on the
  hard slice, you saved the full budget.
- **Attach the test to the Phase-1 objective, not a proxy of a proxy.** A candidate can
  ace an intermediate metric and still move the true objective the wrong way; where you
  have evidence of that gap, test the objective directly and treat intermediate metrics
  as untrusted until shown to correlate.
- **Preregister sign, magnitude, and kill line.** Write "I expect ≥ +X on slice S; if
  ≤ 0, this is dead" *before* running. Retrofitting the threshold to the result is how
  campaigns fool themselves.
- **State the null.** What would the number look like if the idea does nothing?
  Reproducibility jitter, an identical-rerun envelope, a shuffled control. A result
  inside the null envelope is not evidence.

## Distinguishing the three verdicts

After Phase 5, every candidate is exactly one of:
- **Plan** — grounded *and* locally falsifiable *and* it survived the local test. Fund it.
- **Research bet** — grounded but not yet cheaply falsifiable. Keep, but label it; do
  not fund it as if it were a plan. Its first subtask is to *become* falsifiable.
- **Guess** — falsifiable but ungrounded (no mechanism, no literature). Either find the
  mechanism or drop it; do not let a lucky number masquerade as an edge.

## Preregistration (Phase 6)

Before the expensive run, write down: the exact objective, the exact local test and
slice, the threshold, the kill criterion, and the decision each outcome triggers. This
is not bureaucracy — it is what keeps the eventual claim honest, and it is what lets a
*negative* result genuinely prune the search instead of being explained away. A
campaign that never preregisters will re-drift into the same dead directions forever.

## Evidentiality standard

- Prefer **primary** sources; when you cite a mechanism, you should be able to produce
  the paper and the specific result, not a vague recollection.
- For a **cross-disciplinary** transfer, cite (or derive from first principles) *why*
  the borrowed result holds under *your* conditions — the transfer's validity is the
  claim that needs support, more than the borrowed result itself.
- Treat an un-producible citation as a **red flag** that the idea may be a confabulated
  guess wearing the costume of grounding.
