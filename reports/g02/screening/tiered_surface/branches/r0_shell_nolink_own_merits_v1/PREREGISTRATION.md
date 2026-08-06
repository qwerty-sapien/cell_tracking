# R0 shell no-link own-merits screen — preregistration

Status: **LOCKED BEFORE MODEL SCORES OR SELECTED ACTIONS**
Date: 2026-08-01
Mechanism: `R0-SHELL-FREEFREE-OOF-APPEARANCE-NOLINK-B4-V1`

## Objective

Test whether a learned, appearance-grounded no-link score can identify an extremely
small, high-precision set of useful edges from the exact fit-only `10 < distance_um
<= 12`, cap-8 shell.  This lane is a precision-at-budget early falsifier.  The
marginal-value gate below proves that four perfect add-only edits cannot reach the
applicable held-fit graph floor, so no candidate graph is opened in any outcome.

This is not a re-screen of DEAD `R12SHELL`, `MGSHELL`, or `TEMPBG`: it uses a learned
deployment-background no-link model, raw-patch appearance evidence, incumbent-relative
appearance, nested video OOF scoring, and a fixed tiny accepted support.  It never
rescales, reranks, or replaces the incumbent cost over the ordinary candidate pool.

## Frozen population and information boundary

- Family and slice: the 36 fit-train105 `44b6` videos only.
- Candidate eligibility is fixed before learned scoring: exact shell pairs whose
  source has no selected outgoing edge and whose target has no selected incoming edge
  (`topology_code == 0`).  This is the literal missed-link/no-link context and excludes
  the occupied shell background from learned rescoring.  Already-selected pairs are
  impossible in this context.  Division-truth status is not available to selection;
  if selected, it counts false at Gate P because this mechanism targets one-to-one
  continuation edges.
- Existing training rows: all shell continuation positives, all known negatives, and
  the already-frozen hard-plus-hash-uniform deployment-background sample in the banked
  appearance examples.
- Held video labels are unavailable to its model, calibration, ranking, support
  allocation, node-disjoint suppression, or tie-breaking.  Labels are opened only to
  compute precision after the four actions are irrevocably selected.
- No `6bba`, validation12, CV57, Kaggle, GPU, RunPod, or submission access.

Pinned aggregate inputs:

- shell prepare report SHA-256
  `cb882835d9b33bcad016684f988ea81e2fff3713432cbde0dadc9b6879cd6386`;
- appearance report SHA-256
  `8183f347bd89f3405dfdb7e14e564f2b285ef624ab6bd6f5e12d7460b43f7aa8`;
- reachability report SHA-256
  `22a70e0b750d3d59bdd92ca5a82967d35c5d866fafe1c867dbb09b936efc2803`;
- selected-graph conflict prepare SHA-256
  `3965145141c8b58a6b844ed9f9bcee651589fde03785aec6767175dfc1612f1e`;
- source commit `c493a676bfc415aa4f221eb51469c410307238c9`.

## Frozen model and nested scoring

Five outer video folds are assigned by
`int(sha256(identity)[:8], 16) % 5`.  For outer fold `k`, fold `(k+1) % 5` is
calibration-only and the remaining three folds fit one model.  The held outer fold is
never used for fitting or calibration.

The one model is standardized, class-balanced ridge least squares with an unpenalized
intercept, standard-deviation floor `1e-6`, and `lambda=0.01`.  Weighted mean and
variance use the same class-balanced row weights as the normal equations.  There is no feature,
lambda, seed, fold, or architecture search.  Positive rows receive total weight `0.5`;
known-negative plus sampled deployment-background rows receive total weight `0.5`.
Unknown deployment rows are a contamination-tolerant no-link/background proxy, not
asserted GT negatives.

Exactly these three features are used:

1. `pair_ncc`;
2. `source_no_link_probability`;
3. `distance_um`.

For comparability across outer models, each held score is converted to a one-sided
empirical tail probability against every label-blind topology-0 shell row in its
calibration fold: `(1 + count(calibration_score >= held_score)) / (1 + n_calibration)`.
Lower tail probability is better; ties use higher raw score, identity, source id,
target id.

## Fixed accepted budget and local action contract

The accepted budget is exactly **four actions across all 36 held 44b6 videos**.  Greedy
selection follows the frozen score order and accepts a candidate only if both
challenger endpoints are disjoint from all previously accepted actions.  The first
four eligible actions are irrevocable.  Fewer than four is an automatic precision-gate
fail.

Each action inserts one edge between two incumbent-free endpoints.  It removes no edge
and adds no node.  There are no deletions, swaps, global solve, graph-cost change,
occupied-pool rescoring, or post-outcome threshold.  This deliberately replaces less
of the incumbent than the banked local-swap winner.

## Marginal-value gate

The banked identical-population APNCC control has `edge_tp=10327`, `edge_fp=958`, and
`edge_fn=687`, hence edge denominator `D=11972`.  A perfect add-only correction changes
`TP += 1` and `FN -= 1`, leaving `D`, predicted nodes, and node recall fixed.  Therefore
four perfect edits can improve raw edge Jaccard by at most
`4 / 11972 = 0.0003341129`.

The official per-video adjustment factor is
`1 - 0.1 * (num_pred_nodes - estimated_nodes) / estimated_nodes`.  Since predicted
node count is nonnegative, this factor is at most `1.1`; the adjusted improvement is
therefore bounded by `4 * 1.1 / 11972 = 0.0003675242`.  Both ceilings are strictly
below `+0.005`.  Node recall cannot change because no nodes are added or removed.

Consequently, the graph branch is preregistered **closed regardless of precision**.
The exact executed candidate is a no-op with applied edit set size zero and delta
vector `adjusted/raw/node = 0/0/0`.  The four sealed actions and their precision remain
diagnostic evidence about whether the narrow learned no-link idea has own-merits.

## Preregistered precision gate

### Gate P — precision at fixed budget

After the four actions are sealed, official fit-only matching labels each challenger.
Every result other than a non-division continuation truth edge counts false.

- **SUPPORTS_NARROW_EVIDENCE:** precision `== 1.00`, i.e. exactly `4/4` correct.  This
  does not authorize graph execution or promotion.
- **KILL_EARLY:** precision `< 1.00`, fewer than four actions, any selection leakage,
  nondeterminism, or action-contract violation.

In both cases the graph stays unopened and the exact no-op delta vector is reported.
The named B4 mechanism cannot promote at 44b6 by construction.  `6bba` and CV57 remain
closed in all outcomes.

## Four-property audit

- appearance-grounded: raw 3-D patch NCC is required by the three-feature contract;
- narrow: pre-existing free/free no-link context within the 10–12 um shell, fixed
  four-action support;
- node-disjoint: enforced over challenger endpoints before labels open;
- recall-preserving: actions add edges and remove none; the node set is unchanged, and
  graph promotion additionally requires nonnegative raw edge Jaccard and node recall.
  If Gate P fails, the candidate abstains completely and is an exact no-op.

No outcome may amend this preregistration.  A changed budget, feature list, model,
folding, calibration, eligibility rule, or compiler behavior is a new named mechanism.
