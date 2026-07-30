# Recall N0B — reconciling the pool-absence ceiling against tensor reachability

**Disposition: static code/evidence audit. No execution, no data mount, no GPU, no
Kaggle asset, no CPU pilot.** This resolves the apparent conflict between `N0`'s
`+0.052004594313` held 44b6 pool-absent ceiling and the banked fit117 result that the
`10 µm`/cap-8 candidate tensor reaches **99.04%** (`73,999/74,718`) of clean one-to-one
GT edges. The two numbers are not contradictory; they measure different objects, and the
reconciliation changes the read on `N0`'s ceiling materially.

## The apparent conflict

- `extended_radius_reachability_v1.md`: on fit117, only `719/74,718` (0.96%) of clean
  one-to-one GT edges are unreachable at `10 µm`/cap-8, and `718/719` of those fail on
  distance, not neighbor-cap crowding.
- `recall/runs/N0_POOL_RECALL_CEILING.md`: on held 44b6, `626/673` (93.02%) of edge FNs
  are absent from the **frozen occupied-swap action population**, with `534/626` under
  `7 µm` and only `8/626` above `12 µm` — i.e. overwhelmingly *short-range*.

A set that is 93% "absent" but overwhelmingly short-range cannot be radius-excluded from
a 10 µm tensor. Something other than geometric supply is filtering these edges out.

## What "pool-absent" actually measures

`N0` defines pool-absent as: not a member of the pinned **occupied-swap action
population** (`4ab7bb23…76f0f`). Tracing that population's construction
(`scripts/analyze_action_class_oracle.py`) shows it is not the raw candidate tensor
itself — it is a **legality-filtered subset** of it:

```python
# free-add: BOTH endpoints must have zero baseline degree
free_additions = {
    pair for pair in true_current
    if pair not in baseline_edges
    and not outgoing.get(pair[0])
    and not incoming.get(pair[1])
}
# raw "swap" oracle (legacy CG0): ALL correct, reachable, missing edges,
# realized by unconditionally displacing whatever already occupies each endpoint
swap_additions = {pair for pair in true_current if pair not in baseline_edges}
```

`true_current` is exactly the r10/k8-reachable correct-edge set — the same object the
reachability study measured at ~99% coverage. So a held FN can be inside the tensor
(`true_current`) and still be absent from **both** deployed action classes:

- **free-add** requires *both* endpoints to be completely unclaimed by any existing
  edge. In a dense per-frame assignment almost every detection already has some
  (possibly wrong) incident edge, so this filter is nearly always the reason free-add's
  realized oracle is exactly `0` (`P0 action-class surface`, `LIVE_STATE.md`).
- **occupied-swap**, as actually deployed, is further restricted to a **node-disjoint**
  contract (`achievable_headroom_v1.json`: `"swap_contract": "node-disjoint under
  existing per-video cap"`) — no cascading or chained reassignment. A correct edge whose
  fix requires bumping a *chain* of other wrong edges (not a single isolated swap) is
  legally unreachable under this contract even though it sits inside the tensor.

## The campaign's own precedent for the exact same collapse

This is not a new phenomenon — the campaign already measured it once, for the same
action class, on a different (fit117) population:

| Quantity | Value | Source |
|---|---:|---|
| Raw/legacy CG0 "swap" oracle (unconstrained displacement) | `+0.075828` | `p0_action_class_oracle_v1.md:17`, cited as `SWAP-O0` legacy basis in `LIVE_STATE.md` |
| Realized, frozen, node-disjoint occupied-swap oracle (`run_v6`) | `+0.018571397` | `LIVE_STATE.md` (`6e62a29c…6c71b`) |
| Collapse ratio | `0.018571397 / 0.075828 = 24.49%` (a `4.08×` reduction) | derived |

`LIVE_STATE.md` states this correction plainly: "the run_v6 measured what the +0.005
target actually requires" — i.e. the raw CG0/tensor-level oracle overstates realized,
legality-constrained headroom by roughly `4×` for this exact action class.

## Reconciled reading

1. **The candidate tensor is not the bottleneck.** Geometric supply (radius/cap) is
   confirmed adequate (~99% coverage on fit117); `N0`'s own displacement histogram
   (93% of pool-absent FNs under 7 µm) is consistent with this, not contrary to it.
2. **The bottleneck is degree/conflict feasibility under the deployed, node-disjoint
   action contract**, not spatial absence. Most of the `626` pool-absent held FNs
   plausibly sit in local degree-conflict clusters — at least one endpoint already
   claimed by a wrong edge, and the correct fix requires either a merge-like admission
   (blocked: `free-add` needs both ends free) or a chained reassignment (blocked: `swap`
   is node-disjoint, single-hop only).
3. **`N0`'s `+0.052004594313` should not be read as achievable recall headroom.** It is
   the *same kind* of unconstrained-oracle number the campaign already corrected once
   (`+0.075828 → +0.018571397`, `24.49%` realized). Applying that precedent's ratio as a
   rough, non-binding sanity check — not a new committed estimate — puts a plausible
   realized ceiling near `+0.0127` (`0.052004594313 × 0.2449`). That is still above the
   `+0.005` floor in isolation, but it is a generous upper bound assuming the same
   collapse ratio transfers, it ignores FP cost entirely, and — critically — no
   candidate-*generation* mechanism can realize any of it: the limiting resource is the
   **action vocabulary** (node-disjoint single-hop swap plus both-ends-free add), not
   candidate availability. Feeding a perfect geometric oracle through the same two
   action classes would still hit the identical wall.
4. **This reframes `N2`'s `DEAD` verdict for `NDR-FBC-local-translation-v1`.** The
   `3.10%` recovery failure is not merely "this particular generator is weak" — the
   entire mechanism *class* (label-free candidate generation feeding the existing
   free-add/occupied-swap action space) is structurally capped by a wall that has
   nothing to do with candidate quality. A better generator does not move this number.

## What would actually be required — and why it is not proposed here

The only lever consistent with this diagnosis is a **new N1 compiler action kind**: a
conflict-disjoint *batch* of chained reassignments (delete-and-relink a short cascade of
wrong edges in one legal action set), rather than a single node-disjoint swap. This is
explicitly out of scope for a recall/candidate-supply lane — it is a change to the
frozen, `GO`-status N1 feasibility compiler itself (`tiered_surface/n1_feasibility_compiler`),
not a new supplier. Two reasons this is not being proposed as a next step:

- **It likely still falls short.** Even the *unconstrained* raw CG0 swap oracle
  (`+0.075828`, fit117, one family) is itself well under the `+0.044–0.061` local gap
  to public `0.910`; a compound-action compiler extension would recover *some* fraction
  between the realized `+0.018571` and the raw `+0.075828`, optimistically not the whole
  raw figure (chains have their own FP and correctness cost), and 44b6+6bba would need
  to be combined. Best case is informative, not sufficient.
- **It is a different mechanism class entirely** (compiler/action-DSL extension, not
  recall/supply) and would need its own N0 novelty/crowding gate, its own feasibility
  proof against the frozen N1 interface, and explicit owner authorization to touch a
  component the campaign has already frozen as `GO`. Proposing it casually here would be
  exactly the kind of un-gated scope creep the campaign's own discipline forbids.

## Verdict

**Recall-as-candidate-generation is DEAD as a mechanism class**, not only for the one
tested realization. The pool-absent set is real, but it is gated by action-vocabulary
legality (node-disjoint single-hop swap; both-ends-free add), not by candidate supply —
so no generator, however good, can move it. `N2`'s per-mechanism `DEAD` stands; this
adds a class-level verdict on top of it, grounded in the campaign's own prior SWAP-O0
correction rather than new speculation.

**Do not fund another candidate-generation mechanism in this lane without a new named
mechanism that changes the action vocabulary itself** (compound/chained conflict-swap),
and do not propose that compiler change without a fresh, separately gated novelty and
feasibility review — it touches a frozen, `GO`-status component and is not licensed by
today's instruction to reconcile these two results.

## Evidence consulted (all pre-existing; nothing newly executed)

- `dag_prompts/2026-07-30-recall-and-division/recall/runs/N0_POOL_RECALL_CEILING.md`
- `reports/g02/screening/extended_radius_reachability_v1.md`
- `scripts/analyze_action_class_oracle.py` (free-add / swap oracle construction)
- `reports/g02/screening/achievable_headroom_v1.json` (`current_pool_contract.swap_contract`)
- `reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.md` (raw swap oracle `0.075828`/`0.059735`)
- `reports/g02/screening/LIVE_STATE.md` (`SWAP-O0` legacy-CG0-to-`run_v6` correction)
- `reports/g02/screening/tiered_surface/n1_feasibility_compiler` (frozen `GO` interface, `ActionKind.SWAP` is single-action)

## Proposed LEDGER row — do not write

| ID | Mechanism | Gate | Result | Status | Note |
|---|---|---|---|---|---|
| `RECALL-N0B-RECONCILIATION-20260730` | static reconciliation of pool-absence vs tensor reachability | none (analysis only) | pool-absence is action-vocabulary-gated, not supply-gated; realized ceiling plausibly `~+0.0127` by precedent ratio, not `+0.052`; no generator can realize it | **DEAD (mechanism class: label-free candidate generation)** | Do not re-fund a generator without a new compiler action-kind proposal, separately gated. |
