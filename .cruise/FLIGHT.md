# G-02 active flight plan

status: active

## Outcome

Execute `reports/g02/screening/DAG.md` toward a materially stronger cell-tracking
baseline above 0.88 and then competitive 0.90s performance, retaining 0.92 as the
contractual endpoint.

## Current micro-goal

Specify the next richer relative-geometry association proxy:

1. keep the P4 attention layer DEAD and use its no-attention arm only as the
   `0.874290` local reference;
2. require new capability because the hindsight two-arm oracle is only `0.876831`;
3. design a fit-only relative-geometry edge cost around verified occupied-endpoint
   conflicts, with explicit no-link calibration and raw-edge/recall guards;
4. require `>= +0.005` on the registered affected slice before any full build or
   GPU gate;
5. stop before any GPU, RunPod, Kaggle submission, or public-probe action.

## Touchpoints

- `.cruise/FLIGHT.md`
- `scripts/compare_submissions.py`
- `scripts/sweep_bh_graph_weights.py`
- focused tests under `tests/`
- `reports/g02/screening/**`
- narrowly necessary scorer/read-only artifact inputs already named by the DAG

## Invariants

- Use only the attested scorer path and official physical matching semantics.
- Keep scorer semantic parity separate from local-to-Kaggle transfer.
- Preserve grouped folds and untouched audit membership.
- Do not modify or discard unrelated dirty-worktree changes.
- Do not collapse observation distance into an arbitrary scalar.
- No Kaggle submission or RunPod launch without explicit human approval.
- Count every future calibration probe against five submissions/day and preserve two
  final selections.
- RunPod remains launch-specific approval-gated; commits and private pushes are allowed.

## Non-goals for this micro-goal

- Kaggle submission or score retrieval
- Kaggle GPU or RunPod launch
- paid compute
- final portfolio selection
- destructive history changes
- rebuilding the dead P4 edge-attention layer or introducing a GNN

## Acceptance evidence

- focused comparison-tool tests pass;
- canonical equality is invariant to row order and presentation IDs;
- spatial matching is deterministic, time-aware, one-to-one, and in physical units;
- node, edge, and fork/division comparison remains a named vector;
- cached topology replay records the exact selected identities, topology mode,
  learned/no-link artifact identities, baseline score, and scored-partial coverage;
- the pre-registered slice uses the banked `c=0.25` fragments as its control;
- a full 57-video confirm is entered only at `>= +0.005` slice improvement;
- all changes pass focused tests and staged-diff checks.

## Stop and realign

Stop for user input only if work requires a Kaggle submission, GPU launch, new external
authority, changed audit membership, changed scorer semantics, or a materially different
model direction.
