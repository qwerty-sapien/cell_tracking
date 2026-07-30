# NODETOPO N2 result — synthetic event-local falsifier

**Verdict: PARK.** The proposed test is crisp but no isolated neighborhood-signature
fixture or extraction interface exists; creating one is a new implementation request.

## Preregistered test

| Item | Contract |
|---|---|
| Fixture | Two frames with two equally plausible associations, declared neighbor sets, and a controlled collision/division counterexample. |
| Positive criterion | The continuity candidate ranks strictly higher only when its neighbor-set similarity exceeds the alternative by a preregistered margin. |
| Null | Permute neighbor labels while retaining geometry and candidate edges; the score gap must disappear. |
| Kill line | A score gap under permutation, or failure to rank the conserved pair, kills/parks the mechanism. |
| Scope | Standard-library/synthetic objects only; no raw universe, videos, campaign archive, graph score, or model. |

## Feasibility finding

`src/celltrack_alpha/edge_data.py` preserves and hashes sparse candidate topology, and
the action compiler has deterministic synthetic graph fixtures. Neither exposes a local
neighbor-signature extractor or a fixture that separates a conservation signal from
ordinary topology. The normal test environment is also unavailable (stale project
`pytest`; bundled runtime cannot import the eager `scipy` pipeline), so the existing
fixtures cannot be run in isolation today.

The next honest step is a small, separately reviewed fixture/interface implementation;
a Kaggle protocol would be premature.

## Proposed LEDGER row (not written)

`NODETOPO-NEIGHBOR-CONSERVATION | PARK | no isolated extractor/fixture | no compute request`
