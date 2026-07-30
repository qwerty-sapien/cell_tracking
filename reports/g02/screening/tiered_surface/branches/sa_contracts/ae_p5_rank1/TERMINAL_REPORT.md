# AE-P5 C0/C2-KBEST-STABILITY — rank-1 emission

**Verdict: `STRUCTURALLY_INERT_ON_FROZEN_44B6` — rank-1 output is byte-semantically
identical to APNCC. No graph run was executed, and none is needed to know the
44b6 adjusted delta is exactly `0`, below the `+0.005` floor.**

Scope: fit-only 44b6, held labels never read, no graph run, no 6bba, no CV57.
Implemented under the human implementation authorization of 2026-07-30; the
contract's `graph_run_authorized: false` is unchanged and was not exercised.

## Result

| quantity | value |
|---|---:|
| frozen shards | 36 |
| occupied-conflict actions | 45,961 |
| conflict components | 44,161 |
| components applying rank-1 | **0** |
| components falling back to APNCC | **44,161** |
| actions selected | **0** |

Result `4fc8eb98b09ae9fe8bfae376edf9c7dab8f0a32442729c109b5b091856b8c2be`.
Focused tests pass 20/20.

### Fallback reasons

| reason | components |
|---|---:|
| fewer than eight feasible assignments | 44,144 |
| remove-membership above 1/8 | 12 |
| introduce-membership below 7/8 | 5 |

## Why the rule cannot act

The C2 fixed rule requires the **eight** best feasible assignments per component.
The frozen occupied-conflict population does not have components large enough to
supply them.

| component size | count | feasible assignments |
|---:|---:|---:|
| 1 | 42,563 | 2 |
| 2 | 1,425 | 3 |
| 3 | 147 | 4–5 |
| 4 | 23 | 5–8 |
| 5 | 3 | 8–13 |

96.4% of components are singletons, which admit exactly two feasible assignments:
the explicit no-op, or the single action. Feasible-assignment counts on a conflict
path are Fibonacci-like — `2, 3, 5, 8, 13` for sizes 1–5 — so a component needs at
least **four** chained actions before eight assignments exist at all. Only 17
components in the entire population reach eight, and all 17 then fail the
`7/8` introduce / `1/8` remove membership conjunction.

The rule was specified for a combinatorial regime this population does not exhibit.

## Robustness to the ownership interpretation

The contract defines feasibility as "pairwise-disjoint N1 ownership" without
fixing which slots an occupied swap owns. Both readings were computed:

| ownership model | max component | components with ≥8 feasible | components applying rank-1 |
|---|---:|---:|---:|
| target incoming only (looser) | 3 | **0** | **0** |
| source outgoing + target incoming (stricter) | 5 | 17 | **0** |

Zero components apply under either reading, so the finding does not rest on the
ambiguous modelling choice. The looser model is in fact *more* decisive: no
component anywhere reaches eight feasible assignments.

## Consequence

AE-P5's rank-1 output equals the APNCC control on every component, so its 44b6
adjusted, raw, and node deltas are exactly `0 / 0 / 0` by construction. It cannot
clear the `+0.005` adjusted floor. This is a property of the mechanism against
this population, not a measurement, so it needs no graph budget.

This does **not** falsify the broader C2/appearance family, and no tier move is
proposed here. A differently named realizer — one whose feasibility structure
produces larger conflict components, or a stability rule with `k` matched to the
observed component sizes — remains eligible.

## Not done

No graph run, 6bba, CV57, tier move, DEAD promotion, GPU, Kaggle launch, budget
debit, RunPod, or submission.
