# Tier-census schema — machine-readable fallback order

Status: **BINDING format.** The orchestrator maintains `tier_census.json` (single writer) and
renders a summary table into `LIVE_STATE.md`. Subagents emit `state_fragment.json` (same node
schema, their nodes only); they never write the census directly.

## `tier_census.json`

```json
{
  "schema_version": 1,
  "updated_utc": "2026-07-28T00:00:00Z",
  "plan_sha256": "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116",
  "objective": "S(G)=weighted_adjusted_edge_Jaccard + 0.1*division_Jaccard",
  "gate": "44b6->6bba->CV57; adj>=+0.005 both families; raw,node>=0",
  "kaggle_week": { "week_start_utc": "2026-07-27T00:00:00Z", "budget_h": 30.0, "used_h": 0.0,
                   "auto_approve_under_h": 5.0, "hard_kill_h": 5.0 },
  "rollup": { "LIVE": 0, "PARK_A": 13, "PARK_B": 23, "PARK_C": 8, "DEAD": 1 },
  "fallback_order": ["PARK_A", "PARK_B", "PARK_C"],
  "nodes": [
    {
      "id": "DELETE-O0",
      "branch": "association_expansion",
      "owner_agent": "SA-DELETE",
      "tier": "PARK_A",
      "banked": true,
      "evidence_artifact": "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
      "evidence_sha256": "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
      "evidence_basis": "P0 delete class 44b6 ΔS +0.088774 / 6bba +0.074840; retention/no-link framing",
      "oracle_headroom": { "family_44b6_dS": 0.088774, "family_6bba_dS": 0.074840, "control": "CG0_fit117" },
      "graph_result": null,
      "kill_line": "adj<=0 on 44b6 => PARK-C instantiation",
      "resume_condition": "N1 compiler frozen; then fit conservative retention beta",
      "blocked_on": ["N1_FEASIBILITY_COMPILER"],
      "priority": 1,
      "last_change_utc": "2026-07-28T00:45:00Z"
    }
  ]
}
```

### Field rules

- `tier` ∈ {LIVE, PARK_A, PARK_B, PARK_C, DEAD}. A DEAD row requires `human_authorized: true`
  and `named_mechanism`; the orchestrator rejects DEAD without both.
- `banked` MUST be true for any PARK_C or DEAD tier, with a resolvable
  `evidence_artifact`+`evidence_sha256`. Uncited => forced to PARK_B on merge.
- `graph_result` (when run): `{ "family": {"adj":x,"raw":y,"node":z}, ... }`; null until scored.
- `priority` is the numeric fallback key: PARK_A=1, PARK_B=2, PARK_C=3; ties broken by
  descending `oracle_headroom.family_44b6_dS`. This makes the fallback order deterministic and
  machine-consumable: "next thing to try" = lowest `priority`, highest 44b6 headroom, not
  `blocked_on` anything open.
- `quarantined` (optional bool): true for any node carrying a memo-sourced unbanked number;
  such a node is pinned PARK_B and excluded from fallback selection until independently banked.

## `LIVE_STATE.md` rendering (orchestrator writes at each barrier)

A census block + a fallback table (top of the promising queue), e.g.:

```
## Tier census (machine copy: tier_census.json @ <sha>)
LIVE 0 · PARK-A 13 · PARK-B 23 · PARK-C 8 · DEAD 1   | Kaggle 0.0/30.0 h this week

Fallback queue (unblocked, by priority then 44b6 headroom):
1. DELETE-O0   PARK-A  +0.0888  (blocked_on: N1)
2. SWAP-O0     PARK-A  +0.0758  (blocked_on: N1)
...
```
