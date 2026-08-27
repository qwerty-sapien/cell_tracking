#!/usr/bin/env python3
"""Build the human-decision division dossier without fitting or amending."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
SPEC = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_cont/"
    "continuity_target_spec.json"
)
REPLAY = HERE / "ORACLE_ONLY_REPLAY.json"
EVENT_OUTPUT = HERE / "DIVISION_REPAIR_BUNDLES.json"
DOSSIER = HERE / "PLAN_AMENDMENT_DOSSIER.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_text(edges: list[list[int]]) -> str:
    return ", ".join(f"{int(source)}→{int(target)}" for source, target in edges)


def main() -> int:
    for path in (EVENT_OUTPUT, DOSSIER):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    specification = json.loads(SPEC.read_text(encoding="ascii"))
    replay = json.loads(REPLAY.read_text(encoding="ascii"))
    events = specification["events"]
    if len(events) != 38 or replay["repair_bundle_count"] != 38:
        raise RuntimeError("38-bundle identity drift")

    occupied_by_event: dict[str, set[tuple[str, int]]] = {}
    for event in events:
        nodes = [
            int(event["parent"]["node_id"]),
            *(int(row["node_id"]) for row in event["children"]),
            *(
                int(row["node_id"])
                for row in event["persistence_evidence"]
                if row is not None
            ),
        ]
        occupied_by_event[event["event_key"]] = {
            (event["identity"], node) for node in nodes
        }
    overlapping: dict[str, list[str]] = {}
    for event in events:
        key = event["event_key"]
        overlapping[key] = sorted(
            other["event_key"]
            for other in events
            if other["event_key"] != key
            and occupied_by_event[key]
            & occupied_by_event[other["event_key"]]
        )
    if any(overlapping.values()):
        raise RuntimeError("cross-bundle occupied-node overlap drift")

    rows = []
    for ordinal, event in enumerate(events, start=1):
        late = event["late_fork_equivalence"]
        if late["available"]:
            preferred = late["preferred"]
            late_record = {
                "available": True,
                "scorer_equivalence_key": late["scorer_equivalence_key"],
                "continuation_child_id": preferred["continuation_child_id"],
                "linear_required_edges": preferred["linear_continuity"][
                    "required_edges"
                ],
                "late_fork_required_edges": preferred["late_fork"][
                    "required_edges"
                ],
                "total_edit_count": preferred["total_edit_count"],
                "selection_rule": late["selection_rule"],
            }
            risk = (
                "MUTUALLY_EXCLUSIVE_DIRECT_LATE: suppress by scorer-equivalence "
                "key before occupied-node ownership; no cross-bundle node overlap"
            )
        else:
            late_record = {
                "available": False,
                "scorer_equivalence_key": late["scorer_equivalence_key"],
                "reason": late.get("reason", "missing two-branch persistence"),
            }
            risk = (
                "DIRECT_ONLY: no late alternative; equivalence key remains unique "
                "and no cross-bundle node overlap exists"
            )
        rows.append(
            {
                "ordinal": ordinal,
                "event_key": event["event_key"],
                "identity": event["identity"],
                "family": event["family"],
                "gt_divider_id": event["gt_divider_id"],
                "parent_id": int(event["parent"]["node_id"]),
                "child_ids": sorted(
                    int(row["node_id"]) for row in event["children"]
                ),
                "direct_required_edges": event["direct_repair"][
                    "required_edges"
                ],
                "direct_net_add_edges": event["direct_repair"]["net_add_edges"],
                "direct_net_delete_edges": event["direct_repair"][
                    "net_delete_edges"
                ],
                "motif_class": event["motif_class"],
                "late_fork_equivalence": late_record,
                "duplicate_suppression_risk": risk,
                "cross_bundle_occupied_node_overlaps": overlapping[
                    event["event_key"]
                ],
            }
        )
    bundle_artifact = {
        "schema": "celltrack-alpha.division-plan-amendment-bundles.v1",
        "status": "DIAGNOSTIC_ONLY_FITTING_LOCKED",
        "source": {"path": str(SPEC), "sha256": sha256(SPEC)},
        "counts": specification["counts"],
        "duplicate_suppression": {
            "direct_late_same_equivalence_class": 36,
            "direct_only": 2,
            "cross_bundle_occupied_node_overlap_count": 0,
            "required_order": (
                "equivalence-key suppression, then occupied-node ownership, "
                "then atomic frozen-N1 emission"
            ),
            "duplicate_survivors_required": 0,
        },
        "events": rows,
        "fitting": False,
        "plan_amendment": False,
    }
    EVENT_OUTPUT.write_text(
        json.dumps(bundle_artifact, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    event_lines = []
    for row in rows:
        late = row["late_fork_equivalence"]
        if late["available"]:
            late_text = (
                f"yes; linear {edge_text(late['linear_required_edges'])}; "
                f"fork {edge_text(late['late_fork_required_edges'])}"
            )
        else:
            late_text = f"no; {late['reason']}"
        event_lines.append(
            "| {ordinal} | `{event_key}` | {family} | `{direct}` | "
            "`{motif}` | {late} | {risk} |".format(
                ordinal=row["ordinal"],
                event_key=row["event_key"],
                family=row["family"],
                direct=edge_text(row["direct_required_edges"]),
                motif=row["motif_class"],
                late=late_text,
                risk=row["duplicate_suppression_risk"],
            )
        )

    delta = replay["delta"]
    if not math.isclose(
        delta["combined_score_delta"], 0.09206784151854919, abs_tol=1e-15
    ):
        raise RuntimeError("oracle ceiling drift before dossier emission")
    fp_rows = [
        (0, 0.1 * 25 / 42),
        (5, 0.1 * 25 / 47),
        (10, 0.1 * 25 / 52),
    ]
    dossier = f"""# Division plan-amendment dossier

Status: **HUMAN DECISION REQUIRED — NO AMENDMENT MADE — FITTING LOCKED**

This dossier prepares one decision: whether to unlock a fit-only division arm from the
association-continuity prerequisite. It is not the amendment, it grants no execution
authority, and it does not change any frozen gate.

## Decision summary

The old prerequisite has become a poor sequencing rule, but the deployment problem has
not disappeared. The banked local score is `0.881522`; the planning range corresponding
to public `0.910` is `0.9255–0.9422`, leaving `+0.043978–+0.060678`
(reported as `+0.044–+0.061`). An edge-only current-pool projection caps near `0.896`,
still `+0.0295–+0.0462` short of that target range. Division is therefore one of the few
levers with enough arithmetic headroom:

| Zero-FP division TP | Division Jaccard | Combined-score addition |
|---:|---:|---:|
| 12 | `{12 / 42:.12f}` | `+{0.1 * 12 / 42:.12f}` (`+0.0286`) |
| 20 | `{20 / 42:.12f}` | `+{0.1 * 20 / 42:.12f}` (`+0.0476`) |
| 25 | `{25 / 42:.12f}` | `+{0.1 * 25 / 42:.12f}` (`+0.0595`) |

The evidence supports human review of an unlock; it does not establish that a
non-oracle selector can reach those TP counts.

## Why the association-continuity prerequisite is stale

`B1_division_tp.md` measured a late postprocessor on an already fragmented graph:
production added no edge on 13 valid fixtures and restored `0/51` removed fork edges.
That was valid evidence that constants or a wider orphan gate could not repair missing
connectivity after the fact. It predates two material facts:

1. the campaign now has **38 exact, compiler-valid repair bundles** rather than only a
   generic postprocessor; and
2. six subsequent association mechanisms have produced approximately zero or negative
   usable graph lift, with the formal qualifying both-family arm count still `0`.

Keeping division fitting behind an association win therefore makes the only measured
large residual lever depend on a lane that has repeatedly failed. The proposed decision
is about sequencing only: allow a strictly isolated division selector to be falsified
directly. It is not permission to ignore edge FP, duplicate, or held-fold controls.

## Oracle-only ceiling through frozen N1

The hash-pinned diagnostic execution authenticated all 38 Wave-3 actions, compiled every
action through frozen N1, applied the bundles to the frozen CV57 graphs, and ran the
frozen scorer. This dossier freshly reverified its result, all 38 compiler receipts, the
54-check validation, and its zero-swap resource receipt. This is explicitly
**GT-DEPENDENT** and is a ceiling, not achievable gain:

- division TP/FP/FN: `{delta['division_tp']}/{delta['division_fp']}/{delta['division_fn']}`;
- division Jaccard: `{delta['division_jaccard']:.12f}`;
- adjusted/raw/node deltas: `+{delta['adjusted_edge_jaccard_delta']:.12f}` /
  `+{delta['raw_edge_jaccard_delta']:.12f}` /
  `{delta['node_recall_delta']:.12f}`;
- weighted division term: `0.1 × {delta['division_jaccard']:.12f} =
  +{0.1 * delta['division_jaccard']:.12f}`;
- exact combined delta: `+{delta['combined_score_delta']:.12f}`.

Identity check:
`+{delta['adjusted_edge_jaccard_delta']:.12f} + 0.1 ×
{delta['division_jaccard']:.12f} = +{delta['combined_score_delta']:.12f}`.

This quantifies the ceiling requested for the decision. The separate deployment audit
still says only `1/38` is reachable from the current pool/features at perfect selection
(`+0.002406490`); `37/38` require a stronger representation. Oracle headroom is not
selector evidence.

## False-positive risk is binding

For 25 true division forks, FP erodes the division-only score contribution as follows:

| TP | FP | Jaccard `TP/(42+FP)` | Combined addition |
|---:|---:|---:|---:|
{chr(10).join(f"| 25 | {fp} | `{25 / (42 + fp):.12f}` | `+{value:.12f}` |" for fp, value in fp_rows)}

Thus 5 FP reduce `+0.0595` to `+0.0532`; 10 FP reduce it to `+0.0481`. This table is
edge-neutral. Any wrong fork can also damage adjusted/raw edge Jaccard, so it is an
optimistic bound.

## The arm that would run only if a human unlocks it

The existing preregistered `D1-LATE-PERSISTENT-ONE-TO-ONE-FORK` comparison remains
unchanged:

1. `UNCHANGED_GRAPH`;
2. `D0_PARENTAL_SOFTMAX_GREEDY_FORK`;
3. `D1_LATE_PERSISTENT_ONE_TO_ONE_FORK`;
4. `D1_PERSISTENCE_ABLATED`.

Direct and one-frame-late realizations share one scorer-equivalence class. Keep at most
one candidate per equivalence key, then enforce occupied-node ownership, require both
branches to persist, emit the two-edge fork atomically through frozen N1, and require
zero duplicate survivors.

The execution order would be `44b6 → 6bba → CV57`, with **44b6 first**. Kill the
instantiation immediately on 44b6 if division Jaccard is `<=0.15`, any adjusted/raw/node
delta is negative, combined delta is nonpositive, or any duplicate survives. GO remains
division Jaccard `>=0.30`, nonnegative graph vectors, positive combined delta, and D1
attribution beyond D0 on both families. No threshold rescue, family splicing, or
post-outcome change is permitted.

This is a description of the already preregistered arm, not authorization to run it.

## Per-event repair bundles (38/42)

Counts: 9 `44b6`, 29 `6bba`; 37
`SINGLE_DAUGHTER_REPARENT_DIRECT`, 1
`DOUBLE_DAUGHTER_REPARENT_DIRECT`; 36 have a deterministic late-fork equivalent and 2
are direct-only because two-branch persistence is absent. There are no cross-bundle
occupied-node overlaps. For the 36 events with two timing realizations, the duplicate
risk is within-event: direct and late are mutually exclusive representations of the
same scorer event and must be suppressed by equivalence key before node ownership.

| # | Event | Family | Direct required edges | Motif | Preferred late equivalence | Duplicate-suppression risk |
|---:|---|---|---|---|---|---|
{chr(10).join(event_lines)}

Machine-readable bundle details, including net adds/deletes and equivalence keys, are
in `DIVISION_REPAIR_BUNDLES.json`.

## Decision options and barrier

- **Approve a plan amendment:** unlock only the preregistered fit-only D1 arm above,
  preserving its 44b6-first kill line, held-fold discipline, FP accounting, and all
  downstream human gates.
- **Decline or defer:** keep `DIV-F1` dependency-parked and division output OFF.

Until the owner decides, **FITTING REMAINS LOCKED**. No training, fit, graph execution,
6bba, CV57, submission, Kaggle, RunPod, tier move, DEAD promotion, or plan amendment was
performed by this dossier.

## Evidence identities

- frozen plan: `a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116`;
- B1 result: `c9f2b4d1b8e33f5dae456de9e0ac1649b08f2cda8591a67b9d00d4a355dffbb1`;
- 38-bundle target spec: `{sha256(SPEC)}`;
- oracle measurement receipt: `{sha256(REPLAY)}`;
- per-event dossier: `{sha256(EVENT_OUTPUT)}`;
- frozen N1 implementation: `{replay['input_sha256']['n1_implementation']}`.
"""
    DOSSIER.write_text(dossier, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "DOSSIER_COMPLETE_FITTING_LOCKED",
                "events": len(rows),
                "dossier_sha256": sha256(DOSSIER),
                "bundle_sha256": sha256(EVENT_OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
