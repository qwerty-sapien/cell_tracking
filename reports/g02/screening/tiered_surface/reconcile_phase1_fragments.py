#!/usr/bin/env python3
"""Deterministically overlay validated Phase-1 fragments onto the root census."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


TIERS = {"LIVE", "PARK_A", "PARK_B", "PARK_C", "DEAD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--updated-utc", required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, action="append", required=True)
    parser.add_argument("--accept-node", action="append", default=[])
    parser.add_argument("--accept-move", action="append", default=[])
    parser.add_argument("--reject-node", action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    census_path = args.census.resolve()
    census = json.loads(census_path.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.resolve().read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("Phase-1 barrier validation has not passed")
    validated_fragment_hashes = {
        lane["fragment_sha256"] for lane in validation["lanes"].values()
    }
    allowed_moves: set[tuple[str, str, str]] = set()
    for raw in args.accept_move:
        parts = tuple(raw.split(":"))
        if len(parts) != 3:
            raise SystemExit(f"malformed accepted move: {raw}")
        allowed_moves.add(parts)
    rejected_nodes = set(args.reject_node)
    accepted_nodes = set(args.accept_node)
    if accepted_nodes & rejected_nodes:
        raise SystemExit("a node cannot be both accepted and rejected")
    indexed = {node["id"]: node for node in census["nodes"]}
    if len(indexed) != len(census["nodes"]):
        raise SystemExit("duplicate node IDs in root census")

    seen: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for fragment_arg in args.fragment:
        fragment_path = fragment_arg.resolve()
        fragment_digest = sha256(fragment_path)
        if fragment_digest not in validated_fragment_hashes:
            raise SystemExit(
                f"fragment was not hash-bound by the barrier: {fragment_path}"
            )
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        owner = fragment.get("owner_agent")
        for proposed in fragment["nodes"]:
            node_id = proposed["id"]
            if node_id in seen:
                raise SystemExit(f"conflicting Phase-1 fragments for {node_id}")
            seen.add(node_id)
            if node_id not in indexed:
                raise SystemExit(f"fragment proposes unknown census node {node_id}")
            if proposed["tier"] not in TIERS:
                raise SystemExit(f"invalid tier for {node_id}")
            evidence = census_path.parents[4] / proposed["evidence_artifact"]
            if not evidence.is_file() or sha256(evidence) != proposed["evidence_sha256"]:
                raise SystemExit(f"unresolved merged evidence for {node_id}")

            before = indexed[node_id]
            if node_id in rejected_nodes:
                decisions.append(
                    {
                        "node_id": node_id,
                        "owner_agent": owner or proposed.get("owner_agent"),
                        "before_tier": before["tier"],
                        "after_tier": before["tier"],
                        "tier_changed": False,
                        "evidence_artifact": proposed["evidence_artifact"],
                        "evidence_sha256": proposed["evidence_sha256"],
                        "fragment_sha256": fragment_digest,
                        "decision": "REJECT_OVERBROAD_MECHANISM_TIER",
                    }
                )
                continue
            if node_id not in accepted_nodes:
                raise SystemExit(f"node is absent from root acceptance policy: {node_id}")
            proposed_move = (node_id, before["tier"], proposed["tier"])
            if before["tier"] != proposed["tier"] and proposed_move not in allowed_moves:
                raise SystemExit(f"unapproved tier transition: {proposed_move}")
            after = dict(before)
            after.update(proposed)
            indexed[node_id] = after
            decisions.append(
                {
                    "node_id": node_id,
                    "owner_agent": owner or proposed.get("owner_agent"),
                    "before_tier": before["tier"],
                    "after_tier": after["tier"],
                    "tier_changed": before["tier"] != after["tier"],
                    "evidence_artifact": after["evidence_artifact"],
                    "evidence_sha256": after["evidence_sha256"],
                    "fragment_sha256": fragment_digest,
                    "decision": "ACCEPT_CITED_FRAGMENT",
                }
            )

    census["nodes"] = [indexed[node["id"]] for node in census["nodes"]]
    census["active_agents"] = []
    census["phase"] = "PHASE1_BARRIER_COMPLETE"
    census["phase1_worktrees_manifest"]["sha256"] = args.manifest_sha256
    census["rollup"] = dict(
        sorted(Counter(node["tier"] for node in census["nodes"]).items())
    )
    census["updated_utc"] = args.updated_utc
    census_path.write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    decision_record = {
        "schema": "celltrack-alpha.phase1-root-reconciliation.v1",
        "status": "PASS",
        "updated_utc": args.updated_utc,
        "census_path": str(census_path),
        "fragment_count": len(args.fragment),
        "node_decisions": decisions,
        "accepted_move_allowlist": sorted(args.accept_move),
        "accepted_node_allowlist": sorted(accepted_nodes),
        "rejected_node_allowlist": sorted(rejected_nodes),
        "uncited_tier_moves_rejected": 0,
        "memo_numbers_used_for_tiering": 0,
        "family_sequence_interpretation": (
            "A family opens when its graph outcome is computed. Frozen fit117 "
            "rows from both named families remain shared training inputs under "
            "the locked P0/fold population; they are not a 6bba graph execution."
        ),
        "division_dependency_only_reads": (
            "SA-DIVISION additionally authenticated existing root-banked "
            "H1/C1/U0/C2/D1 terminal evidence solely to test the locked "
            "both-family association prerequisite. It consumed no sibling "
            "live output and performed no fitting or family graph execution."
        ),
        "cv57_used": False,
        "submission_action": False,
        "runpod_action": False,
        "gpu_authorizations": 0,
        "kaggle_budget_debit_hours": 0.0,
    }
    args.decisions.resolve().write_text(
        json.dumps(decision_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision_record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
