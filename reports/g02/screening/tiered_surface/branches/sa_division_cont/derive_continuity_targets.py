#!/usr/bin/env python3
"""Derive exact oracle-only association targets for fragmented CV57 divisions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
AUDIT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_division_deep/"
    "oracle_audit_v2.json"
)
FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
AUDIT_SHA256 = "6d1a2754bbe597d53c29863290cba51225d3138b002b83ac4d4ad9c5d306c119"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_graph(path: Path) -> tuple[set[int], set[tuple[int, int]]]:
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] == "node":
                nodes.add(int(row["node_id"]))
            elif row["row_type"] == "edge":
                edges.add((int(row["source_id"]), int(row["target_id"])))
            else:
                raise AssertionError(f"unexpected row type: {row['row_type']}")
    return nodes, edges


def components(
    nodes: Iterable[int], edges: Iterable[tuple[int, int]]
) -> dict[int, int]:
    parent = {int(node): int(node) for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for source, target in sorted(edges):
        union(source, target)
    return {node: find(node) for node in sorted(parent)}


def incident_conflicts(
    edges: set[tuple[int, int]],
    parent: int,
    children: tuple[int, int],
) -> set[tuple[int, int]]:
    return {
        edge
        for edge in edges
        if edge[0] == parent or edge[1] in children
    }


def net_bundle(
    edges: set[tuple[int, int]],
    parent: int,
    children: tuple[int, int],
) -> dict[str, object]:
    required = {(parent, child) for child in children}
    conflicts = incident_conflicts(edges, parent, children)
    net_add = sorted(required - edges)
    net_delete = sorted(conflicts - required)
    preserve = sorted(required & edges)
    after = (edges - conflicts) | required
    return {
        "required_edges": [list(edge) for edge in sorted(required)],
        "net_add_edges": [list(edge) for edge in net_add],
        "net_delete_edges": [list(edge) for edge in net_delete],
        "already_correct_edges": [list(edge) for edge in preserve],
        "owned_incident_conflicts": [list(edge) for edge in sorted(conflicts)],
        "edit_count": len(net_add) + len(net_delete),
        "_after": after,
    }


def public_bundle(bundle: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in bundle.items() if not key.startswith("_")}


def late_candidates(
    edges: set[tuple[int, int]],
    parent: int,
    children: tuple[int, int],
    grandchildren: tuple[int, int],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for continuation in children:
        first = net_bundle(edges, parent, (continuation, continuation))
        first_required = {(parent, continuation)}
        first_conflicts = incident_conflicts(edges, parent, (continuation, continuation))
        after_linear = (edges - first_conflicts) | first_required
        fork = net_bundle(after_linear, continuation, grandchildren)
        final_edges = fork["_after"]
        assert isinstance(final_edges, set)
        candidate = {
            "continuation_child_id": continuation,
            "linear_continuity": public_bundle(first),
            "late_fork": public_bundle(fork),
            "total_edit_count": int(first["edit_count"]) + int(fork["edit_count"]),
            "_after": final_edges,
        }
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (int(row["total_edit_count"]), int(row["continuation_child_id"]))
    )
    return candidates


def run(output: Path) -> dict[str, object]:
    if sha256(AUDIT) != AUDIT_SHA256:
        raise AssertionError("banked Wave-2 division audit hash drifted")
    audit = json.loads(AUDIT.read_text(encoding="ascii"))
    events = [
        row
        for row in audit["events"]
        if row["wave2_classification"] == "PRESENT_FRAGMENTED"
    ]
    if len(events) != 38:
        raise AssertionError("expected exactly 38 fragmented events")

    graph_cache: dict[str, tuple[set[int], set[tuple[int, int]], str]] = {}
    targets: list[dict[str, object]] = []
    for event in sorted(
        events,
        key=lambda row: (
            str(row["identity"]),
            int(row["gt_divider_id"]),
            int(row["event_index"]),
        ),
    ):
        identity = str(event["identity"])
        if identity not in graph_cache:
            fragment = FRAGMENTS / f"{identity.replace(':', '__')}.csv"
            nodes, edges = read_graph(fragment)
            graph_cache[identity] = (nodes, edges, sha256(fragment))
        nodes, edges, fragment_sha = graph_cache[identity]
        action = event["oracle_action"]
        if not isinstance(action, dict) or not action["n1_compiled"]:
            raise AssertionError("fragmented event lacks compiled oracle action")
        parent_row = action["selected_parent"]
        child_rows = action["selected_children"]
        parent = int(parent_row["node_id"])
        children = tuple(sorted(int(row["node_id"]) for row in child_rows))
        if any(int(row["t"]) != int(parent_row["t"]) + 1 for row in child_rows):
            raise AssertionError("direct child timing drifted")

        direct = net_bundle(edges, parent, children)
        direct_after = direct["_after"]
        assert isinstance(direct_after, set)
        component = components(nodes, direct_after)
        if len({component[parent], *(component[node] for node in children)}) != 1:
            raise AssertionError("direct bundle failed to connect event roles")
        correct_count = len(direct["already_correct_edges"])
        if correct_count == 1:
            motif = "SINGLE_DAUGHTER_REPARENT_DIRECT"
        elif correct_count == 0:
            motif = "DOUBLE_DAUGHTER_REPARENT_DIRECT"
        else:
            raise AssertionError("fragmented event unexpectedly has a complete fork")

        persistence_rows = action["persistence_evidence"]
        persistent = bool(action["persistent_both_branches"])
        late: dict[str, object]
        if persistent:
            grandchildren = tuple(
                sorted(int(row["node_id"]) for row in persistence_rows if row is not None)
            )
            if len(grandchildren) != 2:
                raise AssertionError("persistent event lacks two grandchildren")
            candidates = late_candidates(edges, parent, children, grandchildren)
            for candidate in candidates:
                after = candidate["_after"]
                assert isinstance(after, set)
                component = components(nodes, after)
                required_nodes = (
                    parent,
                    int(candidate["continuation_child_id"]),
                    *grandchildren,
                )
                candidate["connects_all_role_and_persistence_nodes"] = (
                    len({component[node] for node in required_nodes}) == 1
                )
                del candidate["_after"]
            late = {
                "available": True,
                "scorer_equivalence_key": action["oracle_equivalence_key"],
                "definition": (
                    "parent-side lineage continues linearly for one frame; a binary "
                    "fork is emitted on that continuation to the two persistent "
                    "grandchild-stage branches"
                ),
                "preferred": candidates[0],
                "alternatives": candidates[1:],
                "selection_rule": (
                    "minimum total net edits, then lowest continuation child id; "
                    "oracle diagnostic only and never a fitted feature"
                ),
            }
        else:
            late = {
                "available": False,
                "scorer_equivalence_key": action["oracle_equivalence_key"],
                "reason": "two distinct grandchild-stage persistence nodes are absent",
            }

        target = {
            "event_key": (
                f"{identity}|gt-divider:{event['gt_divider_id']}|"
                f"event:{event['event_index']}"
            ),
            "identity": identity,
            "family": identity.split(":", 1)[1].split("_", 1)[0],
            "event_index": int(event["event_index"]),
            "gt_divider_id": int(event["gt_divider_id"]),
            "source_fragment_sha256": fragment_sha,
            "wave2_action_sha256": action["action_sha256"],
            "motif_class": motif,
            "role_component_count_before": int(event["role_component_count"]),
            "parent": parent_row,
            "children": child_rows,
            "persistence_evidence": persistence_rows,
            "direct_repair": public_bundle(direct),
            "late_fork_equivalence": late,
            "future_arm_target": (
                "recover this exact endpoint-identity bundle label-free on emitted "
                "association graphs; D1 fork fitting remains separately locked"
            ),
        }
        targets.append(target)

    motif_counts: dict[str, int] = {}
    for row in targets:
        motif = str(row["motif_class"])
        motif_counts[motif] = motif_counts.get(motif, 0) + 1
    late_count = sum(bool(row["late_fork_equivalence"]["available"]) for row in targets)
    payload: dict[str, object] = {
        "schema": "celltrack-alpha.sa-division-continuity-targets.v1",
        "status": "PASS_ORACLE_TARGET_SPEC_FITTING_LOCKED",
        "source": {
            "path": str(AUDIT),
            "sha256": AUDIT_SHA256,
            "fragment_count": len(graph_cache),
        },
        "counts": {
            "fragmented_events": len(targets),
            "family_44b6": sum(row["family"] == "44b6" for row in targets),
            "family_6bba": sum(row["family"] == "6bba" for row in targets),
            "motifs": motif_counts,
            "late_fork_equivalence_available": late_count,
            "direct_only_missing_persistence": len(targets) - late_count,
        },
        "contract": {
            "oracle_only": True,
            "association_arm_must_be_label_free": True,
            "fitting_locked": True,
            "graph_scoring": False,
            "cv57_fitting": False,
            "family_splicing": False,
            "qualifying_dependency": (
                "one named association-continuity arm must pass adjusted >= +0.005 "
                "with raw/node >= 0 on both 44b6 and 6bba and increase label-free "
                "connected fork opportunities"
            ),
        },
        "events": targets,
        "execution": {
            "cpu_local_only": True,
            "gpu": False,
            "kaggle": False,
            "runpod": False,
            "submission": False,
            "budget_debit": False,
            "cellect_touched": False,
            "dead_action": False,
        },
    }
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return payload


def main() -> int:
    output = HERE / "continuity_target_spec.json"
    payload = run(output)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
