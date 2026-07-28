#!/usr/bin/env python3
"""Audit CV57 division fragmentation and compile diagnostic N1 fork actions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path("/Users/agent/Desktop/cell_tracking")
FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
REPORT = ROOT / "reports/g02/division_failure_holdout57_postpatch_v1.json"

PINS = {
    "alignment": (
        ROOT / "reports/g02/screening/tiered_surface/ALIGNMENT.md",
        "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
    ),
    "scope_manifest": (
        ROOT / "reports/g02/screening/tiered_surface/SCOPE_MANIFEST.md",
        "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
    ),
    "n1_interface": (
        ROOT
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        ROOT / "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "folds": (
        ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        ROOT
        / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "division_failure_report": (
        REPORT,
        "10a8e480cbd0134f0fec18c1d302b388c4dd3ecbaf02f6eb3298f79eb40d2b44",
    ),
    "d1_preregistration": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N0_preregistration.json",
        "34b2aa020099e14ad7a73820d97e9baefadec386e0e2d9acbd7ab5c1cb07620f",
    ),
    "d1_arm_manifest": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N1_arm_manifest.json",
        "327af0d29dbde19172a61bf396b951d150040bb357c3fbba90e79249035e4776",
    ),
    "d1_dry_validation": (
        ROOT
        / "reports/g02/screening/top50_parallel_dags/runs/20260727_top50_six_01/"
        "d1_persistent_fork/N1_dry_validation.json",
        "f6a96d868e6ad2a21aea123157bc9bddc169e8453c117e829495d933514deaf9",
    ),
}


@dataclass(frozen=True)
class Node:
    node_id: int
    t: int
    z: int
    y: int
    x: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate() -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for name, (path, expected) in PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"pin mismatch for {name}: {observed}")
        receipts[name] = {"path": str(path), "sha256": observed, "match": True}
    return receipts


def read_fragment(path: Path) -> tuple[dict[int, Node], set[tuple[int, int]]]:
    nodes: dict[int, Node] = {}
    edges: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] == "node":
                node_id = int(row["node_id"])
                nodes[node_id] = Node(
                    node_id=node_id,
                    t=int(row["t"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                )
            elif row["row_type"] == "edge":
                edges.add((int(row["source_id"]), int(row["target_id"])))
            else:
                raise AssertionError(f"unknown row type in {path}: {row['row_type']}")
    if not nodes or any(endpoint not in nodes for edge in edges for endpoint in edge):
        raise AssertionError(f"malformed graph fragment: {path}")
    return nodes, edges


def internal_to_submission_ids(nodes: dict[int, Node]) -> dict[int, int]:
    """Mirror official_eval.prediction_to_graph's deterministic add-node order."""

    ordered = sorted(nodes.values(), key=lambda node: (node.t, node.node_id))
    return {internal_id: node.node_id for internal_id, node in enumerate(ordered)}


def components(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> dict[int, int]:
    parent = {int(node): int(node) for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for source, target in sorted(edges):
        union(source, target)
    return {node: find(node) for node in sorted(parent)}


def classify_roles(
    role_covered: bool,
    parent_ids: list[int],
    daughter_groups: list[list[int]],
    component: dict[int, int],
) -> tuple[str, dict[str, object]]:
    if not role_covered:
        return "ROLE_MISSING", {
            "role_component_count": None,
            "spanning_component_count": 0,
        }
    if len(daughter_groups) != 2:
        raise AssertionError("strict scorer audit requires exactly two daughter lineages")
    required = [*parent_ids, *daughter_groups[0], *daughter_groups[1]]
    missing = sorted(node for node in required if node not in component)
    if missing:
        raise AssertionError(
            f"frozen matched role references missing prediction nodes: {missing}"
        )
    role_components = {component[node] for node in required}
    spanning = (
        {component[node] for node in parent_ids}
        & {component[node] for node in daughter_groups[0]}
        & {component[node] for node in daughter_groups[1]}
    )
    if spanning:
        classification = "PRESENT_CONNECTED_NO_LOCAL_FORK"
    elif len(role_components) >= 2:
        classification = "PRESENT_FRAGMENTED"
    else:
        raise AssertionError("role-covered event is neither spanning nor fragmented")
    return classification, {
        "role_component_count": len(role_components),
        "spanning_component_count": len(spanning),
    }


def choose_oracle_nodes(
    nodes: dict[int, Node],
    parent_ids: list[int],
    daughter_groups: list[list[int]],
) -> tuple[Node, Node, Node, Node | None, Node | None]:
    parents = sorted((nodes[node] for node in parent_ids), key=lambda node: (-node.t, node.node_id))
    parent = parents[0]

    def branch(group: list[int]) -> tuple[Node, Node | None]:
        later = sorted(
            (nodes[node] for node in group if nodes[node].t > parent.t),
            key=lambda node: (node.t, node.node_id),
        )
        if not later:
            raise AssertionError("role-covered branch has no node after selected parent")
        child = later[0]
        grandchildren = [node for node in later[1:] if node.t > child.t]
        return child, grandchildren[0] if grandchildren else None

    child_a, grandchild_a = branch(daughter_groups[0])
    child_b, grandchild_b = branch(daughter_groups[1])
    if child_a.node_id == child_b.node_id:
        raise AssertionError("daughter lineages collapsed to one prediction node")
    if child_b.node_id < child_a.node_id:
        child_a, child_b = child_b, child_a
        grandchild_a, grandchild_b = grandchild_b, grandchild_a
    return parent, child_a, child_b, grandchild_a, grandchild_b


def compile_oracle_action(
    identity: str,
    event_index: int,
    gt_divider_id: int,
    nodes: dict[int, Node],
    edges: set[tuple[int, int]],
    selected: tuple[Node, Node, Node, Node | None, Node | None],
) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from celltrack_alpha.action_compiler import (  # type: ignore
        ActionKind,
        GraphAction,
        action_sha256,
        compile_action,
    )
    from celltrack_alpha.kaggle_calibration import (  # type: ignore
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
    )

    parent, child_a, child_b, grandchild_a, grandchild_b = selected
    graph = SubmissionGraph(
        dataset=identity.split(":", 1)[1],
        nodes={
            node.node_id: SubmissionNode(node.node_id, node.t, node.z, node.y, node.x)
            for node in nodes.values()
        },
        edges=[SubmissionEdge(source, target) for source, target in sorted(edges)],
    )
    add_edges = ((parent.node_id, child_a.node_id), (parent.node_id, child_b.node_id))
    conflicts = tuple(
        sorted(
            edge
            for edge in edges
            if edge[0] == parent.node_id
            or edge[1] in (child_a.node_id, child_b.node_id)
        )
    )
    action = GraphAction(
        action_id=f"D1_ORACLE_{identity.replace(':', '_')}_{event_index:03d}",
        kind=ActionKind.DIVISION_FORK,
        add_edges=add_edges,
        replace_outgoing_from=(parent.node_id,),
        replace_incoming_to=(child_a.node_id, child_b.node_id),
        expected_conflict_edges=conflicts,
    )
    base = {
        "action_id": action.action_id,
        "oracle_equivalence_key": f"{identity}|gt-divider:{gt_divider_id}",
        "action_sha256": action_sha256(action),
        "selected_parent": asdict(parent),
        "selected_children": [asdict(child_a), asdict(child_b)],
        "persistence_evidence": [
            asdict(grandchild_a) if grandchild_a else None,
            asdict(grandchild_b) if grandchild_b else None,
        ],
        "persistent_both_branches": grandchild_a is not None and grandchild_b is not None,
        "equivalence_timing_realization": "DIRECT",
        "expected_conflict_edges": [list(edge) for edge in conflicts],
        "graph_scored": False,
    }
    try:
        compiled = compile_action(graph, action)
    except Exception as exc:
        return {**base, "n1_compiled": False, "compile_error": str(exc)}
    return {
        **base,
        "n1_compiled": True,
        "compile_error": None,
        "before_graph_sha256": compiled.before_graph_sha256,
        "after_graph_sha256": compiled.after_graph_sha256,
        "removed_edges": [list(edge) for edge in compiled.removed_edges],
    }


def suppress_oracle_actions(
    events: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the frozen persistence/equivalence/node-ownership suppression."""

    selected: list[dict[str, object]] = []
    suppressed: list[dict[str, object]] = []
    used_keys: set[str] = set()
    occupied: set[tuple[str, int]] = set()
    candidates = sorted(
        (
            event
            for event in events
            if isinstance(event.get("oracle_action"), dict)
        ),
        key=lambda event: (
            str(event["identity"]),
            int(event["gt_divider_id"]),
            int(event["event_index"]),
            str(event["oracle_action"]["action_sha256"]),  # type: ignore[index]
        ),
    )
    for event in candidates:
        action = event["oracle_action"]
        assert isinstance(action, dict)
        key = f"{event['identity']}|gt-divider:{event['gt_divider_id']}"
        nodes = {
            (str(event["identity"]), int(action["selected_parent"]["node_id"])),
            *(
                (str(event["identity"]), int(node["node_id"]))
                for node in action["selected_children"]
            ),
            *(
                (str(event["identity"]), int(node["node_id"]))
                for node in action["persistence_evidence"]
                if node is not None
            ),
        }
        if not bool(action["persistent_both_branches"]):
            suppressed.append(
                {
                    "action_id": action["action_id"],
                    "equivalence_key": key,
                    "reason": "missing_branch_persistence",
                    "winner": None,
                }
            )
            continue
        if key in used_keys:
            winner = next(
                row["action_id"]
                for row in selected
                if row["equivalence_key"] == key
            )
            suppressed.append(
                {
                    "action_id": action["action_id"],
                    "equivalence_key": key,
                    "reason": "duplicate_equivalence_class",
                    "winner": winner,
                }
            )
            continue
        if nodes & occupied:
            winner = next(
                row["action_id"]
                for row in selected
                if set(row["occupied_nodes"]) & nodes
            )
            suppressed.append(
                {
                    "action_id": action["action_id"],
                    "equivalence_key": key,
                    "reason": "one_to_one_node_conflict",
                    "winner": winner,
                }
            )
            continue
        selected.append(
            {
                "action_id": action["action_id"],
                "action_sha256": action["action_sha256"],
                "equivalence_key": key,
                "occupied_nodes": sorted(nodes),
            }
        )
        used_keys.add(key)
        occupied.update(nodes)
    return selected, suppressed


def run(output: Path) -> dict[str, object]:
    authenticated = authenticate()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["summary"]["gt_divisions"] != 42:
        raise AssertionError("CV57 division denominator drifted")
    if report["summary"]["official_division_tp"] != 0:
        raise AssertionError("expected division-off zero-TP source")

    events: list[dict[str, object]] = []
    fragment_receipts: list[dict[str, object]] = []
    for sample in sorted(report["samples"], key=lambda row: row["identity"]):
        identity = str(sample["identity"])
        fragment = FRAGMENTS / f"{identity.replace(':', '__')}.csv"
        fragment_receipts.append(
            {"identity": identity, "path": str(fragment), "sha256": sha256(fragment)}
        )
        nodes, edges = read_fragment(fragment)
        component = components(nodes, edges)
        id_map = internal_to_submission_ids(nodes)
        for event_index, event in enumerate(sample["divisions"]):
            coverage = event["strict_role_coverage"]
            parent_ids = [
                id_map[int(node)] for node in coverage["parent_side_prediction_ids"]
            ]
            daughter_groups = [
                [id_map[int(node)] for node in group]
                for group in coverage["daughter_prediction_ids"]
            ]
            classification, component_audit = classify_roles(
                bool(coverage["role_covered"]),
                parent_ids,
                daughter_groups,
                component,
            )
            row: dict[str, object] = {
                "identity": identity,
                "event_index": event_index,
                "gt_divider_id": int(event["gt_divider_id"]),
                "banked_classification": str(event["classification"]),
                "id_domain": "submission node IDs converted from frozen scorer-internal IDs",
                "wave2_classification": classification,
                **component_audit,
                "oracle_action": None,
            }
            if classification != "ROLE_MISSING":
                selected = choose_oracle_nodes(nodes, parent_ids, daughter_groups)
                row["oracle_action"] = compile_oracle_action(
                    identity,
                    event_index,
                    int(event["gt_divider_id"]),
                    nodes,
                    edges,
                    selected,
                )
            events.append(row)

    counts = {
        label: sum(event["wave2_classification"] == label for event in events)
        for label in (
            "PRESENT_FRAGMENTED",
            "PRESENT_CONNECTED_NO_LOCAL_FORK",
            "ROLE_MISSING",
        )
    }
    compiled = [
        event["oracle_action"]
        for event in events
        if isinstance(event["oracle_action"], dict)
        and event["oracle_action"]["n1_compiled"]
    ]
    failures = [
        event["oracle_action"]
        for event in events
        if isinstance(event["oracle_action"], dict)
        and not event["oracle_action"]["n1_compiled"]
    ]
    d1_selected, d1_suppressed = suppress_oracle_actions(events)
    suppression_counts = {
        reason: sum(row["reason"] == reason for row in d1_suppressed)
        for reason in (
            "missing_branch_persistence",
            "duplicate_equivalence_class",
            "one_to_one_node_conflict",
        )
    }
    payload: dict[str, object] = {
        "schema": "celltrack-alpha.sa-division-deep-oracle-audit.v1",
        "status": "PASS_ORACLE_ONLY_FITTING_LOCKED",
        "authenticated_inputs": authenticated,
        "fragment_receipts": fragment_receipts,
        "counts": {
            "cv57_gt_divisions": len(events),
            **counts,
            "role_covered": len(events) - counts["ROLE_MISSING"],
            "n1_oracle_actions_compiled": len(compiled),
            "n1_oracle_actions_failed": len(failures),
            "d1_persistent_actions_selected": len(d1_selected),
            "d1_actions_suppressed": len(d1_suppressed),
            "duplicate_survivors": 0,
            **suppression_counts,
        },
        "denominator_notes": {
            "primary": "PRESENT_FRAGMENTED / 42 frozen CV57 GT division FNs",
            "role_covered": "all strict scorer roles present; does not imply scorer TP",
            "historical_prefix52": "35/37 disconnected in an older prediction identity; not pooled or substituted",
            "scorer_equivalence": "role presence is necessary but a TP also requires an eligible local fork and one-to-one scorer pairing",
        },
        "events": events,
        "d1_suppression": {
            "ordering": "identity, GT divider, event index, action SHA; diagnostic oracle has no fitted score",
            "selected": d1_selected,
            "suppressed": d1_suppressed,
        },
        "dependency": {
            "fitting_locked": True,
            "qualifying_both_family_association_arm_count": 0,
            "held_graph_run": False,
            "scorer_run": False,
            "cv57_fit": False,
        },
        "execution": {
            "cpu_cache_only": True,
            "gpu": False,
            "kaggle": False,
            "runpod": False,
            "submission": False,
            "budget_debit": False,
            "cellect_touched": False,
            "dead_verdict": False,
            "quarantined_memo_numbers_used": False,
        },
    }
    if len(events) != 42 or sum(counts.values()) != 42:
        raise AssertionError("event classification is not exhaustive")
    if len(d1_selected) + len(d1_suppressed) != len(compiled):
        raise AssertionError("D1 suppression did not exhaust the compiled oracle population")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
