#!/usr/bin/env python3
"""Generate Arm-A label-free geometry/motion division-fork predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


SCALE = (1.625, 0.40625, 0.40625)
PRIMARY_MAX = 7.65
SECONDARY_MAX = 4.66
SISTER_MIN = 3.0
SISTER_MAX = 8.5
SYMMETRY_MAX = 2.0
MIDPOINT_MAX = 2.0
MOTION_GAIN_MIN = 1.0
COUNT_RADIUS = 8.5
COUNT_DELTA = 1
SCORE_MAX = 3.25
PER_VIDEO_CAP = 1
EXPECTED_SHARED_CONTRACT_SHA256 = (
    "08f3e875e77a5c7f93c56ce779df93f4cbbed1bb8a9a7f183369452c9e29e647"
)


class ArmAError(RuntimeError):
    """Fail-closed Arm-A prediction error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def distance(a: Any, b: Any) -> float:
    return math.sqrt(
        sum(
            ((float(av) - float(bv)) * scale) ** 2
            for av, bv, scale in zip(a.coord_zyx, b.coord_zyx, SCALE, strict=True)
        )
    )


def predicted_coord(previous: Any, parent: Any) -> tuple[float, float, float]:
    return tuple(
        2.0 * float(parent_value) - float(previous_value)
        for previous_value, parent_value in zip(
            previous.coord_zyx, parent.coord_zyx, strict=True
        )
    )


def coord_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(
        sum(
            ((float(av) - float(bv)) * scale) ** 2
            for av, bv, scale in zip(a, b, SCALE, strict=True)
        )
    )


def local_count(nodes: Iterable[Any], center: Iterable[float]) -> int:
    return sum(coord_distance(node.coord_zyx, center) <= COUNT_RADIUS for node in nodes)


def key_for(identity: str, parent: int, branches: Sequence[int]) -> str:
    payload = {
        "identity": identity,
        "observed_parent_node": int(parent),
        "observed_branch_nodes": sorted(int(value) for value in branches),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def incident_conflicts(
    edges: set[tuple[int, int]], parent: int, branches: Sequence[int]
) -> list[list[int]]:
    branch_set = set(int(value) for value in branches)
    return [
        [source, target]
        for source, target in sorted(edges)
        if source == parent or target in branch_set
    ]


def candidates_for_graph(graph: Any, identity: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    by_time: dict[int, list[Any]] = defaultdict(list)
    for node in graph.nodes.values():
        by_time[int(node.t)].append(node)

    births: dict[int, list[Any]] = defaultdict(list)
    for node in graph.nodes.values():
        if not incoming.get(int(node.node_id)):
            births[int(node.t)].append(node)

    raw_parent_secondary_pairs = 0
    hard_geometry_pass = 0
    count_pass = 0
    motion_pass = 0
    score_pass = 0
    proposals: list[dict[str, Any]] = []
    for parent_id in sorted(outgoing):
        children = sorted(set(outgoing[parent_id]))
        predecessors = sorted(set(incoming.get(parent_id, ())))
        if len(children) != 1 or len(predecessors) != 1:
            continue
        parent = graph.nodes[parent_id]
        primary = graph.nodes[children[0]]
        previous = graph.nodes[predecessors[0]]
        if primary.t != parent.t + 1 or parent.t != previous.t + 1:
            continue
        primary_um = distance(parent, primary)
        if primary_um > PRIMARY_MAX:
            continue
        prediction = predicted_coord(previous, parent)
        for secondary in sorted(
            births.get(parent.t + 1, ()), key=lambda node: int(node.node_id)
        ):
            if secondary.node_id == primary.node_id:
                continue
            raw_parent_secondary_pairs += 1
            secondary_um = distance(parent, secondary)
            sister_um = distance(primary, secondary)
            symmetry_um = abs(primary_um - secondary_um)
            midpoint = tuple(
                0.5 * (float(a) + float(b))
                for a, b in zip(primary.coord_zyx, secondary.coord_zyx, strict=True)
            )
            midpoint_residual_um = coord_distance(midpoint, prediction)
            primary_motion_residual_um = coord_distance(primary.coord_zyx, prediction)
            secondary_motion_residual_um = coord_distance(
                secondary.coord_zyx, prediction
            )
            motion_gain_um = (
                min(primary_motion_residual_um, secondary_motion_residual_um)
                - midpoint_residual_um
            )
            if not (
                secondary_um <= SECONDARY_MAX
                and SISTER_MIN <= sister_um <= SISTER_MAX
                and symmetry_um <= SYMMETRY_MAX
                and midpoint_residual_um <= MIDPOINT_MAX
            ):
                continue
            hard_geometry_pass += 1
            current_count = local_count(by_time[parent.t], parent.coord_zyx)
            next_count = local_count(by_time[parent.t + 1], prediction)
            local_count_change = next_count - current_count
            if local_count_change != COUNT_DELTA:
                continue
            count_pass += 1
            if motion_gain_um < MOTION_GAIN_MIN:
                continue
            motion_pass += 1
            score = (
                secondary_um
                + 0.15 * sister_um
                + midpoint_residual_um
                + 0.5 * symmetry_um
                - 0.5 * motion_gain_um
            )
            if score > SCORE_MAX:
                continue
            score_pass += 1
            branches = sorted((int(primary.node_id), int(secondary.node_id)))
            candidate_key = key_for(identity, int(parent.node_id), branches)
            proposals.append(
                {
                    "candidate_key": candidate_key,
                    "identity": identity,
                    "dataset": str(graph.dataset),
                    "observed_parent_node": int(parent.node_id),
                    "observed_branch_nodes": branches,
                    "primary_branch_node": int(primary.node_id),
                    "secondary_branch_node": int(secondary.node_id),
                    "features": {
                        "primary_parent_um": primary_um,
                        "secondary_parent_um": secondary_um,
                        "sister_um": sister_um,
                        "symmetry_um": symmetry_um,
                        "midpoint_motion_residual_um": midpoint_residual_um,
                        "primary_motion_residual_um": primary_motion_residual_um,
                        "secondary_motion_residual_um": secondary_motion_residual_um,
                        "fork_motion_gain_um": motion_gain_um,
                        "local_count_current": current_count,
                        "local_count_next": next_count,
                        "local_count_change": local_count_change,
                        "score": score,
                    },
                    "action": {
                        "kind": "division_fork",
                        "add_edges": [
                            [int(parent.node_id), branches[0]],
                            [int(parent.node_id), branches[1]],
                        ],
                        "replace_outgoing_from": [int(parent.node_id)],
                        "replace_incoming_to": branches,
                        "expected_conflict_edges": incident_conflicts(
                            edges, int(parent.node_id), branches
                        ),
                    },
                }
            )
    return proposals, {
        "raw_parent_secondary_pairs": raw_parent_secondary_pairs,
        "hard_geometry_pass": hard_geometry_pass,
        "local_count_pass": count_pass,
        "motion_pass": motion_pass,
        "score_pass": score_pass,
    }


def compile_selected(repo_root: Path, graphs: dict[str, Any], selected: list[dict[str, Any]]) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    from celltrack_alpha.action_compiler import (  # pylint: disable=import-outside-toplevel
        ActionKind,
        GraphAction,
        compile_action,
    )

    for row in selected:
        action = row["action"]
        graph = graphs[row["dataset"]]
        graph_action = GraphAction(
            action_id=f"armA.{row['candidate_key'][:24]}",
            kind=ActionKind.DIVISION_FORK,
            add_edges=tuple(tuple(pair) for pair in action["add_edges"]),
            replace_outgoing_from=tuple(action["replace_outgoing_from"]),
            replace_incoming_to=tuple(action["replace_incoming_to"]),
            expected_conflict_edges=tuple(
                tuple(pair) for pair in action["expected_conflict_edges"]
            ),
        )
        compiled = compile_action(graph, graph_action)
        row["n1"] = {
            "action_id": graph_action.action_id,
            "action_sha256": compiled.action_sha256,
            "before_graph_sha256": compiled.before_graph_sha256,
            "after_graph_sha256": compiled.after_graph_sha256,
            "removed_edges": [list(pair) for pair in compiled.removed_edges],
        }


def run(
    fragments_dir: Path,
    output: Path,
    receipt: Path,
    repo_root: Path,
    preregistration: Path,
    shared_contract: Path,
) -> dict[str, Any]:
    if sha256(shared_contract) != EXPECTED_SHARED_CONTRACT_SHA256:
        raise ArmAError("shared contract SHA-256 mismatch")
    sys.path.insert(0, str(repo_root / "src"))
    from celltrack_alpha.kaggle_calibration import (  # pylint: disable=import-outside-toplevel
        read_submission,
    )

    paths = sorted(fragments_dir.glob("train__44b6_*.csv"))
    if not paths:
        raise ArmAError("no 44b6 fragment graphs found")
    forbidden = sorted(fragments_dir.glob("train__6bba_*.csv"))
    input_rows: list[dict[str, Any]] = []
    graphs: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)
    for path in paths:
        parsed = read_submission(path)
        if len(parsed) != 1:
            raise ArmAError(f"fragment did not contain exactly one graph: {path}")
        graph = next(iter(parsed.values()))
        if not str(graph.dataset).startswith("44b6_"):
            raise ArmAError(f"non-44b6 graph entered prediction stage: {graph.dataset}")
        identity = f"train:{graph.dataset}"
        graphs[graph.dataset] = graph
        candidates, graph_counts = candidates_for_graph(graph, identity)
        all_candidates.extend(candidates)
        for key, value in graph_counts.items():
            counters[key] += value
        input_rows.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "dataset": graph.dataset,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "eligible_candidates": len(candidates),
            }
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in all_candidates:
        key = str(row["candidate_key"])
        if key in unique:
            raise ArmAError(f"duplicate candidate key before suppression: {key}")
        unique[key] = row

    ranked = sorted(
        unique.values(),
        key=lambda row: (
            float(row["features"]["score"]),
            int(row["observed_parent_node"]),
            tuple(row["observed_branch_nodes"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    used_datasets: set[str] = set()
    used_nodes: set[tuple[str, int]] = set()
    for row in ranked:
        dataset = str(row["dataset"])
        occupied = {
            (dataset, int(row["observed_parent_node"])),
            *((dataset, int(value)) for value in row["observed_branch_nodes"]),
        }
        if dataset in used_datasets or occupied & used_nodes:
            continue
        selected.append(row)
        used_datasets.add(dataset)
        used_nodes.update(occupied)
    if len(selected) > len(paths) * PER_VIDEO_CAP:
        raise ArmAError("per-video selection cap violated")
    compile_selected(repo_root, graphs, selected)

    selected_keys = [str(row["candidate_key"]) for row in selected]
    duplicate_survivors = len(selected_keys) - len(set(selected_keys))
    if duplicate_survivors:
        raise ArmAError("FAIL_CLOSED_DUPLICATE_FORK")
    output_value = {
        "schema": "celltrack-alpha.division-realizer-predictions.v1",
        "arm": "A",
        "name": "geometry_motion",
        "status": "SEALED_BEFORE_EVALUATION",
        "family": "44b6",
        "candidate_universe": {
            "fragment_graph_count": len(paths),
            "non_target_family_files_ignored": len(forbidden),
            **dict(sorted(counters.items())),
            "eligible_unique": len(unique),
            "selected": len(selected),
        },
        "suppression": {
            "per_video_cap": PER_VIDEO_CAP,
            "occupied_node_ownership": True,
            "duplicate_survivors": duplicate_survivors,
            "verdict": "PASS_ZERO_DUPLICATES",
        },
        "predictions": selected,
    }
    atomic_json(output, output_value)
    prediction_sha = sha256(output)
    receipt_value = {
        "schema": "celltrack-alpha.division-realizer-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "prediction_path": str(output),
        "prediction_sha256": prediction_sha,
        "prediction_bytes": output.stat().st_size,
        "preregistration_path": str(preregistration),
        "preregistration_sha256": sha256(preregistration),
        "prediction_program_sha256": sha256(Path(__file__).resolve()),
        "shared_contract_sha256": sha256(shared_contract),
        "input_fragments": input_rows,
        "information_firewall": {
            "target_read": False,
            "oracle_read": False,
            "gt_bundle_read": False,
            "scorer_outcome_read": False,
            "program_has_target_argument": False,
        },
        "duplicate_survivors": duplicate_survivors,
    }
    atomic_json(receipt, receipt_value)
    return receipt_value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--shared-contract", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(
        args.fragments_dir.resolve(),
        args.output.resolve(),
        args.receipt.resolve(),
        args.repo_root.resolve(),
        args.preregistration.resolve(),
        args.shared_contract.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
