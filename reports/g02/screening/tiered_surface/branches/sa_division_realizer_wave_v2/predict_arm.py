#!/usr/bin/env python3
"""Generate one sealed, label-blind full-population division-arm prediction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


SCALE = (1.625, 0.40625, 0.40625)
FAMILY_COUNTS = {"44b6": 21, "6bba": 36}
APPEARANCE_CANDIDATE_CAP = 8


class PredictionError(RuntimeError):
    """Fail-closed prediction or resource error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def verify_freeze(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = freeze["programs"].get(Path(__file__).name)
    observed = sha256_file(Path(__file__).resolve())
    if expected != observed:
        raise PredictionError("prediction program does not match implementation freeze")
    return freeze


def swapout_mib() -> float:
    completed = subprocess.run(
        ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
    )
    page_size = 4096
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if "page size of" in line:
            page_size = int(
                line.split("page size of", 1)[1].split("bytes", 1)[0].strip()
            )
        if ":" in line:
            key, raw = line.split(":", 1)
            token = raw.strip().rstrip(".")
            if token.isdigit():
                values[key.strip()] = int(token)
    if "Swapouts" not in values:
        raise PredictionError("vm_stat did not expose cumulative Swapouts")
    return values["Swapouts"] * page_size / (1024.0 * 1024.0)


def family_from_dataset(dataset: str) -> str:
    family = dataset.split("_", 1)[0]
    if family not in FAMILY_COUNTS:
        raise PredictionError(f"unexpected family: {dataset}")
    return family


def read_graph(path: Path) -> dict[str, Any]:
    nodes: dict[int, dict[str, Any]] = {}
    edges: set[tuple[int, int]] = set()
    dataset: str | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if dataset is None:
                dataset = str(row["dataset"])
            elif dataset != str(row["dataset"]):
                raise PredictionError(f"multi-dataset fragment: {path}")
            if row["row_type"] == "node":
                node_id = int(row["node_id"])
                if node_id in nodes:
                    raise PredictionError(f"duplicate node: {path}:{node_id}")
                nodes[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }
            elif row["row_type"] == "edge":
                edge = (int(row["source_id"]), int(row["target_id"]))
                if edge in edges:
                    raise PredictionError(f"duplicate edge: {path}:{edge}")
                edges.add(edge)
            else:
                raise PredictionError(f"unknown row type: {row['row_type']}")
    if dataset is None:
        raise PredictionError(f"empty fragment: {path}")
    if any(source not in nodes or child not in nodes for source, child in edges):
        raise PredictionError(f"edge references missing node: {path}")
    return {
        "dataset": dataset,
        "family": family_from_dataset(dataset),
        "identity": f"train:{dataset}",
        "nodes": nodes,
        "edges": edges,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory(fragment_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(fragment_dir.glob("train__*.csv"))
    graphs = [read_graph(path) for path in paths]
    counts = {
        family: sum(graph["family"] == family for graph in graphs)
        for family in FAMILY_COUNTS
    }
    if counts != FAMILY_COUNTS:
        raise PredictionError(f"fragment inventory mismatch: {counts}")
    if len(graphs) != sum(FAMILY_COUNTS.values()):
        raise PredictionError("fragment inventory total mismatch")
    return graphs


def adjacency(
    edges: Iterable[tuple[int, int]]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for source, child in sorted(edges):
        outgoing[source].append(child)
        incoming[child].append(source)
    return dict(outgoing), dict(incoming)


def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.sqrt(
        sum(
            (
                (float(left[axis]) - float(right[axis]))
                * SCALE[index]
            )
            ** 2
            for index, axis in enumerate(("z", "y", "x"))
        )
    )


def coord_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum(
            ((float(a) - float(b)) * SCALE[index]) ** 2
            for index, (a, b) in enumerate(zip(left, right, strict=True))
        )
    )


def p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values if value > 0.0)
    if not ordered:
        raise PredictionError("no positive one-frame edge lengths")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def stable_key(identity: str, parent: int, branches: Sequence[int]) -> str:
    encoded = json.dumps(
        [identity, int(parent), sorted(int(value) for value in branches)],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def incident_conflicts(
    edges: set[tuple[int, int]], parent: int, branches: Sequence[int]
) -> list[list[int]]:
    branch_set = set(int(value) for value in branches)
    return [
        [source, child]
        for source, child in sorted(edges)
        if source == int(parent) or child in branch_set
    ]


def direct_action(
    edges: set[tuple[int, int]], parent: int, branches: Sequence[int]
) -> dict[str, Any]:
    branches = sorted(int(value) for value in branches)
    return {
        "kind": "division_fork",
        "add_edges": [[int(parent), branch] for branch in branches],
        "replace_outgoing_from": [int(parent)],
        "replace_incoming_to": branches,
        "expected_conflict_edges": incident_conflicts(edges, parent, branches),
    }


def base_row(
    graph: Mapping[str, Any],
    parent: int,
    branches: Sequence[int],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_key": stable_key(graph["identity"], parent, branches),
        "identity": graph["identity"],
        "dataset": graph["dataset"],
        "family": graph["family"],
        "observed_parent_node": int(parent),
        "observed_branch_nodes": sorted(int(value) for value in branches),
        "event_equivalence_key": stable_key(graph["identity"], parent, branches),
        "action": dict(action),
    }


def geometry_candidates(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing, incoming = adjacency(edges)
    by_time: dict[int, list[dict[str, Any]]] = defaultdict(list)
    births: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node_id, node in nodes.items():
        by_time[int(node["t"])].append(node)
        if not incoming.get(node_id):
            births[int(node["t"])].append(node)
    counts = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for parent_id in sorted(outgoing):
        children = sorted(set(outgoing[parent_id]))
        predecessors = sorted(set(incoming.get(parent_id, ())))
        if len(children) != 1 or len(predecessors) != 1:
            continue
        parent = nodes[parent_id]
        primary = nodes[children[0]]
        previous = nodes[predecessors[0]]
        if not (
            int(primary["t"]) == int(parent["t"]) + 1
            and int(parent["t"]) == int(previous["t"]) + 1
        ):
            continue
        primary_um = distance(parent, primary)
        if primary_um > 7.65:
            continue
        predicted = tuple(
            2.0 * float(parent[axis]) - float(previous[axis])
            for axis in ("z", "y", "x")
        )
        for secondary in births.get(int(parent["t"]) + 1, ()):
            if int(secondary["node_id"]) == int(primary["node_id"]):
                continue
            counts["raw_parent_secondary_pairs"] += 1
            secondary_um = distance(parent, secondary)
            sister_um = distance(primary, secondary)
            symmetry_um = abs(primary_um - secondary_um)
            midpoint = tuple(
                0.5 * (float(primary[axis]) + float(secondary[axis]))
                for axis in ("z", "y", "x")
            )
            midpoint_residual = coord_distance(midpoint, predicted)
            primary_residual = coord_distance(
                tuple(float(primary[axis]) for axis in ("z", "y", "x")),
                predicted,
            )
            secondary_residual = coord_distance(
                tuple(float(secondary[axis]) for axis in ("z", "y", "x")),
                predicted,
            )
            motion_gain = min(primary_residual, secondary_residual) - midpoint_residual
            if not (
                secondary_um <= 4.66
                and 3.0 <= sister_um <= 8.5
                and symmetry_um <= 2.0
                and midpoint_residual <= 2.0
            ):
                continue
            counts["hard_geometry_pass"] += 1
            current_count = sum(
                distance(parent, node) <= 8.5
                for node in by_time[int(parent["t"])]
            )
            next_count = sum(
                coord_distance(
                    tuple(float(node[axis]) for axis in ("z", "y", "x")),
                    predicted,
                )
                <= 8.5
                for node in by_time[int(parent["t"]) + 1]
            )
            if next_count - current_count != 1:
                continue
            counts["count_pass"] += 1
            if motion_gain < 1.0:
                continue
            counts["motion_pass"] += 1
            score = (
                secondary_um
                + 0.15 * sister_um
                + midpoint_residual
                + 0.5 * symmetry_um
                - 0.5 * motion_gain
            )
            if score > 3.25:
                continue
            counts["score_pass"] += 1
            branches = sorted((int(primary["node_id"]), int(secondary["node_id"])))
            row = base_row(
                graph, parent_id, branches, direct_action(edges, parent_id, branches)
            )
            row["features"] = {
                "primary_parent_um": primary_um,
                "secondary_parent_um": secondary_um,
                "sister_um": sister_um,
                "symmetry_um": symmetry_um,
                "midpoint_motion_residual_um": midpoint_residual,
                "fork_motion_gain_um": motion_gain,
                "local_count_change": next_count - current_count,
                "score": score,
            }
            rows.append(row)
    rows.sort(key=lambda row: (float(row["features"]["score"]), row["candidate_key"]))
    return rows, dict(counts)


def persistence_candidates(
    graph: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing, incoming = adjacency(edges)
    radius = p95(
        [
            distance(nodes[source], nodes[child])
            for source, child in edges
            if int(nodes[child]["t"]) == int(nodes[source]["t"]) + 1
        ]
    )
    eligible_by_time: dict[int, list[int]] = defaultdict(list)
    for child2 in sorted(nodes):
        parents2 = incoming.get(child2, [])
        grandchildren2 = outgoing.get(child2, [])
        if len(parents2) != 1 or len(grandchildren2) != 1:
            continue
        grandchild2 = grandchildren2[0]
        successors2 = outgoing.get(grandchild2, [])
        if (
            len(successors2) == 1
            and len(incoming.get(grandchild2, [])) == 1
            and len(outgoing.get(parents2[0], [])) == 1
            and int(nodes[grandchild2]["t"]) == int(nodes[child2]["t"]) + 1
            and int(nodes[successors2[0]]["t"]) == int(nodes[grandchild2]["t"]) + 1
        ):
            eligible_by_time[int(nodes[child2]["t"])].append(child2)
    spatial: dict[int, dict[tuple[int, int, int], list[int]]] = {}
    for frame, child_ids in eligible_by_time.items():
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for child_id in child_ids:
            bucket = tuple(
                math.floor(float(nodes[child_id][axis]) * SCALE[index] / radius)
                for index, axis in enumerate(("z", "y", "x"))
            )
            buckets[bucket].append(child_id)
        spatial[frame] = dict(buckets)
    raw: list[dict[str, Any]] = []
    for parent in sorted(nodes):
        children1 = outgoing.get(parent, [])
        if len(children1) != 1:
            continue
        child1 = children1[0]
        grandchildren1 = outgoing.get(child1, [])
        if (
            len(incoming.get(child1, [])) != 1
            or len(grandchildren1) != 1
        ):
            continue
        grandchild1 = grandchildren1[0]
        successors1 = outgoing.get(grandchild1, [])
        if (
            len(successors1) != 1
            or len(incoming.get(grandchild1, [])) != 1
            or int(nodes[child1]["t"]) != int(nodes[parent]["t"]) + 1
            or int(nodes[grandchild1]["t"]) != int(nodes[child1]["t"]) + 1
            or int(nodes[successors1[0]]["t"]) != int(nodes[grandchild1]["t"]) + 1
        ):
            continue
        parent_bucket = tuple(
            math.floor(float(nodes[parent][axis]) * SCALE[index] / radius)
            for index, axis in enumerate(("z", "y", "x"))
        )
        local_child2: list[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    local_child2.extend(
                        spatial.get(int(nodes[child1]["t"]), {}).get(
                            (
                                parent_bucket[0] + dz,
                                parent_bucket[1] + dy,
                                parent_bucket[2] + dx,
                            ),
                            (),
                        )
                    )
        for child2 in sorted(local_child2):
            if child2 == child1:
                continue
            parent2 = incoming[child2][0]
            grandchild2 = outgoing[child2][0]
            successor2 = outgoing[grandchild2][0]
            occupied = {
                parent,
                child1,
                child2,
                grandchild1,
                grandchild2,
                successors1[0],
                successor2,
            }
            if len(occupied) != 7 or parent2 == parent:
                continue
            distances = (
                distance(nodes[parent], nodes[child2]),
                distance(nodes[child1], nodes[child2]),
                distance(nodes[child1], nodes[grandchild2]),
            )
            if any(value > radius for value in distances):
                continue
            cost = sum(distances) / radius
            branches = sorted((child1, child2))
            late_children = sorted((grandchild1, grandchild2))
            conflicts = [
                [source, child]
                for source, child in sorted(edges)
                if source == child1 or child in set(late_children)
            ]
            row = base_row(
                graph,
                parent,
                branches,
                {
                    "kind": "division_fork",
                    "add_edges": [[child1, child] for child in late_children],
                    "replace_outgoing_from": [child1],
                    "replace_incoming_to": late_children,
                    "expected_conflict_edges": conflicts,
                },
            )
            row.update(
                {
                    "observed_sibling_parent": parent2,
                    "observed_branch_terminals": late_children,
                    "observed_survival_nodes": sorted((successors1[0], successor2)),
                    "occupied_nodes": sorted(occupied),
                    "features": {
                        "branch_survival_depth": 3,
                        "normalized_locality_cost": cost,
                        "radius_um": radius,
                    },
                    "_parent": parent,
                    "_child2": child2,
                }
            )
            raw.append(row)
    costs_by_parent: dict[int, list[float]] = defaultdict(list)
    costs_by_child2: dict[int, list[float]] = defaultdict(list)
    for row in raw:
        cost = float(row["features"]["normalized_locality_cost"])
        costs_by_parent[int(row["_parent"])].append(cost)
        costs_by_child2[int(row["_child2"])].append(cost)
    selected: list[dict[str, Any]] = []
    for row in raw:
        cost = float(row["features"]["normalized_locality_cost"])
        parent_costs = sorted(costs_by_parent[int(row["_parent"])])
        child_costs = sorted(costs_by_child2[int(row["_child2"])])
        if cost != parent_costs[0] or cost != child_costs[0]:
            continue
        parent_runner = parent_costs[1] if len(parent_costs) > 1 else math.inf
        child_runner = child_costs[1] if len(child_costs) > 1 else math.inf
        margin = min(parent_runner, child_runner) / max(cost, 1.0e-12)
        if margin < 1.25:
            continue
        row["features"]["mutual_best_margin"] = margin if math.isfinite(margin) else None
        row["_margin"] = margin
        selected.append(row)
    selected.sort(
        key=lambda row: (
            -float(row["_margin"]),
            float(row["features"]["normalized_locality_cost"]),
            row["candidate_key"],
        )
    )
    clean = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in selected
    ]
    return clean, {
        "radius_um": radius,
        "raw_eligible_candidates": len(raw),
        "mutual_unique_candidates": len(clean),
    }


def appearance_prefilter(
    graph: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing, incoming = adjacency(edges)
    radius = p95(
        [
            distance(nodes[source], nodes[child])
            for source, child in edges
            if int(nodes[child]["t"]) == int(nodes[source]["t"]) + 1
        ]
    )
    births: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        if not incoming.get(node_id):
            births[int(node["t"])].append(node_id)
    rows: list[dict[str, Any]] = []
    for parent in sorted(nodes):
        children = outgoing.get(parent, [])
        if len(children) != 1 or len(incoming.get(parent, [])) != 1:
            continue
        primary = children[0]
        if int(nodes[primary]["t"]) != int(nodes[parent]["t"]) + 1:
            continue
        primary_distance = distance(nodes[parent], nodes[primary])
        if primary_distance > radius:
            continue
        for secondary in births.get(int(nodes[parent]["t"]) + 1, ()):
            if secondary == primary:
                continue
            secondary_distance = distance(nodes[parent], nodes[secondary])
            sister_distance = distance(nodes[primary], nodes[secondary])
            if secondary_distance > radius or sister_distance > 1.5 * radius:
                continue
            branches = sorted((primary, secondary))
            row = base_row(
                graph, parent, branches, direct_action(edges, parent, branches)
            )
            row["patch_nodes"] = {
                "parent": parent,
                "daughter_1": branches[0],
                "daughter_2": branches[1],
            }
            row["features"] = {
                "geometry_cost": (
                    primary_distance + secondary_distance + sister_distance
                )
                / radius,
                "radius_um": radius,
            }
            rows.append(row)
    rows.sort(key=lambda row: (float(row["features"]["geometry_cost"]), row["candidate_key"]))
    return rows[:APPEARANCE_CANDIDATE_CAP], {
        "radius_um": radius,
        "raw_candidates": len(rows),
        "cached_candidates": min(len(rows), APPEARANCE_CANDIDATE_CAP),
    }


def open_raw_array(raw_archive: Path, dataset: str) -> Any:
    fsspec_spec = importlib.util.find_spec("fsspec")
    zarr_spec = importlib.util.find_spec("zarr")
    if fsspec_spec is None or zarr_spec is None:
        raise PredictionError("fsspec/zarr unavailable for bounded appearance cache")
    import fsspec  # pylint: disable=import-outside-toplevel
    import zarr  # pylint: disable=import-outside-toplevel

    mapper = fsspec.get_mapper(f"zip://train/{dataset}.zarr/0::{raw_archive}")
    return zarr.open_array(mapper, mode="r", zarr_format=3)


def patch_summary(array: Any, node: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    t = int(node["t"])
    z, y, x = (int(round(float(node[axis]))) for axis in ("z", "y", "x"))
    slices = (
        slice(max(0, z - 2), min(int(array.shape[1]), z + 3)),
        slice(max(0, y - 12), min(int(array.shape[2]), y + 13)),
        slice(max(0, x - 12), min(int(array.shape[3]), x + 13)),
    )
    patch = np.asarray(array[(t, *slices)], dtype=np.float32)
    if patch.size == 0:
        raise PredictionError("empty appearance patch")
    background = float(np.percentile(patch, 20.0))
    weights = np.maximum(patch - background, 0.0)
    mass = float(weights.sum())
    if not math.isfinite(mass) or mass <= 0.0:
        roundness = 0.0
    else:
        projected = weights.sum(axis=0)
        yy, xx = np.indices(projected.shape, dtype=np.float64)
        total = float(projected.sum())
        cy = float((projected * yy).sum() / total)
        cx = float((projected * xx).sum() / total)
        dy = yy - cy
        dx = xx - cx
        covariance = np.array(
            [
                [
                    float((projected * dy * dy).sum() / total),
                    float((projected * dy * dx).sum() / total),
                ],
                [
                    float((projected * dy * dx).sum() / total),
                    float((projected * dx * dx).sum() / total),
                ],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        roundness = float(eigenvalues[0] / max(eigenvalues[-1], 1.0e-12))
    return {
        "background_p20": background,
        "mass_above_background": mass,
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "voxel_count": int(patch.size),
        "xy_roundness": roundness,
    }


def cache_and_select_appearance(
    graph: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    raw_archive: Path,
    cache_dir: Path,
    prereg_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = cache_dir / f"{graph['dataset']}.json"
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (
            cached.get("fragment_sha256") != graph["sha256"]
            or cached.get("preregistration_sha256") != prereg_sha256
        ):
            raise PredictionError(f"appearance checkpoint drift: {checkpoint}")
    else:
        array = open_raw_array(raw_archive, graph["dataset"])
        cached_rows: list[dict[str, Any]] = []
        for row in candidates:
            summaries = {
                role: patch_summary(array, graph["nodes"][int(node_id)])
                for role, node_id in row["patch_nodes"].items()
            }
            record = dict(row)
            record["intensity_summaries"] = summaries
            cached_rows.append(record)
        cached = {
            "schema": "celltrack-alpha.bounded-appearance-cache.v1",
            "dataset": graph["dataset"],
            "fragment_sha256": graph["sha256"],
            "preregistration_sha256": prereg_sha256,
            "candidate_cap": APPEARANCE_CANDIDATE_CAP,
            "contains_voxels": False,
            "contains_labels": False,
            "candidates": cached_rows,
        }
        atomic_json(checkpoint, cached)
    qualified: list[dict[str, Any]] = []
    for row in cached["candidates"]:
        summaries = row["intensity_summaries"]
        parent_mass = float(summaries["parent"]["mass_above_background"])
        daughter_masses = [
            float(summaries["daughter_1"]["mass_above_background"]),
            float(summaries["daughter_2"]["mass_above_background"]),
        ]
        ratio = sum(daughter_masses) / max(parent_mass, 1.0e-12)
        balance = min(daughter_masses) / max(max(daughter_masses), 1.0e-12)
        parent_roundness = float(summaries["parent"]["xy_roundness"])
        if not (
            0.6 <= ratio <= 1.6
            and balance >= 0.35
            and parent_roundness >= 0.4
        ):
            continue
        record = dict(row)
        record["features"] = {
            **record["features"],
            "daughter_parent_mass_ratio": ratio,
            "daughter_mass_balance": balance,
            "parent_xy_roundness": parent_roundness,
            "appearance_score": abs(math.log(ratio)) + (1.0 - balance),
        }
        qualified.append(record)
    qualified.sort(
        key=lambda row: (
            float(row["features"]["appearance_score"]),
            float(row["features"]["geometry_cost"]),
            row["candidate_key"],
        )
    )
    return qualified[:1], {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cached_candidates": len(cached["candidates"]),
        "qualified_candidates": len(qualified),
        "contains_voxels": False,
        "contains_labels": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.with_suffix(".receipt.json").exists():
        raise PredictionError("refusing to overwrite sealed prediction")
    freeze = verify_freeze(args.implementation_freeze.resolve())
    preregistration = args.preregistration.resolve()
    prereg_sha256 = sha256_file(preregistration)
    expected_prereg = freeze["preregistrations"].get(args.arm)
    if expected_prereg != prereg_sha256:
        raise PredictionError("preregistration does not match implementation freeze")
    graphs = inventory(args.fragments.resolve())
    swap_before = swapout_mib()
    started = time.monotonic()
    time.sleep(args.preflight_seconds)
    swap_after_idle = swapout_mib()
    if swap_after_idle != swap_before:
        raise PredictionError(
            f"zero-swap idle preflight failed: {swap_before}->{swap_after_idle}"
        )
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for graph in graphs:
        if args.arm == "A":
            rows, detail = geometry_candidates(graph)
            chosen = rows[:1]
        elif args.arm == "B":
            rows, detail = persistence_candidates(graph)
            chosen = rows[:1]
        elif args.arm == "C":
            rows, detail = appearance_prefilter(graph)
            chosen, cache_detail = cache_and_select_appearance(
                graph,
                rows,
                args.raw_archive.resolve(),
                args.cache_dir.resolve(),
                prereg_sha256,
            )
            detail = {**detail, **cache_detail}
        else:
            raise PredictionError(f"unsupported arm: {args.arm}")
        selected.extend(chosen)
        diagnostics.append(
            {
                "identity": graph["identity"],
                "family": graph["family"],
                "fragment_bytes": graph["bytes"],
                "fragment_sha256": graph["sha256"],
                "selected": len(chosen),
                **detail,
            }
        )
        if swapout_mib() != swap_before:
            raise PredictionError(f"zero-swap guard tripped after {graph['identity']}")
    candidate_keys = [row["candidate_key"] for row in selected]
    equivalence_keys = [row["event_equivalence_key"] for row in selected]
    if (
        len(candidate_keys) != len(set(candidate_keys))
        or len(equivalence_keys) != len(set(equivalence_keys))
    ):
        raise PredictionError("FAIL_CLOSED_DUPLICATE_FORK")
    per_video: dict[str, int] = defaultdict(int)
    for row in selected:
        per_video[row["identity"]] += 1
    if any(count > 1 for count in per_video.values()):
        raise PredictionError("per-video precision cap violated")
    swap_after = swapout_mib()
    payload = {
        "schema": "celltrack-alpha.division-realizer-predictions.v2",
        "status": "SEALED_BEFORE_EVALUATION",
        "arm": args.arm,
        "scope": {
            "families": FAMILY_COUNTS,
            "video_count": len(graphs),
            "label_or_bundle_read": False,
            "official_scorer_read": False,
            "other_arm_read": False,
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": prereg_sha256,
        },
        "implementation_freeze_sha256": sha256_file(
            args.implementation_freeze.resolve()
        ),
        "selection": {
            "count": len(selected),
            "by_family": {
                family: sum(row["family"] == family for row in selected)
                for family in FAMILY_COUNTS
            },
            "per_video_cap": 1,
            "duplicate_survivors": 0,
            "predictions": selected,
        },
        "diagnostics": diagnostics,
        "resource_guard": {
            "idle_preflight_seconds": args.preflight_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "swapout_mib_before": swap_before,
            "swapout_mib_after_idle": swap_after_idle,
            "swapout_mib_after": swap_after,
            "swap_growth_mib": swap_after - swap_before,
            "passed": swap_after == swap_before,
            "cpu_only": True,
            "kaggle_used": False,
            "gpu_used": False,
        },
        "cache_policy": {
            "bounded_summary_cache": args.arm == "C",
            "full_pool_cache": False,
        },
    }
    atomic_json(args.output.resolve(), payload)
    receipt = {
        "schema": "celltrack-alpha.sealed-prediction-receipt.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "arm": args.arm,
        "path": str(args.output.resolve()),
        "bytes": args.output.resolve().stat().st_size,
        "prediction_sha256": sha256_file(args.output.resolve()),
        "atomic": True,
        "duplicate_survivors": 0,
    }
    atomic_json(args.output.resolve().with_suffix(".receipt.json"), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--implementation-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--preflight-seconds", type=int, default=45)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
