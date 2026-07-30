#!/usr/bin/env python3
"""Generate label-blind D1 late-persistence division candidates on 44b6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


class PredictionError(RuntimeError):
    """The frozen persistence prediction contract failed closed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def swapout_mib() -> float:
    """Return cumulative macOS swapouts in MiB, or fail outside the guard."""
    import subprocess

    completed = subprocess.run(
        ["/usr/bin/vm_stat"],
        check=True,
        capture_output=True,
        text=True,
    )
    page_size = 4096
    for line in completed.stdout.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            break
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        token = raw.strip().rstrip(".")
        if token.isdigit():
            values[key.strip()] = int(token)
    if "Swapouts" not in values:
        raise PredictionError("vm_stat did not expose cumulative Swapouts")
    return values["Swapouts"] * page_size / (1024.0 * 1024.0)


def quantile_p95(values: Sequence[float]) -> float:
    if not values:
        raise PredictionError("no one-frame edges exist for locality calibration")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    value = float(ordered[index])
    if not math.isfinite(value) or value <= 0.0:
        raise PredictionError("invalid p95 edge length")
    return value


def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.sqrt(
        sum((float(left[axis]) - float(right[axis])) ** 2 for axis in ("z", "y", "x"))
    )


def read_graph(path: Path) -> tuple[str, dict[int, dict[str, Any]], set[tuple[int, int]]]:
    nodes: dict[int, dict[str, Any]] = {}
    edges: set[tuple[int, int]] = set()
    dataset: str | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_dataset = str(row["dataset"])
            if dataset is None:
                dataset = row_dataset
            elif dataset != row_dataset:
                raise PredictionError(f"multi-dataset fragment: {path}")
            kind = str(row["row_type"])
            if kind == "node":
                node_id = int(row["node_id"])
                if node_id in nodes:
                    raise PredictionError(f"duplicate node {node_id}: {path}")
                nodes[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }
            elif kind == "edge":
                edge = (int(row["source_id"]), int(row["target_id"]))
                if edge in edges:
                    raise PredictionError(f"duplicate edge {edge}: {path}")
                edges.add(edge)
            else:
                raise PredictionError(f"unknown row type {kind}: {path}")
    if dataset is None or not dataset.startswith("44b6_"):
        raise PredictionError(f"not one 44b6 fragment: {path}")
    if any(source not in nodes or target not in nodes for source, target in edges):
        raise PredictionError(f"edge references missing node: {path}")
    return f"train:{dataset}", nodes, edges


def adjacency(
    edges: Iterable[tuple[int, int]]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for source, target in sorted(edges):
        outgoing.setdefault(source, []).append(target)
        incoming.setdefault(target, []).append(source)
    return outgoing, incoming


def candidate_key(identity: str, parent: int, terminals: Sequence[int]) -> str:
    record = [identity, int(parent), sorted(int(value) for value in terminals)]
    return sha256_bytes(json.dumps(record, separators=(",", ":"), sort_keys=False).encode("ascii"))


def enumerate_video(
    identity: str,
    nodes: Mapping[int, Mapping[str, Any]],
    edges: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outgoing, incoming = adjacency(edges)
    one_frame_lengths = [
        distance(nodes[source], nodes[target])
        for source, target in edges
        if int(nodes[target]["t"]) - int(nodes[source]["t"]) == 1
    ]
    radius = quantile_p95(one_frame_lengths)
    raw: list[dict[str, Any]] = []
    for parent in sorted(nodes):
        children1 = outgoing.get(parent, [])
        if len(children1) != 1:
            continue
        child1 = children1[0]
        if len(incoming.get(child1, [])) != 1:
            continue
        grandchildren1 = outgoing.get(child1, [])
        if len(grandchildren1) != 1:
            continue
        grandchild1 = grandchildren1[0]
        successors1 = outgoing.get(grandchild1, [])
        if len(successors1) != 1 or len(incoming.get(grandchild1, [])) != 1:
            continue
        successor1 = successors1[0]
        if not (
            int(nodes[child1]["t"]) == int(nodes[parent]["t"]) + 1
            and int(nodes[grandchild1]["t"]) == int(nodes[child1]["t"]) + 1
            and int(nodes[successor1]["t"]) == int(nodes[grandchild1]["t"]) + 1
        ):
            continue
        for child2 in sorted(nodes):
            if child2 == child1 or int(nodes[child2]["t"]) != int(nodes[child1]["t"]):
                continue
            parents2 = incoming.get(child2, [])
            grandchildren2 = outgoing.get(child2, [])
            if len(parents2) != 1 or len(grandchildren2) != 1 or parents2[0] == parent:
                continue
            parent2 = parents2[0]
            grandchild2 = grandchildren2[0]
            successors2 = outgoing.get(grandchild2, [])
            if (
                len(successors2) != 1
                or len(incoming.get(grandchild2, [])) != 1
                or len(outgoing.get(parent2, [])) != 1
            ):
                continue
            successor2 = successors2[0]
            occupied = {parent, child1, child2, grandchild1, grandchild2, successor1, successor2}
            if len(occupied) != 7:
                continue
            if not (
                int(nodes[grandchild2]["t"]) == int(nodes[child2]["t"]) + 1
                and int(nodes[successor2]["t"]) == int(nodes[grandchild2]["t"]) + 1
            ):
                continue
            distances = (
                distance(nodes[parent], nodes[child2]),
                distance(nodes[child1], nodes[child2]),
                distance(nodes[child1], nodes[grandchild2]),
            )
            if any(value > radius for value in distances):
                continue
            normalized_cost = sum(distances) / radius
            key = candidate_key(identity, parent, (grandchild1, grandchild2))
            owned_conflicts = sorted(
                set((child1, target) for target in outgoing.get(child1, []))
                | set((source, grandchild1) for source in incoming.get(grandchild1, []))
                | set((source, grandchild2) for source in incoming.get(grandchild2, []))
            )
            raw.append(
                {
                    "identity": identity,
                    "candidate_key": key,
                    "observed_parent": parent,
                    "observed_sibling_parent": parent2,
                    "observed_daughters": [child1, child2],
                    "observed_branch_terminals": sorted([grandchild1, grandchild2]),
                    "observed_survival_nodes": sorted([successor1, successor2]),
                    "late_parent": child1,
                    "late_children": sorted([grandchild1, grandchild2]),
                    "add_edges": sorted([[child1, grandchild1], [child1, grandchild2]]),
                    "expected_conflict_edges": [list(edge) for edge in owned_conflicts],
                    "occupied_nodes": sorted(occupied),
                    "radius": radius,
                    "normalized_locality_cost": normalized_cost,
                    "locality_distances": list(distances),
                    "branch_survival_depth": 3,
                    "_pair_parent": parent,
                    "_pair_child2": child2,
                }
            )
    costs_by_parent: dict[int, list[float]] = {}
    costs_by_child2: dict[int, list[float]] = {}
    for row in raw:
        costs_by_parent.setdefault(int(row["_pair_parent"]), []).append(float(row["normalized_locality_cost"]))
        costs_by_child2.setdefault(int(row["_pair_child2"]), []).append(float(row["normalized_locality_cost"]))
    selected_raw: list[dict[str, Any]] = []
    for row in raw:
        cost = float(row["normalized_locality_cost"])
        parent_costs = sorted(costs_by_parent[int(row["_pair_parent"])])
        child_costs = sorted(costs_by_child2[int(row["_pair_child2"])])
        if cost != parent_costs[0] or cost != child_costs[0]:
            continue
        parent_runner = parent_costs[1] if len(parent_costs) > 1 else math.inf
        child_runner = child_costs[1] if len(child_costs) > 1 else math.inf
        parent_margin = parent_runner / max(cost, 1.0e-12)
        child_margin = child_runner / max(cost, 1.0e-12)
        if min(parent_margin, child_margin) < 1.25:
            continue
        row["mutual_best_margin"] = min(parent_margin, child_margin)
        selected_raw.append(row)
    selected_raw.sort(
        key=lambda row: (
            -int(row["branch_survival_depth"]),
            -float(row["mutual_best_margin"]),
            float(row["normalized_locality_cost"]),
            str(row["candidate_key"]),
        )
    )
    accepted: list[dict[str, Any]] = []
    occupied: set[int] = set()
    keys: set[str] = set()
    for row in selected_raw:
        row_nodes = set(int(value) for value in row["occupied_nodes"])
        if str(row["candidate_key"]) in keys or occupied.intersection(row_nodes):
            continue
        keys.add(str(row["candidate_key"]))
        occupied.update(row_nodes)
        accepted.append({key: value for key, value in row.items() if not key.startswith("_")})
    duplicate_keys = len(accepted) - len({str(row["candidate_key"]) for row in accepted})
    node_claims = [int(node) for row in accepted for node in row["occupied_nodes"]]
    duplicate_nodes = len(node_claims) - len(set(node_claims))
    if duplicate_keys or duplicate_nodes:
        raise PredictionError("FAIL_CLOSED_DUPLICATE_FORK")
    return accepted, {
        "identity": identity,
        "nodes": len(nodes),
        "edges": len(edges),
        "p95_one_frame_edge_length": radius,
        "raw_eligible_candidates": len(raw),
        "mutual_unique_candidates": len(selected_raw),
        "accepted_candidates": len(accepted),
        "duplicate_survivors": 0,
    }


def generate(fragment_dir: Path, preregistration: Path) -> dict[str, Any]:
    prereg_sha = sha256_file(preregistration)
    paths = sorted(fragment_dir.glob("train__44b6_*.csv"))
    if len(paths) != 21:
        raise PredictionError(f"expected 21 44b6 fragments, found {len(paths)}")
    forbidden = sorted(fragment_dir.glob("train__6bba_*.csv"))
    if len(forbidden) != 36:
        raise PredictionError(
            f"authenticated family inventory drift: expected 36 forbidden 6bba fragments, "
            f"found {len(forbidden)}"
        )
    predictions: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    for path in paths:
        identity, nodes, edges = read_graph(path)
        rows, summary = enumerate_video(identity, nodes, edges)
        predictions.extend(rows)
        videos.append(
            {
                **summary,
                "fragment_path": str(path),
                "fragment_sha256": sha256_file(path),
            }
        )
    prediction_keys = [str(row["candidate_key"]) for row in predictions]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise PredictionError("FAIL_CLOSED_DUPLICATE_FORK")
    return {
        "schema": "celltrack-alpha.division-realizer-predictions.v1",
        "arm": "B",
        "rule": "D1_LATE_PERSISTENT_ONE_TO_ONE_FORK",
        "scope": {
            "family": "44b6",
            "video_count": len(paths),
            "target_or_gt_read": False,
            "official_scorer_read": False,
            "other_arm_read": False,
            "dispatch_count_corrected_before_freeze": True
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": prereg_sha,
        },
        "universe": {
            "videos": videos,
            "candidate_count_before_cross_video_suppression": sum(
                int(row["raw_eligible_candidates"]) for row in videos
            ),
        },
        "selection": {
            "count": len(predictions),
            "duplicate_survivors": 0,
            "predictions": predictions,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-seconds", default=45, type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise PredictionError(f"refusing to overwrite sealed output: {args.output}")
    swap_before = swapout_mib()
    idle_started = time.monotonic()
    time.sleep(args.preflight_seconds)
    swap_after_idle = swapout_mib()
    if swap_after_idle != swap_before:
        raise PredictionError(
            f"zero-swap idle preflight failed: {swap_before} -> {swap_after_idle} MiB"
        )
    payload = generate(args.fragments, args.preregistration)
    swap_after_prediction = swapout_mib()
    if swap_after_prediction != swap_before:
        raise PredictionError(
            f"zero-swap prediction guard failed: {swap_before} -> {swap_after_prediction} MiB"
        )
    payload["resource_guard"] = {
        "idle_preflight_seconds": args.preflight_seconds,
        "idle_elapsed_seconds": time.monotonic() - idle_started,
        "swapout_mib_before": swap_before,
        "swapout_mib_after_idle": swap_after_idle,
        "swapout_mib_after_prediction": swap_after_prediction,
        "swap_growth_mib": 0.0,
        "passed": True,
    }
    atomic_write(args.output, canonical_bytes(payload))
    receipt = {
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "atomic": True,
        "sealed_before_target_evaluation": True,
    }
    atomic_write(args.output.with_suffix(".receipt.json"), canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
