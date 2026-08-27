#!/usr/bin/env python3
"""Compare two competition submission graphs without ground truth."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


SUBMISSION_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
SCALE_ZYX_UM = (1.625, 0.40625, 0.40625)
MAX_DISTANCE_UM = 7.0
DENSITY_BINS = (
    ("1-9", 1, 9),
    ("10-24", 10, 24),
    ("25-49", 25, 49),
    ("50-99", 50, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500+", 500, None),
)


class ComparisonError(ValueError):
    """An input cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class Node:
    node_id: int
    t: int
    z: int
    y: int
    x: int

    def physical_coord(self) -> tuple[float, float, float]:
        return (
            self.z * SCALE_ZYX_UM[0],
            self.y * SCALE_ZYX_UM[1],
            self.x * SCALE_ZYX_UM[2],
        )


@dataclass(slots=True)
class Graph:
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: set[tuple[int, int]] = field(default_factory=set)


@dataclass(slots=True)
class SparseStats:
    frames: int = 0
    frames_with_both_sides: int = 0
    frames_with_candidates: int = 0
    candidate_pairs: int = 0
    components: int = 0
    peak_frame_candidates: int = 0
    peak_component_candidates: int = 0
    peak_component_left_nodes: int = 0
    peak_component_right_nodes: int = 0
    peak_component_nodes: int = 0
    peak_assignment_dimension: int = 0

    def merge(self, other: "SparseStats") -> None:
        self.frames += other.frames
        self.frames_with_both_sides += other.frames_with_both_sides
        self.frames_with_candidates += other.frames_with_candidates
        self.candidate_pairs += other.candidate_pairs
        self.components += other.components
        self.peak_frame_candidates = max(
            self.peak_frame_candidates, other.peak_frame_candidates
        )
        self.peak_component_candidates = max(
            self.peak_component_candidates, other.peak_component_candidates
        )
        self.peak_component_left_nodes = max(
            self.peak_component_left_nodes, other.peak_component_left_nodes
        )
        self.peak_component_right_nodes = max(
            self.peak_component_right_nodes, other.peak_component_right_nodes
        )
        self.peak_component_nodes = max(
            self.peak_component_nodes, other.peak_component_nodes
        )
        self.peak_assignment_dimension = max(
            self.peak_assignment_dimension, other.peak_assignment_dimension
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "frames_with_both_sides": self.frames_with_both_sides,
            "frames_with_candidates": self.frames_with_candidates,
            "candidate_pairs": self.candidate_pairs,
            "components": self.components,
            "peak_frame_candidates": self.peak_frame_candidates,
            "peak_component_candidates": self.peak_component_candidates,
            "peak_component_left_nodes": self.peak_component_left_nodes,
            "peak_component_right_nodes": self.peak_component_right_nodes,
            "peak_component_nodes": self.peak_component_nodes,
            "peak_assignment_dimension": self.peak_assignment_dimension,
        }


def _parse_int(row: dict[str, str], field_name: str, location: str) -> int:
    value = row[field_name]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(
            f"{location}: {field_name} is not an integer: {value!r}"
        ) from exc


def resolve_input(spec: str) -> list[Path]:
    """Resolve one CSV, a directory, or a quoted glob to sorted CSV files."""

    raw_path = Path(spec).expanduser()
    matches: list[Path]
    if raw_path.exists():
        matches = [raw_path]
    else:
        matches = [Path(value) for value in glob.glob(spec, recursive=True)]

    files: list[Path] = []
    for match in matches:
        if match.is_dir():
            files.extend(path for path in match.rglob("*.csv") if path.is_file())
        elif match.is_file():
            if match.suffix.lower() != ".csv":
                raise ComparisonError(f"input file is not a CSV: {match}")
            files.append(match)

    unique = sorted({path.resolve() for path in files}, key=lambda path: str(path))
    if not unique:
        raise ComparisonError(f"input matched no CSV files: {spec!r}")
    return unique


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def input_provenance(
    spec: str, paths: Iterable[Path], *, strip_prefix: str | None = None
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in paths:
        file_hash, size = _sha256(path)
        encoded_path = str(path).encode("utf-8")
        combined.update(len(encoded_path).to_bytes(8, "big"))
        combined.update(encoded_path)
        combined.update(bytes.fromhex(file_hash))
        files.append({"path": str(path), "sha256": file_hash, "bytes": size})
    return {
        "input": spec,
        "file_count": len(files),
        "files": files,
        "combined_sha256": combined.hexdigest(),
        "strip_prefix": strip_prefix,
    }


def _read_file(
    path: Path,
    graphs: dict[str, Graph],
    *,
    strip_prefix: str | None,
    dataset_origins: dict[str, str],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
                raise ComparisonError(
                    f"{path}: unexpected header; expected {list(SUBMISSION_COLUMNS)}, "
                    f"got {reader.fieldnames}"
                )
            for row in reader:
                location = f"{path}: row {reader.line_num}"
                if None in row or any(row[name] is None for name in SUBMISSION_COLUMNS):
                    raise ComparisonError(f"{location}: malformed column count")

                raw_dataset = row["dataset"]
                if not raw_dataset or raw_dataset != raw_dataset.strip():
                    raise ComparisonError(
                        f"{location}: invalid dataset {raw_dataset!r}"
                    )
                dataset = raw_dataset
                if strip_prefix is not None and dataset.startswith(strip_prefix):
                    dataset = dataset[len(strip_prefix) :]
                    if not dataset:
                        raise ComparisonError(
                            f"{location}: stripping dataset prefix {strip_prefix!r} "
                            f"from {raw_dataset!r} produces an empty dataset ID"
                        )
                    if dataset != dataset.strip():
                        raise ComparisonError(
                            f"{location}: stripping dataset prefix {strip_prefix!r} "
                            f"from {raw_dataset!r} produces invalid dataset {dataset!r}"
                        )
                previous_origin = dataset_origins.get(dataset)
                if previous_origin is None:
                    dataset_origins[dataset] = raw_dataset
                elif previous_origin != raw_dataset:
                    raise ComparisonError(
                        f"{location}: dataset prefix normalization collision: "
                        f"{previous_origin!r} and {raw_dataset!r} both map to {dataset!r}"
                    )
                row_id = _parse_int(row, "id", location)
                if row_id < 0:
                    raise ComparisonError(f"{location}: id must be non-negative")

                values = {
                    name: _parse_int(row, name, location)
                    for name in (
                        "node_id",
                        "t",
                        "z",
                        "y",
                        "x",
                        "source_id",
                        "target_id",
                    )
                }
                graph = graphs.setdefault(dataset, Graph())
                row_type = row["row_type"]
                if row_type == "node":
                    if values["node_id"] < 0 or min(
                        values["t"], values["z"], values["y"], values["x"]
                    ) < 0:
                        raise ComparisonError(
                            f"{location}: node ID, time, and coordinates must be non-negative"
                        )
                    if values["source_id"] != -1 or values["target_id"] != -1:
                        raise ComparisonError(
                            f"{location}: node row must have source_id=target_id=-1"
                        )
                    node_id = values["node_id"]
                    if node_id in graph.nodes:
                        raise ComparisonError(
                            f"{location}: duplicate node ID {node_id} in dataset {dataset!r}"
                        )
                    graph.nodes[node_id] = Node(
                        node_id=node_id,
                        t=values["t"],
                        z=values["z"],
                        y=values["y"],
                        x=values["x"],
                    )
                elif row_type == "edge":
                    if any(values[name] != -1 for name in ("node_id", "t", "z", "y", "x")):
                        raise ComparisonError(
                            f"{location}: edge row must have node_id=t=z=y=x=-1"
                        )
                    source_id = values["source_id"]
                    target_id = values["target_id"]
                    if source_id < 0 or target_id < 0:
                        raise ComparisonError(
                            f"{location}: edge endpoints must be non-negative"
                        )
                    edge = (source_id, target_id)
                    if edge in graph.edges:
                        raise ComparisonError(
                            f"{location}: duplicate edge {source_id}->{target_id} "
                            f"in dataset {dataset!r}"
                        )
                    graph.edges.add(edge)
                else:
                    raise ComparisonError(
                        f"{location}: unexpected row_type {row_type!r}"
                    )
    except UnicodeDecodeError as exc:
        raise ComparisonError(f"{path}: CSV is not valid UTF-8") from exc
    except csv.Error as exc:
        raise ComparisonError(f"{path}: malformed CSV: {exc}") from exc


def load_graphs(
    paths: Iterable[Path],
    *,
    strict: bool,
    strip_prefix: str | None = None,
) -> dict[str, Graph]:
    graphs: dict[str, Graph] = {}
    dataset_origins: dict[str, str] = {}
    for path in paths:
        _read_file(
            path,
            graphs,
            strip_prefix=strip_prefix,
            dataset_origins=dataset_origins,
        )
    if not graphs:
        raise ComparisonError("input contains no datasets")

    for dataset in sorted(graphs):
        graph = graphs[dataset]
        if not graph.nodes:
            raise ComparisonError(f"dataset {dataset!r} contains no nodes")
        for source_id, target_id in sorted(graph.edges):
            source = graph.nodes.get(source_id)
            target = graph.nodes.get(target_id)
            if source is None or target is None:
                raise ComparisonError(
                    f"dangling edge in dataset {dataset!r}: {source_id}->{target_id}"
                )
            if strict and target.t != source.t + 1:
                raise ComparisonError(
                    f"strict edge validation failed in dataset {dataset!r}: "
                    f"{source_id}(t={source.t})->{target_id}(t={target.t}) "
                    "is backward or nonconsecutive"
                )
    return graphs


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def match_frame(
    left_nodes: Iterable[Node],
    right_nodes: Iterable[Node],
) -> tuple[dict[int, int], list[float], SparseStats]:
    """Return a maximum-cardinality, minimum-distance sparse frame matching."""

    left = sorted(left_nodes, key=lambda node: node.node_id)
    right = sorted(right_nodes, key=lambda node: node.node_id)
    stats = SparseStats(frames=1)
    if left and right:
        stats.frames_with_both_sides = 1
    if not left or not right:
        return {}, [], stats

    left_coords = np.asarray([node.physical_coord() for node in left], dtype=np.float64)
    right_coords = np.asarray([node.physical_coord() for node in right], dtype=np.float64)
    tree = cKDTree(right_coords)
    query_radius = math.nextafter(MAX_DISTANCE_UM, math.inf)
    cutoff_sq = MAX_DISTANCE_UM * MAX_DISTANCE_UM
    candidates: list[tuple[int, int, float]] = []
    union_find = _UnionFind(len(left) + len(right))

    for left_index, coordinate in enumerate(left_coords):
        neighbors = tree.query_ball_point(
            coordinate, r=query_radius, return_sorted=True
        )
        for right_index in neighbors:
            right_int = int(right_index)
            delta = coordinate - right_coords[right_int]
            distance_sq = float(np.dot(delta, delta))
            if distance_sq <= cutoff_sq:
                distance = math.sqrt(distance_sq)
                candidates.append((left_index, right_int, distance))
                union_find.union(left_index, len(left) + right_int)

    stats.candidate_pairs = len(candidates)
    stats.peak_frame_candidates = len(candidates)
    if not candidates:
        return {}, [], stats
    stats.frames_with_candidates = 1

    component_edges: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for candidate in candidates:
        component_edges[union_find.find(candidate[0])].append(candidate)

    ordered_components = sorted(
        component_edges.values(),
        key=lambda edges: (
            min(left[index].node_id for index, _, _ in edges),
            min(right[index].node_id for _, index, _ in edges),
        ),
    )
    mapping: dict[int, int] = {}
    matched_distances: list[float] = []

    for edges in ordered_components:
        left_indices = sorted({value[0] for value in edges}, key=lambda i: left[i].node_id)
        right_indices = sorted({value[1] for value in edges}, key=lambda i: right[i].node_id)
        left_local = {value: index for index, value in enumerate(left_indices)}
        right_local = {value: index for index, value in enumerate(right_indices)}
        left_size = len(left_indices)
        right_size = len(right_indices)
        assignment_size = left_size + right_size

        # The unmatched penalty makes cardinality lexicographically more important
        # than total distance. Invalid pairs cost more than leaving both nodes
        # unmatched. This dense matrix exists only for this sparse component.
        max_pairs = min(left_size, right_size)
        unmatched_cost = (MAX_DISTANCE_UM + 1.0) * (max_pairs + 1)
        invalid_cost = unmatched_cost * (assignment_size + 2)
        costs = np.full(
            (assignment_size, assignment_size), invalid_cost, dtype=np.float64
        )
        costs[:left_size, right_size:] = unmatched_cost
        costs[left_size:, :right_size] = unmatched_cost
        costs[left_size:, right_size:] = 0.0
        for left_index, right_index, distance in edges:
            costs[left_local[left_index], right_local[right_index]] = distance

        rows, columns = linear_sum_assignment(costs)
        edge_distance = {
            (left_local[left_index], right_local[right_index]): distance
            for left_index, right_index, distance in edges
        }
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if row >= left_size or column >= right_size:
                continue
            distance = edge_distance.get((row, column))
            if distance is None:
                continue
            left_node = left[left_indices[row]]
            right_node = right[right_indices[column]]
            mapping[left_node.node_id] = right_node.node_id
            matched_distances.append(distance)

        stats.components += 1
        stats.peak_component_candidates = max(
            stats.peak_component_candidates, len(edges)
        )
        stats.peak_component_left_nodes = max(
            stats.peak_component_left_nodes, left_size
        )
        stats.peak_component_right_nodes = max(
            stats.peak_component_right_nodes, right_size
        )
        stats.peak_component_nodes = max(
            stats.peak_component_nodes, assignment_size
        )
        stats.peak_assignment_dimension = max(
            stats.peak_assignment_dimension, assignment_size
        )

    return mapping, matched_distances, stats


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def overlap_counts(left: int, right: int, intersection: int) -> dict[str, Any]:
    union = left + right - intersection
    return {
        "left": left,
        "right": right,
        "intersection": intersection,
        "left_only": left - intersection,
        "right_only": right - intersection,
        "left_overlap_ratio": _ratio(intersection, left),
        "right_overlap_ratio": _ratio(intersection, right),
        "jaccard_ratio": _ratio(intersection, union),
    }


def _density_label(count: int) -> str:
    for label, lower, upper in DENSITY_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise AssertionError(f"no density bin for count {count}")


def _new_density() -> dict[str, dict[str, int]]:
    return {
        label: {
            "frames": 0,
            "left": 0,
            "right": 0,
            "intersection": 0,
            "candidate_pairs": 0,
        }
        for label, _, _ in DENSITY_BINS
    }


def _density_report(values: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, lower, upper in DENSITY_BINS:
        value = values[label]
        result.append(
            {
                "label": label,
                "min_frame_nodes": lower,
                "max_frame_nodes": upper,
                "frames": value["frames"],
                "candidate_pairs": value["candidate_pairs"],
                "nodes": overlap_counts(
                    value["left"], value["right"], value["intersection"]
                ),
            }
        )
    return result


def _children(graph: Graph) -> dict[int, set[int]]:
    children: dict[int, set[int]] = defaultdict(set)
    for source_id, target_id in graph.edges:
        children[source_id].add(target_id)
    return children


def _motif_counts(
    left_graph: Graph,
    right_graph: Graph,
    mapping: dict[int, int],
    *,
    division: bool,
) -> dict[str, Any]:
    left_children = _children(left_graph)
    right_children = _children(right_graph)
    predicate = (lambda count: count >= 2) if division else (lambda count: count == 1)
    left_motifs = {
        source: targets
        for source, targets in left_children.items()
        if predicate(len(targets))
    }
    right_motifs = {
        source: targets
        for source, targets in right_children.items()
        if predicate(len(targets))
    }
    source_status = 0
    agreement = 0
    for left_source, left_targets in left_motifs.items():
        right_source = mapping.get(left_source)
        if right_source not in right_motifs:
            continue
        source_status += 1
        mapped_targets = {mapping[target] for target in left_targets if target in mapping}
        if len(mapped_targets) == len(left_targets) and mapped_targets == right_motifs[right_source]:
            agreement += 1
    result = overlap_counts(len(left_motifs), len(right_motifs), agreement)
    result["source_status_intersection"] = source_status
    return result


def _distance_report(count: int, total: float, maximum: float) -> dict[str, Any]:
    return {
        "matched_nodes": count,
        "sum_um": total,
        "mean_um": total / count if count else None,
        "max_um": maximum if count else None,
    }


def _compare_dataset(
    left_graph: Graph,
    right_graph: Graph,
    *,
    left_present: bool,
    right_present: bool,
) -> tuple[dict[str, Any], SparseStats, dict[str, dict[str, int]], tuple[int, float, float]]:
    left_frames: dict[int, list[Node]] = defaultdict(list)
    right_frames: dict[int, list[Node]] = defaultdict(list)
    for node in left_graph.nodes.values():
        left_frames[node.t].append(node)
    for node in right_graph.nodes.values():
        right_frames[node.t].append(node)

    times = sorted(set(left_frames) | set(right_frames))
    mapping: dict[int, int] = {}
    sparse = SparseStats()
    density = _new_density()
    distance_count = 0
    distance_sum = 0.0
    distance_max = 0.0
    for frame_time in times:
        left_nodes = left_frames.get(frame_time, [])
        right_nodes = right_frames.get(frame_time, [])
        frame_mapping, distances, frame_sparse = match_frame(left_nodes, right_nodes)
        mapping.update(frame_mapping)
        sparse.merge(frame_sparse)
        distance_count += len(distances)
        distance_sum += math.fsum(distances)
        if distances:
            distance_max = max(distance_max, max(distances))

        density_value = density[_density_label(max(len(left_nodes), len(right_nodes)))]
        density_value["frames"] += 1
        density_value["left"] += len(left_nodes)
        density_value["right"] += len(right_nodes)
        density_value["intersection"] += len(frame_mapping)
        density_value["candidate_pairs"] += frame_sparse.candidate_pairs

    edge_agreement = 0
    for source_id, target_id in left_graph.edges:
        mapped_source = mapping.get(source_id)
        mapped_target = mapping.get(target_id)
        if (
            mapped_source is not None
            and mapped_target is not None
            and (mapped_source, mapped_target) in right_graph.edges
        ):
            edge_agreement += 1

    frame_intersection = len(set(left_frames) & set(right_frames))
    report = {
        "present": {"left": left_present, "right": right_present},
        "frames": overlap_counts(
            len(left_frames), len(right_frames), frame_intersection
        ),
        "nodes": overlap_counts(
            len(left_graph.nodes), len(right_graph.nodes), len(mapping)
        ),
        "edges": overlap_counts(
            len(left_graph.edges), len(right_graph.edges), edge_agreement
        ),
        "division_sources": _motif_counts(
            left_graph, right_graph, mapping, division=True
        ),
        "single_child_sources": _motif_counts(
            left_graph, right_graph, mapping, division=False
        ),
        "matching_distance": _distance_report(
            distance_count, distance_sum, distance_max
        ),
        "density_bins": _density_report(density),
        "sparse_matching": sparse.as_dict(),
    }
    return report, sparse, density, (distance_count, distance_sum, distance_max)


def _sum_overlap(
    reports: Iterable[dict[str, Any]], key: str
) -> dict[str, Any]:
    rows = [report[key] for report in reports]
    return overlap_counts(
        sum(int(row["left"]) for row in rows),
        sum(int(row["right"]) for row in rows),
        sum(int(row["intersection"]) for row in rows),
    )


def compare_graphs(
    left_graphs: dict[str, Graph], right_graphs: dict[str, Graph]
) -> tuple[dict[str, Any], SparseStats]:
    common = set(left_graphs) & set(right_graphs)
    if not common:
        raise ComparisonError("left and right inputs have no overlapping datasets")

    empty = Graph()
    per_dataset: dict[str, dict[str, Any]] = {}
    global_sparse = SparseStats()
    global_density = _new_density()
    global_distance_count = 0
    global_distance_sum = 0.0
    global_distance_max = 0.0

    for dataset in sorted(set(left_graphs) | set(right_graphs)):
        report, sparse, density, distance = _compare_dataset(
            left_graphs.get(dataset, empty),
            right_graphs.get(dataset, empty),
            left_present=dataset in left_graphs,
            right_present=dataset in right_graphs,
        )
        per_dataset[dataset] = report
        global_sparse.merge(sparse)
        for label, _, _ in DENSITY_BINS:
            for key in ("frames", "left", "right", "intersection", "candidate_pairs"):
                global_density[label][key] += density[label][key]
        global_distance_count += distance[0]
        global_distance_sum += distance[1]
        global_distance_max = max(global_distance_max, distance[2])

    dataset_reports = list(per_dataset.values())
    left_datasets = len(left_graphs)
    right_datasets = len(right_graphs)
    global_report: dict[str, Any] = {
        "dataset_presence": overlap_counts(
            left_datasets, right_datasets, len(common)
        ),
        "frames": _sum_overlap(dataset_reports, "frames"),
        "nodes": _sum_overlap(dataset_reports, "nodes"),
        "edges": _sum_overlap(dataset_reports, "edges"),
        "division_sources": _sum_overlap(dataset_reports, "division_sources"),
        "single_child_sources": _sum_overlap(
            dataset_reports, "single_child_sources"
        ),
        "matching_distance": _distance_report(
            global_distance_count, global_distance_sum, global_distance_max
        ),
        "density_bins": _density_report(global_density),
        "sparse_matching": global_sparse.as_dict(),
    }
    global_report["division_sources"]["source_status_intersection"] = sum(
        int(report["division_sources"]["source_status_intersection"])
        for report in dataset_reports
    )
    global_report["single_child_sources"]["source_status_intersection"] = sum(
        int(report["single_child_sources"]["source_status_intersection"])
        for report in dataset_reports
    )
    return {"global": global_report, "datasets": per_dataset}, global_sparse


def build_report(
    left_spec: str,
    right_spec: str,
    *,
    strict: bool = False,
    left_strip_prefix: str | None = None,
    right_strip_prefix: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    left_paths = resolve_input(left_spec)
    right_paths = resolve_input(right_spec)
    left_provenance = input_provenance(
        left_spec, left_paths, strip_prefix=left_strip_prefix
    )
    right_provenance = input_provenance(
        right_spec, right_paths, strip_prefix=right_strip_prefix
    )
    left_graphs = load_graphs(
        left_paths, strict=strict, strip_prefix=left_strip_prefix
    )
    right_graphs = load_graphs(
        right_paths, strict=strict, strip_prefix=right_strip_prefix
    )
    loaded = time.perf_counter()

    comparison, _ = compare_graphs(left_graphs, right_graphs)
    compared = time.perf_counter()
    return {
        "schema_version": 1,
        "kind": "competition_submission_graph_comparison",
        "parameters": {
            "max_distance_um": MAX_DISTANCE_UM,
            "scale_zyx_um": list(SCALE_ZYX_UM),
            "strict": strict,
            "left_strip_prefix": left_strip_prefix,
            "right_strip_prefix": right_strip_prefix,
            "assignment": "maximum-cardinality then minimum-distance",
            "tie_break": "ascending node IDs with deterministic scipy assignment",
            "density_bins": [label for label, _, _ in DENSITY_BINS],
        },
        "provenance": {"left": left_provenance, "right": right_provenance},
        **comparison,
        "timing_seconds": {
            "input_resolution_hashing_and_parsing": round(loaded - started, 6),
            "comparison": round(compared - loaded, 6),
            "total": round(compared - started, 6),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", nargs="?", help="left CSV, directory, or quoted glob")
    parser.add_argument("right", nargs="?", help="right CSV, directory, or quoted glob")
    parser.add_argument("--left", dest="left_option", help=argparse.SUPPRESS)
    parser.add_argument("--right", dest="right_option", help=argparse.SUPPRESS)
    parser.add_argument(
        "--left-strip-prefix",
        help="strip this prefix from left dataset IDs when it is present",
    )
    parser.add_argument(
        "--right-strip-prefix",
        help="strip this prefix from right dataset IDs when it is present",
    )
    parser.add_argument("--out", help="write JSON to this path instead of stdout")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="reject edges that do not advance exactly one frame",
    )
    args = parser.parse_args(argv)
    if args.left and args.left_option:
        parser.error("left input was supplied both positionally and with --left")
    if args.right and args.right_option:
        parser.error("right input was supplied both positionally and with --right")
    args.left = args.left or args.left_option
    args.right = args.right or args.right_option
    if not args.left or not args.right:
        parser.error("both left and right inputs are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.left,
            args.right,
            strict=args.strict,
            left_strip_prefix=args.left_strip_prefix,
            right_strip_prefix=args.right_strip_prefix,
        )
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out and args.out != "-":
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
    except (ComparisonError, OSError) as exc:
        print(f"cmpgraph: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
