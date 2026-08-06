"""Deterministic graph probes for Kaggle-to-local score calibration."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


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
SCALE_ZYX = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class SubmissionNode:
    node_id: int
    t: int
    z: int
    y: int
    x: int

    @property
    def coord_zyx(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x


@dataclass(frozen=True)
class SubmissionEdge:
    source_id: int
    target_id: int


@dataclass
class SubmissionGraph:
    dataset: str
    nodes: dict[int, SubmissionNode]
    edges: list[SubmissionEdge]

    def copy(self) -> "SubmissionGraph":
        return SubmissionGraph(self.dataset, dict(self.nodes), list(self.edges))


def read_submission(
    path: Path,
    *,
    max_outdegree: int | None = 2,
) -> dict[str, SubmissionGraph]:
    """Read a submission while preserving the default competition-safe topology checks.

    ``max_outdegree=None`` is deliberately an opt-in parser policy for
    faithfully evaluating an externally authored graph whose scorer accepts
    higher-degree parents.  It does not weaken the default used by APNCC and
    all ordinary calibration/submission paths.
    """
    graphs: dict[str, SubmissionGraph] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_COLUMNS:
            raise ValueError(f"unexpected submission header in {path}: {reader.fieldnames}")
        for row in reader:
            dataset = str(row["dataset"])
            graph = graphs.setdefault(dataset, SubmissionGraph(dataset, {}, []))
            if row["row_type"] == "node":
                node = SubmissionNode(
                    node_id=int(row["node_id"]),
                    t=int(row["t"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                )
                if node.node_id in graph.nodes:
                    raise ValueError(f"duplicate node ID {node.node_id} in {dataset}")
                graph.nodes[node.node_id] = node
            elif row["row_type"] == "edge":
                graph.edges.append(
                    SubmissionEdge(int(row["source_id"]), int(row["target_id"]))
                )
            else:
                raise ValueError(f"unexpected row_type {row['row_type']!r}")
    validate_graphs(graphs, max_outdegree=max_outdegree)
    return graphs


def write_submission(graphs: dict[str, SubmissionGraph], path: Path) -> None:
    validate_graphs(graphs)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for dataset in sorted(graphs):
            graph = graphs[dataset]
            for node in sorted(graph.nodes.values(), key=lambda value: (value.t, value.node_id)):
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": node.node_id,
                        "t": node.t,
                        "z": node.z,
                        "y": node.y,
                        "x": node.x,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for edge in sorted(graph.edges, key=lambda value: (value.source_id, value.target_id)):
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                    }
                )
                row_id += 1


def validate_graphs(
    graphs: dict[str, SubmissionGraph],
    *,
    max_outdegree: int | None = 2,
) -> None:
    if max_outdegree is not None and max_outdegree < 0:
        raise ValueError("max_outdegree must be non-negative or None")
    if not graphs:
        raise ValueError("submission contains no datasets")
    for dataset, graph in graphs.items():
        if not graph.nodes:
            raise ValueError(f"dataset {dataset} contains no nodes")
        seen_edges: set[tuple[int, int]] = set()
        indegree: Counter[int] = Counter()
        outdegree: Counter[int] = Counter()
        for edge in graph.edges:
            pair = (edge.source_id, edge.target_id)
            if pair in seen_edges:
                raise ValueError(f"duplicate edge {pair} in {dataset}")
            seen_edges.add(pair)
            source = graph.nodes.get(edge.source_id)
            target = graph.nodes.get(edge.target_id)
            if source is None or target is None:
                raise ValueError(f"missing edge endpoint {pair} in {dataset}")
            if target.t <= source.t:
                raise ValueError(f"non-forward edge {pair} in {dataset}")
            indegree[target.node_id] += 1
            outdegree[source.node_id] += 1
        if any(value > 1 for value in indegree.values()):
            raise ValueError(f"dataset {dataset} contains a merge")
        if max_outdegree is not None and any(
            value > max_outdegree for value in outdegree.values()
        ):
            raise ValueError(
                f"dataset {dataset} contains out-degree greater than {max_outdegree}"
            )


def _distance_um(a: SubmissionNode, b: SubmissionNode) -> float:
    return math.sqrt(
        sum(
            ((float(av) - float(bv)) * scale) ** 2
            for av, bv, scale in zip(a.coord_zyx, b.coord_zyx, SCALE_ZYX, strict=True)
        )
    )


def _midpoint_residual_um(
    previous: SubmissionNode,
    parent: SubmissionNode,
    child_a: SubmissionNode,
    child_b: SubmissionNode,
) -> float:
    residual_sq = 0.0
    for prev_v, parent_v, a_v, b_v, scale in zip(
        previous.coord_zyx,
        parent.coord_zyx,
        child_a.coord_zyx,
        child_b.coord_zyx,
        SCALE_ZYX,
        strict=True,
    ):
        predicted = 2.0 * parent_v - prev_v
        midpoint = 0.5 * (a_v + b_v)
        residual_sq += ((midpoint - predicted) * scale) ** 2
    return math.sqrt(residual_sq)


def _stable_fraction(dataset: str, edge: SubmissionEdge) -> float:
    payload = f"{dataset}:{edge.source_id}:{edge.target_id}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def thin_edges(
    graph: SubmissionGraph,
    *,
    fraction: float = 0.10,
) -> tuple[SubmissionGraph, dict[str, int | float]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    result = graph.copy()
    result.edges = [
        edge for edge in graph.edges if _stable_fraction(graph.dataset, edge) >= fraction
    ]
    return result, {
        "before_edges": len(graph.edges),
        "after_edges": len(result.edges),
        "removed_edges": len(graph.edges) - len(result.edges),
        "fraction": fraction,
    }


def prune_long_edges(
    graph: SubmissionGraph,
    *,
    max_distance_um: float = 5.0,
) -> tuple[SubmissionGraph, dict[str, int | float]]:
    if max_distance_um <= 0.0:
        raise ValueError("max_distance_um must be positive")
    result = graph.copy()
    result.edges = [
        edge
        for edge in graph.edges
        if _distance_um(graph.nodes[edge.source_id], graph.nodes[edge.target_id]) <= max_distance_um
    ]
    return result, {
        "before_edges": len(graph.edges),
        "after_edges": len(result.edges),
        "removed_edges": len(graph.edges) - len(result.edges),
        "max_distance_um": max_distance_um,
    }


def prune_short_components(
    graph: SubmissionGraph,
    *,
    max_component_edges: int = 1,
) -> tuple[SubmissionGraph, dict[str, int]]:
    if max_component_edges < 0:
        raise ValueError("max_component_edges must be non-negative")
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)

    components: list[set[int]] = []
    remaining = set(graph.nodes)
    while remaining:
        start = min(remaining)
        component = {start}
        queue: deque[int] = deque([start])
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    edge_counts: dict[int, int] = {}
    component_for_node: dict[int, int] = {}
    for index, component in enumerate(components):
        for node_id in component:
            component_for_node[node_id] = index
        edge_counts[index] = 0
    for edge in graph.edges:
        edge_counts[component_for_node[edge.source_id]] += 1

    removable = {
        index for index, edge_count in edge_counts.items() if edge_count <= max_component_edges
    }
    keep_nodes = {
        node_id
        for node_id, index in component_for_node.items()
        if index not in removable
    }
    if not keep_nodes:
        largest = max(
            range(len(components)),
            key=lambda index: (edge_counts[index], len(components[index]), -index),
        )
        keep_nodes = set(components[largest])
        removable.discard(largest)

    result = SubmissionGraph(
        graph.dataset,
        {node_id: node for node_id, node in graph.nodes.items() if node_id in keep_nodes},
        [
            edge
            for edge in graph.edges
            if edge.source_id in keep_nodes and edge.target_id in keep_nodes
        ],
    )
    return result, {
        "before_nodes": len(graph.nodes),
        "after_nodes": len(result.nodes),
        "removed_nodes": len(graph.nodes) - len(result.nodes),
        "before_edges": len(graph.edges),
        "after_edges": len(result.edges),
        "removed_edges": len(graph.edges) - len(result.edges),
        "removed_components": len(removable),
        "max_component_edges": max_component_edges,
    }


def rescue_divisions(
    graph: SubmissionGraph,
    *,
    max_parent_child_um: float = 7.2,
    min_daughter_separation_um: float = 3.0,
    max_daughter_separation_um: float = 11.0,
    max_symmetry_delta_um: float = 3.0,
    max_midpoint_residual_um: float = 2.5,
    max_forks_per_t: int = 2,
    target_daughter_separation_um: float = 5.0,
) -> tuple[SubmissionGraph, dict[str, int | float]]:
    """Link conservative birth tracks as second daughters of continuing tracks."""

    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_id].append(edge.source_id)
        outgoing[edge.source_id].append(edge.target_id)
    births_by_t: dict[int, list[int]] = defaultdict(list)
    for node in graph.nodes.values():
        if not incoming[node.node_id] and outgoing[node.node_id]:
            births_by_t[node.t].append(node.node_id)

    proposals_by_t: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for parent_id in sorted(outgoing):
        if len(outgoing[parent_id]) != 1 or len(incoming[parent_id]) != 1:
            continue
        parent = graph.nodes[parent_id]
        child_a_id = outgoing[parent_id][0]
        child_a = graph.nodes[child_a_id]
        if child_a.t != parent.t + 1 or not outgoing[child_a_id]:
            continue
        previous = graph.nodes[incoming[parent_id][0]]
        dist_a = _distance_um(parent, child_a)
        if dist_a > max_parent_child_um:
            continue
        for child_b_id in births_by_t.get(parent.t + 1, ()):
            if child_b_id == child_a_id:
                continue
            child_b = graph.nodes[child_b_id]
            dist_b = _distance_um(parent, child_b)
            if dist_b > max_parent_child_um:
                continue
            separation = _distance_um(child_a, child_b)
            if not min_daughter_separation_um <= separation <= max_daughter_separation_um:
                continue
            symmetry = abs(dist_a - dist_b)
            if symmetry > max_symmetry_delta_um:
                continue
            midpoint_residual = _midpoint_residual_um(previous, parent, child_a, child_b)
            if midpoint_residual > max_midpoint_residual_um:
                continue
            score = (
                midpoint_residual
                + 0.35 * symmetry
                + 0.15 * abs(separation - target_daughter_separation_um)
            )
            proposals_by_t[parent.t].append((score, parent_id, child_b_id))

    additions: list[SubmissionEdge] = []
    used_parents: set[int] = set()
    used_births: set[int] = set()
    for t in sorted(proposals_by_t):
        admitted = 0
        for _, parent_id, child_b_id in sorted(proposals_by_t[t]):
            if admitted >= max_forks_per_t:
                break
            if parent_id in used_parents or child_b_id in used_births:
                continue
            additions.append(SubmissionEdge(parent_id, child_b_id))
            used_parents.add(parent_id)
            used_births.add(child_b_id)
            admitted += 1

    result = graph.copy()
    result.edges.extend(additions)
    return result, {
        "before_edges": len(graph.edges),
        "after_edges": len(result.edges),
        "added_division_edges": len(additions),
        "candidate_motifs": sum(len(values) for values in proposals_by_t.values()),
        "max_parent_child_um": max_parent_child_um,
        "min_daughter_separation_um": min_daughter_separation_um,
        "max_daughter_separation_um": max_daughter_separation_um,
        "max_symmetry_delta_um": max_symmetry_delta_um,
        "max_midpoint_residual_um": max_midpoint_residual_um,
        "max_forks_per_t": max_forks_per_t,
        "target_daughter_separation_um": target_daughter_separation_um,
    }


VARIANTS: tuple[
    tuple[str, str, Callable[[SubmissionGraph], tuple[SubmissionGraph, dict[str, int | float]]]],
    ...,
] = (
    ("edge_thin_10", "calibration", lambda graph: thin_edges(graph, fraction=0.10)),
    (
        "long_edge_prune_5um",
        "calibration",
        lambda graph: prune_long_edges(graph, max_distance_um=5.0),
    ),
    (
        "short_component_prune_1edge",
        "holdout",
        lambda graph: prune_short_components(graph, max_component_edges=1),
    ),
    ("division_rescue_conservative", "holdout", rescue_divisions),
)


def apply_variant(
    graphs: dict[str, SubmissionGraph],
    transform: Callable[[SubmissionGraph], tuple[SubmissionGraph, dict[str, int | float]]],
) -> tuple[dict[str, SubmissionGraph], dict[str, dict[str, int | float]]]:
    outputs: dict[str, SubmissionGraph] = {}
    stats: dict[str, dict[str, int | float]] = {}
    for dataset, graph in graphs.items():
        outputs[dataset], stats[dataset] = transform(graph)
    validate_graphs(outputs)
    return outputs, stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_totals(graphs: Iterable[SubmissionGraph]) -> dict[str, int]:
    graph_list = list(graphs)
    outdegree: Counter[tuple[str, int]] = Counter()
    for graph in graph_list:
        outdegree.update((graph.dataset, edge.source_id) for edge in graph.edges)
    return {
        "datasets": len(graph_list),
        "nodes": sum(len(graph.nodes) for graph in graph_list),
        "edges": sum(len(graph.edges) for graph in graph_list),
        "divisions": sum(value >= 2 for value in outdegree.values()),
    }
