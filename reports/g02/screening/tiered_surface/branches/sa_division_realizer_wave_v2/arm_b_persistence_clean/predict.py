#!/usr/bin/env python3
"""Target-blind persistence division arm.

This module intentionally uses only frozen fragment graph topology and coordinates.
It never imports campaign evaluators or reads outcome artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable, Sequence
import uuid


EXPECTED_COLUMNS = (
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
INPUT_NAME = re.compile(r"^train__(44b6|6bba)_[0-9a-f]{8}\.csv$")
FORBIDDEN_READ_TOKENS = (
    "continuity_target",
    "division_cont",
    "division_value",
    "oracle",
    "dossier",
    "bundle",
    "result",
    "wave_terminal",
    "ledger",
    "live_state",
)


class FailClosed(RuntimeError):
    """A preregistered invariant failed; publish nothing."""


@dataclass(frozen=True)
class Node:
    node_id: int
    t: int
    z: int
    y: int
    x: int

    @property
    def xyz(self) -> tuple[int, int, int]:
        return (self.z, self.y, self.x)


@dataclass(frozen=True)
class Graph:
    dataset: str
    nodes: dict[int, Node]
    edges: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class Candidate:
    p: int
    a: int
    b: int
    ga: int
    gb: int
    cost: float

    @property
    def nodes(self) -> frozenset[int]:
        return frozenset((self.p, self.a, self.b, self.ga, self.gb))

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (self.p, self.a, self.b, self.ga, self.gb)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode("ascii")
        + b"\n"
    )


def guarded_read_bytes(path: Path, allowed_root: Path) -> bytes:
    resolved = path.resolve()
    root = allowed_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FailClosed(f"read escaped allowed root: {resolved}") from exc
    lowered = resolved.name.lower()
    if any(token in lowered for token in FORBIDDEN_READ_TOKENS):
        raise FailClosed(f"forbidden read token in filename: {resolved.name}")
    return resolved.read_bytes()


def parse_int(row: dict[str, str], key: str, file_label: str, line: int) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise FailClosed(f"{file_label}:{line}: invalid integer in {key}") from exc


def parse_graph_bytes(data: bytes, file_label: str) -> Graph:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FailClosed(f"{file_label}: input is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise FailClosed(f"{file_label}: unexpected CSV schema")

    nodes: dict[int, Node] = {}
    edges: set[tuple[int, int]] = set()
    csv_ids: set[int] = set()
    datasets: set[str] = set()
    for line, row in enumerate(reader, start=2):
        row_id = parse_int(row, "id", file_label, line)
        if row_id in csv_ids:
            raise FailClosed(f"{file_label}:{line}: duplicate CSV id {row_id}")
        csv_ids.add(row_id)
        dataset = row["dataset"]
        if not dataset:
            raise FailClosed(f"{file_label}:{line}: empty dataset")
        datasets.add(dataset)
        row_type = row["row_type"]
        if row_type == "node":
            node = Node(
                node_id=parse_int(row, "node_id", file_label, line),
                t=parse_int(row, "t", file_label, line),
                z=parse_int(row, "z", file_label, line),
                y=parse_int(row, "y", file_label, line),
                x=parse_int(row, "x", file_label, line),
            )
            if node.node_id in nodes:
                raise FailClosed(
                    f"{file_label}:{line}: duplicate node ID {node.node_id}"
                )
            if parse_int(row, "source_id", file_label, line) != -1:
                raise FailClosed(f"{file_label}:{line}: node source_id is not -1")
            if parse_int(row, "target_id", file_label, line) != -1:
                raise FailClosed(f"{file_label}:{line}: node target_id is not -1")
            nodes[node.node_id] = node
        elif row_type == "edge":
            edge = (
                parse_int(row, "source_id", file_label, line),
                parse_int(row, "target_id", file_label, line),
            )
            if edge in edges:
                raise FailClosed(f"{file_label}:{line}: duplicate edge {edge}")
            edges.add(edge)
            for key in ("node_id", "t", "z", "y", "x"):
                if parse_int(row, key, file_label, line) != -1:
                    raise FailClosed(f"{file_label}:{line}: edge {key} is not -1")
        else:
            raise FailClosed(f"{file_label}:{line}: unknown row_type {row_type!r}")
    if len(datasets) != 1:
        raise FailClosed(f"{file_label}: expected exactly one dataset")
    if not nodes:
        raise FailClosed(f"{file_label}: graph has no nodes")
    graph = Graph(next(iter(datasets)), nodes, frozenset(edges))
    validate_graph(graph, file_label)
    return graph


def adjacency(
    graph: Graph,
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    outgoing_work: dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    incoming_work: dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    for source, target in graph.edges:
        if source not in graph.nodes or target not in graph.nodes:
            raise FailClosed(f"{graph.dataset}: edge has missing endpoint")
        outgoing_work[source].append(target)
        incoming_work[target].append(source)
    outgoing = {
        node_id: tuple(sorted(targets))
        for node_id, targets in outgoing_work.items()
    }
    incoming = {
        node_id: tuple(sorted(sources))
        for node_id, sources in incoming_work.items()
    }
    return outgoing, incoming


def validate_graph(graph: Graph, file_label: str) -> None:
    outgoing, incoming = adjacency(graph)
    for source, target in graph.edges:
        if graph.nodes[target].t - graph.nodes[source].t != 1:
            raise FailClosed(f"{file_label}: edge {(source, target)} is not one-frame")
    for node_id in graph.nodes:
        if len(incoming[node_id]) > 1:
            raise FailClosed(f"{file_label}: node {node_id} has indegree > 1")
        if len(outgoing[node_id]) > 2:
            raise FailClosed(f"{file_label}: node {node_id} has outdegree > 2")


def nearest_rank(values: Sequence[float], proportion: float) -> float:
    if not values:
        raise FailClosed("nearest-rank statistic requested from empty values")
    if not 0.0 < proportion <= 1.0:
        raise FailClosed("nearest-rank proportion outside (0,1]")
    ordered = sorted(values)
    index = max(0, math.ceil(proportion * len(ordered)) - 1)
    return float(ordered[index])


def normalized_vector(
    first: Node, second: Node, scales: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(
        (b - a) / scale
        for a, b, scale in zip(first.xyz, second.xyz, scales, strict=True)
    )


def norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalized_distance(
    first: Node, second: Node, scales: tuple[float, float, float]
) -> float:
    return norm(normalized_vector(first, second, scales))


def derive_normalization(
    graph: Graph,
) -> tuple[tuple[float, float, float], float]:
    outgoing, incoming = adjacency(graph)
    axis_displacements: list[list[float]] = [[], [], []]
    for source, target in graph.edges:
        for axis, (left, right) in enumerate(
            zip(graph.nodes[source].xyz, graph.nodes[target].xyz, strict=True)
        ):
            axis_displacements[axis].append(float(abs(right - left)))
    if any(not values for values in axis_displacements):
        raise FailClosed(f"{graph.dataset}: no edge displacements")
    scales = tuple(
        max(1.0, nearest_rank(values, 0.75)) for values in axis_displacements
    )
    one_to_one_lengths = [
        normalized_distance(graph.nodes[source], graph.nodes[target], scales)
        for source, target in graph.edges
        if len(outgoing[source]) == 1 and len(incoming[target]) == 1
    ]
    if not one_to_one_lengths:
        raise FailClosed(f"{graph.dataset}: no one-to-one edges for normalization")
    q1 = nearest_rank(one_to_one_lengths, 0.25)
    q3 = nearest_rank(one_to_one_lengths, 0.75)
    iqr = q3 - q1
    upper = q3 if iqr == 0.0 else q3 + 1.5 * iqr
    if not math.isfinite(upper) or upper < 0.0:
        raise FailClosed(f"{graph.dataset}: invalid Tukey fence")
    return scales, upper


def candidate_cost(
    graph: Graph,
    p: int,
    a: int,
    b: int,
    ga: int,
    gb: int,
    scales: tuple[float, float, float],
    upper: float,
) -> float | None:
    nodes = graph.nodes
    distances = (
        normalized_distance(nodes[p], nodes[a], scales),
        normalized_distance(nodes[p], nodes[b], scales),
        normalized_distance(nodes[a], nodes[ga], scales),
        normalized_distance(nodes[b], nodes[gb], scales),
        normalized_distance(nodes[a], nodes[b], scales),
        normalized_distance(nodes[ga], nodes[gb], scales),
    )
    motion_a = normalized_vector(nodes[a], nodes[ga], scales)
    motion_b = normalized_vector(nodes[b], nodes[gb], scales)
    disagreement = norm(
        left - right for left, right in zip(motion_a, motion_b, strict=True)
    )
    values = distances + (disagreement,)
    if any(value > upper for value in values):
        return None
    denominator = max(upper, 1.0)
    cost = sum((value / denominator) ** 2 for value in values)
    if not math.isfinite(cost):
        raise FailClosed(f"{graph.dataset}: non-finite candidate cost")
    return cost


def spatial_key(
    node: Node, physical_cells: tuple[float, float, float]
) -> tuple[int, int, int]:
    return tuple(
        math.floor(value / cell)
        for value, cell in zip(node.xyz, physical_cells, strict=True)
    )


def generate_candidates(
    graph: Graph,
) -> tuple[list[Candidate], tuple[float, float, float], float]:
    outgoing, incoming = adjacency(graph)
    scales, upper = derive_normalization(graph)
    physical_cells = tuple(scale * max(upper, 1.0) for scale in scales)

    roots_by_frame_cell: dict[
        tuple[int, tuple[int, int, int]], list[int]
    ] = {}
    for node_id, node in graph.nodes.items():
        if len(incoming[node_id]) == 0 and len(outgoing[node_id]) == 1:
            key = (node.t, spatial_key(node, physical_cells))
            roots_by_frame_cell.setdefault(key, []).append(node_id)
    for values in roots_by_frame_cell.values():
        values.sort()

    candidates: list[Candidate] = []
    for p in sorted(graph.nodes):
        if len(outgoing[p]) != 1:
            continue
        a = outgoing[p][0]
        if incoming[a] != (p,) or len(outgoing[a]) != 1:
            continue
        ga = outgoing[a][0]
        if len(incoming[ga]) != 1:
            continue
        if graph.nodes[a].t != graph.nodes[p].t + 1:
            continue
        if graph.nodes[ga].t != graph.nodes[a].t + 1:
            continue
        center = spatial_key(graph.nodes[p], physical_cells)
        root_ids: list[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cell = (center[0] + dz, center[1] + dy, center[2] + dx)
                    root_ids.extend(
                        roots_by_frame_cell.get((graph.nodes[a].t, cell), ())
                    )
        for b in sorted(root_ids):
            if b in (p, a, ga):
                continue
            gb = outgoing[b][0]
            if gb in (p, a, b, ga):
                continue
            if len(incoming[gb]) != 1:
                continue
            if graph.nodes[gb].t != graph.nodes[b].t + 1:
                continue
            cost = candidate_cost(graph, p, a, b, ga, gb, scales, upper)
            if cost is None:
                continue
            candidates.append(Candidate(p, a, b, ga, gb, cost))
    identities = [candidate.identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise FailClosed(f"{graph.dataset}: duplicate candidate identity")
    return candidates, scales, upper


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def select_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    union_find = UnionFind(len(candidates))
    node_owner: dict[int, int] = {}
    for index, candidate in enumerate(candidates):
        for node_id in candidate.nodes:
            if node_id in node_owner:
                union_find.union(index, node_owner[node_id])
            else:
                node_owner[node_id] = index
    components: dict[int, list[Candidate]] = {}
    for index, candidate in enumerate(candidates):
        components.setdefault(union_find.find(index), []).append(candidate)

    representatives: list[Candidate] = []
    for component in components.values():
        minimum = min(candidate.cost for candidate in component)
        winners = [candidate for candidate in component if candidate.cost == minimum]
        if len(winners) == 1:
            representatives.append(winners[0])
    if not representatives:
        return None
    minimum = min(candidate.cost for candidate in representatives)
    winners = [candidate for candidate in representatives if candidate.cost == minimum]
    if len(winners) != 1:
        return None
    return winners[0]


def apply_candidate(graph: Graph, selected: Candidate | None) -> Graph:
    if selected is None:
        return graph
    outgoing, incoming = adjacency(graph)
    if outgoing[selected.p] != (selected.a,):
        raise FailClosed(f"{graph.dataset}: selected parent boundary changed")
    if incoming[selected.a] != (selected.p,) or incoming[selected.b]:
        raise FailClosed(f"{graph.dataset}: selected daughter boundary changed")
    edges = {
        edge for edge in graph.edges if edge[0] != selected.p
    } | {(selected.p, selected.a), (selected.p, selected.b)}
    if len(edges) != len(graph.edges) + 1:
        raise FailClosed(f"{graph.dataset}: fork did not add exactly one net edge")
    result = Graph(graph.dataset, graph.nodes, frozenset(edges))
    validate_graph(result, graph.dataset)
    return result


def graph_csv_bytes(graph: Graph) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(EXPECTED_COLUMNS)
    row_id = 0
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        writer.writerow(
            (
                row_id,
                graph.dataset,
                "node",
                node.node_id,
                node.t,
                node.z,
                node.y,
                node.x,
                -1,
                -1,
            )
        )
        row_id += 1
    for source, target in sorted(graph.edges):
        writer.writerow(
            (row_id, graph.dataset, "edge", -1, -1, -1, -1, -1, source, target)
        )
        row_id += 1
    return stream.getvalue().encode("utf-8")


def action_record(
    graph: Graph, selected: Candidate | None, scales: tuple[float, ...], upper: float
) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset": graph.dataset,
        "axis_scales": list(scales),
        "edge_length_upper_tukey_fence": upper,
        "selected_fork_count": int(selected is not None),
    }
    if selected is None:
        base["action"] = None
        return base
    action_id = f"{graph.dataset}:persistence:{selected.p}"
    base["action"] = {
        "action_id": action_id,
        "kind": "division_fork",
        "event_nodes": asdict(selected),
        "representation": "canonical_early_of_one_frame_late_equivalence_class",
        "add_edges": [[selected.p, selected.a], [selected.p, selected.b]],
        "delete_edges": [],
        "replace_outgoing_from": [selected.p],
        "replace_incoming_to": [selected.a, selected.b],
        "expected_conflict_edges": [[selected.p, selected.a]],
    }
    return base


def swap_counters() -> dict[str, int]:
    try:
        completed = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FailClosed("could not execute vm_stat") from exc
    counters: dict[str, int] = {}
    for label in ("Swapins", "Swapouts"):
        match = re.search(rf"^{label}:\s+([0-9]+)\.$", completed.stdout, re.MULTILINE)
        if match is None:
            raise FailClosed(f"vm_stat did not expose {label}")
        counters[label.lower()] = int(match.group(1))
    return counters


def assert_zero_swapouts(reference: dict[str, int], observed: dict[str, int]) -> None:
    key = "swapouts"
    if observed[key] != reference[key]:
        raise FailClosed(
            f"zero-swapout guard failed: {key} changed "
            f"{reference[key]}->{observed[key]}"
        )


def idle_preflight(seconds: int) -> tuple[dict[str, int], dict[str, int], float]:
    if seconds != 45:
        raise FailClosed("production preflight must be exactly 45 seconds")
    before = swap_counters()
    started = time.monotonic()
    deadline = started + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
    elapsed = time.monotonic() - started
    after = swap_counters()
    assert_zero_swapouts(before, after)
    if elapsed < float(seconds):
        raise FailClosed("idle preflight elapsed less than 45 seconds")
    return before, after, elapsed


def validate_input_manifest(paths: Sequence[Path]) -> dict[str, int]:
    if len(paths) != 57:
        raise FailClosed(f"expected 57 fragments, found {len(paths)}")
    counts = {"44b6": 0, "6bba": 0}
    for path in paths:
        match = INPUT_NAME.fullmatch(path.name)
        if match is None:
            raise FailClosed(f"unexpected fragment filename: {path.name}")
        counts[match.group(1)] += 1
    if counts != {"44b6": 21, "6bba": 36}:
        raise FailClosed(f"unexpected fragment prefix counts: {counts}")
    return counts


def verify_only_parent_changed(
    before: Graph, after: Graph, selected: Candidate | None
) -> None:
    if before.nodes != after.nodes:
        raise FailClosed(f"{before.dataset}: node table changed")
    changed = before.edges ^ after.edges
    if selected is None:
        if changed:
            raise FailClosed(f"{before.dataset}: noop graph changed")
        return
    if changed != {(selected.p, selected.b)}:
        raise FailClosed(f"{before.dataset}: unexpected edge delta {sorted(changed)}")
    if sum(1 for source, _target in after.edges if source == selected.p) != 2:
        raise FailClosed(f"{before.dataset}: selected parent is not a binary fork")


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def production_run(repo_root: Path, arm_dir: Path) -> dict[str, object]:
    fragment_root = (repo_root / "kaggle_kernel/bhcv57out_v3/fragments").resolve()
    prereg_path = arm_dir / "PREREGISTRATION.json"
    amendment_path = arm_dir / "EXECUTION_GUARD_AMENDMENT.json"
    if not prereg_path.is_file():
        raise FailClosed("PREREGISTRATION.json is missing")
    if not amendment_path.is_file():
        raise FailClosed("EXECUTION_GUARD_AMENDMENT.json is missing")
    prereg_bytes = prereg_path.read_bytes()
    amendment_bytes = amendment_path.read_bytes()
    prereg = json.loads(prereg_bytes)
    amendment = json.loads(amendment_bytes)
    if prereg.get("status_at_creation") != "locked_before_prediction_generation":
        raise FailClosed("preregistration is not locked")
    if prereg.get("design_provenance", {}).get("prediction_generation_had_started_when_locked"):
        raise FailClosed("preregistration provenance is not clean")
    if (
        amendment.get("status")
        != "immutable_locked_before_any_fragment_prediction_outcome"
        or amendment.get("corrected_guard", {}).get("counter") != "Swapouts"
        or amendment.get("prediction_outcomes_known_when_locked") is not False
    ):
        raise FailClosed("execution guard amendment is not valid and locked")

    final_predictions = arm_dir / "predictions"
    final_actions = arm_dir / "ACTIONS.json"
    final_receipt = arm_dir / "RUN_RECEIPT.json"
    for final_path in (final_predictions, final_actions, final_receipt):
        if final_path.exists():
            raise FailClosed(f"fresh-output guard failed: {final_path.name} exists")

    paths = sorted(fragment_root.glob("train__*.csv"), key=lambda path: path.name)
    prefix_counts = validate_input_manifest(paths)
    preflight_before, preflight_after, preflight_elapsed = idle_preflight(45)

    stage = arm_dir / f".staging.{os.getpid()}.{uuid.uuid4().hex}"
    stage_predictions = stage / "predictions"
    published: list[Path] = []
    try:
        stage_predictions.mkdir(parents=True, exist_ok=False)
        input_records: list[dict[str, object]] = []
        output_records: list[dict[str, object]] = []
        action_records: list[dict[str, object]] = []
        action_ids: set[str] = set()
        selected_total = 0

        for path in paths:
            data = guarded_read_bytes(path, fragment_root)
            graph = parse_graph_bytes(data, path.name)
            expected_dataset = path.stem.removeprefix("train__")
            if graph.dataset != expected_dataset:
                raise FailClosed(f"{path.name}: dataset does not match filename")
            candidates, scales, upper = generate_candidates(graph)
            selected = select_candidate(candidates)
            output_graph = apply_candidate(graph, selected)
            verify_only_parent_changed(graph, output_graph, selected)
            if selected is not None:
                selected_total += 1
            record = action_record(graph, selected, scales, upper)
            action = record["action"]
            if isinstance(action, dict):
                action_id = str(action["action_id"])
                if action_id in action_ids:
                    raise FailClosed(f"duplicate action ID: {action_id}")
                action_ids.add(action_id)
            action_records.append(record)

            output_data = graph_csv_bytes(output_graph)
            output_path = stage_predictions / path.name
            atomic_write(output_path, output_data)
            reparsed = parse_graph_bytes(output_data, output_path.name)
            verify_only_parent_changed(graph, reparsed, selected)
            input_records.append(
                {
                    "path": str(path.relative_to(repo_root)),
                    "sha256": sha256_bytes(data),
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "candidate_count": len(candidates),
                }
            )
            output_records.append(
                {
                    "path": f"predictions/{path.name}",
                    "sha256": sha256_bytes(output_data),
                    "node_count": len(output_graph.nodes),
                    "edge_count": len(output_graph.edges),
                    "selected_fork_count": int(selected is not None),
                }
            )

        if selected_total > len(paths):
            raise FailClosed("at-most-one-fork-per-video guard failed")
        if len(output_records) != 57 or len({row["path"] for row in output_records}) != 57:
            raise FailClosed("output cardinality or duplicate-path guard failed")
        if len(action_ids) != selected_total:
            raise FailClosed("action ID count disagrees with selected fork count")

        actions_payload = {
            "schema": "cell-tracking.target-blind-persistence-actions.v1",
            "preregistration_sha256": sha256_bytes(prereg_bytes),
            "video_count": len(action_records),
            "selected_fork_count": selected_total,
            "videos": action_records,
        }
        actions_bytes = canonical_json_bytes(actions_payload)
        atomic_write(stage / "ACTIONS.json", actions_bytes)

        generation_swap = swap_counters()
        assert_zero_swapouts(preflight_before, generation_swap)
        script_bytes = Path(__file__).resolve().read_bytes()
        receipt_payload: dict[str, object] = {
            "schema": "cell-tracking.clean-arm-run-receipt.v1",
            "completed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "status": "pass",
            "runtime": {
                "cpu_only": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "network_used": False,
                "scorer_or_evaluator_used": False,
                "preflight_required_seconds": 45,
                "preflight_elapsed_seconds": preflight_elapsed,
                "swap_before": preflight_before,
                "swap_after_idle": preflight_after,
                "swap_after_generation": generation_swap,
                "zero_swapout_guard": "pass",
            },
            "counts": {
                "input_videos": len(input_records),
                "output_videos": len(output_records),
                "prefixes": prefix_counts,
                "selected_forks": selected_total,
                "maximum_selected_forks_per_video": max(
                    row["selected_fork_count"] for row in output_records
                ),
                "duplicate_input_paths": 0,
                "duplicate_output_paths": 0,
                "duplicate_action_ids": 0,
                "duplicate_output_nodes_or_edges": 0,
            },
            "hashes": {
                "preregistration_sha256": sha256_bytes(prereg_bytes),
                "execution_guard_amendment_sha256": sha256_bytes(amendment_bytes),
                "predictor_sha256": sha256_bytes(script_bytes),
                "actions_sha256": sha256_bytes(actions_bytes),
            },
            "inputs": input_records,
            "outputs": output_records,
            "postconditions": {
                "all_graphs_valid": True,
                "only_selected_parent_outgoing_boundaries_changed": True,
                "one_frame_edges_only": True,
                "indegree_at_most_one": True,
                "outdegree_at_most_two": True,
                "at_most_one_fork_per_video": True,
                "zero_duplicates": True,
            },
        }
        receipt_bytes = canonical_json_bytes(receipt_payload)
        atomic_write(stage / "RUN_RECEIPT.json", receipt_bytes)

        assert_zero_swapouts(preflight_before, swap_counters())
        os.replace(stage_predictions, final_predictions)
        published.append(final_predictions)
        os.replace(stage / "ACTIONS.json", final_actions)
        published.append(final_actions)
        os.replace(stage / "RUN_RECEIPT.json", final_receipt)
        published.append(final_receipt)
        post_publish_swap = swap_counters()
        assert_zero_swapouts(preflight_before, post_publish_swap)
        stage.rmdir()
        return receipt_payload
    except BaseException:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        receipt = production_run(args.repo_root.resolve(), args.arm_dir.resolve())
    except (FailClosed, json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "counts": receipt["counts"],
                "hashes": receipt["hashes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
