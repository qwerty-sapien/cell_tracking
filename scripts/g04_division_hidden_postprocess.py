#!/usr/bin/env python3
"""Apply the frozen G-04 unfiltered-geometry/cap-five division policy.

This module is intentionally independent of campaign source imports so the exact
same bytes can be embedded after the already-scored APNCC Kaggle kernel.  It
discovers dataset names from the freshly generated parent CSV and requires the
mounted test Zarr inventory to match; no public test identity is embedded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile


CSV_FIELDS = (
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
APPEARANCE_CANDIDATE_CAP = 8
POLICY = {
    "name": "unfiltered_geometry_cap5",
    "ranking": "geometry",
    "balance_min": 0.0,
    "ratio_min": 0.0,
    "ratio_max": None,
    "roundness_min": 0.0,
    "cap": 5,
}
SAFE_DATASET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class PostprocessError(RuntimeError):
    """The parent, raw input, transform, or output violated its contract."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(row: Mapping[str, str], name: str, row_number: int) -> int:
    try:
        value = row[name]
        result = int(value)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PostprocessError(f"row {row_number}: {name} is not an integer") from exc
    if str(result) != str(value).strip():
        raise PostprocessError(f"row {row_number}: {name} is not canonical integer syntax")
    return result


def _validate_dataset(value: str) -> str:
    if len(value) > 200 or SAFE_DATASET.fullmatch(value) is None:
        raise PostprocessError(f"unsafe dataset label: {value!r}")
    return value


def read_submission(
    path: Path,
    *,
    max_outdegree: int | None = 2,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Read a submission with strict defaults and an explicit cap override.

    APNCC callers retain the default outdegree-two contract.  A caller can
    set ``max_outdegree=None`` only when evaluating a known external pipeline
    whose competition scorer permits higher-degree parents.
    """

    graphs: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise PostprocessError(f"unexpected submission header: {reader.fieldnames!r}")
        row_count = 0
        for index, row in enumerate(reader):
            row_number = index + 2
            if _integer(row, "id", row_number) != index:
                raise PostprocessError(f"row {row_number}: non-contiguous id")
            dataset = _validate_dataset(str(row.get("dataset", "")))
            graph = graphs.setdefault(dataset, {"dataset": dataset, "nodes": {}, "edges": set()})
            row_type = str(row.get("row_type", ""))
            values = {
                name: _integer(row, name, row_number)
                for name in CSV_FIELDS
                if name not in {"dataset", "row_type"}
            }
            if row_type == "node":
                if values["source_id"] != -1 or values["target_id"] != -1:
                    raise PostprocessError(f"row {row_number}: node edge sentinels drifted")
                node_id = values["node_id"]
                if node_id < 0 or min(values[name] for name in ("t", "z", "y", "x")) < 0:
                    raise PostprocessError(f"row {row_number}: invalid node fields")
                if node_id in graph["nodes"]:
                    raise PostprocessError(f"row {row_number}: duplicate node {dataset}:{node_id}")
                graph["nodes"][node_id] = {
                    "node_id": node_id,
                    "t": values["t"],
                    "z": values["z"],
                    "y": values["y"],
                    "x": values["x"],
                }
            elif row_type == "edge":
                if any(values[name] != -1 for name in ("node_id", "t", "z", "y", "x")):
                    raise PostprocessError(f"row {row_number}: edge node sentinels drifted")
                edge = (values["source_id"], values["target_id"])
                if min(edge) < 0 or edge in graph["edges"]:
                    raise PostprocessError(f"row {row_number}: invalid or duplicate edge {edge}")
                graph["edges"].add(edge)
            else:
                raise PostprocessError(f"row {row_number}: unexpected row type {row_type!r}")
            row_count += 1
    validate_graphs(graphs, max_outdegree=max_outdegree)
    return graphs, row_count


def validate_graphs(
    graphs: Mapping[str, Mapping[str, Any]],
    *,
    max_outdegree: int | None = 2,
) -> dict[str, int]:
    if max_outdegree is not None and max_outdegree < 0:
        raise PostprocessError("max_outdegree must be non-negative or None")
    if not graphs:
        raise PostprocessError("submission contains no datasets")
    totals = {"datasets": len(graphs), "nodes": 0, "edges": 0, "dividing_nodes": 0}
    for dataset, graph in graphs.items():
        nodes = graph["nodes"]
        edges = graph["edges"]
        if not nodes:
            raise PostprocessError(f"{dataset}: graph contains no nodes")
        indegree: dict[int, int] = defaultdict(int)
        outdegree: dict[int, int] = defaultdict(int)
        for source, target in edges:
            if source not in nodes or target not in nodes:
                raise PostprocessError(f"{dataset}: edge endpoint absent: {(source, target)!r}")
            if int(nodes[target]["t"]) != int(nodes[source]["t"]) + 1:
                raise PostprocessError(f"{dataset}: edge is not exactly one frame forward")
            indegree[target] += 1
            outdegree[source] += 1
            if indegree[target] > 1:
                raise PostprocessError(f"{dataset}: merge")
            if max_outdegree is not None and outdegree[source] > max_outdegree:
                raise PostprocessError(
                    f"{dataset}: outdegree > {max_outdegree}"
                )
        totals["nodes"] += len(nodes)
        totals["edges"] += len(edges)
        totals["dividing_nodes"] += sum(value == 2 for value in outdegree.values())
    return totals


def _write_submission(graphs: Mapping[str, Mapping[str, Any]], path: Path) -> int:
    validate_graphs(graphs)
    if path.exists():
        raise PostprocessError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    row_id = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for dataset in sorted(graphs):
                graph = graphs[dataset]
                for node in sorted(
                    graph["nodes"].values(), key=lambda value: (int(value["t"]), int(value["node_id"]))
                ):
                    writer.writerow(
                        {
                            "id": row_id,
                            "dataset": dataset,
                            "row_type": "node",
                            "node_id": int(node["node_id"]),
                            "t": int(node["t"]),
                            "z": int(node["z"]),
                            "y": int(node["y"]),
                            "x": int(node["x"]),
                            "source_id": -1,
                            "target_id": -1,
                        }
                    )
                    row_id += 1
                for source, target in sorted(graph["edges"]):
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
                            "source_id": source,
                            "target_id": target,
                        }
                    )
                    row_id += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return row_id


def adjacency(edges: Iterable[tuple[int, int]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for source, target in sorted(edges):
        outgoing[source].append(target)
        incoming[target].append(source)
    return dict(outgoing), dict(incoming)


def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.sqrt(
        sum(
            ((float(left[axis]) - float(right[axis])) * SCALE_ZYX[index]) ** 2
            for index, axis in enumerate(("z", "y", "x"))
        )
    )


def p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values if value > 0.0)
    if not ordered:
        raise PostprocessError("no positive one-frame edge lengths")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def stable_key(identity: str, parent: int, branches: Sequence[int]) -> str:
    encoded = json.dumps(
        [identity, int(parent), sorted(int(value) for value in branches)],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def appearance_prefilter(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact frozen G-03 appearance prefilter, with no public-ID whitelist."""

    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing, incoming = adjacency(edges)
    radius = p95(
        [
            distance(nodes[source], nodes[target])
            for source, target in edges
            if int(nodes[target]["t"]) == int(nodes[source]["t"]) + 1
        ]
    )
    births: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        if not incoming.get(node_id):
            births[int(node["t"])].append(node_id)
    identity = f"test:{graph['dataset']}"
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
        for secondary in births.get(int(nodes[parent]["t"]) + 1, ()):  # insertion order is frozen
            if secondary == primary:
                continue
            secondary_distance = distance(nodes[parent], nodes[secondary])
            sister_distance = distance(nodes[primary], nodes[secondary])
            if secondary_distance > radius or sister_distance > 1.5 * radius:
                continue
            branches = sorted((primary, secondary))
            key = stable_key(identity, parent, branches)
            branch_set = set(branches)
            conflicts = [
                [source, target]
                for source, target in sorted(edges)
                if source == parent or target in branch_set
            ]
            rows.append(
                {
                    "candidate_key": key,
                    "identity": identity,
                    "dataset": graph["dataset"],
                    "action": {
                        "kind": "division_fork",
                        "add_edges": [[parent, branch] for branch in branches],
                        "replace_outgoing_from": [parent],
                        "replace_incoming_to": branches,
                        "expected_conflict_edges": conflicts,
                    },
                    "patch_nodes": {
                        "parent": parent,
                        "daughter_1": branches[0],
                        "daughter_2": branches[1],
                    },
                    "features": {
                        "geometry_cost": (
                            primary_distance + secondary_distance + sister_distance
                        )
                        / radius,
                        "radius_um": radius,
                    },
                }
            )
    rows.sort(key=lambda row: (float(row["features"]["geometry_cost"]), row["candidate_key"]))
    return rows[:APPEARANCE_CANDIDATE_CAP], {
        "radius_um": radius,
        "raw_candidates": len(rows),
        "cached_candidates": min(len(rows), APPEARANCE_CANDIDATE_CAP),
    }


def patch_summary(array: Any, node: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    t = int(node["t"])
    z, y, x = (int(round(float(node[axis]))) for axis in ("z", "y", "x"))
    if not (0 <= t < int(array.shape[0])):
        raise PostprocessError("node time lies outside raw image")
    slices = (
        slice(max(0, z - 2), min(int(array.shape[1]), z + 3)),
        slice(max(0, y - 12), min(int(array.shape[2]), y + 13)),
        slice(max(0, x - 12), min(int(array.shape[3]), x + 13)),
    )
    patch = np.asarray(array[(t, *slices)], dtype=np.float32)
    if patch.size == 0:
        raise PostprocessError("empty appearance patch")
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
    result = {
        "background_p20": background,
        "mass_above_background": mass,
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "voxel_count": int(patch.size),
        "xy_roundness": roundness,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise PostprocessError("raw patch summary contains a nonfinite value")
    return result


def appearance_features(candidate: Mapping[str, Any]) -> dict[str, float]:
    summaries = candidate["intensity_summaries"]
    mass_1 = float(summaries["daughter_1"]["mass_above_background"])
    mass_2 = float(summaries["daughter_2"]["mass_above_background"])
    parent_mass = float(summaries["parent"]["mass_above_background"])
    larger = max(mass_1, mass_2)
    balance = min(mass_1, mass_2) / larger if larger > 0.0 else 0.0
    ratio = (mass_1 + mass_2) / parent_mass if parent_mass > 0.0 else float("inf")
    mass_error = abs(math.log(ratio)) if ratio > 0.0 and math.isfinite(ratio) else float("inf")
    result = {
        "daughter_mass_balance": balance,
        "daughter_parent_mass_ratio": ratio,
        "parent_xy_roundness": float(summaries["parent"]["xy_roundness"]),
        "absolute_log_mass_error": mass_error,
        "geometry_cost": float(candidate["features"]["geometry_cost"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise PostprocessError("appearance cache contains a nonfinite derived feature")
    return result


def _ordered_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = dict(raw)
        features = appearance_features(candidate)
        if not (
            features["daughter_mass_balance"] >= float(POLICY["balance_min"])
            and features["daughter_parent_mass_ratio"] >= float(POLICY["ratio_min"])
            and features["parent_xy_roundness"] >= float(POLICY["roundness_min"])
        ):
            continue
        candidate["g04_appearance_features"] = features
        eligible.append(candidate)
    return sorted(
        eligible,
        key=lambda row: (
            row["g04_appearance_features"]["geometry_cost"],
            row["g04_appearance_features"]["absolute_log_mass_error"],
            str(row["candidate_key"]),
        ),
    )


def build_disjoint_fork_graph(
    baseline: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = baseline["nodes"]
    baseline_edges = set(baseline["edges"])
    occupied_nodes: set[int] = set()
    owned_conflicts: set[tuple[int, int]] = set()
    selected: list[dict[str, Any]] = []
    incompatible = 0
    overlap_suppressed = 0
    for candidate in _ordered_candidates(candidates):
        action = candidate["action"]
        add_edges = {tuple(int(value) for value in pair) for pair in action["add_edges"]}
        parents = {int(value) for value in action["replace_outgoing_from"]}
        daughters = {int(value) for value in action["replace_incoming_to"]}
        boundary = parents | daughters
        if len(add_edges) != 2 or len(parents) != 1 or len(daughters) != 2:
            incompatible += 1
            continue
        if not boundary <= set(nodes):
            incompatible += 1
            continue
        if any(int(nodes[target]["t"]) != int(nodes[source]["t"]) + 1 for source, target in add_edges):
            incompatible += 1
            continue
        if boundary & occupied_nodes:
            overlap_suppressed += 1
            continue
        actual_conflicts = {
            pair for pair in baseline_edges if pair[0] in parents or pair[1] in daughters
        }
        if actual_conflicts & owned_conflicts:
            overlap_suppressed += 1
            continue
        selected.append(
            {
                "candidate_key": str(candidate["candidate_key"]),
                "add_edges": add_edges,
                "actual_conflicts": actual_conflicts,
                "boundary": boundary,
            }
        )
        occupied_nodes.update(boundary)
        owned_conflicts.update(actual_conflicts)
        if len(selected) >= int(POLICY["cap"]):
            break
    add_union = {pair for row in selected for pair in row["add_edges"]}
    remove_union = {pair for row in selected for pair in row["actual_conflicts"]}
    final_edges = (baseline_edges - remove_union) | add_union
    output = {"dataset": baseline["dataset"], "nodes": dict(nodes), "edges": final_edges}
    outdegree: dict[int, int] = defaultdict(int)
    for source, _target in final_edges:
        outdegree[source] += 1
    return output, {
        "selected_action_count": len(selected),
        "selected_candidate_keys": [row["candidate_key"] for row in selected],
        "incompatible_candidates_seen": incompatible,
        "overlap_suppressed": overlap_suppressed,
        "removed_edges": len(remove_union),
        "added_edges": len(add_union),
        "predicted_forks": sum(value == 2 for value in outdegree.values()),
    }


class RawArrayProvider:
    """Open raw arrays from either a Kaggle directory mount or the local ZIP."""

    def __init__(self, input_root: Path):
        self.input_root = input_root.resolve()
        self._store: Any | None = None

    def inventory(self) -> set[str]:
        if self.input_root.is_file():
            if not zipfile.is_zipfile(self.input_root):
                raise PostprocessError("raw input file is not a ZIP archive")
            with zipfile.ZipFile(self.input_root, "r") as archive:
                result = set()
                for name in archive.namelist():
                    parts = name.split("/")
                    if len(parts) == 4 and parts[0] == "test" and parts[1].endswith(".zarr") and parts[2:] == ["0", "zarr.json"]:
                        result.add(_validate_dataset(parts[1][:-5]))
                return result
        test_root = self.input_root / "test"
        if not test_root.is_dir():
            raise PostprocessError(f"raw input has no test directory: {self.input_root}")
        return {
            _validate_dataset(path.name[:-5])
            for path in test_root.iterdir()
            if path.is_dir() and path.name.endswith(".zarr")
        }

    def __call__(self, dataset: str) -> Any:
        import zarr  # pylint: disable=import-outside-toplevel

        if self.input_root.is_file():
            if self._store is None:
                self._store = zarr.storage.ZipStore(self.input_root, mode="r")
            return zarr.open_array(
                store=self._store,
                path=f"test/{dataset}.zarr/0",
                mode="r",
                zarr_format=3,
            )
        return zarr.open_array(
            str(self.input_root / "test" / f"{dataset}.zarr" / "0"),
            mode="r",
            zarr_format=3,
        )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


class CachedSummaryProvider:
    """Public-parity provider for immutable label-free summary receipts."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir.resolve()

    def inventory(self) -> set[str]:
        return {path.stem for path in self.cache_dir.glob("*.json") if path.is_file()}

    def candidates(
        self, dataset: str, expected: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        path = self.cache_dir / f"{dataset}.json"
        payload = json.loads(path.read_text(encoding="ascii"))
        if (
            payload.get("dataset") != dataset
            or payload.get("contains_labels") is not False
            or payload.get("contains_voxels") is not False
        ):
            raise PostprocessError(f"summary cache provenance drift: {path}")
        cached = payload.get("candidates")
        if not isinstance(cached, list):
            raise PostprocessError(f"summary cache candidates absent: {path}")
        by_key = {str(row.get("candidate_key")): row for row in cached if isinstance(row, dict)}
        expected_keys = [str(row["candidate_key"]) for row in expected]
        if set(by_key) != set(expected_keys) or len(by_key) != len(cached):
            raise PostprocessError(f"summary cache candidate inventory drift: {path}")
        result = []
        for row in expected:
            cached_row = by_key[str(row["candidate_key"])]
            if cached_row.get("patch_nodes") != row.get("patch_nodes"):
                raise PostprocessError(f"summary cache node binding drift: {path}")
            enriched = dict(row)
            enriched["intensity_summaries"] = cached_row["intensity_summaries"]
            result.append(enriched)
        return result


def postprocess_submission(
    parent_path: Path,
    output_path: Path,
    *,
    input_root: Path | None = None,
    summary_cache: Path | None = None,
    receipt_path: Path | None = None,
    expected_output_sha256: str | None = None,
    deadline_seconds: float = 300.0,
) -> dict[str, Any]:
    """Transform a freshly generated parent and emit a byte-authenticated receipt."""

    started = time.monotonic()
    deadline = started + float(deadline_seconds)
    if deadline_seconds <= 0.0:
        raise PostprocessError("deadline_seconds must be positive")
    parent_path = parent_path.resolve()
    output_path = output_path.resolve()
    if not parent_path.is_file() or output_path.exists() or parent_path == output_path:
        raise PostprocessError("parent must exist and output must be a distinct absent path")
    if (input_root is None) == (summary_cache is None):
        raise PostprocessError("select exactly one raw-input or summary-cache provider")
    graphs, parent_rows = read_submission(parent_path)
    parent_totals = validate_graphs(graphs)
    parent_sha = sha256_file(parent_path)

    raw_provider = RawArrayProvider(input_root) if input_root is not None else None
    cache_provider = CachedSummaryProvider(summary_cache) if summary_cache is not None else None
    provider_inventory = (
        raw_provider.inventory() if raw_provider is not None else cache_provider.inventory()  # type: ignore[union-attr]
    )
    if provider_inventory != set(graphs):
        raise PostprocessError(
            "parent/raw dataset inventory mismatch: "
            f"parent={sorted(graphs)} provider={sorted(provider_inventory)}"
        )
    outputs: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    try:
        for dataset in sorted(graphs):
            if time.monotonic() >= deadline:
                raise PostprocessError("postprocess deadline expired")
            graph = graphs[dataset]
            candidates, prefilter = appearance_prefilter(graph)
            if raw_provider is not None:
                array = raw_provider(dataset)
                enriched = []
                for candidate in candidates:
                    if time.monotonic() >= deadline:
                        raise PostprocessError("postprocess deadline expired during patch summaries")
                    record = dict(candidate)
                    record["intensity_summaries"] = {
                        role: patch_summary(array, graph["nodes"][int(node_id)])
                        for role, node_id in candidate["patch_nodes"].items()
                    }
                    enriched.append(record)
            else:
                enriched = cache_provider.candidates(dataset, candidates)  # type: ignore[union-attr]
            transformed, diagnostic = build_disjoint_fork_graph(graph, enriched)
            validate_graphs({dataset: transformed})
            outputs[dataset] = transformed
            diagnostics.append(
                {
                    "identity": f"test:{dataset}",
                    "computed_raw_intensity_summaries": raw_provider is not None,
                    "labels_read": False,
                    "public_identity_pinned": False,
                    "prefilter": prefilter,
                    **diagnostic,
                }
            )
    finally:
        if raw_provider is not None:
            raw_provider.close()
    output_rows = _write_submission(outputs, output_path)
    reread, reread_rows = read_submission(output_path)
    if reread_rows != output_rows or set(reread) != set(graphs):
        raise PostprocessError("written output failed semantic round trip")
    output_sha = sha256_file(output_path)
    if expected_output_sha256 is not None and output_sha != expected_output_sha256:
        raise PostprocessError(
            f"public parity SHA mismatch: expected {expected_output_sha256}, got {output_sha}"
        )
    final_totals = validate_graphs(reread)
    selected_actions = sum(int(row["selected_action_count"]) for row in diagnostics)
    receipt = {
        "schema": "celltrack-alpha.g04-hidden-division-postprocess.v1",
        "status": "PASS",
        "policy": POLICY,
        "dataset_discovery": {
            "source": "fresh_parent_csv_and_mounted_test_inventory",
            "datasets": sorted(graphs),
            "dataset_count": len(graphs),
            "public_identity_pinned": False,
            "inventory_exact_match": True,
        },
        "parent": {
            "path": str(parent_path),
            "sha256": parent_sha,
            "bytes": parent_path.stat().st_size,
            "rows": parent_rows,
            **parent_totals,
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha,
            "bytes": output_path.stat().st_size,
            "rows": output_rows,
            **final_totals,
        },
        "transform": {
            "selected_actions": selected_actions,
            "labels_read": False,
            "computed_raw_intensity_summaries": raw_provider is not None,
            "summary_cache_used": cache_provider is not None,
            "per_dataset_cap": POLICY["cap"],
            "observation_distinct_from_parent": output_sha != parent_sha,
        },
        "per_dataset": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
    }
    if receipt_path is not None:
        atomic_bytes(receipt_path.resolve(), canonical_bytes(receipt))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    provider = parser.add_mutually_exclusive_group(required=True)
    provider.add_argument("--input-root", type=Path)
    provider.add_argument("--summary-cache", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    receipt = postprocess_submission(
        args.parent,
        args.output,
        input_root=args.input_root,
        summary_cache=args.summary_cache,
        receipt_path=args.receipt,
        expected_output_sha256=args.expected_output_sha256,
        deadline_seconds=args.deadline_seconds,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
