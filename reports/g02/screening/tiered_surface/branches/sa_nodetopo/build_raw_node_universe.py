#!/usr/bin/env python3
"""Build and stress the preregistered authentic 44b6 raw-image node universe.

This runner is intentionally branch-local.  It reads the authoritative base tree
through absolute paths, streams one raw Zarr frame at a time, and writes only below
the SA-NODETOPO subtree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking").resolve()
HERE = Path(__file__).resolve().parent
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionCompileError,
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionGraph,
    SubmissionNode,
    read_submission,
)
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_alpha.proposals import detect_candidates_one_timepoint  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.sa-nodetopo-raw-universe-result.v2"
PREREG = HERE / "PREREGISTRATION_RAW_UNIVERSE_v2.json"
PREREG_SHA256 = "b85129a75905c8efb29e4068455d76a4da5917b3f0b8ddefa69819710f4afd06"
FAMILY = "44b6"
ROUTE = "balanced_compact"
BUDGETS = (0, 1, 2, 5, 10, 20, 50)
POSITIVE_RADIUS_UM = 7.0
HARD_FALSE_RADIUS_UM = 14.0
TOPOLOGY_RADIUS_UM = 8.0
PER_VIDEO_HEAP_CAP = 4096


class NodeUniverseError(RuntimeError):
    """The frozen raw-universe contract failed closed."""


@dataclass(frozen=True)
class CandidateRow:
    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int
    score_hex: str
    sources: tuple[str, ...]

    @property
    def coord(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x

    def payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "node_id": self.node_id,
            "score_hex": self.score_hex,
            "sources": list(self.sources),
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "family": FAMILY,
            "node_id": self.node_id,
            "sources": list(self.sources),
            "split": "train",
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.identity_payload())).hexdigest()


@dataclass(frozen=True)
class ActionRecord:
    identity: str
    candidate_identity_sha256: str
    action: GraphAction
    removed_edges: tuple[tuple[int, int], ...]

    @property
    def order_key(self) -> tuple[str, str]:
        return self.candidate_identity_sha256, self.identity


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NodeUniverseError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise NodeUniverseError(f"{label} must be a JSON object")
    return value


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise NodeUniverseError(f"{label} hash mismatch: {path}")


def central_directory_identity(path: Path) -> tuple[str, int]:
    with zipfile.ZipFile(path) as archive:
        rows = [
            {
                "compressed_size": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "filename": info.filename,
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
        ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest(), len(rows)


def authenticate() -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any]]:
    if (
        os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
        or os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
    ):
        raise NodeUniverseError("continuous zero-swap guard is not active")
    verify_file(PREREG, PREREG_SHA256, "preregistration")
    prereg = load_json(PREREG, "preregistration")
    if prereg.get("status") != "FROZEN_BEFORE_RAW_PROPOSAL_OR_STRESS_OUTCOMES":
        raise NodeUniverseError("preregistration status drifted")
    for name, pin in prereg["frozen_inputs"].items():
        if name in {"raw_archive", "passing_probe"}:
            continue
        verify_file(Path(str(pin["path"])), str(pin["sha256"]), name)
    probe = prereg["frozen_inputs"]["passing_probe"]
    verify_file(Path(str(probe["path"])), str(probe["sha256"]), "passing probe")
    archive_pin = prereg["frozen_inputs"]["raw_archive"]
    archive = Path(str(archive_pin["path"]))
    if not archive.is_file() or archive.stat().st_size != int(archive_pin["bytes"]):
        raise NodeUniverseError("raw archive size/identity drifted")
    central_sha, entry_count = central_directory_identity(archive)
    if (
        central_sha != archive_pin["central_directory_sha256"]
        or entry_count != int(archive_pin["entry_count"])
    ):
        raise NodeUniverseError("raw archive central-directory identity drifted")
    p0 = load_json(Path(prereg["frozen_inputs"]["p0"]["path"]), "P0")
    archive_row = p0.get("inputs", {}).get("archive", {})
    if (
        archive_row.get("sha256") != archive_pin["sha256_banked_in_p0"]
        or archive_row.get("bytes") != archive_pin["bytes"]
    ):
        raise NodeUniverseError("P0 raw-archive pin disagrees with preregistration")
    identity_report = load_json(
        Path(prereg["frozen_inputs"]["fit105_identity_source"]["path"]),
        "fit105 identity source",
    )
    identities = tuple(
        sorted(
            str(row["identity"])
            for row in identity_report.get("files", ())
            if str(row.get("identity", "")).startswith(f"train:{FAMILY}_")
        )
    )
    if len(identities) != 36 or len(set(identities)) != 36:
        raise NodeUniverseError("fixed 44b6 identity set is not exactly 36 videos")
    manifest = load_json(
        Path(prereg["frozen_inputs"]["fit117_baseline_manifest"]["path"]),
        "fit117 baseline manifest",
    )
    if any(identity not in manifest.get("videos", {}) for identity in identities):
        raise NodeUniverseError("fit117 manifest does not cover fixed 44b6 identities")
    return prereg, identities, manifest


def candidate_from_node(node: Any) -> CandidateRow:
    return CandidateRow(
        dataset=str(node.dataset),
        node_id=int(node.node_id),
        t=int(node.t),
        z=int(node.z),
        y=int(node.y),
        x=int(node.x),
        score_hex=float(node.score).hex(),
        sources=tuple(sorted(str(value) for value in node.sources)),
    )


def gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(canonical_json_bytes(dict(row)))
                count += 1
    return sha256_file(path), count


def positive_and_hard_false_masks(
    candidates: Sequence[CandidateRow],
    gt_coords: np.ndarray,
    scale_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidates)
    positive = np.zeros(count, dtype=bool)
    hard_false = np.zeros(count, dtype=bool)
    nearest = np.full(count, np.inf, dtype=np.float64)
    if count == 0:
        return positive, hard_false, nearest
    if len(gt_coords) == 0:
        hard_false[:] = True
        return positive, hard_false, nearest
    candidate_coords = np.asarray([row.coord for row in candidates], dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    distances = np.sqrt(
        np.sum(
            ((np.asarray(gt_coords, dtype=np.float64)[:, None, :] - candidate_coords[None, :, :]) * scale) ** 2,
            axis=2,
        )
    )
    nearest[:] = np.min(distances, axis=0)
    cost = distances.copy()
    cost[cost > POSITIVE_RADIUS_UM] = 1.0e12
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols, strict=True):
        if distances[int(row), int(col)] <= POSITIVE_RADIUS_UM:
            positive[int(col)] = True
    hard_false[:] = (~positive) & (nearest > HARD_FALSE_RADIUS_UM)
    return positive, hard_false, nearest


def graph_indices(
    graph: SubmissionGraph,
    scale_zyx: tuple[float, float, float],
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[int, ...]],
    dict[int, tuple[int, ...]],
    set[tuple[int, int]],
]:
    by_t: dict[int, list[SubmissionNode]] = defaultdict(list)
    for node in graph.nodes.values():
        by_t[int(node.t)].append(node)
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scale = np.asarray(scale_zyx, dtype=np.float64)
    for t, nodes in by_t.items():
        ordered = sorted(nodes, key=lambda value: int(value.node_id))
        ids = np.asarray([int(node.node_id) for node in ordered], dtype=np.int64)
        coords = np.asarray([node.coord_zyx for node in ordered], dtype=np.float64) * scale
        frames[t] = ids, coords
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
        pair = (int(edge.source_id), int(edge.target_id))
        edges.add(pair)
        outgoing[pair[0]].append(pair[1])
        incoming[pair[1]].append(pair[0])
    return (
        frames,
        {key: tuple(sorted(value)) for key, value in outgoing.items()},
        {key: tuple(sorted(value)) for key, value in incoming.items()},
        edges,
    )


def nearest_node(
    frame: tuple[np.ndarray, np.ndarray] | None,
    coord_um: np.ndarray,
) -> tuple[int, float] | None:
    if frame is None or len(frame[0]) == 0:
        return None
    ids, coords = frame
    tree = cKDTree(coords)
    distance, index = tree.query(coord_um, k=1)
    return int(ids[int(index)]), float(distance)


def build_action(
    identity: str,
    candidate: CandidateRow,
    graph: SubmissionGraph,
    scale_zyx: tuple[float, float, float],
    indices: tuple[
        dict[int, tuple[np.ndarray, np.ndarray]],
        dict[int, tuple[int, ...]],
        dict[int, tuple[int, ...]],
        set[tuple[int, int]],
    ],
) -> ActionRecord | None:
    frames, _outgoing, _incoming, edges = indices
    if candidate.t <= 0 or candidate.t >= max(frames, default=0):
        return None
    coord_um = np.asarray(candidate.coord, dtype=np.float64) * np.asarray(
        scale_zyx, dtype=np.float64
    )
    predecessor = nearest_node(frames.get(candidate.t - 1), coord_um)
    successor = nearest_node(frames.get(candidate.t + 1), coord_um)
    if (
        predecessor is None
        or successor is None
        or predecessor[1] > TOPOLOGY_RADIUS_UM
        or successor[1] > TOPOLOGY_RADIUS_UM
    ):
        return None
    source = predecessor[0]
    target = successor[0]
    new_id = max(graph.nodes) + int(candidate.node_id)
    conflicts = tuple(
        sorted(pair for pair in edges if pair[0] == source or pair[1] == target)
    )
    action = GraphAction(
        action_id=f"false.{candidate.identity_sha256[:24]}",
        kind=ActionKind.RESTORE_TOPOLOGY,
        add_nodes=(
            SubmissionNode(
                new_id,
                candidate.t,
                candidate.z,
                candidate.y,
                candidate.x,
            ),
        ),
        add_edges=((source, new_id), (new_id, target)),
        replace_outgoing_from=(source,),
        replace_incoming_to=(target,),
        expected_conflict_edges=conflicts,
    )
    compiled = compile_action(graph, action)
    return ActionRecord(
        identity=identity,
        candidate_identity_sha256=candidate.identity_sha256,
        action=action,
        removed_edges=compiled.removed_edges,
    )


def heap_keep_smallest(
    heap: list[tuple[int, str, ActionRecord]], record: ActionRecord, cap: int
) -> None:
    numeric = int(record.candidate_identity_sha256, 16)
    item = (-numeric, record.identity, record)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif numeric < -heap[0][0]:
        heapq.heapreplace(heap, item)


def mutually_compatible(
    ordered: Sequence[ActionRecord],
    limit: int,
) -> tuple[list[ActionRecord], Counter[str]]:
    accepted: list[ActionRecord] = []
    rejected: Counter[str] = Counter()
    owners: dict[str, dict[Any, str]] = defaultdict(dict)
    for record in ordered:
        action = record.action
        conflicts: list[str] = []
        primitives = {
            "node": tuple(int(node.node_id) for node in action.add_nodes),
            "edge": tuple(action.add_edges),
            "removed": record.removed_edges,
            "source": tuple(action.replace_outgoing_from),
            "target": tuple(action.replace_incoming_to),
        }
        for kind, values in primitives.items():
            if any((record.identity, value) in owners[kind] for value in values):
                conflicts.append(kind)
        if conflicts:
            rejected.update({f"ownership_{kind}": 1 for kind in sorted(set(conflicts))})
            continue
        for kind, values in primitives.items():
            for value in values:
                owners[kind][(record.identity, value)] = action.action_id
        accepted.append(record)
        if len(accepted) >= limit:
            break
    return accepted, rejected


def action_payload(record: ActionRecord) -> dict[str, Any]:
    action = record.action
    return {
        "action_id": action.action_id,
        "add_edges": [list(pair) for pair in action.add_edges],
        "add_nodes": [
            {
                "node_id": int(node.node_id),
                "t": int(node.t),
                "x": int(node.x),
                "y": int(node.y),
                "z": int(node.z),
            }
            for node in action.add_nodes
        ],
        "candidate_identity_sha256": record.candidate_identity_sha256,
        "expected_conflict_edges": [
            list(pair) for pair in action.expected_conflict_edges
        ],
        "identity": record.identity,
        "kind": action.kind.value,
        "replace_incoming_to": list(action.replace_incoming_to),
        "replace_outgoing_from": list(action.replace_outgoing_from),
        "removed_edges": [list(pair) for pair in record.removed_edges],
    }


def graph_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return scoreall._prediction_rows(graph.dataset, graph)


def metric_vector(summary: Mapping[str, Any]) -> dict[str, float]:
    adjusted = float(summary["adj_edge_jaccard"])
    division = float(summary.get("division_jaccard", float("nan")))
    return {
        "adjusted": adjusted,
        "raw": float(summary["edge_jaccard"]),
        "node_recall": float(summary["node_recall"]),
        "division": division,
        "score": adjusted + (0.0 if math.isnan(division) else 0.1 * division),
    }


def score_curve(
    archive: Path,
    identities: Sequence[str],
    manifest: Mapping[str, Any],
    selected: Sequence[ActionRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions_by_identity: dict[str, list[ActionRecord]] = defaultdict(list)
    baseline_graphs: dict[str, SubmissionGraph] = {}
    baseline_rows: dict[str, dict[str, Any]] = {}
    scales: dict[str, tuple[float, float, float]] = {}
    gt_by_identity: dict[str, dict[str, Any]] = {}
    with ZipStore(archive, mode="r") as store:
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            fragment = ROOT / "reports/g02/fitcache_117/bh9out/fragments" / f"{slug}.csv"
            expected = manifest["videos"][identity]["files"]["fragments"]["sha256"]
            verify_file(fragment, str(expected), f"baseline fragment {identity}")
            graphs = read_submission(fragment)
            graph = graphs.get(dataset)
            if graph is None or len(graphs) != 1:
                raise NodeUniverseError(f"baseline identity mismatch: {identity}")
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            nodes, edges = graph_rows(graph)
            baseline_graphs[identity] = graph
            scales[identity] = scale
            gt_by_identity[identity] = gt
            baseline_rows[identity] = official_metric_row(nodes, edges, gt, scale)
    baseline_summary = official_summary([baseline_rows[value] for value in identities])
    baseline_vector = metric_vector(baseline_summary)

    cache: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    curve: list[dict[str, Any]] = []
    for budget in BUDGETS:
        chosen = list(selected[:budget])
        by_identity: dict[str, list[ActionRecord]] = defaultdict(list)
        for record in chosen:
            by_identity[record.identity].append(record)
        rows: list[dict[str, Any]] = []
        hashes: dict[str, str] = {}
        batch_hashes: dict[str, str] = {}
        for identity in identities:
            records = by_identity.get(identity, [])
            if not records:
                rows.append(baseline_rows[identity])
                hashes[identity] = graph_sha256(baseline_graphs[identity])
                continue
            key = (identity, len(records))
            cached = cache.get(key)
            if cached is None:
                compiled = compile_batch(
                    baseline_graphs[identity],
                    tuple(record.action for record in records),
                )
                nodes, edges = graph_rows(compiled.graph)
                metric = official_metric_row(
                    nodes,
                    edges,
                    gt_by_identity[identity],
                    scales[identity],
                )
                cached = (
                    metric,
                    {
                        "batch_sha256": compiled.batch_sha256,
                        "graph_sha256": compiled.after_graph_sha256,
                    },
                )
                cache[key] = cached
            rows.append(cached[0])
            hashes[identity] = str(cached[1]["graph_sha256"])
            batch_hashes[identity] = str(cached[1]["batch_sha256"])
        summary = official_summary(rows)
        vector = metric_vector(summary)
        curve.append(
            {
                "budget": budget,
                "selected_action_count": len(chosen),
                "affected_video_count": len(by_identity),
                "affected_action_counts": {
                    key: len(value) for key, value in sorted(by_identity.items())
                },
                "batch_sha256s": batch_hashes,
                "graph_sha256s": hashes,
                "metrics": vector,
                "delta": {
                    key: vector[key] - baseline_vector[key]
                    for key in ("adjusted", "raw", "node_recall", "score")
                },
            }
        )
    return curve, {
        "summary": baseline_summary,
        "vector": baseline_vector,
        "video_rows": baseline_rows,
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise NodeUniverseError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    prereg, identities, manifest = authenticate()
    archive = Path(prereg["frozen_inputs"]["raw_archive"]["path"])
    route = default_route_configs()[ROUTE]
    expected_route = prereg["raw_proposal_contract"]
    if (
        tuple(route.detector_sources) != tuple(expected_route["detector_sources"])
        or tuple(route.threshold_quantiles) != tuple(expected_route["threshold_quantiles"])
        or route.max_points_per_t != expected_route["max_points_per_timepoint"]
    ):
        raise NodeUniverseError("balanced_compact route content drifted")

    video_reports: list[dict[str, Any]] = []
    action_heaps: dict[str, list[tuple[int, str, ActionRecord]]] = {}
    total_support = Counter()
    with ZipStore(archive, mode="r") as store:
        for video_index, identity in enumerate(identities, start=1):
            video_started = time.perf_counter()
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            attrs = dict(group.attrs)
            raw_array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(attrs))
            gt = read_geff_arrays(store, "train", dataset)
            gt_by_t: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            for t, z, y, x in zip(
                gt["nodes"]["t"],
                gt["nodes"]["z"],
                gt["nodes"]["y"],
                gt["nodes"]["x"],
                strict=True,
            ):
                gt_by_t[int(t)].append((int(z), int(y), int(x)))

            slug = identity.replace(":", "__", 1)
            fragment = ROOT / "reports/g02/fitcache_117/bh9out/fragments" / f"{slug}.csv"
            expected_fragment = manifest["videos"][identity]["files"]["fragments"]["sha256"]
            verify_file(fragment, str(expected_fragment), f"baseline fragment {identity}")
            graphs = read_submission(fragment)
            graph = graphs.get(dataset)
            if graph is None or len(graphs) != 1:
                raise NodeUniverseError(f"baseline graph identity mismatch: {identity}")
            indices = graph_indices(graph, scale)

            bank_path = output / "raw_universe" / "videos" / f"{slug}.jsonl.gz"
            frame_counts: list[int] = []
            source_counts: Counter[str] = Counter()
            support = Counter()
            next_node_id = 1
            frame_signatures: dict[int, str] = {}
            heap: list[tuple[int, str, ActionRecord]] = []
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            with bank_path.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_handle, mtime=0
                ) as zipped:
                    for t in range(int(raw_array.shape[0])):
                        volume = np.asarray(raw_array[t])
                        nodes, next_node_id = detect_candidates_one_timepoint(
                            volume=volume,
                            dataset=dataset,
                            t=t,
                            attrs=attrs,
                            scale_zyx=scale,
                            config=route,
                            next_node_id=next_node_id,
                        )
                        candidates = [candidate_from_node(node) for node in nodes]
                        frame_counts.append(len(candidates))
                        frame_hasher = hashlib.sha256()
                        for candidate in candidates:
                            encoded = canonical_json_bytes(candidate.payload())
                            zipped.write(encoded)
                            frame_hasher.update(encoded)
                            source_counts.update(candidate.sources)
                        if t in (0, int(raw_array.shape[0]) - 1):
                            frame_signatures[t] = frame_hasher.hexdigest()
                        gt_coords = np.asarray(
                            gt_by_t.get(t, ()), dtype=np.float64
                        ).reshape((-1, 3))
                        positive, hard_false, nearest = positive_and_hard_false_masks(
                            candidates, gt_coords, scale
                        )
                        support["raw_candidates"] += len(candidates)
                        support["positive_assignments_7um"] += int(np.sum(positive))
                        support["background_candidates"] += int(np.sum(~positive))
                        support["hard_false_candidates_gt14um"] += int(
                            np.sum(hard_false)
                        )
                        support["finite_nearest_gt"] += int(np.isfinite(nearest).sum())
                        for candidate, is_hard_false in zip(
                            candidates, hard_false, strict=True
                        ):
                            if not bool(is_hard_false):
                                continue
                            try:
                                record = build_action(
                                    identity,
                                    candidate,
                                    graph,
                                    scale,
                                    indices,
                                )
                            except ActionCompileError as exc:
                                raise NodeUniverseError(
                                    f"frozen N1 rejected a constructed action for {identity}: {exc}"
                                ) from exc
                            if record is not None:
                                support["topology_joinable_hard_false"] += 1
                                heap_keep_smallest(heap, record, PER_VIDEO_HEAP_CAP)
                        del volume, nodes, candidates, positive, hard_false, nearest
            action_heaps[identity] = heap

            replay: dict[str, Any] = {}
            for t in sorted(frame_signatures):
                volume = np.asarray(raw_array[t])
                replay_nodes, _ = detect_candidates_one_timepoint(
                    volume=volume,
                    dataset=dataset,
                    t=t,
                    attrs=attrs,
                    scale_zyx=scale,
                    config=route,
                    next_node_id=1 + sum(frame_counts[:t]),
                )
                replay_hasher = hashlib.sha256()
                for node in replay_nodes:
                    replay_hasher.update(
                        canonical_json_bytes(candidate_from_node(node).payload())
                    )
                actual = replay_hasher.hexdigest()
                replay[str(t)] = {
                    "expected_sha256": frame_signatures[t],
                    "actual_sha256": actual,
                    "equal": actual == frame_signatures[t],
                }
                if actual != frame_signatures[t]:
                    raise NodeUniverseError(f"deterministic replay mismatch: {identity} t={t}")

            bank_sha = sha256_file(bank_path)
            total_support.update(support)
            video_reports.append(
                {
                    "identity": identity,
                    "raw_zarr_path": f"train/{dataset}.zarr/0",
                    "shape": [int(value) for value in raw_array.shape],
                    "dtype": str(raw_array.dtype),
                    "scale_zyx_um": list(scale),
                    "frame_count": int(raw_array.shape[0]),
                    "frame_candidate_counts": frame_counts,
                    "candidate_count": int(sum(frame_counts)),
                    "source_counts": dict(sorted(source_counts.items())),
                    "support": dict(support),
                    "raw_bank_path": str(bank_path),
                    "raw_bank_sha256": bank_sha,
                    "raw_bank_bytes": bank_path.stat().st_size,
                    "replay": replay,
                    "baseline_fragment_path": str(fragment),
                    "baseline_fragment_sha256": str(expected_fragment),
                    "baseline_graph_sha256": graph_sha256(graph),
                    "elapsed_seconds": time.perf_counter() - video_started,
                }
            )
            print(
                f"raw-universe {video_index:02d}/36 {identity} "
                f"frames={raw_array.shape[0]} candidates={sum(frame_counts)} "
                f"hard_false={support['hard_false_candidates_gt14um']} "
                f"joinable={support['topology_joinable_hard_false']}",
                flush=True,
            )
            del group, raw_array, gt, graph, indices

    pooled: list[ActionRecord] = []
    heap_cutoffs: dict[str, str | None] = {}
    for identity, heap in action_heaps.items():
        records = [item[2] for item in heap]
        records.sort(key=lambda value: value.order_key)
        pooled.extend(records)
        heap_cutoffs[identity] = (
            records[-1].candidate_identity_sha256
            if len(records) == PER_VIDEO_HEAP_CAP
            else None
        )
    pooled.sort(key=lambda value: value.order_key)
    selected, rejected = mutually_compatible(pooled, max(BUDGETS))
    if len(selected) < max(BUDGETS):
        raise NodeUniverseError(
            f"only {len(selected)} mutually compatible actions; need {max(BUDGETS)}"
        )
    minimum_truncation_cutoff = min(
        (value for value in heap_cutoffs.values() if value is not None),
        default=None,
    )
    if (
        minimum_truncation_cutoff is not None
        and selected[-1].candidate_identity_sha256 >= minimum_truncation_cutoff
    ):
        raise NodeUniverseError(
            "heap cap cannot prove the first 50 compatible actions are globally exact"
        )

    action_bank_path = output / "false_node_action_bank.jsonl.gz"
    action_bank_sha, action_bank_count = gzip_jsonl(
        action_bank_path, (action_payload(record) for record in selected)
    )
    curve, baseline = score_curve(archive, identities, manifest, selected)

    raw_manifest = {
        "schema": "celltrack-alpha.sa-nodetopo-raw-universe-manifest.v1",
        "status": "COMPLETE_AUTHENTIC_44B6_RAW_UNIVERSE",
        "family": FAMILY,
        "identity_count": len(identities),
        "identities": list(identities),
        "route": ROUTE,
        "proposal_implementation_sha256": prereg["frozen_inputs"][
            "proposal_implementation"
        ]["sha256"],
        "route_implementation_sha256": prereg["frozen_inputs"][
            "route_implementation"
        ]["sha256"],
        "archive": prereg["frozen_inputs"]["raw_archive"],
        "totals": dict(total_support),
        "videos": video_reports,
    }
    raw_manifest_path = output / "raw_universe" / "manifest.json"
    raw_manifest_path.write_bytes(canonical_json_bytes(raw_manifest))
    raw_manifest_sha = sha256_file(raw_manifest_path)

    verdict = "GO_AUTHENTIC_UNIVERSE_STRESS_DIAGNOSTIC_COMPLETE"
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "verdict": verdict,
        "preregistration": {
            "path": str(PREREG),
            "sha256": PREREG_SHA256,
            "git_commit": "6085104be5e810c4b15923e3e42be695208b07b4",
            "outcomes_opened_after_commit": True,
        },
        "scope": {
            "fit_only": True,
            "family": FAMILY,
            "second_family_opened": False,
            "audit_used": False,
            "test_split_used": False,
            "cv57_used": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
            "model_fit": False,
            "pseudo_masks_used": False,
            "raw_images_used": True,
        },
        "provenance": {
            "archive_bytes": archive.stat().st_size,
            "archive_central_directory_sha256": prereg["frozen_inputs"][
                "raw_archive"
            ]["central_directory_sha256"],
            "raw_universe_manifest_path": str(raw_manifest_path),
            "raw_universe_manifest_sha256": raw_manifest_sha,
            "video_bank_count": len(video_reports),
            "all_replays_equal": all(
                row["equal"]
                for video in video_reports
                for row in video["replay"].values()
            ),
            "labels_in_raw_banks": False,
            "pseudo_masks": False,
        },
        "support": {
            **dict(total_support),
            "exact_background_definition": (
                "every raw proposal not selected by same-time one-to-one "
                "minimum-distance fit-GT assignment within 7 um"
            ),
            "hard_false_definition": "background with nearest same-time fit GT >14 um",
        },
        "topology": {
            "join_radius_um": TOPOLOGY_RADIUS_UM,
            "candidate_heap_cap_per_video": PER_VIDEO_HEAP_CAP,
            "global_order_exact_through_budget_50": True,
            "pooled_heap_action_count": len(pooled),
            "mutually_compatible_action_count": len(selected),
            "ownership_rejections_before_50": dict(rejected),
            "single_compile_count": int(
                total_support["topology_joinable_hard_false"]
            ),
            "single_compile_reject_count": 0,
            "action_bank_path": str(action_bank_path),
            "action_bank_sha256": action_bank_sha,
            "action_bank_count": action_bank_count,
        },
        "baseline": baseline,
        "false_node_stress_curve": curve,
        "decision": {
            "universe_prerequisite": "GO",
            "topology_contract": "GO",
            "stress_curve": "DIAGNOSTIC_ONLY",
            "tier_move_proposed": False,
            "second_family_gate": "CLOSED",
            "graph_promotion_gate_run": False,
        },
        "quarantine": {
            "memo_c2_values_used": False,
            "memo_branch_b_66_223_used": False,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "stream_one_frame_at_a_time": True,
            "stream_one_video_at_a_time": True,
        },
    }
    result_path = output / "report.json"
    result_path.write_bytes(canonical_json_bytes(result))
    print(json.dumps({"status": "COMPLETE", "verdict": verdict}), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
