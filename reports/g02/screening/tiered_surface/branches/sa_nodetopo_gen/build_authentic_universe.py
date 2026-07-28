#!/usr/bin/env python3
"""Build BLOCKPEAK-RESTORE-V1 and compile its bounded action population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import heapq
import json
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
    sys.path.insert(0, str(location))

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
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


PREREG = HERE / "PREREGISTRATION.json"
PREREG_SHA256 = "c6529b5da9e6bbf76f08067fd68d248a56aa8dcb6104512e91905d8d58ecaa87"
PREREG_COMMIT = "cd7f24d30ed17bac9940432a90213a87a1f68e78"
GENERATOR = "BLOCKPEAK-RESTORE-V1"
FAMILY = "44b6"
BLOCK = (4, 16, 16)
MAX_PER_FRAME = 64
POSITIVE_RADIUS_UM = 7.0
HARD_FALSE_RADIUS_UM = 14.0
TOPOLOGY_RADIUS_UM = 16.0
BUDGETS = (0, 1, 2, 5, 10, 20, 50)
PER_VIDEO_HEAP_CAP = 256


class GeneratorError(RuntimeError):
    """Fail-closed contract error."""


@dataclass(frozen=True)
class Candidate:
    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int
    intensity: int

    @property
    def coord(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            canonical_bytes(
                {
                    "dataset": self.dataset,
                    "family": FAMILY,
                    "generator": GENERATOR,
                    "intensity": self.intensity,
                    "split": "train",
                    "t": self.t,
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                }
            )
        ).hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "identity_sha256": self.identity_sha256,
            "intensity": self.intensity,
            "node_id": self.node_id,
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class ActionRecord:
    identity: str
    candidate: Candidate
    action: GraphAction
    removed_edges: tuple[tuple[int, int], ...]

    @property
    def order_key(self) -> tuple[str, str]:
        return self.candidate.identity_sha256, self.identity


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratorError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GeneratorError(f"JSON root must be an object: {path}")
    return value


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
    return hashlib.sha256(canonical_bytes(rows)).hexdigest(), len(rows)


def authenticate() -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    if (
        os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1"
        or os.environ.get("CELLTRACK_HOLDOUT_ACCESS") != "0"
    ):
        raise GeneratorError("continuous zero-swap guard is not active")
    if sha256_file(PREREG) != PREREG_SHA256:
        raise GeneratorError("preregistration hash mismatch")
    prereg = load_json(PREREG)
    receipts: dict[str, Any] = {}
    for label, pin in prereg["frozen_inputs"].items():
        path = Path(pin["path"])
        if label == "raw_archive":
            if not path.is_file() or path.stat().st_size != int(pin["bytes"]):
                raise GeneratorError("raw archive byte identity mismatch")
            central_sha, count = central_directory_identity(path)
            if central_sha != pin["central_directory_sha256"] or count != pin["entry_count"]:
                raise GeneratorError("raw archive central-directory mismatch")
            receipts[label] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "central_directory_sha256": central_sha,
                "entry_count": count,
                "full_sha256": pin["sha256_banked_in_p0"],
                "full_sha256_receipt_source": prereg["frozen_inputs"]["p0"]["path"],
            }
            continue
        observed = sha256_file(path)
        if observed != pin["sha256"]:
            raise GeneratorError(f"frozen input hash mismatch: {label}")
        receipts[label] = {"path": str(path), "sha256": observed}

    identity_source = load_json(
        Path(prereg["frozen_inputs"]["fit105_identity_source"]["path"])
    )
    identities = tuple(
        sorted(
            str(row["identity"])
            for row in identity_source.get("files", ())
            if str(row.get("identity", "")).startswith(f"train:{FAMILY}_")
        )
    )
    if len(identities) != 36 or len(set(identities)) != 36:
        raise GeneratorError("frozen identity set is not exactly 36 unique 44b6 videos")
    manifest = load_json(
        Path(prereg["frozen_inputs"]["fit117_baseline_manifest"]["path"])
    )
    if any(identity not in manifest.get("videos", {}) for identity in identities):
        raise GeneratorError("baseline manifest does not cover the frozen identities")
    return prereg, identities, manifest, receipts


def threshold_from_attrs(attrs: Mapping[str, Any]) -> int:
    try:
        value = float(attrs["image_statistics"]["quantiles"]["0.999"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GeneratorError("missing finite image_statistics 0.999 quantile") from exc
    if not np.isfinite(value):
        raise GeneratorError("non-finite image_statistics 0.999 quantile")
    return int(np.ceil(value))


def block_peaks(volume: np.ndarray, threshold: int) -> list[tuple[int, int, int, int]]:
    if volume.ndim != 3:
        raise GeneratorError(f"raw frame is not ZYX: shape={volume.shape}")
    bz, by, bx = BLOCK
    z_extent = (volume.shape[0] // bz) * bz
    y_extent = (volume.shape[1] // by) * by
    x_extent = (volume.shape[2] // bx) * bx
    peaks: list[tuple[int, int, int, int]] = []
    for z0 in range(0, z_extent, bz):
        slab = np.asarray(volume[z0 : z0 + bz, :y_extent, :x_extent])
        blocks = (
            slab.reshape(bz, y_extent // by, by, x_extent // bx, bx)
            .transpose(1, 3, 0, 2, 4)
            .reshape(-1, bz * by * bx)
        )
        flat_indices = np.argmax(blocks, axis=1)
        intensities = blocks[np.arange(len(blocks)), flat_indices]
        keep = np.flatnonzero(intensities >= threshold)
        for block_index in keep:
            y_block, x_block = divmod(int(block_index), x_extent // bx)
            local_z, rem = divmod(int(flat_indices[block_index]), by * bx)
            local_y, local_x = divmod(rem, bx)
            peaks.append(
                (
                    int(intensities[block_index]),
                    z0 + local_z,
                    y_block * by + local_y,
                    x_block * bx + local_x,
                )
            )
    peaks.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    return peaks[:MAX_PER_FRAME]


def gzip_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as encoded:
            for row in rows:
                encoded.write(canonical_bytes(dict(row)))
                count += 1
    return sha256_file(path), count


def read_gzip_rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GeneratorError(f"non-object row in {path}")
            yield value


def candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    candidate = Candidate(
        dataset=str(row["dataset"]),
        node_id=int(row["node_id"]),
        t=int(row["t"]),
        z=int(row["z"]),
        y=int(row["y"]),
        x=int(row["x"]),
        intensity=int(row["intensity"]),
    )
    if candidate.identity_sha256 != row["identity_sha256"]:
        raise GeneratorError("candidate identity receipt mismatch")
    return candidate


def generate_raw_banks(
    output: Path,
    archive: Path,
    identities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    videos: list[dict[str, Any]] = []
    bank_paths: dict[str, Path] = {}
    with ZipStore(archive, mode="r") as store:
        for video_index, identity in enumerate(identities, start=1):
            started = time.perf_counter()
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            threshold = threshold_from_attrs(dict(group.attrs))
            slug = identity.replace(":", "__", 1)
            bank_path = output / "raw_universe" / "videos" / f"{slug}.jsonl.gz"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            frame_counts: list[int] = []
            replay_expected: dict[int, str] = {}
            next_node_id = 1
            with bank_path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as encoded:
                    for t in range(int(array.shape[0])):
                        frame = np.asarray(array[t])
                        rows = [
                            Candidate(dataset, next_node_id + i, t, z, y, x, intensity)
                            for i, (intensity, z, y, x) in enumerate(
                                block_peaks(frame, threshold)
                            )
                        ]
                        next_node_id += len(rows)
                        hasher = hashlib.sha256()
                        for candidate in rows:
                            payload = candidate.payload()
                            data = canonical_bytes(payload)
                            encoded.write(data)
                            hasher.update(data)
                        frame_counts.append(len(rows))
                        if t in (0, int(array.shape[0]) - 1):
                            replay_expected[t] = hasher.hexdigest()
                        del frame, rows
            replay: dict[str, Any] = {}
            offset = 1
            for t in range(int(array.shape[0])):
                if t not in replay_expected:
                    offset += frame_counts[t]
                    continue
                rows = [
                    Candidate(dataset, offset + i, t, z, y, x, intensity)
                    for i, (intensity, z, y, x) in enumerate(
                        block_peaks(np.asarray(array[t]), threshold)
                    )
                ]
                actual_hasher = hashlib.sha256()
                for candidate in rows:
                    actual_hasher.update(canonical_bytes(candidate.payload()))
                actual = actual_hasher.hexdigest()
                replay[str(t)] = {
                    "expected_sha256": replay_expected[t],
                    "actual_sha256": actual,
                    "equal": actual == replay_expected[t],
                }
                if actual != replay_expected[t]:
                    raise GeneratorError(f"determinism replay failed: {identity} t={t}")
                offset += frame_counts[t]
            bank_sha = sha256_file(bank_path)
            bank_paths[identity] = bank_path
            videos.append(
                {
                    "identity": identity,
                    "raw_zarr_path": f"train/{dataset}.zarr/0",
                    "shape": [int(value) for value in array.shape],
                    "chunks": [int(value) for value in array.chunks],
                    "dtype": str(array.dtype),
                    "threshold_q999_ceil": threshold,
                    "frame_count": int(array.shape[0]),
                    "frame_candidate_counts": frame_counts,
                    "candidate_count": int(sum(frame_counts)),
                    "raw_bank_path": str(bank_path),
                    "raw_bank_sha256": bank_sha,
                    "raw_bank_bytes": bank_path.stat().st_size,
                    "raw_bank_field_names": [
                        "dataset",
                        "identity_sha256",
                        "intensity",
                        "node_id",
                        "t",
                        "x",
                        "y",
                        "z",
                    ],
                    "replay": replay,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"BLOCKPEAK {video_index:02d}/36 {identity} "
                f"frames={array.shape[0]} candidates={sum(frame_counts)}",
                flush=True,
            )
            del group, array
    manifest = {
        "schema": "celltrack-alpha.blockpeak-raw-universe-manifest.v1",
        "status": "COMPLETE",
        "generator": GENERATOR,
        "family": FAMILY,
        "identity_count": len(identities),
        "frame_count": sum(row["frame_count"] for row in videos),
        "candidate_count": sum(row["candidate_count"] for row in videos),
        "labels_in_raw_banks": False,
        "masks_used": False,
        "pseudo_masks_used": False,
        "videos": videos,
    }
    manifest_path = output / "raw_universe" / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest, bank_paths


def positive_false_masks(
    candidates: Sequence[Candidate],
    gt_coords: np.ndarray,
    scale_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidates)
    positive = np.zeros(count, dtype=bool)
    nearest = np.full(count, np.inf, dtype=np.float64)
    if count == 0:
        return positive, ~positive, nearest
    if len(gt_coords) == 0:
        return positive, ~positive, nearest
    candidate_coords = np.asarray([row.coord for row in candidates], dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    distances = np.sqrt(
        np.sum(
            ((gt_coords[:, None, :] - candidate_coords[None, :, :]) * scale) ** 2,
            axis=2,
        )
    )
    nearest[:] = np.min(distances, axis=0)
    cost = distances.copy()
    cost[cost > POSITIVE_RADIUS_UM] = 1.0e12
    gt_rows, candidate_cols = linear_sum_assignment(cost)
    for gt_row, candidate_col in zip(gt_rows, candidate_cols, strict=True):
        if distances[int(gt_row), int(candidate_col)] <= POSITIVE_RADIUS_UM:
            positive[int(candidate_col)] = True
    return positive, ~positive, nearest


def graph_indices(
    graph: SubmissionGraph,
    scale_zyx: tuple[float, float, float],
) -> tuple[dict[int, tuple[np.ndarray, cKDTree]], set[tuple[int, int]]]:
    by_t: dict[int, list[SubmissionNode]] = defaultdict(list)
    for node in graph.nodes.values():
        by_t[int(node.t)].append(node)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    frames: dict[int, tuple[np.ndarray, cKDTree]] = {}
    for t, nodes in by_t.items():
        ordered = sorted(nodes, key=lambda node: int(node.node_id))
        ids = np.asarray([int(node.node_id) for node in ordered], dtype=np.int64)
        coords = (
            np.asarray([node.coord_zyx for node in ordered], dtype=np.float64) * scale
        )
        frames[t] = ids, cKDTree(coords)
    edges = {(int(edge.source_id), int(edge.target_id)) for edge in graph.edges}
    return frames, edges


def nearest(
    frame: tuple[np.ndarray, cKDTree] | None, coord_um: np.ndarray
) -> tuple[int, float] | None:
    if frame is None:
        return None
    distance, index = frame[1].query(coord_um, k=1)
    return int(frame[0][int(index)]), float(distance)


def build_action(
    identity: str,
    candidate: Candidate,
    graph: SubmissionGraph,
    scale_zyx: tuple[float, float, float],
    indices: tuple[dict[int, tuple[np.ndarray, cKDTree]], set[tuple[int, int]]],
) -> ActionRecord | None:
    frames, edges = indices
    if candidate.t <= 0 or candidate.t >= max(frames, default=0):
        return None
    coord_um = np.asarray(candidate.coord, dtype=np.float64) * np.asarray(
        scale_zyx, dtype=np.float64
    )
    predecessor = nearest(frames.get(candidate.t - 1), coord_um)
    successor = nearest(frames.get(candidate.t + 1), coord_um)
    if (
        predecessor is None
        or successor is None
        or predecessor[1] > TOPOLOGY_RADIUS_UM
        or successor[1] > TOPOLOGY_RADIUS_UM
    ):
        return None
    source, target = predecessor[0], successor[0]
    new_id = max(graph.nodes) + candidate.node_id
    conflicts = tuple(sorted(edge for edge in edges if edge[0] == source or edge[1] == target))
    action = GraphAction(
        action_id=f"blockpeak.restore.{candidate.identity_sha256[:24]}",
        kind=ActionKind.RESTORE_TOPOLOGY,
        add_nodes=(
            SubmissionNode(
                new_id, candidate.t, candidate.z, candidate.y, candidate.x
            ),
        ),
        add_edges=((source, new_id), (new_id, target)),
        replace_outgoing_from=(source,),
        replace_incoming_to=(target,),
        expected_conflict_edges=conflicts,
    )
    return ActionRecord(identity, candidate, action, conflicts)


def heap_keep(
    heap: list[tuple[int, str, ActionRecord]], record: ActionRecord
) -> None:
    numeric = int(record.candidate.identity_sha256, 16)
    item = (-numeric, record.identity, record)
    if len(heap) < PER_VIDEO_HEAP_CAP:
        heapq.heappush(heap, item)
    elif numeric < -heap[0][0]:
        heapq.heapreplace(heap, item)


def compatible_population(
    records: Sequence[ActionRecord], limit: int
) -> tuple[list[ActionRecord], Counter[str]]:
    accepted: list[ActionRecord] = []
    rejected: Counter[str] = Counter()
    owners: dict[str, set[tuple[str, Any]]] = defaultdict(set)
    for record in records:
        action = record.action
        primitives: dict[str, tuple[Any, ...]] = {
            "node": tuple(int(node.node_id) for node in action.add_nodes),
            "edge": tuple(action.add_edges),
            "removed": record.removed_edges,
            "source": tuple(action.replace_outgoing_from),
            "target": tuple(action.replace_incoming_to),
        }
        conflicts = [
            kind
            for kind, values in primitives.items()
            if any((record.identity, value) in owners[kind] for value in values)
        ]
        if conflicts:
            rejected.update(f"ownership_{kind}" for kind in sorted(set(conflicts)))
            continue
        for kind, values in primitives.items():
            owners[kind].update((record.identity, value) for value in values)
        accepted.append(record)
        if len(accepted) == limit:
            break
    return accepted, rejected


def action_payload(record: ActionRecord) -> dict[str, Any]:
    action = record.action
    return {
        "action_id": action.action_id,
        "action_kind": action.kind.value,
        "identity": record.identity,
        "candidate_identity_sha256": record.candidate.identity_sha256,
        "candidate_node_id": record.candidate.node_id,
        "add_nodes": [
            {
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": int(node.z),
                "y": int(node.y),
                "x": int(node.x),
            }
            for node in action.add_nodes
        ],
        "add_edges": [list(edge) for edge in action.add_edges],
        "replace_outgoing_from": list(action.replace_outgoing_from),
        "replace_incoming_to": list(action.replace_incoming_to),
        "expected_conflict_edges": [list(edge) for edge in action.expected_conflict_edges],
    }


def compile_single_worker(
    task: tuple[SubmissionGraph, GraphAction, str]
) -> dict[str, Any]:
    graph, action, identity = task
    compiled = compile_action(graph, action)
    if compiled.before_graph_sha256 == compiled.after_graph_sha256:
        raise GeneratorError(f"observation-null restoration: {action.action_id}")
    return {
        "action_id": action.action_id,
        "identity": identity,
        "before_graph_sha256": compiled.before_graph_sha256,
        "after_graph_sha256": compiled.after_graph_sha256,
        "action_sha256": compiled.action_sha256,
    }


def compile_batch_worker(
    task: tuple[int, str, SubmissionGraph, tuple[GraphAction, ...]]
) -> tuple[int, str, dict[str, Any]]:
    budget, identity, graph, actions = task
    compiled = compile_batch(graph, actions)
    return (
        budget,
        identity,
        {
            "action_count": len(actions),
            "batch_sha256": compiled.batch_sha256,
            "before_graph_sha256": compiled.before_graph_sha256,
            "after_graph_sha256": compiled.after_graph_sha256,
        },
    )


def audit_and_compile(
    output: Path,
    archive: Path,
    identities: Sequence[str],
    manifest: Mapping[str, Any],
    bank_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = Counter()
    heaps: dict[str, list[tuple[int, str, ActionRecord]]] = {}
    graphs: dict[str, SubmissionGraph] = {}
    baseline_hashes: dict[str, str] = {}
    per_video: list[dict[str, Any]] = []
    with ZipStore(archive, mode="r") as store:
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
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
            fragment = (
                ROOT
                / "reports/g02/fitcache_117/bh9out/fragments"
                / f"{identity.replace(':', '__', 1)}.csv"
            )
            expected_fragment = manifest["videos"][identity]["files"]["fragments"]["sha256"]
            if sha256_file(fragment) != expected_fragment:
                raise GeneratorError(f"baseline fragment hash mismatch: {identity}")
            loaded = read_submission(fragment)
            graph = loaded.get(dataset)
            if graph is None or len(loaded) != 1:
                raise GeneratorError(f"baseline graph identity mismatch: {identity}")
            graphs[identity] = graph
            baseline_hashes[identity] = graph_sha256(graph)
            indices = graph_indices(graph, scale)
            by_t: dict[int, list[Candidate]] = defaultdict(list)
            for row in read_gzip_rows(bank_paths[identity]):
                candidate = candidate_from_row(row)
                if candidate.dataset != dataset:
                    raise GeneratorError("raw-bank dataset identity mismatch")
                by_t[candidate.t].append(candidate)
            local = Counter()
            heap: list[tuple[int, str, ActionRecord]] = []
            for t in sorted(by_t):
                candidates = by_t[t]
                gt_coords = np.asarray(gt_by_t.get(t, ()), dtype=np.float64).reshape((-1, 3))
                positive, false, nearest_gt = positive_false_masks(
                    candidates, gt_coords, scale
                )
                local["raw_candidates"] += len(candidates)
                local["positive_assignments_7um"] += int(np.sum(positive))
                local["false_candidates"] += int(np.sum(false))
                local["hard_false_gt14um"] += int(
                    np.sum(false & (nearest_gt > HARD_FALSE_RADIUS_UM))
                )
                for candidate, is_false in zip(candidates, false, strict=True):
                    if not bool(is_false):
                        continue
                    action = build_action(identity, candidate, graph, scale, indices)
                    if action is not None:
                        local["topology_joinable_false"] += 1
                        heap_keep(heap, action)
            heaps[identity] = heap
            support.update(local)
            per_video.append(
                {
                    "identity": identity,
                    "raw_bank_sha256": sha256_file(bank_paths[identity]),
                    "baseline_fragment_path": str(fragment),
                    "baseline_fragment_sha256": expected_fragment,
                    "baseline_graph_sha256": baseline_hashes[identity],
                    "support": dict(local),
                    "heap_retained": len(heap),
                }
            )
            print(
                f"AUDIT {identity} false={local['false_candidates']} "
                f"joinable={local['topology_joinable_false']}",
                flush=True,
            )

    pooled: list[ActionRecord] = []
    cutoffs: list[str] = []
    for heap in heaps.values():
        records = sorted((entry[2] for entry in heap), key=lambda record: record.order_key)
        pooled.extend(records)
        if len(records) == PER_VIDEO_HEAP_CAP:
            cutoffs.append(records[-1].candidate.identity_sha256)
    pooled.sort(key=lambda record: record.order_key)
    selected, rejected = compatible_population(pooled, max(BUDGETS))
    if len(selected) < max(BUDGETS):
        raise GeneratorError(
            f"only {len(selected)} compatible restoration actions; need {max(BUDGETS)}"
        )
    if cutoffs and selected[-1].candidate.identity_sha256 >= min(cutoffs):
        raise GeneratorError("bounded heaps cannot prove globally exact first-50 population")
    allowed_kinds = {ActionKind.RESTORE_TOPOLOGY}
    if {record.action.kind for record in selected} != allowed_kinds:
        raise GeneratorError("action population contains a non-restoration action")

    action_path = output / "action_population.jsonl.gz"
    action_sha, action_count = gzip_rows(
        action_path, (action_payload(record) for record in selected)
    )
    audit = {
        "schema": "celltrack-alpha.blockpeak-coverage-identity-audit.v1",
        "status": "PASS",
        "support": dict(support),
        "video_count": len(per_video),
        "videos": per_video,
        "positive_radius_um": POSITIVE_RADIUS_UM,
        "hard_false_radius_um": HARD_FALSE_RADIUS_UM,
        "labels_in_raw_bank": False,
        "identity_failures": 0,
    }
    audit_checkpoint = output / "COVERAGE_IDENTITY_AUDIT.checkpoint.json"
    audit_checkpoint.write_bytes(canonical_bytes(audit))
    population_checkpoint = output / "ACTION_POPULATION.checkpoint.json"
    population_checkpoint.write_bytes(
        canonical_bytes(
            {
                "status": "READY_FOR_FROZEN_N1",
                "action_population_path": str(action_path),
                "action_population_sha256": action_sha,
                "action_population_row_count": action_count,
                "pooled_bounded_action_count": len(pooled),
                "global_order_exact_through_50": True,
                "ownership_rejections_before_50": dict(rejected),
            }
        )
    )
    print("CHECKPOINT audit and exact 50-action population frozen", flush=True)

    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            single_receipts = list(
                executor.map(
                    compile_single_worker,
                    (
                        (graphs[record.identity], record.action, record.identity)
                        for record in selected
                    ),
                    chunksize=1,
                )
            )
    except ActionCompileError as exc:
        raise GeneratorError(f"frozen N1 rejected a retained action: {exc}") from exc
    print(f"N1 singles {len(single_receipts)}/{len(selected)} PASS", flush=True)

    batch_tasks: list[
        tuple[int, str, SubmissionGraph, tuple[GraphAction, ...]]
    ] = []
    selected_by_budget: dict[int, dict[str, list[ActionRecord]]] = {}
    for budget in BUDGETS:
        by_identity: dict[str, list[ActionRecord]] = defaultdict(list)
        for record in selected[:budget]:
            by_identity[record.identity].append(record)
        selected_by_budget[budget] = by_identity
        for identity, records in sorted(by_identity.items()):
            batch_tasks.append(
                (
                    budget,
                    identity,
                    graphs[identity],
                    tuple(record.action for record in records),
                )
            )
    with ProcessPoolExecutor(max_workers=4) as executor:
        batch_results = list(executor.map(compile_batch_worker, batch_tasks, chunksize=1))
    batch_receipts: dict[int, dict[str, Any]] = defaultdict(dict)
    for budget, identity, receipt in batch_results:
        batch_receipts[budget][identity] = receipt
    print(f"N1 budget batches {len(batch_results)}/{len(batch_tasks)} PASS", flush=True)

    curve: list[dict[str, Any]] = []
    for budget in BUDGETS:
        chosen = selected[:budget]
        by_identity = selected_by_budget[budget]
        removed = sum(len(record.removed_edges) for record in chosen)
        curve.append(
            {
                "budget": budget,
                "selected_action_count": len(chosen),
                "affected_video_count": len(by_identity),
                "added_nodes": len(chosen),
                "added_edges": 2 * len(chosen),
                "declared_removed_edges": removed,
                "batch_receipts": batch_receipts[budget],
                "official_scorer_run": False,
                "graph_metric": None,
            }
        )
    population = {
        "schema": "celltrack-alpha.blockpeak-restoration-population.v1",
        "status": "GO",
        "action_kind": "restore_topology",
        "action_kind_count": {"restore_topology": len(selected)},
        "population_count": len(selected),
        "action_population_path": str(action_path),
        "action_population_sha256": action_sha,
        "action_population_row_count": action_count,
        "pooled_bounded_action_count": len(pooled),
        "global_order_exact_through_50": True,
        "ownership_rejections_before_50": dict(rejected),
        "single_n1_compile_passed": len(single_receipts),
        "single_n1_compile_total": len(selected),
        "single_n1_receipts": single_receipts,
        "false_node_budget_stress_curve": curve,
        "official_scorer_run": False,
        "graph_metric_run": False,
        "tier_gate": False,
    }
    return audit, population


def load_existing_raw(
    output: Path, identities: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = output / "raw_universe" / "manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("identity_count") != 36
        or tuple(row["identity"] for row in manifest.get("videos", ())) != tuple(identities)
    ):
        raise GeneratorError("existing raw universe is not the exact complete frozen set")
    bank_paths: dict[str, Path] = {}
    for video in manifest["videos"]:
        path = Path(video["raw_bank_path"])
        if sha256_file(path) != video["raw_bank_sha256"]:
            raise GeneratorError(f"existing raw bank hash mismatch: {video['identity']}")
        bank_paths[video["identity"]] = path
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest, bank_paths


def run(output: Path, resume_raw: bool) -> None:
    started = time.perf_counter()
    prereg, identities, baseline_manifest, receipts = authenticate()
    archive = Path(prereg["frozen_inputs"]["raw_archive"]["path"])
    if resume_raw:
        raw_manifest, bank_paths = load_existing_raw(output, identities)
    else:
        if output.exists():
            raise GeneratorError(f"refusing to overwrite output: {output}")
        output.mkdir(parents=True)
        raw_manifest, bank_paths = generate_raw_banks(output, archive, identities)
    audit, population = audit_and_compile(
        output, archive, identities, baseline_manifest, bank_paths
    )
    audit_path = output / "COVERAGE_IDENTITY_AUDIT.json"
    audit_path.write_bytes(canonical_bytes(audit))
    curve_path = output / "FALSE_NODE_STRESS_CURVE.json"
    curve_path.write_bytes(canonical_bytes(population))
    result = {
        "schema": "celltrack-alpha.sa-nodetopo-gen-result.v1",
        "status": "GO_AUTHENTIC_GENERATOR_PREREQUISITE",
        "mechanism": GENERATOR,
        "preregistration": {
            "path": str(PREREG),
            "sha256": PREREG_SHA256,
            "git_commit": PREREG_COMMIT,
            "outcomes_opened_after_commit": True,
        },
        "frozen_input_receipts": receipts,
        "raw_universe": {
            "manifest_path": raw_manifest["manifest_path"],
            "manifest_sha256": raw_manifest["manifest_sha256"],
            "identity_count": raw_manifest["identity_count"],
            "frame_count": raw_manifest["frame_count"],
            "candidate_count": raw_manifest["candidate_count"],
            "all_replays_equal": all(
                replay["equal"]
                for video in raw_manifest["videos"]
                for replay in video["replay"].values()
            ),
            "labels_in_raw_banks": False,
            "masks_used": False,
            "pseudo_masks_used": False,
        },
        "coverage_identity_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "status": audit["status"],
            "support": audit["support"],
        },
        "action_population": {
            "path": str(curve_path),
            "sha256": sha256_file(curve_path),
            "status": population["status"],
            "count": population["population_count"],
            "kind": population["action_kind"],
            "single_n1_compile_passed": population["single_n1_compile_passed"],
            "single_n1_compile_total": population["single_n1_compile_total"],
            "budgets": list(BUDGETS),
        },
        "decision": {
            "generator_prerequisite": "GO",
            "node_restoration_population": "GO",
            "NODETOPO_O0_graph_gate": "HOLD_FOR_HUMAN_REVIEW",
            "tier_move_proposed": False,
            "graph_score_run": False,
            "second_family_opened": False,
        },
        "scope": {
            "fit_only": True,
            "family": FAMILY,
            "cv57_used": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "cellect_touched": False,
            "submission_action": False,
            "model_fit": False,
            "detector_modified": False,
            "held_graph_scoring": False,
        },
        "quarantine": {
            "memo_c2_values_used": False,
            "memo_branch_b_values_used": False,
            "old_partial_nodetopo_metrics_used": False,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "stream_one_frame_at_a_time": True,
            "stream_one_video_at_a_time": True,
            "maximum_inflight_frames": 1,
        },
    }
    result_path = output / "RESULT.json"
    result_path.write_bytes(canonical_bytes(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "videos": raw_manifest["identity_count"],
                "frames": raw_manifest["frame_count"],
                "candidates": raw_manifest["candidate_count"],
                "actions": population["population_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-raw", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.resume_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
