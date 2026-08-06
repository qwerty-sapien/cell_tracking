#!/usr/bin/env python3
"""Seal and extract raw 3-D+t crops for the G-03 trained fork screen."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402


LANE = ROOT / "reports/g02/screening/tiered_surface/branches/g03_orthogonal_lanes_20260803/lane2_trained_fork"
PREREG = LANE / "PREREGISTRATION.json"
PREREG_SHA256 = "1c75dcc2a5c989b51e8d9d4e86fed7bdbd15c036ab96c48e912adc999184f6ab"
BUNDLES = ROOT / "reports/g02/screening/tiered_surface/branches/sa_division_value/DIVISION_REPAIR_BUNDLES.json"
BUNDLES_SHA256 = "198aeda1d0c0522bd1aedacdbb48e2b343aca0b25cb41d68dd8cb72cbfc87b14"
TARGET_SPEC = ROOT / "reports/g02/screening/tiered_surface/branches/sa_division_cont/continuity_target_spec.json"
TARGET_SPEC_SHA256 = "0926ab647915ad4e2e790252c040867e0bbc45a396155cecfe7f5813d6efe221"
DEV69_SEAL = ROOT / "reports/g03/screening/wave_r/R4_artifacts/bank_extension/dev69/DEV69_INFERENCE_SEAL.json"
DEV69_SEAL_SHA256 = "588397b13ba330192ebb8bd3fb6f226ec61d2f2ba9ee9635728765dde00ee6d8"
H000_SCORE = ROOT / "reports/g03/screening/phase2_zero_init_v1/h000/CANONICAL_SCORE.json"
H000_SCORE_SHA256 = "d7189066b4122ff11e8a82fef616e292a9053f947c0a2e7dce9b3923ef7fc1c0"
DEV69_BANK_ROOT = ROOT / "reports/g03/screening/wave_r/R4_artifacts/bank_extension/dev69/videos"
HELD36_BANK_ROOT = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1/run_v1/kaggle_output/trackastra_native_fit105_bank_v1/videos"
FINAL_GRAPH_ROOT = ROOT / "reports/g02/screening/tiered_surface/branches/sa_native_base_wave_v1/cv57_local_native_v1/fragments"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
RAW_ARCHIVE_BYTES = 87_393_127_165
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MATCH_RADIUS_UM = 7.0
CROP_SHAPE = (3, 16, 32, 32)
DOWNSAMPLED_SHAPE = (3, 8, 16, 16)


class ForkDataError(RuntimeError):
    """The direct fork screen data contract failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ForkDataError(f"{label} SHA-256 mismatch: {path}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ForkDataError(f"expected JSON object: {path}")
    return value


def canonical_json(value: Mapping[str, Any]) -> bytes:
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise ForkDataError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    if path.exists():
        raise ForkDataError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return sha256(path)


def verify_static() -> None:
    require_hash(PREREG, PREREG_SHA256, "lane preregistration")
    require_hash(BUNDLES, BUNDLES_SHA256, "division bundle")
    require_hash(TARGET_SPEC, TARGET_SPEC_SHA256, "continuity target specification")
    require_hash(DEV69_SEAL, DEV69_SEAL_SHA256, "development bank seal")
    require_hash(H000_SCORE, H000_SCORE_SHA256, "held-fit36 population seal")
    if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise ForkDataError("raw archive byte count drifted")


def gt_dividers(gt: Mapping[str, Any]) -> set[int]:
    edges = np.asarray(gt["edges"], dtype=np.int64)
    nodes = gt["nodes"]
    times = {int(node): int(t) for node, t in zip(nodes["node_id"], nodes["t"], strict=True)}
    forward = sum(times.get(int(target), -10) > times.get(int(source), 10) for source, target in edges)
    reverse = sum(times.get(int(source), -10) > times.get(int(target), 10) for source, target in edges)
    if forward < reverse or forward < 0.9 * len(edges):
        raise ForkDataError("GEFF edge direction/time convention drifted")
    counts = Counter(int(source) for source, _ in edges)
    return {node for node, count in counts.items() if count == 2}


def load_bank(identity: str, population: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    slug = identity.replace(":", "__", 1)
    if population == "train":
        path = DEV69_BANK_ROOT / slug / "association_bank.npz"
        with np.load(path, allow_pickle=False) as value:
            ids = np.asarray(value["node_id"], dtype=np.int64)
            frames = np.asarray(value["frame"], dtype=np.int64)
            coords = np.asarray(value["coords_zyx"], dtype=np.float64)
    elif population == "validation":
        path = HELD36_BANK_ROOT / f"{slug}.npz"
        with np.load(path, allow_pickle=False) as value:
            nodes = np.asarray(value["nodes"], dtype=np.float64)
        ids = nodes[:, 0].astype(np.int64)
        frames = nodes[:, 1].astype(np.int64)
        coords = nodes[:, 2:5]
    else:
        raise ForkDataError(f"unknown bank population {population}")
    if (
        len(ids) != len(frames)
        or len(ids) != len(coords)
        or len(ids) != len(set(ids.tolist()))
        or coords.shape != (len(ids), 3)
        or not np.isfinite(coords).all()
    ):
        raise ForkDataError(f"malformed detection bank: {path}")
    return ids, frames, coords, path


def match_to_gt(
    ids: np.ndarray,
    frames: np.ndarray,
    coords: np.ndarray,
    gt: Mapping[str, Any],
) -> dict[int, int]:
    nodes = gt["nodes"]
    result: dict[int, int] = {}
    for frame in sorted(set(frames.tolist()).intersection(set(np.asarray(nodes["t"]).tolist()))):
        pred_rows = np.flatnonzero(frames == frame)
        gt_rows = np.flatnonzero(np.asarray(nodes["t"]) == frame)
        if not len(pred_rows) or not len(gt_rows):
            continue
        gt_coords = np.stack(
            (
                np.asarray(nodes["z"])[gt_rows],
                np.asarray(nodes["y"])[gt_rows],
                np.asarray(nodes["x"])[gt_rows],
            ),
            axis=1,
        ).astype(np.float64)
        distance = np.linalg.norm(
            (coords[pred_rows, None, :] - gt_coords[None, :, :])
            * SCALE_ZYX_UM[None, None, :],
            axis=2,
        )
        cost = np.where(distance <= MATCH_RADIUS_UM, distance, MATCH_RADIUS_UM + 1e6)
        pred_index, gt_index = linear_sum_assignment(cost)
        for left, right in zip(pred_index, gt_index, strict=True):
            if distance[left, right] <= MATCH_RADIUS_UM:
                result[int(ids[pred_rows[left]])] = int(np.asarray(nodes["node_id"])[gt_rows[right]])
    return result


def nearest_gt_within_radius(
    ids: np.ndarray,
    frames: np.ndarray,
    coords: np.ndarray,
    gt: Mapping[str, Any],
) -> dict[int, int]:
    """Map every detection within 7 um of sparse GT, without one-to-one suppression."""

    nodes = gt["nodes"]
    result: dict[int, int] = {}
    for frame in sorted(set(frames.tolist()).intersection(set(np.asarray(nodes["t"]).tolist()))):
        pred_rows = np.flatnonzero(frames == frame)
        gt_rows = np.flatnonzero(np.asarray(nodes["t"]) == frame)
        if not len(pred_rows) or not len(gt_rows):
            continue
        gt_coords = np.stack(
            (
                np.asarray(nodes["z"])[gt_rows],
                np.asarray(nodes["y"])[gt_rows],
                np.asarray(nodes["x"])[gt_rows],
            ),
            axis=1,
        ).astype(np.float64)
        tree = cKDTree(gt_coords * SCALE_ZYX_UM[None, :])
        distances, indices = tree.query(coords[pred_rows] * SCALE_ZYX_UM[None, :], k=1)
        for row, distance, index in zip(pred_rows, distances, indices, strict=True):
            if float(distance) <= MATCH_RADIUS_UM:
                result[int(ids[row])] = int(np.asarray(nodes["node_id"])[gt_rows[int(index)]])
    return result


def hash_order(identity: str, frame: int, positive_id: int, candidate_id: int) -> bytes:
    return hashlib.sha256(
        f"G03-FORK-NEG-V1|{identity}|{frame}|{positive_id}|{candidate_id}".encode("ascii")
    ).digest()


def development_records(
    store: ZipStore, identities: Iterable[str], population: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for identity in sorted(identities):
        dataset = identity.split(":", 1)[1]
        ids, frames, coords, bank_path = load_bank(identity, population)
        gt = read_geff_arrays(store, "train", dataset)
        dividers = gt_dividers(gt)
        matched = match_to_gt(ids, frames, coords, gt)
        nearest = nearest_gt_within_radius(ids, frames, coords, gt)
        image = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
        id_to_row = {int(node): row for row, node in enumerate(ids)}
        positives = sorted(
            node
            for node, gt_node in matched.items()
            if gt_node in dividers and 0 < int(frames[id_to_row[node]]) < int(image.shape[0]) - 1
        )
        used_by_frame: dict[int, set[int]] = defaultdict(set)
        negatives = []
        for positive in positives:
            row = id_to_row[positive]
            frame = int(frames[row])
            eligible = [
                int(node)
                for node in ids.tolist()
                if int(frames[id_to_row[int(node)]]) == frame
                and int(node) != positive
                and int(node) not in used_by_frame[frame]
                and nearest.get(int(node)) not in dividers
            ]
            # Prefer authenticated sparse-GT non-forks, then deterministic
            # far-from-sparse-GT candidates. Sparse GEFF frames can expose
            # fewer than four authenticated non-forks despite hundreds of
            # observed detections; the fallback is recorded rather than hidden.
            eligible.sort(
                key=lambda node: (
                    0 if node in nearest else 1,
                    hash_order(identity, frame, positive, node),
                )
            )
            if len(eligible) < 4:
                raise ForkDataError(
                    f"fewer than four matched non-fork negatives for {identity} node {positive}"
                )
            chosen = eligible[:4]
            used_by_frame[frame].update(chosen)
            for label, nodes_to_add in ((1, [positive]), (0, chosen)):
                for node in nodes_to_add:
                    node_row = id_to_row[node]
                    records.append(
                        {
                            "split": population,
                            "identity": identity,
                            "family": identity.split(":", 1)[1].split("_", 1)[0],
                            "node_id": int(node),
                            "frame": int(frames[node_row]),
                            "zyx": coords[node_row].tolist(),
                            "label": label,
                            "matched_gt_id": int(
                                matched[node]
                                if label == 1
                                else nearest.get(node, -1)
                            ),
                            "label_status": (
                                "positive_7um_assignment_to_gt_fork"
                                if label == 1
                                else "authenticated_nearest_gt_nonfork"
                                if node in nearest
                                else "weak_negative_far_from_sparse_gt"
                            ),
                        }
                    )
                    if label == 0:
                        negatives.append(node)
        reports.append(
            {
                "identity": identity,
                "bank_path": str(bank_path),
                "bank_sha256": sha256(bank_path),
                "matched_detections": len(matched),
                "detections_with_nearest_gt_within_7um": len(nearest),
                "gt_dividers": len(dividers),
                "matched_dividers": len(positives),
                "sampled_negatives": len(negatives),
                "weak_negative_count": sum(
                    row["identity"] == identity
                    and row["label"] == 0
                    and row["label_status"] == "weak_negative_far_from_sparse_gt"
                    for row in records
                ),
            }
        )
    return records, reports


def read_fragment_nodes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, frames, coords = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] != "node":
                continue
            ids.append(int(row["node_id"]))
            frames.append(int(row["t"]))
            coords.append((float(row["z"]), float(row["y"]), float(row["x"])))
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(frames, dtype=np.int64),
        np.asarray(coords, dtype=np.float64),
    )


def seal_final_negatives(output: Path) -> dict[str, Any]:
    verify_static()
    target = load_json(TARGET_SPEC)
    events = target.get("events", [])
    if len(events) != 38:
        raise ForkDataError("final positive count drifted")
    positives_by_identity: dict[str, set[int]] = defaultdict(set)
    for event in events:
        positives_by_identity[str(event["identity"])].add(int(event["parent"]["node_id"]))
    rows: list[dict[str, Any]] = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], set[int], Path]] = {}
        used_by_identity_frame: dict[tuple[str, int], set[int]] = defaultdict(set)
        for event in events:
            identity = str(event["identity"])
            dataset = identity.split(":", 1)[1]
            if identity not in cached:
                fragment = FINAL_GRAPH_ROOT / f"{identity.replace(':', '__', 1)}.csv"
                ids, frames, coords = read_fragment_nodes(fragment)
                gt = read_geff_arrays(store, "train", dataset)
                cached[identity] = (
                    ids,
                    frames,
                    coords,
                    nearest_gt_within_radius(ids, frames, coords, gt),
                    gt_dividers(gt),
                    fragment,
                )
            ids, frames, coords, matched, dividers, fragment = cached[identity]
            id_to_row = {int(node): index for index, node in enumerate(ids)}
            positive_id = int(event["parent"]["node_id"])
            if positive_id not in id_to_row:
                raise ForkDataError(f"positive parent absent from graph: {event['event_key']}")
            positive_row = id_to_row[positive_id]
            frame = int(frames[positive_row])
            if frame != int(event["parent"]["t"]):
                raise ForkDataError("positive parent frame drifted")
            frame_rows = np.flatnonzero(frames == frame)
            physical = coords[frame_rows] * SCALE_ZYX_UM[None, :]
            tree = cKDTree(physical)
            counts = np.asarray([len(value) - 1 for value in tree.query_ball_tree(tree, r=12.0)])
            frame_node_to_local = {int(ids[row]): local for local, row in enumerate(frame_rows)}
            positive_count = int(counts[frame_node_to_local[positive_id]])
            eligible = []
            for row in frame_rows:
                node = int(ids[row])
                gt_node = matched.get(node)
                if (
                    node in positives_by_identity[identity]
                    or node in used_by_identity_frame[(identity, frame)]
                    or (gt_node is not None and gt_node in dividers)
                ):
                    continue
                local = frame_node_to_local[node]
                distance = float(
                    np.linalg.norm((coords[row] - coords[positive_row]) * SCALE_ZYX_UM)
                )
                eligible.append((abs(int(counts[local]) - positive_count), distance, node, row, int(counts[local])))
            if not eligible:
                raise ForkDataError(f"no matched negative candidate: {event['event_key']}")
            density_delta, distance, negative_id, negative_row, negative_count = min(eligible)
            used_by_identity_frame[(identity, frame)].add(negative_id)
            common = {
                "ordinal": int(event["event_index"]) if "event_index" in event else len(rows) // 2,
                "event_key": str(event["event_key"]),
                "identity": identity,
                "family": str(event["family"]),
                "frame": frame,
                "source_fragment_path": str(fragment),
                "source_fragment_sha256": sha256(fragment),
            }
            rows.append(
                common
                | {
                    "label": 1,
                    "node_id": positive_id,
                    "zyx": coords[positive_row].tolist(),
                    "neighborhood_count_12um": positive_count,
                    "match_distance_um": 0.0,
                }
            )
            rows.append(
                common
                | {
                    "label": 0,
                    "node_id": negative_id,
                    "zyx": coords[negative_row].tolist(),
                    "neighborhood_count_12um": negative_count,
                    "density_count_absolute_delta": density_delta,
                    "match_distance_um": distance,
                    "matched_gt_id": matched.get(negative_id),
                }
            )
    result = {
        "schema": "celltrack-alpha.g03-trained-fork-final-population.v1",
        "status": "SEALED_BEFORE_CROP_EXTRACTION_OR_MODEL_FIT",
        "lane_preregistration_sha256": PREREG_SHA256,
        "target_spec_sha256": TARGET_SPEC_SHA256,
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(int(row["label"]) == 0 for row in rows),
        "record_count": len(rows),
        "records": rows,
        "model_score_access": False,
        "crop_access": False,
        "graph_integration": False,
    }
    if result["positive_count"] != 38 or result["negative_count"] != 38:
        raise ForkDataError("sealed final population counts drifted")
    atomic_json(output, result)
    return result


def crop(image: Any, frame: int, center_zyx: Iterable[float]) -> np.ndarray:
    center = np.rint(np.asarray(tuple(center_zyx), dtype=np.float64)).astype(np.int64)
    temporal = np.clip(np.asarray((frame - 1, frame, frame + 1)), 0, image.shape[0] - 1)
    indices = []
    for coordinate, size, limit in zip(center, CROP_SHAPE[1:], image.shape[1:], strict=True):
        start = int(coordinate) - size // 2
        indices.append(np.clip(np.arange(start, start + size), 0, int(limit) - 1))
    value = np.stack(
        [np.asarray(image[int(t)])[np.ix_(indices[0], indices[1], indices[2])] for t in temporal]
    ).astype(np.float32)
    mean = float(value.mean())
    centered = value - mean
    rms = max(float(np.sqrt(np.mean(centered * centered))), 1.0e-6)
    normalized = np.clip(centered / rms, -5.0, 5.0)
    reduced = normalized.reshape(3, 8, 2, 16, 2, 16, 2).mean(axis=(2, 4, 6))
    if reduced.shape != DOWNSAMPLED_SHAPE or not np.isfinite(reduced).all():
        raise ForkDataError("malformed normalized crop")
    return reduced.astype(np.float32)


def extract_records(store: ZipStore, records: list[dict[str, Any]]) -> np.ndarray:
    values = np.empty((len(records),) + DOWNSAMPLED_SHAPE, dtype=np.float32)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_identity[str(record["identity"])].append(index)
    for identity in sorted(by_identity):
        dataset = identity.split(":", 1)[1]
        image = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
        for index in by_identity[identity]:
            record = records[index]
            values[index] = crop(image, int(record["frame"]), record["zyx"])
        print(json.dumps({"identity": identity, "crops": len(by_identity[identity])}), flush=True)
    return values


def extract_development(output: Path) -> dict[str, Any]:
    verify_static()
    dev = load_json(DEV69_SEAL)
    train_ids = [str(value) for value in dev["identities"]]
    held = load_json(H000_SCORE)
    validation_ids = [str(row["identity"]) for row in held["per_video"]]
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        train_records, train_reports = development_records(store, train_ids, "train")
        validation_records, validation_reports = development_records(
            store, validation_ids, "validation"
        )
        records = train_records + validation_records
        values = extract_records(store, records)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.uint8)
    splits = np.asarray([str(row["split"]) for row in records])
    digest = atomic_npz(
        output,
        x=values,
        y=labels,
        split=splits,
        identity=np.asarray([str(row["identity"]) for row in records]),
        node_id=np.asarray([int(row["node_id"]) for row in records], dtype=np.int64),
        frame=np.asarray([int(row["frame"]) for row in records], dtype=np.int16),
    )
    report = {
        "schema": "celltrack-alpha.g03-trained-fork-development-crops.v1",
        "status": "COMPLETE_CROSS_FAMILY_FROM_FIRST_FIT",
        "lane_preregistration_sha256": PREREG_SHA256,
        "output_path": str(output),
        "output_sha256": digest,
        "crop_shape": list(DOWNSAMPLED_SHAPE),
        "train": {
            "family": "6bba",
            "identities": len(train_ids),
            "positives": sum(row["label"] == 1 for row in train_records),
            "negatives": sum(row["label"] == 0 for row in train_records),
            "reports": train_reports,
        },
        "validation": {
            "family": "44b6",
            "identities": len(validation_ids),
            "positives": sum(row["label"] == 1 for row in validation_records),
            "negatives": sum(row["label"] == 0 for row in validation_records),
            "reports": validation_reports,
        },
        "final_38_access": False,
        "gpu_hours": 0.0,
    }
    atomic_json(output.with_suffix(".json"), report)
    return report


def extract_final(population: Path, output: Path) -> dict[str, Any]:
    verify_static()
    sealed = load_json(population)
    if (
        sealed.get("status") != "SEALED_BEFORE_CROP_EXTRACTION_OR_MODEL_FIT"
        or sealed.get("positive_count") != 38
        or sealed.get("negative_count") != 38
    ):
        raise ForkDataError("final population seal is invalid")
    records = [dict(row) for row in sealed["records"]]
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        values = extract_records(store, records)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.uint8)
    digest = atomic_npz(
        output,
        x=values,
        y=labels,
        identity=np.asarray([str(row["identity"]) for row in records]),
        node_id=np.asarray([int(row["node_id"]) for row in records], dtype=np.int64),
        frame=np.asarray([int(row["frame"]) for row in records], dtype=np.int16),
    )
    report = {
        "schema": "celltrack-alpha.g03-trained-fork-final-crops.v1",
        "status": "COMPLETE_38_POSITIVE_38_MATCHED_NEGATIVE_CROPS",
        "population_path": str(population),
        "population_sha256": sha256(population),
        "output_path": str(output),
        "output_sha256": digest,
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "graph_integration": False,
        "gpu_hours": 0.0,
    }
    atomic_json(output.with_suffix(".json"), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-final")
    seal.add_argument("--output", type=Path, required=True)
    development = subparsers.add_parser("extract-development")
    development.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("extract-final")
    final.add_argument("--population", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal-final":
        result = seal_final_negatives(args.output)
    elif args.command == "extract-development":
        result = extract_development(args.output)
    else:
        result = extract_final(args.population, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
