#!/usr/bin/env python3
"""CUDA-canonical fixed-checkpoint h005 association-bank emitter.

This stage runs only after immutable h005 publication.  It opens the exact 36
held-fit 44b6 images and detections, emits complete association banks, and does
not read GEFF or any score.  Graph compilation and scoring remain a separate
local, seal-gated stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback
from typing import Any
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import yaml
import zarr


SCHEMA = "celltrack-alpha.g03-phase2-h005-cuda-bank.v1"
EXPECTED_MODEL_SHA256 = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
EXPECTED_CACHE_MANIFEST_SHA256 = "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28"
EXPECTED_PARENT_H000_SHA256 = "603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
WINDOW_FRAMES = 4
MAX_TOKENS = 1024
INFERENCE_RADIUS_UM = 12.0
INFERENCE_NEIGHBOR_CAP = 16
SOURCE_CHUNK = 128
ADAPTER_BATCH = 128
INTERNAL_DEADLINE_SECONDS = 17_100.0


class EvalError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def atomic_npz(path: Path, *, nodes: np.ndarray, edges: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, nodes=nodes, edges=edges)
    os.replace(temporary, path)
    return sha256(path)


class Phase2CropAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_pool = nn.AvgPool3d((4, 8, 8), stride=(4, 8, 8))
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=True)
        self.activation1 = nn.GELU(approximate="none")
        self.conv2 = nn.Conv3d(16, 48, kernel_size=3, padding=1, bias=True)
        self.activation2 = nn.GELU(approximate="none")
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(48, 12, bias=True)
        self.output_activation = nn.Tanh()

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        value = crops.float()
        mean = value.mean(dim=(1, 2, 3, 4), keepdim=True)
        centered = value - mean
        rms = torch.sqrt(centered.square().mean(dim=(1, 2, 3, 4), keepdim=True))
        value = torch.clamp(centered / rms.clamp_min(1.0e-6), -5.0, 5.0)
        value = self.spatial_pool(value)
        value = self.activation1(self.conv1(value))
        value = self.activation2(self.conv2(value))
        value = self.global_pool(value).flatten(1)
        value = self.output_activation(self.projection(value))
        if value.shape[1:] != (12,) or not bool(torch.isfinite(value).all()):
            raise EvalError("adapter output drift")
        return value


def cache_identity(meta: Mapping[str, Any]):
    from celltrack_alpha import bh_model

    model = meta["contract"]["model"]
    return bh_model.CacheIdentity(
        str(model["config_sha256"]),
        str(model["source_sha256"]),
        str(model["weight_sha256"]),
        int(model["pair_batch_size"]),
        int(model["view_batch_size"]),
    )


def load_detections(cache_root: Path, identity: str):
    from celltrack_alpha import bh_model

    slug = identity.replace(":", "__", 1)
    meta = json.loads((cache_root / f"{slug}.json").read_text(encoding="ascii"))
    return bh_model.load_sparse_cache_detections(
        cache_root / f"{slug}.bhc", expected_identity=cache_identity(meta)
    )


def load_models(asset_root: Path, checkpoint: Mapping[str, Any], device: torch.device):
    from trackastra.model.model import TrackingTransformer

    with zipfile.ZipFile(asset_root / "ctc.zip.bin") as archive:
        config = yaml.safe_load(archive.read("ctc/config.yaml"))
    model = TrackingTransformer(**config)
    model.load_state_dict(checkpoint["model"]["state_dict"], strict=True)
    adapter = Phase2CropAdapter()
    adapter.load_state_dict(checkpoint["adapter"]["state_dict"], strict=True)
    model.to(device).eval()
    adapter.to(device).eval()
    for parameter in (*model.parameters(), *adapter.parameters()):
        parameter.requires_grad_(False)
    return model, adapter


def crop_indices(coords: np.ndarray, shape: Sequence[int]) -> tuple[np.ndarray, ...]:
    centers = np.rint(np.asarray(coords, dtype=np.float64)).astype(np.int64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise EvalError("crop coordinates must have shape [N,3]")
    offsets = (
        np.arange(-8, 8, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
    )
    return tuple(
        np.clip(centers[:, axis, None] + offsets[axis], 0, int(shape[axis]) - 1)
        for axis in range(3)
    )


def crop_batch(frames: Sequence[np.ndarray], coords: np.ndarray) -> torch.Tensor:
    if len(frames) != 3 or any(frame.ndim != 3 for frame in frames):
        raise EvalError("crop source must contain three Z/Y/X frames")
    if any(tuple(frame.shape) != tuple(frames[0].shape) for frame in frames):
        raise EvalError("temporal crop frames have different shapes")
    z_idx, y_idx, x_idx = crop_indices(coords, frames[0].shape)
    channels = [
        frame[
            z_idx[:, :, None, None],
            y_idx[:, None, :, None],
            x_idx[:, None, None, :],
        ].astype(np.float32, copy=False)
        for frame in frames
    ]
    value = np.stack(channels, axis=1)
    if value.shape != (len(coords), 3, 16, 32, 32) or not np.isfinite(value).all():
        raise EvalError("extracted crops are malformed")
    return torch.from_numpy(value)


def embed_video(image: Any, detections: Sequence[Any], adapter, device: torch.device):
    if not detections:
        raise EvalError("video has no detections")
    by_node = {}
    frame_nodes = {}
    channel_sum = np.zeros(12, dtype=np.float64)
    channel_sq = np.zeros(12, dtype=np.float64)
    channel_min = np.full(12, np.inf)
    channel_max = np.full(12, -np.inf)
    nonzero_rows = count = 0
    minimum = min(int(frame.frame_index) for frame in detections)
    maximum = max(int(frame.frame_index) for frame in detections)
    frame_cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for frame in sorted(detections, key=lambda value: int(value.frame_index)):
            t = int(frame.frame_index)
            node_ids = np.asarray(frame.node_ids, dtype=np.int64)
            coords = np.asarray(frame.coords_zyx, dtype=np.float32)
            frame_nodes[t] = (node_ids, coords)
            needed = [min(max(t + offset, minimum), maximum) for offset in (-1, 0, 1)]
            for source_t in needed:
                if source_t not in frame_cache:
                    frame_cache[source_t] = np.asarray(image[source_t], dtype=np.float32)
            outputs = []
            for start in range(0, len(coords), ADAPTER_BATCH):
                crops = crop_batch(
                    [frame_cache[source_t] for source_t in needed],
                    coords[start : start + ADAPTER_BATCH],
                ).to(device=device, non_blocking=True)
                outputs.append(adapter(crops).cpu().numpy().astype(np.float32))
            features = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 12), np.float32)
            for node, coord, feature in zip(node_ids, coords, features, strict=True):
                if int(node) in by_node:
                    raise EvalError("detection node ID repeats")
                by_node[int(node)] = (t, coord, feature)
            if len(features):
                values = features.astype(np.float64)
                channel_sum += values.sum(axis=0)
                channel_sq += (values * values).sum(axis=0)
                channel_min = np.minimum(channel_min, values.min(axis=0))
                channel_max = np.maximum(channel_max, values.max(axis=0))
                nonzero_rows += int(np.count_nonzero(np.any(features != 0, axis=1)))
                count += len(features)
            keep = set(needed)
            for old in list(frame_cache):
                if old not in keep and old < t - 1:
                    del frame_cache[old]
    mean = channel_sum / max(count, 1)
    variance = np.maximum(channel_sq / max(count, 1) - mean * mean, 0.0)
    return by_node, frame_nodes, {
        "node_count": count,
        "nonzero_row_count": nonzero_rows,
        "all_rows_nonzero": nonzero_rows == count and count > 0,
        "channel_mean": mean.tolist(),
        "channel_std": np.sqrt(variance).tolist(),
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
    }


def source_chunks(source_ids, source_coords, target_ids, target_coords):
    source_physical = source_coords * SCALE_ZYX[None]
    target_physical = target_coords * SCALE_ZYX[None]
    tree = cKDTree(target_physical)
    order = np.lexsort((source_coords[:, 2], source_coords[:, 1], source_coords[:, 0]))
    pending = [order[start : start + SOURCE_CHUNK] for start in range(0, len(order), SOURCE_CHUNK)]
    while pending:
        indices = pending.pop(0)
        candidates = []
        required_targets = set()
        for source_index in indices:
            hits = tree.query_ball_point(source_physical[source_index], r=INFERENCE_RADIUS_UM)
            hits = sorted(
                hits,
                key=lambda target_index: (
                    float(np.linalg.norm(target_physical[target_index] - source_physical[source_index])),
                    int(target_ids[target_index]),
                ),
            )[:INFERENCE_NEIGHBOR_CAP]
            row = []
            for target_index in hits:
                distance = float(np.linalg.norm(target_physical[target_index] - source_physical[source_index]))
                row.append((int(target_ids[target_index]), distance))
                required_targets.add(int(target_ids[target_index]))
            candidates.append((int(source_ids[source_index]), row))
        if len(indices) + len(required_targets) > MAX_TOKENS:
            if len(indices) == 1:
                raise EvalError("one source neighborhood exceeds token cap")
            middle = len(indices) // 2
            pending.insert(0, indices[middle:])
            pending.insert(0, indices[:middle])
            continue
        yield candidates, required_targets


def infer_association(model, ordered_nodes, by_node, device):
    coordinates = np.stack(
        [np.concatenate(([by_node[node][0]], by_node[node][1])).astype(np.float32) for node in ordered_nodes]
    )
    features = np.stack([by_node[node][2] for node in ordered_nodes]).astype(np.float32)
    timepoints = torch.from_numpy(coordinates[:, 0].astype(np.int64)).unsqueeze(0).to(device)
    coords = torch.from_numpy(coordinates).unsqueeze(0).to(device)
    wrfeat = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(coords, features=wrfeat)
        association = model.normalize_output(logits, timepoints, coords)[0]
    value = association.float().cpu().numpy()
    if value.shape != (len(ordered_nodes), len(ordered_nodes)) or not np.isfinite(value).all():
        raise EvalError("Trackastra association matrix is malformed")
    return value, {node: index for index, node in enumerate(ordered_nodes)}


def process_video(identity, model, adapter, cache_root, train_root, output_root, device, deadline):
    started = time.perf_counter()
    if time.monotonic() >= deadline - 120:
        raise TimeoutError("evaluation deadline reached")
    detections = load_detections(cache_root, identity)
    dataset = identity.split(":", 1)[1]
    if not dataset.startswith("44b6_"):
        raise EvalError("held-fit family drift")
    image = zarr.open_group(str(train_root / f"{dataset}.zarr"), mode="r")["0"]
    by_node, frame_nodes, feature_evidence = embed_video(image, detections, adapter, device)
    all_frames = sorted(frame_nodes)
    edge_rows = []
    calls = peak_tokens = 0
    for source_t in range(all_frames[0], all_frames[-1]):
        if time.monotonic() >= deadline - 120:
            raise TimeoutError("evaluation deadline reached")
        if source_t not in frame_nodes or source_t + 1 not in frame_nodes:
            continue
        source_ids, source_coords = frame_nodes[source_t]
        target_ids, target_coords = frame_nodes[source_t + 1]
        if len(source_ids) == 0 or len(target_ids) == 0:
            continue
        start = min(max(source_t - 1, all_frames[0]), max(all_frames[0], all_frames[-1] - 3))
        context = [
            int(node)
            for frame in range(start, start + WINDOW_FRAMES)
            if frame in frame_nodes
            for node in frame_nodes[frame][0]
        ]
        for candidates, required_targets in source_chunks(source_ids, source_coords, target_ids, target_coords):
            required = {source for source, _ in candidates} | required_targets
            centroid = np.mean([by_node[node][1] for node in required], axis=0)
            optional = sorted(
                set(context) - required,
                key=lambda node: (float(np.sum((by_node[node][1] - centroid) ** 2)), by_node[node][0], node),
            )
            selected = required | set(optional[: MAX_TOKENS - len(required)])
            ordered = tuple(sorted(selected, key=lambda node: (by_node[node][0], node)))
            association, index = infer_association(model, ordered, by_node, device)
            calls += 1
            peak_tokens = max(peak_tokens, len(ordered))
            for source, targets in candidates:
                for target, distance in targets:
                    edge_rows.append((source, target, float(association[index[source], index[target]]), distance))
    nodes = np.asarray(
        sorted(
            [
                [int(node), int(value[0]), float(value[1][0]), float(value[1][1]), float(value[1][2])]
                for node, value in by_node.items()
            ],
            key=lambda row: (row[1], row[0]),
        ),
        dtype=np.float64,
    ).reshape(-1, 5)
    edges = np.asarray(sorted(edge_rows, key=lambda row: (row[0], row[1])), dtype=np.float64).reshape(-1, 4)
    slug = identity.replace(":", "__", 1)
    bank_path = output_root / "banks" / f"{slug}.npz"
    bank_sha = atomic_npz(bank_path, nodes=nodes, edges=edges)
    report = {
        "schema": SCHEMA,
        "identity": identity,
        "device": str(device),
        "bank_path": str(bank_path.relative_to(output_root)),
        "bank_sha256": bank_sha,
        "bank_bytes": bank_path.stat().st_size,
        "node_count": len(nodes),
        "candidate_edge_count": len(edges),
        "inference_calls": calls,
        "peak_tokens": peak_tokens,
        "wrfeat": feature_evidence,
        "elapsed_seconds": time.perf_counter() - started,
        "gt_access": False,
        "score_access": False,
    }
    atomic_json(output_root / "receipts" / f"{slug}.json", report)
    return report


def aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda value: value["identity"]):
        digest.update(row["identity"].encode("ascii") + b"\0")
        digest.update(row["bank_sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()


def main(
    *,
    asset_root: Path,
    cache_dataset_root: Path,
    train_root: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    output_root: Path,
) -> int:
    started = time.monotonic()
    deadline = started + INTERNAL_DEADLINE_SECONDS
    if sha256(asset_root / "model.pt") != EXPECTED_MODEL_SHA256:
        raise EvalError("base model carrier mismatch")
    manifest_path = cache_dataset_root / "cache_dataset_manifest.json"
    if sha256(manifest_path) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise EvalError("fit105 cache manifest mismatch")
    if sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise EvalError("h005 checkpoint mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "celltrack-alpha.g03-phase2-h005.v1"
        or checkpoint.get("checkpoint_tag") != "h005"
        or checkpoint.get("immutable_final") is not True
        or checkpoint.get("parent_h000_sha256") != EXPECTED_PARENT_H000_SHA256
        or not 4.75 <= float(checkpoint.get("cumulative_t4_hours", -1)) <= 5.25
    ):
        raise EvalError("h005 checkpoint contract drift")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    identities = sorted(value for value in manifest["identities"] if value.startswith("train:44b6_"))
    if len(identities) != 36:
        raise EvalError("held-fit36 population drift")
    cache_root = cache_dataset_root / "reports/g02/fitcache_117/bh9out/cache"
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "banks").mkdir()
    (output_root / "receipts").mkdir()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise EvalError("two-T4 evaluation shape unavailable")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker(device_index: int, selected: list[str]) -> None:
        device = torch.device(f"cuda:{device_index}")
        try:
            torch.cuda.set_device(device)
            model, adapter = load_models(asset_root, checkpoint, device)
            for identity in selected:
                row = process_video(
                    identity, model, adapter, cache_root, train_root, output_root, device, deadline
                )
                with lock:
                    results.append(row)
                print(json.dumps({"EVAL": row}, sort_keys=True), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as exc:
            with lock:
                errors.append(
                    {
                        "device": str(device),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

    threads = [
        threading.Thread(target=worker, args=(index, identities[index::2]), daemon=False)
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    status = "SEALED_ALL_36_SCORE_BLIND" if not errors and len(results) == 36 else "PARTIAL_ERROR"
    seal = {
        "schema": "celltrack-alpha.g03-phase2-h005-cuda-bank-seal.v1",
        "status": status,
        "checkpoint": {"sha256": expected_checkpoint_sha256, "cumulative_t4_hours": checkpoint["cumulative_t4_hours"]},
        "population": {"name": "held-fit36", "identity_count": 36, "identities": identities},
        "inference": {
            "window_frames": WINDOW_FRAMES,
            "max_tokens": MAX_TOKENS,
            "radius_um": INFERENCE_RADIUS_UM,
            "neighbor_cap": INFERENCE_NEIGHBOR_CAP,
            "augmentation": False,
        },
        "aggregate_bank_sha256": aggregate(results) if len(results) == 36 else None,
        "bank_bytes": sum(row["bank_bytes"] for row in results),
        "candidate_edge_count": sum(row["candidate_edge_count"] for row in results),
        "results": sorted(results, key=lambda row: row["identity"]),
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "gt_access": False,
        "score_access": False,
        "cv57_access": False,
        "submission": False,
    }
    atomic_json(output_root / "BANK_SEAL.json", seal)
    print(json.dumps({"FINAL": status, "elapsed_seconds": seal["elapsed_seconds"]}), flush=True)
    return 0 if status == "SEALED_ALL_36_SCORE_BLIND" else 1
