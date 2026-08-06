#!/usr/bin/env python3
"""Fixed G-03 h005 end-to-end Trackastra/crop-adapter training core.

The Kaggle wrapper supplies only authenticated inputs and calls ``main``.  This
module deliberately has no CLI surface: the sole scientific candidate is fixed
by constants below.  Held-fit 44b6, CV57, audit, and test paths are never
constructed by this training core.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import yaml
import zarr


SCHEMA = "celltrack-alpha.g03-phase2-h005-training.v1"
CHECKPOINT_SCHEMA = "celltrack-alpha.g03-phase2-h005.v1"
RUN_SEED = 20260803
TRAIN_FAMILY = "6bba"
TRAIN_IDENTITY_COUNT = 69
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MATCH_RADIUS_UM = 7.0
WINDOW_FRAMES = 4
MAX_TOKENS = 384
ADAPTER_CROP_SHAPE = (3, 16, 32, 32)
BASE_LR = 1.0e-5
ADAPTER_LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 1.0
WARMUP_HOURS = 0.25
COSINE_HORIZON_HOURS = 90.0
TARGET_CUDA_SECONDS = 17_100.0  # 4.75 T4-hours, lower edge of fixed h005 window
PROCESS_HARD_KILL_SECONDS = 17_700.0  # 4.9167 hours, strictly below five hours
CHECKPOINT_EVERY_SECONDS = 900.0
MAX_PEAK_VRAM_BYTES = 15_000_000_000
MAX_CALIBRATION_STEP_SECONDS = 90.0
EXPECTED_MODEL_SHA256 = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
EXPECTED_ARCHIVE_SHA256 = "2ede08481b2e1f4ea23a41dd3001e2151ecbce7d978be9829ad230f0a60aea0e"
EXPECTED_H000_SHA256 = "603e02f8f152db642a177d57763a07bebf84518f868abbbe1e0246187b018f95"
EXPECTED_CACHE_MANIFEST_SHA256 = "ff7b205f58ad70c69954fed1c3e60a100d246c1e374a849c6ea7183934e98f28"


class H005Error(RuntimeError):
    """The fixed h005 run failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().to(device="cpu").contiguous().numpy()
        encoded = name.encode("utf-8")
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(array.shape).to_bytes(8, "big"))
        for extent in array.shape:
            digest.update(int(extent).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
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


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(value), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sha256(path)


class Phase2CropAdapter(nn.Module):
    """The immutable Phase-2 22,684-parameter crop adapter."""

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

    @staticmethod
    def normalize(crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 5 or tuple(crops.shape[1:]) != ADAPTER_CROP_SHAPE:
            raise H005Error("crop shape drift")
        value = crops.float()
        mean = value.mean(dim=(1, 2, 3, 4), keepdim=True)
        centered = value - mean
        rms = torch.sqrt(centered.square().mean(dim=(1, 2, 3, 4), keepdim=True))
        return torch.clamp(centered / rms.clamp_min(1.0e-6), -5.0, 5.0)

    def hidden(self, crops: torch.Tensor) -> torch.Tensor:
        value = self.spatial_pool(self.normalize(crops))
        value = self.activation1(self.conv1(value))
        value = self.activation2(self.conv2(value))
        return self.global_pool(value).flatten(1)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        value = self.output_activation(self.projection(self.hidden(crops)))
        if value.shape[1:] != (12,) or not bool(torch.isfinite(value).all()):
            raise H005Error("adapter output drift")
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


def load_gt(train_root: Path, dataset: str) -> dict[str, Any]:
    group = zarr.open_group(str(train_root / f"{dataset}.geff"), mode="r")
    nodes = {
        key: np.asarray(group[path], dtype=np.int64)
        for key, path in {
            "id": "nodes/ids",
            "t": "nodes/props/t/values",
            "z": "nodes/props/z/values",
            "y": "nodes/props/y/values",
            "x": "nodes/props/x/values",
        }.items()
    }
    edges = np.asarray(group["edges/ids"], dtype=np.int64)
    return {"nodes": nodes, "edges": edges}


def frame_rows(detections: Sequence[Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for frame in detections:
        result[int(frame.frame_index)] = (
            np.asarray(frame.node_ids, dtype=np.int64),
            np.asarray(frame.coords_zyx, dtype=np.float32),
        )
    return result


def match_predictions(
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]], gt: Mapping[str, Any]
) -> dict[int, int]:
    nodes = gt["nodes"]
    gt_by_t: dict[int, list[int]] = {}
    for index, t in enumerate(nodes["t"]):
        gt_by_t.setdefault(int(t), []).append(index)
    result: dict[int, int] = {}
    for t, (pred_ids, pred_coords) in frames.items():
        gt_indices = gt_by_t.get(t, [])
        if len(pred_ids) == 0 or not gt_indices:
            continue
        gt_coords = np.stack(
            [nodes[axis][gt_indices] for axis in ("z", "y", "x")], axis=1
        ).astype(np.float64)
        delta = (pred_coords[:, None, :].astype(np.float64) - gt_coords[None])
        distances = np.sqrt(np.sum((delta * SCALE_ZYX[None, None]) ** 2, axis=2))
        feasible = distances <= MATCH_RADIUS_UM
        if not bool(feasible.any()):
            continue
        cost = np.where(feasible, distances, MATCH_RADIUS_UM + 1.0e6)
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            if distances[row, col] <= MATCH_RADIUS_UM:
                result[int(pred_ids[row])] = int(nodes["id"][gt_indices[col]])
    return result


def build_samples(
    identity: str,
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]],
    gt: Mapping[str, Any],
    pred_to_gt: Mapping[int, int],
) -> list[dict[str, Any]]:
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()}
    gt_coords = {
        int(node): np.asarray((z, y, x), dtype=np.float64)
        for node, z, y, x in zip(
            gt["nodes"]["id"],
            gt["nodes"]["z"],
            gt["nodes"]["y"],
            gt["nodes"]["x"],
            strict=True,
        )
    }
    gt_time = {
        int(node): int(t)
        for node, t in zip(gt["nodes"]["id"], gt["nodes"]["t"], strict=True)
    }
    samples = []
    for source_gt, target_gt in np.asarray(gt["edges"], dtype=np.int64):
        source_gt, target_gt = int(source_gt), int(target_gt)
        if source_gt not in gt_to_pred or target_gt not in gt_to_pred:
            continue
        source_t = gt_time[source_gt]
        target_t = gt_time[target_gt]
        if target_t <= source_t or target_t - source_t > 2:
            continue
        midpoint = (gt_coords[source_gt] + gt_coords[target_gt]) / 2.0
        digest = hashlib.sha256(
            f"{identity}:{source_gt}:{target_gt}".encode("ascii")
        ).hexdigest()
        samples.append(
            {
                "sample_id": digest,
                "source_gt": source_gt,
                "target_gt": target_gt,
                "source_t": source_t,
                "start_t": max(0, source_t - 1),
                "midpoint": midpoint,
            }
        )
    samples.sort(key=lambda row: (row["source_t"], row["sample_id"]))
    return samples


def selected_window(
    sample: Mapping[str, Any],
    frames: Mapping[int, tuple[np.ndarray, np.ndarray]],
    pred_to_gt: Mapping[int, int],
    gt_edges: np.ndarray,
) -> dict[str, Any]:
    candidates = []
    midpoint = np.asarray(sample["midpoint"], dtype=np.float64)
    for t in range(int(sample["start_t"]), int(sample["start_t"]) + WINDOW_FRAMES):
        ids, coords = frames.get(
            t,
            (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)),
        )
        for node, coord in zip(ids, coords, strict=True):
            distance2 = float(np.sum(((coord - midpoint) * SCALE_ZYX) ** 2))
            candidates.append((distance2, t, int(node), np.asarray(coord, dtype=np.float32)))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    mandatory_gt = {int(sample["source_gt"]), int(sample["target_gt"])}
    mandatory = {node for node, gt_id in pred_to_gt.items() if gt_id in mandatory_gt}
    chosen = [row for row in candidates if row[2] in mandatory]
    chosen_nodes = {row[2] for row in chosen}
    for row in candidates:
        if len(chosen) >= MAX_TOKENS:
            break
        if row[2] not in chosen_nodes:
            chosen.append(row)
            chosen_nodes.add(row[2])
    chosen.sort(key=lambda row: (row[1], row[2]))
    if not mandatory.issubset(chosen_nodes) or len(chosen) < 2:
        raise H005Error("mandatory positive endpoints missing from training window")
    node_ids = np.asarray([row[2] for row in chosen], dtype=np.int64)
    times = np.asarray([row[1] for row in chosen], dtype=np.int64)
    coords = np.stack([row[3] for row in chosen]).astype(np.float32)
    index = {int(node): idx for idx, node in enumerate(node_ids)}
    gt_by_pred = [pred_to_gt.get(int(node)) for node in node_ids]
    adjacency: dict[int, set[int]] = {}
    for source, target in np.asarray(gt_edges, dtype=np.int64):
        adjacency.setdefault(int(source), set()).add(int(target))
    related: dict[int, set[int]] = {}
    for gt_id in {value for value in gt_by_pred if value is not None}:
        reached = set()
        frontier = {int(gt_id)}
        for _ in range(2):
            frontier = {nxt for node in frontier for nxt in adjacency.get(node, set())}
            reached.update(frontier)
        related[int(gt_id)] = reached
    assoc = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    for i, left in enumerate(gt_by_pred):
        if left is None:
            continue
        for j, right in enumerate(gt_by_pred):
            if right is None or i == j:
                continue
            if right in related.get(left, set()) or left in related.get(right, set()):
                assoc[i, j] = 1.0
    source_pred = next(node for node, gt_id in pred_to_gt.items() if gt_id == sample["source_gt"])
    target_pred = next(node for node, gt_id in pred_to_gt.items() if gt_id == sample["target_gt"])
    if assoc[index[source_pred], index[target_pred]] != 1.0:
        raise H005Error("anchor edge is not positive in association target")
    return {"node_ids": node_ids, "times": times, "coords": coords, "assoc": assoc}


class FrameCache:
    def __init__(self, image: Any, maximum: int = 10) -> None:
        self.image = image
        self.maximum = maximum
        self.values: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, t: int) -> np.ndarray:
        t = min(max(int(t), 0), int(self.image.shape[0]) - 1)
        if t not in self.values:
            self.values[t] = np.asarray(self.image[t], dtype=np.float32)
            while len(self.values) > self.maximum:
                self.values.popitem(last=False)
        self.values.move_to_end(t)
        return self.values[t]


def crop_indices(coords: np.ndarray, shape: Sequence[int]) -> tuple[np.ndarray, ...]:
    centers = np.rint(np.asarray(coords, dtype=np.float64)).astype(np.int64)
    offsets = (
        np.arange(-8, 8, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
    )
    return tuple(
        np.clip(centers[:, axis, None] + offsets[axis], 0, int(shape[axis]) - 1)
        for axis in range(3)
    )


def extract_crops(
    frame_cache: FrameCache, times: np.ndarray, coords: np.ndarray
) -> torch.Tensor:
    output = np.empty((len(coords), *ADAPTER_CROP_SHAPE), dtype=np.float32)
    for t in np.unique(times):
        mask = np.flatnonzero(times == t)
        frames = [frame_cache.get(int(t) + offset) for offset in (-1, 0, 1)]
        z_idx, y_idx, x_idx = crop_indices(coords[mask], frames[0].shape)
        for channel, frame in enumerate(frames):
            output[mask, channel] = frame[
                z_idx[:, :, None, None],
                y_idx[:, None, :, None],
                x_idx[:, None, None, :],
            ]
    if not np.isfinite(output).all():
        raise H005Error("nonfinite raw crop")
    return torch.from_numpy(output)


def augment(crops: torch.Tensor, sample_id: str, global_step: int) -> torch.Tensor:
    """Stateless object-centred flips; evaluation never calls this function."""
    digest = hashlib.sha256(
        f"{RUN_SEED}:{global_step}:{sample_id}".encode("ascii")
    ).digest()
    value = crops
    if digest[0] & 1:
        value = torch.flip(value, dims=(-1,))
    if digest[1] & 1:
        value = torch.flip(value, dims=(-2,))
    if digest[2] & 1:
        value = torch.flip(value, dims=(-3,))
    return value.contiguous()


def schedule_factor(cuda_seconds: float) -> float:
    hours = cuda_seconds / 3600.0
    warmup = min(hours / WARMUP_HOURS, 1.0)
    phase = min(hours / COSINE_HORIZON_HOURS, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    return warmup * cosine


def association_loss(model, coords, features, times, target):
    from trackastra.utils import blockwise_causal_norm, blockwise_sum

    logits = model(coords, features)
    logits = logits.clamp(
        torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    normalized = torch.stack(
        [
            blockwise_causal_norm(row, row_times, mode="quiet_softmax")
            for row, row_times in zip(logits, times, strict=True)
        ]
    )
    parental = torch.nn.functional.binary_cross_entropy(
        normalized, target, reduction="none"
    )
    loss = 0.01 * raw + parental
    with torch.no_grad():
        sums1 = torch.stack(
            [blockwise_sum(row, row_times, dim=-1) for row, row_times in zip(target, times, strict=True)]
        )
        sums2 = torch.stack(
            [blockwise_sum(row, row_times, dim=-2) for row, row_times in zip(target, times, strict=True)]
        )
        block_sum = target * (sums1 + sums2)
        weights = 1.0 + (block_sum == 2).float() + 20.0 * (block_sum > 2).float()
    loss = loss * weights
    dt = times.unsqueeze(1) - times.unsqueeze(2)
    mask = ((dt > 0) & (dt <= 2)).float()
    per_sample = (loss * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    prefactor = mask.sum(dim=(1, 2)).pow(0.2)
    result = (per_sample * prefactor / prefactor.sum().clamp_min(1.0e-7)).sum()
    return result, logits


def make_checkpoint(
    *,
    model,
    adapter,
    optimizer,
    global_step: int,
    cuda_seconds: float,
    epoch: int,
    identity_ordinal: int,
    sample_ordinal: int,
    previous_checkpoint_sha256: str | None,
    final: bool,
) -> dict[str, Any]:
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    adapter_state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_tag": "h005" if final else "h005-partial",
        "immutable_final": bool(final),
        "parent_h000_sha256": EXPECTED_H000_SHA256,
        "trackastra_model_sha256": EXPECTED_MODEL_SHA256,
        "cumulative_t4_hours": cuda_seconds / 3600.0,
        "global_step": int(global_step),
        "model": {
            "state_dict": model_state,
            "state_tensor_sha256": state_tensor_sha256(model_state),
        },
        "adapter": {
            "state_dict": adapter_state,
            "state_tensor_sha256": state_tensor_sha256(adapter_state),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "kind": "elapsed-time-warmup-cosine",
            "warmup_hours": WARMUP_HOURS,
            "horizon_hours": COSINE_HORIZON_HOURS,
            "last_factor": schedule_factor(cuda_seconds),
        },
        "sampler": {
            "epoch": int(epoch),
            "identity_ordinal": int(identity_ordinal),
            "sample_ordinal": int(sample_ordinal),
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
    }


def load_model(asset_root: Path, device: torch.device):
    from trackastra.model.model import TrackingTransformer

    with zipfile.ZipFile(asset_root / "ctc.zip.bin") as archive:
        config = yaml.safe_load(archive.read("ctc/config.yaml"))
    model = TrackingTransformer(**config)
    state = torch.load(
        asset_root / "model.pt", map_location="cpu", mmap=True, weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.to(device)


def validate_inputs(asset_root: Path, cache_dataset_root: Path, h000_path: Path) -> tuple[Path, list[str]]:
    if sha256(asset_root / "model.pt") != EXPECTED_MODEL_SHA256:
        raise H005Error("Trackastra model mismatch")
    if sha256(asset_root / "ctc.zip.bin") != EXPECTED_ARCHIVE_SHA256:
        raise H005Error("Trackastra archive mismatch")
    if sha256(h000_path) != EXPECTED_H000_SHA256:
        raise H005Error("h000 mismatch")
    manifest_path = cache_dataset_root / "cache_dataset_manifest.json"
    if sha256(manifest_path) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise H005Error("fit105 cache manifest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    identities = sorted(
        value for value in manifest["identities"] if value.startswith("train:6bba_")
    )
    if len(identities) != TRAIN_IDENTITY_COUNT:
        raise H005Error("development identity population drift")
    cache_root = cache_dataset_root / "reports/g02/fitcache_117/bh9out/cache"
    if any(not (cache_root / f"{identity.replace(':', '__', 1)}.bhc").is_file() for identity in identities):
        raise H005Error("development cache incomplete")
    return cache_root, identities


def main(
    *,
    asset_root: Path,
    cache_dataset_root: Path,
    train_root: Path,
    h000_path: Path,
    output_root: Path,
) -> int:
    process_started = time.monotonic()
    hard_deadline = process_started + PROCESS_HARD_KILL_SECONDS
    output_root.mkdir(parents=True, exist_ok=False)
    log_path = output_root / "TRAINING.jsonl"
    cache_root, identities = validate_inputs(asset_root, cache_dataset_root, h000_path)
    if not train_root.is_dir() or any("44b6" in identity for identity in identities):
        raise H005Error("training source isolation failed")
    h000 = torch.load(h000_path, map_location="cpu", weights_only=False)
    if h000.get("checkpoint_tag") != "h000" or h000["adapter"]["output_probe_nonzero"] != 0:
        raise H005Error("h000 payload drift")
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise H005Error("T4 CUDA device unavailable")
    device = torch.device("cuda:0")
    if "T4" not in torch.cuda.get_device_name(device):
        raise H005Error("execution device is not a T4")
    model = load_model(asset_root, device)
    adapter = Phase2CropAdapter().to(device)
    adapter.load_state_dict(h000["adapter"]["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in adapter.parameters()) != 22_684:
        raise H005Error("adapter parameter census drift")
    if int(torch.count_nonzero(adapter.projection.weight)) or int(torch.count_nonzero(adapter.projection.bias)):
        raise H005Error("h000 projection is not exact zero")
    model.train()
    adapter.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": BASE_LR, "name": "trackastra"},
            {"params": list(adapter.parameters()), "lr": ADAPTER_LR, "name": "crop_adapter"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    trainable = sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"])
    if trainable != 27_456_880 + 22_684:
        raise H005Error("end-to-end trainability census drift")
    cuda_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    global_step = 0
    last_checkpoint_seconds = 0.0
    previous_checkpoint_sha = None
    losses: list[float] = []
    calibration = None
    final_cursor = (0, 0, 0)
    stop = False
    epoch = 0
    while not stop:
        epoch += 1
        for identity_ordinal, identity in enumerate(identities):
            dataset = identity.split(":", 1)[1]
            # This is the only raw/GT path construction before h005 publication.
            if not dataset.startswith("6bba_"):
                raise H005Error("non-development path construction")
            detections = load_detections(cache_root, identity)
            frames = frame_rows(detections)
            gt = load_gt(train_root, dataset)
            pred_to_gt = match_predictions(frames, gt)
            samples = build_samples(identity, frames, gt, pred_to_gt)
            if not samples:
                raise H005Error(f"no authenticated positives for {identity}")
            image = zarr.open_group(str(train_root / f"{dataset}.zarr"), mode="r")["0"]
            frame_cache = FrameCache(image)
            for sample_ordinal, sample in enumerate(samples):
                now = time.monotonic()
                cuda_seconds = now - cuda_started
                if now >= hard_deadline - 180.0:
                    raise H005Error("hard deadline approached before h005 window")
                if cuda_seconds >= TARGET_CUDA_SECONDS:
                    final_cursor = (epoch, identity_ordinal, sample_ordinal)
                    stop = True
                    break
                batch = selected_window(sample, frames, pred_to_gt, gt["edges"])
                crops = augment(
                    extract_crops(frame_cache, batch["times"], batch["coords"]),
                    sample["sample_id"],
                    global_step,
                ).to(device=device, non_blocking=True)
                coords_np = np.concatenate(
                    (batch["times"][:, None].astype(np.float32), batch["coords"]), axis=1
                )
                coords = torch.from_numpy(coords_np).unsqueeze(0).to(device)
                times = torch.from_numpy(batch["times"]).unsqueeze(0).to(device)
                target = torch.from_numpy(batch["assoc"]).unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                step_started = time.monotonic()
                features = adapter(crops).unsqueeze(0)
                loss, _ = association_loss(model, coords, features, times, target)
                if not bool(torch.isfinite(loss)):
                    raise H005Error("nonfinite association loss")
                loss.backward()
                parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if not parameters or any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                    raise H005Error("missing or nonfinite gradients")
                torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
                if time.monotonic() >= hard_deadline - 120.0:
                    raise H005Error("hard deadline reached before optimizer step")
                cuda_seconds = time.monotonic() - cuda_started
                factor = schedule_factor(cuda_seconds)
                optimizer.param_groups[0]["lr"] = BASE_LR * factor
                optimizer.param_groups[1]["lr"] = ADAPTER_LR * factor
                optimizer.step()
                torch.cuda.synchronize(device)
                step_seconds = time.monotonic() - step_started
                global_step += 1
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                peak = int(torch.cuda.max_memory_allocated(device))
                if calibration is None:
                    calibration = {
                        "step_seconds": step_seconds,
                        "tokens": len(batch["node_ids"]),
                        "peak_vram_bytes": peak,
                        "status": "PASS",
                    }
                    if peak > MAX_PEAK_VRAM_BYTES or step_seconds > MAX_CALIBRATION_STEP_SECONDS:
                        calibration["status"] = "FAIL_RESOURCE_CEILING"
                        atomic_json(output_root / "RESOURCE_CALIBRATION.json", calibration)
                        raise H005Error("resource calibration ceiling failed")
                    atomic_json(output_root / "RESOURCE_CALIBRATION.json", calibration)
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "identity": identity,
                    "sample_id": sample["sample_id"],
                    "tokens": len(batch["node_ids"]),
                    "positives": int(batch["assoc"].sum()),
                    "loss": loss_value,
                    "step_seconds": step_seconds,
                    "cuda_seconds": time.monotonic() - cuda_started,
                    "lr_factor": factor,
                }
                with log_path.open("a", encoding="ascii") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                elapsed = time.monotonic() - cuda_started
                if elapsed - last_checkpoint_seconds >= CHECKPOINT_EVERY_SECONDS:
                    partial = make_checkpoint(
                        model=model,
                        adapter=adapter,
                        optimizer=optimizer,
                        global_step=global_step,
                        cuda_seconds=elapsed,
                        epoch=epoch,
                        identity_ordinal=identity_ordinal,
                        sample_ordinal=sample_ordinal + 1,
                        previous_checkpoint_sha256=previous_checkpoint_sha,
                        final=False,
                    )
                    partial_path = output_root / "h005-partial.pt"
                    previous_checkpoint_sha = atomic_torch_save(partial_path, partial)
                    last_checkpoint_seconds = elapsed
                del crops, coords, times, target, features, loss
            del detections, frames, gt, pred_to_gt, samples, image, frame_cache
            if stop:
                break
    cuda_seconds = time.monotonic() - cuda_started
    if not 4.75 <= cuda_seconds / 3600.0 <= 5.25:
        raise H005Error("fixed h005 cumulative-hour window missed")
    if time.monotonic() >= hard_deadline:
        raise H005Error("process hard kill exceeded")
    epoch, identity_ordinal, sample_ordinal = final_cursor
    checkpoint = make_checkpoint(
        model=model,
        adapter=adapter,
        optimizer=optimizer,
        global_step=global_step,
        cuda_seconds=cuda_seconds,
        epoch=epoch,
        identity_ordinal=identity_ordinal,
        sample_ordinal=sample_ordinal,
        previous_checkpoint_sha256=previous_checkpoint_sha,
        final=True,
    )
    checkpoint_path = output_root / "h005.pt"
    checkpoint_sha = atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE_H005_IMMUTABLE_CHECKPOINT",
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
            "cumulative_t4_hours": cuda_seconds / 3600.0,
            "global_step": global_step,
            "model_state_tensor_sha256": checkpoint["model"]["state_tensor_sha256"],
            "adapter_state_tensor_sha256": checkpoint["adapter"]["state_tensor_sha256"],
        },
        "scope": {
            "training_family": TRAIN_FAMILY,
            "training_identity_count": TRAIN_IDENTITY_COUNT,
            "held_fit_44b6_access": False,
            "cv57_access": False,
            "audit_access": False,
            "test_access": False,
            "submission": False,
        },
        "training": {
            "end_to_end": True,
            "trackastra_parameters": 27_456_880,
            "adapter_parameters": 22_684,
            "optimizer_steps": global_step,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_mean": float(np.mean(losses)),
            "base_lr": BASE_LR,
            "adapter_lr": ADAPTER_LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRAD_CLIP,
            "augmentation": "stateless object-centred independent z/y/x flips",
        },
        "resource_calibration": calibration,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "process_elapsed_seconds": time.monotonic() - process_started,
            "process_hard_kill_seconds": PROCESS_HARD_KILL_SECONDS,
        },
    }
    atomic_json(output_root / "TRAINING_REPORT.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0

