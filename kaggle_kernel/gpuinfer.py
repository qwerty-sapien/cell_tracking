#!/usr/bin/env python3
"""Bounded, image-only dual-GPU inference for the compact G02 checkpoint.

The normal Kaggle invocation is::

    python fastsrc.zip --mode submit --root /kaggle/input/biohub-cell-tracking-during-development \
      --checkpoint /kaggle/input/<checkpoint-dataset>/t03-v6.pt

Use ``--mode benchmark`` to require the exact 203-image registry.  Benchmark
workers install a read audit that rejects GEFF, annotation, scorer, and split
manifest paths.  ``--dry-run --mock`` is a tiny CPU plumbing check; mock mode
is deliberately unavailable for a production run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence
import zipfile


CKPT_SHA = "712646e9e263da018239dc46e927905708d31405c0b1774d20a260d85f480987"
CKPT_BYTES = 6_375_639
DOWN = (1, 4, 4)
CSV_FIELDS = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
)
EXPECTED_ALL = 203
EXPECTED_TEST = 4


class FastError(RuntimeError):
    """Raised before an unsafe read or after an invariant violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_deadline(deadline_epoch: float, stage: str) -> None:
    if time.time() >= deadline_epoch:
        raise FastError(f"runtime deadline reached during {stage}")


def _atomic_json(path: Path, value: Mapping[str, Any], deadline_epoch: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if deadline_epoch is not None:
            _check_deadline(deadline_epoch, f"{path.name} finalization")
        os.replace(name, path)
        if deadline_epoch is not None:
            try:
                _check_deadline(deadline_epoch, f"{path.name} finalization")
            except FastError:
                path.unlink(missing_ok=True)
                raise
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def forbidden_read(path: str | os.PathLike[str]) -> str | None:
    """Return why *path* is not an image/runtime input, or ``None``."""

    text = os.fspath(path).replace("\\", "/").casefold()
    parts = [part for part in text.split("/") if part]
    for part in parts:
        if part.endswith(".geff") or ".geff/" in text:
            return "GEFF"
        stem = part.rsplit(".", 1)[0]
        if "annotation" in stem or stem in {"label", "labels", "ground_truth", "gt"}:
            return "annotation"
        if "scorer" in stem or stem in {"score", "scores", "metric", "metrics"}:
            return "scorer"
        if stem in {"split", "splits", "dataset_split", "dataset_splits"} or stem.endswith("_splits"):
            return "split"
    return None


def _install_read_audit() -> dict[str, int]:
    state = {"blocked": 0}

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        reading = mode is None or (isinstance(mode, str) and ("r" in mode or "+" in mode))
        if isinstance(flags, int):
            reading = reading or (flags & os.O_WRONLY == 0)
        reason = forbidden_read(args[0]) if reading else None
        if reason:
            state["blocked"] += 1
            raise PermissionError(f"benchmark image-only policy denied {reason} read: {os.fspath(args[0])}")

    sys.addaudithook(audit)
    return state


def _registry_bytes(path: Path | None) -> bytes:
    if path is not None:
        reason = forbidden_read(path)
        if reason:
            raise FastError(f"registry path is forbidden ({reason}): {path}")
        return path.read_bytes()
    loose = Path(__file__).with_name("fastreg.json")
    if loose.is_file():
        return loose.read_bytes()
    entry = Path(sys.argv[0]).resolve()
    if entry.is_file() and zipfile.is_zipfile(entry):
        with zipfile.ZipFile(entry) as archive:
            return archive.read("fastreg.json")
    raise FastError("--registry is required outside the packaged entrypoint")


def load_registry(path: Path | None) -> dict[str, Any]:
    try:
        registry = json.loads(_registry_bytes(path))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise FastError("invalid compact image registry") from exc
    if not isinstance(registry, dict) or registry.get("artifact") != "fast_image_registry":
        raise FastError("registry is not a fast_image_registry projection")
    records = registry.get("records")
    if not isinstance(records, list) or not records:
        raise FastError("registry has no image records")
    identities: set[str] = set()
    total_chunks = 0
    for row in records:
        if not isinstance(row, dict):
            raise FastError("registry record is not an object")
        split, dataset = row.get("split"), row.get("dataset")
        shape, chunks = row.get("shape"), row.get("chunks")
        identity = f"{split}:{dataset}"
        if split not in {"train", "test"} or not isinstance(dataset, str) or identity in identities:
            raise FastError("registry image identity is invalid or duplicated")
        if not isinstance(shape, list) or len(shape) != 4 or any(not isinstance(v, int) or v < 1 for v in shape):
            raise FastError(f"registry shape is invalid for {identity}")
        if not isinstance(chunks, int) or chunks < 1:
            raise FastError(f"registry chunk count is invalid for {identity}")
        identities.add(identity)
        total_chunks += chunks
    counts = registry.get("counts")
    if not isinstance(counts, dict) or counts.get("videos") != len(records) or counts.get("image_chunks") != total_chunks:
        raise FastError("registry aggregate counts do not match records")
    return registry


def discover_jobs(
    root: Path,
    registry: Mapping[str, Any],
    mode: str,
    dry_run: bool,
    max_datasets: int | None,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in registry["records"] if mode == "benchmark" or row["split"] == "test"]
    expected = EXPECTED_ALL if mode == "benchmark" else EXPECTED_TEST
    if not dry_run and len(selected) != expected:
        raise FastError(f"{mode} requires exactly {expected} registered images, got {len(selected)}")
    if dry_run:
        selected = selected[: max_datasets or 1]
    elif max_datasets is not None:
        raise FastError("--max-datasets is dry-run only")
    if not selected:
        raise FastError("no registered datasets selected")
    jobs: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in selected:
        image = root / row["split"] / f"{row['dataset']}.zarr"
        if not image.is_dir():
            missing.append(f"{row['split']}:{row['dataset']}")
            continue
        jobs.append({**row, "path": str(image), "identity": f"{row['split']}:{row['dataset']}"})
    if missing:
        raise FastError(f"registered image coverage missing {len(missing)} paths: {missing[:8]}")
    if len(jobs) != len(selected) or len({job["identity"] for job in jobs}) != len(jobs):
        raise FastError("discovered image coverage is incomplete or duplicated")
    return jobs


def shard_jobs(jobs: Sequence[Mapping[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Greedily balance immutable image jobs by registered voxel work."""

    if workers < 1:
        raise ValueError("workers must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    weighted = sorted(
        (dict(job) for job in jobs),
        key=lambda row: (-math.prod(row["shape"]), row["identity"]),
    )
    for job in weighted:
        rank = min(range(workers), key=lambda index: (loads[index], index))
        shards[rank].append(job)
        loads[rank] += math.prod(job["shape"])
    flat = [job["identity"] for shard in shards for job in shard]
    if len(flat) != len(jobs) or len(set(flat)) != len(flat):
        raise FastError("GPU sharding lost or duplicated a dataset")
    return shards


def _parse_scale(attrs: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        multiscale = attrs["multiscales"][0]
        axes = [str(axis["name"] if isinstance(axis, dict) else axis).strip().casefold() for axis in multiscale["axes"]]
        transforms = multiscale["datasets"][0]["coordinateTransformations"]
        scale = next(item["scale"] for item in transforms if item.get("type") == "scale")
        values = tuple(float(scale[axes.index(axis)]) for axis in ("z", "y", "x"))
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
        raise FastError("image has no usable OME-NGFF z/y/x scale") from exc
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise FastError("image scale must be positive and finite")
    return values


def _quantiles(attrs: Mapping[str, Any]) -> tuple[float, float]:
    try:
        raw = attrs["image_statistics"]["quantiles"]
        by_float = {float(key): float(value) for key, value in raw.items()}
        low = next(value for key, value in by_float.items() if abs(key - 0.001) <= 1e-8)
        high = next(value for key, value in by_float.items() if abs(key - 0.999) <= 1e-8)
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise FastError("image is missing finite 0.001/0.999 quantiles") from exc
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise FastError("image quantile range is invalid")
    return low, high


def _chunk_count(shape: Sequence[int], chunks: Sequence[int]) -> int:
    if len(shape) != len(chunks) or any(int(value) < 1 for value in chunks):
        raise FastError("Zarr chunk metadata is invalid")
    return math.prod(math.ceil(int(size) / int(chunk)) for size, chunk in zip(shape, chunks, strict=True))


def _pool_kernel(distance_um: float, voxel_um: Sequence[float]) -> tuple[int, int, int]:
    values = []
    for scale in voxel_um:
        width = max(1, round(distance_um / float(scale)))
        values.append(width if width % 2 else width + 1)
    return tuple(values)  # type: ignore[return-value]


def sparse_link(
    previous: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    scale: Sequence[float],
    radius: float,
    neighbor_cap: int,
    threshold: float,
    max_candidates: int,
) -> tuple[list[tuple[int, int]], int]:
    """Bounded physical-space k-nearest linking without a dense N-by-N matrix."""

    if radius <= 0 or neighbor_cap < 1 or max_candidates < 1:
        raise ValueError("invalid sparse-link bounds")
    if not previous or not current:
        return [], 0
    query_cap = min(neighbor_cap, len(previous))
    if len(current) * query_cap > max_candidates:
        raise FastError("sparse pair-evaluation ceiling would be exceeded")
    ordered = sorted(previous, key=lambda node: int(node["node_id"]))
    source_points = [
        tuple(float(node[axis]) * float(scale[i]) for i, axis in enumerate(("z", "y", "x")))
        for node in ordered
    ]
    target_points = [
        tuple(float(node[axis]) * float(scale[i]) for i, axis in enumerate(("z", "y", "x")))
        for node in current
    ]
    predicted_points = [
        tuple(point[i] + float(node["backward_um"][i]) for i in range(3))
        for node, point in zip(current, target_points, strict=True)
    ]

    neighbor_rows: list[list[tuple[float, int]]] = []
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(source_points).query(
            predicted_points, k=query_cap, distance_upper_bound=radius, workers=1,
        )
        if query_cap == 1:
            distances = [[value] for value in distances]
            indices = [[value] for value in indices]
        neighbor_rows = [
            [(float(distance), int(index)) for distance, index in zip(row_d, row_i, strict=True)
             if math.isfinite(float(distance)) and int(index) < len(ordered)]
            for row_d, row_i in zip(distances, indices, strict=True)
        ]
    except ImportError:
        # The production image provides SciPy.  This deterministic emergency
        # path enumerates and evaluates no more than query_cap pairs per target.
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(source_points):
            cell = tuple(math.floor(value / radius) for value in point)
            bucket = buckets.setdefault(cell, [])
            if len(bucket) < query_cap:
                bucket.append(index)
        offsets = sorted(
            ((dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
            key=lambda offset: (sum(value * value for value in offset), offset),
        )
        for predicted in predicted_points:
            center = tuple(math.floor(value / radius) for value in predicted)
            selected: list[int] = []
            for dz, dy, dx in offsets:
                for index in buckets.get((center[0] + dz, center[1] + dy, center[2] + dx), ()):
                    selected.append(index)
                    if len(selected) == query_cap:
                        break
                if len(selected) == query_cap:
                    break
            neighbor_rows.append([
                (math.sqrt(sum((source_points[index][axis] - predicted[axis]) ** 2 for axis in range(3))), index)
                for index in selected
            ])

    candidates: list[tuple[float, float, int, int]] = []
    for target, target_point, neighbors in zip(current, target_points, neighbor_rows, strict=True):
        ranked: list[tuple[float, float, int]] = []
        for residual, index in neighbors[:query_cap]:
            source_point = source_points[index]
            actual = math.sqrt(sum((source_point[i] - target_point[i]) ** 2 for i in range(3)))
            if residual <= radius and actual <= radius:
                ranked.append((residual, actual, int(ordered[index]["node_id"])))
        ranked.sort()
        for residual, actual, source_id in ranked:
            score = max(0.0, 1.0 - residual / radius)
            if score >= threshold:
                candidates.append((-score, actual, source_id, int(target["node_id"])))
                if len(candidates) > max_candidates:
                    raise FastError("sparse candidate ceiling exceeded")
    candidates.sort()
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    edges: list[tuple[int, int]] = []
    for _negative_score, _actual, source, target in candidates:
        if source in used_sources or target in used_targets:
            continue
        edges.append((source, target))
        used_sources.add(source)
        used_targets.add(target)
    if len(edges) > min(len(previous), len(current)):
        raise FastError("sparse linker emitted an impossible edge count")
    return edges, len(candidates)


def _build_model(torch: Any) -> Any:
    """Define the exact released TemporalUNet + compact adapter tensor schema."""

    nn, functional = torch.nn, torch.nn.functional

    def conv_block(in_channels: int, out_channels: int) -> Any:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
        )

    class TemporalAttention(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)

        def forward(self, x: Any) -> Any:
            batch, steps, channels = x.shape[:3]
            spatial = x.shape[3:]
            count = math.prod(spatial)
            value = x.reshape(batch, steps, channels, count).permute(0, 3, 1, 2).reshape(batch * count, steps, channels)
            value = self.norm(value)
            value, _ = self.attn(value, value, value, need_weights=False)
            value = value.reshape(batch, count, steps, channels).permute(0, 2, 3, 1).reshape(batch, steps, channels, *spatial)
            return x + value

    class TemporalUNet3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = (32, 64, 128)
            self.gradient_checkpointing = True
            self.encoder_blocks = nn.ModuleList()
            self.temporal_blocks = nn.ModuleList()
            previous = 1
            for index, channels in enumerate(layers):
                self.encoder_blocks.append(conv_block(previous, channels))
                self.temporal_blocks.append(nn.Identity() if index == 0 else TemporalAttention(channels))
                previous = channels
            self.pool = nn.MaxPool3d(2, 2)
            self.upsamples = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            for index in range(len(layers) - 1, 0, -1):
                self.upsamples.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
                self.decoder_blocks.append(conv_block(layers[index] + layers[index - 1], layers[index - 1]))
            self.head = nn.Conv3d(layers[0], 32, 1)

        def forward(self, value: Any) -> Any:
            batch, steps = value.shape[:2]
            value = value.reshape(batch * steps, *value.shape[2:])
            skips = []
            for index, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks, strict=True)):
                if index:
                    value = self.pool(value)
                value = block(value)
                value = temporal(value.reshape(batch, steps, *value.shape[1:])).reshape(batch * steps, *value.shape[1:])
                if index < len(self.encoder_blocks) - 1:
                    skips.append(value)
            for upsample, block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips), strict=True):
                value = upsample(value)
                if value.shape[2:] != skip.shape[2:]:
                    value = functional.interpolate(value, size=skip.shape[2:], mode="trilinear", align_corners=False)
                value = block(torch.cat((value, skip), dim=1))
            value = self.head(value)
            return value.reshape(batch, steps, *value.shape[1:])

    class Specialist(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reduce = nn.Conv3d(32, 8, 1)
            self.expand = nn.Conv3d(8, 32, 1)
            self.calibrate = nn.Conv3d(6, 6, 1)

        def forward(self, features: Any, universal: Any, aux_head: Any) -> Any:
            adjusted = features + self.expand(functional.gelu(self.reduce(features)))
            branch = torch.cat((universal[:, :1], aux_head(adjusted)), dim=1)
            return branch + self.calibrate(branch)

    class Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = TemporalUNet3D()
            self.detect_head = nn.Conv3d(32, 1, 1)
            self.aux_head = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, 5, 1))
            self.specialists = nn.ModuleDict({name: Specialist() for name in ("sparse_clean", "dense_crowded", "low_snr_depth")})

        def forward(self, images: Any) -> Any:
            batch, steps, _channels, z, y, x = images.shape
            features = self.backbone(images).reshape(batch * steps, 32, z, y, x)
            output = torch.cat((self.detect_head(features), self.aux_head(features)), dim=1)
            return output.reshape(batch, steps, 6, z, y, x)

    return Adapter()


def _load_model(checkpoint: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    if checkpoint.stat().st_size != CKPT_BYTES or _sha256(checkpoint) != CKPT_SHA:
        raise FastError("compact checkpoint identity mismatch")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FastError("weights-only checkpoint load failed") from exc
    required = {
        "completed_steps", "completed_windows", "elapsed_seconds", "execution", "identity",
        "loss_components", "model", "optimizer", "scaler", "source_groups", "source_records",
        "source_video_ids", "status",
    }
    if not isinstance(state, dict) or set(state) != required or state.get("status") != "completed":
        raise FastError("compact checkpoint completion/schema mismatch")
    identity, model_state = state.get("identity"), state.get("model")
    if not isinstance(identity, dict) or identity.get("scenario") != "universal" or not isinstance(model_state, dict):
        raise FastError("compact checkpoint is not the universal model")
    model = _build_model(torch)
    if tuple(model.state_dict()) != tuple(model_state):
        raise FastError("compact checkpoint model tensor schema mismatch")
    model.load_state_dict(model_state, strict=True)
    model.to(torch.device(device)).eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return torch, model, {
        "checkpoint_sha256": CKPT_SHA,
        "fold": int(identity["fold"]),
        "completed_steps": int(state["completed_steps"]),
        "model_tensors": len(model_state),
        "model_loads": 1,
    }


def _forward(torch: Any, model: Any, pairs: Any, device: str, amp: bool) -> Any:
    pairs = pairs.to(torch.device(device), non_blocking=device.startswith("cuda"))
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp)
    with torch.inference_mode(), context:
        output = model(pairs)
    if tuple(output.shape[:2]) != (len(pairs), 2) or output.shape[2] != 6 or not bool(torch.isfinite(output).all()):
        raise FastError("model output schema or finiteness invariant failed")
    return output


def _peak_coords(torch: Any, output: Any, kernel: Sequence[int], threshold: float) -> Any:
    logits = output[0:1].unsqueeze(0).float()
    pooled = torch.nn.functional.max_pool3d(logits, tuple(kernel), stride=1, padding=tuple(v // 2 for v in kernel))
    return torch.nonzero((logits == pooled) & (torch.sigmoid(logits) >= threshold))[:, 2:]


def _parity_check(torch: Any, model: Any, sample: Any, device: str, kernel: Sequence[int], threshold: float) -> dict[str, Any]:
    started = time.perf_counter()
    first = _forward(torch, model, sample, device, False)
    second = _forward(torch, model, sample, device, False)
    deterministic = bool(torch.equal(first, second))
    mixed = _forward(torch, model, sample, device, True)
    left, right = first.float(), mixed.float()
    center_delta = (left[:, :, 0] - right[:, :, 0]).abs()
    fp_coords = _peak_coords(torch, left[0, 0], kernel, threshold)
    mp_coords = _peak_coords(torch, right[0, 0], kernel, threshold)
    coordinate_equal = bool(torch.equal(fp_coords, mp_coords))
    report = {
        "deterministic_fp32": deterministic,
        "peak_coordinates_equal": coordinate_equal,
        "center_max_abs": float(center_delta.max().item()),
        "center_mean_abs": float(center_delta.mean().item()),
        "seconds": time.perf_counter() - started,
    }
    report["passed"] = bool(
        deterministic and coordinate_equal and report["center_max_abs"] <= 0.05
        and report["center_mean_abs"] <= 0.005
    )
    return report


def _extract_nodes(
    torch: Any,
    np: Any,
    output: Any,
    frame: int,
    next_id: int,
    kernel: Sequence[int],
    threshold: float,
    max_points: int,
) -> tuple[list[dict[str, Any]], int]:
    coords = _peak_coords(torch, output, kernel, threshold)
    logits = output[0].float()
    fallback = not len(coords)
    if len(coords):
        scores = torch.sigmoid(logits[coords[:, 0], coords[:, 1], coords[:, 2]]).detach().cpu().numpy()
        points = coords.detach().cpu().numpy()
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], -scores))[:max_points]
        points, scores = points[order], scores[order]
    else:
        flat_index = int(torch.argmax(logits.reshape(-1)).item())
        point = np.asarray(np.unravel_index(flat_index, tuple(logits.shape)), dtype=np.int64)
        points = point.reshape(1, 3)
        coordinate = tuple(int(value) for value in point)
        scores = np.asarray([float(torch.sigmoid(logits[coordinate]).item())], dtype=np.float32)
    nodes: list[dict[str, Any]] = []
    for point, score in zip(points, scores, strict=True):
        z, y, x = (int(value) for value in point)
        backward = [float(value) for value in output[1:4, z, y, x].float().detach().cpu().tolist()]
        if not all(math.isfinite(value) for value in backward):
            raise FastError("nonfinite motion head output")
        nodes.append({
            "node_id": next_id, "t": frame,
            "z": z * DOWN[0], "y": y * DOWN[1], "x": x * DOWN[2],
            "score": float(score), "backward_um": backward, "fallback": fallback,
        })
        next_id += 1
    return nodes, next_id


def _write_nodes(writer: Any, dataset: str, nodes: Sequence[Mapping[str, Any]], row_id: int) -> int:
    for node in nodes:
        writer.writerow({
            "id": row_id, "dataset": dataset, "row_type": "node", "node_id": node["node_id"],
            "t": node["t"], "z": node["z"], "y": node["y"], "x": node["x"],
            "source_id": -1, "target_id": -1,
        })
        row_id += 1
    return row_id


def _write_edges(writer: Any, dataset: str, edges: Sequence[tuple[int, int]], row_id: int) -> int:
    for source, target in edges:
        writer.writerow({
            "id": row_id, "dataset": dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source, "target_id": target,
        })
        row_id += 1
    return row_id


def _open_image(job: Mapping[str, Any], max_frames: int | None) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import zarr

    group = zarr.open_group(str(job["path"]), mode="r")
    array = group["0"]
    shape = tuple(int(value) for value in array.shape)
    chunks = tuple(int(value) for value in array.chunks)
    if shape != tuple(job["shape"]):
        raise FastError(f"registered shape mismatch for {job['identity']}: {shape}")
    expected_chunks = (1, *shape[1:])
    if chunks != expected_chunks:
        raise FastError(
            f"Zarr chunks must be exactly one frame by one spatial volume for {job['identity']}: "
            f"{chunks} != {expected_chunks}"
        )
    available_chunks = _chunk_count(shape, chunks)
    if available_chunks != int(job["chunks"]):
        raise FastError(f"registered chunk mismatch for {job['identity']}: {available_chunks}")
    frames = shape[0] if max_frames is None else min(shape[0], max_frames)
    if frames < 2:
        raise FastError("inference requires at least two frames")
    attrs = dict(group.attrs)
    low, high = _quantiles(attrs)
    metadata = {
        "np": np, "frames": frames, "scale": _parse_scale(attrs), "low": low, "high": high,
        "available_chunks": available_chunks, "frame_chunk_geometry": True,
        "dtype_bytes": int(np.dtype(array.dtype).itemsize),
    }
    return array, attrs, metadata


def _frame(array: Any, np: Any, index: int, low: float, high: float) -> Any:
    raw = np.asarray(array[index, ::DOWN[0], ::DOWN[1], ::DOWN[2]], dtype=np.float32)
    value = np.clip((raw - low) / (high - low), 0.0, 1.0)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise FastError("native Zarr frame read produced invalid data")
    return value


def _mock_nodes(
    np: Any, value: Any, frame: int, next_id: int, max_points: int, threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], int]:
    flat = value.ravel()
    eligible = np.flatnonzero(flat >= threshold)
    fallback = not len(eligible)
    if fallback:
        indices = np.asarray([int(np.argmax(flat))])
    else:
        order = np.lexsort((eligible, -flat[eligible]))
        indices = eligible[order[:min(max_points, 8, len(eligible))]]
    points = np.column_stack(np.unravel_index(indices, value.shape))
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    nodes = []
    for z, y, x in points:
        nodes.append({"node_id": next_id, "t": frame, "z": int(z), "y": int(y) * 4,
                      "x": int(x) * 4, "score": float(value[z, y, x]),
                      "backward_um": [0.0, 0.0, 0.0], "fallback": fallback})
        next_id += 1
    return nodes, next_id


def _require_frame_nodes(nodes: Sequence[Mapping[str, Any]], identity: str, frame: int) -> None:
    if not nodes:
        raise FastError(f"processed frame emitted no nodes: {identity} frame {frame}")


def _count_frame_fallback(nodes: Sequence[Mapping[str, Any]]) -> int:
    count = sum(bool(node.get("fallback")) for node in nodes)
    if count and (count != 1 or len(nodes) != 1):
        raise FastError("fallback extraction must emit exactly one node")
    return int(count == 1)


def _dataset_run(
    job: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    total_start = time.perf_counter()
    deadline_epoch = float(config["deadline_epoch"])
    _check_deadline(deadline_epoch, f"{job['identity']} startup")
    stages = {name: 0.0 for name in ("zarr_open", "read", "transfer_infer", "nms", "link", "write", "parity")}
    opened = time.perf_counter()
    array, _attrs, metadata = _open_image(job, config.get("max_frames"))
    _check_deadline(deadline_epoch, f"{job['identity']} Zarr open")
    stages["zarr_open"] = time.perf_counter() - opened
    np, frame_count = metadata["np"], int(metadata["frames"])
    label = str(job["dataset"]) if config["mode"] == "submit" else str(job["identity"])
    fragment = output_dir / "frags" / f"{job['split']}__{job['dataset']}.csv"
    if fragment.exists():
        raise FastError(f"refusing to overwrite fragment: {fragment}")
    temporary = fragment.with_suffix(".tmp")
    node_total = edge_total = candidate_total = max_pair_candidates = chunks_read = fallback_frames = 0
    next_id = 1
    previous: list[dict[str, Any]] = []
    row_id = 0
    write_started = time.perf_counter()
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        stages["write"] += time.perf_counter() - write_started

        if config["mock"]:
            for frame_index in range(frame_count):
                _check_deadline(deadline_epoch, f"{job['identity']} frame {frame_index} read")
                read_started = time.perf_counter()
                value = _frame(array, np, frame_index, metadata["low"], metadata["high"])
                stages["read"] += time.perf_counter() - read_started
                chunks_read += 1
                _check_deadline(deadline_epoch, f"{job['identity']} frame {frame_index} read")
                nms_started = time.perf_counter()
                current, next_id = _mock_nodes(
                    np, value, frame_index, next_id, int(config["max_points"]), float(config["det_threshold"]),
                )
                stages["nms"] += time.perf_counter() - nms_started
                _require_frame_nodes(current, str(job["identity"]), frame_index)
                fallback_frames += _count_frame_fallback(current)
                link_started = time.perf_counter()
                edges, candidate_count = sparse_link(previous, current, metadata["scale"], float(config["edge_radius"]),
                                                     int(config["neighbor_cap"]), float(config["edge_threshold"]),
                                                     int(config["max_candidates"])) if previous else ([], 0)
                stages["link"] += time.perf_counter() - link_started
                write_started = time.perf_counter()
                row_id = _write_nodes(writer, label, current, row_id)
                row_id = _write_edges(writer, label, edges, row_id)
                stages["write"] += time.perf_counter() - write_started
                node_total += len(current); edge_total += len(edges); candidate_total += candidate_count
                max_pair_candidates = max(max_pair_candidates, candidate_count)
                previous = current
                _check_deadline(deadline_epoch, f"{job['identity']} frame {frame_index} finalization")
        else:
            torch, model = runtime["torch"], runtime["model"]
            device, batch_size = str(config["device"]), int(config["batch"])
            voxel_um = tuple(metadata["scale"][i] * DOWN[i] for i in range(3))
            kernel = _pool_kernel(float(config["nms_um"]), voxel_um)
            pending = None
            cached = None
            for start in range(0, frame_count - 1, batch_size):
                _check_deadline(deadline_epoch, f"{job['identity']} batch {start} read")
                stop = min(frame_count - 1, start + batch_size)
                values = []
                if cached is None:
                    read_started = time.perf_counter()
                    cached = _frame(array, np, start, metadata["low"], metadata["high"])
                    stages["read"] += time.perf_counter() - read_started
                    chunks_read += 1
                    _check_deadline(deadline_epoch, f"{job['identity']} frame {start} read")
                values.append(cached)
                for frame_index in range(start + 1, stop + 1):
                    read_started = time.perf_counter()
                    values.append(_frame(array, np, frame_index, metadata["low"], metadata["high"]))
                    stages["read"] += time.perf_counter() - read_started
                    chunks_read += 1
                    _check_deadline(deadline_epoch, f"{job['identity']} frame {frame_index} read")
                cached = values[-1]
                pairs_np = np.stack([np.stack((values[i], values[i + 1])) for i in range(len(values) - 1)])
                pairs = torch.from_numpy(pairs_np).unsqueeze(2)
                if device.startswith("cuda"):
                    pairs = pairs.pin_memory()
                _check_deadline(deadline_epoch, f"{job['identity']} batch {start} inference")
                precision = runtime["precision"]
                if not precision["checked"] and config["precision"] != "fp32":
                    parity = _parity_check(torch, model, pairs[:1], device, kernel, float(config["det_threshold"]))
                    stages["parity"] += float(parity["seconds"])
                    precision.update({"checked": True, "report": parity})
                    if config["precision"] == "fp16" and not parity["passed"]:
                        raise FastError(f"mixed precision parity failed: {parity}")
                    precision["amp"] = bool(parity["passed"])
                    _check_deadline(deadline_epoch, f"{job['identity']} precision parity")
                infer_started = time.perf_counter()
                outputs = _forward(torch, model, pairs, device, bool(precision["amp"]))
                stages["transfer_infer"] += time.perf_counter() - infer_started
                _check_deadline(deadline_epoch, f"{job['identity']} batch {start} inference")
                for offset in range(len(outputs)):
                    absolute = start + offset
                    first_output, second_output = outputs[offset, 0], outputs[offset, 1]
                    ready: list[tuple[int, Any]] = []
                    if absolute == 0:
                        ready.append((0, first_output))
                    else:
                        if pending is None:
                            raise FastError("sliding output coverage lost an interior frame")
                        ready.append((absolute, (pending.float() + first_output.float()) * 0.5))
                    pending = second_output
                    for frame_index, output in ready:
                        nms_started = time.perf_counter()
                        current, next_id = _extract_nodes(torch, np, output, frame_index, next_id, kernel,
                                                          float(config["det_threshold"]), int(config["max_points"]))
                        stages["nms"] += time.perf_counter() - nms_started
                        _require_frame_nodes(current, str(job["identity"]), frame_index)
                        fallback_frames += _count_frame_fallback(current)
                        link_started = time.perf_counter()
                        edges, candidate_count = sparse_link(previous, current, metadata["scale"], float(config["edge_radius"]),
                                                             int(config["neighbor_cap"]), float(config["edge_threshold"]),
                                                             int(config["max_candidates"])) if previous else ([], 0)
                        stages["link"] += time.perf_counter() - link_started
                        write_started = time.perf_counter()
                        row_id = _write_nodes(writer, label, current, row_id)
                        row_id = _write_edges(writer, label, edges, row_id)
                        stages["write"] += time.perf_counter() - write_started
                        node_total += len(current); edge_total += len(edges); candidate_total += candidate_count
                        max_pair_candidates = max(max_pair_candidates, candidate_count)
                        previous = current
                        _check_deadline(deadline_epoch, f"{job['identity']} frame {frame_index} finalization")
                del outputs, pairs, pairs_np
            if pending is None:
                raise FastError("sliding output did not cover the last frame")
            nms_started = time.perf_counter()
            current, next_id = _extract_nodes(torch, np, pending, frame_count - 1, next_id, kernel,
                                              float(config["det_threshold"]), int(config["max_points"]))
            stages["nms"] += time.perf_counter() - nms_started
            _require_frame_nodes(current, str(job["identity"]), frame_count - 1)
            fallback_frames += _count_frame_fallback(current)
            link_started = time.perf_counter()
            edges, candidate_count = sparse_link(previous, current, metadata["scale"], float(config["edge_radius"]),
                                                 int(config["neighbor_cap"]), float(config["edge_threshold"]),
                                                 int(config["max_candidates"])) if previous else ([], 0)
            stages["link"] += time.perf_counter() - link_started
            write_started = time.perf_counter()
            row_id = _write_nodes(writer, label, current, row_id)
            row_id = _write_edges(writer, label, edges, row_id)
            stages["write"] += time.perf_counter() - write_started
            node_total += len(current); edge_total += len(edges); candidate_total += candidate_count
            max_pair_candidates = max(max_pair_candidates, candidate_count)

        _check_deadline(deadline_epoch, f"{job['identity']} fragment write")
        write_started = time.perf_counter()
        handle.flush()
        os.fsync(handle.fileno())
        stages["write"] += time.perf_counter() - write_started
    _check_deadline(deadline_epoch, f"{job['identity']} fragment finalization")
    os.replace(temporary, fragment)
    if chunks_read != frame_count:
        raise FastError(f"native Zarr logical chunk coverage mismatch: {chunks_read} != {frame_count}")
    if config.get("max_frames") is None and chunks_read != int(job["chunks"]):
        raise FastError(f"full image chunk coverage mismatch for {job['identity']}")
    if max_pair_candidates > int(config["max_candidates"]):
        raise FastError("candidate memory bound was exceeded")
    if not 0 <= fallback_frames <= frame_count:
        raise FastError("fallback frame accounting is invalid")
    return {
        "identity": job["identity"], "csv_dataset": label, "fragment": str(fragment), "frames": frame_count,
        "available_chunks": metadata["available_chunks"], "chunks_read": chunks_read,
        "frame_chunk_geometry": metadata["frame_chunk_geometry"],
        "nodes": node_total, "edges": edge_total, "candidate_edges": candidate_total,
        "fallback_frames": fallback_frames,
        "max_pair_candidates": max_pair_candidates, "dense_nxn": False,
        "stages_seconds": stages, "total_seconds": time.perf_counter() - total_start,
    }


def _worker(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    config = job["config"]
    output_dir = Path(job["output_dir"])
    status_path = Path(job["status"])
    timing_path = Path(job["timing"])
    runtime: dict[str, Any] = {"precision": {"checked": config["precision"] == "fp32", "amp": False, "report": None}}
    audit = {"blocked": 0}
    try:
        deadline_epoch = float(config["deadline_epoch"])
        _check_deadline(deadline_epoch, "worker startup")
        if config["mode"] == "benchmark":
            # Finish dependency imports before the audit, then enforce the
            # policy for the checkpoint and every dataset read.
            import numpy  # noqa: F401
            import zarr  # noqa: F401
            if not config["mock"]:
                import torch  # noqa: F401
            audit = _install_read_audit()
        if not config["mock"]:
            torch, model, restore = _load_model(Path(config["checkpoint"]), str(config["device"]))
            _check_deadline(deadline_epoch, "model loading")
            runtime.update({"torch": torch, "model": model, "restore": restore})
        else:
            runtime["restore"] = {"mock": True, "model_loads": 1}
        results = []
        with timing_path.open("a", encoding="ascii") as timing:
            for dataset in job["jobs"]:
                if time.time() >= float(config["deadline_epoch"]):
                    raise FastError("two-hour runtime deadline reached before image coverage completed")
                result = _dataset_run(dataset, config, output_dir, runtime)
                result["worker"] = int(job["rank"])
                timing.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                timing.flush()
                results.append(result)
                print(json.dumps({"done": result["identity"], "gpu": job["rank"], "seconds": result["total_seconds"]}), flush=True)
        _check_deadline(deadline_epoch, "worker status finalization")
        _atomic_json(status_path, {
            "ok": True, "rank": job["rank"], "device": config["device"], "datasets": results,
            "restore": runtime["restore"], "precision": runtime["precision"], "denied_reads": audit["blocked"],
        }, deadline_epoch)
        return 0
    except Exception as exc:
        _atomic_json(status_path, {
            "ok": False, "rank": job.get("rank"), "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-8000:], "denied_reads": audit["blocked"],
        })
        return 1


def _merge(
    output: Path,
    jobs: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
    deadline_epoch: float | None = None,
) -> dict[str, Any]:
    by_identity: dict[str, Mapping[str, Any]] = {}
    for status in statuses:
        for result in status["datasets"]:
            identity = str(result["identity"])
            if identity in by_identity:
                raise FastError(f"duplicate completed dataset: {identity}")
            by_identity[identity] = result
    expected = [str(job["identity"]) for job in jobs]
    if set(by_identity) != set(expected) or len(by_identity) != len(expected):
        raise FastError("completed output coverage does not match registered jobs")
    row_id = node_total = edge_total = 0
    if output.exists():
        raise FastError(f"refusing to overwrite merged CSV: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for identity in expected:
                if deadline_epoch is not None:
                    _check_deadline(deadline_epoch, f"{identity} CSV merge")
                result = by_identity[identity]
                expected_label = str(result["csv_dataset"])
                nodes: set[int] = set()
                node_frames: set[int] = set()
                edges: list[tuple[int, int]] = []
                dataset_rows = 0
                with Path(result["fragment"]).open(newline="", encoding="utf-8") as source:
                    reader = csv.DictReader(source)
                    if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                        raise FastError(f"fragment schema mismatch: {identity}")
                    for row in reader:
                        if deadline_epoch is not None:
                            _check_deadline(deadline_epoch, f"{identity} CSV merge")
                        if row["dataset"] != expected_label:
                            raise FastError(f"dataset label mismatch in {identity}")
                        dataset_rows += 1
                        row["id"] = str(row_id)
                        writer.writerow(row)
                        row_id += 1
                        if row["row_type"] == "node":
                            node_id = int(row["node_id"])
                            if node_id in nodes:
                                raise FastError(f"duplicate node ID in {identity}")
                            nodes.add(node_id); node_frames.add(int(row["t"])); node_total += 1
                        elif row["row_type"] == "edge":
                            edges.append((int(row["source_id"]), int(row["target_id"]))); edge_total += 1
                        else:
                            raise FastError(f"unknown row type in {identity}")
                if not dataset_rows:
                    raise FastError(f"dataset is absent from merged CSV: {identity}")
                if node_frames != set(range(int(result["frames"]))):
                    raise FastError(f"one-or-more processed frames have no CSV nodes: {identity}")
                if any(source not in nodes or target_id not in nodes for source, target_id in edges):
                    raise FastError(f"dangling sparse edge in {identity}")
                if len(nodes) != int(result["nodes"]) or len(edges) != int(result["edges"]):
                    raise FastError(f"fragment accounting mismatch in {identity}")
            target.flush(); os.fsync(target.fileno())
        if deadline_epoch is not None:
            _check_deadline(deadline_epoch, "merged CSV finalization")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"rows": row_id, "nodes": node_total, "edges": edge_total, "datasets": len(by_identity)}


def _terminate_workers(processes: Sequence[Any]) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        process.terminate()
    stop = time.monotonic() + 5.0
    for process in active:
        try:
            process.wait(timeout=max(0.0, stop - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
    for process in active:
        if process.poll() is None:
            process.wait()


def _wait_workers(processes: Sequence[Any], deadline_epoch: float) -> list[int]:
    while True:
        codes = [process.poll() for process in processes]
        if all(code is not None for code in codes):
            return [int(code) for code in codes]
        if any(code not in (None, 0) for code in codes):
            _terminate_workers(processes)
            return [int(process.wait()) for process in processes]
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            _terminate_workers(processes)
            raise FastError("runtime deadline reached while waiting for workers")
        time.sleep(min(0.05, remaining))


def _parse_devices(value: str, dry_run: bool) -> list[str]:
    if value.strip().casefold() == "cpu":
        if not dry_run:
            raise FastError("CPU is dry-run only")
        return ["cpu"]
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise FastError("--gpus must be 'cpu' or comma-separated CUDA indices")
    if len(values) != 2 and not dry_run:
        raise FastError("production runtime requires exactly two GPU shards")
    return [f"cuda:{item}" for item in values]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("benchmark", "submit"), default="submit")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/fastout"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--nms-um", type=float, default=3.0)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--edge-radius", type=float, default=8.0)
    parser.add_argument("--neighbor-cap", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--max-candidates", type=int, default=4096)
    parser.add_argument("--deadline-min", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-file")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.root is None:
        raise FastError("--root is required")
    if not args.mock and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise FastError("--checkpoint is required and must exist")
    if args.mock and not args.dry_run:
        raise FastError("--mock is dry-run only")
    if args.max_frames is not None and (not args.dry_run or args.max_frames < 2):
        raise FastError("--max-frames >=2 is dry-run only")
    if args.max_datasets is not None and args.max_datasets < 1:
        raise FastError("--max-datasets must be positive")
    if not 1 <= args.batch <= 16 or not 1 <= args.max_points <= 4096:
        raise FastError("batch or point cap is outside bounded range")
    if not 1 <= args.neighbor_cap <= 16 or not args.max_points * args.neighbor_cap <= args.max_candidates <= 65536:
        raise FastError("candidate cap must cover max_points*neighbor_cap and remain <=65536")
    if not 0 < args.edge_threshold < 1 or not 0 < args.det_threshold < 1:
        raise FastError("probability thresholds must be in (0,1)")
    if args.deadline_min <= 0 or (not args.dry_run and args.deadline_min > 120):
        raise FastError("production deadline must be in (0,120] minutes")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_job is not None:
        return _worker(args.worker_job)
    _validate_args(args)
    started_epoch, started = time.time(), time.perf_counter()
    deadline_epoch = started_epoch + args.deadline_min * 60
    registry = load_registry(args.registry)
    jobs = discover_jobs(args.root.resolve(), registry, args.mode, args.dry_run, args.max_datasets)
    devices = _parse_devices(args.gpus, args.dry_run)
    shards = shard_jobs(jobs, len(devices))
    _check_deadline(deadline_epoch, "inference setup")
    output_dir = args.out.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FastError(f"refusing to overwrite nonempty output directory: {output_dir}")
    (output_dir / "frags").mkdir(parents=True, exist_ok=True)
    output_name = args.output_file or ("submission.csv" if args.mode == "submit" else "benchmark.csv")
    if Path(output_name).name != output_name or not output_name.endswith(".csv"):
        raise FastError("--output-file must be one CSV basename")
    config_base = {
        "mode": args.mode, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "precision": args.precision, "batch": args.batch, "det_threshold": args.det_threshold,
        "nms_um": args.nms_um, "max_points": args.max_points, "edge_radius": args.edge_radius,
        "neighbor_cap": args.neighbor_cap, "edge_threshold": args.edge_threshold,
        "max_candidates": args.max_candidates, "max_frames": args.max_frames,
        "deadline_epoch": deadline_epoch, "mock": args.mock,
    }
    entry = Path(sys.argv[0]).resolve()
    processes: list[tuple[subprocess.Popen[Any], Any]] = []
    status_paths: list[Path] = []
    try:
        for rank, (device, shard) in enumerate(zip(devices, shards, strict=True)):
            _check_deadline(deadline_epoch, "worker launch")
            status = output_dir / f"worker{rank}.json"
            timing = output_dir / f"gpu{rank}.jsonl"
            job_path = output_dir / f"job{rank}.json"
            _atomic_json(job_path, {
                "rank": rank, "jobs": shard, "output_dir": str(output_dir), "status": str(status),
                "timing": str(timing), "config": {**config_base, "device": device},
            }, deadline_epoch)
            command = [sys.executable, "-B", str(entry), "--worker-job", str(job_path)]
            process = subprocess.Popen(command, cwd=Path.cwd())
            processes.append((process, job_path)); status_paths.append(status)
        return_codes = _wait_workers([process for process, _job in processes], deadline_epoch)
    except Exception:
        _terminate_workers([process for process, _job in processes])
        (output_dir / "stats.json").unlink(missing_ok=True)
        raise
    _check_deadline(deadline_epoch, "worker status collection")
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_paths if path.is_file()]
    if len(statuses) != len(processes) or any(code != 0 for code in return_codes) or any(not row.get("ok") for row in statuses):
        failures = [row for row in statuses if not row.get("ok")]
        raise FastError(f"GPU worker failure: {failures or return_codes}")
    _check_deadline(deadline_epoch, "CSV merge")
    merged = _merge(output_dir / output_name, jobs, statuses, deadline_epoch)
    _check_deadline(deadline_epoch, "coverage finalization")
    elapsed = time.perf_counter() - started
    results = [result for status in statuses for result in status["datasets"]]
    expected_chunks = sum(int(job["chunks"]) for job in jobs)
    available_chunks = sum(int(result["available_chunks"]) for result in results)
    full_frames = args.max_frames is None
    read_chunks = sum(int(result["chunks_read"]) for result in results)
    fallback_frames = sum(int(result["fallback_frames"]) for result in results)
    fallback_datasets = sum(int(result["fallback_frames"]) > 0 for result in results)
    chunk_geometry = all(bool(result.get("frame_chunk_geometry")) for result in results)
    if (not chunk_geometry or available_chunks != expected_chunks
            or (full_frames and read_chunks != expected_chunks)):
        raise FastError("final registered/native Zarr chunk assertion failed")
    if args.mode == "benchmark" and not args.dry_run and merged["datasets"] != EXPECTED_ALL:
        raise FastError("final 203-dataset benchmark coverage assertion failed")
    if args.mode == "benchmark" and any(int(status["denied_reads"]) != 0 for status in statuses):
        raise FastError("a forbidden benchmark read was attempted")
    summary = {
        "artifact": "g02_fast_runtime", "schema_version": 1, "mode": args.mode,
        "dry_run": args.dry_run, "mock": args.mock, "devices": devices,
        "coverage": {"expected": len(jobs), "completed": merged["datasets"],
                     "registered_chunks": expected_chunks, "available_chunks": available_chunks,
                     "logical_chunks_read": read_chunks, "frame_chunk_geometry": chunk_geometry,
                     "fallback_frames": fallback_frames, "datasets_with_fallback": fallback_datasets,
                     "asserted": True},
        "memory_contract": {"dense_nxn": False, "max_points_per_frame": args.max_points,
                            "neighbor_cap": args.neighbor_cap, "max_candidates_per_pair": args.max_candidates},
        "runtime": {"elapsed_seconds": elapsed, "deadline_seconds": args.deadline_min * 60,
                    "under_deadline": elapsed <= args.deadline_min * 60,
                    "models_persistent": True, "model_loads": sum(int(row["restore"]["model_loads"]) for row in statuses),
                    "batch_pairs": args.batch},
        "image_only": {"geff": False, "annotations": False, "scorer": False, "split_manifest": False,
                       "denied_read_attempts": sum(int(row["denied_reads"]) for row in statuses)},
        "output": {"path": str(output_dir / output_name), **merged},
        "workers": statuses,
    }
    _check_deadline(deadline_epoch, "statistics finalization")
    _atomic_json(output_dir / "stats.json", summary, deadline_epoch)
    print(json.dumps({"output": summary["output"], "coverage": summary["coverage"], "runtime": summary["runtime"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
