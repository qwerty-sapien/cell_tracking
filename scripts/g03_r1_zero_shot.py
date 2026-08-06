#!/usr/bin/env python3
"""Seal, emit, and score the G-03 R1 untrained crop-adapter baseline.

The ``infer`` command never reads GEFF/ground truth.  It creates all 36 frozen
44b6 fragments and seals their hashes.  The separate ``score`` command refuses
to run until that inference seal is complete and hash-verified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from typing import Any
import zipfile

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import yaml
import zarr
from zarr.storage import ZipStore


SCHEMA = "celltrack-alpha.g03-r1-zero-shot.v1"
CONTRACT_SCHEMA = "celltrack-alpha.g03-r1-zero-shot-contract.v1"
MEMBER_MAP_SCHEMA = "celltrack-alpha.g03-r1-44b6-member-map.v1"
ADAPTER_SCHEMA = "celltrack-alpha.g03-r1-untrained-crop-adapter.v1"
PREFLIGHT_SCHEMA = "celltrack-alpha.g03-r1-preflight.v1"
VIDEO_SCHEMA = "celltrack-alpha.g03-r1-zero-shot-video.v1"
SEAL_SCHEMA = "celltrack-alpha.g03-r1-inference-seal.v1"
SCORE_SCHEMA = "celltrack-alpha.g03-r1-score.v1"
BANK_SCHEMA = "celltrack-alpha.g03-r1-association-bank.v1"


class R1Error(RuntimeError):
    """The sealed one-shot R1 measurement failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise R1Error(f"refusing to overwrite artifact: {path}")
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
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R1Error(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise R1Error(f"{label} must be a JSON object")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise R1Error(f"{label} SHA-256 mismatch: {path}")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def configure_runtime(contract: Mapping[str, Any]) -> None:
    runtime = contract["runtime"]
    torch.set_num_threads(int(runtime["torch_num_threads"]))
    torch.set_num_interop_threads(int(runtime["torch_num_interop_threads"]))
    torch.manual_seed(int(contract["adapter"]["seed"]))
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != runtime["torch_version"]:
        raise R1Error("PyTorch version drifted")
    if np.__version__ != runtime["numpy_version"]:
        raise R1Error("NumPy version drifted")
    if zarr.__version__ != runtime["zarr_version"]:
        raise R1Error("Zarr version drifted")


class CropAdapter(nn.Module):
    """Tiny untrained 3-D CNN over a three-frame intensity crop."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_pool = nn.AvgPool3d(
            kernel_size=(4, 8, 8), stride=(4, 8, 8)
        )
        self.conv = nn.Conv3d(3, 8, kernel_size=3, padding=1, bias=True)
        self.activation = nn.GELU(approximate="none")
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(8, 12, bias=True)
        self.output_norm = nn.LayerNorm(12, elementwise_affine=False)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.ndim != 5 or tuple(crops.shape[1:]) != (3, 16, 32, 32):
            raise R1Error("adapter crops must have shape [N,3,16,32,32]")
        value = crops.to(dtype=torch.float32)
        mean = value.mean(dim=(1, 2, 3, 4), keepdim=True)
        centered = value - mean
        rms = torch.sqrt(
            centered.square().mean(dim=(1, 2, 3, 4), keepdim=True)
        ).clamp_min(1.0e-6)
        value = torch.clamp(centered / rms, -5.0, 5.0)
        value = self.spatial_pool(value)
        value = self.activation(self.conv(value))
        value = self.global_pool(value).flatten(1)
        value = self.output_norm(self.projection(value))
        if value.shape[1:] != (12,) or not bool(torch.isfinite(value).all()):
            raise R1Error("adapter emitted malformed wrfeat")
        return value


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        array = tensor.numpy()
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


def create_adapter(path: Path, seed: int) -> dict[str, Any]:
    if path.exists():
        raise R1Error(f"refusing to overwrite adapter: {path}")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    adapter = CropAdapter().eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    state = adapter.state_dict()
    payload = {
        "schema": ADAPTER_SCHEMA,
        "seed": seed,
        "architecture": (
            "AvgPool3d(4,8,8)->Conv3d(3,8,k3,p1)->GELU->"
            "AdaptiveAvgPool3d(1)->Linear(8,12)->LayerNorm(no-affine)"
        ),
        "parameter_count": sum(p.numel() for p in adapter.parameters()),
        "state_tensor_sha256": state_tensor_sha256(state),
        "state_dict": state,
        "training_steps": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        key: payload[key]
        for key in (
            "schema",
            "seed",
            "architecture",
            "parameter_count",
            "state_tensor_sha256",
            "training_steps",
        )
    } | {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_adapter(contract: Mapping[str, Any]) -> CropAdapter:
    record = contract["adapter"]
    path = Path(record["path"])
    require_hash(path, record["file_sha256"], "adapter checkpoint")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, dict)
        or value.get("schema") != ADAPTER_SCHEMA
        or int(value.get("seed", -1)) != int(record["seed"])
        or int(value.get("training_steps", -1)) != 0
        or int(value.get("parameter_count", -1)) != int(record["parameter_count"])
        or value.get("state_tensor_sha256") != record["state_tensor_sha256"]
    ):
        raise R1Error("adapter checkpoint contract drifted")
    state = value.get("state_dict")
    if not isinstance(state, Mapping) or state_tensor_sha256(state) != record["state_tensor_sha256"]:
        raise R1Error("adapter state tensor hash drifted")
    adapter = CropAdapter().eval()
    adapter.load_state_dict(state, strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in adapter.parameters()) != int(
        record["parameter_count"]
    ):
        raise R1Error("adapter parameter count drifted")
    return adapter


def create_member_map(
    *, raw_manifest: Path, h1_report: Path, output: Path
) -> dict[str, Any]:
    raw = load_json(raw_manifest, "raw fit105 manifest")
    report = load_json(h1_report, "H1 44b6 report")
    identities = tuple(str(value) for value in report["scope"]["video_ids"])
    if (
        len(identities) != 36
        or len(set(identities)) != 36
        or any(not value.startswith("train:44b6_") for value in identities)
    ):
        raise R1Error("H1 report does not define the exact 36-video population")
    rows = raw.get("files")
    if not isinstance(rows, list):
        raise R1Error("raw manifest has no file list")
    videos = []
    covered: set[str] = set()
    for identity in identities:
        dataset = identity.split(":", 1)[1]
        prefix = f"train/{dataset}.zarr/"
        members = [dict(row) for row in rows if str(row.get("path", "")).startswith(prefix)]
        members.sort(key=lambda row: str(row["path"]))
        if not members or any(set(row) != {"path", "bytes", "sha256"} for row in members):
            raise R1Error(f"raw member coverage is malformed for {identity}")
        covered.add(identity)
        videos.append(
            {
                "identity": identity,
                "dataset": dataset,
                "member_count": len(members),
                "members": members,
            }
        )
    result = {
        "schema": MEMBER_MAP_SCHEMA,
        "scope": {
            "fit_train_44b6_only": True,
            "identity_count": len(identities),
            "identities": list(identities),
            "gt_members_included": False,
            "test_members_included": False,
        },
        "source_raw_manifest": {
            "path": str(raw_manifest),
            "sha256": sha256(raw_manifest),
        },
        "source_h1_report": {
            "path": str(h1_report),
            "sha256": sha256(h1_report),
        },
        "videos": videos,
        "aggregate": {
            "video_count": len(videos),
            "member_count": sum(row["member_count"] for row in videos),
            "bytes": sum(
                int(member["bytes"])
                for row in videos
                for member in row["members"]
            ),
        },
    }
    atomic_json(output, result)
    return result


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_hash(path, expected_sha256, "R1 contract")
    value = load_json(path, "R1 contract")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "SEALED_BEFORE_HELD_FIT_SCORE_READ"
        or value.get("scope", {}).get("identity_count") != 36
        or value.get("scope", {}).get("family") != "44b6"
        or value.get("scope", {}).get("training") is not False
        or value.get("scope", {}).get("gpu") is not False
        or value.get("scope", {}).get("kaggle") is not False
        or value.get("scope", {}).get("cv57") is not False
        or value.get("scope", {}).get("submission") is not False
        or value.get("selection", {}).get("alternatives") != 1
        or value.get("selection", {}).get("post_result_tuning") is not False
        or value.get("compiler", {}).get("divisions") is not False
    ):
        raise R1Error("R1 contract is not the sealed one-shot design")
    require_hash(Path(__file__).resolve(), value["implementation_sha256"], "R1 implementation")
    return value


def verify_cache_population(contract: Mapping[str, Any]) -> dict[str, Any]:
    record = contract["inputs"]["cache_manifest"]
    path = Path(record["path"])
    require_hash(path, record["sha256"], "fit105 cache manifest")
    manifest = load_json(path, "fit105 cache manifest")
    identities = tuple(contract["scope"]["identities"])
    files = [
        dict(row)
        for row in manifest.get("files", [])
        if str(row.get("identity", "")) in identities
    ]
    if len(files) != 72 or {str(row["identity"]) for row in files} != set(identities):
        raise R1Error("44b6 cache manifest coverage drifted")
    for row in files:
        candidate = Path(contract["inputs"]["repository_root"]) / str(row["path"])
        if candidate.stat().st_size != int(row["bytes"]):
            raise R1Error(f"cache byte count drifted: {candidate}")
        require_hash(candidate, str(row["sha256"]), str(row["identity"]))
    return {"identity_count": len(identities), "file_count": len(files)}


def verify_member_map(contract: Mapping[str, Any]) -> dict[str, Any]:
    record = contract["inputs"]["member_map"]
    path = Path(record["path"])
    require_hash(path, record["sha256"], "44b6 raw member map")
    value = load_json(path, "44b6 raw member map")
    identities = tuple(contract["scope"]["identities"])
    if (
        value.get("schema") != MEMBER_MAP_SCHEMA
        or tuple(value.get("scope", {}).get("identities", [])) != identities
        or value.get("scope", {}).get("gt_members_included") is not False
        or value.get("scope", {}).get("test_members_included") is not False
        or int(value.get("aggregate", {}).get("video_count", 0)) != 36
    ):
        raise R1Error("raw member map scope drifted")
    raw_rows = {
        str(member["path"]): member
        for video in value["videos"]
        for member in video["members"]
    }
    archive = Path(contract["inputs"]["raw_archive"]["path"])
    with zipfile.ZipFile(archive) as handle:
        infos = {info.filename: info for info in handle.infolist()}
    if not set(raw_rows) <= set(infos):
        raise R1Error("archive is missing sealed raw members")
    for name, row in raw_rows.items():
        if infos[name].file_size != int(row["bytes"]):
            raise R1Error(f"raw archive member size drifted: {name}")
    return {
        "identity_count": 36,
        "member_count": len(raw_rows),
        "bytes": sum(int(row["bytes"]) for row in raw_rows.values()),
    }


def verify_scorer(contract: Mapping[str, Any]) -> dict[str, str]:
    record = contract["inputs"]["scorer_parity"]
    parity_path = Path(record["path"])
    require_hash(parity_path, record["sha256"], "scorer parity receipt")
    parity = load_json(parity_path, "scorer parity receipt")
    expected = parity.get("source_hashes")
    if not isinstance(expected, Mapping):
        raise R1Error("scorer parity receipt has no source hashes")
    root = Path(contract["inputs"]["repository_root"])
    observed = {}
    for relative, wanted in contract["inputs"]["scorer_sources"].items():
        path = root / relative
        require_hash(path, wanted, f"scorer source {relative}")
        if expected.get(relative) != wanted:
            raise R1Error(f"scorer parity does not pin {relative}")
        observed[relative] = wanted
    return observed


def verify_implementation_dependencies(contract: Mapping[str, Any]) -> dict[str, str]:
    root = Path(contract["inputs"]["repository_root"])
    observed = {}
    for relative, wanted in contract["inputs"]["implementation_dependencies"].items():
        path = root / relative
        require_hash(path, wanted, f"implementation dependency {relative}")
        observed[relative] = wanted
    return observed


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    configure_runtime(contract)
    raw = contract["inputs"]["raw_archive"]
    raw_path = Path(raw["path"])
    if raw_path.stat().st_size != int(raw["bytes"]):
        raise R1Error("raw archive byte count drifted")
    require_hash(raw_path, raw["sha256"], "raw competition archive")
    checkpoint = contract["inputs"]["trackastra_checkpoint"]
    require_hash(Path(checkpoint["model_path"]), checkpoint["model_sha256"], "Trackastra model")
    require_hash(Path(checkpoint["archive_path"]), checkpoint["archive_sha256"], "Trackastra archive")
    source_root = Path(contract["inputs"]["trackastra_source_root"])
    if tree_sha256(source_root) != contract["inputs"]["trackastra_source_tree_sha256"]:
        raise R1Error("Trackastra source tree drifted")
    member_evidence = verify_member_map(contract)
    cache_evidence = verify_cache_population(contract)
    scorer_evidence = verify_scorer(contract)
    dependency_evidence = verify_implementation_dependencies(contract)
    adapter = load_adapter(contract)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS_SCORE_BLIND_PREFLIGHT",
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "implementation_sha256": contract["implementation_sha256"],
        "raw_archive": raw,
        "raw_member_map": member_evidence,
        "fit105_cache": cache_evidence,
        "trackastra_checkpoint": checkpoint,
        "trackastra_source_tree_sha256": contract["inputs"]["trackastra_source_tree_sha256"],
        "adapter": {
            "path": contract["adapter"]["path"],
            "file_sha256": contract["adapter"]["file_sha256"],
            "state_tensor_sha256": contract["adapter"]["state_tensor_sha256"],
            "parameter_count": sum(p.numel() for p in adapter.parameters()),
            "training_steps": 0,
        },
        "scorer_sources": scorer_evidence,
        "implementation_dependencies": dependency_evidence,
        "held_fit_score_read": False,
        "gt_access": False,
        "gpu_hours": 0.0,
    }
    atomic_json(args.output, report)
    return report


def install_trackastra(source_root: Path):
    if any(name == "trackastra" or name.startswith("trackastra.") for name in sys.modules):
        raise R1Error("refusing to mix an existing Trackastra import")
    if "matplotlib" not in sys.modules:
        sys.modules["matplotlib"] = types.ModuleType("matplotlib")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from trackastra.model.model import TrackingTransformer

    return TrackingTransformer


def load_trackastra(contract: Mapping[str, Any]):
    checkpoint = contract["inputs"]["trackastra_checkpoint"]
    source_root = Path(contract["inputs"]["trackastra_source_root"])
    TrackingTransformer = install_trackastra(source_root)
    archive_path = Path(checkpoint["archive_path"])
    with zipfile.ZipFile(archive_path) as archive:
        config = yaml.safe_load(archive.read("ctc/config.yaml"))
    if config != contract["model_config"]:
        raise R1Error("Trackastra model configuration drifted")
    with torch.device("meta"):
        model = TrackingTransformer(**config)
    state = torch.load(
        Path(checkpoint["model_path"]), map_location="cpu", mmap=True, weights_only=True
    )
    model.load_state_dict(state, strict=True, assign=True)
    model.to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        checkpoint["parameter_count"]
    ):
        raise R1Error("Trackastra parameter count drifted")
    return model


def configure_repository_imports(contract: Mapping[str, Any]) -> None:
    root = Path(contract["inputs"]["repository_root"])
    for location in (root / "src", root / "scripts"):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))


def cache_identity(meta: Mapping[str, Any], contract: Mapping[str, Any]):
    configure_repository_imports(contract)
    from celltrack_alpha import bh_model

    model = meta["contract"]["model"]
    return bh_model.CacheIdentity(
        str(model["config_sha256"]),
        str(model["source_sha256"]),
        str(model["weight_sha256"]),
        int(model["pair_batch_size"]),
        int(model["view_batch_size"]),
    )


def load_detections(identity: str, contract: Mapping[str, Any]):
    configure_repository_imports(contract)
    from celltrack_alpha import bh_model

    root = Path(contract["inputs"]["repository_root"])
    cache_root = root / contract["inputs"]["cache_root_relative"]
    slug = identity.replace(":", "__", 1)
    meta_path = cache_root / f"{slug}.json"
    meta = load_json(meta_path, f"{identity} cache metadata")
    return bh_model.load_sparse_cache_detections(
        cache_root / f"{slug}.bhc", expected_identity=cache_identity(meta, contract)
    )


def crop_indices(coords: np.ndarray, shape: Sequence[int]) -> tuple[np.ndarray, ...]:
    centers = np.rint(np.asarray(coords, dtype=np.float64)).astype(np.int64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise R1Error("crop coordinates must have shape [N,3]")
    offsets = (
        np.arange(-8, 8, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
        np.arange(-16, 16, dtype=np.int64),
    )
    return tuple(
        np.clip(centers[:, axis, None] + offsets[axis][None, :], 0, int(shape[axis]) - 1)
        for axis in range(3)
    )


def extract_crop_batch(
    frames: Sequence[np.ndarray], coords: np.ndarray
) -> torch.Tensor:
    if len(frames) != 3 or any(frame.ndim != 3 for frame in frames):
        raise R1Error("crop source must contain three Z/Y/X frames")
    shape = tuple(int(value) for value in frames[0].shape)
    if any(tuple(frame.shape) != shape for frame in frames):
        raise R1Error("temporal crop frames have different shapes")
    z_idx, y_idx, x_idx = crop_indices(coords, shape)
    channels = []
    for frame in frames:
        patch = frame[
            z_idx[:, :, None, None],
            y_idx[:, None, :, None],
            x_idx[:, None, None, :],
        ]
        channels.append(np.asarray(patch, dtype=np.float32))
    stacked = np.stack(channels, axis=1)
    if stacked.shape != (len(coords), 3, 16, 32, 32) or not np.isfinite(stacked).all():
        raise R1Error("extracted crops are malformed")
    return torch.from_numpy(stacked)


def embed_video(
    *,
    image: Any,
    detections: Sequence[Any],
    adapter: CropAdapter,
    batch_size: int,
) -> tuple[dict[int, tuple[int, np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    frame_rows = {
        int(frame.frame_index): (
            np.asarray(frame.node_ids, dtype=np.int64),
            np.asarray(frame.coords_zyx, dtype=np.float32),
        )
        for frame in detections
    }
    if not frame_rows:
        raise R1Error("video has no detections")
    min_frame, max_frame = min(frame_rows), max(frame_rows)
    frame_cache: dict[int, np.ndarray] = {}
    by_node: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    inference_frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    feature_sum = np.zeros(12, dtype=np.float64)
    feature_sumsq = np.zeros(12, dtype=np.float64)
    feature_min = np.full(12, np.inf, dtype=np.float64)
    feature_max = np.full(12, -np.inf, dtype=np.float64)
    nonzero_rows = 0
    count = 0

    def get_frame(index: int) -> np.ndarray:
        bounded = min(max(index, min_frame), max_frame)
        if bounded not in frame_cache:
            value = np.asarray(image[bounded])
            if value.ndim != 3 or not np.issubdtype(value.dtype, np.number):
                raise R1Error("raw image frame is not numeric Z/Y/X")
            frame_cache[bounded] = value
        return frame_cache[bounded]

    with torch.inference_mode():
        for frame_index in sorted(frame_rows):
            node_ids, coords = frame_rows[frame_index]
            output = []
            source_frames = [
                get_frame(frame_index - 1),
                get_frame(frame_index),
                get_frame(frame_index + 1),
            ]
            for start in range(0, len(node_ids), batch_size):
                stop = min(len(node_ids), start + batch_size)
                crops = extract_crop_batch(source_frames, coords[start:stop])
                features = adapter(crops).to(device="cpu", dtype=torch.float32).numpy()
                output.append(features)
            wrfeat = np.concatenate(output, axis=0) if output else np.empty((0, 12), dtype=np.float32)
            if wrfeat.shape != (len(node_ids), 12) or not np.isfinite(wrfeat).all():
                raise R1Error("wrfeat population is malformed")
            inference_frames[frame_index] = (node_ids, coords)
            for node, coord, feature in zip(node_ids, coords, wrfeat, strict=True):
                node_id = int(node)
                if node_id in by_node:
                    raise R1Error("detection node ID repeats")
                by_node[node_id] = (frame_index, coord, feature)
            if len(wrfeat):
                feature64 = wrfeat.astype(np.float64, copy=False)
                feature_sum += feature64.sum(axis=0)
                feature_sumsq += np.square(feature64).sum(axis=0)
                feature_min = np.minimum(feature_min, feature64.min(axis=0))
                feature_max = np.maximum(feature_max, feature64.max(axis=0))
                nonzero_rows += int(
                    np.count_nonzero(np.linalg.norm(feature64, axis=1) > 0.0)
                )
                count += len(wrfeat)
            for stale in [value for value in frame_cache if value < frame_index]:
                del frame_cache[stale]
    if count == 0:
        raise R1Error("video has no detection features")
    mean = feature_sum / count
    variance = np.maximum(feature_sumsq / count - mean * mean, 0.0)
    evidence = {
        "node_count": count,
        "nonzero_row_count": nonzero_rows,
        "channel_mean": mean.tolist(),
        "channel_std": np.sqrt(variance).tolist(),
        "channel_min": feature_min.tolist(),
        "channel_max": feature_max.tolist(),
        "all_rows_nonzero": nonzero_rows == count,
        "image_conditioned_channel_count": int(np.count_nonzero(variance > 1.0e-12)),
    }
    if evidence["all_rows_nonzero"] is not True or evidence["image_conditioned_channel_count"] < 1:
        raise R1Error("adapter did not emit genuine nonzero image-conditioned wrfeat")
    return by_node, inference_frames, evidence


def source_chunks(
    source_ids: np.ndarray,
    source_coords: np.ndarray,
    target_ids: np.ndarray,
    target_coords: np.ndarray,
    contract: Mapping[str, Any],
):
    compiler = contract["compiler"]
    scale = np.asarray(compiler["scale_zyx_um"], dtype=np.float64)
    source_physical = source_coords * scale[None, :]
    target_physical = target_coords * scale[None, :]
    tree = cKDTree(target_physical)
    order = np.lexsort((source_coords[:, 2], source_coords[:, 1], source_coords[:, 0]))
    chunk_size = int(compiler["initial_source_chunk"])
    pending = [order[start : start + chunk_size] for start in range(0, len(order), chunk_size)]
    while pending:
        indices = pending.pop(0)
        candidates = []
        required_targets: set[int] = set()
        for source_index in indices:
            hits = tree.query_ball_point(
                source_physical[source_index], r=float(compiler["inference_radius_um"])
            )
            hits = sorted(
                hits,
                key=lambda target_index: (
                    float(np.linalg.norm(target_physical[target_index] - source_physical[source_index])),
                    int(target_ids[target_index]),
                ),
            )[: int(compiler["inference_neighbor_cap"])]
            row = []
            for target_index in hits:
                distance = float(
                    np.linalg.norm(target_physical[target_index] - source_physical[source_index])
                )
                row.append((int(target_ids[target_index]), distance))
                required_targets.add(int(target_ids[target_index]))
            candidates.append((int(source_ids[source_index]), row))
        required_count = len(indices) + len(required_targets)
        if required_count > int(compiler["max_tokens"]):
            if len(indices) == 1:
                raise R1Error("one source neighborhood exceeds max tokens")
            middle = len(indices) // 2
            pending.insert(0, indices[middle:])
            pending.insert(0, indices[:middle])
            continue
        yield candidates, required_targets


def infer_association(model: nn.Module, ordered_nodes: Sequence[int], by_node: Mapping[int, tuple[int, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, dict[int, int]]:
    coordinates = np.stack(
        [
            np.concatenate(
                (
                    np.asarray((by_node[node][0],), dtype=np.float32),
                    np.asarray(by_node[node][1], dtype=np.float32),
                )
            )
            for node in ordered_nodes
        ]
    )
    wrfeat = np.stack([np.asarray(by_node[node][2], dtype=np.float32) for node in ordered_nodes])
    timepoints = torch.from_numpy(coordinates[:, 0].astype(np.int64))
    coords = torch.from_numpy(coordinates).unsqueeze(0)
    features = torch.from_numpy(wrfeat).unsqueeze(0)
    with torch.inference_mode():
        logits = model(coords, features=features)
        association = model.normalize_output(logits, timepoints.unsqueeze(0), coords)[0]
    value = association.detach().to(device="cpu", dtype=torch.float32).numpy()
    if value.shape != (len(ordered_nodes), len(ordered_nodes)) or not np.isfinite(value).all():
        raise R1Error("Trackastra association matrix is malformed")
    return value, {node: index for index, node in enumerate(ordered_nodes)}


def run_video(identity: str, model: nn.Module, adapter: CropAdapter, store: Any, contract: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    detections = load_detections(identity, contract)
    dataset = identity.split(":", 1)[1]
    image = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
    by_node, frame_nodes, wrfeat_evidence = embed_video(
        image=image,
        detections=detections,
        adapter=adapter,
        batch_size=int(contract["adapter"]["batch_size"]),
    )
    all_frames = sorted(frame_nodes)
    min_frame, max_frame = all_frames[0], all_frames[-1]
    edge_source: list[int] = []
    edge_target: list[int] = []
    edge_weight: list[float] = []
    edge_distance: list[float] = []
    calls = peak_tokens = 0
    compiler = contract["compiler"]
    for source_t in range(min_frame, max_frame):
        if source_t not in frame_nodes or source_t + 1 not in frame_nodes:
            continue
        source_ids, source_coords = frame_nodes[source_t]
        target_ids, target_coords = frame_nodes[source_t + 1]
        if len(source_ids) == 0 or len(target_ids) == 0:
            continue
        start = min(
            max(source_t - 1, min_frame),
            max(min_frame, max_frame - int(compiler["window_frames"]) + 1),
        )
        context_candidates = [
            int(node)
            for frame in range(start, start + int(compiler["window_frames"]))
            if frame in frame_nodes
            for node in frame_nodes[frame][0]
        ]
        for candidates, required_targets in source_chunks(
            source_ids, source_coords, target_ids, target_coords, contract
        ):
            required_sources = {source for source, _ in candidates}
            required = required_sources | required_targets
            centroid = np.mean([by_node[node][1] for node in required], axis=0)
            optional = sorted(
                set(context_candidates) - required,
                key=lambda node: (
                    float(np.sum((by_node[node][1] - centroid) ** 2)),
                    by_node[node][0],
                    node,
                ),
            )
            selected = required | set(
                optional[: int(compiler["max_tokens"]) - len(required)]
            )
            ordered = tuple(sorted(selected, key=lambda node: (by_node[node][0], node)))
            association, index = infer_association(model, ordered, by_node)
            calls += 1
            peak_tokens = max(peak_tokens, len(ordered))
            for source, targets in candidates:
                for target, distance in targets:
                    edge_source.append(source)
                    edge_target.append(target)
                    edge_weight.append(float(association[index[source], index[target]]))
                    edge_distance.append(distance)
    nodes = sorted(
        (
            int(node),
            int(value[0]),
            float(value[1][0]),
            float(value[1][1]),
            float(value[1][2]),
        )
        for node, value in by_node.items()
    )
    nodes.sort(key=lambda row: (row[1], row[0]))
    edges = sorted(
        zip(edge_source, edge_target, edge_weight, edge_distance, strict=True),
        key=lambda row: (row[0], row[1]),
    )
    return (
        np.asarray(nodes, dtype=np.float64).reshape(-1, 5),
        np.asarray(edges, dtype=np.float64).reshape(-1, 4),
        {
            "identity": identity,
            "node_count": len(nodes),
            "candidate_edge_count": len(edges),
            "inference_calls": calls,
            "peak_tokens": peak_tokens,
            "elapsed_seconds": time.perf_counter() - started,
            "wrfeat": wrfeat_evidence,
        },
    )


def selected_edges(edges: np.ndarray, contract: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    compiler = contract["compiler"]
    by_source: dict[int, list[tuple[int, float, float]]] = {}
    for source, target, weight, distance in edges:
        if distance <= float(compiler["selected_radius_um"]) + 1.0e-9:
            by_source.setdefault(int(source), []).append(
                (int(target), float(weight), float(distance))
            )
    eligible = []
    for source, candidates in by_source.items():
        for target, weight, distance in sorted(
            candidates, key=lambda row: (row[2], row[0])
        )[: int(compiler["selected_neighbor_cap"])]:
            eligible.append((source, target, weight, distance))
    incoming: set[int] = set()
    outgoing: set[int] = set()
    result = []
    for source, target, weight, _distance in sorted(
        eligible, key=lambda row: (-row[2], row[0], row[1])
    ):
        if weight < float(compiler["association_threshold"]):
            break
        if target in incoming or source in outgoing:
            continue
        incoming.add(target)
        outgoing.add(source)
        result.append((source, target))
    return tuple(sorted(result))


def graph_from_arrays(identity: str, nodes: np.ndarray, edges: np.ndarray, contract: Mapping[str, Any]):
    configure_repository_imports(contract)
    from celltrack_alpha.kaggle_calibration import (
        SubmissionEdge,
        SubmissionGraph,
        SubmissionNode,
        validate_graphs,
    )

    selected = selected_edges(edges, contract)
    used = {node for pair in selected for node in pair}
    node_rows = {int(row[0]): row for row in nodes if int(row[0]) in used}
    dataset = identity.split(":", 1)[1]
    graph = SubmissionGraph(
        dataset=dataset,
        nodes={
            node: SubmissionNode(
                node,
                int(row[1]),
                int(round(float(row[2]))),
                int(round(float(row[3]))),
                int(round(float(row[4]))),
            )
            for node, row in node_rows.items()
        },
        edges=[SubmissionEdge(source, target) for source, target in selected],
    )
    validate_graphs({dataset: graph})
    return graph


def write_graph(path: Path, graph: Any, contract: Mapping[str, Any]) -> str:
    configure_repository_imports(contract)
    from celltrack_alpha.kaggle_calibration import write_submission

    if path.exists():
        raise R1Error(f"refusing to overwrite graph fragment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        write_submission({graph.dataset: graph}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return sha256(path)


def write_association_bank(
    path: Path, identity: str, nodes: np.ndarray, edges: np.ndarray
) -> str:
    """Persist every frozen candidate-pair probability before graph decoding."""
    if path.exists():
        raise R1Error(f"refusing to overwrite association bank: {path}")
    if nodes.ndim != 2 or nodes.shape[1] != 5:
        raise R1Error("association-bank nodes are malformed")
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise R1Error("association-bank edges are malformed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    metadata = canonical_json(
        {
            "schema": BANK_SCHEMA,
            "identity": identity,
            "probability_dtype": "float32",
            "distance_unit": "micrometre",
            "complete_candidate_bank": True,
        }
    )
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.frombuffer(metadata, dtype=np.uint8),
                node_id=nodes[:, 0].astype(np.int64),
                frame=nodes[:, 1].astype(np.int32),
                coords_zyx=nodes[:, 2:5].astype(np.float32),
                edge_source=edges[:, 0].astype(np.int64),
                edge_target=edges[:, 1].astype(np.int64),
                probability=edges[:, 2].astype(np.float32),
                distance_um=edges[:, 3].astype(np.float32),
            )
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


def aggregate_fragment_sha(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda value: str(value["identity"])):
        digest.update(str(row["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["fragment_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def aggregate_bank_sha(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda value: str(value["identity"])):
        digest.update(str(row["identity"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["association_bank_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def completed_video(identity: str, output: Path, contract_sha256: str) -> dict[str, Any] | None:
    slug = identity.replace(":", "__", 1)
    fragment = output / "fragments" / f"{slug}.csv"
    bank = output / "association_banks" / f"{slug}.npz"
    receipt_path = output / "receipts" / f"{slug}.json"
    if not fragment.exists() and not bank.exists() and not receipt_path.exists():
        return None
    if not fragment.is_file() or not bank.is_file() or not receipt_path.is_file():
        raise R1Error(f"partial completed-video state: {identity}")
    receipt = load_json(receipt_path, f"{identity} receipt")
    if (
        receipt.get("schema") != VIDEO_SCHEMA
        or receipt.get("identity") != identity
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("fragment_sha256") != sha256(fragment)
        or receipt.get("association_bank_sha256") != sha256(bank)
        or receipt.get("association_bank_complete") is not True
        or receipt.get("held_fit_score_read") is not False
        or receipt.get("gt_access") is not False
    ):
        raise R1Error(f"completed-video receipt drifted: {identity}")
    return receipt


def infer(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    configure_runtime(contract)
    preflight_receipt = load_json(args.preflight, "preflight receipt")
    require_hash(args.preflight, args.preflight_sha256, "preflight receipt")
    if (
        preflight_receipt.get("schema") != PREFLIGHT_SCHEMA
        or preflight_receipt.get("status") != "PASS_SCORE_BLIND_PREFLIGHT"
        or preflight_receipt.get("contract", {}).get("sha256") != args.contract_sha256
        or preflight_receipt.get("held_fit_score_read") is not False
    ):
        raise R1Error("preflight receipt does not authorize score-blind inference")
    if (args.output / "INFERENCE_SEAL.json").exists():
        raise R1Error("inference is already sealed; repeat forbidden")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fragments").mkdir(exist_ok=True)
    (args.output / "association_banks").mkdir(exist_ok=True)
    (args.output / "receipts").mkdir(exist_ok=True)
    model = load_trackastra(contract)
    adapter = load_adapter(contract)
    identities = tuple(contract["scope"]["identities"])
    deadline_seconds = float(contract["runtime"]["hard_wall_seconds"])
    started = time.monotonic()
    accumulated_prior_seconds = 0.0
    records = []
    raw_path = Path(contract["inputs"]["raw_archive"]["path"])
    with ZipStore(raw_path, mode="r") as store:
        for ordinal, identity in enumerate(identities, start=1):
            if (
                accumulated_prior_seconds + time.monotonic() - started
                >= deadline_seconds - 120.0
            ):
                raise TimeoutError("R1 hard wall reached before all fragments sealed")
            existing = completed_video(identity, args.output, args.contract_sha256)
            if existing is not None:
                records.append(existing)
                accumulated_prior_seconds += float(
                    existing["runtime"]["elapsed_seconds"]
                )
                continue
            nodes, edges, runtime = run_video(identity, model, adapter, store, contract)
            graph = graph_from_arrays(identity, nodes, edges, contract)
            slug = identity.replace(":", "__", 1)
            fragment = args.output / "fragments" / f"{slug}.csv"
            bank = args.output / "association_banks" / f"{slug}.npz"
            bank_hash = write_association_bank(bank, identity, nodes, edges)
            fragment_hash = write_graph(fragment, graph, contract)
            receipt = {
                "schema": VIDEO_SCHEMA,
                "identity": identity,
                "ordinal": ordinal,
                "contract_sha256": args.contract_sha256,
                "fragment_path": str(fragment),
                "fragment_sha256": fragment_hash,
                "fragment_bytes": fragment.stat().st_size,
                "association_bank_path": str(bank),
                "association_bank_sha256": bank_hash,
                "association_bank_bytes": bank.stat().st_size,
                "association_bank_complete": True,
                "prediction_nodes": len(graph.nodes),
                "prediction_edges": len(graph.edges),
                "predicted_forks": 0,
                "runtime": runtime,
                "adapter_training_steps": 0,
                "gt_access": False,
                "held_fit_score_read": False,
                "gpu_hours": 0.0,
            }
            atomic_json(args.output / "receipts" / f"{slug}.json", receipt)
            records.append(receipt)
            print(
                f"R1 infer {ordinal:02d}/36 {identity} "
                f"nodes={len(graph.nodes)} edges={len(graph.edges)} "
                f"elapsed={runtime['elapsed_seconds']:.1f}s",
                flush=True,
            )
    if len(records) != 36 or {row["identity"] for row in records} != set(identities):
        raise R1Error("inference did not cover the sealed population")
    if any(
        row["runtime"]["wrfeat"]["all_rows_nonzero"] is not True
        or row["runtime"]["wrfeat"]["image_conditioned_channel_count"] < 1
        for row in records
    ):
        raise R1Error("one video lacks real image-conditioned wrfeat")
    seal = {
        "schema": SEAL_SCHEMA,
        "status": "SEALED_ALL_36_BEFORE_SCORE_READ",
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "preflight": {"path": str(args.preflight), "sha256": args.preflight_sha256},
        "population": {"family": "44b6", "identity_count": 36, "identities": list(identities)},
        "adapter": {
            "file_sha256": contract["adapter"]["file_sha256"],
            "state_tensor_sha256": contract["adapter"]["state_tensor_sha256"],
            "parameter_count": contract["adapter"]["parameter_count"],
            "training_steps": 0,
        },
        "compiler": contract["compiler"],
        "aggregate": {
            "fragment_sha256": aggregate_fragment_sha(records),
            "association_bank_sha256": aggregate_bank_sha(records),
            "association_bank_bytes": sum(
                int(row["association_bank_bytes"]) for row in records
            ),
            "node_count": sum(int(row["prediction_nodes"]) for row in records),
            "edge_count": sum(int(row["prediction_edges"]) for row in records),
            "candidate_edge_count": sum(int(row["runtime"]["candidate_edge_count"]) for row in records),
            "inference_calls": sum(int(row["runtime"]["inference_calls"]) for row in records),
            "peak_tokens": max(int(row["runtime"]["peak_tokens"]) for row in records),
            "predicted_forks": 0,
            "elapsed_seconds": accumulated_prior_seconds
            + time.monotonic()
            - started,
            "current_process_elapsed_seconds": time.monotonic() - started,
        },
        "videos": records,
        "gt_access": False,
        "held_fit_score_read": False,
        "gpu_hours": 0.0,
    }
    atomic_json(args.output / "INFERENCE_SEAL.json", seal)
    return seal


def score(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract, args.contract_sha256)
    configure_runtime(contract)
    require_hash(args.inference_seal, args.inference_seal_sha256, "inference seal")
    seal = load_json(args.inference_seal, "inference seal")
    identities = tuple(contract["scope"]["identities"])
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("status") != "SEALED_ALL_36_BEFORE_SCORE_READ"
        or seal.get("contract", {}).get("sha256") != args.contract_sha256
        or tuple(seal.get("population", {}).get("identities", [])) != identities
        or seal.get("gt_access") is not False
        or seal.get("held_fit_score_read") is not False
        or seal.get("adapter", {}).get("training_steps") != 0
    ):
        raise R1Error("inference seal is not score-eligible")
    records = seal["videos"]
    if aggregate_fragment_sha(records) != seal["aggregate"]["fragment_sha256"]:
        raise R1Error("sealed aggregate fragment hash drifted")
    fragment_root = args.inference_seal.parent / "fragments"
    bank_root = args.inference_seal.parent / "association_banks"
    for row in records:
        slug = str(row["identity"]).replace(":", "__", 1)
        fragment_path = fragment_root / f"{slug}.csv"
        bank_path = bank_root / f"{slug}.npz"
        require_hash(fragment_path, row["fragment_sha256"], str(row["identity"]))
        require_hash(
            bank_path, row["association_bank_sha256"],
            f"{str(row['identity'])} association bank",
        )
    if aggregate_bank_sha(records) != seal["aggregate"]["association_bank_sha256"]:
        raise R1Error("sealed aggregate association-bank hash drifted")
    verify_scorer(contract)
    configure_repository_imports(contract)
    import scoreall

    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(fragment_root)), identities
    )
    raw_path = Path(contract["inputs"]["raw_archive"]["path"])
    started = time.perf_counter()
    with ZipStore(raw_path, mode="r") as store:
        rows, provenance, coverage = scoreall.score_fragment_input(store, manifest)
    summary = scoreall.official_summary(rows)
    edge_counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    best = float(contract["decision"]["best_known_score"])
    floor = float(contract["decision"]["neighborhood_floor"])
    observed = float(summary["score"])
    branch = "A" if observed >= floor or observed > best else "B"
    result = {
        "schema": SCORE_SCHEMA,
        "status": "COMPLETE_ONE_SHOT_ZERO_SHOT_MEASUREMENT",
        "decision_reference": {
            "best_known_score": best,
            "neighborhood_floor": floor,
            "apncc_44b6_reference": float(
                contract["decision"]["apncc_44b6_reference"]
            ),
        },
        "candidate": {
            "score": observed,
            "adj_edge_jaccard": float(summary["adj_edge_jaccard"]),
            "edge_jaccard": float(summary["edge_jaccard"]),
            "raw_edge_jaccard": float(summary["edge_jaccard"]),
            "node_recall": float(summary["node_recall"]),
            "edge_tp": edge_counts["edge_tp"],
            "edge_fp": edge_counts["edge_fp"],
            "edge_fn": edge_counts["edge_fn"],
            "division_tp": int(summary["division_tp"]),
            "division_fp": int(summary["division_fp"]),
            "division_fn": int(summary["division_fn"]),
            "division_jaccard": float(summary["division_jaccard"]),
            "branch": branch,
        },
        "execution": {
            "gpu_h": 0.0,
            "trained": False,
            "training_steps": 0,
            "elapsed_seconds": float(seal["aggregate"]["elapsed_seconds"])
            + time.perf_counter()
            - started,
            "inference_elapsed_seconds": float(
                seal["aggregate"]["elapsed_seconds"]
            ),
            "scoring_elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "config_sha256": args.contract_sha256,
            "prereg_sha256": contract["preregistration"]["sha256"],
            "adapter_weight_sha256": contract["adapter"]["file_sha256"],
            "adapter_state_tensor_sha256": contract["adapter"][
                "state_tensor_sha256"
            ],
            "archive_sha256": contract["inputs"]["raw_archive"]["sha256"],
            "association_bank_sha256": seal["aggregate"][
                "association_bank_sha256"
            ],
            "aggregate_graph_sha256": seal["aggregate"]["fragment_sha256"],
            "scorer_input_aggregate_sha256": provenance["aggregate_sha256"],
            "frozen_scorer_source_hashes": contract["inputs"][
                "scorer_sources"
            ],
            "sealed_before_score": True,
        },
        "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
        "inference_seal": {"path": str(args.inference_seal), "sha256": args.inference_seal_sha256},
        "score": summary,
        "edge_counts": edge_counts,
        "coverage": coverage,
        "input_provenance": provenance,
        "per_video": rows,
        "decision": {
            "best_known_score": best,
            "neighborhood_floor": floor,
            "apncc_44b6_reference": float(contract["decision"]["apncc_44b6_reference"]),
            "delta_vs_best_known_score": observed - best,
            "delta_vs_neighborhood_floor": observed - floor,
            "delta_vs_apncc_44b6_reference": observed
            - float(contract["decision"]["apncc_44b6_reference"]),
            "branch": branch,
            "branch_rule": "A iff score >= neighborhood_floor or score is the new best; otherwise B",
            "new_best": observed > best,
        },
        "measurement": {
            "training_steps": 0,
            "alternatives_scored": 1,
            "post_result_tuning": False,
            "repeat": False,
            "cv57_access": False,
            "submission": False,
            "gpu_hours": 0.0,
            "scoring_elapsed_seconds": time.perf_counter() - started,
        },
    }
    atomic_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    make_adapter = sub.add_parser("make-adapter")
    make_adapter.add_argument("--output", type=Path, required=True)
    make_adapter.add_argument("--seed", type=int, required=True)
    make_map = sub.add_parser("make-member-map")
    make_map.add_argument("--raw-manifest", type=Path, required=True)
    make_map.add_argument("--h1-report", type=Path, required=True)
    make_map.add_argument("--output", type=Path, required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--contract", type=Path, required=True)
    pre.add_argument("--contract-sha256", required=True)
    pre.add_argument("--output", type=Path, required=True)
    emit = sub.add_parser("infer")
    emit.add_argument("--contract", type=Path, required=True)
    emit.add_argument("--contract-sha256", required=True)
    emit.add_argument("--preflight", type=Path, required=True)
    emit.add_argument("--preflight-sha256", required=True)
    emit.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("score")
    measure.add_argument("--contract", type=Path, required=True)
    measure.add_argument("--contract-sha256", required=True)
    measure.add_argument("--inference-seal", type=Path, required=True)
    measure.add_argument("--inference-seal-sha256", required=True)
    measure.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "make-adapter":
        report = create_adapter(args.output, args.seed)
    elif args.command == "make-member-map":
        report = create_member_map(
            raw_manifest=args.raw_manifest,
            h1_report=args.h1_report,
            output=args.output,
        )
    elif args.command == "preflight":
        report = preflight(args)
    elif args.command == "infer":
        report = infer(args)
    elif args.command == "score":
        report = score(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
