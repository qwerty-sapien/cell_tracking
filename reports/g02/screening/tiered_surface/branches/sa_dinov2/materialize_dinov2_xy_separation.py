#!/usr/bin/env python3
"""Materialize only frozen 44b6 DINOv2 XY separation-population features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
GATE = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
POPULATION_ROOT = GATE / "run_v1/population"
REQUEST_ROOT = GATE / "run_v1/feature_requests"
POPULATION_MANIFEST = GATE / "run_v1/population_manifest.json"
AMENDMENT = GATE / "PREREGISTRATION_v3_per_class_authenticated_weight.json"
SEALED_B0 = GATE / "run_v3/B0_SEALED.json"
STAGING_CONTRACT = HERE / "R5_XY_BETA_STAGING_CONTRACT.json"
FEATURIZER_PATH = HERE / "dinov2_xy_action_featurizer.py"

AMENDMENT_SHA256 = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
SEALED_B0_SHA256 = "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
STAGING_CONTRACT_SHA256 = "c8e3df7739a3691b6ad9c93234ecc8db9696992c126de3ff343def38a93d5610"
FEATURIZER_SHA256 = "f93a202a7e662859f92442a4a6932ad371af513c65627838f2ebfd7bfecb9b49"
SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
WEIGHT_BYTES = 88_283_115
RAW_ARCHIVE_BYTES = 87_393_127_165
RAW_CENTRAL_DIRECTORY_SHA256 = "5edffe5596e0918f92336ffb269b17df4f2e5f41b55a67bfa01e0253d58cdc77"
RAW_ENTRY_COUNT = 24_886
MAX_SECONDS = 5 * 3600
SOFT_STOP_SECONDS = MAX_SECONDS - 300
PROJECTION_SAMPLE_ENDPOINTS = 4096
CLASSES = ("delete", "occupied_swap")
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)
Z_OFFSETS = np.arange(-16, 17, dtype=np.float32) * 0.25
YX_OFFSETS = np.arange(-16, 17, dtype=np.int64)
CENTRAL_PATCH_INDICES = (119, 120, 135, 136)


class MaterializationError(RuntimeError):
    """Fail-closed local materialization error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


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


def load_featurizer():
    specification = importlib.util.spec_from_file_location(
        "dinov2_xy_action_featurizer_frozen", FEATURIZER_PATH
    )
    if specification is None or specification.loader is None:
        raise MaterializationError("cannot load frozen DINOv2 action featurizer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def authenticate_inputs(
    raw_archive: Path,
    source: Path,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    pins = (
        (AMENDMENT, AMENDMENT_SHA256),
        (SEALED_B0, SEALED_B0_SHA256),
        (POPULATION_MANIFEST, POPULATION_MANIFEST_SHA256),
        (STAGING_CONTRACT, STAGING_CONTRACT_SHA256),
        (FEATURIZER_PATH, FEATURIZER_SHA256),
    )
    for path, expected in pins:
        observed = sha256(path)
        if observed != expected:
            raise MaterializationError(f"frozen input hash mismatch: {path}: {observed}")
    if raw_archive.stat().st_size != RAW_ARCHIVE_BYTES:
        raise MaterializationError("raw archive byte size mismatch")
    directory_sha, entry_count = central_directory_identity(raw_archive)
    if (
        directory_sha != RAW_CENTRAL_DIRECTORY_SHA256
        or entry_count != RAW_ENTRY_COUNT
    ):
        raise MaterializationError("raw archive central-directory identity mismatch")
    if not (source / ".git").exists():
        raise MaterializationError("pinned DINOv2 source checkout is absent")
    import subprocess

    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise MaterializationError(f"DINOv2 source revision drift: {revision}")
    if checkpoint.stat().st_size != WEIGHT_BYTES or sha256(checkpoint) != WEIGHT_SHA256:
        raise MaterializationError("DINOv2 checkpoint identity mismatch")

    manifest = json.loads(POPULATION_MANIFEST.read_text())
    expected_requests: dict[tuple[str, str], str] = {}
    for action_class in CLASSES:
        for row in manifest["classes"][action_class]["videos"]:
            request = ROOT / row["request_path"]
            population = ROOT / row["population_path"]
            if sha256(request) != row["request_sha256"]:
                raise MaterializationError(f"feature request hash mismatch: {request}")
            if sha256(population) != row["population_sha256"]:
                raise MaterializationError(f"population hash mismatch: {population}")
            expected_requests[(action_class, request.name)] = row["request_sha256"]
    observed_requests = {
        (path.parent.name, path.name)
        for path in REQUEST_ROOT.glob("*/*.npz")
    }
    if observed_requests != set(expected_requests):
        raise MaterializationError("feature-request inventory mismatch")
    return manifest, expected_requests


def exact_uint16_quantile(histogram: np.ndarray, quantile: float) -> float:
    histogram = np.asarray(histogram, dtype=np.int64)
    if histogram.shape != (65536,) or np.any(histogram < 0):
        raise MaterializationError("invalid uint16 histogram")
    count = int(histogram.sum())
    if count <= 0 or not 0.0 <= quantile <= 1.0:
        raise MaterializationError("invalid quantile request")
    position = (count - 1) * float(quantile)
    low_rank = int(math.floor(position))
    high_rank = int(math.ceil(position))
    cumulative = np.cumsum(histogram)
    low_value = int(np.searchsorted(cumulative, low_rank + 1, side="left"))
    high_value = int(np.searchsorted(cumulative, high_rank + 1, side="left"))
    return float(low_value + (position - low_rank) * (high_value - low_value))


def video_quantiles(array: zarr.Array) -> tuple[float, float]:
    if str(array.dtype) != "uint16" or tuple(array.shape[1:]) != (64, 256, 256):
        raise MaterializationError(
            f"raw array contract drift: shape={array.shape}, dtype={array.dtype}"
        )
    histogram = np.zeros(65536, dtype=np.int64)
    for frame_index in range(int(array.shape[0])):
        frame = np.asarray(array[frame_index])
        histogram += np.bincount(frame.reshape(-1), minlength=65536)
    low = exact_uint16_quantile(histogram, 0.01)
    high = exact_uint16_quantile(histogram, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise MaterializationError(f"degenerate video normalization: {low}, {high}")
    return low, high


def pseudo_rgb_batches(
    frame: np.ndarray,
    coordinates_zyx: np.ndarray,
    low: float,
    high: float,
    batch_size: int,
) -> Iterator[tuple[slice, np.ndarray]]:
    frame = np.asarray(frame)
    coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
    if frame.shape != (64, 256, 256) or frame.dtype != np.uint16:
        raise MaterializationError("raw frame contract drift")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise MaterializationError("endpoint-coordinate contract drift")
    for offset in range(0, len(coordinates), batch_size):
        stop = min(offset + batch_size, len(coordinates))
        batch = coordinates[offset:stop]
        z_position = batch[:, 0, None].astype(np.float32) + Z_OFFSETS[None, :]
        z_floor = np.floor(z_position).astype(np.int64)
        z_weight = (z_position - z_floor).astype(np.float32)
        z_ceil = z_floor + (z_weight > 0).astype(np.int64)
        z_floor = np.clip(z_floor, 0, frame.shape[0] - 1)
        z_ceil = np.clip(z_ceil, 0, frame.shape[0] - 1)
        y_index = np.clip(batch[:, 1, None] + YX_OFFSETS[None, :], 0, frame.shape[1] - 1)
        x_index = np.clip(batch[:, 2, None] + YX_OFFSETS[None, :], 0, frame.shape[2] - 1)
        lower = frame[
            z_floor[:, :, None, None],
            y_index[:, None, :, None],
            x_index[:, None, None, :],
        ].astype(np.float32)
        upper = frame[
            z_ceil[:, :, None, None],
            y_index[:, None, :, None],
            x_index[:, None, None, :],
        ].astype(np.float32)
        volume = lower + (upper - lower) * z_weight[:, :, None, None]
        del lower, upper
        np.clip(volume, low, high, out=volume)
        volume -= low
        volume /= high - low
        pseudo = np.stack(
            (
                volume.max(axis=1),
                volume.mean(axis=1, dtype=np.float32),
                volume[:, 16],
            ),
            axis=1,
        ).astype(np.float32)
        if pseudo.shape != (len(batch), 3, 33, 33) or not np.isfinite(pseudo).all():
            raise MaterializationError("invalid pseudo-RGB crop")
        yield slice(offset, stop), pseudo


def load_model(source: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14

    state_dict = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise MaterializationError("checkpoint is not a tensor-only state dict")
    model = dinov2_vits14(pretrained=False)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise MaterializationError("strict DINOv2 load failed")
    model.eval().requires_grad_(False).to(device)
    return model


def endpoint_rows(request: dict[str, np.ndarray], role: str) -> np.ndarray:
    return np.column_stack(
        (
            request[f"{role}_t"].astype(np.int64),
            request[f"{role}_zyx"].astype(np.int64),
        )
    )


def structured_rows(rows: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(rows, dtype=np.int64)
    return contiguous.view(
        np.dtype([("t", "<i8"), ("z", "<i8"), ("y", "<i8"), ("x", "<i8")])
    ).reshape(-1)


def map_rows(unique_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    unique_structured = structured_rows(unique_rows)
    rows_structured = structured_rows(rows)
    indices = np.searchsorted(unique_structured, rows_structured)
    if np.any(indices >= len(unique_rows)) or not np.array_equal(
        unique_structured[indices], rows_structured
    ):
        raise MaterializationError("endpoint mapping is incomplete")
    return indices


def embed_video(
    model: torch.nn.Module,
    device: torch.device,
    array: zarr.Array,
    unique_rows: np.ndarray,
    crop_batch: int,
    forward_batch: int,
    started: float,
    completed_before: int,
    total_endpoints: int,
) -> tuple[np.ndarray, tuple[float, float]]:
    low, high = video_quantiles(array)
    embeddings = np.empty((len(unique_rows), 768), dtype=np.float32)
    written = np.zeros(len(unique_rows), dtype=bool)
    mean = torch.tensor(IMAGE_NET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_NET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    for frame_index in np.unique(unique_rows[:, 0]):
        if time.monotonic() - started >= SOFT_STOP_SECONDS:
            raise MaterializationError("local five-hour soft-stop approached")
        index = np.flatnonzero(unique_rows[:, 0] == frame_index)
        frame = np.asarray(array[int(frame_index)])
        for local_slice, pseudo in pseudo_rgb_batches(
            frame, unique_rows[index, 1:], low, high, crop_batch
        ):
            for forward_offset in range(0, len(pseudo), forward_batch):
                forward_stop = min(forward_offset + forward_batch, len(pseudo))
                tensor = torch.from_numpy(
                    pseudo[forward_offset:forward_stop]
                ).to(device)
                tensor = F.interpolate(
                    tensor,
                    size=(224, 224),
                    mode="bicubic",
                    align_corners=False,
                )
                tensor = (tensor - mean) / std
                with torch.inference_mode():
                    result = model.forward_features(tensor)
                class_token = result["x_norm_clstoken"]
                patch_token = result["x_norm_patchtokens"][
                    :, CENTRAL_PATCH_INDICES
                ].mean(dim=1)
                embedded = torch.cat((class_token, patch_token), dim=1)
                values = embedded.float().cpu().numpy()
                destination = index[
                    local_slice.start + forward_offset : local_slice.start + forward_stop
                ]
                embeddings[destination] = values
                written[destination] = True
                completed = completed_before + int(written.sum())
                if completed >= PROJECTION_SAMPLE_ENDPOINTS:
                    elapsed = time.monotonic() - started
                    projected = elapsed / completed * total_endpoints
                    if projected >= MAX_SECONDS:
                        raise MaterializationError(
                            "authenticated local projection exceeds five hours: "
                            f"{projected / 3600:.3f}"
                        )
    if not written.all() or not np.isfinite(embeddings).all():
        raise MaterializationError("endpoint embedding coverage is incomplete")
    return embeddings, (low, high)


def permuted_features(
    featurizer: Any,
    request: dict[str, np.ndarray],
    source: np.ndarray,
    target: np.ndarray,
    identity: str,
) -> tuple[np.ndarray, np.ndarray]:
    a1 = featurizer.action_features(source, target)
    permuted_source = np.empty_like(source)
    permuted_target = np.empty_like(target)
    for role, values, output in (
        ("source", source, permuted_source),
        ("target", target, permuted_target),
    ):
        frames = request[f"{role}_t"]
        node_ids = request[f"{role}_id"]
        for frame in np.unique(frames):
            index = np.flatnonzero(frames == frame)
            output[index] = featurizer.within_frame_permutation(
                values[index],
                node_ids[index],
                identity=identity,
                frame=int(frame),
                supplier_contract_sha256=STAGING_CONTRACT_SHA256,
            )
    a0 = featurizer.action_features(permuted_source, permuted_target)
    if not np.isfinite(a1).all() or not np.isfinite(a0).all():
        raise MaterializationError("nonfinite action features")
    return a1, a0


def load_request(path: Path) -> dict[str, np.ndarray]:
    required = {
        "source_id", "target_id", "source_t", "target_t",
        "source_zyx", "target_zyx",
    }
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != required:
            raise MaterializationError(f"request field drift: {path}")
        request = {key: loaded[key] for key in loaded.files}
    count = len(request["source_id"])
    if any(len(request[key]) != count for key in required):
        raise MaterializationError(f"request length drift: {path}")
    return request


def plan_requests() -> tuple[dict[str, dict[str, Path]], int]:
    videos: dict[str, dict[str, Path]] = {}
    endpoint_total = 0
    for path in sorted(REQUEST_ROOT.glob("*/*.npz")):
        identity = path.stem.replace("__", ":", 1)
        videos.setdefault(identity, {})[path.parent.name] = path
    for identity, paths in videos.items():
        rows = []
        for path in paths.values():
            request = load_request(path)
            rows.extend((endpoint_rows(request, "source"), endpoint_rows(request, "target")))
        endpoint_total += len(np.unique(np.concatenate(rows), axis=0))
    return videos, endpoint_total


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output.exists():
        raise MaterializationError(f"refusing to overwrite output: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise MaterializationError("MPS is unavailable in this execution context")
    device = torch.device(args.device)
    population_manifest, expected_requests = authenticate_inputs(
        args.raw_archive, args.source, args.checkpoint
    )
    videos, total_endpoints = plan_requests()
    if len(videos) != 41 or len(expected_requests) != 77 or total_endpoints != 1_417_863:
        raise MaterializationError(
            "frozen request plan drift: "
            f"videos={len(videos)}, files={len(expected_requests)}, "
            f"endpoints={total_endpoints}"
        )
    featurizer = load_featurizer()
    model = load_model(args.source, args.checkpoint, device)
    args.output.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    completed_endpoints = 0
    with ZipStore(args.raw_archive, mode="r") as store:
        for ordinal, (identity, paths) in enumerate(sorted(videos.items()), start=1):
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            requests = {
                action_class: load_request(path)
                for action_class, path in paths.items()
            }
            all_rows = []
            for request in requests.values():
                all_rows.extend(
                    (endpoint_rows(request, "source"), endpoint_rows(request, "target"))
                )
            unique_rows = np.unique(np.concatenate(all_rows), axis=0)
            embeddings, quantiles = embed_video(
                model,
                device,
                array,
                unique_rows,
                args.crop_batch,
                args.forward_batch,
                started,
                completed_endpoints,
                total_endpoints,
            )
            completed_endpoints += len(unique_rows)
            for action_class, request in sorted(requests.items()):
                request_path = paths[action_class]
                source_index = map_rows(unique_rows, endpoint_rows(request, "source"))
                target_index = map_rows(unique_rows, endpoint_rows(request, "target"))
                a1, a0 = permuted_features(
                    featurizer,
                    request,
                    embeddings[source_index],
                    embeddings[target_index],
                    identity,
                )
                destination = args.output / action_class / request_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, a1=a1, a0=a0)
                files.append(
                    {
                        "identity": identity,
                        "action_class": action_class,
                        "count": len(a1),
                        "request_sha256": expected_requests[
                            (action_class, request_path.name)
                        ],
                        "output": str(destination.relative_to(args.output)),
                        "output_sha256": sha256(destination),
                    }
                )
            print(
                f"{ordinal}/{len(videos)} {identity} "
                f"endpoints={len(unique_rows)} q01={quantiles[0]:.6f} "
                f"q99={quantiles[1]:.6f}",
                flush=True,
            )
    if completed_endpoints != total_endpoints or len(files) != len(expected_requests):
        raise MaterializationError("terminal feature inventory drift")
    elapsed = time.monotonic() - started
    result = {
        "schema": "celltrack-alpha.dinov2-xy-action-features.v1",
        "status": "COMPLETE",
        "supplier_id": "F0-DINOV2-VITS14-XY-V1",
        "supplier_contract_sha256": STAGING_CONTRACT_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "sealed_B0_sha256": SEALED_B0_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "source_revision": SOURCE_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "crop_batch": args.crop_batch,
        "forward_batch": args.forward_batch,
        "unique_endpoint_count": total_endpoints,
        "video_count": len(videos),
        "file_count": len(files),
        "files": files,
        "elapsed_seconds": elapsed,
        "projection_seconds_at_completion": elapsed,
        "population_actions": {
            action_class: int(
                sum(
                    row["count"]
                    for row in population_manifest["classes"][action_class]["videos"]
                )
            )
            for action_class in CLASSES
        },
        "action_feature_names": list(featurizer.FEATURE_NAMES),
        "full_pool_cache": False,
        "endpoint_cache_persisted": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
        "barrier": "HOLD_FOR_HUMAN_REVIEW_BEFORE_ANY_GRAPH_PREREGISTRATION",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "files": len(files),
                "endpoints": total_endpoints,
                "seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--crop-batch", type=int, default=64)
    parser.add_argument("--forward-batch", type=int, default=64)
    args = parser.parse_args()
    if args.crop_batch <= 0 or args.forward_batch <= 0:
        raise SystemExit("batch sizes must be positive")
    try:
        run(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "celltrack-alpha.dinov2-xy-action-features.v1",
                    "status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "pid": os.getpid(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
