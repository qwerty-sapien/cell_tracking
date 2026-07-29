#!/usr/bin/env python3
"""Self-contained, bounded DINOv2 R5 XY-beta Kaggle GPU materializer."""

from __future__ import annotations

# Parent step 0 is standard-library only. The first runtime action is the
# recursive mounted-input inventory; no content resolution or heavy import may
# precede it.
import argparse
import base64
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import zlib


EMBEDDED_CONFIG_ZLIB_BASE64 = "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"
WORKING = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
SCHEMA = "celltrack-alpha.dinov2-kaggle-gpu-runtime.v1"


class RuntimeContractError(RuntimeError):
    """Fail-closed runtime contract violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def dump_mount_inventory(input_root: Path) -> tuple[list[dict], float]:
    # This print is deliberately the first executable step-0 action.
    print(
        json.dumps(
            {"event": "STEP0_MOUNT_INVENTORY_BEGIN", "root": str(input_root)},
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.monotonic()
    rows = []
    for directory, directory_names, file_names in os.walk(
        input_root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        root = Path(directory)
        for name in file_names:
            path = root / name
            if not path.is_file():
                raise RuntimeContractError(f"mounted non-file entry: {path}")
            row = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            rows.append(row)
            print(
                json.dumps(
                    {"event": "STEP0_MOUNT_FILE", **row},
                    sort_keys=True,
                ),
                flush=True,
            )
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "event": "STEP0_MOUNT_INVENTORY_END",
                "file_count": len(rows),
                "bytes": sum(row["bytes"] for row in rows),
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return rows, elapsed


def decode_config() -> dict:
    try:
        value = json.loads(
            zlib.decompress(base64.b64decode(EMBEDDED_CONFIG_ZLIB_BASE64))
        )
    except Exception as error:
        raise RuntimeContractError("embedded config decode failed") from error
    if value.get("schema") != "celltrack-alpha.dinov2-kaggle-gpu-embedded-config.v1":
        raise RuntimeContractError("embedded config schema drift")
    return value


def unique_by_hash(rows: list[dict], expected: str, label: str) -> Path:
    matches = [Path(row["path"]) for row in rows if row["sha256"] == expected]
    if len(matches) != 1:
        raise RuntimeContractError(
            f"{label} content resolution expected one match, found {len(matches)}"
        )
    return matches[0]


def competition_root(rows: list[dict], required_video_ids: list[str]) -> Path:
    paths = {row["path"] for row in rows}
    roots = None
    for identity in required_video_ids:
        dataset = identity.split(":", 1)[1]
        suffix = f"/train/{dataset}.zarr/0/zarr.json"
        matches = [path[: -len(suffix)] for path in paths if path.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeContractError(
                f"competition video {identity} expected one mounted root, found {len(matches)}"
            )
        observed = set(matches)
        roots = observed if roots is None else roots & observed
    if roots is None or len(roots) != 1:
        raise RuntimeContractError("competition root resolution is not unique")
    return Path(next(iter(roots)))


def resolve_assets(rows: list[dict], config: dict) -> dict:
    resolved_requests = {}
    for request in config["requests"]:
        digest = request["sha256"]
        path = unique_by_hash(rows, digest, f"request {request['identity']} {request['action_class']}")
        if path.stat().st_size <= 0:
            raise RuntimeContractError("empty request payload")
        resolved_requests[digest] = str(path)
    wheels = {}
    for wheel in config["wheels"]:
        path = unique_by_hash(rows, wheel["sha256"], f"wheel {wheel['name']}")
        if path.stat().st_size != wheel["bytes"]:
            raise RuntimeContractError("wheel byte-size drift")
        wheels[wheel["sha256"]] = str(path)
    weight = unique_by_hash(rows, config["weight"]["sha256"], "DINOv2 weight")
    source = unique_by_hash(rows, config["source"]["sha256"], "DINOv2 source tar")
    if (
        weight.stat().st_size != config["weight"]["bytes"]
        or source.stat().st_size != config["source"]["bytes"]
    ):
        raise RuntimeContractError("weight or source byte-size drift")
    root = competition_root(rows, config["required_video_ids"])
    if config["partial_reuse"]["reused_rows"] != 0:
        raise RuntimeContractError("partial feature reuse is forbidden in this package")
    return {
        "weight": str(weight),
        "source_tar": str(source),
        "requests": resolved_requests,
        "wheels": wheels,
        "competition_root": str(root),
    }


def install_offline_wheels(config: dict, resolved: dict) -> None:
    wheel_paths = [resolved["wheels"][row["sha256"]] for row in config["wheels"]]
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        *wheel_paths,
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        raise RuntimeContractError("offline dependency installation failed")


def extract_source(source_tar: Path) -> Path:
    destination = WORKING / "dinov2_source_checkout"
    destination.mkdir(exist_ok=False)
    with tarfile.open(source_tar, "r") as archive:
        archive.extractall(destination, filter="data")
    source = destination / "dinov2_source"
    if not (source / "dinov2/hub/backbones.py").is_file():
        raise RuntimeContractError("extracted DINOv2 source is incomplete")
    return source


def run_worker(
    action_class: str,
    runtime_path: Path,
    output: Path,
    hard_deadline_epoch: float,
) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        action_class,
        "--runtime",
        str(runtime_path),
        "--output",
        str(output),
        "--deadline-epoch",
        str(hard_deadline_epoch),
    ]
    remaining = max(1.0, hard_deadline_epoch - time.time())
    try:
        result = subprocess.run(
            command,
            text=True,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeContractError(f"{action_class} hard timeout") from error
    if result.returncode != 0:
        raise RuntimeContractError(f"{action_class} worker failed closed")
    manifest_path = output / "class_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeContractError(f"{action_class} terminal manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeContractError(f"{action_class} terminal manifest incomplete")
    return manifest


def parent_main() -> int:
    # Keep this call first. Do not insert resolution, decoding, timing, or heavy
    # imports above it.
    rows, inventory_seconds = dump_mount_inventory(INPUT_ROOT)
    run_started_epoch = time.time() - inventory_seconds
    WORKING.mkdir(parents=True, exist_ok=True)
    inventory_path = WORKING / "DINOV2_MOUNT_INVENTORY.json"
    inventory_path.write_bytes(
        canonical_bytes(
            {
                "schema": "celltrack-alpha.kaggle-mounted-input-inventory.v1",
                "status": "COMPLETE",
                "rows": rows,
                "file_count": len(rows),
                "bytes": sum(row["bytes"] for row in rows),
                "elapsed_seconds": inventory_seconds,
            }
        )
    )

    # Resolution starts only after the complete listing is printed and saved.
    config = decode_config()
    resolved = resolve_assets(rows, config)
    hard_deadline_epoch = run_started_epoch + config["limits"]["hard_seconds"]
    if time.time() >= hard_deadline_epoch:
        raise RuntimeContractError("mount inventory exhausted the hard runtime")
    step0 = {
        "schema": "celltrack-alpha.dinov2-kaggle-gpu-step0.v1",
        "status": "PASS",
        "first_action": "FULL_RECURSIVE_MOUNT_INVENTORY_WITH_SIZE_AND_SHA256",
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "inventory_file_count": len(rows),
        "inventory_bytes": sum(row["bytes"] for row in rows),
        "inventory_elapsed_seconds": inventory_seconds,
        "resolution": resolved,
        "external_bundle_manifest_used": False,
        "partial_reused_rows": 0,
        "heavy_work_started": False,
    }
    (WORKING / "DINOV2_STEP0_PASS.json").write_bytes(canonical_bytes(step0))
    install_offline_wheels(config, resolved)
    source = extract_source(Path(resolved["source_tar"]))
    runtime = {
        "config": config,
        "resolved": resolved,
        "source": str(source),
        "run_started_epoch": run_started_epoch,
    }
    runtime_path = WORKING / "DINOV2_RUNTIME_CONFIG.json"
    runtime_path.write_bytes(canonical_bytes(runtime))

    output = WORKING / "dinov2_r5_features"
    output.mkdir(exist_ok=False)
    class_manifests = {}
    status = "COMPLETE"
    error = None
    try:
        for action_class in ("delete", "occupied_swap"):
            class_output = output / action_class
            class_output.mkdir()
            class_manifests[action_class] = run_worker(
                action_class,
                runtime_path,
                class_output,
                hard_deadline_epoch,
            )
    except Exception as caught:
        status = "FAIL_CLOSED"
        error = {"type": type(caught).__name__, "message": str(caught)}

    elapsed = time.time() - run_started_epoch
    terminal = {
        "schema": "celltrack-alpha.dinov2-kaggle-gpu-terminal.v1",
        "status": status,
        "error": error,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "hard_limit_seconds": config["limits"]["hard_seconds"],
        "within_hard_limit": elapsed < config["limits"]["hard_seconds"],
        "step0_sha256": sha256(WORKING / "DINOV2_STEP0_PASS.json"),
        "inventory_sha256": sha256(inventory_path),
        "class_manifests": {
            key: {
                "status": value["status"],
                "file_count": value["file_count"],
                "action_count": value["action_count"],
                "manifest_sha256": sha256(output / key / "class_manifest.json"),
            }
            for key, value in class_manifests.items()
        },
        "complete_class_count": len(class_manifests),
        "partial_reused_rows": 0,
        "full_pool_cache": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
        "barrier": "HOLD_FOR_LOCAL_FROZEN_GATE_EVALUATION",
    }
    (WORKING / "DINOV2_GPU_TERMINAL.json").write_bytes(canonical_bytes(terminal))
    print(json.dumps(terminal, sort_keys=True), flush=True)
    if status != "COMPLETE":
        return 2
    return 0


def exact_uint16_quantile(np, histogram, quantile: float) -> float:
    histogram = np.asarray(histogram, dtype=np.int64)
    if histogram.shape != (65536,) or np.any(histogram < 0):
        raise RuntimeContractError("invalid uint16 histogram")
    count = int(histogram.sum())
    if count <= 0 or not 0.0 <= quantile <= 1.0:
        raise RuntimeContractError("invalid quantile request")
    position = (count - 1) * float(quantile)
    low_rank = int(math.floor(position))
    high_rank = int(math.ceil(position))
    cumulative = np.cumsum(histogram)
    low_value = int(np.searchsorted(cumulative, low_rank + 1, side="left"))
    high_value = int(np.searchsorted(cumulative, high_rank + 1, side="left"))
    return float(low_value + (position - low_rank) * (high_value - low_value))


def video_quantiles(np, array) -> tuple[float, float]:
    if str(array.dtype) != "uint16" or tuple(array.shape[1:]) != (64, 256, 256):
        raise RuntimeContractError(
            f"raw array contract drift: shape={array.shape}, dtype={array.dtype}"
        )
    histogram = np.zeros(65536, dtype=np.int64)
    for frame_index in range(int(array.shape[0])):
        frame = np.asarray(array[frame_index])
        histogram += np.bincount(frame.reshape(-1), minlength=65536)
    low = exact_uint16_quantile(np, histogram, 0.01)
    high = exact_uint16_quantile(np, histogram, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeContractError("degenerate video normalization")
    return low, high


def pseudo_rgb_batches(np, frame, coordinates_zyx, low, high, batch_size):
    z_offsets = np.arange(-16, 17, dtype=np.float32) * 0.25
    yx_offsets = np.arange(-16, 17, dtype=np.int64)
    frame = np.asarray(frame)
    coordinates = np.asarray(coordinates_zyx, dtype=np.int64)
    if frame.shape != (64, 256, 256) or frame.dtype != np.uint16:
        raise RuntimeContractError("raw frame contract drift")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise RuntimeContractError("endpoint-coordinate contract drift")
    for offset in range(0, len(coordinates), batch_size):
        stop = min(offset + batch_size, len(coordinates))
        batch = coordinates[offset:stop]
        z_position = batch[:, 0, None].astype(np.float32) + z_offsets[None, :]
        z_floor = np.floor(z_position).astype(np.int64)
        z_weight = (z_position - z_floor).astype(np.float32)
        z_ceil = z_floor + (z_weight > 0).astype(np.int64)
        z_floor = np.clip(z_floor, 0, frame.shape[0] - 1)
        z_ceil = np.clip(z_ceil, 0, frame.shape[0] - 1)
        y_index = np.clip(
            batch[:, 1, None] + yx_offsets[None, :], 0, frame.shape[1] - 1
        )
        x_index = np.clip(
            batch[:, 2, None] + yx_offsets[None, :], 0, frame.shape[2] - 1
        )
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
            raise RuntimeContractError("invalid pseudo-RGB crop")
        yield slice(offset, stop), pseudo


def load_request(np, path: Path) -> dict:
    required = {
        "source_id",
        "target_id",
        "source_t",
        "target_t",
        "source_zyx",
        "target_zyx",
    }
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != required:
            raise RuntimeContractError(f"request field drift: {path}")
        request = {key: loaded[key] for key in loaded.files}
    count = len(request["source_id"])
    if any(len(request[key]) != count for key in required):
        raise RuntimeContractError(f"request length drift: {path}")
    return request


def endpoint_rows(np, request: dict, role: str):
    return np.column_stack(
        (
            request[f"{role}_t"].astype(np.int64),
            request[f"{role}_zyx"].astype(np.int64),
        )
    )


def structured_rows(np, rows):
    contiguous = np.ascontiguousarray(rows, dtype=np.int64)
    return contiguous.view(
        np.dtype([("t", "<i8"), ("z", "<i8"), ("y", "<i8"), ("x", "<i8")])
    ).reshape(-1)


def map_rows(np, unique_rows, rows):
    unique_structured = structured_rows(np, unique_rows)
    rows_structured = structured_rows(np, rows)
    indices = np.searchsorted(unique_structured, rows_structured)
    if np.any(indices >= len(unique_rows)) or not np.array_equal(
        unique_structured[indices], rows_structured
    ):
        raise RuntimeContractError("endpoint mapping is incomplete")
    return indices


def action_features(np, source, target):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 768:
        raise RuntimeContractError("DINOv2 endpoint feature shape drift")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RuntimeContractError("nonfinite endpoint embedding")
    difference = source - target
    source_norm = np.linalg.norm(source, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    norm_scale = np.sqrt(768.0)
    cosine_denominator = np.maximum(source_norm * target_norm, 1e-8)
    centered_source = source - source.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    correlation_denominator = np.maximum(
        np.linalg.norm(centered_source, axis=1)
        * np.linalg.norm(centered_target, axis=1),
        1e-8,
    )
    result = np.stack(
        (
            np.sum(source * target, axis=1) / cosine_denominator,
            np.linalg.norm(difference, axis=1) / norm_scale,
            np.mean(np.abs(difference), axis=1),
            np.mean(source * target, axis=1),
            source_norm / norm_scale,
            target_norm / norm_scale,
            (source_norm - target_norm) / norm_scale,
            np.sum(centered_source * centered_target, axis=1)
            / correlation_denominator,
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(source), 8) or not np.isfinite(result).all():
        raise RuntimeContractError("invalid action features")
    return result


def within_frame_permutation(
    np,
    values,
    node_ids,
    *,
    identity: str,
    frame: int,
    supplier_contract_sha256: str,
):
    values = np.asarray(values, dtype=np.float32)
    node_ids = np.asarray(node_ids)
    if values.ndim != 2 or values.shape[1] != 768:
        raise RuntimeContractError("embedding shape drift")
    if node_ids.ndim != 1 or len(node_ids) != len(values):
        raise RuntimeContractError("node-id alignment drift")
    unique_ids, first = np.unique(node_ids, return_index=True)
    vectors = values[first]
    seed = (
        "R-SEPARATION-FALSIFIER-44B6"
        f"{supplier_contract_sha256}{identity}{int(frame)}"
    )
    order = sorted(
        range(len(unique_ids)),
        key=lambda index: (
            hashlib.sha256(
                f"{seed}{int(unique_ids[index])}".encode("ascii")
            ).digest(),
            int(unique_ids[index]),
        ),
    )
    replacement = {
        int(unique_ids[index]): vectors[order[(position + 1) % len(order)]]
        for position, index in enumerate(order)
    }
    if not order:
        return np.empty_like(values)
    return np.stack([replacement[int(node)] for node in node_ids]).astype(np.float32)


def permuted_features(np, request, source, target, identity, supplier_contract_sha):
    a1 = action_features(np, source, target)
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
            output[index] = within_frame_permutation(
                np,
                values[index],
                node_ids[index],
                identity=identity,
                frame=int(frame),
                supplier_contract_sha256=supplier_contract_sha,
            )
    a0 = action_features(np, permuted_source, permuted_target)
    return a1, a0


def load_model(torch, source: Path, checkpoint: Path, device):
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14

    state_dict = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise RuntimeContractError("checkpoint is not a tensor-only state dict")
    model = dinov2_vits14(pretrained=False)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeContractError("strict DINOv2 load failed")
    del state_dict
    model.eval().requires_grad_(False).to(device)
    return model


def embed_video(
    np,
    torch,
    functional,
    model,
    device,
    array,
    unique_rows,
    *,
    crop_batch: int,
    forward_batch: int,
    soft_deadline_epoch: float,
):
    low, high = video_quantiles(np, array)
    embeddings = np.empty((len(unique_rows), 768), dtype=np.float32)
    written = np.zeros(len(unique_rows), dtype=bool)
    mean = torch.tensor(
        (0.485, 0.456, 0.406), dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225), dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    for frame_index in np.unique(unique_rows[:, 0]):
        if time.time() >= soft_deadline_epoch:
            raise RuntimeContractError("five-hour soft-stop approached")
        index = np.flatnonzero(unique_rows[:, 0] == frame_index)
        frame = np.asarray(array[int(frame_index)])
        for local_slice, pseudo in pseudo_rgb_batches(
            np, frame, unique_rows[index, 1:], low, high, crop_batch
        ):
            for forward_offset in range(0, len(pseudo), forward_batch):
                forward_stop = min(forward_offset + forward_batch, len(pseudo))
                tensor = torch.from_numpy(
                    pseudo[forward_offset:forward_stop]
                ).to(device)
                tensor = functional.interpolate(
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
                    :, (119, 120, 135, 136)
                ].mean(dim=1)
                values = torch.cat((class_token, patch_token), dim=1).float().cpu().numpy()
                destination = index[
                    local_slice.start
                    + forward_offset : local_slice.start
                    + forward_stop
                ]
                embeddings[destination] = values
                written[destination] = True
                del tensor, result, class_token, patch_token, values
        del frame
    if not written.all() or not np.isfinite(embeddings).all():
        raise RuntimeContractError("endpoint embedding coverage is incomplete")
    return embeddings, (low, high)


def worker_main(args: argparse.Namespace) -> int:
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import zarr
    from zarr.storage import LocalStore

    runtime = json.loads(args.runtime.read_text(encoding="ascii"))
    config = runtime["config"]
    resolved = runtime["resolved"]
    if args.worker not in ("delete", "occupied_swap"):
        raise RuntimeContractError("unknown action class")
    if not torch.cuda.is_available():
        raise RuntimeContractError("CUDA unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = load_model(
        torch,
        Path(runtime["source"]),
        Path(resolved["weight"]),
        device,
    )
    rows = [
        row for row in config["requests"] if row["action_class"] == args.worker
    ]
    if not rows:
        raise RuntimeContractError("empty class request plan")
    total_endpoints = config["class_unique_endpoint_counts"][args.worker]
    expected_actions = config["class_action_counts"][args.worker]
    completed_endpoints = 0
    completed_actions = 0
    files = []
    started = time.monotonic()
    projection = None
    soft_deadline_epoch = (
        args.deadline_epoch
        - (config["limits"]["hard_seconds"] - config["limits"]["soft_seconds"])
    )
    for ordinal, row in enumerate(rows, start=1):
        identity = row["identity"]
        request_path = Path(resolved["requests"][row["sha256"]])
        if sha256(request_path) != row["sha256"]:
            raise RuntimeContractError("resolved request changed after step 0")
        request = load_request(np, request_path)
        unique_rows = np.unique(
            np.concatenate(
                (
                    endpoint_rows(np, request, "source"),
                    endpoint_rows(np, request, "target"),
                )
            ),
            axis=0,
        )
        if len(unique_rows) != row["unique_endpoints"]:
            raise RuntimeContractError("request unique-endpoint count drift")
        dataset = identity.split(":", 1)[1]
        store = LocalStore(Path(resolved["competition_root"]), read_only=True)
        try:
            array = zarr.open_array(
                store=store,
                path=f"train/{dataset}.zarr/0",
                mode="r",
            )
            embeddings, quantiles = embed_video(
                np,
                torch,
                functional,
                model,
                device,
                array,
                unique_rows,
                crop_batch=64,
                forward_batch=64,
                soft_deadline_epoch=soft_deadline_epoch,
            )
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        source_index = map_rows(
            np, unique_rows, endpoint_rows(np, request, "source")
        )
        target_index = map_rows(
            np, unique_rows, endpoint_rows(np, request, "target")
        )
        a1, a0 = permuted_features(
            np,
            request,
            embeddings[source_index],
            embeddings[target_index],
            identity,
            config["pins"]["supplier_contract"],
        )
        destination = args.output / f"{identity.replace(':', '__', 1)}.npz"
        np.savez_compressed(destination, a1=a1, a0=a0)
        files.append(
            {
                "identity": identity,
                "action_class": args.worker,
                "count": int(len(a1)),
                "request_sha256": row["sha256"],
                "output": destination.name,
                "output_sha256": sha256(destination),
            }
        )
        completed_endpoints += len(unique_rows)
        completed_actions += len(a1)
        if ordinal == 2:
            elapsed = time.monotonic() - started
            projection = elapsed / completed_endpoints * total_endpoints
            event = {
                "event": "AUTHENTICATED_CLASS_PROJECTION",
                "action_class": args.worker,
                "completed_videos": ordinal,
                "completed_unique_endpoints": completed_endpoints,
                "total_unique_endpoints": total_endpoints,
                "elapsed_seconds": elapsed,
                "projected_class_seconds": projection,
                "remaining_hard_seconds": args.deadline_epoch - time.time(),
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            if projection >= args.deadline_epoch - time.time():
                raise RuntimeContractError("authenticated class projection exceeds hard limit")
        print(
            json.dumps(
                {
                    "event": "VIDEO_COMPLETE_AND_RELEASED",
                    "action_class": args.worker,
                    "ordinal": ordinal,
                    "video_count": len(rows),
                    "identity": identity,
                    "actions": len(a1),
                    "unique_endpoints": len(unique_rows),
                    "q01": quantiles[0],
                    "q99": quantiles[1],
                    "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            request,
            unique_rows,
            embeddings,
            source_index,
            target_index,
            a1,
            a0,
            array,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if completed_endpoints != total_endpoints or completed_actions != expected_actions:
        raise RuntimeContractError("terminal class coverage drift")
    elapsed = time.monotonic() - started
    manifest = {
        "schema": "celltrack-alpha.dinov2-kaggle-gpu-class.v1",
        "status": "COMPLETE",
        "supplier_id": config["supplier_id"],
        "action_class": args.worker,
        "amendment_sha256": config["pins"]["amendment"],
        "sealed_B0_sha256": config["pins"]["sealed_b0"],
        "population_manifest_sha256": config["pins"]["population_manifest"],
        "supplier_contract_sha256": config["pins"]["supplier_contract"],
        "source_revision": config["source"]["revision"],
        "source_tree": config["source"]["tree"],
        "weight_sha256": config["weight"]["sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "crop_batch": 64,
        "forward_batch": 64,
        "unique_endpoint_count": total_endpoints,
        "file_count": len(files),
        "action_count": completed_actions,
        "files": files,
        "elapsed_seconds": elapsed,
        "projection_seconds_after_two_videos": projection,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "per_video_load_release": True,
        "partial_reused_rows": 0,
        "full_pool_cache": False,
        "endpoint_cache_persisted": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
    }
    (args.output / "class_manifest.json").write_bytes(canonical_bytes(manifest))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("delete", "occupied_swap"))
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    if args.worker and (
        args.runtime is None or args.output is None or args.deadline_epoch is None
    ):
        parser.error("worker mode requires runtime, output, and deadline")
    return args


if __name__ == "__main__":
    # Parent mode calls parent_main without argument parsing so its first
    # runtime operation remains dump_mount_inventory. Worker mode is a later,
    # separately guarded subprocess entered only after step 0 has passed.
    parsed = parse_args() if "--worker" in sys.argv else argparse.Namespace(worker=None)
    try:
        if parsed.worker:
            raise SystemExit(worker_main(parsed))
        raise SystemExit(parent_main())
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL_CLOSED",
            "error_type": type(error).__name__,
            "error": str(error),
            "worker": parsed.worker,
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        if not parsed.worker:
            WORKING.mkdir(parents=True, exist_ok=True)
            (WORKING / "DINOV2_GPU_TERMINAL.json").write_bytes(
                canonical_bytes(failure)
            )
        raise SystemExit(2)
