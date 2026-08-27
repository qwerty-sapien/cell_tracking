#!/usr/bin/env python3
"""Generate frozen occupied-swap sliced-W1 features; never fit or score."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


BOOT_STARTED = time.monotonic()
DATASET_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/g02-occ-sliced-w1-input-v1"
)
WORK_ROOT = Path("/kaggle/working/occ_sliced_w1_v1")
WORK_ROOT.mkdir(parents=True, exist_ok=False)
RUN_LOG_PATH = WORK_ROOT / "RUN_LOG.jsonl"
OUTPUT_ROOT = WORK_ROOT / "features"

INPUT_MANIFEST_SHA256 = "806c7fd4198792b651b2c76536b26f0ae53195fa6ad5cc20252c636b465d3a94"
REQUEST_MANIFEST_SHA256 = "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
SUPPLIER_FREEZE_SHA256 = "2f7e6eb4d3d61904af3be2df1b015945647e56ef958d69e86da37840d1f5f8f5"
WHEEL_SHA256 = {
    "donfig-0.8.1.post1-py3-none-any.whl": "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d",
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3",
    "zarr-3.2.1-py3-none-any.whl": "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7",
}
EXPECTED_REQUEST_COUNT = 45_961
EXPECTED_REQUEST_FILE_COUNT = 36
EXPECTED_POPULATION_FILE_COUNT = 36
EXPECTED_WHEEL_COUNT = 3
SUPPLIER_FREEZE = {
    "schema": "celltrack-alpha.occ-sliced-w1-supplier-freeze.v1",
    "pilot_sha256": "7a5d1ce5d0391651292cb5cb4dd337e55959e74d871e46f5d9a1ef3cb4e610d6",
    "core_sha256": "cd62cdbe3ad4971d5aba8e4545b9e7674c0a36c950a92963a0f8ab8df182a379",
    "crop_radius_zyx": [3, 12, 12],
    "normalization_quantiles": [0.5, 0.99],
    "projection_bins": 25,
    "projection_count": 10,
    "output_mode": "features_only_no_submission",
}


class KernelError(RuntimeError):
    """The remote experiment violated a frozen invariant."""


def canonical_bytes(value: object) -> bytes:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log_lookup(stage: str, event: str, **fields: object) -> None:
    record = {
        "schema": "celltrack-alpha.run-log.v1",
        "elapsed_seconds": time.monotonic() - BOOT_STARTED,
        "stage": stage,
        "event": event,
        **fields,
    }
    with RUN_LOG_PATH.open("ab") as handle:
        handle.write(canonical_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


def _run_failure(stage: str, error: BaseException, **state: object) -> None:
    log_lookup(
        stage,
        "_run_failure",
        error_type=type(error).__name__,
        error=str(error),
        traceback=traceback.format_exc(),
        state=state,
    )


def resolve_declared_asset(record: object, label: str) -> Path:
    if not isinstance(record, dict):
        raise KernelError(f"{label}: malformed manifest record")
    relative = str(record.get("path", ""))
    if not relative or Path(relative).name != relative:
        raise KernelError(f"{label}: asset path is not a flat basename")
    path = DATASET_ROOT / relative
    if not path.is_file():
        raise KernelError(f"{label}: declared asset is absent")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise KernelError(f"{label}: byte-count drift")
    if sha256(path) != str(record.get("sha256", "")):
        raise KernelError(f"{label}: SHA-256 drift")
    return path


def run_step0() -> dict[str, object]:
    try:
        freeze_hash = hashlib.sha256(canonical_bytes(SUPPLIER_FREEZE)).hexdigest()
        if freeze_hash != SUPPLIER_FREEZE_SHA256:
            raise KernelError("embedded supplier freeze drift")
        input_manifest_path = DATASET_ROOT / "occ_sliced_w1_input_manifest.json"
        observed_manifest_sha = (
            sha256(input_manifest_path) if input_manifest_path.is_file() else None
        )
        log_lookup(
            "step0",
            "inventory_input",
            root=str(DATASET_ROOT),
            exists=input_manifest_path.is_file(),
            bytes=(input_manifest_path.stat().st_size if input_manifest_path.is_file() else None),
            observed_sha256=observed_manifest_sha,
            expected_sha256=INPUT_MANIFEST_SHA256,
        )
        if observed_manifest_sha != INPUT_MANIFEST_SHA256:
            raise KernelError("dedicated input manifest mismatch")
        bundle = json.loads(input_manifest_path.read_text(encoding="ascii"))
        if (
            bundle.get("schema") != "celltrack-alpha.occ-sliced-w1-input.v1"
            or bundle.get("status") != "IMMUTABLE_UPLOAD_READY"
            or bundle.get("dataset_id")
            != "onebitbrain/g02-occ-sliced-w1-input-v1"
            or bundle.get("dataset_version") != 1
        ):
            raise KernelError("dedicated input manifest semantic drift")
        request_manifest_path = resolve_declared_asset(
            bundle.get("source_request_manifest"), "request manifest"
        )
        population_manifest_path = resolve_declared_asset(
            bundle.get("source_population_manifest"), "population manifest"
        )
        if sha256(request_manifest_path) != REQUEST_MANIFEST_SHA256:
            raise KernelError("request manifest pin drift")
        if sha256(population_manifest_path) != POPULATION_MANIFEST_SHA256:
            raise KernelError("population manifest pin drift")
        request_manifest = json.loads(request_manifest_path.read_text(encoding="ascii"))
        population_manifest = json.loads(population_manifest_path.read_text(encoding="ascii"))
        if population_manifest.get("status") != "COMPLETE":
            raise KernelError("population status drift")
        source_rows = sorted(
            (
                {
                    "identity": str(row.get("identity")),
                    "action_count": int(row.get("action_count", -1)),
                    "bytes": int(row.get("bytes", -1)),
                    "path": str(row.get("path")),
                    "sha256": str(row.get("sha256")),
                }
                for row in request_manifest.get("requests", [])
                if row.get("action_class") == "occupied_swap"
            ),
            key=lambda row: row["identity"],
        )
        rows = bundle.get("requests", [])
        if (
            not isinstance(rows, list)
            or len(rows) != EXPECTED_REQUEST_FILE_COUNT
            or sum(int(row.get("action_count", -1)) for row in rows)
            != EXPECTED_REQUEST_COUNT
            or bundle.get("request_file_count") != EXPECTED_REQUEST_FILE_COUNT
            or bundle.get("request_count") != EXPECTED_REQUEST_COUNT
            or request_manifest.get("population_manifest_sha256")
            != POPULATION_MANIFEST_SHA256
            or rows != source_rows
        ):
            raise KernelError("occupied-swap request population drift")
        if (
            population_manifest.get("totals", {})
            .get("occupied_swap", {})
            .get("actions")
            != EXPECTED_REQUEST_COUNT
        ):
            raise KernelError("population manifest total drift")
        requests = []
        semantics: set[str] = set()
        for row in rows:
            identity = str(row.get("identity"))
            digest = str(row.get("sha256"))
            count = int(row.get("action_count", -1))
            if not identity.startswith("train:44b6_") or identity in semantics or count <= 0:
                raise KernelError("malformed or duplicate occupied-swap request row")
            semantics.add(identity)
            path = resolve_declared_asset(row, f"request {identity}")
            requests.append(
                {
                    "identity": identity,
                    "action_count": count,
                    "sha256": digest,
                    "path": str(path),
                }
            )
        requests.sort(key=lambda row: str(row["identity"]))
        wheel_rows = bundle.get("wheels", [])
        if not isinstance(wheel_rows, list):
            raise KernelError("wheel records malformed")
        wheel_records = {str(row.get("path")): row for row in wheel_rows}
        wheels = {}
        for name, digest in sorted(WHEEL_SHA256.items()):
            row = wheel_records.get(name)
            if not isinstance(row, dict) or row.get("sha256") != digest:
                raise KernelError(f"wheel manifest drift: {name}")
            wheels[name] = str(resolve_declared_asset(row, f"wheel {name}"))
        if (
            len(wheels) != EXPECTED_WHEEL_COUNT
            or bundle.get("wheel_count") != EXPECTED_WHEEL_COUNT
            or len(wheel_records) != EXPECTED_WHEEL_COUNT
        ):
            raise KernelError("wheel count drift")
        result = {
            "schema": "celltrack-alpha.occ-sliced-w1-step0.v1",
            "status": "PASS",
            "dataset_source": "onebitbrain/g02-occ-sliced-w1-input-v1",
            "dataset_version": 1,
            "input_manifest_sha256": INPUT_MANIFEST_SHA256,
            "checkpoint_sha256": SUPPLIER_FREEZE_SHA256,
            "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
            "request_manifest_sha256": REQUEST_MANIFEST_SHA256,
            "request_count": EXPECTED_REQUEST_COUNT,
            "request_file_count": EXPECTED_REQUEST_FILE_COUNT,
            "population_file_count": EXPECTED_POPULATION_FILE_COUNT,
            "wheel_count": EXPECTED_WHEEL_COUNT,
            "enable_gpu": False,
            "heavy_work_started": False,
            "competition_path_visits": 0,
            "requests": requests,
            "wheels": wheels,
        }
        log_lookup("step0", "stage_end", **result)
        return result
    except BaseException as error:
        _run_failure("step0", error, heavy_work_started=False)
        raise


STEP0 = run_step0()


if importlib.util.find_spec("zarr") is None:
    wheel_paths = [str(STEP0["wheels"][name]) for name in sorted(WHEEL_SHA256)]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", *wheel_paths],
        check=True,
    )

import numpy as np
import zarr


COMPETITION_ROOT = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development"
)
CROP_RADIUS_ZYX = (3, 12, 12)
TOTAL_ACTIONS = 45_961
PROJECTION_LIMIT_SECONDS = 3600.0
SOFT_STOP_SECONDS = 3500.0
CHUNK = 96
PROJECTION_DIRECTIONS = np.asarray(
    (
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, -1.0),
    ),
    dtype=np.float64,
)
PROJECTION_DIRECTIONS /= np.linalg.norm(PROJECTION_DIRECTIONS, axis=1, keepdims=True)
FEATURE_NAMES = (
    "wasserstein_z",
    "wasserstein_y",
    "wasserstein_x",
    "wasserstein_radial",
    "wasserstein_proj_z_plus_y",
    "wasserstein_proj_z_minus_y",
    "wasserstein_proj_z_plus_x",
    "wasserstein_proj_z_minus_x",
    "wasserstein_proj_y_plus_x",
    "wasserstein_proj_y_minus_x",
    "absolute_log_mass_ratio",
    "centroid_shift_normalized",
    "source_log_total_mass",
    "target_log_total_mass",
)


def extract_reflect_patch(
    array: zarr.Array, frame: int, center_zyx: np.ndarray
) -> np.ndarray:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 4 or frame < 0 or frame >= shape[0]:
        raise KernelError("invalid raw TZYX array or frame")
    center = np.asarray(center_zyx, dtype=np.int64)
    radius = np.asarray(CROP_RADIUS_ZYX, dtype=np.int64)
    starts = center - radius
    stops = center + radius + 1
    clipped_start = np.maximum(starts, 0)
    clipped_stop = np.minimum(stops, np.asarray(shape[1:], dtype=np.int64))
    if np.any(clipped_stop <= clipped_start):
        raise KernelError("endpoint crop misses raw volume")
    patch = np.asarray(
        array[
            frame,
            int(clipped_start[0]) : int(clipped_stop[0]),
            int(clipped_start[1]) : int(clipped_stop[1]),
            int(clipped_start[2]) : int(clipped_stop[2]),
        ],
        dtype=np.float32,
    )
    padding = tuple(
        (int(lo), int(hi))
        for lo, hi in zip(
            (starts * -1).clip(min=0),
            (stops - np.asarray(shape[1:], dtype=np.int64)).clip(min=0),
            strict=True,
        )
    )
    if any(lo or hi for lo, hi in padding):
        mode = "reflect" if all(size > 1 for size in patch.shape) else "edge"
        patch = np.pad(patch, padding, mode=mode)
    if patch.shape != (7, 25, 25) or not np.isfinite(patch).all():
        raise KernelError("invalid reflected crop")
    return patch


def normalized_mass(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(patches, dtype=np.float32).reshape(len(patches), -1)
    background = np.quantile(flat, 0.50, axis=1, keepdims=True)
    high = np.quantile(flat, 0.99, axis=1, keepdims=True)
    foreground = np.clip(
        (flat - background) / np.maximum(high - background, 1.0e-6), 0.0, 1.0
    ).astype(np.float32, copy=False)
    total = foreground.sum(axis=1)
    blank = total <= 1.0e-8
    if blank.any():
        foreground[blank] = 1.0
        total[blank] = foreground.shape[1]
    mass = foreground / foreground.sum(axis=1, keepdims=True)
    return mass.astype(np.float32), total.astype(np.float32)


def histogram_w1(first: np.ndarray, second: np.ndarray, spacing: float) -> np.ndarray:
    return np.sum(np.abs(np.cumsum(first - second, axis=1)), axis=1) * spacing


def binned_mass(
    mass: np.ndarray, values: np.ndarray, bins: int, low: float, high: float
) -> np.ndarray:
    indices = np.floor((values - low) / (high - low) * bins).astype(np.int64)
    indices = np.clip(indices, 0, bins - 1)
    output = np.zeros((len(mass), bins), dtype=np.float64)
    for column in range(mass.shape[1]):
        output[:, indices[column]] += mass[:, column]
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1.0e-12)
    return output


def sliced_features(source_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
    source_mass, source_total = normalized_mass(source_patches)
    target_mass, target_total = normalized_mass(target_patches)
    shape = (7, 25, 25)
    grid = np.stack(np.unravel_index(np.arange(source_mass.shape[1]), shape), axis=1).astype(np.float64)
    radius = np.asarray(CROP_RADIUS_ZYX, dtype=np.float64)
    coordinates = (grid - radius) / radius
    columns = []
    source_grid = source_mass.reshape((len(source_mass), *shape))
    target_grid = target_mass.reshape((len(target_mass), *shape))
    for axis, size in enumerate(shape):
        reduce_axes = tuple(index + 1 for index in range(3) if index != axis)
        columns.append(
            histogram_w1(
                source_grid.sum(axis=reduce_axes),
                target_grid.sum(axis=reduce_axes),
                2.0 / (size - 1),
            )
        )
    radial = np.linalg.norm(coordinates, axis=1)
    radial_high = float(np.sqrt(3.0))
    columns.append(
        histogram_w1(
            binned_mass(source_mass, radial, 25, 0.0, radial_high),
            binned_mass(target_mass, radial, 25, 0.0, radial_high),
            radial_high / 24.0,
        )
    )
    projection_high = float(np.sqrt(2.0))
    for direction in PROJECTION_DIRECTIONS:
        projection = coordinates @ direction
        columns.append(
            histogram_w1(
                binned_mass(source_mass, projection, 25, -projection_high, projection_high),
                binned_mass(target_mass, projection, 25, -projection_high, projection_high),
                2.0 * projection_high / 24.0,
            )
        )
    source_centroid = source_mass @ coordinates
    target_centroid = target_mass @ coordinates
    columns.extend(
        (
            np.abs(np.log1p(source_total) - np.log1p(target_total)),
            np.linalg.norm(source_centroid - target_centroid, axis=1),
            np.log1p(source_total),
            np.log1p(target_total),
        )
    )
    output = np.stack(columns, axis=1).astype(np.float32)
    if output.shape != (len(source_patches), len(FEATURE_NAMES)) or not np.isfinite(output).all():
        raise KernelError("malformed sliced-W1 feature output")
    return output


def permutation_indices(
    node_ids: np.ndarray, frames: np.ndarray, identity: str
) -> np.ndarray:
    replacement = np.empty(len(node_ids), dtype=np.int64)
    for frame in np.unique(frames):
        rows = np.flatnonzero(frames == frame)
        unique_ids, first = np.unique(node_ids[rows], return_index=True)
        first_rows = rows[first]
        order = sorted(
            range(len(unique_ids)),
            key=lambda index: (
                hashlib.sha256(
                    f"OCC-SLICED-W1-A0-V1{identity}{int(frame)}{int(unique_ids[index])}".encode("ascii")
                ).digest(),
                int(unique_ids[index]),
            ),
        )
        row_by_id = {}
        for position, unique_index in enumerate(order):
            next_index = order[(position + 1) % len(order)]
            row_by_id[int(unique_ids[unique_index])] = int(first_rows[next_index])
        replacement[rows] = [row_by_id[int(node_ids[row])] for row in rows]
    return replacement


def load_request(path: Path, expected_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        request = {key: loaded[key] for key in loaded.files}
    required = {
        "source_id",
        "target_id",
        "source_t",
        "target_t",
        "source_zyx",
        "target_zyx",
    }
    if set(request) != required or len(request["source_id"]) != expected_count:
        raise KernelError("request schema/count drift")
    if not np.array_equal(request["target_t"], request["source_t"] + 1):
        raise KernelError("request contains non-adjacent transition")
    return request


def video_features(
    array: zarr.Array, request: dict[str, np.ndarray], identity: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    count = len(request["source_id"])
    source = np.stack(
        [
            extract_reflect_patch(
                array, int(request["source_t"][index]), request["source_zyx"][index]
            )
            for index in range(count)
        ]
    )
    target = np.stack(
        [
            extract_reflect_patch(
                array, int(request["target_t"][index]), request["target_zyx"][index]
            )
            for index in range(count)
        ]
    )
    a1 = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
    a0 = np.empty_like(a1)
    permutation = permutation_indices(request["target_id"], request["target_t"], identity)
    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        a1[start:stop] = sliced_features(source[start:stop], target[start:stop])
        a0[start:stop] = sliced_features(source[start:stop], target[permutation[start:stop]])
    if not np.isfinite(a1).all() or not np.isfinite(a0).all():
        raise KernelError("feature archive contains non-finite values")
    moved = int(np.count_nonzero(permutation != np.arange(count)))
    return a1, a0, {"actions": count, "endpoint_crops": 2 * count, "a0_rows_moved": moved}


def main() -> int:
    feature_started = time.monotonic()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    files = []
    preprocess_counts = {"actions": 0, "endpoint_crops": 0, "a0_rows_moved": 0}
    for ordinal, record in enumerate(STEP0["requests"], start=1):
        if time.monotonic() - feature_started >= SOFT_STOP_SECONDS:
            raise KernelError("bounded runtime soft stop reached")
        identity = str(record["identity"])
        dataset = identity.split(":", 1)[1]
        raw = COMPETITION_ROOT / "train" / f"{dataset}.zarr" / "0"
        if not raw.is_dir():
            raise KernelError(f"missing exact competition zarr: {dataset}")
        array = zarr.open_array(str(raw), mode="r")
        request_path = Path(str(record["path"]))
        if sha256(request_path) != str(record["sha256"]):
            raise KernelError(f"request changed after step0: {identity}")
        request = load_request(request_path, int(record["action_count"]))
        a1, a0, counts = video_features(array, request, identity)
        for key, value in counts.items():
            preprocess_counts[key] += value
        destination = OUTPUT_ROOT / (identity.replace(":", "__", 1) + ".npz")
        np.savez_compressed(destination, a1=a1, a0=a0)
        files.append(
            {
                "identity": identity,
                "actions": len(a1),
                "request_sha256": record["sha256"],
                "output": destination.name,
                "output_sha256": sha256(destination),
                **counts,
            }
        )
        elapsed = time.monotonic() - feature_started
        projection = elapsed / preprocess_counts["actions"] * TOTAL_ACTIONS
        log_lookup(
            "features",
            "stage_end",
            ordinal=ordinal,
            identities=EXPECTED_REQUEST_FILE_COUNT,
            identity=identity,
            projection_seconds=projection,
            preprocess_counts=dict(preprocess_counts),
        )
        if preprocess_counts["actions"] >= 256 and projection > PROJECTION_LIMIT_SECONDS:
            raise KernelError(
                f"worker projection exceeds 1.0 h: {projection / 3600.0:.6f}"
            )
    if preprocess_counts["actions"] != TOTAL_ACTIONS:
        raise KernelError("completed action count drift")
    manifest = {
        "schema": "celltrack-alpha.occ-sliced-w1-features.v1",
        "status": "COMPLETE",
        "supplier_freeze_sha256": SUPPLIER_FREEZE_SHA256,
        "request_manifest_sha256": REQUEST_MANIFEST_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "parameters": SUPPLIER_FREEZE,
        "preprocess_counts": preprocess_counts,
        "feature_elapsed_seconds": time.monotonic() - feature_started,
        "files": files,
        "selection_authorized": False,
        "graph_authorized": False,
        "competition_artifact_authorized": False,
    }
    manifest_path = WORK_ROOT / "FEATURE_MANIFEST.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    log_lookup(
        "complete",
        "stage_end",
        status="COMPLETE",
        manifest_sha256=sha256(manifest_path),
        preprocess_counts=preprocess_counts,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        _run_failure("runtime", error, preprocess_counts="see prior RUN_LOG.jsonl")
        raise
