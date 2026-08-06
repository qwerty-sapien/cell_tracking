#!/usr/bin/env python3
"""Materialize the preregistered G-03 dense-motion separation features.

This supplier is intentionally limited to raw 44b6 frame pairs and the banked
Phase-2 action requests.  It never reads oracle labels and never runs a graph.
The unchanged frozen separation evaluator consumes the emitted A1/A0 arrays.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import map_coordinates, shift
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "reports/g02/screening/tiered_surface/branches/g03_orthogonal_lanes_20260803/lane1_dense_motion"
PREREG = LANE / "PREREGISTRATION.json"
PREREG_SHA256 = "05146b759a7147e12b890b2ad8fea4bfee4a42819e054e387a285ab8c4e7c4ca"
SYNTHETIC_AMENDMENT = LANE / "SYNTHETIC_ONLY_AMENDMENT.json"
SYNTHETIC_AMENDMENT_SHA256 = "6cf60c56b8866faec2809f40cf46123c3e155886beb34e1b6cb752f820a20538"
POPULATION_ROOT = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/run_v1"
POPULATION_MANIFEST = POPULATION_ROOT / "population_manifest.json"
POPULATION_MANIFEST_SHA256 = "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
REQUEST_MANIFEST = ROOT / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate/ACTION_REQUEST_MANIFEST_V2.json"
REQUEST_MANIFEST_SHA256 = "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
RAW_ARCHIVE_SHA256 = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
RAW_ARCHIVE_BYTES = 87_393_127_165
AMENDMENT_SHA256 = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
CLASSES = ("delete", "occupied_swap")
COARSE_SHAPE = (16, 64, 64)
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
FEATURE_NAMES = (
    "predicted_dz_um",
    "predicted_dy_um",
    "predicted_dx_um",
    "observed_minus_predicted_dz_um",
    "observed_minus_predicted_dy_um",
    "observed_minus_predicted_dx_um",
    "residual_norm_um",
    "cosine_predicted_observed",
)
A0_SEED = "G03-DENSE-MOTION-A0-V1"


class DenseMotionError(RuntimeError):
    """The supplier cannot proceed without violating its frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise DenseMotionError(f"{label} SHA-256 mismatch: {path}")


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
        raise DenseMotionError(f"refusing to overwrite {path}")
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
        raise DenseMotionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DenseMotionError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DenseMotionError(f"expected JSON object: {path}")
    return value


def verify_static_inputs(*, verify_raw_digest: bool) -> dict[str, Any]:
    require_hash(PREREG, PREREG_SHA256, "lane preregistration")
    require_hash(
        SYNTHETIC_AMENDMENT,
        SYNTHETIC_AMENDMENT_SHA256,
        "synthetic-only amendment",
    )
    require_hash(POPULATION_MANIFEST, POPULATION_MANIFEST_SHA256, "population manifest")
    require_hash(REQUEST_MANIFEST, REQUEST_MANIFEST_SHA256, "request manifest")
    if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
        raise DenseMotionError("raw archive byte count drifted")
    raw_digest = sha256(RAW_ARCHIVE) if verify_raw_digest else None
    if raw_digest is not None and raw_digest != RAW_ARCHIVE_SHA256:
        raise DenseMotionError("raw archive SHA-256 drifted")
    requests = load_json(REQUEST_MANIFEST)
    if (
        requests.get("request_count") != 1_391_458
        or requests.get("request_file_count") != 77
        or requests.get("classes", {}).get("delete", {}).get("request_count") != 1_345_497
        or requests.get("classes", {}).get("occupied_swap", {}).get("request_count") != 45_961
    ):
        raise DenseMotionError("request manifest population counts drifted")
    population = load_json(POPULATION_MANIFEST)
    totals = population.get("totals", {})
    if (
        totals.get("delete", {}).get("actions") != 1_345_497
        or totals.get("delete", {}).get("beneficial") != 1_118
        or totals.get("occupied_swap", {}).get("actions") != 45_961
        or totals.get("occupied_swap", {}).get("beneficial") != 134
    ):
        raise DenseMotionError("population manifest action counts drifted")
    return {
        "preregistration_sha256": PREREG_SHA256,
        "synthetic_amendment_sha256": SYNTHETIC_AMENDMENT_SHA256,
        "population_manifest_sha256": POPULATION_MANIFEST_SHA256,
        "request_manifest_sha256": REQUEST_MANIFEST_SHA256,
        "raw_archive_bytes": RAW_ARCHIVE_BYTES,
        "raw_archive_sha256": raw_digest,
    }


def coarse_pair(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != moving.shape or reference.ndim != 3:
        raise DenseMotionError("frame-pair shapes differ or are not 3-D")
    values = []
    for volume in (reference, moving):
        values.append(
            resize(
                np.asarray(volume),
                COARSE_SHAPE,
                order=1,
                mode="edge",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32, copy=False)
        )
    joined = np.concatenate(tuple(value.ravel() for value in values))
    low, high = np.percentile(joined, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        raise DenseMotionError("nonfinite frame-pair percentile")
    if high - low < 1.0e-6:
        return np.zeros(COARSE_SHAPE, dtype=np.float32), np.zeros(
            COARSE_SHAPE, dtype=np.float32
        )
    normalized = tuple(
        np.clip((value - low) / (high - low), 0.0, 1.0).astype(
            np.float32, copy=False
        )
        for value in values
    )
    return normalized[0], normalized[1]


def dense_field(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    first, second = coarse_pair(reference, moving)
    flow = optical_flow_tvl1(
        first,
        second,
        attachment=75.0,
        tightness=0.3,
        num_warp=5,
        num_iter=15,
        tol=1.0e-4,
        prefilter=True,
        dtype=np.float32,
    )
    value = np.asarray(flow, dtype=np.float32)
    if value.shape != (3,) + COARSE_SHAPE or not np.isfinite(value).all():
        raise DenseMotionError("TV-L1 emitted malformed field")
    return value


def sample_native_flow(
    flow: np.ndarray, source_zyx: np.ndarray, native_shape: tuple[int, int, int]
) -> np.ndarray:
    source = np.asarray(source_zyx, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise DenseMotionError("source coordinates must be [N,3]")
    native = np.asarray(native_shape, dtype=np.float64)
    coarse = np.asarray(COARSE_SHAPE, dtype=np.float64)
    coordinate_scale = (coarse - 1.0) / (native - 1.0)
    vector_scale = (native - 1.0) / (coarse - 1.0)
    sample_coordinates = (source * coordinate_scale[None, :]).T
    sampled = np.stack(
        [
            map_coordinates(
                flow[component],
                sample_coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            * vector_scale[component]
            for component in range(3)
        ],
        axis=1,
    )
    if not np.isfinite(sampled).all():
        raise DenseMotionError("nonfinite sampled native flow")
    return sampled


def action_features(
    flow: np.ndarray,
    source_zyx: np.ndarray,
    target_zyx: np.ndarray,
    native_shape: tuple[int, int, int],
) -> np.ndarray:
    predicted_native = sample_native_flow(flow, source_zyx, native_shape)
    predicted = predicted_native * SCALE_ZYX_UM[None, :]
    observed = (
        np.asarray(target_zyx, dtype=np.float64)
        - np.asarray(source_zyx, dtype=np.float64)
    ) * SCALE_ZYX_UM[None, :]
    residual = observed - predicted
    residual_norm = np.linalg.norm(residual, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(observed, axis=1)
    cosine = np.divide(
        np.sum(predicted * observed, axis=1),
        denominator,
        out=np.zeros(len(observed), dtype=np.float64),
        where=denominator > 1.0e-12,
    )
    result = np.concatenate(
        (predicted, residual, residual_norm[:, None], cosine[:, None]), axis=1
    ).astype(np.float32)
    if result.shape != (len(source_zyx), 8) or not np.isfinite(result).all():
        raise DenseMotionError("malformed action feature matrix")
    return result


def permutation_key(
    action_class: str,
    identity: str,
    frame: int,
    row: int,
    source_id: int,
    target_id: int,
) -> bytes:
    value = (
        f"{A0_SEED}|{action_class}|{identity}|{frame}|{row}|"
        f"{source_id}|{target_id}"
    ).encode("ascii")
    return hashlib.sha256(value).digest()


def permuted_control(
    a1: np.ndarray,
    *,
    action_class: str,
    identity: str,
    source_t: np.ndarray,
    source_id: np.ndarray,
    target_id: np.ndarray,
) -> np.ndarray:
    a0 = np.empty_like(a1)
    for frame in sorted(np.unique(source_t).tolist()):
        rows = np.flatnonzero(source_t == frame)
        if len(rows) == 1:
            a0[rows] = a1[rows]
            continue
        ordered = sorted(
            rows.tolist(),
            key=lambda row: permutation_key(
                action_class,
                identity,
                int(frame),
                int(row),
                int(source_id[row]),
                int(target_id[row]),
            ),
        )
        destination = np.asarray(ordered, dtype=np.int64)
        source = np.roll(destination, 1)
        a0[destination] = a1[source]
    if not np.isfinite(a0).all():
        raise DenseMotionError("permuted control contains nonfinite values")
    return a0


def synthetic_preflight() -> dict[str, Any]:
    shape = COARSE_SHAPE
    grid = np.indices(shape, dtype=np.float32)
    center = np.asarray((8.0, 31.0, 32.0), dtype=np.float32)
    squared = sum(((grid[index] - center[index]) / (2.5, 7.0, 7.0)[index]) ** 2 for index in range(3))
    reference = np.exp(-0.5 * squared).astype(np.float32)
    translations = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, -2.0, 0.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, -2.0),
        (0.0, 2.0, -2.0),
    )
    z, y, x = (slice(4, 13), slice(18, 46), slice(18, 46))
    weight = reference[z, y, x]
    rows = []
    spatial_stds = []
    for translation in translations:
        expected = np.asarray(translation, dtype=np.float64)
        moving = shift(
            reference,
            shift=translation,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        flow = dense_field(reference, moving)
        observed = np.asarray(
            [
                np.sum(flow[component, z, y, x] * weight) / np.sum(weight)
                for component in range(3)
            ]
        )
        rows.append(
            {
                "expected_coarse_flow_zyx": expected.tolist(),
                "observed_weighted_flow_zyx": observed.tolist(),
                "endpoint_error_coarse_voxels": float(
                    np.linalg.norm(observed - expected)
                ),
            }
        )
        spatial_stds.append([float(np.std(flow[index])) for index in range(3)])
    error = max(float(row["endpoint_error_coarse_voxels"]) for row in rows)
    result = {
        "schema": "celltrack-alpha.g03-dense-motion-synthetic-preflight.v1",
        "status": "PASS" if error <= 1.0 else "FAIL",
        "translations": rows,
        "maximum_endpoint_error_coarse_voxels": error,
        "maximum_error": 1.0,
        "maximum_flow_spatial_std": np.max(spatial_stds, axis=0).tolist(),
        "spatially_varying": bool(np.max(spatial_stds) > 1.0e-4),
    }
    if result["status"] != "PASS" or not result["spatially_varying"]:
        raise DenseMotionError(f"synthetic field preflight failed: {result}")
    return result


def request_paths() -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {action_class: {} for action_class in CLASSES}
    for action_class in CLASSES:
        for path in sorted((POPULATION_ROOT / "feature_requests" / action_class).glob("*.npz")):
            identity = path.stem.replace("__", ":", 1)
            result[action_class][identity] = path
    if len(result["delete"]) != 41 or len(result["occupied_swap"]) != 36:
        raise DenseMotionError("request-file identity coverage drifted")
    if not set(result["occupied_swap"]) <= set(result["delete"]):
        raise DenseMotionError("occupied-swap identity is absent from delete requests")
    return result


def load_request(path: Path) -> dict[str, np.ndarray]:
    required = (
        "source_id",
        "target_id",
        "source_t",
        "target_t",
        "source_zyx",
        "target_zyx",
    )
    with np.load(path, allow_pickle=False) as value:
        if tuple(value.files) != required:
            raise DenseMotionError(f"request schema drifted: {path}")
        result = {key: np.asarray(value[key]) for key in required}
    count = len(result["source_id"])
    if any(len(result[key]) != count for key in required):
        raise DenseMotionError(f"request lengths differ: {path}")
    if not np.all(result["target_t"] == result["source_t"] + 1):
        raise DenseMotionError(f"non-adjacent request: {path}")
    return result


def run_extraction(output: Path, *, verify_raw_digest: bool) -> dict[str, Any]:
    if output.exists():
        raise DenseMotionError(f"refusing to reuse output root: {output}")
    static = verify_static_inputs(verify_raw_digest=verify_raw_digest)
    synthetic = synthetic_preflight()
    paths = request_paths()
    output.mkdir(parents=True)
    for action_class in CLASSES:
        (output / "features" / action_class).mkdir(parents=True)
    started = time.perf_counter()
    file_rows: list[dict[str, Any]] = []
    field_pair_count = 0
    action_count = 0
    identity_rows: list[dict[str, Any]] = []
    with ZipStore(RAW_ARCHIVE, mode="r") as store:
        for ordinal, identity in enumerate(sorted(paths["delete"]), start=1):
            identity_started = time.perf_counter()
            dataset = identity.split(":", 1)[1]
            image = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            if image.ndim != 4:
                raise DenseMotionError(f"raw image is not TZYX: {identity}")
            native_shape = tuple(int(value) for value in image.shape[1:])
            loaded: dict[str, dict[str, np.ndarray]] = {}
            rows_by_frame: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
            for action_class in CLASSES:
                path = paths[action_class].get(identity)
                if path is None:
                    continue
                request = load_request(path)
                loaded[action_class] = request
                for frame in np.unique(request["source_t"]):
                    rows_by_frame[int(frame)].append(
                        (action_class, np.flatnonzero(request["source_t"] == frame))
                    )
            a1_by_class = {
                action_class: np.empty((len(request["source_id"]), 8), dtype=np.float32)
                for action_class, request in loaded.items()
            }
            field_stds = []
            for frame in sorted(rows_by_frame):
                if frame < 0 or frame + 1 >= image.shape[0]:
                    raise DenseMotionError(f"request frame outside raw image: {identity} t={frame}")
                flow = dense_field(np.asarray(image[frame]), np.asarray(image[frame + 1]))
                field_pair_count += 1
                field_stds.append([float(np.std(flow[index])) for index in range(3)])
                for action_class, rows in rows_by_frame[frame]:
                    request = loaded[action_class]
                    a1_by_class[action_class][rows] = action_features(
                        flow,
                        request["source_zyx"][rows],
                        request["target_zyx"][rows],
                        native_shape,
                    )
            for action_class, request in loaded.items():
                a1 = a1_by_class[action_class]
                a0 = permuted_control(
                    a1,
                    action_class=action_class,
                    identity=identity,
                    source_t=request["source_t"],
                    source_id=request["source_id"],
                    target_id=request["target_id"],
                )
                target = output / "features" / action_class / paths[action_class][identity].name
                digest = atomic_npz(target, a1=a1, a0=a0)
                action_count += len(a1)
                file_rows.append(
                    {
                        "action_class": action_class,
                        "identity": identity,
                        "actions": len(a1),
                        "path": str(target),
                        "sha256": digest,
                        "bytes": target.stat().st_size,
                    }
                )
            elapsed = time.perf_counter() - identity_started
            identity_row = {
                "ordinal": ordinal,
                "identity": identity,
                "frame_pairs": len(rows_by_frame),
                "native_shape_zyx": list(native_shape),
                "mean_field_component_std": np.mean(field_stds, axis=0).tolist(),
                "elapsed_seconds": elapsed,
            }
            identity_rows.append(identity_row)
            print(json.dumps(identity_row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    if action_count != 1_391_458 or len(file_rows) != 77:
        raise DenseMotionError("terminal feature coverage mismatch")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    peak_rss_mb = peak_rss / (1024 * 1024) if os.uname().sysname == "Darwin" else peak_rss / 1024
    manifest = {
        "schema": "celltrack-alpha.g03-dense-motion-feature-manifest.v1",
        "status": "COMPLETE_SCORE_BLIND_FEATURES",
        "amendment_sha256": AMENDMENT_SHA256,
        "lane_preregistration_sha256": PREREG_SHA256,
        "synthetic_amendment_sha256": SYNTHETIC_AMENDMENT_SHA256,
        "static_inputs": static,
        "synthetic_preflight": synthetic,
        "feature_names": list(FEATURE_NAMES),
        "a0_seed": A0_SEED,
        "classes": {
            action_class: {
                "files": sum(row["action_class"] == action_class for row in file_rows),
                "actions": sum(
                    int(row["actions"])
                    for row in file_rows
                    if row["action_class"] == action_class
                ),
            }
            for action_class in CLASSES
        },
        "aggregate": {
            "identity_count": len(identity_rows),
            "feature_file_count": len(file_rows),
            "action_count": action_count,
            "field_pair_count": field_pair_count,
            "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss_mb,
        },
        "files": file_rows,
        "identities": identity_rows,
        "score_blind": True,
        "oracle_label_access": False,
        "graph": False,
        "gpu_hours": 0.0,
        "cv57": False,
        "submission": False,
        "runpod": False,
    }
    atomic_json(output / "FEATURE_MANIFEST.json", manifest)
    print(json.dumps(manifest["aggregate"], sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--verify-raw-digest", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--verify-raw-digest", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        static = verify_static_inputs(verify_raw_digest=args.verify_raw_digest)
        result = {
            "schema": "celltrack-alpha.g03-dense-motion-preflight.v1",
            "status": "PASS",
            "static_inputs": static,
            "synthetic": synthetic_preflight(),
            "versions": {
                "numpy": np.__version__,
                "zarr": zarr.__version__,
            },
        }
        atomic_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    run_extraction(args.output, verify_raw_digest=args.verify_raw_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
