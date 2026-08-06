#!/usr/bin/env python3
"""Build the private stride-(1,4,4) Biohub training carrier for Colab.

The output contains competition *training* data and must remain private to
participants who accepted the competition rules.  It never reads the test split.
Raw uint16 intensities are retained after the exact V32 strided read; q0.001/q0.999
normalisation is deferred to the loader so its arithmetic remains reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid
import zipfile

import numpy as np
import zarr
from zarr.storage import ZipStore

try:
    from colab_division_harness import (
        FullStageLogger,
        HarnessError,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_idempotent_json,
    )
except ModuleNotFoundError:  # Supports imports as scripts.preprocess_division_colab_data.
    from scripts.colab_division_harness import (
        FullStageLogger,
        HarnessError,
        canonical_json_bytes,
        sha256_bytes,
        sha256_file,
        write_idempotent_json,
    )


SCHEMA = "celltrack-alpha.colab-division-preprocess.v1"
VIDEO_SCHEMA = "celltrack-alpha.colab-division-video.v1"
DEFAULT_ARCHIVE_SHA256 = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
DEFAULT_ARCHIVE_BYTES = 87_393_127_165
DEFAULT_ARCHIVE = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_OUTPUT = Path("data/derived/division_temporal_colab_v1")
FRAME_FILE = "frames_stride_1_4_4.npy"
LINEAGE_FILE = "lineage.npz"
RECEIPT_FILE = "RECEIPT.json"
CONTRACT_FILE = "CONTRACT.json"
MANIFEST_FILE = "DATASET_MANIFEST.json"
READY_FILE = "READY.json"
LOG_FILE = "logs/PREPROCESS_LOG.jsonl"


class PreprocessError(HarnessError):
    """The source or derived carrier violated its frozen contract."""


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    """Durably write an unpublished file before its directory is renamed."""
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class PreprocessConfig:
    split: str = "train"
    downsample_zyx: tuple[int, int, int] = (1, 4, 4)
    expected_dataset_count: int = 199
    expected_frames_per_dataset: int = 100
    expected_raw_shape_zyx: tuple[int, int, int] = (64, 256, 256)
    expected_output_shape_zyx: tuple[int, int, int] = (64, 64, 64)
    expected_total_parent_events: int = 151
    expected_total_division_edges: int = 302
    output_dtype: str = "uint16"
    normalization: str = "(raw-q0.001)/(q0.999-q0.001+1e-6), clamp_min_0, cast_fp16_at_load"
    parent_definition: str = "source node with outdegree >=2"
    daughter_assignment_scope: str = "labels_carried_only; existing conditional pair head is not rebuilt"


def _dataset_names_from_infos(infos: list[zipfile.ZipInfo], split: str) -> list[str]:
    prefix = f"{split}/"
    names = {
        info.filename.split("/")[1][:-5]
        for info in infos
        if info.filename.startswith(prefix)
        and len(info.filename.split("/")) >= 3
        and info.filename.split("/")[1].endswith(".zarr")
    }
    return sorted(names)


def _member_record(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "path": info.filename,
        "bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "crc32": f"{info.CRC:08x}",
        "compression": info.compress_type,
    }


def inventory_archive(
    archive: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    split: str,
) -> tuple[dict[str, Any], list[str], dict[str, list[dict[str, Any]]]]:
    if not archive.is_file() or archive.stat().st_size != expected_bytes:
        raise PreprocessError("competition archive is absent or byte size drifted")
    observed_sha = sha256_file(archive)
    if observed_sha != expected_sha256:
        raise PreprocessError("competition archive SHA-256 drifted")
    with zipfile.ZipFile(archive, "r") as handle:
        infos = handle.infolist()
    datasets = _dataset_names_from_infos(infos, split)
    records = [_member_record(info) for info in infos]
    per_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        prefixes = (f"{split}/{dataset}.zarr/", f"{split}/{dataset}.geff/")
        per_dataset[dataset] = [
            record for record in records if record["path"].startswith(prefixes)
        ]
    central_sha = sha256_bytes(canonical_json_bytes(records))
    train_records = [row for row in records if row["path"].startswith(f"{split}/")]
    return (
        {
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "sha256": observed_sha,
            "entry_count": len(records),
            "central_inventory_sha256": central_sha,
            "selected_split": split,
            "selected_member_count": len(train_records),
            "selected_member_inventory_sha256": sha256_bytes(
                canonical_json_bytes(train_records)
            ),
            "test_members_read": 0,
        },
        datasets,
        per_dataset,
    )


def _array_digest(arrays: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, raw in arrays:
        value = np.ascontiguousarray(raw)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Create an NPZ with fixed member order and timestamps."""
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.save(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _spatial_scale(attrs: dict[str, Any]) -> tuple[float, float, float]:
    try:
        multiscales = attrs["multiscales"]
        axes = multiscales[0]["axes"]
        values = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        by_name = {str(axis["name"]).lower(): float(values[index]) for index, axis in enumerate(axes)}
        return by_name["z"], by_name["y"], by_name["x"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PreprocessError("OME-Zarr spatial scale metadata is malformed") from exc


def _quantiles(attrs: dict[str, Any]) -> tuple[float, float]:
    try:
        quantiles = attrs["image_statistics"]["quantiles"]
        return float(quantiles["0.001"]), float(quantiles["0.999"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessError("OME-Zarr q0.001/q0.999 metadata is malformed") from exc


def _load_lineage(store: ZipStore, split: str, dataset: str) -> dict[str, np.ndarray]:
    base = f"{split}/{dataset}.geff"
    node_id = np.asarray(zarr.open_array(store=store, path=f"{base}/nodes/ids", mode="r")[:])
    props = {
        axis: np.asarray(
            zarr.open_array(
                store=store, path=f"{base}/nodes/props/{axis}/values", mode="r"
            )[:]
        )
        for axis in ("t", "z", "y", "x")
    }
    edges = np.asarray(zarr.open_array(store=store, path=f"{base}/edges/ids", mode="r")[:])
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise PreprocessError(f"{dataset}: GEFF edges are not (N,2)")
    if any(len(values) != len(node_id) for values in props.values()):
        raise PreprocessError(f"{dataset}: GEFF node-property lengths drifted")
    return {"node_id": node_id, "edges": edges, **props}


def _division_arrays(lineage: dict[str, np.ndarray], downsample: tuple[int, int, int]) -> dict[str, np.ndarray]:
    node_id = lineage["node_id"]
    edges = lineage["edges"]
    sources, counts = np.unique(edges[:, 0], return_counts=True)
    parent_ids = sources[counts >= 2]
    index = {int(value): position for position, value in enumerate(node_id)}
    try:
        parent_indices = np.asarray([index[int(value)] for value in parent_ids], dtype=np.int64)
    except KeyError as exc:
        raise PreprocessError("division parent edge references an absent node") from exc
    coords = np.column_stack([lineage[axis] for axis in ("z", "y", "x")]).astype(
        np.int64, copy=False
    )
    ds = np.asarray(downsample, dtype=np.float32)
    return {
        "division_parent_node_id": parent_ids,
        "division_parent_t": lineage["t"][parent_indices],
        "division_parent_coords_zyx": coords[parent_indices],
        "division_parent_coords_ds_zyx": coords[parent_indices].astype(np.float32) / ds,
    }


def _verify_video(root: Path, *, expected_contract_sha256: str | None = None) -> dict[str, Any]:
    receipt_path = root / RECEIPT_FILE
    try:
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessError(f"cannot read video receipt: {root}") from exc
    if receipt.get("schema") != VIDEO_SCHEMA:
        raise PreprocessError(f"video receipt schema drifted: {root.name}")
    if expected_contract_sha256 is not None and receipt.get("contract_sha256") != expected_contract_sha256:
        raise PreprocessError(f"video contract binding drifted: {root.name}")
    expected_files = {FRAME_FILE, LINEAGE_FILE, RECEIPT_FILE}
    if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
        raise PreprocessError(f"video has undeclared or missing files: {root.name}")
    for key, filename in (("frames", FRAME_FILE), ("lineage", LINEAGE_FILE)):
        record = receipt["outputs"][key]
        path = root / filename
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise PreprocessError(f"video output hash drifted: {root.name}/{filename}")
    return receipt


def process_dataset(
    *,
    store: ZipStore,
    dataset: str,
    output_root: Path,
    config: PreprocessConfig,
    contract_sha256: str,
    member_records: list[dict[str, Any]],
    logger: FullStageLogger,
) -> tuple[dict[str, Any], bool]:
    final = output_root / "videos" / dataset
    if final.exists():
        receipt = _verify_video(final, expected_contract_sha256=contract_sha256)
        logger.emit(
            "dataset_resume_verified",
            dataset=dataset,
            examined_count=1,
            modified_count=0,
            counts={"examined": 1, "modified": 0, "resumed": 1},
        )
        return receipt, True

    staging = output_root / "videos" / f".{dataset}.partial-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    image_path = f"{config.split}/{dataset}.zarr/0"
    group_path = f"{config.split}/{dataset}.zarr"
    image = zarr.open_array(store=store, path=image_path, mode="r")
    group = zarr.open_group(store=store, path=group_path, mode="r")
    raw_shape = tuple(int(value) for value in image.shape)
    expected_raw = (config.expected_frames_per_dataset, *config.expected_raw_shape_zyx)
    if raw_shape != expected_raw or str(image.dtype) != config.output_dtype:
        raise PreprocessError(
            f"{dataset}: expected raw {expected_raw}/{config.output_dtype}, observed {raw_shape}/{image.dtype}"
        )
    dz, dy, dx = config.downsample_zyx
    output_shape = (
        raw_shape[0],
        (raw_shape[1] + dz - 1) // dz,
        (raw_shape[2] + dy - 1) // dy,
        (raw_shape[3] + dx - 1) // dx,
    )
    if output_shape[1:] != config.expected_output_shape_zyx:
        raise PreprocessError(f"{dataset}: downsampled image shape drifted")
    raw_voxels = int(np.prod(raw_shape))
    output_voxels = int(np.prod(output_shape))
    source_digest = hashlib.sha256()
    source_digest.update(canonical_json_bytes({"shape": raw_shape, "dtype": str(image.dtype)}))
    frame_path = staging / FRAME_FILE
    with logger.stage(
        "raw_downsample",
        counts={
            "examined": raw_shape[0],
            "modified": raw_shape[0],
            "raw_voxels_examined": raw_voxels,
            "output_voxels_written": output_voxels,
        },
    ):
        output = np.lib.format.open_memmap(
            frame_path, mode="w+", dtype=image.dtype, shape=output_shape
        )
        for t in range(raw_shape[0]):
            raw = np.ascontiguousarray(image[t])
            source_digest.update(raw.tobytes(order="C"))
            output[t] = raw[::dz, ::dy, ::dx]
        output.flush()
        del output
        with frame_path.open("rb") as handle:
            os.fsync(handle.fileno())

    with logger.stage("lineage_export", counts={"examined": 1, "modified": 1}):
        lineage = _load_lineage(store, config.split, dataset)
        division = _division_arrays(lineage, config.downsample_zyx)
        coords = np.column_stack([lineage[axis] for axis in ("z", "y", "x")]).astype(
            np.int64, copy=False
        )
        arrays = {
            "node_id": lineage["node_id"],
            "t": lineage["t"],
            "coords_zyx": coords,
            "coords_ds_zyx": coords.astype(np.float32)
            / np.asarray(config.downsample_zyx, dtype=np.float32),
            "edges": lineage["edges"],
            **division,
        }
        lineage_bytes = _deterministic_npz(arrays)
        lineage_path = staging / LINEAGE_FILE
        _write_bytes_fsync(lineage_path, lineage_bytes)

    spatial_scale = _spatial_scale(dict(group.attrs))
    q_low, q_high = _quantiles(dict(group.attrs))
    source_graph_sha = _array_digest(
        [
            ("node_id", lineage["node_id"]),
            ("t", lineage["t"]),
            ("z", lineage["z"]),
            ("y", lineage["y"]),
            ("x", lineage["x"]),
            ("edges", lineage["edges"]),
        ]
    )
    source_member_sha = sha256_bytes(canonical_json_bytes(member_records))
    source_counts = {
        "frames": raw_shape[0],
        "raw_voxels": raw_voxels,
        "node_rows": len(lineage["node_id"]),
        "edge_rows": len(lineage["edges"]),
        "parent_events": len(division["division_parent_node_id"]),
        "division_edges": int(
            sum(
                count
                for count in np.unique(lineage["edges"][:, 0], return_counts=True)[1]
                if count >= 2
            )
        ),
    }
    modified_counts = {
        "datasets": 1,
        "frames_spatially_strided": raw_shape[0],
        "coordinates_transformed": len(lineage["node_id"]),
        "daughter_assignments_rebuilt": 0,
    }
    examined_counts = {
        "datasets": 1,
        "frames": raw_shape[0],
        "raw_voxels": raw_voxels,
        "node_rows": len(lineage["node_id"]),
        "edge_rows": len(lineage["edges"]),
    }
    receipt = {
        "schema": VIDEO_SCHEMA,
        "dataset": dataset,
        "contract_sha256": contract_sha256,
        "source": {
            "raw_shape_tzyx": list(raw_shape),
            "raw_dtype": str(image.dtype),
            "raw_voxel_sha256": source_digest.hexdigest(),
            "lineage_logical_sha256": source_graph_sha,
            "zip_member_count": len(member_records),
            "zip_member_inventory_sha256": source_member_sha,
            "q_low_0.001": q_low,
            "q_high_0.999": q_high,
            "voxel_size_zyx": list(spatial_scale),
        },
        "derived": {
            "shape_tzyx": list(output_shape),
            "dtype": str(image.dtype),
            "downsample_zyx": list(config.downsample_zyx),
            "voxel_size_zyx": [
                spatial_scale[index] * config.downsample_zyx[index]
                for index in range(3)
            ],
            "normalization": config.normalization,
        },
        "counts": source_counts,
        "examined_counts": examined_counts,
        "modified_counts": modified_counts,
        "outputs": {
            "frames": {
                "path": FRAME_FILE,
                "bytes": frame_path.stat().st_size,
                "sha256": sha256_file(frame_path),
            },
            "lineage": {
                "path": LINEAGE_FILE,
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_bytes_fsync(staging / RECEIPT_FILE, canonical_json_bytes(receipt))
    _ = _verify_video(staging, expected_contract_sha256=contract_sha256)
    _fsync_directory(staging)
    os.replace(staging, final)
    _fsync_directory(final.parent)
    logger.emit(
        "dataset_published",
        dataset=dataset,
        elapsed_seconds=receipt["elapsed_seconds"],
        examined_count=sum(examined_counts.values()),
        modified_count=sum(modified_counts.values()),
        examined_counts=examined_counts,
        modified_counts=modified_counts,
        parent_events=source_counts["parent_events"],
    )
    return receipt, False


def _contract(
    *, archive_inventory: dict[str, Any], config: PreprocessConfig
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "FROZEN_PREPROCESS_ONLY",
        "source": archive_inventory,
        "config": asdict(config),
        "actual_v32_input_resolution": {
            "raw_frame_shape_zyx": list(config.expected_raw_shape_zyx),
            "stride_zyx": list(config.downsample_zyx),
            "model_frame_shape_zyx": list(config.expected_output_shape_zyx),
            "note": "V32 does not stride Z; its public default is 1,4,4",
        },
        "data_security": {
            "contains_competition_training_data": True,
            "contains_competition_test_data": False,
            "public_upload_allowed": False,
            "allowed_transport": "private storage restricted to competition participants",
        },
        "scientific_scope": {
            "training_objective_frozen": False,
            "reason": "await APNCC-vs-V32 taxonomy",
            "daughter_assignment_head_rebuilt": False,
        },
    }


def _aggregate_counts(receipts: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("frames", "raw_voxels", "node_rows", "edge_rows", "parent_events", "division_edges")
    return {key: sum(int(receipt["counts"][key]) for receipt in receipts) for key in keys}


def verify_output(root: Path) -> dict[str, Any]:
    try:
        contract = json.loads((root / CONTRACT_FILE).read_text(encoding="ascii"))
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="ascii"))
        ready = json.loads((root / READY_FILE).read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessError("derived carrier is missing a valid root receipt") from exc
    contract_sha = sha256_file(root / CONTRACT_FILE)
    manifest_sha = sha256_file(root / MANIFEST_FILE)
    if (
        contract.get("schema") != SCHEMA
        or manifest.get("schema") != SCHEMA
        or ready.get("schema") != SCHEMA
        or manifest.get("contract_sha256") != contract_sha
        or ready.get("manifest_sha256") != manifest_sha
    ):
        raise PreprocessError("derived carrier root binding drifted")
    records = manifest.get("videos")
    if not isinstance(records, list):
        raise PreprocessError("derived carrier video manifest is malformed")
    observed: list[dict[str, Any]] = []
    for record in records:
        receipt = _verify_video(
            root / "videos" / str(record["dataset"]),
            expected_contract_sha256=contract_sha,
        )
        receipt_sha = sha256_file(root / "videos" / str(record["dataset"]) / RECEIPT_FILE)
        if receipt_sha != record.get("receipt_sha256"):
            raise PreprocessError(f"video receipt hash drifted: {record.get('dataset')}")
        observed.append(receipt)
    if _aggregate_counts(observed) != manifest.get("counts"):
        raise PreprocessError("derived carrier aggregate counts drifted")
    log_record = manifest.get("log", {})
    log_path = root / str(log_record.get("path", ""))
    if (
        not log_path.is_file()
        or sha256_file(log_path) != log_record.get("sha256_at_completion")
    ):
        raise PreprocessError("derived carrier completion log hash drifted")
    return {
        "status": "PASS",
        "contract_sha256": contract_sha,
        "manifest_sha256": manifest_sha,
        "video_count": len(observed),
        "counts": manifest["counts"],
        "payload_bytes": int(manifest["payload_bytes"]),
    }


def preprocess(
    *,
    archive: Path,
    output: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
    config: PreprocessConfig,
    resume: bool,
) -> dict[str, Any]:
    if output.exists() and not resume:
        raise PreprocessError("output exists; pass --resume to verify and continue")
    if (output / READY_FILE).exists():
        return verify_output(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    run_id = f"preprocess-{time.time_ns()}"
    logger = FullStageLogger(output / LOG_FILE, run_id=run_id, config=asdict(config))
    with logger.stage("input_inventory", counts={"examined": 1, "modified": 0}):
        archive_record, datasets, member_records = inventory_archive(
            archive,
            expected_sha256=expected_archive_sha256,
            expected_bytes=expected_archive_bytes,
            split=config.split,
        )
    if len(datasets) != config.expected_dataset_count:
        raise PreprocessError(
            f"expected {config.expected_dataset_count} training datasets, observed {len(datasets)}"
        )
    contract = _contract(archive_inventory=archive_record, config=config)
    write_idempotent_json(output / CONTRACT_FILE, contract)
    contract_sha = sha256_file(output / CONTRACT_FILE)
    receipts: list[dict[str, Any]] = []
    resumed_count = 0
    with ZipStore(archive, mode="r") as store:
        for ordinal, dataset in enumerate(datasets, start=1):
            logger.emit(
                "dataset_start",
                dataset=dataset,
                ordinal=ordinal,
                dataset_count=len(datasets),
                examined_count=1,
                modified_count=0,
            )
            receipt, resumed = process_dataset(
                store=store,
                dataset=dataset,
                output_root=output,
                config=config,
                contract_sha256=contract_sha,
                member_records=member_records[dataset],
                logger=logger,
            )
            receipts.append(receipt)
            resumed_count += int(resumed)
            print(
                f"[{ordinal:03d}/{len(datasets):03d}] {dataset} "
                f"{'resume-verified' if resumed else 'published'}",
                flush=True,
            )
    counts = _aggregate_counts(receipts)
    if (
        counts["parent_events"] != config.expected_total_parent_events
        or counts["division_edges"] != config.expected_total_division_edges
    ):
        raise PreprocessError("exact parent-event/division-edge census drifted")
    videos = [
        {
            "dataset": receipt["dataset"],
            "receipt_sha256": sha256_file(
                output / "videos" / receipt["dataset"] / RECEIPT_FILE
            ),
            "frame_sha256": receipt["outputs"]["frames"]["sha256"],
            "lineage_sha256": receipt["outputs"]["lineage"]["sha256"],
        }
        for receipt in receipts
    ]
    payload_bytes = sum(
        int(receipt["outputs"][kind]["bytes"])
        for receipt in receipts
        for kind in ("frames", "lineage")
    )
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE_PRIVATE_TRAINING_CARRIER",
        "contract_sha256": contract_sha,
        "videos": videos,
        "video_inventory_sha256": sha256_bytes(canonical_json_bytes(videos)),
        "counts": counts,
        "payload_bytes": payload_bytes,
        "image_payload_bytes_without_npy_headers": (
            counts["frames"]
            * int(np.prod(config.expected_output_shape_zyx))
            * np.dtype(config.output_dtype).itemsize
        ),
        "resumed_video_count": resumed_count,
        "modified_vs_examined": {
            "datasets_examined": len(datasets),
            "datasets_modified_this_run": len(datasets) - resumed_count,
            "datasets_resumed_verified": resumed_count,
            "frames_examined": counts["frames"],
            "frames_spatially_strided": counts["frames"],
            "daughter_assignments_rebuilt": 0,
        },
        "log": {"path": LOG_FILE},
    }
    logger.emit(
        "preprocess_payload_complete",
        examined_count=len(datasets),
        modified_count=len(datasets) - resumed_count,
        counts=counts,
        payload_bytes=payload_bytes,
    )
    manifest["log"]["sha256_at_completion"] = sha256_file(output / LOG_FILE)
    write_idempotent_json(output / MANIFEST_FILE, manifest)
    write_idempotent_json(
        output / READY_FILE,
        {
            "schema": SCHEMA,
            "status": "READY",
            "manifest_sha256": sha256_file(output / MANIFEST_FILE),
            "contract_sha256": contract_sha,
        },
    )
    return verify_output(output)


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers") from exc
    if len(result) != 3 or any(part <= 0 for part in result):
        raise argparse.ArgumentTypeError("shape must be three positive integers")
    return result  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--expected-archive-sha256", default=DEFAULT_ARCHIVE_SHA256)
    build.add_argument("--expected-archive-bytes", type=int, default=DEFAULT_ARCHIVE_BYTES)
    build.add_argument("--resume", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_output(args.output)
    else:
        result = preprocess(
            archive=args.archive,
            output=args.output,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_archive_bytes=args.expected_archive_bytes,
            config=PreprocessConfig(),
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
