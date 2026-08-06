#!/usr/bin/env python3
"""Authenticate Jose Freitas's ``biohub_synthetic`` notebook payload.

The authenticator is intentionally read-only with respect to the payload.  Image
arrays are never materialized: their NPY headers are checked directly inside the
NPZ container.  Only the comparatively small centroid, graph, and voxel arrays
are loaded for structural validation.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


REPORT_SCHEMA = "celltrack-alpha.synth-div-payload-authentication.v1"
SOURCE_ID = "josefreitasalvesneto-biohub-synthetic-dataset-v3"
DEFAULT_STATIC_COUNT = 1_539
DEFAULT_SEQUENCE_COUNT = 2_174
DEFAULT_TOTAL_NODES = 4_056_226
DEFAULT_TOTAL_DIVISIONS = 165_267
DEFAULT_MAX_RUNTIME_SECONDS = 3_600.0
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SMALL_ARRAY_BYTES = 64 * 1024 * 1024
STATIC_MEMBERS = frozenset({"volume", "centroids", "voxel_um"})
SEQUENCE_MEMBERS = frozenset(
    {"volumes", "nodes", "edges", "divisions", "voxel_um_pooled"}
)
OPTIONAL_GENERATOR_FILES = frozenset({"progress.log"})


class AuthenticationError(RuntimeError):
    """The payload cannot be authenticated."""


@dataclass(frozen=True)
class ExpectedCounts:
    static: int = DEFAULT_STATIC_COUNT
    sequences: int = DEFAULT_SEQUENCE_COUNT
    nodes: int = DEFAULT_TOTAL_NODES
    divisions: int = DEFAULT_TOTAL_DIVISIONS

    def validate(self) -> None:
        for label, value in (
            ("static", self.static),
            ("sequences", self.sequences),
            ("nodes", self.nodes),
            ("divisions", self.divisions),
        ):
            _require(type(value) is int and value >= 0, f"expected {label} must be a non-negative integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "static": self.static,
            "sequences": self.sequences,
            "nodes": self.nodes,
            "divisions": self.divisions,
        }


class _Deadline:
    def __init__(self, seconds: float) -> None:
        _require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) > 0,
            "max runtime must be a finite positive number of seconds",
        )
        self.seconds = float(seconds)
        self._end = time.monotonic() + self.seconds

    def check(self, operation: str) -> None:
        if time.monotonic() > self._end:
            raise AuthenticationError(
                f"runtime cap of {self.seconds:g} seconds exceeded during {operation}"
            )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthenticationError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(actual == expected, f"{label} keys drifted: expected {sorted(expected)}, observed {sorted(actual)}")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_int(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def _safe_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value != "", f"{label} must be a non-empty path string")
    _require("\\" not in value and "\x00" not in value, f"{label} is not a canonical POSIX path")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and value == relative.as_posix()
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} is not a safe canonical relative path: {value!r}",
    )
    _require(
        all(ord(character) >= 32 and ord(character) != 127 for character in value),
        f"{label} contains a control character",
    )
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthenticationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AuthenticationError(f"JSON contains non-finite number {value}")


def _load_json(path: Path, label: str, deadline: _Deadline) -> dict[str, Any]:
    deadline.check(f"reading {label}")
    _require(path.is_file() and not path.is_symlink(), f"missing regular {label}: {path.name}")
    size = path.stat().st_size
    _require(0 < size <= MAX_JSON_BYTES, f"{label} size is invalid: {size} bytes")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except AuthenticationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthenticationError(f"invalid {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _validate_metadata_schema(metadata: Mapping[str, Any], expected: ExpectedCounts) -> dict[str, Any]:
    keys = {
        "n_static",
        "n_sequences",
        "total_gb",
        "seq_len",
        "total_divisions",
        "total_nodes",
        "division_rate",
        "voxel_native_um",
        "pooling",
        "real_count_range",
        "shell_kind",
        "motion_calibration",
        "note_division_rate",
        "config",
    }
    _exact_keys(metadata, keys, "metadata.json")
    _require(_nonnegative_int(metadata["n_static"], "metadata n_static") == expected.static, "metadata static count drifted")
    _require(
        _nonnegative_int(metadata["n_sequences"], "metadata n_sequences") == expected.sequences,
        "metadata sequence count drifted",
    )
    _require(
        _nonnegative_int(metadata["total_nodes"], "metadata total_nodes") == expected.nodes,
        "metadata total node count drifted",
    )
    _require(
        _nonnegative_int(metadata["total_divisions"], "metadata total_divisions")
        == expected.divisions,
        "metadata total division count drifted",
    )
    seq_len = _positive_int(metadata["seq_len"], "metadata seq_len")
    _require(_is_number(metadata["total_gb"]) and float(metadata["total_gb"]) >= 0, "metadata total_gb is invalid")
    _require(_is_number(metadata["division_rate"]) and 0 <= float(metadata["division_rate"]) <= 1, "metadata division_rate is invalid")
    expected_rate = expected.divisions / max(expected.nodes, 1)
    _require(
        math.isclose(float(metadata["division_rate"]), expected_rate, rel_tol=1e-15, abs_tol=1e-15),
        "metadata division_rate does not match total_divisions / total_nodes",
    )

    voxel = metadata["voxel_native_um"]
    _require(
        isinstance(voxel, list)
        and len(voxel) == 3
        and all(_is_number(item) and float(item) > 0 for item in voxel),
        "metadata voxel_native_um must contain three finite positive numbers",
    )
    real_range = metadata["real_count_range"]
    _require(
        isinstance(real_range, list)
        and len(real_range) == 2
        and all(_is_number(item) and float(item) >= 0 for item in real_range)
        and float(real_range[0]) <= float(real_range[1]),
        "metadata real_count_range is invalid",
    )
    _require(isinstance(metadata["pooling"], str) and metadata["pooling"], "metadata pooling is invalid")
    _require(metadata["shell_kind"] is None or isinstance(metadata["shell_kind"], str), "metadata shell_kind is invalid")
    _require(
        isinstance(metadata["note_division_rate"], str) and metadata["note_division_rate"],
        "metadata note_division_rate is invalid",
    )
    motion = metadata["motion_calibration"]
    _require(isinstance(motion, dict), "metadata motion_calibration must be an object")
    _exact_keys(
        motion,
        {"median_step_um_per_frame", "lag1_persistence", "sister_separation_um"},
        "metadata motion_calibration",
    )
    _require(all(_is_number(value) for value in motion.values()), "metadata motion calibration contains an invalid number")
    config = metadata["config"]
    _require(
        isinstance(config, dict)
        and all(isinstance(key, str) and key.startswith("SYNTH_") for key in config)
        and all(isinstance(value, str) for value in config.values()),
        "metadata config must map SYNTH_* names to strings",
    )
    return {
        "seq_len": seq_len,
        "total_gb": float(metadata["total_gb"]),
        "voxel_native_um": [float(item) for item in voxel],
    }


def _validate_manifest_schema(
    manifest: Mapping[str, Any], expected: ExpectedCounts, seq_len: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _exact_keys(manifest, {"static", "sequences"}, "manifest.json")
    static_rows = manifest["static"]
    sequence_rows = manifest["sequences"]
    _require(isinstance(static_rows, list), "manifest static must be a list")
    _require(isinstance(sequence_rows, list), "manifest sequences must be a list")
    _require(len(static_rows) == expected.static, "manifest static count drifted")
    _require(len(sequence_rows) == expected.sequences, "manifest sequence count drifted")

    normalized_static: list[dict[str, Any]] = []
    for index, row in enumerate(static_rows):
        _require(isinstance(row, dict), f"manifest static row {index} must be an object")
        _exact_keys(row, {"file", "n_cells", "shape", "bytes"}, f"manifest static row {index}")
        path = _safe_relative(row["file"], f"manifest static row {index} file")
        _require(path == f"static/vol_{index:05d}.npz", f"manifest static path/order drifted at row {index}: {path}")
        shape = row["shape"]
        _require(
            isinstance(shape, list)
            and len(shape) == 3
            and all(type(item) is int and item > 0 for item in shape),
            f"manifest static row {index} shape is invalid",
        )
        normalized_static.append(
            {
                "file": path,
                "n_cells": _nonnegative_int(row["n_cells"], f"manifest static row {index} n_cells"),
                "shape": list(shape),
                "bytes": _positive_int(row["bytes"], f"manifest static row {index} bytes"),
            }
        )

    normalized_sequences: list[dict[str, Any]] = []
    for index, row in enumerate(sequence_rows):
        _require(isinstance(row, dict), f"manifest sequence row {index} must be an object")
        _exact_keys(
            row,
            {"file", "T", "n_nodes", "n_edges", "n_divisions", "bytes"},
            f"manifest sequence row {index}",
        )
        path = _safe_relative(row["file"], f"manifest sequence row {index} file")
        _require(path == f"sequences/seq_{index:04d}.npz", f"manifest sequence path/order drifted at row {index}: {path}")
        row_t = _positive_int(row["T"], f"manifest sequence row {index} T")
        _require(row_t == seq_len, f"manifest sequence row {index} T differs from metadata seq_len")
        normalized_sequences.append(
            {
                "file": path,
                "T": row_t,
                "n_nodes": _positive_int(row["n_nodes"], f"manifest sequence row {index} n_nodes"),
                "n_edges": _nonnegative_int(row["n_edges"], f"manifest sequence row {index} n_edges"),
                "n_divisions": _nonnegative_int(
                    row["n_divisions"], f"manifest sequence row {index} n_divisions"
                ),
                "bytes": _positive_int(row["bytes"], f"manifest sequence row {index} bytes"),
            }
        )
    return normalized_static, normalized_sequences


def _inventory_files(root: Path, deadline: _Deadline) -> list[Path]:
    _require(root.is_dir() and not root.is_symlink(), f"payload root is not a regular directory: {root}")
    paths: list[Path] = []
    try:
        for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            deadline.check("inventory traversal")
            dirnames.sort()
            filenames.sort()
            base = Path(directory)
            for name in dirnames:
                candidate = base / name
                mode = candidate.lstat().st_mode
                _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"payload contains non-directory or symlink: {candidate.relative_to(root)}")
            for name in filenames:
                candidate = base / name
                mode = candidate.lstat().st_mode
                _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"payload contains non-regular file or symlink: {candidate.relative_to(root)}")
                _safe_relative(candidate.relative_to(root).as_posix(), "payload member")
                paths.append(candidate)
    except OSError as error:
        raise AuthenticationError(f"cannot inventory payload: {error}") from error
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _sha256_file(path: Path, deadline: _Deadline) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                deadline.check(f"hashing {path.name}")
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise AuthenticationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_tree_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash sorted ``(path, file digest)`` records using the repository tree convention."""
    digest = hashlib.sha256()
    previous: str | None = None
    for row in records:
        path = _safe_relative(row.get("path"), "tree record path")
        _require(previous is None or previous < path, "tree records are not strictly path-sorted")
        file_sha = row.get("sha256")
        _require(
            isinstance(file_sha, str)
            and len(file_sha) == 64
            and all(character in "0123456789abcdef" for character in file_sha),
            f"tree record has invalid SHA-256: {path}",
        )
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
        previous = path
    return digest.hexdigest()


def _npz_infos(archive: zipfile.ZipFile, expected: frozenset[str], label: str) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    _require(all(not info.is_dir() for info in infos), f"{label} contains a directory member")
    names = [info.filename for info in infos]
    _require(len(names) == len(set(names)), f"{label} contains duplicate archive member names")
    expected_names = {f"{name}.npy" for name in expected}
    _require(set(names) == expected_names, f"{label} member names drifted: expected {sorted(expected_names)}, observed {sorted(names)}")
    for info in infos:
        _require(not (info.flag_bits & 0x1), f"{label} contains encrypted member {info.filename}")
        _require(info.file_size > 0, f"{label} contains empty member {info.filename}")
    return {info.filename[:-4]: info for info in infos}


@dataclass(frozen=True)
class _ArrayHeader:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool
    header_bytes: int
    array_bytes: int


def _member_header(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, label: str
) -> _ArrayHeader:
    try:
        with archive.open(info, "r") as handle:
            version = np.lib.format.read_magic(handle)
            _require(version in ((1, 0), (2, 0), (3, 0)), f"{label} uses unsupported NPY version {version}")
            length_bytes = handle.read(2 if version == (1, 0) else 4)
            _require(
                len(length_bytes) == (2 if version == (1, 0) else 4),
                f"{label} has a truncated NPY header length",
            )
            header_length = int.from_bytes(length_bytes, "little")
            _require(0 < header_length <= 10_000, f"{label} NPY header is too large")
            header_encoding = "utf-8" if version == (3, 0) else "latin1"
            header_raw = handle.read(header_length)
            _require(len(header_raw) == header_length, f"{label} has a truncated NPY header")
            header_value = ast.literal_eval(header_raw.decode(header_encoding).strip())
            _require(isinstance(header_value, dict), f"{label} NPY header is not a dictionary")
            _exact_keys(header_value, {"descr", "fortran_order", "shape"}, f"{label} NPY header")
            shape = header_value["shape"]
            fortran_order = header_value["fortran_order"]
            dtype = np.dtype(header_value["descr"])
            _require(type(fortran_order) is bool, f"{label} has invalid fortran_order")
            header_bytes = handle.tell()
    except AuthenticationError:
        raise
    except Exception as error:
        raise AuthenticationError(f"cannot read {label} NPY header: {error}") from error
    _require(isinstance(shape, tuple) and all(type(item) is int and item >= 0 for item in shape), f"{label} has invalid shape")
    dtype = np.dtype(dtype)
    _require(not dtype.hasobject, f"{label} has forbidden object dtype")
    array_bytes = math.prod(shape) * dtype.itemsize
    _require(
        info.file_size == header_bytes + array_bytes,
        f"{label} member byte length does not match its NPY header",
    )
    return _ArrayHeader(shape, dtype, bool(fortran_order), header_bytes, array_bytes)


def _load_member_array(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    header: _ArrayHeader,
    label: str,
    deadline: _Deadline,
) -> np.ndarray[Any, Any]:
    deadline.check(f"loading {label}")
    _require(header.array_bytes <= MAX_SMALL_ARRAY_BYTES, f"{label} exceeds the small-array safety cap")
    try:
        with archive.open(info, "r") as handle:
            array = np.lib.format.read_array(handle, allow_pickle=False)
    except Exception as error:
        raise AuthenticationError(f"cannot load {label}: {error}") from error
    _require(array.shape == header.shape and array.dtype == header.dtype, f"{label} changed between header and load")
    return array


def _stream_member_crc(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    label: str,
    deadline: _Deadline,
) -> None:
    """Read a member in bounded chunks so ``zipfile`` verifies its CRC."""
    observed = 0
    try:
        with archive.open(info, "r") as handle:
            while True:
                deadline.check(f"verifying {label} CRC")
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise AuthenticationError(f"cannot verify {label} CRC: {error}") from error
    _require(observed == info.file_size, f"{label} streamed byte count drifted")


def _require_dtype(header: _ArrayHeader, dtype: Any, label: str) -> None:
    _require(header.dtype == np.dtype(dtype), f"{label} dtype drifted: expected {np.dtype(dtype)}, observed {header.dtype}")


def _validate_static_npz(
    path: Path,
    row: Mapping[str, Any],
    native_voxel: Sequence[float],
    deadline: _Deadline,
) -> tuple[int, tuple[int, int, int]]:
    deadline.check(f"validating {row['file']}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = _npz_infos(archive, STATIC_MEMBERS, row["file"])
            volume_header = _member_header(archive, infos["volume"], f"{row['file']}:volume")
            centroid_header = _member_header(archive, infos["centroids"], f"{row['file']}:centroids")
            voxel_header = _member_header(archive, infos["voxel_um"], f"{row['file']}:voxel_um")
            _require_dtype(volume_header, np.uint16, f"{row['file']}:volume")
            _require_dtype(centroid_header, np.float32, f"{row['file']}:centroids")
            _require_dtype(voxel_header, np.float32, f"{row['file']}:voxel_um")
            expected_shape = tuple(row["shape"])
            _require(volume_header.shape == expected_shape, f"{row['file']} volume shape differs from manifest")
            _require(
                centroid_header.shape == (row["n_cells"], 3),
                f"{row['file']} centroid shape differs from manifest n_cells",
            )
            _require(voxel_header.shape == (3,), f"{row['file']} voxel_um shape drifted")
            _stream_member_crc(
                archive, infos["volume"], f"{row['file']}:volume", deadline
            )
            centroids = _load_member_array(
                archive, infos["centroids"], centroid_header, f"{row['file']}:centroids", deadline
            )
            voxel = _load_member_array(
                archive, infos["voxel_um"], voxel_header, f"{row['file']}:voxel_um", deadline
            )
    except AuthenticationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise AuthenticationError(f"invalid static NPZ {row['file']}: {error}") from error
    _require(np.isfinite(centroids).all(), f"{row['file']} centroids contain non-finite values")
    if len(centroids):
        _require((centroids >= 0).all(), f"{row['file']} centroids contain negative coordinates")
        _require(
            all((centroids[:, axis] < expected_shape[axis]).all() for axis in range(3)),
            f"{row['file']} centroid lies outside the volume",
        )
    _require(
        np.array_equal(voxel.astype(np.float64), np.asarray(native_voxel, dtype=np.float32).astype(np.float64)),
        f"{row['file']} voxel_um differs from metadata",
    )
    return len(centroids), expected_shape


def _integer_column(values: np.ndarray[Any, Any], label: str) -> np.ndarray[Any, Any]:
    _require(np.isfinite(values).all(), f"{label} contains non-finite values")
    _require(np.equal(values, np.floor(values)).all(), f"{label} contains non-integer values")
    return values.astype(np.int64)


def _validate_sequence_structure(
    nodes: np.ndarray[Any, Any],
    edges: np.ndarray[Any, Any],
    divisions: np.ndarray[Any, Any],
    row: Mapping[str, Any],
) -> None:
    label = row["file"]
    node_count = len(nodes)
    times = _integer_column(nodes[:, 0], f"{label} node times")
    track_ids = _integer_column(nodes[:, 4], f"{label} track ids")
    _require((track_ids >= 0).all(), f"{label} contains negative track id")
    _require(np.isfinite(nodes[:, 1:4]).all(), f"{label} node coordinates contain non-finite values")
    _require((nodes[:, 1:4] >= 0).all(), f"{label} node coordinates contain negative values")
    _require((times >= 0).all() and (times < row["T"]).all(), f"{label} node time is out of range")
    _require(np.all(times[:-1] <= times[1:]), f"{label} nodes are not time-sorted")
    _require(np.array_equal(np.unique(times), np.arange(row["T"])), f"{label} does not cover every declared time")

    edge64 = edges.astype(np.int64, copy=False)
    _require((edge64 >= 0).all() and (edge64 < node_count).all(), f"{label} edge index is out of range")
    if len(edge64):
        _require((edge64[:, 0] != edge64[:, 1]).all(), f"{label} contains a self edge")
        _require(len(np.unique(edge64, axis=0)) == len(edge64), f"{label} contains duplicate edges")
        source = edge64[:, 0]
        target = edge64[:, 1]
        _require(np.equal(times[target], times[source] + 1).all(), f"{label} edge does not advance exactly one frame")
        _require(np.equal(track_ids[target], track_ids[source]).all(), f"{label} edge changes track id")
    else:
        source = np.empty(0, dtype=np.int64)
        target = np.empty(0, dtype=np.int64)

    indegree = np.bincount(target, minlength=node_count)
    outdegree = np.bincount(source, minlength=node_count)
    _require((indegree[times == 0] == 0).all(), f"{label} time-zero node has a parent")
    _require((indegree[times > 0] == 1).all(), f"{label} non-initial node does not have exactly one parent")
    _require((outdegree[times == row["T"] - 1] == 0).all(), f"{label} terminal node has an outgoing edge")
    _require(
        np.isin(outdegree[times < row["T"] - 1], (1, 2)).all(),
        f"{label} non-terminal node must have one continuation or two children",
    )
    observed_divisions = np.flatnonzero(outdegree == 2).astype(np.int64)
    division64 = divisions.astype(np.int64, copy=False)
    _require(
        len(division64) == len(np.unique(division64)),
        f"{label} divisions contains duplicate parent indices",
    )
    _require(
        np.array_equal(division64, np.sort(division64)),
        f"{label} divisions is not sorted",
    )
    _require(
        np.array_equal(division64, observed_divisions),
        f"{label} divisions does not exactly identify the outdegree-two parents",
    )


def _validate_sequence_npz(
    path: Path,
    row: Mapping[str, Any],
    expected_pooled_voxel: Sequence[float],
    deadline: _Deadline,
) -> tuple[int, int, int, tuple[int, ...], tuple[float, float, float]]:
    deadline.check(f"validating {row['file']}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = _npz_infos(archive, SEQUENCE_MEMBERS, row["file"])
            headers = {
                name: _member_header(archive, info, f"{row['file']}:{name}")
                for name, info in infos.items()
            }
            _require_dtype(headers["volumes"], np.uint16, f"{row['file']}:volumes")
            _require_dtype(headers["nodes"], np.float32, f"{row['file']}:nodes")
            _require_dtype(headers["edges"], np.int32, f"{row['file']}:edges")
            _require_dtype(headers["divisions"], np.int32, f"{row['file']}:divisions")
            _require_dtype(headers["voxel_um_pooled"], np.float32, f"{row['file']}:voxel_um_pooled")
            volume_shape = headers["volumes"].shape
            _require(len(volume_shape) == 4 and volume_shape[0] == row["T"] and all(item > 0 for item in volume_shape), f"{row['file']} volumes shape is invalid")
            _require(headers["nodes"].shape == (row["n_nodes"], 5), f"{row['file']} node shape differs from manifest")
            _require(headers["edges"].shape == (row["n_edges"], 2), f"{row['file']} edge shape differs from manifest")
            _require(headers["divisions"].shape == (row["n_divisions"],), f"{row['file']} division shape differs from manifest")
            _require(headers["voxel_um_pooled"].shape == (3,), f"{row['file']} voxel_um_pooled shape drifted")
            _stream_member_crc(
                archive, infos["volumes"], f"{row['file']}:volumes", deadline
            )
            arrays = {
                name: _load_member_array(
                    archive, infos[name], headers[name], f"{row['file']}:{name}", deadline
                )
                for name in ("nodes", "edges", "divisions", "voxel_um_pooled")
            }
    except AuthenticationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise AuthenticationError(f"invalid sequence NPZ {row['file']}: {error}") from error
    _validate_sequence_structure(
        arrays["nodes"], arrays["edges"], arrays["divisions"], row
    )
    voxel = arrays["voxel_um_pooled"]
    _require(np.isfinite(voxel).all() and (voxel > 0).all(), f"{row['file']} pooled voxel metadata is invalid")
    _require(
        np.array_equal(
            voxel.astype(np.float64),
            np.asarray(expected_pooled_voxel, dtype=np.float32).astype(np.float64),
        ),
        f"{row['file']} voxel_um_pooled differs from native voxel metadata and XY pooling",
    )
    return (
        len(arrays["nodes"]),
        len(arrays["edges"]),
        len(arrays["divisions"]),
        volume_shape,
        tuple(float(item) for item in voxel),
    )


def authenticate_payload(
    payload: Path,
    *,
    expected: ExpectedCounts = ExpectedCounts(),
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
) -> dict[str, Any]:
    """Return a deterministic authentication report or raise ``AuthenticationError``."""
    expected.validate()
    deadline = _Deadline(max_runtime_seconds)
    requested_root = Path(payload)
    _require(
        requested_root.is_dir() and not requested_root.is_symlink(),
        f"payload root is invalid: {requested_root}",
    )
    root = requested_root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), f"payload root is invalid: {root}")
    metadata = _load_json(root / "metadata.json", "metadata.json", deadline)
    manifest = _load_json(root / "manifest.json", "manifest.json", deadline)
    metadata_values = _validate_metadata_schema(metadata, expected)
    static_rows, sequence_rows = _validate_manifest_schema(
        manifest, expected, metadata_values["seq_len"]
    )

    manifest_paths = {row["file"] for row in [*static_rows, *sequence_rows]}
    required_paths = {"metadata.json", "manifest.json", *manifest_paths}
    inventory = _inventory_files(root, deadline)
    actual_paths = {path.relative_to(root).as_posix() for path in inventory}
    missing = sorted(required_paths - actual_paths)
    extra = sorted(actual_paths - required_paths - OPTIONAL_GENERATOR_FILES)
    _require(not missing and not extra, f"payload inventory drifted: missing={missing}, extra={extra}")

    row_by_path = {row["file"]: row for row in [*static_rows, *sequence_rows]}
    for relative, row in row_by_path.items():
        observed = (root / relative).stat().st_size
        _require(observed == row["bytes"], f"manifest byte count drifted for {relative}: expected {row['bytes']}, observed {observed}")
    manifest_payload_bytes = sum(row["bytes"] for row in row_by_path.values())
    declared_payload_bytes = metadata_values["total_gb"] * (1024**3)
    _require(
        math.isclose(declared_payload_bytes, manifest_payload_bytes, rel_tol=0, abs_tol=0.25),
        f"metadata total_gb does not match manifest NPZ bytes: declared {declared_payload_bytes}, observed {manifest_payload_bytes}",
    )

    file_records: list[dict[str, Any]] = []
    for path in inventory:
        deadline.check("file hashing")
        file_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path, deadline),
            }
        )
    tree_sha256 = _canonical_tree_sha256(file_records)

    static_cells = 0
    static_shapes: set[tuple[int, int, int]] = set()
    for row in static_rows:
        cells, shape = _validate_static_npz(
            root / row["file"], row, metadata_values["voxel_native_um"], deadline
        )
        static_cells += cells
        static_shapes.add(shape)

    total_nodes = 0
    total_edges = 0
    total_divisions = 0
    sequence_shapes: set[tuple[int, ...]] = set()
    pooled_voxels: set[tuple[float, float, float]] = set()
    native_voxel = metadata_values["voxel_native_um"]
    expected_pooled_voxel = [
        native_voxel[0],
        native_voxel[1] * 4,
        native_voxel[2] * 4,
    ]
    for row in sequence_rows:
        nodes, edges, divisions, shape, pooled_voxel = _validate_sequence_npz(
            root / row["file"], row, expected_pooled_voxel, deadline
        )
        total_nodes += nodes
        total_edges += edges
        total_divisions += divisions
        sequence_shapes.add(shape)
        pooled_voxels.add(pooled_voxel)
    _require(total_nodes == expected.nodes, f"sequence node aggregate drifted: {total_nodes}")
    _require(total_divisions == expected.divisions, f"sequence division aggregate drifted: {total_divisions}")
    _require(
        sum(row["n_nodes"] for row in sequence_rows) == total_nodes
        and sum(row["n_edges"] for row in sequence_rows) == total_edges
        and sum(row["n_divisions"] for row in sequence_rows) == total_divisions,
        "manifest lineage aggregates drifted",
    )
    deadline.check("finalizing authentication")

    return {
        "schema": REPORT_SCHEMA,
        "status": "AUTHENTICATED",
        "source_id": SOURCE_ID,
        "expected": expected.as_dict(),
        "runtime_cap_seconds": float(max_runtime_seconds),
        "payload": {
            "file_count": len(file_records),
            "total_bytes": sum(row["bytes"] for row in file_records),
            "manifest_npz_bytes": manifest_payload_bytes,
            "tree_sha256": tree_sha256,
            "tree_sha256_algorithm": "sha256(concat(u64be(len(path_utf8)),path_utf8,file_sha256_bytes))",
            "files": file_records,
        },
        "metadata": {
            "sha256": next(row["sha256"] for row in file_records if row["path"] == "metadata.json"),
            "seq_len": metadata_values["seq_len"],
            "voxel_native_um": metadata_values["voxel_native_um"],
        },
        "manifest": {
            "sha256": next(row["sha256"] for row in file_records if row["path"] == "manifest.json"),
            "static_count": len(static_rows),
            "sequence_count": len(sequence_rows),
        },
        "validated": {
            "static_cells": static_cells,
            "static_volume_shapes": [list(shape) for shape in sorted(static_shapes)],
            "sequence_nodes": total_nodes,
            "sequence_edges": total_edges,
            "sequence_divisions": total_divisions,
            "sequence_volume_shapes": [list(shape) for shape in sorted(sequence_shapes)],
            "pooled_voxel_um": [list(voxel) for voxel in sorted(pooled_voxels)],
            "image_arrays_materialized": False,
            "loaded_npz_members": [
                "centroids",
                "voxel_um",
                "nodes",
                "edges",
                "divisions",
                "voxel_um_pooled",
            ],
        },
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("ascii")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="biohub_synthetic payload directory")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON authentication report")
    parser.add_argument("--expected-static-count", type=int, default=DEFAULT_STATIC_COUNT)
    parser.add_argument("--expected-sequence-count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--expected-total-nodes", type=int, default=DEFAULT_TOTAL_NODES)
    parser.add_argument("--expected-total-divisions", type=int, default=DEFAULT_TOTAL_DIVISIONS)
    parser.add_argument("--max-runtime-seconds", type=float, default=DEFAULT_MAX_RUNTIME_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected = ExpectedCounts(
        static=args.expected_static_count,
        sequences=args.expected_sequence_count,
        nodes=args.expected_total_nodes,
        divisions=args.expected_total_divisions,
    )
    reported_runtime_cap = (
        float(args.max_runtime_seconds)
        if math.isfinite(float(args.max_runtime_seconds))
        else None
    )
    payload = args.payload.resolve(strict=False)
    output = args.output.resolve(strict=False)
    if output == payload or payload in output.parents:
        error = "output must be outside the payload tree"
        report = {
            "schema": REPORT_SCHEMA,
            "status": "REJECTED",
            "source_id": SOURCE_ID,
            "expected": expected.as_dict(),
            "runtime_cap_seconds": reported_runtime_cap,
            "error": error,
        }
        atomic_write_json(args.output, report)
        print(error, file=sys.stderr)
        return 2
    try:
        report = authenticate_payload(
            args.payload,
            expected=expected,
            max_runtime_seconds=args.max_runtime_seconds,
        )
        code = 0
    except (AuthenticationError, FileNotFoundError, OSError) as error:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "REJECTED",
            "source_id": SOURCE_ID,
            "expected": expected.as_dict(),
            "runtime_cap_seconds": reported_runtime_cap,
            "error": str(error),
        }
        print(f"payload rejected: {error}", file=sys.stderr)
        code = 1
    atomic_write_json(args.output, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
