"""Label-free, sparse V32 replay-cache primitives.

This module deliberately contains no model, scorer, ground-truth, or ranker code.
It defines the durable hand-off between one immutable V32 inference run and
offline association experiments.  The hand-off has three layers:

``pre_ilp``
    The fused sparse candidate graph immediately before the ILP solver.
``post_ilp``
    The selected graph returned by that same solver, including selected nodes
    with no selected incident edge.
``base_final``
    The unmodified V32 output used as the zero-bonus replay oracle.

Arrays use explicit, non-object dtypes and are stored as ordinary ``.npy``
files with ``allow_pickle=False``.  A cache is accepted only when its top-level
manifest and every per-dataset manifest/file hash validate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CACHE_SCHEMA = "celltrack-alpha.v32-preilp-replay-cache.v1"
DATASET_SCHEMA = "celltrack-alpha.v32-preilp-replay-dataset.v1"
PRODUCER_SCHEMA = "celltrack-alpha.v32-preilp-producer.v1"
POPULATION_SCHEMA = "celltrack-alpha.v32-preilp-population.v1"
OPEN_AUDIT_SCHEMA = "celltrack-alpha.v32-label-free-open-audit.v1"
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
IMMUTABLE_PRODUCER_HASHES = {
    "public_notebook_sha256": "3f995a97a51c9517bdf8d8da2f606ffd184a51c841de87c962d911a336b7847d",
    "base_predictor_sha256": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "primary_manifest_sha256": "bc20f1f04cfb682af3b27a836ce9a57f44a2fe27508dc59039200b2a23188103",
    "primary_weight_sha256": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "primary_config_sha256": "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a",
    "secondary_manifest_sha256": "7f01c0b2f9491606a1b543339e072230e25317d9c70fb9a18aea59e002a995eb",
    "secondary_weight_sha256": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
    "secondary_config_sha256": "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a",
}
IMMUTABLE_POPULATION_HASHES = {
    "population_file_sha256": "6c8153d79bcda20e76d74de141928a054b0d123f3e80088a4fbe4c392ea9d5ae",
    "population_self_sha256": "4b18591377fea4faba6e9f7fdcb230acc0b5a43f79dea2b7d532c2b02dc75dc5",
    "raw_archive_sha256": "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19",
}

NODE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("node_id", "<i8"),
        ("t", "<i4"),
        ("z_vox", "<f8"),
        ("y_vox", "<f8"),
        ("x_vox", "<f8"),
        ("z_um", "<f8"),
        ("y_um", "<f8"),
        ("x_um", "<f8"),
    ],
    align=False,
)
PRE_EDGE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("source_id", "<i8"),
        ("target_id", "<i8"),
        ("edge_prob", "<f8"),
        ("edge_dist_grid", "<f8"),
    ],
    align=False,
)
POST_EDGE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("source_id", "<i8"),
        ("target_id", "<i8"),
        ("edge_prob", "<f8"),
    ],
    align=False,
)
FINAL_EDGE_DTYPE = np.dtype(
    [
        ("ordinal", "<i8"),
        ("source_id", "<i8"),
        ("target_id", "<i8"),
    ],
    align=False,
)
FRAME_PAIR_DTYPE = np.dtype(
    [
        ("source_t", "<i4"),
        ("target_t", "<i4"),
        ("source_count", "<i8"),
        ("target_count", "<i8"),
        ("candidate_start", "<i8"),
        ("candidate_count", "<i8"),
        ("skipped_empty", "u1"),
    ],
    align=False,
)

ARRAY_SPECS: dict[str, tuple[str, np.dtype[Any]]] = {
    "pre_ilp_nodes": ("PRE_ILP_NODES.npy", NODE_DTYPE),
    "pre_ilp_edges": ("PRE_ILP_EDGES.npy", PRE_EDGE_DTYPE),
    "post_ilp_nodes": ("POST_ILP_NODES.npy", NODE_DTYPE),
    "post_ilp_edges": ("POST_ILP_EDGES.npy", POST_EDGE_DTYPE),
    "frame_pairs": ("FRAME_PAIRS.npy", FRAME_PAIR_DTYPE),
    "base_final_nodes": ("BASE_FINAL_NODES.npy", NODE_DTYPE),
    "base_final_edges": ("BASE_FINAL_EDGES.npy", FINAL_EDGE_DTYPE),
}
DATASET_MANIFEST_NAME = "DATASET_MANIFEST.json"
TOP_MANIFEST_NAME = "CACHE_MANIFEST.json"
OPEN_AUDIT_NAME = "OPEN_AUDIT.jsonl"
BASE_FRAGMENT_NAME = "BASE_SUBMISSION_FRAGMENT.csv"
SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
_DATASET_FILE_NAMES = {
    *(filename for filename, _dtype in ARRAY_SPECS.values()),
    DATASET_MANIFEST_NAME,
    BASE_FRAGMENT_NAME,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_OPEN_MARKERS = (
    ".geff",
    "ground_truth",
    "ground-truth",
    "/labels/",
    "\\labels\\",
    "/gt/",
    "\\gt\\",
    "/scorer/",
    "\\scorer\\",
    "b0_outcomes",
    "/m3/",
    "\\m3\\",
    "m3_",
    "-m3-",
    "association_ranker",
    "local_association_ranker",
)


class V32CacheError(ValueError):
    """The V32 cache is incomplete, ambiguous, or not label-free."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise V32CacheError("non-finite JSON value is forbidden")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, strict JSON bytes with a terminal newline."""

    return (
        json.dumps(
            _json_safe(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order values rather than the NPY container."""

    value = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": value.dtype.descr, "shape": list(value.shape)}
    )
    digest = hashlib.sha256(header)
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise V32CacheError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256_bytes(payload)


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def atomic_write_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V32CacheError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "semantic_sha256": semantic_array_sha256(array),
        "dtype": array.dtype.descr,
        "shape": list(array.shape),
        "rows": int(len(array)),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V32CacheError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V32CacheError(f"{label} must be a JSON object")
    return value


def _require_sha(value: Any, label: str) -> str:
    text = str(value)
    if _HEX64.fullmatch(text) is None:
        raise V32CacheError(f"{label} must be a lowercase SHA-256")
    return text


def _require_dataset(dataset: str) -> str:
    if _DATASET.fullmatch(dataset) is None or ".." in dataset:
        raise V32CacheError(f"unsafe dataset name {dataset!r}")
    return dataset


def assert_label_free_open_paths(paths: Iterable[str | Path]) -> tuple[str, ...]:
    """Fail closed if an input-open audit contains a protected data surface."""

    accepted: list[str] = []
    for raw in paths:
        text = str(raw)
        normalized = text.replace("\\", "/").lower()
        padded = f"/{normalized.strip('/')}"
        if any(marker in padded for marker in _FORBIDDEN_OPEN_MARKERS):
            raise V32CacheError(f"protected input path in open audit: {text}")
        accepted.append(text)
    return tuple(accepted)


def write_open_audit(path: Path, paths: Iterable[str | Path]) -> str:
    accepted = assert_label_free_open_paths(paths)
    lines = [
        canonical_json_bytes(
            {"schema": OPEN_AUDIT_SCHEMA, "event": "input_open", "path": item}
        )
        for item in accepted
    ]
    return atomic_write_bytes(path, b"".join(lines))


def expected_runtime_split_bytes(identities: Sequence[str]) -> bytes:
    """Reconstruct the exact split JSON written by the accepted V32 notebook."""

    datasets: list[str] = []
    for identity in identities:
        if not isinstance(identity, str) or not identity.startswith("train:"):
            raise V32CacheError(f"invalid split identity {identity!r}")
        datasets.append(_require_dataset(identity.split(":", 1)[1]))
    if not datasets or len(set(datasets)) != len(datasets):
        raise V32CacheError("runtime split identities must be nonempty and unique")
    return json.dumps(
        [{"split": 0, "train": [], "test": sorted(datasets)}], indent=2
    ).encode("utf-8")


def expected_runtime_split_sha256(identities: Sequence[str]) -> str:
    return sha256_bytes(expected_runtime_split_bytes(identities))


def make_nodes(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    """Build the canonical node array from explicit IDs and voxel positions."""

    values = list(rows)
    result = np.empty(len(values), dtype=NODE_DTYPE)
    for ordinal, row in enumerate(values):
        node_id = int(row["node_id"])
        t = int(row["t"])
        z = float(row.get("z_vox", row.get("z")))
        y = float(row.get("y_vox", row.get("y")))
        x = float(row.get("x_vox", row.get("x")))
        result[ordinal] = (
            ordinal,
            node_id,
            t,
            z,
            y,
            x,
            z * VOXEL_SCALE_UM[0],
            y * VOXEL_SCALE_UM[1],
            x * VOXEL_SCALE_UM[2],
        )
    return result


def make_pre_edges(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    values = list(rows)
    result = np.empty(len(values), dtype=PRE_EDGE_DTYPE)
    for ordinal, row in enumerate(values):
        result[ordinal] = (
            ordinal,
            int(row["source_id"]),
            int(row["target_id"]),
            float(row["edge_prob"]),
            float(row.get("edge_dist_grid", row.get("edge_dist"))),
        )
    return result


def make_post_edges(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    values = list(rows)
    result = np.empty(len(values), dtype=POST_EDGE_DTYPE)
    for ordinal, row in enumerate(values):
        result[ordinal] = (
            ordinal,
            int(row["source_id"]),
            int(row["target_id"]),
            float(row["edge_prob"]),
        )
    return result


def make_final_edges(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    values = list(rows)
    result = np.empty(len(values), dtype=FINAL_EDGE_DTYPE)
    for ordinal, row in enumerate(values):
        result[ordinal] = (
            ordinal,
            int(row["source_id"]),
            int(row["target_id"]),
        )
    return result


def make_frame_pairs(
    pre_nodes: np.ndarray, pre_edges: np.ndarray, *, frame_count: int
) -> np.ndarray:
    """Create an exact frame-pair census for candidate coverage checks."""

    if frame_count <= 0:
        raise V32CacheError("frame_count must be positive")
    result = np.empty(max(0, frame_count - 1), dtype=FRAME_PAIR_DTYPE)
    node_times = {
        int(node_id): int(t)
        for node_id, t in zip(pre_nodes["node_id"], pre_nodes["t"], strict=True)
    }
    cursor = 0
    for t in range(frame_count - 1):
        source_count = int(np.count_nonzero(pre_nodes["t"] == t))
        target_count = int(np.count_nonzero(pre_nodes["t"] == t + 1))
        start = cursor
        while cursor < len(pre_edges):
            source_id = int(pre_edges["source_id"][cursor])
            source_t = node_times.get(source_id)
            if source_t != t:
                break
            cursor += 1
        result[t] = (
            t,
            t + 1,
            source_count,
            target_count,
            start,
            cursor - start,
            int(source_count == 0 or target_count == 0),
        )
    if cursor != len(pre_edges):
        raise V32CacheError("pre-ILP edges are not grouped by consecutive source frame")
    return result


def _validate_dtype(name: str, array: np.ndarray, expected: np.dtype[Any]) -> None:
    if array.ndim != 1:
        raise V32CacheError(f"{name} must be one-dimensional")
    if array.dtype != expected:
        raise V32CacheError(f"{name} dtype drift: {array.dtype.descr!r} != {expected.descr!r}")
    if array.dtype.hasobject:
        raise V32CacheError(f"{name} may not contain objects")


def _validate_ordinals(name: str, array: np.ndarray) -> None:
    expected = np.arange(len(array), dtype=np.int64)
    if not np.array_equal(array["ordinal"], expected):
        raise V32CacheError(f"{name} ordinals must be contiguous and ordered")


def _validate_nodes(name: str, nodes: np.ndarray, *, frame_count: int) -> None:
    _validate_dtype(name, nodes, NODE_DTYPE)
    _validate_ordinals(name, nodes)
    ids = nodes["node_id"]
    if len(np.unique(ids)) != len(ids):
        raise V32CacheError(f"{name} contains duplicate node IDs")
    if np.any(nodes["t"] < 0) or np.any(nodes["t"] >= frame_count):
        raise V32CacheError(f"{name} contains an out-of-range frame")
    numeric = ["z_vox", "y_vox", "x_vox", "z_um", "y_um", "x_um"]
    if any(not np.isfinite(nodes[field]).all() for field in numeric):
        raise V32CacheError(f"{name} contains non-finite positions")
    for voxel, microns, scale in (
        ("z_vox", "z_um", VOXEL_SCALE_UM[0]),
        ("y_vox", "y_um", VOXEL_SCALE_UM[1]),
        ("x_vox", "x_um", VOXEL_SCALE_UM[2]),
    ):
        if not np.array_equal(nodes[microns], nodes[voxel] * scale):
            raise V32CacheError(f"{name} has inconsistent {microns}")


def _edge_keys(edges: np.ndarray) -> list[tuple[int, int]]:
    return list(
        zip(
            (int(value) for value in edges["source_id"]),
            (int(value) for value in edges["target_id"]),
            strict=True,
        )
    )


def _validate_edges(
    name: str,
    edges: np.ndarray,
    expected_dtype: np.dtype[Any],
    *,
    nodes: np.ndarray,
    require_consecutive: bool,
) -> None:
    _validate_dtype(name, edges, expected_dtype)
    _validate_ordinals(name, edges)
    keys = _edge_keys(edges)
    if len(set(keys)) != len(keys):
        raise V32CacheError(f"{name} contains duplicate directed edge keys")
    node_time = {
        int(node_id): int(t)
        for node_id, t in zip(nodes["node_id"], nodes["t"], strict=True)
    }
    for source, target in keys:
        if source not in node_time or target not in node_time:
            raise V32CacheError(f"{name} contains a dangling edge {source}->{target}")
        if require_consecutive and node_time[target] != node_time[source] + 1:
            raise V32CacheError(f"{name} contains a nonconsecutive edge {source}->{target}")
    for field in ("edge_prob", "edge_dist_grid"):
        if field in edges.dtype.names and not np.isfinite(edges[field]).all():
            raise V32CacheError(f"{name} contains non-finite {field}")
    if "edge_prob" in edges.dtype.names and (
        np.any(edges["edge_prob"] < 0.0) or np.any(edges["edge_prob"] > 1.0)
    ):
        raise V32CacheError(f"{name} contains edge probability outside [0,1]")
    if "edge_dist_grid" in edges.dtype.names and np.any(edges["edge_dist_grid"] < 0.0):
        raise V32CacheError(f"{name} contains a negative grid distance")


def validate_dataset_arrays(
    arrays: Mapping[str, np.ndarray], *, frame_count: int
) -> dict[str, int]:
    missing = sorted(set(ARRAY_SPECS) - set(arrays))
    extra = sorted(set(arrays) - set(ARRAY_SPECS))
    if missing or extra:
        raise V32CacheError(f"dataset array keys drift; missing={missing}, extra={extra}")
    pre_nodes = arrays["pre_ilp_nodes"]
    pre_edges = arrays["pre_ilp_edges"]
    post_nodes = arrays["post_ilp_nodes"]
    post_edges = arrays["post_ilp_edges"]
    final_nodes = arrays["base_final_nodes"]
    final_edges = arrays["base_final_edges"]
    frame_pairs = arrays["frame_pairs"]

    _validate_nodes("pre_ilp_nodes", pre_nodes, frame_count=frame_count)
    _validate_nodes("post_ilp_nodes", post_nodes, frame_count=frame_count)
    _validate_nodes("base_final_nodes", final_nodes, frame_count=frame_count)
    _validate_edges(
        "pre_ilp_edges", pre_edges, PRE_EDGE_DTYPE, nodes=pre_nodes, require_consecutive=True
    )
    _validate_edges(
        "post_ilp_edges", post_edges, POST_EDGE_DTYPE, nodes=post_nodes, require_consecutive=True
    )
    _validate_edges(
        "base_final_edges", final_edges, FINAL_EDGE_DTYPE, nodes=final_nodes,
        require_consecutive=True,
    )

    pre_ids = {int(value) for value in pre_nodes["node_id"]}
    post_ids = {int(value) for value in post_nodes["node_id"]}
    if not post_ids <= pre_ids:
        raise V32CacheError("post-ILP nodes are not a subset of pre-ILP root nodes")
    pre_node_by_id = {int(row["node_id"]): row for row in pre_nodes}
    preserved_node_fields = (
        "t", "z_vox", "y_vox", "x_vox", "z_um", "y_um", "x_um"
    )
    for post_node in post_nodes:
        node_id = int(post_node["node_id"])
        pre_node = pre_node_by_id[node_id]
        if any(post_node[field] != pre_node[field] for field in preserved_node_fields):
            raise V32CacheError(
                f"post-ILP node attributes drift from pre-ILP node {node_id}"
            )
    pre_by_key = {
        key: float(prob)
        for key, prob in zip(_edge_keys(pre_edges), pre_edges["edge_prob"], strict=True)
    }
    for key, probability in zip(_edge_keys(post_edges), post_edges["edge_prob"], strict=True):
        if key not in pre_by_key:
            raise V32CacheError(f"post-ILP edge {key} is absent from pre-ILP candidates")
        if float(probability) != pre_by_key[key]:
            raise V32CacheError(f"post-ILP edge probability drift for {key}")

    _validate_dtype("frame_pairs", frame_pairs, FRAME_PAIR_DTYPE)
    if len(frame_pairs) != max(0, frame_count - 1):
        raise V32CacheError("frame-pair census cardinality drift")
    if len(frame_pairs):
        if not np.array_equal(frame_pairs["source_t"], np.arange(frame_count - 1)):
            raise V32CacheError("frame-pair source times are not complete and ordered")
        if not np.array_equal(frame_pairs["target_t"], np.arange(1, frame_count)):
            raise V32CacheError("frame-pair target times are not complete and ordered")
    cursor = 0
    node_time = {
        int(node_id): int(t)
        for node_id, t in zip(pre_nodes["node_id"], pre_nodes["t"], strict=True)
    }
    for row in frame_pairs:
        t = int(row["source_t"])
        start = int(row["candidate_start"])
        count = int(row["candidate_count"])
        if start != cursor or count < 0 or start + count > len(pre_edges):
            raise V32CacheError("frame-pair candidate slices are not contiguous")
        expected_source = int(np.count_nonzero(pre_nodes["t"] == t))
        expected_target = int(np.count_nonzero(pre_nodes["t"] == t + 1))
        if int(row["source_count"]) != expected_source or int(row["target_count"]) != expected_target:
            raise V32CacheError("frame-pair node census drift")
        if int(row["skipped_empty"]) != int(expected_source == 0 or expected_target == 0):
            raise V32CacheError("frame-pair empty flag drift")
        for edge in pre_edges[start : start + count]:
            if node_time[int(edge["source_id"])] != t or node_time[int(edge["target_id"])] != t + 1:
                raise V32CacheError("candidate lies outside its declared frame-pair slice")
        cursor += count
    if cursor != len(pre_edges):
        raise V32CacheError("frame-pair census does not cover all candidates")

    return {key: int(len(value)) for key, value in arrays.items()}


def _validate_dataset_metadata(metadata: Mapping[str, Any], *, dataset: str) -> dict[str, Any]:
    value = dict(metadata)
    if value.get("schema") != DATASET_SCHEMA:
        raise V32CacheError("dataset metadata schema drift")
    if "status" in value and value.get("status") != "PASS":
        raise V32CacheError("dataset metadata status drift")
    if value.get("dataset") != dataset:
        raise V32CacheError("dataset metadata identity drift")
    if value.get("identity") != f"train:{dataset}":
        raise V32CacheError("dataset canonical identity drift")
    if value.get("population_role") not in {"held_fit44b6", "held_fit6bba", "cv57", "canary"}:
        raise V32CacheError("unknown population role")
    if not isinstance(value.get("frame_count"), int) or int(value["frame_count"]) <= 0:
        raise V32CacheError("frame_count must be a positive integer")
    if tuple(value.get("voxel_scale_um", ())) != VOXEL_SCALE_UM:
        raise V32CacheError("voxel scale drift")
    if value.get("label_free") is not True or value.get("evaluate") is not False:
        raise V32CacheError("dataset cache is not explicitly label-free/evaluate=false")
    if value.get("image_access_mode") != "raw_zarr_images_only":
        raise V32CacheError("image access mode is not fail-closed")
    for key in (
        "raw_archive_sha256",
        "raw_zarr_metadata_sha256",
        "effective_config_sha256",
        "opened_path_audit_sha256",
    ):
        _require_sha(value.get(key), key)
    return value


def write_dataset_cache(
    root: Path,
    *,
    dataset: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    base_submission_fragment: bytes,
) -> dict[str, Any]:
    """Atomically publish one complete dataset cache without overwriting."""

    dataset = _require_dataset(dataset)
    checked_metadata = _validate_dataset_metadata(metadata, dataset=dataset)
    frame_count = int(checked_metadata["frame_count"])
    counts = validate_dataset_arrays(arrays, frame_count=frame_count)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "datasets" / dataset
    if destination.exists():
        raise V32CacheError(f"refusing to overwrite dataset cache {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dataset}.", suffix=".tmp", dir=destination.parent)
    )
    try:
        files: dict[str, Any] = {}
        for key, (filename, expected_dtype) in ARRAY_SPECS.items():
            value = np.asarray(arrays[key])
            _validate_dtype(key, value, expected_dtype)
            files[key] = atomic_write_npy(staging / filename, value)
        files["base_submission_fragment"] = {
            "path": BASE_FRAGMENT_NAME,
            "bytes": len(base_submission_fragment),
            "sha256": atomic_write_bytes(staging / BASE_FRAGMENT_NAME, base_submission_fragment),
        }
        manifest = {
            **checked_metadata,
            "status": "PASS",
            "counts": counts,
            "files": files,
        }
        manifest_sha = atomic_write_json(staging / DATASET_MANIFEST_NAME, manifest)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "dataset": dataset,
        "identity": f"train:{dataset}",
        "path": f"datasets/{dataset}/{DATASET_MANIFEST_NAME}",
        "sha256": manifest_sha,
        "counts": counts,
    }


def _validate_producer(producer: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(producer)
    if value.get("schema") != PRODUCER_SCHEMA:
        raise V32CacheError("producer schema drift")
    required_hashes = (
        "public_notebook_sha256",
        "builder_sha256",
        "package_source_sha256",
        "source_freeze_sha256",
        "base_predictor_sha256",
        "runtime_predictor_sha256",
        "terminal_receipt_sha256",
        "terminal_inventory_sha256",
        "terminal_open_audit_sha256",
        "terminal_submission_sha256",
        "runtime_split_sha256",
        "primary_manifest_sha256",
        "primary_weight_sha256",
        "primary_config_sha256",
        "secondary_manifest_sha256",
        "secondary_weight_sha256",
        "secondary_config_sha256",
        "effective_environment_sha256",
    )
    for key in required_hashes:
        _require_sha(value.get(key), f"producer.{key}")
    for key, expected in IMMUTABLE_PRODUCER_HASHES.items():
        if value.get(key) != expected:
            raise V32CacheError(f"producer immutable binding drift for {key}")
    if value.get("label_free") is not True or value.get("m3_mounted") is not False:
        raise V32CacheError("producer must be label-free with M3 absent")
    if value.get("evaluate") is not False:
        raise V32CacheError("producer evaluate flag must be false")
    return value


def validate_producer_bindings(producer: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a complete producer receipt against immutable V32 assets."""

    return _validate_producer(producer)


def _validate_population(population: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(population)
    if value.get("schema") != POPULATION_SCHEMA:
        raise V32CacheError("population schema drift")
    identities = value.get("identities")
    if not isinstance(identities, list) or not identities:
        raise V32CacheError("population identities must be a nonempty list")
    if len(set(identities)) != len(identities):
        raise V32CacheError("population contains duplicate identities")
    for identity in identities:
        if not isinstance(identity, str) or not identity.startswith("train:"):
            raise V32CacheError(f"invalid population identity {identity!r}")
        _require_dataset(identity.split(":", 1)[1])
    role = value.get("role")
    if role not in {"held_fit44b6", "held_fit6bba", "cv57", "canary"}:
        raise V32CacheError("unknown population role")
    if role == "canary" and identities != ["train:44b6_cf8fed6b"]:
        raise V32CacheError("canary population is frozen at train:44b6_cf8fed6b")
    if role == "held_fit44b6" and (
        len(identities) != 36
        or any(not identity.startswith("train:44b6_") for identity in identities)
    ):
        raise V32CacheError("held_fit44b6 population count/prefix drift")
    if role == "held_fit6bba" and (
        len(identities) != 69
        or any(not identity.startswith("train:6bba_") for identity in identities)
    ):
        raise V32CacheError("held_fit6bba population count/prefix drift")
    if role == "cv57" and len(identities) != 57:
        raise V32CacheError("cv57 population count drift")
    for key in ("population_file_sha256", "population_self_sha256", "raw_archive_sha256"):
        _require_sha(value.get(key), f"population.{key}")
    for key, expected in IMMUTABLE_POPULATION_HASHES.items():
        if value.get(key) != expected:
            raise V32CacheError(f"population immutable binding drift for {key}")
    return value


def write_cache_manifest(
    root: Path,
    *,
    producer: Mapping[str, Any],
    population: Mapping[str, Any],
    dataset_entries: Sequence[Mapping[str, Any]],
    open_audit_sha256: str,
) -> dict[str, Any]:
    """Seal a complete cache.  The top manifest is always published last."""

    checked_producer = _validate_producer(producer)
    checked_population = _validate_population(population)
    _require_sha(open_audit_sha256, "open_audit_sha256")
    expected = list(checked_population["identities"])
    if checked_producer["runtime_split_sha256"] != expected_runtime_split_sha256(expected):
        raise V32CacheError("producer runtime split differs from the frozen population")
    observed = [str(entry.get("identity")) for entry in dataset_entries]
    if observed != expected:
        raise V32CacheError("dataset entries do not exactly match ordered population identities")
    verified_entries: list[dict[str, Any]] = []
    for raw in dataset_entries:
        entry = dict(raw)
        dataset = _require_dataset(str(entry.get("dataset")))
        path = root / str(entry.get("path"))
        expected_path = root / "datasets" / dataset / DATASET_MANIFEST_NAME
        if path != expected_path or not path.is_file():
            raise V32CacheError(f"dataset manifest path drift for {dataset}")
        actual = sha256_file(path)
        if actual != _require_sha(entry.get("sha256"), f"{dataset}.manifest"):
            raise V32CacheError(f"dataset manifest hash drift for {dataset}")
        validated = validate_dataset_cache(root, dataset)
        if validated["population_role"] != checked_population["role"]:
            raise V32CacheError(f"{dataset} population role differs from the top manifest")
        if validated["raw_archive_sha256"] != checked_population["raw_archive_sha256"]:
            raise V32CacheError(f"{dataset} raw archive differs from the frozen population")
        if (
            validated["effective_config_sha256"]
            != checked_producer["effective_environment_sha256"]
        ):
            raise V32CacheError(f"{dataset} effective config differs from the producer")
        if validated["opened_path_audit_sha256"] != open_audit_sha256:
            raise V32CacheError(f"{dataset} is not bound to the sealed open audit")
        if entry.get("counts") != validated["counts"]:
            raise V32CacheError(f"dataset entry count drift for {dataset}")
        verified_entries.append(entry)
    if checked_producer["terminal_open_audit_sha256"] != open_audit_sha256:
        raise V32CacheError("sealed open audit differs from terminal runtime audit")
    reconstructed_submission = reconstruct_terminal_submission(
        root, [entry["dataset"] for entry in verified_entries]
    )
    if sha256_bytes(reconstructed_submission) != checked_producer["terminal_submission_sha256"]:
        raise V32CacheError("base fragments do not reconstruct the terminal submission")
    manifest = {
        "schema": CACHE_SCHEMA,
        "status": "PASS",
        "producer": checked_producer,
        "population": checked_population,
        "open_audit": {"path": OPEN_AUDIT_NAME, "sha256": open_audit_sha256},
        "datasets": verified_entries,
    }
    sha = atomic_write_json(root / TOP_MANIFEST_NAME, manifest)
    return {**manifest, "manifest_sha256": sha}


def load_dataset_arrays(
    root: Path, dataset: str, *, mmap_mode: str | None = "r"
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset = _require_dataset(dataset)
    directory = root / "datasets" / dataset
    metadata = _load_json(directory / DATASET_MANIFEST_NAME, "dataset manifest")
    arrays: dict[str, np.ndarray] = {}
    for key, (filename, _dtype) in ARRAY_SPECS.items():
        try:
            arrays[key] = np.load(directory / filename, allow_pickle=False, mmap_mode=mmap_mode)
        except (OSError, ValueError) as exc:
            raise V32CacheError(f"cannot load {dataset}/{filename}: {exc}") from exc
    return metadata, arrays


def validate_dataset_cache(root: Path, dataset: str) -> dict[str, Any]:
    dataset = _require_dataset(dataset)
    directory = root / "datasets" / dataset
    if not directory.is_dir() or directory.is_symlink():
        raise V32CacheError(f"{dataset} cache directory is absent or a symlink")
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != _DATASET_FILE_NAMES:
        raise V32CacheError(f"{dataset} cache file-set drift")
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise V32CacheError(f"{dataset} cache contains a non-regular file: {path.name}")
    metadata, arrays = load_dataset_arrays(root, dataset)
    checked = _validate_dataset_metadata(metadata, dataset=dataset)
    if metadata.get("status") != "PASS":
        raise V32CacheError(f"{dataset} dataset manifest status drift")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != {*ARRAY_SPECS, "base_submission_fragment"}:
        raise V32CacheError(f"{dataset} file manifest keys drift")
    for key, (filename, _dtype) in ARRAY_SPECS.items():
        record = files.get(key)
        if not isinstance(record, dict) or record.get("path") != filename:
            raise V32CacheError(f"{dataset}/{key} file record drift")
        path = directory / filename
        if sha256_file(path) != _require_sha(record.get("sha256"), f"{dataset}.{key}"):
            raise V32CacheError(f"{dataset}/{key} container hash mismatch")
        if semantic_array_sha256(arrays[key]) != _require_sha(
            record.get("semantic_sha256"), f"{dataset}.{key}.semantic"
        ):
            raise V32CacheError(f"{dataset}/{key} semantic hash mismatch")
        if record.get("dtype") != json.loads(json.dumps(arrays[key].dtype.descr)):
            raise V32CacheError(f"{dataset}/{key} dtype receipt drift")
        if record.get("shape") != list(arrays[key].shape):
            raise V32CacheError(f"{dataset}/{key} shape receipt drift")
        if record.get("rows") != len(arrays[key]):
            raise V32CacheError(f"{dataset}/{key} row receipt drift")
        if record.get("bytes") != path.stat().st_size:
            raise V32CacheError(f"{dataset}/{key} byte receipt drift")
    counts = validate_dataset_arrays(arrays, frame_count=int(checked["frame_count"]))
    fragment = files["base_submission_fragment"]
    fragment_path = directory / BASE_FRAGMENT_NAME
    if not isinstance(fragment, dict) or fragment.get("path") != BASE_FRAGMENT_NAME:
        raise V32CacheError(f"{dataset} base submission fragment record drift")
    if not fragment_path.is_file() or sha256_file(fragment_path) != _require_sha(
        fragment.get("sha256"), f"{dataset}.base_submission_fragment"
    ):
        raise V32CacheError(f"{dataset} base submission fragment hash mismatch")
    if fragment.get("bytes") != fragment_path.stat().st_size:
        raise V32CacheError(f"{dataset} base submission fragment byte receipt drift")
    fragment_nodes, fragment_edges, canonical_fragment = parse_submission_fragment(
        fragment_path, dataset=dataset
    )
    final_nodes = arrays["base_final_nodes"]
    if not np.array_equal(fragment_nodes["node_id"], final_nodes["node_id"]):
        raise V32CacheError(f"{dataset} base fragment node IDs drift from base-final graph")
    if not np.array_equal(fragment_nodes["t"], final_nodes["t"]):
        raise V32CacheError(f"{dataset} base fragment node frames drift from base-final graph")
    for axis in ("z", "y", "x"):
        expected = np.maximum(0, np.rint(final_nodes[f"{axis}_vox"])).astype(np.float64)
        if not np.array_equal(fragment_nodes[f"{axis}_vox"], expected):
            raise V32CacheError(
                f"{dataset} base fragment {axis} coordinates drift from base-final graph"
            )
    if _edge_keys(fragment_edges) != _edge_keys(arrays["base_final_edges"]):
        raise V32CacheError(f"{dataset} base fragment edges drift from base-final graph")
    if canonical_fragment != fragment_path.read_bytes():
        raise V32CacheError(f"{dataset} base submission fragment is not canonical")
    if metadata.get("counts") != counts:
        raise V32CacheError(f"{dataset} count receipt drift")
    return {
        "dataset": dataset,
        "identity": f"train:{dataset}",
        "status": "PASS",
        "counts": counts,
        "population_role": checked["population_role"],
        "raw_archive_sha256": checked["raw_archive_sha256"],
        "raw_zarr_metadata_sha256": checked["raw_zarr_metadata_sha256"],
        "effective_config_sha256": checked["effective_config_sha256"],
        "manifest_sha256": sha256_file(directory / DATASET_MANIFEST_NAME),
        "opened_path_audit_sha256": checked["opened_path_audit_sha256"],
        "semantic_sha256": sha256_bytes(
            canonical_json_bytes(
                {key: semantic_array_sha256(value) for key, value in arrays.items()}
            )
        ),
    }


def validate_cache(
    root: Path,
    *,
    expected_identities: Sequence[str],
    expected_producer: Mapping[str, Any],
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise V32CacheError("cache root is absent or a symlink")
    if {path.name for path in root.iterdir()} != {
        "datasets", OPEN_AUDIT_NAME, TOP_MANIFEST_NAME,
    }:
        raise V32CacheError("sealed cache root file-set drift")
    dataset_root = root / "datasets"
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise V32CacheError("sealed cache datasets root is absent or a symlink")
    manifest_path = root / TOP_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise V32CacheError("cache manifest is absent or non-regular")
    manifest = _load_json(manifest_path, "cache manifest")
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("status") != "PASS":
        raise V32CacheError("cache manifest schema/status drift")
    producer = _validate_producer(manifest.get("producer", {}))
    checked_expected_producer = _validate_producer(expected_producer)
    if producer != checked_expected_producer:
        raise V32CacheError("cache producer differs from independently expected bindings")
    population = _validate_population(manifest.get("population", {}))
    identities = list(population["identities"])
    if identities != list(expected_identities):
        raise V32CacheError("cache population differs from expected identities")
    if producer["runtime_split_sha256"] != expected_runtime_split_sha256(identities):
        raise V32CacheError("producer runtime split differs from the frozen population")
    open_audit = manifest.get("open_audit")
    if not isinstance(open_audit, dict) or open_audit.get("path") != OPEN_AUDIT_NAME:
        raise V32CacheError("open-audit record drift")
    audit_path = root / OPEN_AUDIT_NAME
    if audit_path.is_symlink() or not audit_path.is_file() or sha256_file(audit_path) != _require_sha(
        open_audit.get("sha256"), "open_audit.sha256"
    ):
        raise V32CacheError("open-audit hash mismatch")
    if open_audit["sha256"] != producer["terminal_open_audit_sha256"]:
        raise V32CacheError("sealed open audit differs from terminal runtime audit")
    opened: list[str] = []
    opened_events: list[dict[str, Any]] = []
    for line_number, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V32CacheError(f"invalid open audit line {line_number}") from exc
        if not isinstance(event, dict):
            raise V32CacheError(f"open audit event must be an object at line {line_number}")
        if event.get("schema") != OPEN_AUDIT_SCHEMA or event.get("event") != "input_open":
            raise V32CacheError(f"open audit event drift at line {line_number}")
        opened.append(str(event.get("path")))
        opened_events.append(event)
    assert_label_free_open_paths(opened)
    if not opened:
        raise V32CacheError("open audit must contain at least one input-open event")
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise V32CacheError("cache datasets must be a list")
    if [entry.get("identity") for entry in entries if isinstance(entry, dict)] != identities:
        raise V32CacheError("cache dataset order/cardinality drift")
    dataset_entries_on_disk = list(dataset_root.iterdir())
    if any(not path.is_dir() or path.is_symlink() for path in dataset_entries_on_disk):
        raise V32CacheError("cache datasets root contains a non-directory or symlink")
    observed_directories = sorted(path.name for path in dataset_entries_on_disk)
    expected_directories = sorted(identity.split(":", 1)[1] for identity in identities)
    if observed_directories != expected_directories:
        raise V32CacheError("cache contains missing or extra dataset directories")
    results: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise V32CacheError("cache dataset entry must be an object")
        dataset = _require_dataset(str(entry.get("dataset")))
        expected_path = f"datasets/{dataset}/{DATASET_MANIFEST_NAME}"
        if entry.get("path") != expected_path:
            raise V32CacheError(f"dataset path drift for {dataset}")
        result = validate_dataset_cache(root, dataset)
        if result["population_role"] != population["role"]:
            raise V32CacheError(f"{dataset} population role differs from the top manifest")
        if result["raw_archive_sha256"] != population["raw_archive_sha256"]:
            raise V32CacheError(f"{dataset} raw archive differs from the frozen population")
        if result["effective_config_sha256"] != producer["effective_environment_sha256"]:
            raise V32CacheError(f"{dataset} effective config differs from the producer")
        matching_raw_events = [
            event
            for event in opened_events
            if str(event.get("path", "")).replace("\\", "/").endswith(
                f"/{dataset}.zarr/0"
            )
        ]
        if (
            len(matching_raw_events) != 1
            or matching_raw_events[0].get("metadata_sha256")
            != result["raw_zarr_metadata_sha256"]
        ):
            raise V32CacheError(
                f"{dataset} raw Zarr metadata differs from the actual-open audit"
            )
        if result["opened_path_audit_sha256"] != open_audit["sha256"]:
            raise V32CacheError(f"{dataset} is not bound to the sealed open audit")
        if result["manifest_sha256"] != _require_sha(entry.get("sha256"), f"{dataset}.sha256"):
            raise V32CacheError(f"top-level dataset hash drift for {dataset}")
        if entry.get("counts") != result["counts"]:
            raise V32CacheError(f"top-level dataset count drift for {dataset}")
        results.append(result)
    reconstructed_submission = reconstruct_terminal_submission(
        root, [result["dataset"] for result in results]
    )
    if sha256_bytes(reconstructed_submission) != producer["terminal_submission_sha256"]:
        raise V32CacheError("base fragments do not reconstruct the terminal submission")
    return {
        "schema": CACHE_SCHEMA,
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "population_role": population["role"],
        "identities": identities,
        "dataset_count": len(results),
        "datasets": results,
        "opened_input_count": len(opened),
    }


def parse_submission_fragment(
    path: Path, *, dataset: str
) -> tuple[np.ndarray, np.ndarray, bytes]:
    """Read one exact V32 CSV fragment and return rounded final graph arrays."""

    dataset = _require_dataset(dataset)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    output_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = SUBMISSION_COLUMNS
        if reader.fieldnames != required:
            raise V32CacheError("submission columns drift")
        for row in reader:
            if row["dataset"] != dataset:
                continue
            output_rows.append(row)
            if row["row_type"] == "node":
                nodes.append(
                    {
                        "node_id": int(row["node_id"]),
                        "t": int(row["t"]),
                        "z": float(row["z"]),
                        "y": float(row["y"]),
                        "x": float(row["x"]),
                    }
                )
            elif row["row_type"] == "edge":
                edges.append(
                    {"source_id": int(row["source_id"]), "target_id": int(row["target_id"])}
                )
            else:
                raise V32CacheError(f"unknown row_type {row['row_type']!r}")
    if not output_rows:
        raise V32CacheError(f"submission has no rows for {dataset}")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SUBMISSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    return make_nodes(nodes), make_final_edges(edges), buffer.getvalue().encode("utf-8")


def reconstruct_terminal_submission(root: Path, datasets: Sequence[str]) -> bytes:
    """Rebuild the exact CRLF V32 submission from canonical per-dataset fragments."""

    import io

    rows_by_id: dict[int, dict[str, str]] = {}
    for dataset in datasets:
        dataset = _require_dataset(dataset)
        path = root / "datasets" / dataset / BASE_FRAGMENT_NAME
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SUBMISSION_COLUMNS:
                raise V32CacheError(f"{dataset} base fragment columns drift")
            for row in reader:
                if row["dataset"] != dataset:
                    raise V32CacheError(f"{dataset} base fragment identity drift")
                try:
                    row_id = int(row["id"])
                except (TypeError, ValueError) as exc:
                    raise V32CacheError(f"{dataset} has a non-integer submission row ID") from exc
                if row_id < 0 or row_id in rows_by_id or str(row_id) != row["id"]:
                    raise V32CacheError("submission row IDs are negative, duplicate, or noncanonical")
                rows_by_id[row_id] = row
    if sorted(rows_by_id) != list(range(len(rows_by_id))):
        raise V32CacheError("submission row IDs are not globally contiguous from zero")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SUBMISSION_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows_by_id[row_id] for row_id in range(len(rows_by_id)))
    return buffer.getvalue().encode("utf-8")


__all__ = [
    "ARRAY_SPECS",
    "BASE_FRAGMENT_NAME",
    "CACHE_SCHEMA",
    "DATASET_SCHEMA",
    "FINAL_EDGE_DTYPE",
    "FRAME_PAIR_DTYPE",
    "IMMUTABLE_POPULATION_HASHES",
    "IMMUTABLE_PRODUCER_HASHES",
    "NODE_DTYPE",
    "OPEN_AUDIT_NAME",
    "OPEN_AUDIT_SCHEMA",
    "POPULATION_SCHEMA",
    "POST_EDGE_DTYPE",
    "PRE_EDGE_DTYPE",
    "PRODUCER_SCHEMA",
    "TOP_MANIFEST_NAME",
    "SUBMISSION_COLUMNS",
    "VOXEL_SCALE_UM",
    "V32CacheError",
    "assert_label_free_open_paths",
    "atomic_write_json",
    "canonical_json_bytes",
    "expected_runtime_split_bytes",
    "expected_runtime_split_sha256",
    "load_dataset_arrays",
    "make_final_edges",
    "make_frame_pairs",
    "make_nodes",
    "make_post_edges",
    "make_pre_edges",
    "parse_submission_fragment",
    "reconstruct_terminal_submission",
    "semantic_array_sha256",
    "sha256_file",
    "validate_cache",
    "validate_dataset_arrays",
    "validate_dataset_cache",
    "validate_producer_bindings",
    "write_cache_manifest",
    "write_dataset_cache",
    "write_open_audit",
]
