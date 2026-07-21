"""Frozen scorer and split-integrity primitives for G-02.

This module intentionally uses only ZIP central-directory metadata plus a
strictly bounded number of small members.  It is the sole entry point for
turning a development archive into the immutable scorer, registry, split, and
fit-provenance manifests consumed by later G-02 work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)
OFFICIAL_MATCH_DISTANCE_UM = 7.0
OFFICIAL_ADJUSTMENT_ALPHA = 0.1
OFFICIAL_DIVISION_WEIGHT = 0.1
MAX_SIGNATURE_CHUNKS_PER_VIDEO = 3
MAX_TRAIN_VIDEOS = 199
MAX_TEST_VIDEOS = 4
MAX_TOTAL_VIDEOS = 203
# Compatibility alias: legacy callers use MAX_VIDEOS for the training/runtime
# proxy.  It is deliberately not the total archive bound.
MAX_VIDEOS = MAX_TRAIN_VIDEOS
MAX_METADATA_MEMBER_BYTES = 64 * 1024
MAX_METADATA_MEMBERS_PER_VIDEO = 32
# T-02's archive inventory is capped at 203 total videos.  Comparing every
# complete signature pair is therefore still bounded (203 choose 2) and, unlike
# an approximate coarse bucket, cannot silently omit a possible near duplicate.
MAX_NEAR_CANDIDATE_PAIRS_PER_BUCKET = (
    MAX_TOTAL_VIDEOS * (MAX_TOTAL_VIDEOS - 1) // 2
)
MAX_AUDIT_DP_STATES = 100000

NODE_COLUMNS = ("dataset", "node_id", "t", "z", "y", "x")
EDGE_COLUMNS = ("dataset", "source_id", "target_id")


class IntegrityError(ValueError):
    """Raised when a registry, split, or provenance invariant is unsafe."""


class SubmissionIntegrityError(IntegrityError):
    """Raised before malformed submission rows reach the official scorer."""


@dataclass(frozen=True)
class ValidatedSubmission:
    """Submission rows partitioned by dataset without changing their values."""

    expected_datasets: tuple[str, ...]
    nodes_by_dataset: Mapping[str, tuple[dict[str, Any], ...]]
    edges_by_dataset: Mapping[str, tuple[dict[str, Any], ...]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert values to strict, portable JSON without non-finite literals."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic ASCII JSON suitable for manifest identity hashes."""

    safe_value = _json_safe(value)
    return (
        json.dumps(
            safe_value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def manifest_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    """Atomically replace one manifest with canonical JSON bytes."""

    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def source_hashes() -> dict[str, str | None]:
    """Return source identities without reading any dataset payload."""

    root = _project_root()
    paths = (
        "src/celltrack_alpha/g02_integrity.py",
        "src/celltrack_alpha/official_eval.py",
        "src/celltrack_alpha/scorer_parity.py",
        "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
        "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py",
    )
    return {relative: _file_sha256(root / relative) for relative in paths}


def scorer_reference_identities() -> dict[str, Any]:
    """Expose fixed official metric identities and score constants for reports."""

    hashes = source_hashes()
    return {
        "metric_source": "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py",
        "division_metric_source": (
            "external/kaggle-cell-tracking-competition/src/"
            "tracking_cellmot/division_metrics.py"
        ),
        "metric_source_sha256": hashes[
            "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py"
        ],
        "division_metric_source_sha256": hashes[
            "external/kaggle-cell-tracking-competition/src/"
            "tracking_cellmot/division_metrics.py"
        ],
        "constants": {
            "scale_zyx_um_per_voxel": list(DEFAULT_SCALE_ZYX),
            "node_match_distance_um": OFFICIAL_MATCH_DISTANCE_UM,
            "adjustment_alpha": OFFICIAL_ADJUSTMENT_ALPHA,
            "division_score_weight": OFFICIAL_DIVISION_WEIGHT,
        },
    }


def _require_mapping(row: Any, row_type: str, index: int) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise SubmissionIntegrityError(f"{row_type}[{index}] must be an object")
    return row


def _require_columns(
    row: Mapping[str, Any], columns: Sequence[str], row_type: str, index: int
) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise SubmissionIntegrityError(
            f"{row_type}[{index}] is missing required columns: {', '.join(missing)}"
        )
    unexpected = sorted(set(row) - set(columns))
    if unexpected:
        raise SubmissionIntegrityError(
            f"{row_type}[{index}] has unexpected columns: {', '.join(unexpected)}"
        )


def _require_dataset(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionIntegrityError(f"{context}.dataset must be a non-empty string")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubmissionIntegrityError(f"{context} must be an integer")
    return value


def _require_coordinate(value: Any, context: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionIntegrityError(f"{context} must be a finite number")
    if not math.isfinite(float(value)):
        raise SubmissionIntegrityError(f"{context} must be a finite number")
    return value


def validate_submission_rows(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    expected_datasets: Iterable[str],
) -> ValidatedSubmission:
    """Reject malformed submission-shaped rows before official scoring.

    The official wrapper deliberately de-duplicates and ignores invalid edges for
    legacy compatibility.  This gate is stricter: it preserves rows verbatim
    after validation and rejects every condition that would otherwise be
    normalized away.
    """

    expected = tuple(sorted(set(expected_datasets)))
    if not expected or any(not isinstance(dataset, str) or not dataset for dataset in expected):
        raise SubmissionIntegrityError("expected_datasets must contain non-empty dataset strings")
    expected_set = set(expected)
    nodes_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    node_times: dict[str, dict[int, int]] = defaultdict(dict)

    for index, raw_row in enumerate(nodes):
        row = _require_mapping(raw_row, "nodes", index)
        _require_columns(row, NODE_COLUMNS, "nodes", index)
        context = f"nodes[{index}]"
        dataset = _require_dataset(row["dataset"], context)
        if dataset not in expected_set:
            raise SubmissionIntegrityError(f"{context}.dataset is not expected: {dataset!r}")
        node_id = _require_int(row["node_id"], f"{context}.node_id")
        timepoint = _require_int(row["t"], f"{context}.t")
        for coordinate in ("z", "y", "x"):
            _require_coordinate(row[coordinate], f"{context}.{coordinate}")
        if node_id in node_times[dataset]:
            raise SubmissionIntegrityError(
                f"duplicate node_id {node_id} in dataset {dataset!r}"
            )
        node_times[dataset][node_id] = timepoint
        nodes_by_dataset[dataset].append(dict(row))

    missing_datasets = expected_set - set(nodes_by_dataset)
    if missing_datasets:
        raise SubmissionIntegrityError(
            "missing node rows for datasets: " + ", ".join(sorted(missing_datasets))
        )

    edges_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_edges: set[tuple[str, int, int]] = set()
    for index, raw_row in enumerate(edges):
        row = _require_mapping(raw_row, "edges", index)
        _require_columns(row, EDGE_COLUMNS, "edges", index)
        context = f"edges[{index}]"
        dataset = _require_dataset(row["dataset"], context)
        if dataset not in expected_set:
            raise SubmissionIntegrityError(f"{context}.dataset is not expected: {dataset!r}")
        source_id = _require_int(row["source_id"], f"{context}.source_id")
        target_id = _require_int(row["target_id"], f"{context}.target_id")
        if source_id == target_id:
            raise SubmissionIntegrityError(f"{context} contains a self-edge")
        if source_id not in node_times[dataset] or target_id not in node_times[dataset]:
            raise SubmissionIntegrityError(
                f"{context} endpoint is absent from dataset {dataset!r}"
            )
        edge_key = (dataset, source_id, target_id)
        if edge_key in seen_edges:
            raise SubmissionIntegrityError(
                f"duplicate directed edge {source_id}->{target_id} in dataset {dataset!r}"
            )
        if node_times[dataset][source_id] >= node_times[dataset][target_id]:
            raise SubmissionIntegrityError(
                f"{context} is not forward in time in dataset {dataset!r}"
            )
        seen_edges.add(edge_key)
        edges_by_dataset[dataset].append(dict(row))

    return ValidatedSubmission(
        expected_datasets=expected,
        nodes_by_dataset={key: tuple(value) for key, value in sorted(nodes_by_dataset.items())},
        edges_by_dataset={key: tuple(value) for key, value in sorted(edges_by_dataset.items())},
    )


def score_submission_rows(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    expected_datasets: Iterable[str],
    ground_truth_by_dataset: Mapping[str, dict[str, Any]],
    scale_zyx_by_dataset: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Validate first, then evaluate each dataset with the official wrapper."""

    validated = validate_submission_rows(nodes, edges, expected_datasets)
    missing_gt = set(validated.expected_datasets) - set(ground_truth_by_dataset)
    if missing_gt:
        raise IntegrityError("missing ground truth for datasets: " + ", ".join(sorted(missing_gt)))

    from .official_eval import official_metric_row, official_summary

    rows: list[dict[str, Any]] = []
    scales = scale_zyx_by_dataset or {}
    for dataset in validated.expected_datasets:
        row = official_metric_row(
            list(validated.nodes_by_dataset[dataset]),
            list(validated.edges_by_dataset.get(dataset, ())),
            ground_truth_by_dataset[dataset],
            scales.get(dataset, DEFAULT_SCALE_ZYX),
        )
        rows.append({"dataset": dataset, **row})
    return {
        "scorer_reference": scorer_reference_identities(),
        "rows": rows,
        "summary": official_summary(rows),
    }


def _member_descriptor(info: zipfile.ZipInfo, kind: str, member_path: str) -> dict[str, Any]:
    return {
        "artifact": kind,
        "member_path": member_path,
        "crc32": f"{info.CRC:08x}",
        "compressed_size": int(info.compress_size),
        "uncompressed_size": int(info.file_size),
    }


def _parse_video_member(info: zipfile.ZipInfo) -> tuple[str, str, str, str] | None:
    """Return ``(video_id, split, kind, relative_member_path)`` for one member."""

    parts = info.filename.split("/")
    if len(parts) < 3 or parts[0] not in {"train", "test"}:
        return None
    root = parts[1]
    if root.endswith(".zarr"):
        dataset = root[: -len(".zarr")]
        kind = "image"
    elif root.endswith(".geff"):
        dataset = root[: -len(".geff")]
        kind = "geff"
    else:
        return None
    if not dataset:
        return None
    relative_member_path = "/".join(parts[2:])
    if not relative_member_path:
        return None
    split = parts[0]
    return f"{split}:{dataset}", split, kind, relative_member_path


def _is_metadata_member(member_path: str) -> bool:
    leaf = member_path.rsplit("/", 1)[-1]
    return leaf in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}


def _selected_indices(length: int, limit: int) -> list[int]:
    if length <= 0:
        return []
    if limit == 1 or length == 1:
        return [0]
    if length <= limit:
        return list(range(length))
    return [index * (length - 1) // (limit - 1) for index in range(limit)]


def _metadata_summary(parsed: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("zarr_format", "data_type", "dtype", "shape", "chunks"):
        value = parsed.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            summary[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float)) and not isinstance(item, bool)
            for item in value
        ):
            summary[key] = value
    return summary


def _shape_first_axis(parsed_by_member: Mapping[tuple[str, str], Mapping[str, Any]], suffix: str) -> int | None:
    for (kind, path), payload in sorted(parsed_by_member.items()):
        if kind != "geff" or not path.endswith(suffix):
            continue
        shape = payload.get("shape")
        if isinstance(shape, list) and shape and isinstance(shape[0], int) and not isinstance(shape[0], bool):
            return int(shape[0])
    return None


def _compact_metadata(
    parsed_by_member: Mapping[tuple[str, str], Mapping[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    image_shapes: list[list[Any]] = []
    image_dtypes: list[str] = []
    for (kind, _path), payload in sorted(parsed_by_member.items()):
        if kind != "image":
            continue
        shape = payload.get("shape")
        if isinstance(shape, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in shape):
            image_shapes.append([int(item) for item in shape])
        dtype = payload.get("data_type", payload.get("dtype"))
        if isinstance(dtype, str):
            image_dtypes.append(dtype)
    image_shape = image_shapes[0] if image_shapes else None
    image_dtype = image_dtypes[0] if image_dtypes else None
    node_count = _shape_first_axis(parsed_by_member, "nodes/ids/zarr.json")
    edge_count = _shape_first_axis(parsed_by_member, "edges/ids/zarr.json")
    time_count = _shape_first_axis(parsed_by_member, "nodes/props/t/values/zarr.json")
    family = dataset.split("_", 1)[0][:16] or "unknown"
    depth = image_shape[1] if image_shape is not None and len(image_shape) > 1 else None
    if depth is None:
        depth_bin = "unknown"
    elif depth <= 16:
        depth_bin = "shallow"
    elif depth <= 64:
        depth_bin = "medium"
    else:
        depth_bin = "deep"
    if node_count is None:
        density_bin = "unlabeled"
    elif node_count < 128:
        density_bin = "sparse"
    elif node_count < 1024:
        density_bin = "moderate"
    else:
        density_bin = "dense"
    return {
        "image_shape": image_shape,
        "image_dtype": image_dtype,
        "geff_node_count": node_count,
        "geff_edge_count": edge_count,
        "geff_time_value_count": time_count,
        "stratum": f"{family}|{depth_bin}|{density_bin}",
    }


def _chunk_signature(payload: bytes) -> dict[str, Any]:
    if not payload:
        raise IntegrityError("empty image chunk cannot provide a duplicate signature")
    histogram = [0] * 16
    for byte in payload:
        histogram[byte >> 4] += 1
    block_count = min(32, len(payload))
    blocks = []
    for index in range(block_count):
        start = index * len(payload) // block_count
        end = (index + 1) * len(payload) // block_count
        blocks.append(hashlib.sha256(payload[start:end]).hexdigest())
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "length": len(payload),
        "histogram16": histogram,
        "block_sha256": blocks,
    }


def _near_chunk_evidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    left_length = int(left["length"])
    right_length = int(right["length"])
    larger = max(left_length, right_length)
    if not larger or abs(left_length - right_length) * 100 > larger * 5:
        return False, {"reason": "chunk_length_delta_exceeds_5_percent"}
    left_histogram = [int(value) for value in left["histogram16"]]
    right_histogram = [int(value) for value in right["histogram16"]]
    histogram_numerator = sum(
        abs(a * right_length - b * left_length)
        for a, b in zip(left_histogram, right_histogram, strict=True)
    )
    histogram_denominator = left_length * right_length
    left_blocks = list(left["block_sha256"])
    right_blocks = list(right["block_sha256"])
    shared = min(len(left_blocks), len(right_blocks))
    matching_blocks = sum(
        1
        for left_block, right_block in zip(left_blocks, right_blocks)
        if left_block == right_block
    )
    blocks_pass = shared > 0 and matching_blocks * 100 >= shared * 80
    histogram_pass = histogram_numerator * 100 <= histogram_denominator * 4
    return histogram_pass and blocks_pass, {
        "histogram_l1_numerator": histogram_numerator,
        "histogram_l1_denominator": histogram_denominator,
        "matching_blocks": matching_blocks,
        "block_count": shared,
    }


def _near_record_evidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    left_chunks = left["signature"]["chunks"]
    right_chunks = right["signature"]["chunks"]
    if len(left_chunks) != len(right_chunks):
        return False, {"reason": "sample_count_mismatch"}
    evidence = []
    for left_chunk, right_chunk in zip(left_chunks, right_chunks, strict=True):
        same, chunk_evidence = _near_chunk_evidence(
            left_chunk["signature"], right_chunk["signature"]
        )
        evidence.append({"match": same, **chunk_evidence})
        if not same:
            return False, {"chunks": evidence}
    return True, {"chunks": evidence}


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def _append_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    members: Sequence[dict[str, Any]],
    checks: list[dict[str, Any]],
    bucket_kind: str,
    bucket_key: str,
) -> None:
    count = len(members)
    possible = count * (count - 1) // 2
    if possible > MAX_NEAR_CANDIDATE_PAIRS_PER_BUCKET:
        checks.append(
            {
                "state": "unresolved",
                "reason": "candidate_bucket_pair_limit",
                "bucket_kind": bucket_kind,
                "bucket_key": bucket_key,
                "candidate_video_ids": [record["video_id"] for record in members],
                "candidate_pair_count": possible,
            }
        )
        return
    for left_index, left in enumerate(members):
        for right in members[left_index + 1 :]:
            pairs.append((left, right))


def _group_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {record["video_id"]: record for record in records}
    union_find = _UnionFind(by_id)
    exact_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_buckets[record["exact_identity_sha256"]].append(record)
    for bucket in exact_buckets.values():
        for record in bucket[1:]:
            union_find.union(bucket[0]["video_id"], record["video_id"])

    checks: list[dict[str, Any]] = []
    candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    members = sorted(records, key=lambda record: record["video_id"])
    unresolved_pairs = [
        (left, right)
        for left_index, left in enumerate(members)
        for right in members[left_index + 1 :]
        if (
            left["signature"]["status"] != "complete"
            or right["signature"]["status"] != "complete"
        )
        and left["exact_identity_sha256"] != right["exact_identity_sha256"]
    ]
    if len(unresolved_pairs) > MAX_NEAR_CANDIDATE_PAIRS_PER_BUCKET:
        checks.append(
            {
                "state": "unresolved",
                "reason": "unresolved_candidate_pair_limit",
                "bucket_kind": "complete_registry",
                "candidate_video_ids": [record["video_id"] for record in members],
                "candidate_pair_count": len(unresolved_pairs),
            }
        )
    else:
        for left, right in unresolved_pairs:
            unresolved_record = (
                left if left["signature"]["status"] != "complete" else right
            )
            checks.append(
                {
                    "left_video_id": left["video_id"],
                    "right_video_id": right["video_id"],
                    "state": "unresolved",
                    "reason": unresolved_record["signature"]["reason"],
                    "bucket_kind": "complete_registry",
                }
            )

    complete = [
        record for record in members if record["signature"]["status"] == "complete"
    ]
    _append_pairs(
        candidate_pairs,
        complete,
        checks,
        "complete_registry",
        "all-complete-signatures",
    )

    seen_pairs: set[tuple[str, str]] = set()
    for left, right in candidate_pairs:
        pair = tuple(sorted((left["video_id"], right["video_id"])))
        if pair in seen_pairs or left["exact_identity_sha256"] == right["exact_identity_sha256"]:
            continue
        seen_pairs.add(pair)
        near, evidence = _near_record_evidence(left, right)
        state = "near_duplicate" if near else "distinct"
        checks.append(
            {
                "left_video_id": pair[0],
                "right_video_id": pair[1],
                "state": state,
                "evidence": evidence,
            }
        )
        if near:
            union_find.union(pair[0], pair[1])

    groups_by_root: dict[str, list[str]] = defaultdict(list)
    for video_id in sorted(by_id):
        groups_by_root[union_find.find(video_id)].append(video_id)
    groups: list[dict[str, Any]] = []
    for member_ids in sorted(groups_by_root.values()):
        stratum_counts = Counter(by_id[video_id]["compact_metadata"]["stratum"] for video_id in member_ids)
        stratum = sorted(stratum_counts, key=lambda item: (-stratum_counts[item], item))[0]
        group_id = "g02-" + hashlib.sha256(
            "\n".join(member_ids).encode("utf-8")
        ).hexdigest()[:20]
        splits = sorted({by_id[video_id]["split"] for video_id in member_ids})
        groups.append(
            {
                "group_id": group_id,
                "video_ids": member_ids,
                "video_count": len(member_ids),
                "splits": splits,
                "stratum": stratum,
                "stratum_counts": dict(sorted(stratum_counts.items())),
                "contains_nontrain_member": any(split != "train" for split in splits),
            }
        )
        for video_id in member_ids:
            by_id[video_id]["group_id"] = group_id
    return groups, sorted(
        checks,
        key=lambda check: (
            check.get("state", ""),
            check.get("left_video_id", ""),
            check.get("right_video_id", ""),
            check.get("bucket_key", ""),
        ),
    )


def build_dataset_registry(
    zip_path: Path,
    *,
    signature_chunks_per_video: int = MAX_SIGNATURE_CHUNKS_PER_VIDEO,
    max_uncompressed_chunk_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Build a bounded video registry without extracting full videos.

    At most ``signature_chunks_per_video`` image members are sampled per video.
    Each sample streams no more than ``max_uncompressed_chunk_bytes`` after ZIP
    decompression; oversized members contribute a bounded prefix.  GEFF and
    image metadata reads are independently capped.
    """

    if not 1 <= signature_chunks_per_video <= MAX_SIGNATURE_CHUNKS_PER_VIDEO:
        raise IntegrityError(
            "signature_chunks_per_video must be between 1 and "
            f"{MAX_SIGNATURE_CHUNKS_PER_VIDEO}"
        )
    if not 0 < max_uncompressed_chunk_bytes <= 1024 * 1024:
        raise IntegrityError(
            "max_uncompressed_chunk_bytes must be between 1 and 1048576"
        )
    if not zip_path.is_file():
        raise IntegrityError(f"ZIP path does not exist: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        central_directory = [
            {
                "filename": info.filename,
                "crc32": f"{info.CRC:08x}",
                "compressed_size": int(info.compress_size),
                "uncompressed_size": int(info.file_size),
            }
            for info in sorted(infos, key=lambda item: item.filename)
        ]
        raw_videos: dict[str, dict[str, Any]] = {}
        for info in infos:
            parsed = _parse_video_member(info)
            if parsed is None:
                continue
            video_id, split, kind, relative_member_path = parsed
            record = raw_videos.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "split": split,
                    "dataset": video_id.split(":", 1)[1],
                    "members": [],
                },
            )
            record["members"].append((kind, relative_member_path, info))

        train_videos = sum(record["split"] == "train" for record in raw_videos.values())
        test_videos = sum(record["split"] == "test" for record in raw_videos.values())
        total_videos = len(raw_videos)
        if train_videos > MAX_TRAIN_VIDEOS:
            raise IntegrityError(
                "ZIP contains "
                f"{train_videos} training videos; G-02 is bounded to "
                f"{MAX_TRAIN_VIDEOS} training videos"
            )
        if test_videos > MAX_TEST_VIDEOS:
            raise IntegrityError(
                "ZIP contains "
                f"{test_videos} test videos; G-02 is bounded to "
                f"{MAX_TEST_VIDEOS} test videos"
            )
        if total_videos > MAX_TOTAL_VIDEOS:
            raise IntegrityError(
                "ZIP contains "
                f"{total_videos} videos; G-02 is bounded to "
                f"{MAX_TOTAL_VIDEOS} total videos"
            )

        records: list[dict[str, Any]] = []
        actual_metadata_bytes = 0
        actual_chunk_bytes = 0
        for video_id, raw in sorted(raw_videos.items()):
            members = sorted(raw["members"], key=lambda item: (item[0], item[1]))
            member_vector = [
                _member_descriptor(info, kind, relative_path)
                for kind, relative_path, info in members
            ]
            metadata_candidates = [
                item for item in members if _is_metadata_member(item[1])
            ][:MAX_METADATA_MEMBERS_PER_VIDEO]
            metadata_members: list[dict[str, Any]] = []
            parsed_metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
            for kind, relative_path, info in metadata_candidates:
                metadata_entry = {
                    "artifact": kind,
                    "member_path": relative_path,
                    "uncompressed_size": int(info.file_size),
                }
                if info.file_size > MAX_METADATA_MEMBER_BYTES:
                    metadata_entry["state"] = "oversized"
                    metadata_members.append(metadata_entry)
                    continue
                try:
                    payload = archive.read(info)
                    actual_metadata_bytes += len(payload)
                    if len(payload) != info.file_size:
                        raise IntegrityError("ZIP member length did not match central directory")
                    metadata_entry["sha256"] = hashlib.sha256(payload).hexdigest()
                    parsed = json.loads(payload.decode("utf-8"))
                    if not isinstance(parsed, Mapping):
                        raise IntegrityError("metadata JSON root must be an object")
                    parsed_metadata[(kind, relative_path)] = parsed
                    metadata_entry["state"] = "selected"
                    metadata_entry["summary"] = _metadata_summary(parsed)
                except (UnicodeDecodeError, json.JSONDecodeError, IntegrityError, OSError, zipfile.BadZipFile) as exc:
                    metadata_entry["state"] = "unreadable"
                    metadata_entry["error"] = type(exc).__name__
                metadata_members.append(metadata_entry)

            image_chunks = [
                item
                for item in members
                if item[0] == "image" and not _is_metadata_member(item[1])
            ]
            selected_chunks: list[dict[str, Any]] = []
            signature_errors: list[str] = []
            for index in _selected_indices(len(image_chunks), signature_chunks_per_video):
                _kind, relative_path, info = image_chunks[index]
                selected = {
                    "member_path": relative_path,
                    "uncompressed_size": int(info.file_size),
                    "declared_uncompressed_size": int(info.file_size),
                    "central_directory_crc32": f"{info.CRC:08x}",
                }
                try:
                    with archive.open(info) as image_member:
                        payload = image_member.read(max_uncompressed_chunk_bytes)
                    if len(payload) > max_uncompressed_chunk_bytes:
                        raise IntegrityError("ZIP stream exceeded bounded image read limit")
                    actual_chunk_bytes += len(payload)
                    if not payload:
                        raise IntegrityError("empty image chunk cannot provide a duplicate signature")
                    if (
                        info.file_size <= max_uncompressed_chunk_bytes
                        and len(payload) != info.file_size
                    ):
                        raise IntegrityError("ZIP member length did not match central directory")
                    selected["state"] = "selected"
                    selected["sampling_mode"] = (
                        "full"
                        if info.file_size <= max_uncompressed_chunk_bytes
                        else "prefix"
                    )
                    selected["truncated"] = info.file_size > max_uncompressed_chunk_bytes
                    selected["sampled_bytes"] = len(payload)
                    selected["signature"] = _chunk_signature(payload)
                except (IntegrityError, OSError, zipfile.BadZipFile) as exc:
                    selected["state"] = "unreadable"
                    selected["error"] = type(exc).__name__
                    signature_errors.append(f"unreadable:{relative_path}")
                selected_chunks.append(selected)
            if not image_chunks:
                signature_errors.append("missing_image_chunks")
            if not selected_chunks:
                signature_errors.append("missing_selected_chunks")
            if any(chunk["state"] != "selected" for chunk in selected_chunks):
                signature_errors.append("incomplete_selected_chunks")
            complete_chunks = [
                chunk for chunk in selected_chunks if chunk["state"] == "selected"
            ]
            signature = {
                "status": "complete" if not signature_errors else "unresolved",
                "reason": ";".join(sorted(set(signature_errors))) if signature_errors else None,
                "chunks": complete_chunks,
                "selected_chunks": selected_chunks,
            }
            compact = _compact_metadata(parsed_metadata, raw["dataset"])
            exact_identity = manifest_sha256(member_vector)
            records.append(
                {
                    "video_id": video_id,
                    "split": raw["split"],
                    "dataset": raw["dataset"],
                    "member_vector": member_vector,
                    "exact_identity_sha256": exact_identity,
                    "selected_metadata_members": metadata_members,
                    "compact_metadata": compact,
                    "signature": signature,
                }
            )

    if not records:
        raise IntegrityError("ZIP central directory contains no train/test .zarr videos")
    groups, near_checks = _group_records(records)
    unresolved_checks = [check for check in near_checks if check["state"] == "unresolved"]
    selected_chunk_count = sum(len(record["signature"]["selected_chunks"]) for record in records)
    unresolved_chunk_count = sum(
        1
        for record in records
        for chunk in record["signature"]["selected_chunks"]
        if chunk["state"] != "selected"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "dataset_registry_v1",
        "parameters": {
            "signature_chunks_per_video": signature_chunks_per_video,
            "max_uncompressed_chunk_bytes": max_uncompressed_chunk_bytes,
            "max_metadata_member_bytes": MAX_METADATA_MEMBER_BYTES,
            "max_metadata_members_per_video": MAX_METADATA_MEMBERS_PER_VIDEO,
            "max_near_candidate_pairs_per_bucket": MAX_NEAR_CANDIDATE_PAIRS_PER_BUCKET,
            "max_train_videos": MAX_TRAIN_VIDEOS,
            "max_test_videos": MAX_TEST_VIDEOS,
            "max_total_videos": MAX_TOTAL_VIDEOS,
        },
        "source": {
            "zip_path": str(zip_path),
            "zip_central_directory_sha256": manifest_sha256(central_directory),
            "zip_entry_count": len(central_directory),
        },
        "source_hashes": source_hashes(),
        "records": sorted(records, key=lambda record: record["video_id"]),
        "groups": sorted(groups, key=lambda group: group["group_id"]),
        "near_duplicate_checks": near_checks,
        "counts": {
            "videos": len(records),
            "train_videos": sum(record["split"] == "train" for record in records),
            "test_videos": sum(record["split"] == "test" for record in records),
            "groups": len(groups),
            "selected_signature_chunks": selected_chunk_count,
            "unresolved_signature_chunks": unresolved_chunk_count,
            "unresolved_candidate_duplicate_checks": len(unresolved_checks),
        },
        "bounded_reads": {
            "central_directory_only_for_identity": True,
            "actual_metadata_bytes_read": actual_metadata_bytes,
            "actual_image_chunk_bytes_read": actual_chunk_bytes,
            "maximum_metadata_bytes": len(records)
            * MAX_METADATA_MEMBERS_PER_VIDEO
            * MAX_METADATA_MEMBER_BYTES,
            "maximum_image_chunk_bytes": len(records)
            * signature_chunks_per_video
            * max_uncompressed_chunk_bytes,
            "max_train_videos": MAX_TRAIN_VIDEOS,
            "max_test_videos": MAX_TEST_VIDEOS,
            "max_total_videos": MAX_TOTAL_VIDEOS,
            "maximum_videos": MAX_TOTAL_VIDEOS,
            "maximum_videos_scope": "total videos across train and test",
            "full_video_extraction": False,
        },
        "grouping_status": "blocked" if unresolved_checks else "ready",
    }


def _require_registry_ready(registry: Mapping[str, Any]) -> None:
    if registry.get("artifact") != "dataset_registry_v1":
        raise IntegrityError("expected a dataset_registry_v1 manifest")
    unresolved = [
        check
        for check in registry.get("near_duplicate_checks", [])
        if check.get("state") == "unresolved"
    ]
    if unresolved:
        raise IntegrityError(
            "cannot freeze splits while candidate duplicate signatures are unresolved"
        )


def _seed_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _stratum_targets(groups: Sequence[Mapping[str, Any]], audit_videos: int) -> tuple[list[str], tuple[int, ...]]:
    population = Counter()
    for group in groups:
        population[str(group["stratum"])] += int(group["video_count"])
    total = sum(population.values())
    if not total:
        raise IntegrityError("no eligible training videos are available for audit selection")
    strata = sorted(population)
    targets = {stratum: audit_videos * population[stratum] // total for stratum in strata}
    remaining = audit_videos - sum(targets.values())
    remainders = sorted(
        strata,
        key=lambda stratum: (-(audit_videos * population[stratum] % total), stratum),
    )
    for stratum in remainders[:remaining]:
        targets[stratum] += 1
    return strata, tuple(targets[stratum] for stratum in strata)


def _select_exact_stratified_audit(
    groups: Sequence[Mapping[str, Any]], audit_videos: int, seed: str
) -> list[str]:
    strata, targets = _stratum_targets(groups, audit_videos)
    stratum_index = {stratum: index for index, stratum in enumerate(strata)}
    ordered_groups = sorted(
        groups,
        key=lambda group: (_seed_rank(seed, str(group["group_id"])), str(group["group_id"])),
    )
    zero = tuple(0 for _ in strata)
    states: dict[tuple[int, tuple[int, ...]], tuple[str, ...]] = {(0, zero): ()}
    for group in ordered_groups:
        size = int(group["video_count"])
        stratum = str(group["stratum"])
        index = stratum_index[stratum]
        next_states = dict(states)
        for (count, distribution), selected in states.items():
            if count + size > audit_videos:
                continue
            updated = list(distribution)
            updated[index] += size
            key = (count + size, tuple(updated))
            candidate = selected + (str(group["group_id"]),)
            existing = next_states.get(key)
            if existing is None or candidate < existing:
                next_states[key] = candidate
        if len(next_states) > MAX_AUDIT_DP_STATES:
            raise IntegrityError("bounded audit selector exceeded its deterministic state limit")
        states = next_states
    candidates = [
        (distribution, selected)
        for (count, distribution), selected in states.items()
        if count == audit_videos
    ]
    if not candidates:
        raise IntegrityError(
            f"cannot reserve exactly {audit_videos} audit videos without splitting a group"
        )
    _, selected = min(
        candidates,
        key=lambda item: (
            sum(abs(actual - target) for actual, target in zip(item[0], targets, strict=True)),
            item[1],
        ),
    )
    return sorted(selected)


def build_grouped_split_manifest(
    registry: Mapping[str, Any],
    *,
    fold_count: int = 3,
    audit_videos: int = 20,
    seed: str = "g02-split-v1",
) -> dict[str, Any]:
    """Create an exact-size audit and grouped folds from a ready registry."""

    _require_registry_ready(registry)
    if fold_count != 3:
        raise IntegrityError("G-02 requires exactly three grouped folds")
    if audit_videos != 20:
        raise IntegrityError("G-02 requires exactly 20 untouched audit videos")
    groups = [dict(group) for group in registry.get("groups", [])]
    eligible = [
        group
        for group in groups
        if group.get("splits") == ["train"] and not group.get("contains_nontrain_member")
    ]
    excluded = [group for group in groups if group not in eligible]
    audit_group_ids = _select_exact_stratified_audit(eligible, audit_videos, seed)
    audit_group_set = set(audit_group_ids)
    remainder = [group for group in eligible if group["group_id"] not in audit_group_set]
    if len(remainder) < fold_count:
        raise IntegrityError("fewer than three non-audit groups remain for grouped CV")

    folds = [
        {"group_ids": [], "video_count": 0, "strata": Counter()}
        for _ in range(fold_count)
    ]
    for group in sorted(
        remainder,
        key=lambda item: (
            -int(item["video_count"]),
            _seed_rank(seed, str(item["group_id"])),
            str(item["group_id"]),
        ),
    ):
        stratum = str(group["stratum"])
        destination = min(
            range(fold_count),
            key=lambda index: (
                folds[index]["strata"][stratum],
                folds[index]["video_count"],
                len(folds[index]["group_ids"]),
                index,
            ),
        )
        folds[destination]["group_ids"].append(group["group_id"])
        folds[destination]["video_count"] += int(group["video_count"])
        folds[destination]["strata"][stratum] += int(group["video_count"])
    if any(not fold["group_ids"] for fold in folds):
        raise IntegrityError("grouped CV assignment produced an empty fold")

    group_by_id = {group["group_id"]: group for group in eligible}
    audit_video_ids = sorted(
        video_id
        for group_id in audit_group_ids
        for video_id in group_by_id[group_id]["video_ids"]
    )
    assignments = {group_id: "audit" for group_id in audit_group_ids}
    fold_payload: dict[str, Any] = {}
    non_audit_group_ids = {group["group_id"] for group in remainder}
    for index, fold in enumerate(folds):
        held_out_ids = sorted(fold["group_ids"])
        held_out_set = set(held_out_ids)
        fit_ids = sorted(non_audit_group_ids - held_out_set)
        assignments.update({group_id: f"fold_{index}" for group_id in held_out_ids})
        fold_payload[str(index)] = {
            "held_out_group_ids": held_out_ids,
            "held_out_video_ids": sorted(
                video_id
                for group_id in held_out_ids
                for video_id in group_by_id[group_id]["video_ids"]
            ),
            "fit_group_ids": fit_ids,
            "fit_video_ids": sorted(
                video_id
                for group_id in fit_ids
                for video_id in group_by_id[group_id]["video_ids"]
            ),
            "held_out_video_count": fold["video_count"],
            "held_out_strata": dict(sorted(fold["strata"].items())),
        }
    strata, targets = _stratum_targets(eligible, audit_videos)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "grouped_3fold_audit20_v1",
        "parameters": {
            "fold_count": fold_count,
            "audit_videos": audit_videos,
            "seed": seed,
            "grouping_policy": "exact_identity_and_bounded_near_signature_union",
        },
        "dataset_registry_sha256": manifest_sha256(registry),
        "source_hashes": source_hashes(),
        "audit": {
            "group_ids": audit_group_ids,
            "video_ids": audit_video_ids,
            "video_count": len(audit_video_ids),
            "untouched": True,
        },
        "folds": fold_payload,
        "group_assignments": dict(sorted(assignments.items())),
        "strata": {
            "names": strata,
            "audit_target_video_counts": dict(zip(strata, targets, strict=True)),
            "eligible_group_count": len(eligible),
            "excluded_group_ids": sorted(group["group_id"] for group in excluded),
        },
        "counts": {
            "eligible_train_groups": len(eligible),
            "excluded_nontrain_or_mixed_groups": len(excluded),
            "audit_videos": len(audit_video_ids),
            "remainder_videos": sum(fold["video_count"] for fold in folds),
        },
    }


def build_fit_provenance(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Emit the only group identities permitted for each fitting activity."""

    if split_manifest.get("artifact") != "grouped_3fold_audit20_v1":
        raise IntegrityError("expected a grouped_3fold_audit20_v1 manifest")
    audit_groups = set(split_manifest["audit"]["group_ids"])
    folds = split_manifest.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {"0", "1", "2"}:
        raise IntegrityError("split manifest must contain folds 0, 1, and 2")
    provenance_folds: dict[str, Any] = {}
    for fold_id, fold in sorted(folds.items()):
        held_out = set(fold["held_out_group_ids"])
        fit = set(fold["fit_group_ids"])
        if audit_groups & held_out or audit_groups & fit or held_out & fit:
            raise IntegrityError(f"fold {fold_id} has audit/holdout fitting leakage")
        sources = sorted(fit)
        provenance_folds[str(fold_id)] = {
            "held_out_group_ids": sorted(held_out),
            "forbidden_fit_group_ids": sorted(audit_groups | held_out),
            "fit_sources": {
                "training": sources,
                "calibration": sources,
                "router": sources,
                "threshold": sources,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "fit_provenance_v1",
        "grouped_split_sha256": manifest_sha256(split_manifest),
        "dataset_registry_sha256": split_manifest["dataset_registry_sha256"],
        "source_hashes": source_hashes(),
        "policy": (
            "audit and held-out fold groups are forbidden from training, calibration, "
            "router, and threshold fitting"
        ),
        "audit_group_ids": sorted(audit_groups),
        "folds": provenance_folds,
    }


def assert_fit_sources_allowed(
    provenance: Mapping[str, Any],
    fold_id: str,
    role: str,
    source_group_ids: Iterable[str],
) -> None:
    """Fail closed when a later fitting call supplies a forbidden group."""

    if role not in {"training", "calibration", "router", "threshold"}:
        raise IntegrityError(f"unknown fitting role: {role}")
    try:
        fold = provenance["folds"][str(fold_id)]
        allowed = set(fold["fit_sources"][role])
    except (KeyError, TypeError) as exc:
        raise IntegrityError(f"fit provenance lacks fold {fold_id!r}") from exc
    supplied = set(source_group_ids)
    unsafe = supplied - allowed
    if unsafe:
        raise IntegrityError(
            f"{role} fitting includes forbidden groups for fold {fold_id}: "
            + ", ".join(sorted(unsafe))
        )


def build_scorer_parity_report() -> dict[str, Any]:
    """Run the existing official wrapper fixtures and freeze their identities."""

    from .scorer_parity import run_scorer_parity

    fixture_report = dict(run_scorer_parity())
    fixture_report["official_src"] = (
        "external/kaggle-cell-tracking-competition/src"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "scorer_parity_v1",
        "scorer_reference": scorer_reference_identities(),
        "source_hashes": source_hashes(),
        "fixture_report": fixture_report,
        "calibration_pair_ledger": {
            "status": "no_observed_kaggle_pairs",
            "pairs": [],
            "local_score": None,
            "site_score": None,
            "residual": None,
        },
    }


def freeze_integrity_outputs(
    zip_path: Path,
    output_dir: Path,
    *,
    fold_count: int = 3,
    audit_videos: int = 20,
    signature_chunks_per_video: int = MAX_SIGNATURE_CHUNKS_PER_VIDEO,
    max_uncompressed_chunk_bytes: int = 1024 * 1024,
    seed: str = "g02-split-v1",
) -> dict[str, Any]:
    """Build and atomically write all four deterministic G-02 manifests."""

    parity = build_scorer_parity_report()
    registry = build_dataset_registry(
        zip_path,
        signature_chunks_per_video=signature_chunks_per_video,
        max_uncompressed_chunk_bytes=max_uncompressed_chunk_bytes,
    )
    split = build_grouped_split_manifest(
        registry,
        fold_count=fold_count,
        audit_videos=audit_videos,
        seed=seed,
    )
    provenance = build_fit_provenance(split)
    provenance["related_manifest_hashes"] = {
        "scorer_parity_v1.json": manifest_sha256(parity),
        "dataset_registry_v1.json": manifest_sha256(registry),
        "grouped_3fold_audit20_v1.json": manifest_sha256(split),
    }
    artifacts = {
        "scorer_parity_v1.json": parity,
        "dataset_registry_v1.json": registry,
        "grouped_3fold_audit20_v1.json": split,
        "fit_provenance_v1.json": provenance,
    }
    hashes = {
        name: atomic_write_json(output_dir / name, artifact)
        for name, artifact in artifacts.items()
    }
    return {
        "output_dir": str(output_dir),
        "artifact_sha256": dict(sorted(hashes.items())),
        "dataset_registry_sha256": manifest_sha256(registry),
        "grouped_split_sha256": manifest_sha256(split),
        "fit_provenance_sha256": manifest_sha256(provenance),
        "scorer_parity_sha256": manifest_sha256(parity),
    }
