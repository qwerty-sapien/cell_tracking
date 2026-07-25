#!/usr/bin/env python3
"""Production runner for the frozen V45 BioHub detector and sparse graph.

The module is intentionally import-safe: torch, SciPy, Zarr, and the tracking
modules are imported only by preflight or execution entry points.  Submission
mode opens image Zarrs only.  A local competition archive is read in place
through :class:`zarr.storage.ZipStore`; it is never extracted.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "reports/g02/dataset_registry_v1.json"
DEFAULT_SPLITS = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_EDGE_AUXILIARY_CHECKPOINT = ROOT / "reports/g02/v45f4.pt"

RUNNER_SCHEMA = "celltrack-alpha.bh-runner.v1"
CACHE_META_SCHEMA = "celltrack-alpha.bh-cache-meta.v1"
CACHE_INTENT_SCHEMA = "celltrack-alpha.bh-cache-intent.v1"
STATUS_SCHEMA = "celltrack-alpha.bh-video-status.v1"
TRANSACTION_SCHEMA = "celltrack-alpha.bh-video-transaction.v1"
FEATURE_SCHEMA = "celltrack-alpha.bh-video-features.v1"
RUNG_SCHEMA = "celltrack-alpha.bh-cv-rungs.v1"
PREFLIGHT_SCHEMA = "celltrack-alpha.bh-preflight.v1"
ACCEL_GATE_SCHEMA = "celltrack-alpha.bh-d4-accelerated-score-gate.v1"
A0_AUDIT_SCHEMA = "celltrack-alpha.bh-a0-run-audit.v1"
RUNTIME_ZARR_IDENTITY_SCHEMA = "celltrack-alpha.bh-runtime-zarr-identity.v1"
P0_CALIBRATION_SCHEMA = "celltrack-alpha.p0-public-calibration.v1"
REGISTRY_EXACT_IDENTITY = "registry-exact-image-manifest-v1"
RUNTIME_METADATA_IDENTITY = "runtime-zarr-metadata-v1"
MAX_RUNTIME_METADATA_BYTES = 1_000_000

FRAME_COUNT = 100
DEFAULT_BATCH_SIZE = 1
DEFAULT_DEADLINE_MINUTES = 360.0
SOURCE_BUDGET_BYTES = 1_000_000
CHECKPOINT_EXPOSURE = "unknown"
IMAGE_SUPPORT_SCALE_ZYX = (1.625, 0.40625, 0.40625)
IMAGE_SUPPORT_MAX_CORRECTION_UM = 3.2
IMAGE_SUPPORT_EPSILON_UM = 1.0e-12

MOTION_ROUTE = "motion"
LOCKED_GRAPH_VARIANT = "v45-motion-lazyrefine-locked"
A0_GRAPH_VARIANT = "a0"
PRODUCTION_D4_VIEWS = tuple(range(8))
TEST_ONLY_ONE_VIEW = (0,)
A0_GATE_RADIUS_UM = 10.0
A0_GATE_NEIGHBOR_CAP = 8
GRAPH_VARIANTS = {
    LOCKED_GRAPH_VARIANT: {
        "learned_edge_probabilities": None,
        "image_refinement": "lazy-p20-intensity-centroid",
        "config": "V45_GRAPH_CONFIG",
        "locked": True,
    },
    A0_GRAPH_VARIANT: {
        "learned_edge_probabilities": "a0-original-candidate-coo",
        "image_refinement": "lazy-p20-intensity-centroid",
        "config": "V45_GRAPH_CONFIG",
        "checkpoint_scorer": "A0CheckpointScorer.from_loaded_model",
        "scorer_lifecycle": "once-per-worker-after-load_v45_model",
        "cache_inputs": ["FrameDet", "PairContext"],
        "adjacent_contexts": "all-exactly-once",
        "candidate_gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": A0_GATE_RADIUS_UM,
            "neighbor_cap": A0_GATE_NEIGHBOR_CAP,
            "scale_zyx": list(IMAGE_SUPPORT_SCALE_ZYX),
        },
        "probability_merge": "original-coo-duplicate-free",
        "full_frame_dense_matrix": False,
        "audit_schema": A0_AUDIT_SCHEMA,
        "default": False,
        "locked": True,
    },
}
P0_APPROVED_MOTION_WEIGHTS = {
    "p0-c0": 0.0,
    "p0-c2": 2.0,
}

CSV_FIELDS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)


class RunnerError(RuntimeError):
    """Base fail-closed runner error."""


class PreflightError(RunnerError):
    """The pinned runtime is unavailable or incompatible."""


class DiscoveryError(RunnerError):
    """Requested registry identities do not map one-to-one to image Zarrs."""


class DeadlineError(RunnerError):
    """The hard global deadline was reached."""


class ResumeHashError(RunnerError):
    """A fragment, status, or cache belongs to another immutable run."""


class OutputValidationError(RunnerError):
    """A graph fragment or merged submission violates the public schema."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    source = Path(path)
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid JSON in {source}: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_slug(split: str, dataset: str) -> str:
    value = f"{split}__{dataset}"
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise DiscoveryError(f"unsafe dataset identity {split}:{dataset}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: str | os.PathLike[str], payload: bytes, deadline_epoch: float | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        check_deadline(deadline_epoch, f"{destination.name} finalization")
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
        check_deadline(deadline_epoch, f"{destination.name} finalization")
    except Exception:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def atomic_json(path: str | os.PathLike[str], value: Any, deadline_epoch: float | None = None) -> None:
    atomic_bytes(path, canonical_json(value) + b"\n", deadline_epoch)


def atomic_create_bytes(
    path: str | os.PathLike[str], payload: bytes, deadline_epoch: float | None = None
) -> None:
    """Atomically publish a new immutable file without replacing an existing one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        check_deadline(deadline_epoch, f"{destination.name} finalization")
        os.link(temporary, destination)
        os.unlink(temporary)
        temporary = None
        _fsync_directory(destination.parent)
        check_deadline(deadline_epoch, f"{destination.name} finalization")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def atomic_create_json(
    path: str | os.PathLike[str], value: Any, deadline_epoch: float | None = None
) -> None:
    atomic_create_bytes(path, canonical_json(value) + b"\n", deadline_epoch)


def check_deadline(deadline_epoch: float | None, stage: str) -> None:
    if deadline_epoch is not None and time.time() >= float(deadline_epoch):
        raise DeadlineError(f"global deadline reached during {stage}")


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    split: str
    dataset: str
    video_id: str
    exact_identity_sha256: str
    image_shape: tuple[int, int, int, int]
    identity_provenance: str = REGISTRY_EXACT_IDENTITY

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegistryRecord":
        try:
            split = str(value["split"])
            dataset = str(value["dataset"])
            video_id = str(value["video_id"])
            digest = str(value["exact_identity_sha256"])
            shape = tuple(value["compact_metadata"]["image_shape"])
        except (KeyError, TypeError) as exc:
            raise DiscoveryError("registry record lacks a required image identity field") from exc
        if split not in {"train", "test"} or video_id != f"{split}:{dataset}":
            raise DiscoveryError(f"invalid registry identity {video_id!r}")
        _safe_slug(split, dataset)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DiscoveryError(f"invalid exact identity hash for {video_id}")
        if len(shape) != 4 or any(type(item) is not int or item <= 0 for item in shape):
            raise DiscoveryError(f"invalid registered image shape for {video_id}")
        return cls(split, dataset, video_id, digest, shape)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class VideoJob:
    record: RegistryRecord
    input_kind: str
    input_path: str
    zarr_path: str

    @property
    def identity(self) -> str:
        return self.record.video_id

    @property
    def slug(self) -> str:
        return _safe_slug(self.record.split, self.record.dataset)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record": {
                **asdict(self.record),
                "image_shape": list(self.record.image_shape),
            },
            "input_kind": self.input_kind,
            "input_path": self.input_path,
            "zarr_path": self.zarr_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VideoJob":
        raw = dict(value["record"])
        raw["image_shape"] = tuple(raw["image_shape"])
        record = RegistryRecord(**raw)
        return cls(record, str(value["input_kind"]), str(value["input_path"]), str(value["zarr_path"]))


def load_registry(path: str | os.PathLike[str] = DEFAULT_REGISTRY) -> tuple[RegistryRecord, ...]:
    payload = strict_json_load(path)
    if not isinstance(payload, Mapping) or payload.get("artifact") != "dataset_registry_v1":
        raise DiscoveryError("registry is not dataset_registry_v1")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise DiscoveryError("registry has no records")
    records = tuple(RegistryRecord.from_mapping(item) for item in raw_records if isinstance(item, Mapping))
    if len(records) != len(raw_records):
        raise DiscoveryError("registry contains a non-object record")
    identities = [record.video_id for record in records]
    if len(set(identities)) != len(identities):
        raise DiscoveryError("registry contains duplicate video identities")
    counts = payload.get("counts")
    if isinstance(counts, Mapping) and "videos" in counts and counts["videos"] != len(records):
        raise DiscoveryError("registry video count does not match records")
    return records


def select_records(
    records: Sequence[RegistryRecord],
    *,
    split: str,
    requested: Sequence[str] | None = None,
) -> tuple[RegistryRecord, ...]:
    if split not in {"train", "test"}:
        raise DiscoveryError("split must be train or test")
    available = {record.video_id: record for record in records if record.split == split}
    by_dataset = {record.dataset: record for record in records if record.split == split}
    if requested is None:
        selected = list(available.values())
    else:
        values = [str(item).strip() for item in requested]
        if any(not item for item in values) or len(set(values)) != len(values):
            raise DiscoveryError("requested datasets must be non-empty and unique")
        selected = []
        missing = []
        for value in values:
            record = available.get(value) if ":" in value else by_dataset.get(value)
            if record is None:
                missing.append(value)
            else:
                selected.append(record)
        if missing:
            raise DiscoveryError(f"requested datasets are absent from the exact registry: {missing}")
    if not selected:
        raise DiscoveryError("no datasets selected")
    if len({record.video_id for record in selected}) != len(selected):
        raise DiscoveryError("selected registry identities are not one-to-one")
    return tuple(sorted(selected, key=lambda record: record.video_id))


def _zip_image_counts(archive: Path, records: Sequence[RegistryRecord]) -> dict[str, int]:
    wanted = {f"{record.split}/{record.dataset}.zarr/0/zarr.json": record.video_id for record in records}
    counts = {identity: 0 for identity in wanted.values()}
    try:
        with zipfile.ZipFile(archive, mode="r") as handle:
            for item in handle.infolist():
                identity = wanted.get(item.filename)
                if identity is not None:
                    counts[identity] += 1
    except (OSError, zipfile.BadZipFile) as exc:
        raise DiscoveryError(f"invalid competition ZIP {archive}: {exc}") from exc
    return counts


def discover_jobs(input_path: str | os.PathLike[str], records: Sequence[RegistryRecord]) -> tuple[VideoJob, ...]:
    """Map every exact registry identity to one image Zarr without reading labels."""

    source = Path(input_path).resolve()
    if not records:
        raise DiscoveryError("job discovery requires at least one selected registry record")
    jobs: list[VideoJob] = []
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise DiscoveryError(f"local input file is not a ZIP archive: {source}")
        counts = _zip_image_counts(source, records)
        invalid = {identity: count for identity, count in counts.items() if count != 1}
        if invalid:
            raise DiscoveryError(f"requested ZIP image identities must occur exactly once: {invalid}")
        jobs = [
            VideoJob(record, "zip", str(source), f"{record.split}/{record.dataset}.zarr")
            for record in records
        ]
    elif source.is_dir():
        direct = source.name.endswith(".zarr")
        if direct and len(records) != 1:
            raise DiscoveryError("a direct Zarr directory can satisfy exactly one requested dataset")
        for record in records:
            image = source if direct else source / record.split / f"{record.dataset}.zarr"
            metadata = image / "0" / "zarr.json"
            if not image.is_dir() or not metadata.is_file():
                raise DiscoveryError(f"registered directory Zarr is missing: {record.video_id}")
            jobs.append(VideoJob(record, "directory", str(source), str(image)))
    else:
        raise DiscoveryError(f"input path does not exist: {source}")
    identities = [job.identity for job in jobs]
    if len(identities) != len(records) or len(set(identities)) != len(identities):
        raise DiscoveryError("discovery lost or duplicated a requested dataset")
    return tuple(sorted(jobs, key=lambda job: job.identity))


def _runtime_metadata_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_RUNTIME_METADATA_BYTES:
            raise DiscoveryError(
                f"runtime Zarr metadata size must be in [1, {MAX_RUNTIME_METADATA_BYTES}]: {path}"
            )
        return path.read_bytes()
    except OSError as exc:
        raise DiscoveryError(f"cannot read runtime Zarr metadata {path}: {exc}") from exc


def _runtime_zip_metadata_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size <= 0 or info.file_size > MAX_RUNTIME_METADATA_BYTES:
        raise DiscoveryError(
            "runtime ZIP Zarr metadata size must be in "
            f"[1, {MAX_RUNTIME_METADATA_BYTES}]: {info.filename}"
        )
    try:
        payload = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DiscoveryError(f"cannot read runtime ZIP Zarr metadata {info.filename}: {exc}") from exc
    if len(payload) != info.file_size:
        raise DiscoveryError(f"runtime ZIP Zarr metadata length mismatch: {info.filename}")
    return payload


def _runtime_submit_record(dataset: str, group_metadata: bytes, array_metadata: bytes) -> RegistryRecord:
    _safe_slug("test", dataset)
    try:
        group = json.loads(
            group_metadata.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
        array = json.loads(
            array_metadata.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"invalid runtime Zarr metadata for test:{dataset}: {exc}") from exc
    if not isinstance(group, Mapping) or group.get("node_type") != "group":
        raise DiscoveryError(f"test:{dataset} root metadata is not a Zarr group")
    if not isinstance(array, Mapping) or array.get("node_type") != "array":
        raise DiscoveryError(f"test:{dataset} image metadata is not a Zarr array")
    raw_shape = array.get("shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 4
        or any(type(item) is not int or item <= 0 for item in raw_shape)
    ):
        raise DiscoveryError(f"test:{dataset} runtime image shape is not four positive integers")
    shape = tuple(raw_shape)
    identity = {
        "schema": RUNTIME_ZARR_IDENTITY_SCHEMA,
        "split": "test",
        "dataset": dataset,
        "metadata": [
            {
                "path": "zarr.json",
                "bytes": len(group_metadata),
                "sha256": hashlib.sha256(group_metadata).hexdigest(),
            },
            {
                "path": "0/zarr.json",
                "bytes": len(array_metadata),
                "sha256": hashlib.sha256(array_metadata).hexdigest(),
            },
        ],
    }
    return RegistryRecord(
        "test",
        dataset,
        f"test:{dataset}",
        canonical_sha256(identity),
        shape,  # type: ignore[arg-type]
        RUNTIME_METADATA_IDENTITY,
    )


def discover_submit_jobs(input_path: str | os.PathLike[str]) -> tuple[VideoJob, ...]:
    """Discover every mounted test image Zarr without relying on public test IDs.

    Kaggle code submissions are rerun against hidden test identities, so a
    packaged public registry cannot define submission membership. Only the two
    bounded Zarr metadata documents are read here; labels and image chunks are
    not opened during discovery.
    """

    source = Path(input_path).resolve()
    jobs: list[VideoJob] = []
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise DiscoveryError(f"local submission input is not a ZIP archive: {source}")
        try:
            with zipfile.ZipFile(source, mode="r") as archive:
                by_name: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
                for info in archive.infolist():
                    by_name[info.filename].append(info)
                array_names = []
                for name in by_name:
                    parts = name.split("/")
                    if (
                        len(parts) == 4
                        and parts[0] == "test"
                        and parts[1].endswith(".zarr")
                        and parts[2:] == ["0", "zarr.json"]
                    ):
                        array_names.append(name)
                for array_name in sorted(array_names):
                    dataset = array_name.split("/", 2)[1][:-5]
                    group_name = f"test/{dataset}.zarr/zarr.json"
                    group_records = by_name.get(group_name, [])
                    if len(by_name[array_name]) != 1 or not group_records:
                        raise DiscoveryError(
                            f"runtime ZIP test:{dataset} must have one image and at least one group metadata record"
                        )
                    # Zarr's ZipStore appends group metadata when attributes are
                    # updated. Zip readers resolve duplicate names to the final
                    # record, so mirror that behavior while still requiring one
                    # immutable array metadata record per test image.
                    group_metadata = _runtime_zip_metadata_bytes(archive, group_records[-1])
                    array_metadata = _runtime_zip_metadata_bytes(archive, by_name[array_name][0])
                    record = _runtime_submit_record(dataset, group_metadata, array_metadata)
                    jobs.append(
                        VideoJob(record, "zip", str(source), f"test/{dataset}.zarr")
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            raise DiscoveryError(f"invalid submission ZIP {source}: {exc}") from exc
    elif source.is_dir():
        test_root = source / "test"
        if not test_root.is_dir():
            raise DiscoveryError(f"submission input has no test directory: {source}")
        candidates = sorted(path for path in test_root.iterdir() if path.name.endswith(".zarr"))
        for image in candidates:
            dataset = image.name[:-5]
            if not image.is_dir():
                raise DiscoveryError(f"runtime test Zarr is not a directory: test:{dataset}")
            group_metadata = _runtime_metadata_bytes(image / "zarr.json")
            array_metadata = _runtime_metadata_bytes(image / "0" / "zarr.json")
            record = _runtime_submit_record(dataset, group_metadata, array_metadata)
            jobs.append(VideoJob(record, "directory", str(source), str(image)))
    else:
        raise DiscoveryError(f"submission input path does not exist: {source}")
    identities = [job.identity for job in jobs]
    if not jobs or len(set(identities)) != len(identities):
        raise DiscoveryError("runtime submission discovery found no test Zarrs or duplicate identities")
    return tuple(sorted(jobs, key=lambda job: job.identity))


def shard_jobs(jobs: Sequence[VideoJob], shard_count: int = 2) -> tuple[tuple[VideoJob, ...], ...]:
    """Deterministically balance whole videos; a video is never split."""

    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    shards: list[list[VideoJob]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    ordered = sorted(jobs, key=lambda job: (-math.prod(job.record.image_shape), job.identity))
    for job in ordered:
        rank = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[rank].append(job)
        loads[rank] += math.prod(job.record.image_shape)
    flat = [job.identity for shard in shards for job in shard]
    if len(flat) != len(jobs) or len(set(flat)) != len(flat):
        raise DiscoveryError("deterministic sharding lost or duplicated a video")
    return tuple(tuple(shard) for shard in shards)


def _module_version(module: Any, distribution: str) -> str:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else importlib.metadata.version(distribution)


def source_size_report(
    *,
    budget_bytes: int = SOURCE_BUDGET_BYTES,
    paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    if paths is None:
        paths = (
            Path(__file__).resolve(),
            ROOT / "src/celltrack_alpha/bh_model.py",
            ROOT / "src/celltrack_alpha/bh_graph.py",
            ROOT / "src/celltrack_alpha/bh_edge.py",
            ROOT / "src/celltrack_alpha/bh_route.py",
        )
    files = []
    total = 0
    for path in paths:
        if not path.is_file():
            raise PreflightError(f"required source package member is missing: {path}")
        size = path.stat().st_size
        total += size
        files.append({"path": str(path), "bytes": size, "sha256": sha256_file(path)})
    report = {
        "budget_bytes": int(budget_bytes),
        "source_bytes": total,
        "under_budget": total <= int(budget_bytes),
        "files": files,
    }
    if not report["under_budget"]:
        raise PreflightError(f"source package exceeds {budget_bytes} bytes: {total}")
    return report


def preflight(
    *,
    support_root: str | os.PathLike[str] = DEFAULT_SUPPORT,
    report_path: str | os.PathLike[str] | None = None,
    require_cuda: bool = True,
    source_budget_bytes: int = SOURCE_BUDGET_BYTES,
    pair_batch_size: int = DEFAULT_BATCH_SIZE,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Import and pin every runtime dependency before any image data read."""

    try:
        numpy = importlib.import_module("numpy")
        scipy = importlib.import_module("scipy")
        torch = importlib.import_module("torch")
        zarr = importlib.import_module("zarr")
        bh_model = importlib.import_module("celltrack_alpha.bh_model")
        bh_graph = importlib.import_module("celltrack_alpha.bh_graph")
        bh_edge = importlib.import_module("celltrack_alpha.bh_edge")
        bh_route = importlib.import_module("celltrack_alpha.bh_route")
    except (ImportError, ModuleNotFoundError) as exc:
        raise PreflightError(f"required import failed before data reads: {exc}") from exc

    versions = {
        "python": platform.python_version(),
        "torch": _module_version(torch, "torch"),
        "numpy": _module_version(numpy, "numpy"),
        "scipy": _module_version(scipy, "scipy"),
        "zarr": _module_version(zarr, "zarr"),
    }
    if sys.version_info < (3, 11) or int(versions["zarr"].split(".", 1)[0]) != 3:
        raise PreflightError(f"incompatible Python/Zarr versions: {versions}")
    if not hasattr(zarr, "storage") or not hasattr(zarr.storage, "ZipStore"):
        raise PreflightError("Zarr v3 runtime lacks storage.ZipStore")
    if (
        bh_model.V45_CONFIG.detection_tta != "D4-8-view"
        or bh_model.V45_CONFIG.edge_feature_tta is not False
        or bh_model.V45_CONFIG.pair_batch_size != 1
        or bh_model.V45_CONFIG.view_batch_size != 1
    ):
        raise PreflightError("bh_model is not the locked detection-only D4 V45 configuration")
    try:
        numerical_config_sha256 = bh_model.v45_config_sha(
            pair_batch_size=pair_batch_size,
            view_batch_size=view_batch_size,
            d4_views=d4_views,
        )
    except (AttributeError, ValueError) as exc:
        raise PreflightError(f"invalid pair/view batching mode: {exc}") from exc
    if bh_graph.V45_GRAPH_CONFIG is not bh_graph.V45_CONFIG:
        raise PreflightError("bh_graph locked configuration aliases disagree")
    if (
        bh_graph.V45_GRAPH_CONFIG.learned_bonus != 1.0
        or bh_graph.V45_GRAPH_CONFIG.raw_distance_weight != 0.05
        or bh_graph.V45_GRAPH_CONFIG.as_dict().get("learned_bonus") != 1.0
        or bh_graph.V45_GRAPH_CONFIG.as_dict().get("raw_distance_weight") != 0.05
    ):
        raise PreflightError("submit graph learned-bonus/distance constants are not locked")
    if (
        not hasattr(bh_route, "extract_features")
        or set(GRAPH_VARIANTS) != {LOCKED_GRAPH_VARIANT, A0_GRAPH_VARIANT}
        or not hasattr(bh_edge, "A0CheckpointScorer")
        or bh_edge.SparseEdgeConfig().radius_um != A0_GATE_RADIUS_UM
        or bh_edge.SparseEdgeConfig().neighbor_cap != A0_GATE_NEIGHBOR_CAP
        or tuple(bh_edge.SparseEdgeConfig().scale_zyx) != IMAGE_SUPPORT_SCALE_ZYX
        or bh_edge.A0ScorerConfig().neighbor_cap != A0_GATE_NEIGHBOR_CAP
    ):
        raise PreflightError("runner graph/route integration is incompatible")

    try:
        bundle = bh_model.validate_artifact_bundle(support_root)
    except Exception as exc:
        raise PreflightError(f"pinned V45 checkpoint validation failed: {exc}") from exc
    cuda_count = int(torch.cuda.device_count())
    if require_cuda and cuda_count < 2:
        raise PreflightError(f"production dual-GPU execution requires 2 CUDA devices, found {cuda_count}")
    offline_command = (
        "python -m pip install --no-index "
        "--find-links /kaggle/input/bh-v45-offline-wheels/wheels "
        + " ".join(f"{name}=={versions[name]}" for name in ("torch", "numpy", "scipy", "zarr"))
    )
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "ok": True,
        "versions": versions,
        "cuda_count": cuda_count,
        "cuda_required": bool(require_cuda),
        "checkpoint": {
            "path": str(bundle.weight_path),
            "bytes": bundle.weight_path.stat().st_size,
            "hashes": asdict(bundle.hashes),
            "model_config_sha256": bh_model.V45_CONFIG_SHA256,
            "exposure": CHECKPOINT_EXPOSURE,
        },
        "d4_execution": view_batch_report(
            pair_batch_size=pair_batch_size,
            view_batch_size=view_batch_size,
            accelerated_gate=None,
            require_accelerated_gate=False,
            d4_views=d4_views,
        ),
        "numerical_config_sha256": numerical_config_sha256,
        "offline_wheel_install_command": offline_command,
        "runtime_installation": "forbidden",
        "network": "forbidden",
        "source_package": source_size_report(budget_bytes=source_budget_bytes),
    }
    if report_path is not None:
        atomic_json(report_path, report)
    return report


def read_quantile_attrs(attrs: Mapping[str, Any]) -> tuple[float, float]:
    """Read the exact q0.001/q0.999 metadata; never estimate it from pixels."""

    try:
        nested = attrs.get("image_statistics", {}).get("quantiles", {})
        candidates = dict(nested) if isinstance(nested, Mapping) else {}
        if "q0.001" in attrs:
            candidates["0.001"] = attrs["q0.001"]
        if "q0.999" in attrs:
            candidates["0.999"] = attrs["q0.999"]
        values = {float(key.removeprefix("q")): float(value) for key, value in candidates.items()}
        low = next(value for key, value in values.items() if abs(key - 0.001) <= 1.0e-12)
        high = next(value for key, value in values.items() if abs(key - 0.999) <= 1.0e-12)
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise RunnerError("image attrs lack exact finite q0.001/q0.999 values") from exc
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise RunnerError("image q0.001/q0.999 values are not a finite increasing range")
    return low, high


def read_scale_attrs(attrs: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        multiscale = attrs["multiscales"][0]
        axes = [
            str(axis["name"] if isinstance(axis, Mapping) else axis).strip().casefold()
            for axis in multiscale["axes"]
        ]
        transformations = multiscale["datasets"][0]["coordinateTransformations"]
        scale = next(item["scale"] for item in transformations if item.get("type") == "scale")
        result = tuple(float(scale[axes.index(axis)]) for axis in ("z", "y", "x"))
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
        raise RunnerError("image attrs lack an unambiguous OME-NGFF z/y/x scale") from exc
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise RunnerError("OME-NGFF z/y/x scale must be positive and finite")
    return result  # type: ignore[return-value]


def submit_input_contract_report(
    jobs: Sequence[VideoJob],
    *,
    expected_frames: int = FRAME_COUNT,
) -> dict[str, Any]:
    """Validate hidden-test geometry and normalization metadata before inference.

    The three top-level flags are intentionally stable so a failed Kaggle
    rerun can be diagnosed from ``bhdiag.json`` without relaxing any locked
    V45 assumptions.
    """

    zarr = importlib.import_module("zarr")
    rows: list[dict[str, Any]] = []
    flags = {
        "frame_count_ne_100": False,
        "missing_quantile_or_multiscale_metadata": False,
        "voxel_scale_ne_locked": False,
    }
    for job in jobs:
        store = None
        owns_store = False
        row: dict[str, Any] = {
            "identity": job.identity,
            "image_shape": list(job.record.image_shape),
            "frame_count": int(job.record.image_shape[0]),
            "expected_frame_count": int(expected_frames),
        }
        row["frame_count_ok"] = row["frame_count"] == expected_frames
        flags["frame_count_ne_100"] |= not row["frame_count_ok"]
        try:
            if job.input_kind == "zip":
                store = zarr.storage.ZipStore(job.input_path, mode="r")
                owns_store = True
                group = zarr.open_group(
                    store=store,
                    path=job.zarr_path,
                    mode="r",
                    zarr_format=3,
                )
            elif job.input_kind == "directory":
                group = zarr.open_group(job.zarr_path, mode="r", zarr_format=3)
            else:
                raise DiscoveryError(f"unsupported input kind {job.input_kind!r}")
            attrs = dict(group.attrs)
            try:
                row["quantiles"] = list(read_quantile_attrs(attrs))
                row["quantile_metadata_ok"] = True
            except RunnerError as exc:
                row["quantile_metadata_ok"] = False
                row["quantile_error"] = str(exc)
            try:
                scale = read_scale_attrs(attrs)
                row["multiscale_metadata_ok"] = True
                row["voxel_scale_zyx"] = list(scale)
                row["voxel_scale_ok"] = all(
                    abs(left - right) <= 1.0e-9
                    for left, right in zip(scale, IMAGE_SUPPORT_SCALE_ZYX, strict=True)
                )
            except RunnerError as exc:
                row["multiscale_metadata_ok"] = False
                row["voxel_scale_ok"] = False
                row["multiscale_error"] = str(exc)
        finally:
            if owns_store and store is not None and callable(getattr(store, "close", None)):
                store.close()
        metadata_missing = not row.get("quantile_metadata_ok", False) or not row.get(
            "multiscale_metadata_ok", False
        )
        flags["missing_quantile_or_multiscale_metadata"] |= metadata_missing
        flags["voxel_scale_ne_locked"] |= (
            row.get("multiscale_metadata_ok") is True and row.get("voxel_scale_ok") is not True
        )
        row["ok"] = (
            row["frame_count_ok"]
            and not metadata_missing
            and row.get("voxel_scale_ok") is True
        )
        rows.append(row)
    triggered = [name for name, value in flags.items() if value]
    return {
        "schema": "celltrack-alpha.bh-submit-input-contract.v1",
        "ok": not triggered,
        "video_count": len(rows),
        "locked": {
            "frame_count": int(expected_frames),
            "voxel_scale_zyx": list(IMAGE_SUPPORT_SCALE_ZYX),
            "quantiles": [0.001, 0.999],
        },
        "fail_closed": flags,
        "triggered": triggered,
        "videos": rows,
    }


@dataclass(slots=True)
class VideoHandle:
    array: Any
    attrs: dict[str, Any]
    store: Any | None
    owns_store: bool

    def close(self) -> None:
        if self.owns_store and self.store is not None:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()


def _zarr_format(value: Any) -> int | None:
    metadata = getattr(value, "metadata", None)
    result = getattr(metadata, "zarr_format", None)
    return int(result) if result is not None else None


def open_video(
    job: VideoJob,
    *,
    expected_frames: int = FRAME_COUNT,
    shared_zip_store: Any | None = None,
) -> VideoHandle:
    """Open exactly one registered image path with Zarr v3."""

    zarr = importlib.import_module("zarr")
    store = shared_zip_store
    owns = False
    if job.input_kind == "zip":
        if store is None:
            store = zarr.storage.ZipStore(job.input_path, mode="r")
            owns = True
        group = zarr.open_group(store=store, path=job.zarr_path, mode="r", zarr_format=3)
        array = zarr.open_array(store=store, path=f"{job.zarr_path}/0", mode="r", zarr_format=3)
    elif job.input_kind == "directory":
        group = zarr.open_group(job.zarr_path, mode="r", zarr_format=3)
        array = zarr.open_array(str(Path(job.zarr_path) / "0"), mode="r", zarr_format=3)
    else:
        raise DiscoveryError(f"unsupported input kind {job.input_kind!r}")
    try:
        shape = tuple(int(value) for value in array.shape)
        if _zarr_format(group) not in {None, 3} or _zarr_format(array) not in {None, 3}:
            raise RunnerError(f"{job.identity} is not a Zarr v3 image")
        if shape != job.record.image_shape:
            raise RunnerError(f"registered/native shape mismatch for {job.identity}: {shape}")
        if shape[0] != expected_frames:
            raise RunnerError(f"{job.identity} must contain exactly {expected_frames} frames, got {shape[0]}")
        attrs = dict(group.attrs)
        read_quantile_attrs(attrs)
        scale = read_scale_attrs(attrs)
        locked = (1.625, 0.40625, 0.40625)
        if any(abs(left - right) > 1.0e-9 for left, right in zip(scale, locked, strict=True)):
            raise RunnerError(f"{job.identity} scale {scale} is incompatible with locked V45 graph {locked}")
        return VideoHandle(array, attrs, store, owns)
    except Exception:
        if owns and store is not None and callable(getattr(store, "close", None)):
            store.close()
        raise


def read_all_frames(array: Any, frame_count: int, deadline_epoch: float | None) -> list[Any]:
    """Read all frames one logical frame at a time, preserving original voxels."""

    numpy = importlib.import_module("numpy")
    frames = []
    for frame_index in range(frame_count):
        check_deadline(deadline_epoch, f"frame {frame_index} read")
        value = numpy.asarray(array[frame_index])
        if value.ndim != 3 or not numpy.issubdtype(value.dtype, numpy.number):
            raise RunnerError(f"frame {frame_index} is not a numeric Z/Y/X volume")
        frames.append(value)
        check_deadline(deadline_epoch, f"frame {frame_index} read")
    if len(frames) != frame_count:
        raise RunnerError("logical frame streaming did not cover the whole video")
    return frames


class LazyImageRefiner:
    """Load and cache only frames requested by bh_graph gap refinement."""

    def __init__(
        self,
        source: Any,
        refine: Callable[..., Any],
        *,
        deadline_epoch: float | None,
        capacity: int = 3,
    ) -> None:
        self.source = source
        self.refine = refine
        self.deadline_epoch = deadline_epoch
        self.capacity = max(1, int(capacity))
        self.cache: OrderedDict[int, Any] = OrderedDict()
        self.reads = 0
        self.hits = 0

    def _frame(self, frame: int) -> Any:
        numpy = importlib.import_module("numpy")
        if frame in self.cache:
            self.hits += 1
            value = self.cache.pop(frame)
            self.cache[frame] = value
            return value
        check_deadline(self.deadline_epoch, f"lazy refinement frame {frame} read")
        value = numpy.asarray(self.source[frame])
        self.reads += 1
        self.cache[frame] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def __call__(self, frame: int, coord_zyx: tuple[float, float, float]) -> Any:
        check_deadline(self.deadline_epoch, f"lazy refinement frame {frame}")
        return self.refine(self._frame(int(frame)), coord_zyx)

    def report(self) -> dict[str, int | bool]:
        return {"lazy": True, "frame_reads": self.reads, "cache_hits": self.hits, "capacity": self.capacity}


def detections_to_graph_nodes(detections: Sequence[Any], graph_node_type: type[Any]) -> tuple[Any, ...]:
    """Convert canonical bh_model FrameDet records without rounding coordinates."""

    result = []
    seen: set[int] = set()
    for frame in detections:
        for node in frame.nodes:
            if int(node.node_id) in seen:
                raise RunnerError(f"duplicate canonical node ID {node.node_id}")
            seen.add(int(node.node_id))
            result.append(
                graph_node_type(
                    node_id=int(node.node_id),
                    t=int(node.t),
                    z=float(node.z),
                    y=float(node.y),
                    x=float(node.x),
                    score=float(node.score),
                    synthetic=False,
                )
            )
    return tuple(sorted(result, key=lambda node: (node.t, node.node_id)))


def _nearest_link_summaries(detections: Sequence[Any], scale_zyx: Sequence[float]) -> dict[str, list[float]]:
    numpy = importlib.import_module("numpy")
    spatial = importlib.import_module("scipy.spatial")
    nearest_neighbor: list[float] = []
    margins: list[float] = []
    unmatched6: list[float] = []
    unmatched10: list[float] = []
    scale = numpy.asarray(scale_zyx, dtype=numpy.float64)
    for frame in detections:
        points = numpy.asarray(frame.coords_zyx, dtype=numpy.float64) * scale[None, :]
        if len(points) > 1:
            distances, _ = spatial.cKDTree(points).query(points, k=2, workers=1)
            nearest_neighbor.extend(float(value) for value in distances[:, 1])
    for source, target in zip(detections, detections[1:]):
        left = numpy.asarray(source.coords_zyx, dtype=numpy.float64) * scale[None, :]
        right = numpy.asarray(target.coords_zyx, dtype=numpy.float64) * scale[None, :]
        if not len(left):
            continue
        if not len(right):
            nearest = numpy.full(len(left), numpy.inf)
        else:
            k = min(2, len(right))
            distances, _ = spatial.cKDTree(right).query(left, k=k, workers=1)
            if k == 1:
                nearest = numpy.asarray(distances).reshape(-1)
            else:
                nearest = distances[:, 0]
                margins.extend(float(value) for value in (distances[:, 1] - distances[:, 0]))
        unmatched6.extend(float(value > 6.0) for value in nearest)
        unmatched10.extend(float(value > 10.0) for value in nearest)
    return {
        "nearest_neighbor_um": nearest_neighbor,
        "assignment_margins": margins,
        "unmatched_at_6um": unmatched6,
        "unmatched_at_10um": unmatched10,
    }


def build_route_summary(
    detections: Sequence[Any],
    *,
    frame_shape_zyx: Sequence[int],
    scale_zyx: Sequence[float],
    q_low: float,
    q_high: float,
    frames: Sequence[Any] | None = None,
    graph_result: Any | None = None,
    route_module: Any | None = None,
) -> dict[str, Any]:
    """Build and extract the cheap, identity-free bh_route feature summary."""

    numpy = importlib.import_module("numpy")
    route_module = route_module or importlib.import_module("celltrack_alpha.bh_route")
    counts = [len(frame.node_ids) for frame in detections]
    confidences: list[float] = []
    border: list[float] = []
    depth_edge: list[float] = []
    intensities: list[float] = []
    z_size, y_size, x_size = (int(value) for value in frame_shape_zyx)
    for frame in detections:
        for index, coord in enumerate(frame.coords_zyx):
            z, y, x = (int(value) for value in coord)
            confidences.append(float(frame.scores[index]))
            border.append(float(y <= 1 or x <= 1 or y >= y_size - 2 or x >= x_size - 2))
            depth_edge.append(float(z <= 1 or z >= z_size - 2))
            if frames is not None:
                raw = float(frames[int(frame.frame_index)][z, y, x])
                intensities.append(min(1.0, max(0.0, (raw - q_low) / (q_high - q_low))))
    summary: dict[str, Any] = {
        "frame_shape_zyx": list(frame_shape_zyx),
        "voxel_size_um": list(scale_zyx),
        "spatial_ndim": 3,
        "frame_count": len(detections),
        "detections_per_frame": counts,
        "detection_confidences": confidences,
        "border_flags": border,
        "depth_edge_flags": depth_edge,
        "detection_intensities": intensities,
        "gap_lengths": [],
        "division_motif_count": 0,
        "division_motif_opportunities": max(1, sum(counts)),
        **_nearest_link_summaries(detections, scale_zyx),
    }
    if graph_result is not None:
        outdegree: dict[int, int] = defaultdict(int)
        gap_edges = 0
        for edge in graph_result.edges:
            outdegree[int(edge.source_id)] += 1
            gap_edges += int("gap" in str(edge.kind).casefold())
        summary["gap_demand_fraction"] = min(1.0, gap_edges / max(1, len(graph_result.edges)))
        summary["division_motif_count"] = sum(value == 2 for value in outdegree.values())
    features = route_module.extract_features(summary)
    return {
        "schema": FEATURE_SCHEMA,
        "feature_schema": route_module.FEATURE_SCHEMA,
        "summary": summary,
        "features": features,
    }


def load_validated_router(path: str | os.PathLike[str] | None, route_module: Any | None = None) -> tuple[Any | None, str | None]:
    if path is None:
        return None, None
    route_module = route_module or importlib.import_module("celltrack_alpha.bh_route")
    source = Path(path)
    try:
        encoded = source.read_text(encoding="ascii")
        router = route_module.ScenarioRouter.from_json(encoded)
    except Exception as exc:
        raise RunnerError(f"explicit router JSON failed validation: {exc}") from exc
    return router, hashlib.sha256(encoded.encode("ascii")).hexdigest()


def view_batch_report(
    *,
    pair_batch_size: int,
    view_batch_size: int,
    accelerated_gate: Mapping[str, Any] | None,
    require_accelerated_gate: bool = True,
    d4_views: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Describe exact sequential or explicitly gated accelerated D4 execution."""

    if type(pair_batch_size) is not int or pair_batch_size <= 0:
        raise RunnerError("pair_batch_size must be a positive integer")
    if type(view_batch_size) is not int or not 1 <= view_batch_size <= 8:
        raise RunnerError("view_batch_size must be an integer in [1, 8]")
    selected_views = PRODUCTION_D4_VIEWS if d4_views is None else tuple(d4_views)
    if selected_views not in (PRODUCTION_D4_VIEWS, TEST_ONLY_ONE_VIEW):
        raise RunnerError("runner D4 views must be all eight production views or explicit test-only view 0")
    if view_batch_size > len(selected_views):
        raise RunnerError("view_batch_size exceeds the selected D4 view count")
    if selected_views == TEST_ONLY_ONE_VIEW:
        if accelerated_gate is not None or view_batch_size != 1:
            raise RunnerError("test-only one-view execution must be sequential and ungated")
        return {
            "pair_batch_size": pair_batch_size,
            "view_batch_size": 1,
            "d4_views": [0],
            "mode": "test-only-one-view",
            "production_eligible": False,
            "exact_code_path_control": False,
            "target_t4_numeric_parity": "not-applicable",
            "parity_claim": "local-port-gate-only",
            "observed_count_mismatch_cause": None,
            "real_t4_ab_required": False,
            "accelerated_score_gate_sha256": None,
        }
    if view_batch_size == 1:
        if accelerated_gate is not None:
            raise RunnerError("an accelerated score gate cannot be attached to exact sequential D4")
        return {
            "pair_batch_size": pair_batch_size,
            "view_batch_size": 1,
            "d4_views": list(PRODUCTION_D4_VIEWS),
            "mode": "exact-sequential",
            "production_eligible": True,
            "exact_code_path_control": True,
            "target_t4_numeric_parity": "unverified",
            "parity_claim": "none-pending-real-t4-a-b",
            "observed_count_mismatch_cause": "unknown-cpu-vs-t4-drift-plausible-batching-not-proven-causal",
            "real_t4_ab_required": True,
            "accelerated_score_gate_sha256": None,
        }
    if require_accelerated_gate and accelerated_gate is None:
        raise RunnerError("accelerated D4 production requires an explicit validated local score gate")
    gate_sha = None if accelerated_gate is None else accelerated_gate.get("sha256")
    if accelerated_gate is not None and (
        accelerated_gate.get("passed") is not True
        or accelerated_gate.get("pair_batch_size") != pair_batch_size
        or accelerated_gate.get("view_batch_size") != view_batch_size
        or not isinstance(gate_sha, str)
    ):
        raise RunnerError("accelerated D4 score gate does not match the requested batching mode")
    return {
        "pair_batch_size": pair_batch_size,
        "view_batch_size": view_batch_size,
        "d4_views": list(PRODUCTION_D4_VIEWS),
        "mode": "accelerated-experimental",
        "production_eligible": True,
        "exact_code_path_control": False,
        "target_t4_numeric_parity": "not-claimed",
        "parity_claim": "none",
        "observed_count_mismatch_cause": "unknown-batching-not-proven-causal",
        "real_t4_ab_required": True,
        "accelerated_score_gate_sha256": gate_sha,
    }


def load_accelerated_score_gate(
    path: str | os.PathLike[str] | None,
    *,
    pair_batch_size: int,
    view_batch_size: int,
    checkpoint_sha256: str,
    numerical_config_sha256: str,
) -> dict[str, Any] | None:
    """Validate a self-digesting, fixed-local-score production gate."""

    if view_batch_size == 1:
        if path is not None:
            raise RunnerError("--accelerated-score-gate is invalid for sequential D4")
        return None
    if path is None:
        raise RunnerError("accelerated D4 production requires --accelerated-score-gate")
    payload = strict_json_load(path)
    required = {
        "artifact",
        "checkpoint_sha256",
        "local_rung_sha256",
        "minimum_score",
        "numerical_config_sha256",
        "official_score",
        "pair_batch_size",
        "passed",
        "sha256",
        "view_batch_size",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise RunnerError("accelerated D4 score gate has missing or unexpected fields")
    core = {key: value for key, value in payload.items() if key != "sha256"}
    if payload["artifact"] != ACCEL_GATE_SCHEMA or payload["sha256"] != canonical_sha256(core):
        raise RunnerError("accelerated D4 score gate schema or SHA-256 is invalid")
    if (
        payload["pair_batch_size"] != pair_batch_size
        or payload["view_batch_size"] != view_batch_size
        or payload["checkpoint_sha256"] != checkpoint_sha256
        or payload["numerical_config_sha256"] != numerical_config_sha256
    ):
        raise RunnerError("accelerated D4 score gate model/batching identity mismatch")
    rung_sha = payload["local_rung_sha256"]
    if not isinstance(rung_sha, str) or len(rung_sha) != 64 or any(char not in "0123456789abcdef" for char in rung_sha):
        raise RunnerError("accelerated D4 score gate local rung SHA-256 is invalid")
    try:
        score = float(payload["official_score"])
        minimum = float(payload["minimum_score"])
    except (TypeError, ValueError) as exc:
        raise RunnerError("accelerated D4 score gate scores must be numeric") from exc
    if (
        not math.isfinite(score)
        or not math.isfinite(minimum)
        or payload["passed"] is not True
        or score < minimum
    ):
        raise RunnerError("accelerated D4 did not pass its fixed local score gate")
    return dict(payload)


def choose_route(feature_record: Mapping[str, Any], router: Any | None) -> dict[str, Any]:
    if router is None:
        return {
            "route": MOTION_ROUTE,
            "source": "conservative-default",
            "router_used": False,
        }
    prediction = router.predict(feature_record)
    if prediction.route != MOTION_ROUTE:
        raise RunnerError(
            f"validated router selected unsupported route {prediction.route!r}; only {MOTION_ROUTE!r} is locked"
        )
    return {
        "route": prediction.route,
        "source": "explicit-validated-router",
        "router_used": True,
        "confidence": float(prediction.confidence),
        "margin": float(prediction.margin),
        "ood": bool(prediction.ood),
        "used_default": bool(prediction.used_default),
    }


def _identity_dict(identity: Any) -> dict[str, Any]:
    return {
        "config_sha256": str(identity.config_sha256),
        "source_sha256": str(identity.source_sha256),
        "weight_sha256": str(identity.weight_sha256),
        "pair_batch_size": int(identity.pair_batch_size),
        "view_batch_size": int(identity.view_batch_size),
    }


def inference_contract(
    job: VideoJob,
    cache_identity: Any,
    *,
    pair_batch_size: int = DEFAULT_BATCH_SIZE,
    view_batch_size: int = 1,
    frame_count: int = FRAME_COUNT,
    d4_views: Sequence[int] | None = None,
) -> dict[str, Any]:
    if (
        int(cache_identity.pair_batch_size) != pair_batch_size
        or int(cache_identity.view_batch_size) != view_batch_size
    ):
        raise RunnerError("bh_model cache identity batching fields disagree with runner settings")
    d4 = view_batch_report(
        pair_batch_size=pair_batch_size,
        view_batch_size=view_batch_size,
        accelerated_gate=None,
        require_accelerated_gate=False,
        d4_views=d4_views,
    )
    value = {
        "schema": CACHE_META_SCHEMA,
        "model": _identity_dict(cache_identity),
        "inference": {
            "api": "bh_model.infer_video",
            "pair_batch_size": int(pair_batch_size),
            "view_batch_size": int(view_batch_size),
            "d4_views": d4["d4_views"],
            "d4_mode": d4["mode"],
            "exact_code_path_control": d4["exact_code_path_control"],
            "target_t4_numeric_parity": d4["target_t4_numeric_parity"],
            "real_t4_ab_required": True,
            "frame_count": int(frame_count),
            "learned_dense_transformer": False,
        },
    }
    if job.record.identity_provenance == REGISTRY_EXACT_IDENTITY:
        value["data_exact_identity_sha256"] = job.record.exact_identity_sha256
    elif job.record.identity_provenance == RUNTIME_METADATA_IDENTITY:
        value["data_runtime_metadata_identity_sha256"] = job.record.exact_identity_sha256
    else:
        raise RunnerError(
            f"unsupported data identity provenance for {job.identity}: "
            f"{job.record.identity_provenance}"
        )
    value["data_identity_provenance"] = job.record.identity_provenance
    return {**value, "sha256": canonical_sha256(value)}


def p0_calibration_contract(
    experiment_id: str | None,
    motion_error_weight: float | None,
) -> dict[str, Any] | None:
    if experiment_id is None and motion_error_weight is None:
        return None
    if experiment_id not in P0_APPROVED_MOTION_WEIGHTS or motion_error_weight is None:
        raise RunnerError("P0 calibration requires an approved experiment ID and motion weight")
    try:
        weight = float(motion_error_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunnerError("P0 calibration motion weight must be finite") from exc
    expected = P0_APPROVED_MOTION_WEIGHTS[experiment_id]
    if not math.isfinite(weight) or weight != expected:
        raise RunnerError(
            f"P0 calibration {experiment_id} requires motion weight {expected:g}"
        )
    return {
        "schema": P0_CALIBRATION_SCHEMA,
        "experiment_id": experiment_id,
        "purpose": "public-metric-parity-calibration",
        "motion_error_weight": weight,
        "production_eligible": False,
        "submit_eligible": True,
    }


def run_contract(
    inference_value: Mapping[str, Any],
    *,
    graph_variant: str,
    router_sha256: str | None,
    view_execution: Mapping[str, Any],
    edge_artifact: Mapping[str, Any] | None = None,
    p0_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if graph_variant not in GRAPH_VARIANTS:
        raise RunnerError(f"unsupported graph variant {graph_variant!r}")
    value = {
        "schema": RUNNER_SCHEMA,
        "inference_sha256": inference_value["sha256"],
        "graph_variant": graph_variant,
        "graph_config": GRAPH_VARIANTS[graph_variant],
        "route_default": MOTION_ROUTE,
        "router_sha256": router_sha256,
        "view_execution": dict(view_execution),
        "checkpoint_exposure": CHECKPOINT_EXPOSURE,
    }
    if edge_artifact is not None:
        value["edge_artifact"] = dict(edge_artifact)
    if p0_calibration is not None:
        value["p0_calibration"] = dict(p0_calibration)
    return {**value, "sha256": canonical_sha256(value)}


_A0_SUM_FIELDS = (
    "source_count",
    "target_count",
    "candidate_count",
    "component_count",
    "attention_component_count",
    "fallback_component_count",
    "node_budget_fallback_component_count",
    "attention_budget_fallback_component_count",
    "unsupported_fallback_component_count",
    "attention_candidate_count",
    "fallback_candidate_count",
    "batch_count",
)
_A0_MAX_FIELDS = (
    "max_component_source_nodes",
    "max_component_target_nodes",
    "max_component_nodes",
    "max_component_candidates",
    "max_component_attention_cells",
    "max_batch_components",
    "max_padded_source_nodes",
    "max_padded_target_nodes",
    "peak_padding_nodes",
    "peak_attention_cells",
    "peak_pair_chunk",
)
_A0_FIXED_FIELDS = (
    "neighbor_cap",
    "max_component_nodes_budget",
    "attention_cell_budget",
    "pair_chunk_size",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _a0_scorer_identity(scorer: Any, expected_weight_sha256: str) -> dict[str, Any]:
    """Validate and summarize the worker-local frozen A0 checkpoint view."""

    if not _is_sha256(expected_weight_sha256):
        raise RunnerError("A0 expected checkpoint identity is invalid")
    if getattr(scorer, "pretrained_weight_sha256", None) != expected_weight_sha256:
        raise RunnerError("A0 scorer checkpoint identity differs from the inference cache")
    architecture = getattr(scorer, "architecture", None)
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("full_frame_matrix") is not False
        or architecture.get("pair_scope") != "original-candidate-coo-only"
    ):
        raise RunnerError("A0 scorer lacks original-COO sparse provenance")
    config = getattr(scorer, "config", None)
    if getattr(config, "neighbor_cap", None) != A0_GATE_NEIGHBOR_CAP:
        raise RunnerError("A0 scorer neighbor cap differs from the production gate")
    architecture_sha256 = getattr(scorer, "architecture_sha256", None)
    frozen_subset_sha256 = getattr(scorer, "frozen_subset_sha256", None)
    if not _is_sha256(architecture_sha256) or not _is_sha256(frozen_subset_sha256):
        raise RunnerError("A0 scorer architecture or frozen subset identity is invalid")
    return {
        "weight_sha256": expected_weight_sha256,
        "architecture_sha256": architecture_sha256,
        "frozen_subset_sha256": frozen_subset_sha256,
        "factory": "A0CheckpointScorer.from_loaded_model",
        "full_frame_dense_matrix": False,
    }


def build_a0_scorer(
    graph_variant: str,
    model: Any,
    bundle: Any,
    cache_identity: Any,
    *,
    edge_module: Any | None = None,
) -> Any | None:
    """Construct the sole worker A0 scorer, or remain inert for motion."""

    if graph_variant == LOCKED_GRAPH_VARIANT:
        return None
    if graph_variant != A0_GRAPH_VARIANT:
        raise RunnerError(f"unsupported graph variant {graph_variant!r}")
    edge_module = edge_module or importlib.import_module("celltrack_alpha.bh_edge")
    if getattr(getattr(bundle, "hashes", None), "weight_sha256", None) != getattr(
        cache_identity, "weight_sha256", None
    ):
        raise RunnerError("A0 bundle and cache checkpoint identities disagree")
    scorer = edge_module.A0CheckpointScorer.from_loaded_model(model, bundle)
    _a0_scorer_identity(scorer, str(cache_identity.weight_sha256))
    return scorer


def pinned_edge_artifact_record(
    artifact_path: str | os.PathLike[str] | None,
    artifact_sha256: str | None,
) -> dict[str, Any] | None:
    """Validate the CLI pin exactly as v45feval does and record its identity."""

    if artifact_path is None:
        if artifact_sha256 is not None:
            raise RunnerError("--edge-artifact-sha256 requires --edge-artifact")
        return None
    if not _is_sha256(artifact_sha256):
        raise RunnerError("--edge-artifact requires a lowercase SHA-256")
    source = Path(artifact_path)
    if not source.is_file() or sha256_file(source) != artifact_sha256:
        raise RunnerError("edge artifact SHA-256 does not match the pinned value")
    payload = strict_json_load(source)
    if not isinstance(payload, Mapping):
        raise RunnerError("edge artifact must be a JSON object")
    config = payload.get("config")
    pretrained = payload.get("pretrained")
    auxiliary = payload.get("auxiliary")
    exposure = payload.get("exposure")
    if not all(isinstance(value, Mapping) for value in (config, pretrained, auxiliary, exposure)):
        raise RunnerError("edge artifact identity metadata is incomplete")
    hidden = config.get("residual_hidden")
    if type(hidden) is not int or hidden <= 0:
        raise RunnerError("edge artifact residual hidden width is invalid")
    auxiliary_enabled = auxiliary.get("enabled") is True
    identity = {
        "architecture_sha256": payload.get("architecture_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "pretrained_weight_sha256": pretrained.get("checkpoint_weight_sha256"),
        "frozen_subset_sha256": pretrained.get("frozen_subset_sha256"),
        "residual_weight_sha256": (
            payload.get("weights", {}).get("residual_weight_sha256")
            if isinstance(payload.get("weights"), Mapping)
            else None
        ),
        "auxiliary_checkpoint_sha256": auxiliary.get("checkpoint_sha256") if auxiliary_enabled else None,
        "auxiliary_state_sha256": (
            auxiliary.get("identity", {}).get("auxiliary_state_sha256")
            if auxiliary_enabled and isinstance(auxiliary.get("identity"), Mapping)
            else None
        ),
        "exposure": dict(exposure),
    }
    if any(not _is_sha256(value) for key, value in identity.items() if key != "exposure" and value is not None):
        raise RunnerError("edge artifact contains an invalid identity SHA-256")
    return json.loads(
        canonical_json(
            {
                "path": str(source.resolve()),
                "sha256": artifact_sha256,
                "identity": identity,
                "fitted_parameter_count": (16 + 1) * hidden + hidden + 2,
                "runtime_trainable_parameter_count": 0,
                "division_enabled": False,
            }
        ).decode("ascii")
    )


def load_sparse_edge_scorer(
    artifact_path: str | os.PathLike[str] | None,
    artifact_sha256: str | None,
    auxiliary_checkpoint: str | os.PathLike[str],
    model: Any,
    bundle: Any,
    *,
    edge_module: Any | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Restore one hash-pinned fit-only scorer and freeze it for inference."""

    record = pinned_edge_artifact_record(artifact_path, artifact_sha256)
    if record is None:
        return None, None
    source = Path(artifact_path)
    payload = strict_json_load(source)
    auxiliary_metadata = payload.get("auxiliary") if isinstance(payload, Mapping) else None
    auxiliary = None
    if isinstance(auxiliary_metadata, Mapping) and auxiliary_metadata.get("enabled") is True:
        checkpoint_sha256 = auxiliary_metadata.get("checkpoint_sha256")
        if not _is_sha256(checkpoint_sha256):
            raise RunnerError("edge artifact auxiliary checkpoint identity is invalid")
        checkpoint = Path(auxiliary_checkpoint)
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha256:
            raise RunnerError("edge artifact auxiliary checkpoint SHA-256 does not match")
        fast_module = importlib.import_module("celltrack_alpha.v45_fast")
        head_module = importlib.import_module("celltrack_alpha.v45_head")
        target_module = importlib.import_module("celltrack_alpha.t_adapter")
        auxiliary, _identity = fast_module.load_fast_readout(
            checkpoint,
            bundle,
            expected_sha256=checkpoint_sha256,
            target_sha256=target_module.TARGET_CONTRACT_SHA256,
            expected_core_state_sha256=head_module.v45_state_sha256(model),
        )
    edge_module = edge_module or importlib.import_module("celltrack_alpha.bh_edge")
    scorer, identity = edge_module.load_edge_artifact(
        source,
        model,
        expected_pretrained_weight_sha256=bundle.hashes.weight_sha256,
        auxiliary=auxiliary,
    )
    fitted_parameter_count = int(scorer.trainable_parameter_count)
    if fitted_parameter_count <= 0 or scorer.config.division_enabled is not False:
        raise RunnerError("edge artifact must contain a fitted scorer with divisions disabled")
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    scorer.eval()
    if scorer.trainable_parameter_count != 0:
        raise RunnerError("fit-only edge scorer remained trainable during inference")
    observed_identity = json.loads(canonical_json(asdict(identity)).decode("ascii"))
    if observed_identity != record["identity"] or fitted_parameter_count != record["fitted_parameter_count"]:
        raise RunnerError("loaded edge artifact identity differs from its pinned JSON record")
    return scorer, record


def score_sparse_edge_cache(
    cache: Any,
    scorer: Any,
    *,
    edge_module: Any | None = None,
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[str, Any]]:
    """Port v45feval's adjacent sparse scorer over a persisted run_bh cache."""

    edge_module = edge_module or importlib.import_module("celltrack_alpha.bh_edge")
    if not isinstance(scorer, edge_module.SparseEdgeScorer):
        raise RunnerError("edge artifact execution requires SparseEdgeScorer")
    detections = getattr(cache, "detections", None)
    contexts = getattr(cache, "contexts", None)
    if not isinstance(detections, tuple) or not isinstance(contexts, tuple) or not contexts:
        raise RunnerError("edge artifact execution requires complete cached detections and contexts")
    by_frame = {int(detection.frame_index): detection for detection in detections}
    if len(by_frame) != len(detections):
        raise RunnerError("edge artifact cache detections have duplicate frame identities")
    pairs = []
    for context in contexts:
        source = by_frame.get(int(context.source_frame))
        target = by_frame.get(int(context.target_frame))
        if source is None or target is None:
            raise RunnerError("edge artifact context references a missing detection frame")
        pairs.append(scorer.score_pair(context, source, target))
    probabilities = edge_module.learned_edge_probability_map(pairs)
    no_link: dict[int, float] = {}
    for pair in pairs:
        for source_id, probability in zip(
            pair.no_link_source_ids, pair.no_link_probabilities, strict=True
        ):
            source_id_int = int(source_id)
            if source_id_int in no_link:
                raise RunnerError("duplicate edge artifact no-link probability across frame pairs")
            no_link[source_id_int] = float(probability)
    candidate_count = sum(int(pair.stats.candidate_count) for pair in pairs)
    if candidate_count != len(probabilities):
        raise RunnerError("edge artifact candidate probability mapping is incomplete")
    audit = {
        "mode": "fit-only-v45-sparse-legacy",
        "learned_association_active": True,
        "topology_mode": "legacy",
        "division_enabled": False,
        "pair_count": len(pairs),
        "candidate_count": candidate_count,
        "probability_count": len(probabilities),
        "no_link_probability_count": len(no_link),
        "peak_pair_chunk": max((int(pair.stats.peak_pair_chunk) for pair in pairs), default=0),
        "neighbor_cap": int(scorer.config.neighbor_cap),
        "full_frame_dense_matrix": False,
    }
    return probabilities, no_link, audit


def _merge_a0_association(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RunnerError("A0 audit aggregation requires at least one pair")
    result: dict[str, Any] = {}
    for field in _A0_SUM_FIELDS:
        values = [record.get(field) for record in records]
        if any(type(value) is not int or value < 0 for value in values):
            raise RunnerError(f"A0 audit field {field!r} is invalid")
        result[field] = sum(values)
    for field in _A0_MAX_FIELDS:
        values = [record.get(field) for record in records]
        if any(type(value) is not int or value < 0 for value in values):
            raise RunnerError(f"A0 audit field {field!r} is invalid")
        result[field] = max(values)
    for field in _A0_FIXED_FIELDS:
        values = {record.get(field) for record in records}
        if len(values) != 1 or any(type(value) is not int or value <= 0 for value in values):
            raise RunnerError(f"A0 audit invariant {field!r} disagrees across pairs")
        result[field] = values.pop()
    if result["neighbor_cap"] != A0_GATE_NEIGHBOR_CAP:
        raise RunnerError("A0 association audit does not attest the production neighbor cap")
    component_denominator = max(result["component_count"], 1)
    candidate_denominator = max(result["candidate_count"], 1)
    result.update(
        {
            "attention_component_fraction": (
                result["attention_component_count"] / component_denominator
            ),
            "fallback_component_fraction": (
                result["fallback_component_count"] / component_denominator
            ),
            "attention_candidate_fraction": (
                result["attention_candidate_count"] / candidate_denominator
            ),
            "fallback_candidate_fraction": (
                result["fallback_candidate_count"] / candidate_denominator
            ),
        }
    )
    if (
        result["attention_component_count"] + result["fallback_component_count"]
        != result["component_count"]
        or result["attention_candidate_count"] + result["fallback_candidate_count"]
        != result["candidate_count"]
    ):
        raise RunnerError("A0 aggregate association accounting does not balance")
    return result


def score_a0_cache(
    cache: Any,
    scorer: Any,
    cache_identity: Any,
    *,
    edge_module: Any | None = None,
) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Score every persisted adjacent context and merge only original COO rows."""

    edge_module = edge_module or importlib.import_module("celltrack_alpha.bh_edge")
    numpy = importlib.import_module("numpy")
    scorer_identity = _a0_scorer_identity(scorer, str(cache_identity.weight_sha256))
    detections = getattr(cache, "detections", None)
    contexts = getattr(cache, "contexts", None)
    if not isinstance(detections, tuple) or len(detections) < 2:
        raise RunnerError("A0 cache lacks ordered persisted FrameDet records")
    if not isinstance(contexts, tuple) or len(contexts) != len(detections) - 1:
        raise RunnerError("A0 cache has a persisted adjacent-context gap")
    if any(not isinstance(frame, edge_module.FrameDet) for frame in detections):
        raise RunnerError("A0 cache detections are not FrameDet records")
    if any(not isinstance(context, edge_module.PairContext) for context in contexts):
        raise RunnerError("A0 cache contexts are not PairContext records")

    gate_config = edge_module.SparseEdgeConfig(
        radius_um=A0_GATE_RADIUS_UM,
        neighbor_cap=A0_GATE_NEIGHBOR_CAP,
    )
    if tuple(gate_config.scale_zyx) != IMAGE_SUPPORT_SCALE_ZYX:
        raise RunnerError("A0 candidate gate physical scale differs from locked V45 geometry")
    probabilities: dict[tuple[int, int], float] = {}
    pair_audits: list[dict[str, Any]] = []
    peak_radius_neighbors = 0
    peak_retained_neighbors = 0
    for index, context in enumerate(contexts):
        source = detections[index]
        target = detections[index + 1]
        expected_pair = (source.frame_index, target.frame_index)
        if target.frame_index != source.frame_index + 1 or context.frame_pair != expected_pair:
            raise RunnerError("A0 cache has a persisted adjacent-context gap")
        candidates = edge_module.gate_sparse_candidates(
            source,
            target,
            config=gate_config,
        )
        if (
            candidates.stats.radius_um != A0_GATE_RADIUS_UM
            or candidates.stats.neighbor_cap != A0_GATE_NEIGHBOR_CAP
        ):
            raise RunnerError("A0 candidates lack production 10um/cap8 gate provenance")
        scored = scorer.score_pair(context, source, target, candidates=candidates)
        if not isinstance(scored, edge_module.A0PairProbabilities):
            raise RunnerError("A0 scorer returned an unsupported probability container")
        if scored.source_frame != expected_pair[0] or scored.target_frame != expected_pair[1]:
            raise RunnerError("A0 scored pair provenance differs from its persisted context")
        if (
            not numpy.array_equal(scored.source_ids, candidates.source_ids)
            or not numpy.array_equal(scored.target_ids, candidates.target_ids)
            or len(scored.probabilities) != len(candidates)
        ):
            raise RunnerError("A0 scored probabilities are not the original candidate COO rows")
        for raw_source, raw_target, raw_probability in zip(
            scored.source_ids,
            scored.target_ids,
            scored.probabilities,
            strict=True,
        ):
            pair = (int(raw_source), int(raw_target))
            probability = float(raw_probability)
            if pair in probabilities:
                raise RunnerError("duplicate A0 probability pair across adjacent contexts")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RunnerError("A0 probability is nonfinite or outside [0, 1]")
            probabilities[pair] = probability
        pair_audits.append(asdict(scored.stats))
        peak_radius_neighbors = max(
            peak_radius_neighbors, int(candidates.stats.peak_radius_neighbors)
        )
        peak_retained_neighbors = max(
            peak_retained_neighbors, int(candidates.stats.peak_retained_neighbors)
        )

    association = _merge_a0_association(pair_audits)
    if association["candidate_count"] != len(probabilities):
        raise RunnerError("A0 original COO and duplicate-free map cardinalities disagree")
    audit = {
        "schema": A0_AUDIT_SCHEMA,
        "scope": "video",
        "video_count": 1,
        "frame_count": len(detections),
        "pair_count": len(contexts),
        "checkpoint": scorer_identity,
        "cache": {
            "record_types": ["FrameDet", "PairContext"],
            "detection_count": len(detections),
            "context_count": len(contexts),
            "complete_adjacent_coverage": True,
        },
        "gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": A0_GATE_RADIUS_UM,
            "neighbor_cap": A0_GATE_NEIGHBOR_CAP,
            "scale_zyx": list(IMAGE_SUPPORT_SCALE_ZYX),
            "candidate_count": len(probabilities),
            "peak_radius_neighbors": peak_radius_neighbors,
            "peak_retained_neighbors": peak_retained_neighbors,
            "full_frame_dense_matrix": False,
        },
        "probabilities": {
            "scope": "original-candidate-coo-only",
            "candidate_count": association["candidate_count"],
            "unique_pair_count": len(probabilities),
            "finite_unit_interval": True,
        },
        "association": association,
    }
    return probabilities, audit


def aggregate_a0_audits(
    audits: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Deterministically aggregate video A0 audits for worker/run outputs."""

    if not audits or scope not in {"worker", "run"}:
        raise RunnerError("A0 aggregate scope or input is invalid")
    for audit in audits:
        if audit.get("schema") != A0_AUDIT_SCHEMA:
            raise RunnerError("A0 aggregate contains an incompatible audit schema")
    checkpoints = [audit.get("checkpoint") for audit in audits]
    if any(checkpoint != checkpoints[0] for checkpoint in checkpoints):
        raise RunnerError("A0 aggregate checkpoint identities disagree")
    associations = [audit.get("association") for audit in audits]
    if any(not isinstance(record, Mapping) for record in associations):
        raise RunnerError("A0 aggregate association audit is missing")
    association = _merge_a0_association(associations)  # type: ignore[arg-type]
    gate_rows = [audit.get("gate") for audit in audits]
    if any(not isinstance(row, Mapping) for row in gate_rows):
        raise RunnerError("A0 aggregate gate audit is missing")
    result = {
        "schema": A0_AUDIT_SCHEMA,
        "scope": scope,
        "video_count": sum(int(audit.get("video_count", 0)) for audit in audits),
        "frame_count": sum(int(audit.get("frame_count", 0)) for audit in audits),
        "pair_count": sum(int(audit.get("pair_count", 0)) for audit in audits),
        "checkpoint": checkpoints[0],
        "gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": A0_GATE_RADIUS_UM,
            "neighbor_cap": A0_GATE_NEIGHBOR_CAP,
            "scale_zyx": list(IMAGE_SUPPORT_SCALE_ZYX),
            "candidate_count": sum(int(row.get("candidate_count", 0)) for row in gate_rows),
            "peak_radius_neighbors": max(
                int(row.get("peak_radius_neighbors", 0)) for row in gate_rows
            ),
            "peak_retained_neighbors": max(
                int(row.get("peak_retained_neighbors", 0)) for row in gate_rows
            ),
            "full_frame_dense_matrix": False,
        },
        "probabilities": {
            "scope": "original-candidate-coo-only",
            "candidate_count": association["candidate_count"],
            "unique_pair_count": sum(
                int(audit.get("probabilities", {}).get("unique_pair_count", 0))
                for audit in audits
            ),
            "finite_unit_interval": True,
        },
        "association": association,
    }
    if result["probabilities"]["unique_pair_count"] != association["candidate_count"]:
        raise RunnerError("A0 aggregate probability cardinalities disagree")
    return result


def empty_a0_worker_audit(scorer: Any, cache_identity: Any) -> dict[str, Any]:
    """Record a constructed A0 singleton for a worker assigned no videos."""

    config = scorer.config
    association = {
        **{field: 0 for field in _A0_SUM_FIELDS},
        **{field: 0 for field in _A0_MAX_FIELDS},
        "neighbor_cap": int(config.neighbor_cap),
        "max_component_nodes_budget": int(config.max_component_nodes),
        "attention_cell_budget": int(config.attention_cell_budget),
        "pair_chunk_size": int(config.pair_chunk_size),
        "attention_component_fraction": 0.0,
        "fallback_component_fraction": 0.0,
        "attention_candidate_fraction": 0.0,
        "fallback_candidate_fraction": 0.0,
    }
    return {
        "schema": A0_AUDIT_SCHEMA,
        "scope": "worker",
        "video_count": 0,
        "frame_count": 0,
        "pair_count": 0,
        "checkpoint": _a0_scorer_identity(scorer, str(cache_identity.weight_sha256)),
        "gate": {
            "algorithm": "physical-cKDTree",
            "radius_um": A0_GATE_RADIUS_UM,
            "neighbor_cap": A0_GATE_NEIGHBOR_CAP,
            "scale_zyx": list(IMAGE_SUPPORT_SCALE_ZYX),
            "candidate_count": 0,
            "peak_radius_neighbors": 0,
            "peak_retained_neighbors": 0,
            "full_frame_dense_matrix": False,
        },
        "probabilities": {
            "scope": "original-candidate-coo-only",
            "candidate_count": 0,
            "unique_pair_count": 0,
            "finite_unit_interval": True,
        },
        "association": association,
    }


def _cache_paths(cache_dir: Path, job: VideoJob) -> tuple[Path, Path]:
    return cache_dir / f"{job.slug}.bhc", cache_dir / f"{job.slug}.json"


def _cache_intent_path(cache_dir: Path, job: VideoJob) -> Path:
    return cache_dir / f"{job.slug}.intent.json"


def _validate_intent(path: Path, expected: Mapping[str, Any], label: str) -> bool:
    if not path.exists():
        return False
    try:
        observed = strict_json_load(path)
    except RunnerError as exc:
        raise ResumeHashError(f"invalid {label} intent {path}") from exc
    if observed != expected:
        raise ResumeHashError(f"{label} intent contract mismatch for {path.stem}")
    return True


def _claim_intent(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
    deadline_epoch: float | None,
) -> None:
    if _validate_intent(path, expected, label):
        return
    try:
        atomic_create_json(path, expected, deadline_epoch)
    except FileExistsError:
        if not _validate_intent(path, expected, label):
            raise ResumeHashError(f"failed to claim {label} intent {path}")


def _remove_owned_artifacts(
    paths: Sequence[Path],
    *,
    intent_path: Path,
    expected_intent: Mapping[str, Any],
    label: str,
) -> None:
    if not _validate_intent(intent_path, expected_intent, label):
        raise ResumeHashError(f"refusing to remove unowned {label} artifacts")
    parents: set[Path] = set()
    for path in paths:
        if path.exists():
            path.unlink()
            parents.add(path.parent)
    for parent in parents:
        _fsync_directory(parent)


def _cache_intent(
    cache_dir: Path,
    job: VideoJob,
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    cache_path, meta_path = _cache_paths(cache_dir, job)
    intent_path = _cache_intent_path(cache_dir, job)
    intent = {
        "schema": CACHE_INTENT_SCHEMA,
        "identity": job.identity,
        "contract": dict(contract),
        "artifacts": {"cache": str(cache_path), "meta": str(meta_path)},
    }
    return intent_path, intent


def load_inference_cache(
    cache_dir: Path,
    job: VideoJob,
    expected_contract: Mapping[str, Any],
    cache_identity: Any,
    model_module: Any,
    deadline_epoch: float | None = None,
) -> tuple[Any, dict[str, Any]] | None:
    cache_path, meta_path = _cache_paths(cache_dir, job)
    intent_path, expected_intent = _cache_intent(cache_dir, job, expected_contract)
    owned = _validate_intent(intent_path, expected_intent, "inference cache")
    cache_exists = cache_path.exists()
    meta_exists = meta_path.exists()
    if cache_exists != meta_exists:
        if not owned:
            raise ResumeHashError(f"unowned orphaned inference cache/meta pair for {job.identity}")
        check_deadline(deadline_epoch, f"{job.identity} inference cache recovery")
        _remove_owned_artifacts(
            (cache_path, meta_path),
            intent_path=intent_path,
            expected_intent=expected_intent,
            label="inference cache",
        )
        return None
    if not cache_exists:
        if not owned:
            check_deadline(deadline_epoch, f"{job.identity} inference cache intent")
            _claim_intent(intent_path, expected_intent, "inference cache", deadline_epoch)
        return None
    try:
        meta = strict_json_load(meta_path)
        if not isinstance(meta, Mapping) or meta.get("schema") != CACHE_META_SCHEMA:
            raise ResumeHashError(f"invalid inference cache metadata for {job.identity}")
        if meta.get("contract") != expected_contract:
            raise ResumeHashError(f"inference cache source/checkpoint/config hash mismatch for {job.identity}")
        if meta.get("cache_sha256") != sha256_file(cache_path):
            raise ResumeHashError(f"inference cache byte hash mismatch for {job.identity}")
        cache = model_module.load_sparse_cache(cache_path, expected_identity=cache_identity)
        return cache, dict(meta)
    except Exception as exc:
        if not owned:
            if isinstance(exc, ResumeHashError):
                raise
            raise ResumeHashError(f"invalid inference cache for {job.identity}") from exc
        check_deadline(deadline_epoch, f"{job.identity} inference cache recovery")
        _remove_owned_artifacts(
            (cache_path, meta_path),
            intent_path=intent_path,
            expected_intent=expected_intent,
            label="inference cache",
        )
        return None


def _save_inference_cache_bytes(
    cache_path: Path,
    cache: Any,
    model_module: Any,
    deadline_epoch: float | None,
) -> None:
    descriptor, staging_text = tempfile.mkstemp(
        prefix=f".{cache_path.name}.", suffix=".stage", dir=cache_path.parent
    )
    os.close(descriptor)
    os.unlink(staging_text)
    staging = Path(staging_text)
    try:
        model_module.save_sparse_cache(staging, cache)
        check_deadline(deadline_epoch, f"{cache_path.name} finalization")
        os.link(staging, cache_path)
        staging.unlink()
        _fsync_directory(cache_path.parent)
        check_deadline(deadline_epoch, f"{cache_path.name} finalization")
    finally:
        staging.unlink(missing_ok=True)


def save_inference_cache(
    cache_dir: Path,
    job: VideoJob,
    contract: Mapping[str, Any],
    cache_identity: Any,
    inference: Any,
    pregraph_features: Mapping[str, Any],
    model_module: Any,
    deadline_epoch: float | None,
) -> tuple[Any, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path = _cache_paths(cache_dir, job)
    intent_path, expected_intent = _cache_intent(cache_dir, job, contract)
    if cache_path.exists() or meta_path.exists():
        raise ResumeHashError(f"refusing to overwrite existing inference cache for {job.identity}")
    _claim_intent(intent_path, expected_intent, "inference cache", deadline_epoch)
    cache = model_module.SparseInferenceCache(
        cache_identity,
        tuple(inference.detections),
        tuple(inference.contexts),
        (),
    )
    _save_inference_cache_bytes(cache_path, cache, model_module, deadline_epoch)
    check_deadline(deadline_epoch, f"{job.identity} inference cache")
    meta = {
        "schema": CACHE_META_SCHEMA,
        "contract": dict(contract),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "pregraph_features": pregraph_features,
    }
    atomic_create_json(meta_path, meta, deadline_epoch)
    return cache, meta


def public_round_coordinate(value: Any) -> int:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OutputValidationError("node coordinate is non-finite")
    return int(round(numeric))


def _image_support_measurement(
    raw_coords: Sequence[Any],
    shape_tzyx: Sequence[int],
    scale_zyx: Sequence[float] = IMAGE_SUPPORT_SCALE_ZYX,
) -> tuple[
    tuple[float, float, float],
    tuple[int | None, int | None, int | None],
    tuple[float, float, float] | None,
    float | None,
]:
    """Measure the minimum physical correction to continuous image support."""

    try:
        shape = tuple(int(value) for value in shape_tzyx)
        scale = tuple(float(value) for value in scale_zyx)
        raw = tuple(float(value) for value in raw_coords)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OutputValidationError("image-support inputs must be numeric") from exc
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise OutputValidationError(f"image shape must contain four positive dimensions: {shape!r}")
    if len(scale) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in scale):
        raise OutputValidationError(f"image scale must contain three positive finite values: {scale!r}")
    if len(raw) != 3:
        raise OutputValidationError("node coordinates must contain exactly three z/y/x values")
    rounded = tuple(
        int(round(value)) if math.isfinite(value) else None
        for value in raw
    )
    if not all(math.isfinite(value) for value in raw):
        return raw, rounded, None, None  # type: ignore[return-value]
    bounds = tuple(float(value - 1) for value in shape[1:])
    projected = tuple(
        min(max(value, 0.0), bound)
        for value, bound in zip(raw, bounds, strict=True)
    )
    correction_um = math.hypot(
        *(
            (value - supported) * axis_scale
            for value, supported, axis_scale in zip(raw, projected, scale, strict=True)
        )
    )
    return raw, rounded, projected, correction_um  # type: ignore[return-value]


def _node_validation_context(
    dataset: str,
    node_id: int,
    raw: Sequence[Any],
    rounded: Sequence[Any],
    shape_tzyx: Sequence[int],
    correction_um: float | None,
) -> str:
    return (
        f"dataset={dataset!r} node_id={node_id} raw_coords_zyx={tuple(raw)!r} "
        f"rounded_coords_zyx={tuple(rounded)!r} image_shape_tzyx={tuple(shape_tzyx)!r} "
        f"correction_um={correction_um!r}"
    )


def project_graph_result_to_image_support(
    dataset: str,
    graph_result: Any,
    shape_tzyx: Sequence[int],
    *,
    scale_zyx: Sequence[float] = IMAGE_SUPPORT_SCALE_ZYX,
    max_correction_um: float = IMAGE_SUPPORT_MAX_CORRECTION_UM,
    epsilon_um: float = IMAGE_SUPPORT_EPSILON_UM,
) -> Any:
    """Project a completed graph to image support under the V45 refinement bound.

    This is deliberately a post-smoothing/public-output constraint.  It does
    not make arbitrary invalid coordinates acceptable: non-finite values and
    corrections beyond ``refine_max_shift_um`` fail closed with node context.
    """

    try:
        limit = float(max_correction_um)
        epsilon = float(epsilon_um)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OutputValidationError("image-support correction bound must be numeric") from exc
    if not math.isfinite(limit) or limit < 0.0 or not math.isfinite(epsilon) or epsilon < 0.0:
        raise OutputValidationError("image-support correction bound and epsilon must be finite and nonnegative")

    projected_nodes: list[Any] = []
    projected_count = 0
    maximum_observed = 0.0
    for node in tuple(graph_result.nodes):
        node_id = int(node.node_id)
        raw, rounded, projected, correction_um = _image_support_measurement(
            (node.z, node.y, node.x), shape_tzyx, scale_zyx
        )
        context = _node_validation_context(
            dataset, node_id, raw, rounded, shape_tzyx, correction_um
        )
        if projected is None or correction_um is None:
            raise OutputValidationError(f"graph node has non-finite coordinates: {context}")
        if correction_um > limit + epsilon:
            raise OutputValidationError(
                "image-support correction exceeds refine_max_shift_um "
                f"limit={limit!r} epsilon_um={epsilon!r}: {context}"
            )
        maximum_observed = max(maximum_observed, correction_um)
        if projected != raw:
            projected_count += 1
            projected_nodes.append(
                replace(node, z=projected[0], y=projected[1], x=projected[2])
            )
        else:
            projected_nodes.append(node)

    edge_count = len(tuple(graph_result.edges))
    summary = {
        "boundary": "post-smoothing/public-output",
        "scale_zyx": [float(value) for value in scale_zyx],
        "limit_um": limit,
        "epsilon_um": epsilon,
        "projected_nodes": projected_count,
        "max_correction_um": maximum_observed,
        "node_count": len(projected_nodes),
        "edge_count": edge_count,
        "node_count_preserved": True,
        "topology_preserved": True,
    }
    stats = {
        **dict(graph_result.stats),
        "image_support_projected_nodes": projected_count,
        "image_support_max_correction_um": maximum_observed,
        "image_support_projection_limit_um": limit,
    }
    provenance = {
        **dict(graph_result.provenance),
        "image_support_projection": summary,
    }
    return replace(
        graph_result,
        nodes=tuple(projected_nodes),
        stats=stats,
        provenance=provenance,
    )


def validate_graph_result(
    graph_result: Any,
    shape_tzyx: Sequence[int],
    dataset: str = "<unknown>",
) -> None:
    nodes = tuple(graph_result.nodes)
    edges = tuple(graph_result.edges)
    node_by_id: dict[int, Any] = {}
    max_t, max_z, max_y, max_x = (int(value) for value in shape_tzyx)
    for node in nodes:
        node_id = int(node.node_id)
        if node_id < 0 or node_id in node_by_id:
            raise OutputValidationError("graph has a negative or duplicate node ID")
        raw, rounded, _projected, correction_um = _image_support_measurement(
            (node.z, node.y, node.x), shape_tzyx
        )
        context = _node_validation_context(
            dataset, node_id, raw, rounded, shape_tzyx, correction_um
        )
        if not 0 <= int(node.t) < max_t:
            raise OutputValidationError(f"graph node frame lies outside the image: {context}")
        if any(value is None for value in rounded):
            raise OutputValidationError(f"graph node has non-finite coordinates: {context}")
        if not (0 <= rounded[0] < max_z and 0 <= rounded[1] < max_y and 0 <= rounded[2] < max_x):
            raise OutputValidationError(
                f"publicly rounded graph node lies outside the image: {context}"
            )
        node_by_id[node_id] = node
    if not node_by_id:
        raise OutputValidationError("graph emitted no nodes")
    seen: set[tuple[int, int]] = set()
    indegree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    for edge in edges:
        pair = (int(edge.source_id), int(edge.target_id))
        if pair in seen or pair[0] not in node_by_id or pair[1] not in node_by_id:
            raise OutputValidationError("graph edge is duplicate or references an unknown node")
        seen.add(pair)
        delta = int(node_by_id[pair[1]].t) - int(node_by_id[pair[0]].t)
        if delta != 1:
            raise OutputValidationError(
                "graph edges must be frame-adjacent; gap/division structure requires graph-inserted nodes"
            )
        indegree[pair[1]] += 1
        outdegree[pair[0]] += 1
        if indegree[pair[1]] > 1 or outdegree[pair[0]] > 2:
            raise OutputValidationError("graph contains a merge or outdegree greater than two")


def graph_rows(dataset: str, graph_result: Any, shape_tzyx: Sequence[int]) -> list[dict[str, Any]]:
    """Create public rows; floating graph coordinates are rounded only here."""

    validate_graph_result(graph_result, shape_tzyx, dataset)
    rows: list[dict[str, Any]] = []
    for node in sorted(graph_result.nodes, key=lambda item: (int(item.t), int(item.node_id))):
        rows.append(
            {
                "id": len(rows),
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": public_round_coordinate(node.z),
                "y": public_round_coordinate(node.y),
                "x": public_round_coordinate(node.x),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for edge in sorted(graph_result.edges, key=lambda item: (int(item.source_id), int(item.target_id))):
        rows.append(
            {
                "id": len(rows),
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(edge.source_id),
                "target_id": int(edge.target_id),
            }
        )
    return rows


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise OutputValidationError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OutputValidationError(f"{field} must be an integer") from exc
    if str(result) != str(value).strip() and not isinstance(value, int):
        raise OutputValidationError(f"{field} must use canonical integer syntax")
    return result


def validate_submission_rows(rows: Sequence[Mapping[str, Any]], expected: Sequence[RegistryRecord]) -> dict[str, int]:
    expected_by_dataset = {record.dataset: record for record in expected}
    if len(expected_by_dataset) != len(expected):
        raise OutputValidationError("expected dataset labels are duplicated")
    node_by_dataset: dict[str, dict[int, tuple[int, int, int, int]]] = defaultdict(dict)
    edges_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_datasets: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != set(CSV_FIELDS):
            raise OutputValidationError("submission row schema mismatch")
        if _integer(row["id"], "id") != index:
            raise OutputValidationError("submission id is not contiguous from zero")
        dataset = str(row["dataset"])
        if dataset not in expected_by_dataset:
            raise OutputValidationError(f"unexpected submission dataset {dataset!r}")
        seen_datasets.add(dataset)
        values = {name: _integer(row[name], name) for name in CSV_FIELDS if name not in {"dataset", "row_type"}}
        if row["row_type"] == "node":
            if values["source_id"] != -1 or values["target_id"] != -1 or values["node_id"] < 0:
                raise OutputValidationError("invalid node row sentinels")
            shape = expected_by_dataset[dataset].image_shape
            point = (values["t"], values["z"], values["y"], values["x"])
            if any(value < 0 or value >= bound for value, bound in zip(point, shape, strict=True)):
                raise OutputValidationError("node row lies outside its registered image")
            if values["node_id"] in node_by_dataset[dataset]:
                raise OutputValidationError("duplicate node ID in a dataset")
            node_by_dataset[dataset][values["node_id"]] = point
        elif row["row_type"] == "edge":
            if any(values[name] != -1 for name in ("node_id", "t", "z", "y", "x")):
                raise OutputValidationError("invalid edge row sentinels")
            edges_by_dataset[dataset].append((values["source_id"], values["target_id"]))
        else:
            raise OutputValidationError(f"unknown row_type {row['row_type']!r}")
    if seen_datasets != set(expected_by_dataset):
        raise OutputValidationError("submission does not cover every requested dataset")
    for dataset in expected_by_dataset:
        nodes = node_by_dataset[dataset]
        if not nodes:
            raise OutputValidationError(f"dataset {dataset} has no node rows")
        seen_edges: set[tuple[int, int]] = set()
        indegree: dict[int, int] = defaultdict(int)
        outdegree: dict[int, int] = defaultdict(int)
        for source, target in edges_by_dataset[dataset]:
            if source not in nodes or target not in nodes or (source, target) in seen_edges:
                raise OutputValidationError("edge has a missing node reference or is duplicated")
            seen_edges.add((source, target))
            if nodes[target][0] != nodes[source][0] + 1:
                raise OutputValidationError("submission edge is not frame-adjacent")
            indegree[target] += 1
            outdegree[source] += 1
            if indegree[target] > 1 or outdegree[source] > 2:
                raise OutputValidationError("submission lineage has a merge or outdegree > 2")
    return {
        "rows": len(rows),
        "nodes": sum(len(value) for value in node_by_dataset.values()),
        "edges": sum(len(value) for value in edges_by_dataset.values()),
        "datasets": len(seen_datasets),
    }


def write_rows_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    deadline_epoch: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ResumeHashError(f"refusing to overwrite output fragment {path}")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                check_deadline(deadline_epoch, f"{path.name} row write")
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        check_deadline(deadline_epoch, f"{path.name} finalization")
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
        check_deadline(deadline_epoch, f"{path.name} finalization")
    finally:
        temporary.unlink(missing_ok=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise OutputValidationError(f"CSV schema mismatch in {path}")
        return list(reader)


def _resume_result(
    status_path: Path,
    fragment_path: Path,
    expected_run: Mapping[str, Any],
    *,
    feature_path: Path | None = None,
    expected_identity: str | None = None,
) -> dict[str, Any] | None:
    if status_path.exists() != fragment_path.exists():
        raise ResumeHashError(f"orphaned status/fragment pair for {status_path.stem}")
    if not status_path.exists():
        return None
    status = strict_json_load(status_path)
    if not isinstance(status, Mapping) or status.get("schema") != STATUS_SCHEMA or status.get("ok") is not True:
        raise ResumeHashError(f"invalid completed status {status_path}")
    if status.get("run_contract") != expected_run:
        raise ResumeHashError(f"resume run hash mismatch for {status_path.stem}")
    if expected_identity is not None and status.get("identity") != expected_identity:
        raise ResumeHashError(f"resume identity mismatch for {status_path.stem}")
    if status.get("fragment_sha256") != sha256_file(fragment_path):
        raise ResumeHashError(f"resume fragment byte hash mismatch for {status_path.stem}")
    result_value = status.get("result")
    if not isinstance(result_value, Mapping):
        raise ResumeHashError(f"resume result is invalid for {status_path.stem}")
    result = dict(result_value)
    if feature_path is not None:
        if not feature_path.is_file():
            raise ResumeHashError(f"resume feature summary is missing for {status_path.stem}")
        if status.get("feature_sha256") != sha256_file(feature_path):
            raise ResumeHashError(f"resume feature byte hash mismatch for {status_path.stem}")
        if result.get("fragment") != str(fragment_path) or result.get("feature_summary") != str(feature_path):
            raise ResumeHashError(f"resume artifact path mismatch for {status_path.stem}")
    result["resumed"] = True
    return result


def _video_transaction(
    output_dir: Path,
    job: VideoJob,
    inference_value: Mapping[str, Any],
    run_value: Mapping[str, Any],
    *,
    feature_path: Path,
    fragment_path: Path,
    status_path: Path,
) -> tuple[Path, dict[str, Any]]:
    transaction_path = output_dir / "transactions" / f"{job.slug}.json"
    transaction = {
        "schema": TRANSACTION_SCHEMA,
        "identity": job.identity,
        "inference_contract": dict(inference_value),
        "run_contract": dict(run_value),
        "artifacts": {
            "feature": str(feature_path),
            "fragment": str(fragment_path),
            "status": str(status_path),
        },
    }
    return transaction_path, transaction


def _prepare_video_transaction(
    output_dir: Path,
    job: VideoJob,
    inference_value: Mapping[str, Any],
    run_value: Mapping[str, Any],
    *,
    feature_path: Path,
    fragment_path: Path,
    status_path: Path,
    deadline_epoch: float | None,
) -> dict[str, Any] | None:
    transaction_path, expected_transaction = _video_transaction(
        output_dir,
        job,
        inference_value,
        run_value,
        feature_path=feature_path,
        fragment_path=fragment_path,
        status_path=status_path,
    )
    owned = _validate_intent(transaction_path, expected_transaction, "video transaction")
    if status_path.exists() or fragment_path.exists():
        if status_path.exists() and fragment_path.exists():
            return _resume_result(
                status_path,
                fragment_path,
                run_value,
                feature_path=feature_path,
                expected_identity=job.identity,
            )
        if status_path.exists():
            raise ResumeHashError(f"orphaned completed status for {job.identity}")
    incomplete = tuple(path for path in (feature_path, fragment_path) if path.exists())
    if incomplete:
        if not owned:
            raise ResumeHashError(f"refusing to remove unowned incomplete outputs for {job.identity}")
        check_deadline(deadline_epoch, f"{job.identity} output recovery")
        _remove_owned_artifacts(
            incomplete,
            intent_path=transaction_path,
            expected_intent=expected_transaction,
            label="video transaction",
        )
    if not owned:
        check_deadline(deadline_epoch, f"{job.identity} transaction intent")
        _claim_intent(transaction_path, expected_transaction, "video transaction", deadline_epoch)
    return None


def process_video(
    job: VideoJob,
    *,
    model: Any,
    cache_identity: Any,
    output_dir: Path,
    cache_dir: Path,
    graph_variant: str = LOCKED_GRAPH_VARIANT,
    pair_batch_size: int = DEFAULT_BATCH_SIZE,
    view_batch_size: int = 1,
    d4_views: Sequence[int] | None = None,
    expected_frames: int = FRAME_COUNT,
    deadline_epoch: float | None = None,
    router: Any | None = None,
    router_sha256: str | None = None,
    accelerated_gate: Mapping[str, Any] | None = None,
    shared_zip_store: Any | None = None,
    model_module: Any | None = None,
    graph_module: Any | None = None,
    route_module: Any | None = None,
    infer_video_fn: Callable[..., Any] | None = None,
    a0_scorer: Any | None = None,
    edge_scorer: Any | None = None,
    edge_artifact_identity: Mapping[str, Any] | None = None,
    edge_module: Any | None = None,
    p0_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or contract-safely resume one whole video transaction."""

    if pair_batch_size != 1:
        raise RunnerError("production bh_model pair batching is locked to pair_batch_size=1")
    if graph_variant == LOCKED_GRAPH_VARIANT and a0_scorer is not None:
        raise RunnerError("motion graph must not receive an A0 scorer")
    if graph_variant == A0_GRAPH_VARIANT and a0_scorer is None:
        raise RunnerError("A0 graph requires the worker-local checkpoint scorer")
    if edge_scorer is not None and (graph_variant != LOCKED_GRAPH_VARIANT or a0_scorer is not None):
        raise RunnerError("edge artifact association requires the locked motion graph without A0")
    if (edge_scorer is None) != (edge_artifact_identity is None):
        raise RunnerError("edge scorer and artifact identity must be supplied together")
    model_module = model_module or importlib.import_module("celltrack_alpha.bh_model")
    graph_module = graph_module or importlib.import_module("celltrack_alpha.bh_graph")
    route_module = route_module or importlib.import_module("celltrack_alpha.bh_route")
    infer_video_fn = infer_video_fn or model_module.infer_video
    view_execution = view_batch_report(
        pair_batch_size=pair_batch_size,
        view_batch_size=view_batch_size,
        accelerated_gate=accelerated_gate,
        d4_views=d4_views,
    )
    infer_value = inference_contract(
        job,
        cache_identity,
        pair_batch_size=pair_batch_size,
        view_batch_size=view_batch_size,
        frame_count=expected_frames,
        d4_views=d4_views,
    )
    run_value = run_contract(
        infer_value,
        graph_variant=graph_variant,
        router_sha256=router_sha256,
        view_execution=view_execution,
        edge_artifact=edge_artifact_identity,
        p0_calibration=p0_calibration,
    )
    fragment_path = output_dir / "fragments" / f"{job.slug}.csv"
    status_path = output_dir / "status" / f"{job.slug}.json"
    feature_path = output_dir / "features" / f"{job.slug}.json"
    resumed = _prepare_video_transaction(
        output_dir,
        job,
        infer_value,
        run_value,
        feature_path=feature_path,
        fragment_path=fragment_path,
        status_path=status_path,
        deadline_epoch=deadline_epoch,
    )
    if resumed is not None:
        return resumed

    check_deadline(deadline_epoch, f"{job.identity} startup")
    handle = open_video(job, expected_frames=expected_frames, shared_zip_store=shared_zip_store)
    started = time.perf_counter()
    frames: list[Any] | None = None
    try:
        q_low, q_high = read_quantile_attrs(handle.attrs)
        scale = read_scale_attrs(handle.attrs)
        loaded = load_inference_cache(
            cache_dir,
            job,
            infer_value,
            cache_identity,
            model_module,
            deadline_epoch,
        )
        if loaded is None:
            frames = read_all_frames(handle.array, expected_frames, deadline_epoch)
            check_deadline(deadline_epoch, f"{job.identity} bh_model.infer_video")
            inference = infer_video_fn(
                model,
                frames,
                q_low=q_low,
                q_high=q_high,
                start_frame_index=0,
                node_id_start=0,
                batch_size=pair_batch_size,
                view_batch_size=view_batch_size,
                d4_views=d4_views,
            )
            check_deadline(deadline_epoch, f"{job.identity} bh_model.infer_video")
            if (
                getattr(inference, "pair_batch_size", pair_batch_size) != pair_batch_size
                or getattr(inference, "view_batch_size", view_batch_size) != view_batch_size
                or tuple(getattr(inference, "d4_views", view_execution["d4_views"]))
                != tuple(view_execution["d4_views"])
            ):
                raise RunnerError("bh_model inference D4/batching provenance disagrees with the request")
            pregraph = build_route_summary(
                inference.detections,
                frame_shape_zyx=job.record.image_shape[1:],
                scale_zyx=scale,
                q_low=q_low,
                q_high=q_high,
                frames=frames,
                route_module=route_module,
            )
            cache, _meta = save_inference_cache(
                cache_dir,
                job,
                infer_value,
                cache_identity,
                inference,
                pregraph,
                model_module,
                deadline_epoch,
            )
            cache_reused = False
        else:
            cache, meta = loaded
            pregraph = meta.get("pregraph_features")
            if not isinstance(pregraph, Mapping):
                raise ResumeHashError(f"cache lacks persisted bh_route features for {job.identity}")
            cache_reused = True
        route_decision = choose_route(pregraph, router)
        graph_nodes = detections_to_graph_nodes(cache.detections, graph_module.GraphNode)
        learned_probabilities = None
        source_no_link_probabilities = None
        a0_audit = None
        edge_audit = None
        if graph_variant == A0_GRAPH_VARIANT:
            learned_probabilities, a0_audit = score_a0_cache(
                cache,
                a0_scorer,
                cache_identity,
                edge_module=edge_module,
            )
            a0_audit["cache"]["reused"] = cache_reused
        elif edge_scorer is not None:
            learned_probabilities, source_no_link_probabilities, edge_audit = score_sparse_edge_cache(
                cache,
                edge_scorer,
                edge_module=edge_module,
            )
            edge_audit["cache_reused"] = cache_reused
        refiner_source = frames if frames is not None else handle.array
        refiner = LazyImageRefiner(
            refiner_source,
            graph_module.intensity_centroid_refine,
            deadline_epoch=deadline_epoch,
        )
        check_deadline(deadline_epoch, f"{job.identity} bh_graph calibration")
        graph_options: dict[str, Any] = {
            "learned_edge_probabilities": learned_probabilities,
            "image_refine_callback": refiner,
        }
        if edge_scorer is not None:
            graph_options.update(
                {
                    "source_no_link_probabilities": source_no_link_probabilities,
                    "topology_mode": "legacy",
                    "division_enabled": False,
                }
            )
        sweep_provenance = None
        if p0_calibration is not None:
            sweep_provenance = {
                "schema": graph_module.EXPERIMENTAL_SWEEP_SCHEMA,
                "experiment_id": p0_calibration["experiment_id"],
                "purpose": p0_calibration["purpose"],
                "varied_fields": ["motion_error_weight"],
                "production_eligible": False,
                "submit_eligible": True,
            }
            graph_options.update(
                {
                    "_experimental_sweep_provenance": sweep_provenance,
                    "_experimental_motion_error_weight": p0_calibration[
                        "motion_error_weight"
                    ],
                }
            )
        graph_result = graph_module.calibrate_v45_graph(graph_nodes, **graph_options)
        check_deadline(deadline_epoch, f"{job.identity} bh_graph calibration")
        if graph_result.provenance.get("full_frame_dense_matrix") is not False:
            raise RunnerError("bh_graph did not attest sparse graph execution")
        if p0_calibration is None:
            if (
                graph_result.provenance.get("production_eligible") is not True
                or "experimental_sweep" in graph_result.provenance
            ):
                raise RunnerError("production submission rejected experimental graph configuration")
        elif (
            graph_result.provenance.get("production_eligible") is not False
            or graph_result.provenance.get("experimental_sweep") != sweep_provenance
            or graph_result.provenance.get("motion_error_weight")
            != p0_calibration["motion_error_weight"]
        ):
            raise RunnerError("P0 calibration graph lost its non-production provenance")
        if a0_audit is not None:
            a0_audit["graph"] = {
                "variant": A0_GRAPH_VARIANT,
                "learned_probability_count": len(learned_probabilities),
                "full_frame_dense_matrix": False,
            }
        if edge_audit is not None:
            edge_audit["graph"] = {
                "variant": LOCKED_GRAPH_VARIANT,
                "learned_probability_count": len(learned_probabilities),
                "full_frame_dense_matrix": False,
                "division_enabled": False,
            }
        graph_result = project_graph_result_to_image_support(
            job.record.dataset,
            graph_result,
            job.record.image_shape,
            scale_zyx=graph_module.V45_GRAPH_CONFIG.scale_zyx,
            max_correction_um=graph_module.V45_GRAPH_CONFIG.refine_max_shift_um,
        )
        projection = dict(graph_result.provenance["image_support_projection"])
        postgraph = build_route_summary(
            cache.detections,
            frame_shape_zyx=job.record.image_shape[1:],
            scale_zyx=scale,
            q_low=q_low,
            q_high=q_high,
            frames=frames,
            graph_result=graph_result,
            route_module=route_module,
        )
        feature_record = {
            "schema": FEATURE_SCHEMA,
            "identity": job.identity,
            "pregraph": pregraph,
            "postgraph": postgraph,
            "route": route_decision,
            "graph_variant": graph_variant,
            "view_execution": view_execution,
            "run_contract": run_value,
            "image_support_projection": projection,
        }
        if a0_audit is not None:
            feature_record["a0_audit"] = a0_audit
        if edge_audit is not None:
            feature_record["edge_artifact"] = dict(edge_artifact_identity)
            feature_record["edge_association"] = edge_audit
        atomic_create_json(feature_path, feature_record, deadline_epoch)
        rows = graph_rows(job.record.dataset, graph_result, job.record.image_shape)
        for index, row in enumerate(rows):
            row["id"] = index
        validate_submission_rows(rows, [job.record])
        write_rows_atomic(fragment_path, rows, deadline_epoch=deadline_epoch)
        result = {
            "identity": job.identity,
            "dataset": job.record.dataset,
            "fragment": str(fragment_path),
            "feature_summary": str(feature_path),
            "frames": expected_frames,
            "nodes": len(graph_result.nodes),
            "edges": len(graph_result.edges),
            "rows": len(rows),
            "cache_reused": cache_reused,
            "cache_contract_sha256": infer_value["sha256"],
            "run_contract_sha256": run_value["sha256"],
            "route": route_decision,
            "graph_variant": graph_variant,
            "image_support_projection": projection,
            "lazy_refinement": refiner.report(),
            "seconds": time.perf_counter() - started,
            "resumed": False,
        }
        if a0_audit is not None:
            result["a0_audit"] = a0_audit
        if edge_audit is not None:
            result["edge_artifact"] = dict(edge_artifact_identity)
            result["edge_association"] = edge_audit
        status = {
            "schema": STATUS_SCHEMA,
            "ok": True,
            "identity": job.identity,
            "run_contract": run_value,
            "fragment_sha256": sha256_file(fragment_path),
            "feature_sha256": sha256_file(feature_path),
            "result": result,
        }
        if a0_audit is not None:
            status["a0_audit"] = a0_audit
        if edge_audit is not None:
            status["edge_artifact"] = dict(edge_artifact_identity)
            status["edge_association"] = edge_audit
        atomic_create_json(status_path, status, deadline_epoch)
        return result
    finally:
        handle.close()


def merge_fragments(
    output_path: Path,
    jobs: Sequence[VideoJob],
    statuses: Sequence[Mapping[str, Any]],
    *,
    deadline_epoch: float | None = None,
) -> dict[str, int]:
    """Merge in registry identity order, independent of shard completion order."""

    if output_path.exists():
        raise OutputValidationError(f"refusing to overwrite merged submission {output_path}")
    by_identity: dict[str, Mapping[str, Any]] = {}
    for status in statuses:
        identity = str(status.get("identity", ""))
        if identity in by_identity:
            raise OutputValidationError(f"duplicate completed status for {identity}")
        if status.get("schema") != STATUS_SCHEMA or status.get("ok") is not True:
            raise OutputValidationError(f"unsuccessful video status for {identity}")
        by_identity[identity] = status
    ordered_jobs = sorted(jobs, key=lambda job: job.identity)
    if set(by_identity) != {job.identity for job in ordered_jobs}:
        raise OutputValidationError("completed fragments do not cover exactly the requested datasets")
    rows: list[dict[str, Any]] = []
    for job in ordered_jobs:
        check_deadline(deadline_epoch, f"{job.identity} deterministic merge")
        status = by_identity[job.identity]
        fragment = Path(status["result"]["fragment"])
        if status.get("fragment_sha256") != sha256_file(fragment):
            raise ResumeHashError(f"fragment hash mismatch during merge for {job.identity}")
        fragment_rows = read_rows(fragment)
        if not fragment_rows:
            raise OutputValidationError(f"empty fragment for {job.identity}")
        for row in fragment_rows:
            row["id"] = len(rows)
            rows.append(row)
    summary = validate_submission_rows(rows, [job.record for job in ordered_jobs])
    write_rows_atomic(output_path, rows, deadline_epoch=deadline_epoch)
    return summary


def generate_cv_rungs(
    split_manifest_path: str | os.PathLike[str] = DEFAULT_SPLITS,
    *,
    registry_path: str | os.PathLike[str] | None = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Generate deterministic nested 10/30/100 OOF rungs from own held-out folds."""

    source = Path(split_manifest_path)
    manifest = strict_json_load(source)
    if not isinstance(manifest, Mapping) or manifest.get("artifact") != "grouped_3fold_audit20_v1":
        raise RunnerError("CV source is not grouped_3fold_audit20_v1")
    if registry_path is not None:
        registry_sha = sha256_file(registry_path)
        if manifest.get("dataset_registry_sha256") != registry_sha:
            raise RunnerError("CV split references another dataset registry hash")
    audit = set(manifest["audit"]["video_ids"])
    fold_by_video: dict[str, int] = {}
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {"0", "1", "2"}:
        raise RunnerError("CV split must contain folds 0, 1, and 2")
    for fold_text, fold_record in sorted(folds.items()):
        held = fold_record.get("held_out_video_ids")
        if not isinstance(held, list):
            raise RunnerError(f"fold {fold_text} lacks held_out_video_ids")
        for video_id in held:
            identity = str(video_id)
            if identity in audit:
                raise RunnerError("audit video leaked into held-out CV candidates")
            if identity in fold_by_video:
                raise RunnerError(f"video has multiple held-out folds: {identity}")
            fold_by_video[identity] = int(fold_text)
    if len(fold_by_video) < 100:
        raise RunnerError("fewer than 100 non-audit held-out videos are available")
    seed = str(manifest["dataset_registry_sha256"])
    ordered = sorted(
        fold_by_video,
        key=lambda identity: (hashlib.sha256(f"{seed}\0{identity}".encode("ascii")).hexdigest(), identity),
    )
    rungs: dict[str, list[dict[str, Any]]] = {}
    by_fold: dict[str, dict[str, list[str]]] = {}
    for size in (10, 30, 100):
        records = [
            {
                "video_id": identity,
                "dataset": identity.split(":", 1)[1],
                "held_out_fold": fold_by_video[identity],
                "prediction_source": f"future-fold-{fold_by_video[identity]}-oof",
                "checkpoint_exposure": CHECKPOINT_EXPOSURE,
            }
            for identity in ordered[:size]
        ]
        rungs[str(size)] = records
        by_fold[str(size)] = {
            str(fold): [record["video_id"] for record in records if record["held_out_fold"] == fold]
            for fold in range(3)
        }
    if [record["video_id"] for record in rungs["10"]] != [record["video_id"] for record in rungs["30"][:10]]:
        raise RunnerError("10/30 CV rungs are not nested")
    if [record["video_id"] for record in rungs["30"]] != [record["video_id"] for record in rungs["100"][:30]]:
        raise RunnerError("30/100 CV rungs are not nested")
    selected = {record["video_id"] for record in rungs["100"]}
    if selected & audit:
        raise RunnerError("audit video leaked into fixed CV rungs")
    return {
        "schema": RUNG_SCHEMA,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "dataset_registry_sha256": manifest["dataset_registry_sha256"],
        "checkpoint_exposure": CHECKPOINT_EXPOSURE,
        "audit_excluded": True,
        "audit_video_count": len(audit),
        "selection": "sha256(dataset_registry_sha256, NUL, video_id)",
        "rungs": rungs,
        "by_fold": by_fold,
        "nested": True,
        "future_fold_specific_oof_supported": True,
    }


def _worker_entry(job_path_text: str) -> None:
    job_path = Path(job_path_text)
    payload = strict_json_load(job_path)
    aggregate_path = Path(payload["aggregate_status"])
    results: list[dict[str, Any]] = []
    active_identity: str | None = None
    store = None
    try:
        deadline_epoch = float(payload["deadline_epoch"])
        check_deadline(deadline_epoch, "worker startup")
        bh_model = importlib.import_module("celltrack_alpha.bh_model")
        bh_route = importlib.import_module("celltrack_alpha.bh_route")
        model, bundle = bh_model.load_v45_model(payload["support_root"], device=payload["device"])
        identity = bh_model.CacheIdentity.from_artifact(
            bundle,
            pair_batch_size=int(payload["pair_batch_size"]),
            view_batch_size=int(payload["view_batch_size"]),
            d4_views=payload.get("d4_views"),
        )
        if _identity_dict(identity) != payload["cache_identity"]:
            raise ResumeHashError("worker strict model identity differs from parent preflight")
        edge_module = (
            importlib.import_module("celltrack_alpha.bh_edge")
            if payload["graph_variant"] == A0_GRAPH_VARIANT or payload.get("edge_artifact") is not None
            else None
        )
        a0_scorer = build_a0_scorer(
            payload["graph_variant"],
            model,
            bundle,
            identity,
            edge_module=edge_module,
        )
        edge_scorer, edge_artifact_identity = load_sparse_edge_scorer(
            payload.get("edge_artifact"),
            payload.get("edge_artifact_sha256"),
            payload.get("edge_auxiliary_checkpoint", str(DEFAULT_EDGE_AUXILIARY_CHECKPOINT)),
            model,
            bundle,
            edge_module=edge_module,
        )
        expected_edge_identity = payload.get("edge_artifact_identity")
        if edge_artifact_identity != expected_edge_identity:
            raise ResumeHashError("worker edge artifact identity differs from parent validation")
        router = None
        router_sha = payload.get("router_sha256")
        if payload.get("router_json") is not None:
            router, observed_sha = load_validated_router(payload["router_json"], bh_route)
            if observed_sha != router_sha:
                raise ResumeHashError("worker router hash differs from parent preflight")
        jobs = [VideoJob.from_dict(item) for item in payload["jobs"]]
        if jobs and jobs[0].input_kind == "zip":
            zarr = importlib.import_module("zarr")
            store = zarr.storage.ZipStore(jobs[0].input_path, mode="r")
        for job in jobs:
            active_identity = job.identity
            check_deadline(deadline_epoch, f"worker {payload['rank']} before {job.identity}")
            result = process_video(
                job,
                model=model,
                cache_identity=identity,
                output_dir=Path(payload["output_dir"]),
                cache_dir=Path(payload["cache_dir"]),
                graph_variant=payload["graph_variant"],
                pair_batch_size=int(payload["pair_batch_size"]),
                view_batch_size=int(payload["view_batch_size"]),
                d4_views=payload.get("d4_views"),
                expected_frames=FRAME_COUNT,
                deadline_epoch=deadline_epoch,
                router=router,
                router_sha256=router_sha,
                accelerated_gate=payload.get("accelerated_gate"),
                shared_zip_store=store,
                a0_scorer=a0_scorer,
                edge_scorer=edge_scorer,
                edge_artifact_identity=edge_artifact_identity,
                edge_module=edge_module,
                p0_calibration=payload.get("p0_calibration"),
            )
            results.append(result)
        worker_status = {
            "ok": True,
            "rank": payload["rank"],
            "device": payload["device"],
            "results": results,
        }
        if a0_scorer is not None:
            video_audits = [result["a0_audit"] for result in results]
            worker_audit = (
                aggregate_a0_audits(video_audits, scope="worker")
                if video_audits
                else empty_a0_worker_audit(a0_scorer, identity)
            )
            worker_audit["worker"] = {
                "rank": payload["rank"],
                "device": payload["device"],
                "scorer_instances": 1,
            }
            worker_status["a0_audit"] = worker_audit
        atomic_json(
            aggregate_path,
            worker_status,
            deadline_epoch,
        )
    except Exception as exc:
        atomic_json(
            aggregate_path,
            {
                "ok": False,
                "rank": payload.get("rank"),
                "device": payload.get("device"),
                "identity": active_identity,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-12000:],
                "results": results,
            },
        )
        raise
    finally:
        if store is not None and callable(getattr(store, "close", None)):
            store.close()


def _bounded_worker_failure(
    processes: Sequence[Any],
    aggregate_paths: Sequence[Path],
    observed_exitcodes: Sequence[int | None],
) -> str:
    """Surface child diagnostics without copying job payloads or image data."""

    allowed_result_fields = (
        "identity",
        "dataset",
        "frames",
        "nodes",
        "edges",
        "rows",
        "cache_reused",
        "graph_variant",
        "image_support_projection",
        "a0_audit",
        "seconds",
        "resumed",
    )
    workers: list[dict[str, Any]] = []
    for path in aggregate_paths:
        if not path.is_file():
            continue
        try:
            record = strict_json_load(path)
        except RunnerError as exc:
            workers.append({"aggregate": path.name, "error": str(exc)[:2000]})
            continue
        if not isinstance(record, Mapping):
            workers.append({"aggregate": path.name, "error": "worker status is not an object"})
            continue
        raw_results = record.get("results")
        safe_results = []
        if isinstance(raw_results, list):
            for result in raw_results:
                if isinstance(result, Mapping):
                    safe_results.append(
                        {field: result[field] for field in allowed_result_fields if field in result}
                    )
        workers.append(
            {
                "rank": record.get("rank"),
                "device": record.get("device"),
                "identity": record.get("identity"),
                "error": str(record.get("error", ""))[:4000],
                "results": safe_results,
                "traceback": str(record.get("traceback", ""))[-12000:],
            }
        )
    details = {
        "observed_exitcodes": list(observed_exitcodes),
        "final_exitcodes": [process.exitcode for process in processes],
        "workers": workers,
    }
    return json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def wait_spawned(
    processes: Sequence[Any],
    deadline_epoch: float,
    aggregate_paths: Sequence[Path] = (),
) -> None:
    while True:
        codes = [process.exitcode for process in processes]
        if all(code is not None for code in codes):
            if any(code != 0 for code in codes):
                raise RunnerError(
                    "spawned worker failure: "
                    + _bounded_worker_failure(processes, aggregate_paths, codes)
                )
            return
        if any(code not in {None, 0} for code in codes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            raise RunnerError(
                "spawned worker failure: "
                + _bounded_worker_failure(processes, aggregate_paths, codes)
            )
        if time.time() >= deadline_epoch:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            raise DeadlineError("global deadline reached while waiting for spawned workers")
        time.sleep(0.05)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started_epoch = time.time()
    deadline_epoch = started_epoch + float(args.deadline_min) * 60.0
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = output_dir / args.output_file
    if submission.exists():
        raise RunnerError(f"refusing to overwrite existing submission {submission}")
    d4_views = TEST_ONLY_ONE_VIEW if args.test_only_one_view else PRODUCTION_D4_VIEWS
    edge_artifact_identity = pinned_edge_artifact_record(
        args.edge_artifact,
        args.edge_artifact_sha256,
    )
    p0_calibration = p0_calibration_contract(
        args.p0_experiment_id,
        args.p0_motion_error_weight,
    )
    preflight_report = preflight(
        support_root=args.support_root,
        report_path=output_dir / "preflight.json",
        require_cuda=not args.allow_cpu,
        source_budget_bytes=args.source_budget_bytes,
        pair_batch_size=args.pair_batch_size,
        view_batch_size=args.view_batch_size,
        d4_views=d4_views,
    )
    check_deadline(deadline_epoch, "post-preflight")
    requested = [item.strip() for item in args.datasets.split(",")] if args.datasets else None
    if args.mode == "submit":
        if args.split != "test" or requested is not None:
            raise RunnerError("submit mode is image-only and must discover the complete mounted test split")
        jobs = discover_submit_jobs(args.input)
        input_contract = submit_input_contract_report(jobs)
        atomic_json(output_dir / "input_contract.json", input_contract, deadline_epoch)
        if not input_contract["ok"]:
            raise RunnerError(
                "hidden submission input violated locked V45 assumptions: "
                + ", ".join(input_contract["triggered"])
            )
    else:
        records = load_registry(args.registry)
        selected = select_records(records, split=args.split, requested=requested)
        jobs = discover_jobs(args.input, selected)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if args.allow_cpu:
        if devices != ["cpu"]:
            raise RunnerError("--allow-cpu requires --devices cpu")
    elif len(devices) != 2 or any(not item.startswith("cuda:") for item in devices):
        raise RunnerError("production requires exactly two devices such as cuda:0,cuda:1")
    if args.pair_batch_size != 1:
        raise RunnerError("production pair batch size is locked to 1")
    router, router_sha = load_validated_router(args.router_json)
    del router
    bh_model = importlib.import_module("celltrack_alpha.bh_model")
    hashes = preflight_report["checkpoint"]["hashes"]
    numerical_config_sha256 = bh_model.v45_config_sha(
        pair_batch_size=args.pair_batch_size,
        view_batch_size=args.view_batch_size,
        d4_views=d4_views,
    )
    accelerated_gate = load_accelerated_score_gate(
        args.accelerated_score_gate,
        pair_batch_size=args.pair_batch_size,
        view_batch_size=args.view_batch_size,
        checkpoint_sha256=hashes["weight_sha256"],
        numerical_config_sha256=numerical_config_sha256,
    )
    cache_identity = bh_model.CacheIdentity(
        numerical_config_sha256,
        hashes["source_sha256"],
        hashes["weight_sha256"],
        args.pair_batch_size,
        args.view_batch_size,
    )
    cache_dir = (args.cache_dir or (output_dir / "cache")).resolve()
    shards = shard_jobs(jobs, len(devices))
    context = multiprocessing.get_context("spawn")
    processes = []
    aggregate_paths: list[Path] = []
    for rank, (device, shard) in enumerate(zip(devices, shards, strict=True)):
        aggregate = output_dir / f"worker{rank}.json"
        job_path = output_dir / f"worker{rank}.job.json"
        aggregate_paths.append(aggregate)
        atomic_json(
            job_path,
            {
                "rank": rank,
                "device": device,
                "jobs": [job.as_dict() for job in shard],
                "output_dir": str(output_dir),
                "cache_dir": str(cache_dir),
                "support_root": str(Path(args.support_root).resolve()),
                "graph_variant": args.graph_variant,
                "pair_batch_size": args.pair_batch_size,
                "view_batch_size": args.view_batch_size,
                "d4_views": list(d4_views),
                "accelerated_gate": accelerated_gate,
                "edge_artifact": str(Path(args.edge_artifact).resolve()) if args.edge_artifact else None,
                "edge_artifact_sha256": args.edge_artifact_sha256,
                "edge_artifact_identity": edge_artifact_identity,
                "p0_calibration": p0_calibration,
                "edge_auxiliary_checkpoint": str(DEFAULT_EDGE_AUXILIARY_CHECKPOINT.resolve()),
                "router_json": str(Path(args.router_json).resolve()) if args.router_json else None,
                "router_sha256": router_sha,
                "cache_identity": _identity_dict(cache_identity),
                "deadline_epoch": deadline_epoch,
                "aggregate_status": str(aggregate),
            },
            deadline_epoch,
        )
        process = context.Process(target=_worker_entry, args=(str(job_path),), name=f"bh-gpu-{rank}")
        process.start()
        processes.append(process)
    try:
        wait_spawned(processes, deadline_epoch, aggregate_paths)
        aggregates = [strict_json_load(path) for path in aggregate_paths]
        if any(not item.get("ok") for item in aggregates):
            raise RunnerError(f"one or more workers failed: {aggregates}")
        statuses = [
            strict_json_load(output_dir / "status" / f"{job.slug}.json")
            for job in jobs
        ]
        merged = merge_fragments(submission, jobs, statuses, deadline_epoch=deadline_epoch)
        check_deadline(deadline_epoch, "submission report")
        report = {
            "schema": RUNNER_SCHEMA,
            "ok": True,
            "mode": args.mode,
            "deadline_minutes": args.deadline_min,
            "elapsed_seconds": time.time() - started_epoch,
            "devices": devices,
            "model_loads": len(devices),
            "whole_video_shards": [[job.identity for job in shard] for shard in shards],
            "submission": {"path": str(submission), "sha256": sha256_file(submission), **merged},
            "requested_identities": [job.identity for job in jobs],
            "input_discovery": {
                "mode": "runtime-mounted-test" if args.mode == "submit" else "exact-registry",
                "identity_provenance": sorted(
                    {job.record.identity_provenance for job in jobs}
                ),
            },
            "graph_variant": args.graph_variant,
            "view_execution": view_batch_report(
                pair_batch_size=args.pair_batch_size,
                view_batch_size=args.view_batch_size,
                accelerated_gate=accelerated_gate,
                d4_views=d4_views,
            ),
            "router_sha256": router_sha,
            "checkpoint_exposure": CHECKPOINT_EXPOSURE,
            "preflight": preflight_report,
            "workers": aggregates,
        }
        if p0_calibration is not None:
            report["p0_calibration"] = p0_calibration
        if edge_artifact_identity is not None:
            report["edge_artifact"] = edge_artifact_identity
        if args.graph_variant == A0_GRAPH_VARIANT:
            video_audits = [
                result["a0_audit"]
                for aggregate in aggregates
                for result in aggregate["results"]
            ]
            run_audit = aggregate_a0_audits(video_audits, scope="run")
            run_audit["workers"] = {
                "worker_count": len(aggregates),
                "scorer_instances": sum(
                    int(aggregate["a0_audit"]["worker"]["scorer_instances"])
                    for aggregate in aggregates
                ),
            }
            report["a0_audit"] = run_audit
        atomic_json(output_dir / "run.json", report, deadline_epoch)
        return report
    except Exception:
        submission.with_name(f".{submission.name}.tmp").unlink(missing_ok=True)
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Kaggle directory root or local competition ZIP")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--out-dir", type=Path, default=Path("bh-out"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-file", default="submission.csv")
    parser.add_argument("--mode", choices=("submit", "cv"), default="submit")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--datasets", help="Comma-separated registry dataset names or split:dataset identities")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument(
        "--pair-batch-size",
        "--batch-size",
        dest="pair_batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Adjacent frame-pair batch size; production is locked to 1",
    )
    parser.add_argument(
        "--view-batch-size",
        type=int,
        default=1,
        help="D4 views per encode: 1 is exact sequential; 2-8 are gated experimental modes",
    )
    parser.add_argument("--accelerated-score-gate", type=Path)
    parser.add_argument("--deadline-min", type=float, default=DEFAULT_DEADLINE_MINUTES)
    parser.add_argument("--graph-variant", choices=tuple(GRAPH_VARIANTS), default=LOCKED_GRAPH_VARIANT)
    parser.add_argument("--edge-artifact", type=Path)
    parser.add_argument("--edge-artifact-sha256")
    parser.add_argument(
        "--p0-experiment-id",
        choices=tuple(P0_APPROVED_MOTION_WEIGHTS),
        help="Approved non-production public metric-parity calibration ID",
    )
    parser.add_argument(
        "--p0-motion-error-weight",
        type=float,
        help="Motion weight fixed by --p0-experiment-id",
    )
    parser.add_argument(
        "--test-only-one-view",
        action="store_true",
        help="Parity testing only: run unaugmented D4 view 0 in CV/train mode; never valid for submission",
    )
    parser.add_argument("--router-json", type=Path)
    parser.add_argument("--source-budget-bytes", type=int, default=SOURCE_BUDGET_BYTES)
    parser.add_argument("--allow-cpu", action="store_true", help="Testing only; production requires two CUDA devices")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cv-rungs-out", type=Path)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.deadline_min <= 0 or args.deadline_min > DEFAULT_DEADLINE_MINUTES:
        raise RunnerError(
            f"hard deadline must be in (0, {DEFAULT_DEADLINE_MINUTES:g}] minutes"
        )
    if Path(args.output_file).name != args.output_file or not args.output_file.endswith(".csv"):
        raise RunnerError("--output-file must be one CSV basename")
    if args.source_budget_bytes <= 0:
        raise RunnerError("source budget must be positive")
    if args.pair_batch_size != 1:
        raise RunnerError("production pair batch size is locked to 1")
    if not 1 <= args.view_batch_size <= 8:
        raise RunnerError("--view-batch-size must be in [1, 8]")
    has_edge_artifact = args.edge_artifact is not None or args.edge_artifact_sha256 is not None
    if has_edge_artifact and (args.edge_artifact is None or args.edge_artifact_sha256 is None):
        raise RunnerError("--edge-artifact and --edge-artifact-sha256 must be supplied together")
    if has_edge_artifact and args.graph_variant != LOCKED_GRAPH_VARIANT:
        raise RunnerError("edge artifact association cannot be combined with --graph-variant a0")
    p0_calibration = p0_calibration_contract(
        args.p0_experiment_id,
        args.p0_motion_error_weight,
    )
    if p0_calibration is not None and (
        args.mode != "submit"
        or args.split != "test"
        or args.graph_variant != LOCKED_GRAPH_VARIANT
        or not has_edge_artifact
        or args.router_json is not None
        or args.accelerated_score_gate is not None
        or args.test_only_one_view
        or args.view_batch_size != 1
    ):
        raise RunnerError(
            "P0 calibration requires full submit/test, the locked edge-assisted motion graph, "
            "sequential D4 execution, and no router"
        )
    if args.test_only_one_view and (
        args.mode != "cv"
        or args.split != "train"
        or args.view_batch_size != 1
        or not has_edge_artifact
        or args.accelerated_score_gate is not None
    ):
        raise RunnerError(
            "--test-only-one-view requires CV/train, view batch size 1, a pinned edge artifact, and no accelerated gate"
        )
    if args.mode == "submit" and (args.split != "test" or args.datasets is not None):
        raise RunnerError("submit mode must discover the complete mounted test split")
    if args.input is None and not args.preflight_only and args.cv_rungs_out is None:
        raise RunnerError("--input is required for inference")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.cv_rungs_out is not None:
        manifest = generate_cv_rungs(args.split_manifest, registry_path=args.registry)
        atomic_json(args.cv_rungs_out, manifest)
        if args.input is None and not args.preflight_only:
            print(json.dumps({"cv_rungs": str(args.cv_rungs_out), "sha256": sha256_file(args.cv_rungs_out)}))
            return 0
    if args.preflight_only:
        report = preflight(
            support_root=args.support_root,
            report_path=args.out_dir / "preflight.json",
            require_cuda=not args.allow_cpu,
            source_budget_bytes=args.source_budget_bytes,
            pair_batch_size=args.pair_batch_size,
            view_batch_size=args.view_batch_size,
            d4_views=TEST_ONLY_ONE_VIEW if args.test_only_one_view else PRODUCTION_D4_VIEWS,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    report = run_pipeline(args)
    print(json.dumps({"submission": report["submission"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
