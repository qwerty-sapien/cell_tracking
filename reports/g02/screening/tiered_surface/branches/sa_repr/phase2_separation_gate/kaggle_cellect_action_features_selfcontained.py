#!/usr/bin/env python3
"""Generate only frozen 44b6 CELLECT action features on one bounded Kaggle GPU run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

# BEGIN INLINE CONTENT-ADDRESSED STEP 0
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from collections import defaultdict

SAFE_WEIGHT_NAME = "cellect_uext_x3rd_149_0_4p6540.pth"
BUNDLE_MANIFEST_NAME = "consolidated_bundle_manifest.json"


class Step0InputError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise Step0InputError(f"input root is not a directory: {root}")
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            by_hash[sha256(candidate)].append(candidate)
    return dict(by_hash)


def _unique_content(
    by_hash: dict[str, list[Path]],
    expected_sha256: str,
    *,
    label: str,
    expected_bytes: int | None = None,
) -> Path:
    matches = by_hash.get(expected_sha256, [])
    if len(matches) != 1:
        raise Step0InputError(
            f"{label} content resolution expected one match, found {len(matches)}"
        )
    path = matches[0]
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise Step0InputError(f"{label} byte count drift")
    return path


def _json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Step0InputError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise Step0InputError(f"{label} is not a JSON object")
    return value


def _record_by_role(
    manifest: dict[str, object], role: str
) -> list[dict[str, object]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise Step0InputError("bundle manifest has no file list")
    return [
        record
        for record in files
        if isinstance(record, dict) and record.get("role") == role
    ]


def _verified_file(
    by_hash: dict[str, list[Path]],
    record: dict[str, object],
    *,
    label: str,
) -> Path:
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise Step0InputError(f"{label} has invalid content pins")
    return _unique_content(
        by_hash,
        expected_sha256,
        label=label,
        expected_bytes=expected_bytes,
    )


def _safe_tar_member(name: str) -> bool:
    member = Path(name)
    return (
        bool(name)
        and name.isascii()
        and not member.is_absolute()
        and ".." not in member.parts
        and member.name == SAFE_WEIGHT_NAME
    )


def _resolve_checkpoint(
    bundle_root: Path,
    work_root: Path,
    expected_sha256: str,
) -> Path:
    matches: list[tuple[str, Path]] = []
    for suffix in ("*.pth", "*.bin"):
        for candidate in sorted(bundle_root.rglob(suffix)):
            if candidate.is_file() and sha256(candidate) == expected_sha256:
                matches.append((f"file:{candidate}", candidate))
    extraction_root = work_root / "cellect_checkpoint_step0"
    for archive_ordinal, archive in enumerate(sorted(bundle_root.rglob("*.tar"))):
        if not archive.is_file():
            continue
        try:
            handle = tarfile.open(archive, mode="r:")
        except tarfile.TarError as error:
            raise Step0InputError(f"invalid checkpoint tar: {archive}") from error
        with handle:
            for member_ordinal, member in enumerate(handle):
                if not member.isfile():
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise Step0InputError("checkpoint tar member is unreadable")
                digest = hashlib.sha256()
                payload = bytearray()
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
                    payload.extend(block)
                if digest.hexdigest() != expected_sha256:
                    continue
                if not _safe_tar_member(member.name):
                    raise Step0InputError(
                        f"matching checkpoint has unsafe member name: {member.name!r}"
                    )
                extraction_root.mkdir(parents=True, exist_ok=True)
                target = extraction_root / (
                    f"{archive_ordinal:04d}_{member_ordinal:04d}_{SAFE_WEIGHT_NAME}"
                )
                target.write_bytes(payload)
                matches.append((f"tar:{archive}:{member.name}", target))
    if len(matches) != 1:
        raise Step0InputError(
            f"checkpoint content resolution expected one match, found {len(matches)}"
        )
    source, checkpoint = matches[0]
    if source.startswith("file:") and bundle_root != work_root:
        extraction_root.mkdir(parents=True, exist_ok=True)
        target = extraction_root / SAFE_WEIGHT_NAME
        shutil.copyfile(checkpoint, target)
        checkpoint = target
    if sha256(checkpoint) != expected_sha256:
        raise Step0InputError("checkpoint changed during step-0 resolution")
    return checkpoint


def run_cellect_step0(
    input_root: Path,
    work_root: Path,
    *,
    expected_bundle_manifest_sha256: str,
    expected_request_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    expected_population_manifest_sha256: str,
    expected_wheels: dict[str, str],
) -> dict[str, object]:
    """Authenticate every CELLECT input before NumPy, torch, pip, or model work."""
    started = time.monotonic()
    by_hash = _inventory(input_root)
    bundle_manifest_path = _unique_content(
        by_hash,
        expected_bundle_manifest_sha256,
        label="bundle manifest",
    )
    manifest = _json(bundle_manifest_path, "bundle manifest")
    if manifest.get("status") != "LOCAL_CONSOLIDATED_BYTE_EXACT":
        raise Step0InputError("bundle manifest status is not frozen complete")

    request_records = _record_by_role(manifest, "request_manifest")
    population_records = _record_by_role(manifest, "population_manifest")
    if len(request_records) != 1:
        raise Step0InputError(
            f"expected one request manifest record, found {len(request_records)}"
        )
    if len(population_records) != 1:
        raise Step0InputError(
            f"expected one population manifest record, found {len(population_records)}"
        )
    if request_records[0].get("sha256") != expected_request_manifest_sha256:
        raise Step0InputError("request manifest content pin drift")
    request_manifest_path = _verified_file(
        by_hash, request_records[0], label="request manifest"
    )
    population_manifest_path = _verified_file(
        by_hash, population_records[0], label="population manifest"
    )
    if sha256(population_manifest_path) != expected_population_manifest_sha256:
        raise Step0InputError("population manifest SHA-256 drift")
    request_manifest = _json(request_manifest_path, "request manifest")
    request_count = request_manifest.get("request_count")
    request_file_count = request_manifest.get("request_file_count")
    requests = request_manifest.get("requests")
    if (
        not isinstance(request_count, int)
        or request_count <= 0
        or not isinstance(request_file_count, int)
        or request_file_count <= 0
        or not isinstance(requests, list)
        or len(requests) != request_file_count
    ):
        raise Step0InputError("request manifest is empty or has invalid counts")
    if (
        request_manifest.get("population_manifest_sha256")
        != expected_population_manifest_sha256
    ):
        raise Step0InputError("request manifest population pin drift")
    request_records_out: list[dict[str, object]] = []
    observed_actions = 0
    seen_semantics: set[tuple[str, str]] = set()
    seen_hashes: set[str] = set()
    for ordinal, request in enumerate(requests):
        if not isinstance(request, dict):
            raise Step0InputError("request manifest contains a malformed row")
        action_class = request.get("action_class")
        identity = request.get("identity")
        action_count = request.get("action_count")
        request_sha256 = request.get("sha256")
        if action_class not in {"delete", "occupied_swap"}:
            raise Step0InputError("request payload has invalid action class")
        if not isinstance(identity, str) or not identity.startswith("train:44b6_"):
            raise Step0InputError("request payload has invalid frozen identity")
        if not isinstance(action_count, int) or action_count <= 0:
            raise Step0InputError("request payload has nonpositive action count")
        if not isinstance(request_sha256, str) or request_sha256 in seen_hashes:
            raise Step0InputError("request payload hash is absent or duplicated")
        semantic_key = (action_class, identity)
        if semantic_key in seen_semantics:
            raise Step0InputError("request payload semantic identity is duplicated")
        seen_hashes.add(request_sha256)
        seen_semantics.add(semantic_key)
        path = _verified_file(
            by_hash, request, label=f"request payload {ordinal}"
        )
        request_records_out.append(
            {
                "ordinal": ordinal,
                "action_class": action_class,
                "identity": identity,
                "action_count": action_count,
                "bytes": request["bytes"],
                "sha256": request_sha256,
                "resolved_path": str(path),
            }
        )
        observed_actions += action_count
    if observed_actions != request_count:
        raise Step0InputError("request_count disagrees with request rows")

    wheel_records = _record_by_role(manifest, "offline_wheel")
    wheels: dict[str, str] = {}
    for filename, expected_sha256 in sorted(expected_wheels.items()):
        matches = [
            record
            for record in wheel_records
            if Path(str(record.get("path", ""))).name == filename
        ]
        if len(matches) != 1:
            raise Step0InputError(
                f"expected one offline wheel {filename}, found {len(matches)}"
            )
        wheel = _verified_file(
            by_hash, matches[0], label=f"offline wheel {filename}"
        )
        if sha256(wheel) != expected_sha256:
            raise Step0InputError(f"offline wheel SHA-256 drift: {filename}")
        wheels[filename] = str(wheel)

    safe_loader_records = _record_by_role(manifest, "safe_loader")
    if len(safe_loader_records) != 1:
        raise Step0InputError("expected exactly one CELLECT safe loader")
    safe_loader = _verified_file(
        by_hash, safe_loader_records[0], label="CELLECT safe loader"
    )
    checkpoint = _resolve_checkpoint(
        input_root, work_root, expected_checkpoint_sha256
    )
    elapsed = time.monotonic() - started
    result = {
        "schema": "celltrack-alpha.cellect-step0-preflight.v1",
        "status": "PASS_ALL_REQUIRED_INPUTS",
        "elapsed_seconds": elapsed,
        "bundle_manifest_path": str(bundle_manifest_path),
        "bundle_manifest_sha256": expected_bundle_manifest_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": expected_request_manifest_sha256,
        "request_count": request_count,
        "request_file_count": request_file_count,
        "request_records": request_records_out,
        "population_manifest_path": str(population_manifest_path),
        "population_manifest_sha256": expected_population_manifest_sha256,
        "wheels": wheels,
        "safe_loader_path": str(safe_loader),
        "heavy_work_started": False,
    }
    print(json.dumps({"step": 0, **result}, sort_keys=True), flush=True)
    return result

# END INLINE CONTENT-ADDRESSED STEP 0

BOOT_STARTED = time.monotonic()

WEIGHT_SHA = "f3e8e7303976dc5fb6e52ade12d1ff7953d2a28e30c06274b9f309562cd89ce5"
SAFE_WEIGHT_NAME = "cellect_uext_x3rd_149_0_4p6540.pth"
POPULATION_MANIFEST_SHA = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
BUNDLE_MANIFEST_SHA = (
    "ae3bc9b2bbf5e7b4e544bcb44159a2fc0d6250b3178a81192f2d8e9f867fecd3"
)
REQUEST_MANIFEST_SHA = (
    "82111af54e92a4118efbaea4ed0e5aef770f08a252df8e9f868ecea03b81bc83"
)
CELLECT_INPUT_ROOT = Path(
    "/kaggle/input/datasets/onebitbrain/g02-cellect-r7-published-assets-v1"
)
WHEEL_SHA256 = {
    "donfig-0.8.1.post1-py3-none-any.whl": (
        "2a3175ce74a06109ff9307d90a230f81215cbac9a751f4d1c6194644b8204f9d"
    ),
    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        "c3a09e22140f2c691f7df26303ff8fa2dadcf26d7d0828398c0bc09b69e5efa3"
    ),
    "zarr-3.2.1-py3-none-any.whl": (
        "f78cdd3d9687ad0e9f9cba2c5683b64f0c52589c19f685eeabe872e93cc0d2c7"
    ),
}


class FeatureError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_tar_member(name: str) -> bool:
    member = Path(name)
    return (
        bool(name)
        and name.isascii()
        and not member.is_absolute()
        and ".." not in member.parts
        and member.name == SAFE_WEIGHT_NAME
    )


def resolve_weight_by_content(
    input_root: Path,
    work_root: Path,
    expected_sha256: str = WEIGHT_SHA,
) -> Path:
    """Resolve the one exact checkpoint from direct files or uncompressed tar members."""
    started = time.monotonic()
    observations: list[dict[str, object]] = []
    matches: list[tuple[str, Path]] = []

    for suffix in ("*.pth", "*.bin"):
        for candidate in sorted(input_root.rglob(suffix)):
            if not candidate.is_file():
                continue
            digest = sha256(candidate)
            observations.append(
                {
                    "kind": "file",
                    "path": str(candidate),
                    "bytes": candidate.stat().st_size,
                    "sha256": digest,
                }
            )
            if digest == expected_sha256:
                matches.append((f"file:{candidate}", candidate))

    extraction_root = work_root / "cellect_checkpoint_step0"
    extraction_root.mkdir(parents=True, exist_ok=True)
    for archive_ordinal, archive in enumerate(sorted(input_root.rglob("*.tar"))):
        if not archive.is_file():
            continue
        try:
            handle = tarfile.open(archive, mode="r:")
        except tarfile.TarError as error:
            raise FeatureError(f"invalid uncompressed tar {archive}: {error}") from error
        with handle:
            for member_ordinal, member in enumerate(handle):
                if not member.isfile():
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise FeatureError(f"cannot read tar member {archive}:{member.name}")
                temporary = extraction_root / (
                    f"candidate_{archive_ordinal:04d}_{member_ordinal:04d}.partial"
                )
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as destination:
                    for block in iter(lambda: extracted.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        destination.write(block)
                observed_sha256 = digest.hexdigest()
                observations.append(
                    {
                        "kind": "tar_member",
                        "path": str(archive),
                        "member": member.name,
                        "bytes": size,
                        "sha256": observed_sha256,
                    }
                )
                if observed_sha256 != expected_sha256:
                    temporary.unlink()
                    continue
                if not _safe_tar_member(member.name):
                    temporary.unlink()
                    raise FeatureError(
                        f"matching checkpoint uses unsafe or unexpected tar member name: {member.name!r}"
                    )
                resolved = extraction_root / SAFE_WEIGHT_NAME
                if resolved.exists():
                    resolved = extraction_root / (
                        f"{archive_ordinal:04d}_{member_ordinal:04d}_{SAFE_WEIGHT_NAME}"
                    )
                temporary.replace(resolved)
                matches.append((f"tar:{archive}:{member.name}", resolved))

    if len(matches) != 1:
        summary = json.dumps(
            {
                "expected_sha256": expected_sha256,
                "match_count": len(matches),
                "matches": [source for source, _path in matches],
                "observations": observations,
            },
            sort_keys=True,
        )
        for _source, path in matches:
            if extraction_root in path.parents and path.exists():
                path.unlink()
        raise FeatureError(f"checkpoint content resolution failed: {summary}")

    source, resolved = matches[0]
    if source.startswith("file:"):
        destination = extraction_root / SAFE_WEIGHT_NAME
        shutil.copyfile(resolved, destination)
        resolved = destination
    if sha256(resolved) != expected_sha256:
        raise FeatureError("checkpoint changed during step-0 resolution")
    print(
        json.dumps(
            {
                "step": 0,
                "checkpoint_source": source,
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": expected_sha256,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return resolved


# Step 0 deliberately authenticates every required input before NumPy, torch,
# zarr, dependency installation, model loading, or inference.
STEP0 = (
    run_cellect_step0(
        CELLECT_INPUT_ROOT,
        Path("/kaggle/working"),
        expected_bundle_manifest_sha256=BUNDLE_MANIFEST_SHA,
        expected_request_manifest_sha256=REQUEST_MANIFEST_SHA,
        expected_checkpoint_sha256=WEIGHT_SHA,
        expected_population_manifest_sha256=POPULATION_MANIFEST_SHA,
        expected_wheels=WHEEL_SHA256,
    )
    if __name__ == "__main__"
    else None
)
RESOLVED_WEIGHT = Path(str(STEP0["checkpoint_path"])) if STEP0 is not None else None

import numpy as np
import torch

if importlib.util.find_spec("zarr") is None:
    if STEP0 is None:
        raise RuntimeError("step-0 input preflight did not run")
    wheels = STEP0["wheels"]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheels["donfig-0.8.1.post1-py3-none-any.whl"]),
            str(
                wheels[
                    "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                ]
            ),
            str(wheels["zarr-3.2.1-py3-none-any.whl"]),
        ]
    )

import zarr

AMENDMENT_SHA = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
SUPPLIER_SHA = "8d34445eafa19dcd135b59747c6ccb4314b0e2ca19f28e80fcfa752bab9d3664"
MODEL_SOURCE_SHA = "2394797f4cb680cd45136c83792df67a3ee8915b6925994a6135d4425998b0c0"
ASSET_DATASET_VERSION = 4
TILE_ZYX = (16, 64, 64)
BATCH = 8
HARD_KILL_SECONDS = 5 * 3600
SOFT_STOP_SECONDS = HARD_KILL_SECONDS - 300
OUTPUT = Path("/kaggle/working/cellect_action_features")


def locate_one(pattern: str) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(pattern))
    if len(matches) != 1:
        raise FeatureError(f"expected one {pattern}, found {len(matches)}")
    return matches[0]


def action_features(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 64:
        raise FeatureError("embedding pair shape drift")
    difference = source - target
    src_norm = np.linalg.norm(source, axis=1)
    tgt_norm = np.linalg.norm(target, axis=1)
    denominator = np.maximum(src_norm * tgt_norm, 1e-8)
    centered_source = source - source.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    corr_denominator = np.maximum(
        np.linalg.norm(centered_source, axis=1)
        * np.linalg.norm(centered_target, axis=1),
        1e-8,
    )
    return np.stack(
        (
            np.sum(source * target, axis=1) / denominator,
            np.linalg.norm(difference, axis=1) / 8.0,
            np.mean(np.abs(difference), axis=1),
            np.mean(source * target, axis=1),
            src_norm / 8.0,
            tgt_norm / 8.0,
            (src_norm - tgt_norm) / 8.0,
            np.sum(centered_source * centered_target, axis=1) / corr_denominator,
        ),
        axis=1,
    ).astype(np.float32)


def permutation(values: np.ndarray, node_ids: np.ndarray, identity: str, frame: int) -> np.ndarray:
    unique_ids, first = np.unique(node_ids, return_index=True)
    vectors = values[first]
    seed = f"R-SEPARATION-FALSIFIER-44B6{SUPPLIER_SHA}{identity}{frame}"
    order = sorted(
        range(len(unique_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}{int(unique_ids[index])}".encode("ascii")).digest(),
            int(unique_ids[index]),
        ),
    )
    replacement = {}
    if order:
        for position, index in enumerate(order):
            replacement[int(unique_ids[index])] = vectors[order[(position + 1) % len(order)]]
    return np.stack([replacement[int(node)] for node in node_ids]).astype(np.float32)


def preprocess(volume: torch.Tensor) -> torch.Tensor:
    positive = volume > 0
    if not bool(positive.any().item()):
        raise FeatureError("tile contains no positive intensity")
    minimum = volume[positive].amin()
    return torch.log1p(torch.where(volume < minimum, minimum, volume) + 1900.0)


def count_tile_jobs(request: dict[str, np.ndarray]) -> int:
    source_keys = np.column_stack(
        (
            request["source_t"],
            request["source_zyx"][:, 0] // TILE_ZYX[0],
            request["source_zyx"][:, 1] // TILE_ZYX[1],
            request["source_zyx"][:, 2] // TILE_ZYX[2],
        )
    )
    target_keys = np.column_stack(
        (
            request["target_t"] - 1,
            request["target_zyx"][:, 0] // TILE_ZYX[0],
            request["target_zyx"][:, 1] // TILE_ZYX[1],
            request["target_zyx"][:, 2] // TILE_ZYX[2],
        )
    )
    return int(len(np.unique(np.concatenate((source_keys, target_keys)), axis=0)))


def group_endpoint_indices(
    frames: np.ndarray, coordinates: np.ndarray, *, target_role: bool
) -> dict[tuple[int, int, int, int], np.ndarray]:
    keys = np.column_stack(
        (
            frames - int(target_role),
            coordinates[:, 0] // TILE_ZYX[0],
            coordinates[:, 1] // TILE_ZYX[1],
            coordinates[:, 2] // TILE_ZYX[2],
        )
    )
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return {
        tuple(int(value) for value in key): order[offsets[index] : offsets[index + 1]]
        for index, key in enumerate(unique)
    }


def infer_video(
    model: torch.nn.Module,
    array: zarr.Array,
    request: dict[str, np.ndarray],
    started: float,
    completed_before: int,
    total_planned_jobs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = len(request["source_id"])
    source_embedding = np.empty((count, 64), dtype=np.float32)
    target_embedding = np.empty((count, 64), dtype=np.float32)
    source_written = np.zeros(count, dtype=bool)
    target_written = np.zeros(count, dtype=bool)
    source_groups = group_endpoint_indices(
        request["source_t"], request["source_zyx"], target_role=False
    )
    target_groups = group_endpoint_indices(
        request["target_t"], request["target_zyx"], target_role=True
    )
    empty = np.empty(0, dtype=np.int64)
    tile_jobs = [
        (
            key[0],
            key[1] * TILE_ZYX[0],
            key[2] * TILE_ZYX[1],
            key[3] * TILE_ZYX[2],
            source_groups.get(key, empty),
            target_groups.get(key, empty),
        )
        for key in sorted(source_groups.keys() | target_groups.keys())
    ]
    if len(tile_jobs) != count_tile_jobs(request):
        raise FeatureError("tile grouping drift")
    completed = 0
    for offset in range(0, len(tile_jobs), BATCH):
        if time.monotonic() - started >= SOFT_STOP_SECONDS:
            raise FeatureError("five-hour hard-kill guard approached")
        jobs = tile_jobs[offset : offset + BATCH]
        volumes = []
        for frame, z0, y0, x0, _src, _tgt in jobs:
            forward = np.asarray(
                array[frame : frame + 2, z0 : z0 + 16, y0 : y0 + 64, x0 : x0 + 64],
                dtype=np.float32,
            )
            volumes.extend((forward, forward[::-1].copy()))
        tensor = torch.from_numpy(np.stack(volumes)).permute(0, 1, 3, 4, 2).to("cuda")
        with torch.inference_mode():
            embedding = model(preprocess(tensor)).embedding.float().cpu().numpy()
        for job_index, (_frame, z0, y0, x0, src, tgt) in enumerate(jobs):
            forward = embedding[2 * job_index]
            reverse = embedding[2 * job_index + 1]
            if len(src):
                point = request["source_zyx"][src] - np.asarray((z0, y0, x0))
                source_embedding[src] = forward[:, point[:, 1], point[:, 2], point[:, 0]].T
                source_written[src] = True
            if len(tgt):
                point = request["target_zyx"][tgt] - np.asarray((z0, y0, x0))
                target_embedding[tgt] = reverse[:, point[:, 1], point[:, 2], point[:, 0]].T
                target_written[tgt] = True
        completed += len(jobs)
        global_completed = completed_before + completed
        if global_completed >= min(64, total_planned_jobs):
            elapsed = time.monotonic() - started
            projection = elapsed / global_completed * total_planned_jobs
            if projection >= HARD_KILL_SECONDS:
                raise FeatureError(f"authenticated projection exceeds five hours: {projection / 3600:.3f}")
    if not source_written.all() or not target_written.all():
        raise FeatureError("endpoint feature coverage is incomplete")
    return source_embedding, target_embedding, len(tile_jobs)


def main() -> int:
    started = BOOT_STARTED
    if RESOLVED_WEIGHT is None or STEP0 is None:
        raise FeatureError("step-0 all-input preflight did not run")
    if not torch.cuda.is_available():
        raise FeatureError("CUDA is required for the bounded feature sample")
    model_source = Path(str(STEP0["safe_loader_path"]))
    weight = RESOLVED_WEIGHT
    if sha256(weight) != WEIGHT_SHA:
        raise FeatureError("CELLECT weight hash mismatch")
    if sha256(model_source) != MODEL_SOURCE_SHA:
        raise FeatureError("CELLECT safe-loader hash mismatch")
    specification = importlib.util.spec_from_file_location(
        "cellect_authenticated_model", model_source
    )
    if specification is None or specification.loader is None:
        raise FeatureError("cannot create CELLECT safe-loader module specification")
    model_module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = model_module
    specification.loader.exec_module(model_module)
    load_published_backbone = model_module.load_published_backbone

    model = load_published_backbone(weight, device="cuda").eval()
    request_records = STEP0["request_records"]
    if not request_records:
        raise FeatureError("step-0 returned no frozen feature requests")
    planned_jobs = 0
    for record in request_records:
        request_path = Path(str(record["resolved_path"]))
        with np.load(request_path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        planned_jobs += count_tile_jobs(request)
    if planned_jobs < 64:
        raise FeatureError("insufficient tile jobs for the frozen runtime projection")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = []
    total_jobs = 0
    for ordinal, record in enumerate(request_records, start=1):
        request_path = Path(str(record["resolved_path"]))
        action_class = str(record["action_class"])
        identity = str(record["identity"])
        dataset = identity.split(":", 1)[1]
        raw = locate_one(f"{dataset}.zarr")
        array = zarr.open_array(str(raw / "0"), mode="r")
        with np.load(request_path, allow_pickle=False) as loaded:
            request = {key: loaded[key] for key in loaded.files}
        source, target, jobs = infer_video(
            model,
            array,
            request,
            started,
            total_jobs,
            planned_jobs,
        )
        a1 = action_features(source, target)
        a0_source = np.empty_like(source)
        a0_target = np.empty_like(target)
        for frame in np.unique(request["source_t"]):
            index = np.flatnonzero(request["source_t"] == frame)
            a0_source[index] = permutation(source[index], request["source_id"][index], identity, int(frame))
        for frame in np.unique(request["target_t"]):
            index = np.flatnonzero(request["target_t"] == frame)
            a0_target[index] = permutation(target[index], request["target_id"][index], identity, int(frame))
        a0 = action_features(a0_source, a0_target)
        destination = OUTPUT / action_class / (
            identity.replace(":", "__", 1) + ".npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, a1=a1, a0=a0)
        files.append(
            {
                "identity": identity,
                "action_class": action_class,
                "count": len(a1),
                "request_sha256": sha256(request_path),
                "output": str(destination.relative_to(OUTPUT)),
                "output_sha256": sha256(destination),
                "tile_jobs": jobs,
            }
        )
        total_jobs += jobs
        print(f"{ordinal}/{len(request_records)} {action_class} {identity} actions={len(a1)} jobs={jobs}", flush=True)
    if total_jobs != planned_jobs:
        raise FeatureError(f"tile-job plan drift: {total_jobs} != {planned_jobs}")
    manifest = {
        "schema": "celltrack-alpha.cellect-action-features.v1",
        "status": "COMPLETE",
        "amendment_sha256": AMENDMENT_SHA,
        "supplier_sha256": SUPPLIER_SHA,
        "weight_sha256": WEIGHT_SHA,
        "safe_loader_sha256": MODEL_SOURCE_SHA,
        "asset_dataset_version": ASSET_DATASET_VERSION,
        "device": torch.cuda.get_device_name(0),
        "tile_zyx": list(TILE_ZYX),
        "batch": BATCH,
        "action_feature_names": [
            "cosine",
            "euclidean_over_sqrt64",
            "mean_absolute_difference",
            "mean_product",
            "source_norm_over_sqrt64",
            "target_norm_over_sqrt64",
            "norm_difference_over_sqrt64",
            "centered_correlation",
        ],
        "files": files,
        "total_tile_jobs": total_jobs,
        "planned_tile_jobs": planned_jobs,
        "elapsed_seconds": time.monotonic() - started,
        "full_pool_cache": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files), "jobs": total_jobs, "seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
