#!/usr/bin/env python3
"""Pure-stdlib, fail-closed Kaggle input preflight used before heavy imports."""

from __future__ import annotations

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
