#!/usr/bin/env python3
"""Pure-stdlib prototype for content-addressed CELLECT request discovery.

This module deliberately does not import NumPy, Torch, zarr, pip, or CELLECT
model code. It proves the transport/discovery contract only. The orchestrator
must port the verified contract into the shared kernel before N3 can open.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ACTION_CLASSES = frozenset({"delete", "occupied_swap"})


class DiscoveryError(RuntimeError):
    """Raised when content-addressed discovery cannot prove a unique input."""


@dataclass(frozen=True)
class DiscoveryPins:
    bundle_manifest_sha256: str
    request_manifest_sha256: str
    population_manifest_sha256: str
    request_file_count: int
    request_count: int


@dataclass(frozen=True)
class RequestRecord:
    ordinal: int
    action_class: str
    identity: str
    action_count: int
    bytes: int
    sha256: str
    resolved_path: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiscoveryError(f"{label} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise DiscoveryError(f"{label} is not a JSON object")
    return value


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise DiscoveryError(f"{label} is not a lowercase SHA-256")
    return value


def _inventory(root: Path) -> tuple[dict[str, list[Path]], int]:
    if not root.is_dir():
        raise DiscoveryError(f"input root is not a directory: {root}")
    by_hash: dict[str, list[Path]] = defaultdict(list)
    file_count = 0
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        by_hash[sha256(candidate)].append(candidate)
        file_count += 1
    return dict(by_hash), file_count


def _unique_content(
    by_hash: dict[str, list[Path]],
    expected_sha256: str,
    *,
    label: str,
    expected_bytes: int | None = None,
) -> Path:
    _validate_sha256(expected_sha256, f"{label} expected hash")
    matches = by_hash.get(expected_sha256, [])
    if len(matches) != 1:
        raise DiscoveryError(
            f"{label} content resolution expected one match, found {len(matches)}"
        )
    path = matches[0]
    if expected_bytes is not None:
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise DiscoveryError(f"{label} has invalid expected byte count")
        if path.stat().st_size != expected_bytes:
            raise DiscoveryError(f"{label} byte count drift")
    return path


def _records_by_role(
    manifest: dict[str, Any], role: str
) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DiscoveryError("bundle manifest has no file list")
    return [
        record
        for record in files
        if isinstance(record, dict) and record.get("role") == role
    ]


def discover_requests_by_content(
    input_root: Path,
    pins: DiscoveryPins,
) -> dict[str, Any]:
    """Resolve the frozen request manifest and payloads by bytes and SHA only."""
    if pins.request_file_count <= 0 or pins.request_count <= 0:
        raise DiscoveryError("frozen request pins must be positive")

    by_hash, inventory_file_count = _inventory(input_root)
    bundle_path = _unique_content(
        by_hash,
        pins.bundle_manifest_sha256,
        label="bundle manifest",
    )
    bundle = _json_object(bundle_path, "bundle manifest")
    if bundle.get("status") != "LOCAL_CONSOLIDATED_BYTE_EXACT":
        raise DiscoveryError("bundle manifest is not frozen complete")

    request_manifest_records = _records_by_role(bundle, "request_manifest")
    if len(request_manifest_records) != 1:
        raise DiscoveryError(
            "bundle manifest must contain exactly one request-manifest record"
        )
    request_manifest_record = request_manifest_records[0]
    record_manifest_hash = _validate_sha256(
        request_manifest_record.get("sha256"), "request-manifest record hash"
    )
    if record_manifest_hash != pins.request_manifest_sha256:
        raise DiscoveryError("request-manifest pin disagrees with bundle manifest")
    request_manifest_path = _unique_content(
        by_hash,
        pins.request_manifest_sha256,
        label="request manifest",
        expected_bytes=request_manifest_record.get("bytes"),
    )
    request_manifest = _json_object(request_manifest_path, "request manifest")
    if request_manifest.get("status") != "FROZEN_COMPLETE":
        raise DiscoveryError("request manifest is not frozen complete")

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
        raise DiscoveryError("request manifest is empty or has invalid counts")
    if request_count != pins.request_count:
        raise DiscoveryError("request count disagrees with frozen pin")
    if request_file_count != pins.request_file_count:
        raise DiscoveryError("request-file count disagrees with frozen pin")
    if (
        request_manifest.get("population_manifest_sha256")
        != pins.population_manifest_sha256
    ):
        raise DiscoveryError("request manifest population pin drift")

    bundle_payload_records = _records_by_role(bundle, "feature_request")
    if len(bundle_payload_records) != pins.request_file_count:
        raise DiscoveryError("bundle feature-request coverage is incomplete")
    bundle_payload_by_hash: dict[str, dict[str, Any]] = {}
    for record in bundle_payload_records:
        digest = _validate_sha256(record.get("sha256"), "bundle payload hash")
        if digest in bundle_payload_by_hash:
            raise DiscoveryError("bundle manifest contains duplicate request hashes")
        bundle_payload_by_hash[digest] = record

    resolved: list[RequestRecord] = []
    seen_hashes: set[str] = set()
    seen_semantic_keys: set[tuple[str, str]] = set()
    observed_actions = 0
    for ordinal, request in enumerate(requests):
        if not isinstance(request, dict):
            raise DiscoveryError("request manifest contains a malformed row")
        action_class = request.get("action_class")
        identity = request.get("identity")
        action_count = request.get("action_count")
        expected_bytes = request.get("bytes")
        digest = _validate_sha256(request.get("sha256"), "request payload hash")
        if action_class not in ALLOWED_ACTION_CLASSES:
            raise DiscoveryError(f"request row has invalid action class: {action_class}")
        if not isinstance(identity, str) or not identity.startswith("train:44b6_"):
            raise DiscoveryError("request row has invalid frozen identity")
        if not isinstance(action_count, int) or action_count <= 0:
            raise DiscoveryError("request payload has nonpositive action count")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise DiscoveryError("request payload has invalid byte count")
        if digest in seen_hashes:
            raise DiscoveryError("request manifest contains duplicate payload hashes")
        semantic_key = (action_class, identity)
        if semantic_key in seen_semantic_keys:
            raise DiscoveryError("request manifest contains duplicate semantic rows")
        seen_hashes.add(digest)
        seen_semantic_keys.add(semantic_key)

        bundle_record = bundle_payload_by_hash.get(digest)
        if bundle_record is None:
            raise DiscoveryError("request payload absent from bundle file records")
        for field, value in (
            ("action_class", action_class),
            ("identity", identity),
            ("action_count", action_count),
            ("bytes", expected_bytes),
        ):
            if bundle_record.get(field) != value:
                raise DiscoveryError(f"bundle/request semantic drift: {field}")
        path = _unique_content(
            by_hash,
            digest,
            label=f"request payload {ordinal}",
            expected_bytes=expected_bytes,
        )
        resolved.append(
            RequestRecord(
                ordinal=ordinal,
                action_class=action_class,
                identity=identity,
                action_count=action_count,
                bytes=expected_bytes,
                sha256=digest,
                resolved_path=str(path),
            )
        )
        observed_actions += action_count

    if set(bundle_payload_by_hash) != seen_hashes:
        raise DiscoveryError("bundle/request manifest payload sets disagree")
    if observed_actions != pins.request_count:
        raise DiscoveryError("request action total disagrees with frozen pin")

    class_file_counts = Counter(record.action_class for record in resolved)
    class_action_counts: Counter[str] = Counter()
    for record in resolved:
        class_action_counts[record.action_class] += record.action_count
    return {
        "schema": "celltrack-alpha.cellect-content-addressed-step0.v1",
        "status": "PASS_UNIQUE_CONTENT_DISCOVERY",
        "input_root": str(input_root),
        "inventory_file_count": inventory_file_count,
        "bundle_manifest": {
            "resolved_path": str(bundle_path),
            "sha256": pins.bundle_manifest_sha256,
        },
        "request_manifest": {
            "resolved_path": str(request_manifest_path),
            "sha256": pins.request_manifest_sha256,
        },
        "population_manifest_sha256": pins.population_manifest_sha256,
        "request_file_count": len(resolved),
        "request_count": observed_actions,
        "class_file_counts": dict(sorted(class_file_counts.items())),
        "class_action_counts": dict(sorted(class_action_counts.items())),
        "request_records": [asdict(record) for record in resolved],
        "heavy_work_started": False,
    }
