#!/usr/bin/env python3
"""Fail-closed artifacts for the G-03 cross-session resume proof.

This module deliberately has no torch dependency.  The Kaggle proof kernels use it
for the durable boundary around their torch checkpoint, while local tests exercise
the boundary without pretending that a CPU-only workstation proves CUDA resume.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


SCHEMA = "celltrack-alpha.g03-checkpoint-handoff.v1"
READY_FILE = "READY.json"
INVENTORY_FILE = "INVENTORY.json"
MANIFEST_FILE = "MANIFEST.json"


class ResumeProofError(RuntimeError):
    """An integrity or idempotence invariant failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only JSON representation used in an integrity comparison."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResumeProofError(f"unsafe artifact path: {value!r}")
    return path


def _write_bytes_fsync(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeProofError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ResumeProofError(f"{label} must be a JSON object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    required = {
        "schema",
        "handoff_id",
        "producer_kernel",
        "code_sha256",
        "config_sha256",
        "state",
    }
    missing = sorted(required - set(value))
    if missing or value.get("schema") != SCHEMA:
        raise ResumeProofError(f"handoff manifest drifted; missing={missing}")
    if not all(isinstance(value[key], str) and value[key] for key in required - {"state"}):
        raise ResumeProofError("handoff manifest has malformed identity fields")
    if not isinstance(value["state"], Mapping):
        raise ResumeProofError("handoff manifest lacks state object")
    return value


def _inventory_from_tree(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {INVENTORY_FILE, READY_FILE}:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _expected_tree_digest(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(records))


def verify_handoff(root: Path) -> dict[str, Any]:
    """Verify a completely published handoff and return its normalized receipt."""
    if not root.is_dir():
        raise ResumeProofError(f"handoff directory is absent: {root}")
    ready = _read_json(root / READY_FILE, "handoff ready marker")
    inventory = _read_json(root / INVENTORY_FILE, "handoff inventory")
    manifest = _validate_manifest(_read_json(root / MANIFEST_FILE, "handoff manifest"))
    records = inventory.get("files")
    if (
        inventory.get("schema") != SCHEMA
        or not isinstance(records, list)
        or not isinstance(inventory.get("tree_sha256"), str)
        or ready.get("schema") != SCHEMA
        or ready.get("handoff_id") != manifest["handoff_id"]
        or ready.get("tree_sha256") != inventory["tree_sha256"]
    ):
        raise ResumeProofError("handoff publication markers drifted")
    normalized: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ResumeProofError("handoff inventory record is malformed")
        try:
            relative = _safe_relative_path(str(raw["path"]))
            expected_bytes = int(raw["bytes"])
            expected_sha = str(raw["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResumeProofError("handoff inventory record is malformed") from exc
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ResumeProofError(f"handoff payload size mismatch: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ResumeProofError(f"handoff payload hash mismatch: {relative}")
        normalized.append(
            {"path": relative.as_posix(), "bytes": expected_bytes, "sha256": actual_sha}
        )
    if normalized != records or _expected_tree_digest(normalized) != inventory["tree_sha256"]:
        raise ResumeProofError("handoff inventory digest drifted")
    actual = _inventory_from_tree(root)
    if actual != records:
        raise ResumeProofError("handoff contains undeclared, missing, or changed files")
    return {
        "schema": SCHEMA,
        "handoff_id": manifest["handoff_id"],
        "tree_sha256": inventory["tree_sha256"],
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "files": normalized,
        "manifest": manifest,
    }


def publish_handoff(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Atomically publish an immutable handoff or confirm the identical prior one.

    An incomplete prior directory is never repaired in place.  A completed directory
    is accepted only when it is byte-for-byte the object this invocation would write.
    """
    checked_manifest = _validate_manifest(manifest)
    if MANIFEST_FILE in payloads or INVENTORY_FILE in payloads or READY_FILE in payloads:
        raise ResumeProofError("payload names collide with protocol files")
    expected_payloads = {
        _safe_relative_path(name).as_posix(): bytes(value)
        for name, value in payloads.items()
    }
    if not expected_payloads:
        raise ResumeProofError("handoff has no payloads")
    if root.exists():
        receipt = verify_handoff(root)
        existing_manifest = _read_json(root / MANIFEST_FILE, "existing handoff manifest")
        if canonical_json_bytes(existing_manifest) != canonical_json_bytes(checked_manifest):
            raise ResumeProofError("existing handoff manifest conflicts with restart")
        existing_payloads = {
            record["path"]: (root / record["path"]).read_bytes()
            for record in receipt["files"]
            if record["path"] != MANIFEST_FILE
        }
        if existing_payloads != expected_payloads:
            raise ResumeProofError("existing handoff payload conflicts with restart")
        return receipt

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.partial-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o755)
        for name, value in expected_payloads.items():
            _write_bytes_fsync(staging / name, value)
        _write_bytes_fsync(staging / MANIFEST_FILE, canonical_json_bytes(checked_manifest))
        records = _inventory_from_tree(staging)
        inventory = {
            "schema": SCHEMA,
            "handoff_id": checked_manifest["handoff_id"],
            "files": records,
            "tree_sha256": _expected_tree_digest(records),
        }
        _write_bytes_fsync(staging / INVENTORY_FILE, canonical_json_bytes(inventory))
        _fsync_directory(staging)
        _write_bytes_fsync(
            staging / READY_FILE,
            canonical_json_bytes(
                {
                    "schema": SCHEMA,
                    "handoff_id": checked_manifest["handoff_id"],
                    "tree_sha256": inventory["tree_sha256"],
                }
            ),
        )
        _fsync_directory(staging)
        os.replace(staging, root)
        _fsync_directory(root.parent)
    except Exception:
        # Leaving a distinct partial directory is intentional forensic evidence; it
        # cannot be mistaken for a READY handoff because the final path was untouched.
        raise
    return verify_handoff(root)


def write_idempotent_json(path: Path, value: Mapping[str, Any]) -> tuple[bool, str]:
    """Write a result once; later equal calls pass and conflicting calls fail."""
    encoded = canonical_json_bytes(dict(value))
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise ResumeProofError(f"idempotent result conflicts: {path}")
        return False, sha256_bytes(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.tmp-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            # link(2) is an atomic create-if-absent operation on this filesystem;
            # unlike replace it cannot silently overwrite a concurrent result.
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != encoded:
                raise ResumeProofError(f"idempotent result conflicts: {path}")
            return False, sha256_bytes(encoded)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return True, sha256_bytes(encoded)


__all__ = [
    "INVENTORY_FILE",
    "MANIFEST_FILE",
    "READY_FILE",
    "ResumeProofError",
    "SCHEMA",
    "canonical_json_bytes",
    "publish_handoff",
    "sha256_bytes",
    "sha256_file",
    "verify_handoff",
    "write_idempotent_json",
]
