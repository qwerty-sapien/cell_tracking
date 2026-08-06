#!/usr/bin/env python3
"""Durable, fail-closed infrastructure for the Colab division-detector lane.

The scientific objective is deliberately external to this module.  This file
ports the proven G-03 Phase-1 durability boundary: complete optimizer and RNG
state, immutable hash inventories, atomic publication, exact resume checks,
resource logging, and a provider-specific hours ledger.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import sys
import tempfile
import time
from typing import Any
import uuid


ARTIFACT_SCHEMA = "celltrack-alpha.colab-atomic-artifact.v1"
CHECKPOINT_SCHEMA = "celltrack-alpha.colab-division-checkpoint.v1"
LEDGER_SCHEMA = "celltrack-alpha.colab-hours-ledger.v2"
LOG_SCHEMA = "celltrack-alpha.full-stage-log.v1"
OBJECTIVE_STATUS = "FROZEN_AFTER_APNCC_V32_TAXONOMY"
DEPLOYMENT_GATE_SHA256 = "a1c4b2f97aa5d28f88f5d2d4e35a2d9f35db6b09934920ecb9e982521c74b634"
OWNER_SCOPE_SHA256 = "ac4270f896168b47945bf661d8aed0fb833510d3ec2b37159e7f19e1828381a6"
READY_FILE = "READY.json"
INVENTORY_FILE = "INVENTORY.json"
MANIFEST_FILE = "MANIFEST.json"
LATEST_FILE = "LATEST.json"


class HarnessError(RuntimeError):
    """A durability, provenance, or governance check failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
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


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(fd)


def _write_fsync(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_replace_bytes(path: Path, value: bytes) -> str:
    """Atomically replace one mutable pointer or log-side receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.tmp-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(value)


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> str:
    return atomic_replace_bytes(path, canonical_json_bytes(dict(value)))


def write_idempotent_json(path: Path, value: Mapping[str, Any]) -> tuple[bool, str]:
    """Create once without hard links; identical retries pass and conflicts stop."""
    encoded = canonical_json_bytes(dict(value))
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise HarnessError(f"idempotent JSON conflicts: {path}")
        return False, sha256_bytes(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != encoded:
            raise HarnessError(f"idempotent JSON conflicts: {path}")
        return False, sha256_bytes(encoded)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return True, sha256_bytes(encoded)


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HarnessError(f"unsafe artifact path: {value!r}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
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


def _tree_digest(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(records))


def verify_artifact(root: Path) -> dict[str, Any]:
    """Verify an immutable artifact, including absence of undeclared files."""
    if not root.is_dir():
        raise HarnessError(f"artifact directory is absent: {root}")
    ready = _read_json(root / READY_FILE, "artifact ready marker")
    inventory = _read_json(root / INVENTORY_FILE, "artifact inventory")
    manifest = _read_json(root / MANIFEST_FILE, "artifact manifest")
    records = inventory.get("files")
    if (
        ready.get("schema") != ARTIFACT_SCHEMA
        or inventory.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("artifact_schema") != ARTIFACT_SCHEMA
        or not isinstance(records, list)
        or ready.get("artifact_id") != manifest.get("artifact_id")
        or inventory.get("artifact_id") != manifest.get("artifact_id")
        or ready.get("tree_sha256") != inventory.get("tree_sha256")
    ):
        raise HarnessError("artifact publication markers drifted")
    normalized: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise HarnessError("artifact inventory record is malformed")
        relative = _safe_relative_path(str(raw.get("path", "")))
        path = root / relative
        try:
            expected_bytes = int(raw["bytes"])
            expected_sha = str(raw["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessError("artifact inventory record is malformed") from exc
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise HarnessError(f"artifact payload size mismatch: {relative}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise HarnessError(f"artifact payload hash mismatch: {relative}")
        normalized.append(
            {"path": relative.as_posix(), "bytes": expected_bytes, "sha256": observed_sha}
        )
    if normalized != records or _tree_digest(normalized) != inventory.get("tree_sha256"):
        raise HarnessError("artifact inventory digest drifted")
    if _tree_inventory(root) != records:
        raise HarnessError("artifact has undeclared, missing, or changed files")
    return {
        "artifact_id": manifest["artifact_id"],
        "tree_sha256": inventory["tree_sha256"],
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "files": records,
        "manifest": manifest,
    }


def publish_artifact(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Atomically publish an immutable directory or verify an identical retry."""
    checked = dict(manifest)
    artifact_id = checked.get("artifact_id")
    if (
        checked.get("artifact_schema") != ARTIFACT_SCHEMA
        or not isinstance(artifact_id, str)
        or not artifact_id
    ):
        raise HarnessError("artifact manifest identity is invalid")
    forbidden = {MANIFEST_FILE, INVENTORY_FILE, READY_FILE}
    if forbidden.intersection(payloads):
        raise HarnessError("payload collides with an artifact protocol file")
    expected = {
        _safe_relative_path(name).as_posix(): bytes(value)
        for name, value in payloads.items()
    }
    if not expected:
        raise HarnessError("artifact has no payloads")
    if root.exists():
        receipt = verify_artifact(root)
        if canonical_json_bytes(receipt["manifest"]) != canonical_json_bytes(checked):
            raise HarnessError("existing artifact manifest conflicts with retry")
        observed = {
            row["path"]: (root / row["path"]).read_bytes()
            for row in receipt["files"]
            if row["path"] != MANIFEST_FILE
        }
        if observed != expected:
            raise HarnessError("existing artifact payload conflicts with retry")
        return receipt

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    for name, value in expected.items():
        _write_fsync(staging / name, value)
    _write_fsync(staging / MANIFEST_FILE, canonical_json_bytes(checked))
    records = _tree_inventory(staging)
    inventory = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": artifact_id,
        "files": records,
        "tree_sha256": _tree_digest(records),
    }
    _write_fsync(staging / INVENTORY_FILE, canonical_json_bytes(inventory))
    _fsync_directory(staging)
    _write_fsync(
        staging / READY_FILE,
        canonical_json_bytes(
            {
                "schema": ARTIFACT_SCHEMA,
                "artifact_id": artifact_id,
                "tree_sha256": inventory["tree_sha256"],
            }
        ),
    )
    _fsync_directory(staging)
    os.replace(staging, root)
    _fsync_directory(root.parent)
    return verify_artifact(root)


def rss_bytes() -> int:
    """Return peak RSS in bytes on Linux/Colab and macOS."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def linux_swapout_bytes() -> int | None:
    try:
        rows = Path("/proc/vmstat").read_text(encoding="ascii").splitlines()
        pages = next(int(row.split()[1]) for row in rows if row.startswith("pswpout "))
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, StopIteration, ValueError, IndexError):
        return None


class SwapGuard:
    """Fail after sustained swapout growth, matching the proven G-03 boundary."""

    def __init__(
        self,
        *,
        growth_limit_bytes: int = 256 * 1024 * 1024,
        sustain_seconds: float = 15.0,
    ) -> None:
        if growth_limit_bytes <= 0 or sustain_seconds <= 0:
            raise HarnessError("swap guard thresholds are invalid")
        initial = linux_swapout_bytes()
        if initial is None:
            raise HarnessError("cannot read Linux pswpout swap guard")
        self.growth_limit_bytes = int(growth_limit_bytes)
        self.sustain_seconds = float(sustain_seconds)
        self.initial_bytes = initial
        self.current_bytes = initial
        self.exceeded_at: float | None = None
        self.samples = 0

    def sample(self) -> None:
        observed = linux_swapout_bytes()
        if observed is None:
            raise HarnessError("cannot read Linux pswpout swap guard")
        self.current_bytes = observed
        self.samples += 1
        growth = self.current_bytes - self.initial_bytes
        now = time.monotonic()
        if growth < self.growth_limit_bytes:
            self.exceeded_at = None
            return
        if self.exceeded_at is None:
            self.exceeded_at = now
            return
        if now - self.exceeded_at >= self.sustain_seconds:
            raise HarnessError(
                "sustained swapout violation: "
                f"{growth} bytes >= {self.growth_limit_bytes} for "
                f"{now - self.exceeded_at:.1f}s"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "metric": "Linux pswpout bytes from process start",
            "initial_bytes": self.initial_bytes,
            "current_bytes": self.current_bytes,
            "growth_bytes": self.current_bytes - self.initial_bytes,
            "limit_bytes": self.growth_limit_bytes,
            "sustain_seconds": self.sustain_seconds,
            "samples": self.samples,
        }


class FullStageLogger:
    """Append-only JSONL stage logger with timings, RSS, and explicit counts."""

    def __init__(self, path: Path, *, run_id: str, config: Mapping[str, Any]):
        self.path = path
        self.run_id = run_id
        self.config = dict(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emit(
            "config_echo",
            config=self.config,
            modified_count=0,
            examined_count=0,
        )

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "schema": LOG_SCHEMA,
            "run_id": self.run_id,
            "event": event,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_seconds": time.monotonic(),
            "peak_rss_bytes": rss_bytes(),
            "swapout_bytes": linux_swapout_bytes(),
            **fields,
        }
        encoded = canonical_json_bytes(record)
        with self.path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def stage(self, name: str, *, counts: Mapping[str, int] | None = None) -> "Stage":
        return Stage(self, name, dict(counts or {}))


def validate_full_jsonl(value: bytes) -> list[dict[str, Any]]:
    """Validate the complete checkpoint-bound log, not just its file hash."""
    try:
        lines = value.decode("ascii").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError("checkpoint JSONL log is not valid ASCII JSONL") from exc
    if (
        not rows
        or any(not isinstance(row, dict) or row.get("schema") != LOG_SCHEMA for row in rows)
        or rows[0].get("event") != "config_echo"
    ):
        raise HarnessError("checkpoint JSONL log lacks the full logging contract")
    return rows


class Stage(AbstractContextManager["Stage"]):
    def __init__(self, logger: FullStageLogger, name: str, counts: dict[str, int]):
        self.logger = logger
        self.name = name
        self.counts = counts
        self.started = 0.0

    def __enter__(self) -> "Stage":
        self.started = time.monotonic()
        self.logger.emit(
            f"{self.name}_start",
            counts=self.counts,
            examined_count=int(self.counts.get("examined", 0)),
            modified_count=int(self.counts.get("modified", 0)),
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.logger.emit(
            f"{self.name}_end",
            status="PASS" if exc is None else "FAIL",
            elapsed_seconds=time.monotonic() - self.started,
            counts=self.counts,
            examined_count=int(self.counts.get("examined", 0)),
            modified_count=int(self.counts.get("modified", 0)),
            error_type=None if exc is None else type(exc).__name__,
        )
        return False


def _state_digest(value: Any, torch: Any, np: Any) -> str:
    """Hash state semantically instead of relying on pickle byte stability."""
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(array.shape)))
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: repr(key)):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                add(child)
        elif item is None:
            digest.update(b"none\0")
        elif isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("ascii"))
            digest.update(b"\0")
        else:
            raise HarnessError(f"unsupported checkpoint value: {type(item)!r}")

    add(value)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _runtime_fingerprint(torch: Any) -> dict[str, Any]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "devices": devices,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def runtime_fingerprint() -> dict[str, Any]:
    """Return the exact assigned runtime fingerprint without running a workload."""
    import torch

    value = _runtime_fingerprint(torch)
    if not value["devices"]:
        raise HarnessError("a CUDA GPU was not assigned to this Colab session")
    return value


def assigned_gpu_model() -> str:
    """Return the provider-assigned model name exactly as PyTorch reports it."""
    return str(runtime_fingerprint()["devices"][0]["name"])


def _validate_sampler_state(value: Mapping[str, Any]) -> None:
    if (
        "epoch" not in value
        or "cursor" not in value
        or int(value["epoch"]) < 0
        or int(value["cursor"]) < 0
        or not _is_sha256(value.get("permutation_sha256"))
    ):
        raise HarnessError(
            "sampler state must bind a nonnegative epoch/cursor and permutation SHA-256"
        )


def capture_training_state(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sampler_state: Mapping[str, Any],
    global_step: int,
    epoch: int,
    provenance: Mapping[str, str],
) -> tuple[dict[str, Any], str, bytes]:
    """Capture the complete G-03 state boundary for a real training run."""
    import numpy as np
    import torch

    required = {"code_sha256", "config_sha256", "input_manifest_sha256"}
    if set(provenance) != required or any(not _is_sha256(provenance[key]) for key in required):
        raise HarnessError("checkpoint provenance must contain exactly three SHA-256 values")
    _validate_sampler_state(sampler_state)
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "sampler": dict(sampler_state),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "provenance": dict(provenance),
        "runtime": _runtime_fingerprint(torch),
    }
    digest = _state_digest(state, torch, np)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return state, digest, buffer.getvalue()


def verify_serialized_training_state(
    serialized: bytes,
    *,
    expected_state_digest: str,
    expected_provenance: Mapping[str, str],
) -> dict[str, Any]:
    """CPU-only semantic self-test for a pulled checkpoint before training compute."""
    import numpy as np
    import torch

    if not _is_sha256(expected_state_digest):
        raise HarnessError("expected checkpoint semantic digest is invalid")
    state = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
        raise HarnessError("checkpoint state schema is invalid")
    if state.get("provenance") != dict(expected_provenance):
        raise HarnessError("checkpoint provenance drifted before resume")
    sampler_state = state.get("sampler")
    if not isinstance(sampler_state, Mapping):
        raise HarnessError("checkpoint sampler state is malformed")
    _validate_sampler_state(sampler_state)
    observed = _state_digest(state, torch, np)
    if observed != expected_state_digest:
        raise HarnessError("checkpoint semantic digest mismatch before resume")
    return {
        "status": "PASS_SEMANTIC_DIGEST_BEFORE_COMPUTE",
        "state_digest": observed,
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "sampler": dict(sampler_state),
        "provenance": dict(state["provenance"]),
        "producer_runtime": dict(state.get("runtime", {})),
    }


def restore_training_state(
    serialized: bytes,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    expected_provenance: Mapping[str, str],
    runtime_policy: str = "EXACT_RUNTIME",
) -> dict[str, Any]:
    """Verify and restore a complete checkpoint; return its sampler/cursor metadata."""
    import numpy as np
    import torch

    state = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
        raise HarnessError("checkpoint state schema is invalid")
    if state.get("provenance") != dict(expected_provenance):
        raise HarnessError("checkpoint provenance drifted")
    producer_runtime = state.get("runtime")
    current_runtime = _runtime_fingerprint(torch)
    if runtime_policy not in {
        "EXACT_RUNTIME",
        "ALLOW_LOGGED_GPU_TRANSITION_NO_BIT_IDENTICAL_CLAIM",
    }:
        raise HarnessError("checkpoint runtime policy is invalid")
    if producer_runtime != current_runtime and runtime_policy == "EXACT_RUNTIME":
        raise HarnessError("checkpoint runtime/device/determinism fingerprint drifted")
    if producer_runtime != current_runtime:
        adjudicate_resume_runtime(
            producer_runtime if isinstance(producer_runtime, Mapping) else {},
            current_runtime,
            allow_gpu_transition=True,
        )
    sampler_state = state.get("sampler")
    if not isinstance(sampler_state, Mapping):
        raise HarnessError("checkpoint sampler state is malformed")
    _validate_sampler_state(sampler_state)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise HarnessError("checkpoint omitted the configured scheduler")
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        if state["scaler"] is None:
            raise HarnessError("checkpoint omitted the configured scaler")
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
    return {
        "sampler": state["sampler"],
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "state_digest": _state_digest(state, torch, np),
    }


class AtomicCheckpointStore:
    """Immutable step directories plus one atomic, monotonic LATEST pointer."""

    def __init__(self, root: Path):
        self.root = root
        self.checkpoints = root / "checkpoints"

    def publish_bytes(
        self,
        *,
        global_step: int,
        publication_index: int,
        checkpoint: bytes,
        log_jsonl: bytes,
        state_digest: str,
        provenance: Mapping[str, str],
        session: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            global_step < 0
            or publication_index < 0
            or not _is_sha256(state_digest)
            or not log_jsonl
        ):
            raise HarnessError("checkpoint step, publication, digest, or JSONL log is invalid")
        validate_full_jsonl(log_jsonl)
        tag = f"step-{global_step:012d}-pub-{publication_index:06d}"
        latest_path = self.root / LATEST_FILE
        if latest_path.exists():
            previous = _read_json(latest_path, "latest checkpoint pointer")
            previous_key = (
                int(previous.get("global_step", -1)),
                int(previous.get("publication_index", -1)),
            )
            if previous_key > (global_step, publication_index):
                raise HarnessError("checkpoint publication would regress LATEST")
        manifest = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "artifact_id": f"colab-division-{tag}",
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "global_step": global_step,
            "publication_index": publication_index,
            "state_digest": state_digest,
            "checkpoint_sha256": sha256_bytes(checkpoint),
            "log_sha256": sha256_bytes(log_jsonl),
            "provenance": dict(provenance),
            "session": dict(session),
        }
        artifact = publish_artifact(
            self.checkpoints / tag,
            manifest=manifest,
            payloads={
                "state/checkpoint.pt": checkpoint,
                "logs/session.jsonl": log_jsonl,
            },
        )
        pointer = {
            "schema": CHECKPOINT_SCHEMA,
            "tag": tag,
            "global_step": global_step,
            "publication_index": publication_index,
            "tree_sha256": artifact["tree_sha256"],
            "manifest_sha256": artifact["manifest_sha256"],
            "state_digest": state_digest,
        }
        if latest_path.exists():
            previous = _read_json(latest_path, "latest checkpoint pointer")
            previous_key = (
                int(previous.get("global_step", -1)),
                int(previous.get("publication_index", -1)),
            )
            if previous_key == (global_step, publication_index) and previous != pointer:
                raise HarnessError("same-publication checkpoint conflicts with LATEST")
        atomic_replace_json(latest_path, pointer)
        return {**pointer, "artifact": artifact}

    def verify_latest(self) -> dict[str, Any]:
        pointer = _read_json(self.root / LATEST_FILE, "latest checkpoint pointer")
        tag = str(pointer.get("tag", ""))
        expected_tag = (
            f"step-{int(pointer.get('global_step', -1)):012d}"
            f"-pub-{int(pointer.get('publication_index', -1)):06d}"
        )
        if tag != expected_tag:
            raise HarnessError("latest checkpoint tag/step/publication drifted")
        artifact = verify_artifact(self.checkpoints / tag)
        if (
            artifact["tree_sha256"] != pointer.get("tree_sha256")
            or artifact["manifest_sha256"] != pointer.get("manifest_sha256")
            or artifact["manifest"].get("state_digest") != pointer.get("state_digest")
        ):
            raise HarnessError("latest checkpoint pointer does not bind the artifact")
        checkpoint_path = self.checkpoints / tag / "state/checkpoint.pt"
        log_path = self.checkpoints / tag / "logs/session.jsonl"
        if sha256_file(checkpoint_path) != artifact["manifest"].get("checkpoint_sha256"):
            raise HarnessError("checkpoint payload does not bind its manifest")
        if sha256_file(log_path) != artifact["manifest"].get("log_sha256"):
            raise HarnessError("checkpoint JSONL log does not bind its manifest")
        return {
            "pointer": pointer,
            "artifact": artifact,
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
        }

    def read_latest_bytes(self) -> tuple[bytes, dict[str, Any]]:
        receipt = self.verify_latest()
        return receipt["checkpoint_path"].read_bytes(), receipt

    def pull_latest_verified(
        self,
        local_root: Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Pull the LATEST immutable artifact by its manifest hash and self-test it."""
        remote = self.verify_latest()  # Absence or any mismatch fails; no fresh restart.
        pointer = remote["pointer"]
        if (
            expected_manifest_sha256 is not None
            and pointer["manifest_sha256"] != expected_manifest_sha256
        ):
            raise HarnessError("LATEST manifest differs from the requested checkpoint")
        payloads = {
            row["path"]: (self.checkpoints / pointer["tag"] / row["path"]).read_bytes()
            for row in remote["artifact"]["files"]
            if row["path"] != MANIFEST_FILE
        }
        local_artifact = publish_artifact(
            local_root / "checkpoints" / pointer["tag"],
            manifest=remote["artifact"]["manifest"],
            payloads=payloads,
        )
        if (
            local_artifact["manifest_sha256"] != pointer["manifest_sha256"]
            or local_artifact["tree_sha256"] != pointer["tree_sha256"]
        ):
            raise HarnessError("pulled checkpoint differs from the manifest-bound source")
        checkpoint_path = local_root / "checkpoints" / pointer["tag"] / "state/checkpoint.pt"
        semantic = verify_serialized_training_state(
            checkpoint_path.read_bytes(),
            expected_state_digest=pointer["state_digest"],
            expected_provenance=remote["artifact"]["manifest"]["provenance"],
        )
        return {
            "status": "PASS_PULLED_AND_SEMANTICALLY_VERIFIED",
            "pointer": pointer,
            "checkpoint_path": checkpoint_path,
            "log_path": local_root / "checkpoints" / pointer["tag"] / "logs/session.jsonl",
            "semantic": semantic,
            "producer_session": remote["artifact"]["manifest"]["session"],
        }


def publish_training_checkpoint(
    *,
    store: AtomicCheckpointStore,
    publication_index: int,
    log_path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sampler_state: Mapping[str, Any],
    global_step: int,
    epoch: int,
    provenance: Mapping[str, str],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture and push state plus the full log through one immutable boundary."""
    _, digest, checkpoint = capture_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler_state=sampler_state,
        global_step=global_step,
        epoch=epoch,
        provenance=provenance,
    )
    if not log_path.is_file():
        raise HarnessError("full session JSONL log is absent at checkpoint time")
    log_jsonl = log_path.read_bytes()
    validate_full_jsonl(log_jsonl)
    receipt = store.publish_bytes(
        global_step=global_step,
        publication_index=publication_index,
        checkpoint=checkpoint,
        log_jsonl=log_jsonl,
        state_digest=digest,
        provenance=provenance,
        session=session,
    )
    verified = store.verify_latest()
    if verified["pointer"] != {key: receipt[key] for key in verified["pointer"]}:
        raise HarnessError("published training checkpoint did not become exact LATEST")
    return receipt


def adjudicate_resume_runtime(
    producer_runtime: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
    *,
    allow_gpu_transition: bool,
) -> dict[str, Any]:
    """Classify exact versus logged cross-GPU resume without conflating claims."""
    previous = producer_runtime.get("devices", [])
    current = current_runtime.get("devices", [])
    if (
        not isinstance(previous, list)
        or not isinstance(current, list)
        or len(previous) != 1
        or len(current) != 1
    ):
        raise HarnessError("resume requires one exact producer GPU and one assigned GPU")
    exact_runtime = dict(producer_runtime) == dict(current_runtime)
    same_gpu_model = (
        previous[0].get("name") == current[0].get("name")
        and previous[0].get("capability") == current[0].get("capability")
    )
    if not exact_runtime and not allow_gpu_transition:
        raise HarnessError("checkpoint runtime differs and cross-GPU resume is not allowed")
    return {
        "status": (
            "BIT_IDENTICAL_RUNTIME_MATCH"
            if exact_runtime
            else "LOGGED_RUNTIME_TRANSITION_NO_BIT_IDENTICAL_CLAIM"
        ),
        "exact_runtime_match": exact_runtime,
        "same_gpu_model_and_capability": same_gpu_model,
        "producer_gpu_model": str(previous[0].get("name")),
        "assigned_gpu_model": str(current[0].get("name")),
        "producer_runtime_sha256": sha256_bytes(canonical_json_bytes(dict(producer_runtime))),
        "assigned_runtime_sha256": sha256_bytes(canonical_json_bytes(dict(current_runtime))),
    }


@dataclass
class ColabSessionGuard:
    """Checkpoint cadence and graceful-yield policy for finite Colab sessions."""

    checkpoint_interval_seconds: float = 600.0
    session_limit_seconds: float = 10.0 * 3600.0
    safety_margin_seconds: float = 15.0 * 60.0

    def __post_init__(self) -> None:
        if (
            self.checkpoint_interval_seconds <= 0
            or self.safety_margin_seconds <= 0
            or self.session_limit_seconds <= self.safety_margin_seconds
        ):
            raise HarnessError("Colab session guard timings are invalid")
        self.started = time.monotonic()
        self.last_checkpoint = self.started

    def checkpoint_due(self, *, epoch_finished: bool = False) -> bool:
        return epoch_finished or (
            time.monotonic() - self.last_checkpoint >= self.checkpoint_interval_seconds
        )

    def mark_checkpoint(self) -> None:
        self.last_checkpoint = time.monotonic()

    def graceful_yield_due(self) -> bool:
        return (
            time.monotonic() - self.started
            >= self.session_limit_seconds - self.safety_margin_seconds
        )


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HarnessError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise HarnessError("ledger timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def gpu_model_bucket(assigned_model: str) -> str:
    """Normalize provider spelling only; never collapse different GPU models."""
    value = " ".join(str(assigned_model).strip().split())
    if not value:
        raise HarnessError("Colab did not report an assigned GPU model")
    aliases = {
        "Tesla T4": "NVIDIA T4",
        "NVIDIA Tesla T4": "NVIDIA T4",
        "T4": "NVIDIA T4",
        "NVIDIA T4": "NVIDIA T4",
    }
    return aliases.get(value, value)


def _validate_ledger_row(row: Any, line_number: int | None = None) -> dict[str, Any]:
    label = f" line {line_number}" if line_number is not None else ""
    if (
        not isinstance(row, dict)
        or row.get("schema") != LEDGER_SCHEMA
        or row.get("budget_scope") != "colab_only_by_assigned_gpu_model"
        or float(row.get("kaggle_gpu_hours_included", -1)) != 0.0
        or "cumulative_colab_hours" in row
        or "total_colab_hours" in row
    ):
        raise HarnessError(f"Colab ledger{label} commingles or naively sums budgets")
    event = row.get("event")
    if event not in {"LEDGER_INITIALIZED", "SESSION_START", "SESSION_END"}:
        raise HarnessError(f"Colab ledger{label} has an invalid event")
    if event in {"SESSION_START", "SESSION_END"}:
        model = str(row.get("assigned_gpu_model", ""))
        if (
            not row.get("session_id")
            or row.get("gpu_model_bucket") != gpu_model_bucket(model)
            or not _is_sha256(row.get("hardware_fingerprint_sha256"))
            or not _is_sha256(row.get("receipt_sha256"))
        ):
            raise HarnessError(f"Colab ledger{label} lacks assigned-hardware provenance")
    if event == "SESSION_END" and (
        row.get("duration_basis")
        not in {
            "EXACT_RECORDED_END",
            "UPPER_BOUND_LAST_CHECKPOINT_PLUS_CADENCE",
            "UPPER_BOUND_SESSION_LIMIT_NO_ATTRIBUTABLE_CHECKPOINT",
        }
        or float(row.get("duration_uncertainty_seconds", -1.0)) < 0.0
    ):
        raise HarnessError(f"Colab ledger{label} lacks a duration accounting basis")
    return row


def read_colab_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"Colab ledger line {line_number} is invalid") from exc
        rows.append(_validate_ledger_row(row, line_number))
    return rows


def colab_hours_by_gpu_model(path: Path) -> dict[str, float]:
    """Sum only within exact assigned-model buckets; never emit a cross-GPU total."""
    result: dict[str, float] = {}
    for row in read_colab_ledger(path):
        if row.get("event") != "SESSION_END":
            continue
        bucket = str(row["gpu_model_bucket"])
        result[bucket] = result.get(bucket, 0.0) + float(row["session_wall_hours"])
    return dict(sorted(result.items()))


def append_colab_session_start(
    path: Path,
    *,
    session_id: str,
    started_utc: str,
    assigned_gpu_model: str,
    hardware_fingerprint_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Record assigned hardware immediately, so even a disconnect is attributable."""
    started = _parse_utc(started_utc)
    record = {
        "schema": LEDGER_SCHEMA,
        "budget_scope": "colab_only_by_assigned_gpu_model",
        "provider": "Google Colab",
        "event": "SESSION_START",
        "session_id": session_id,
        "started_utc": started.isoformat(),
        "assigned_gpu_model": assigned_gpu_model,
        "gpu_model_bucket": gpu_model_bucket(assigned_gpu_model),
        "hardware_fingerprint_sha256": hardware_fingerprint_sha256,
        "receipt_sha256": receipt_sha256,
        "kaggle_gpu_hours_included": 0.0,
    }
    _validate_ledger_row(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        try:
            rows = [
                _validate_ledger_row(json.loads(line))
                for line in handle.read().decode("ascii").splitlines()
                if line.strip()
            ]
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HarnessError("Colab ledger is not valid ASCII JSONL") from exc
        for row in rows:
            if row.get("session_id") == session_id and row.get("event") == "SESSION_START":
                if row != record:
                    raise HarnessError("Colab session start conflicts with an existing receipt")
                return dict(row)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def append_colab_session(
    path: Path,
    *,
    session_id: str,
    started_utc: str,
    ended_utc: str,
    assigned_gpu_model: str,
    hardware_fingerprint_sha256: str,
    status: str,
    receipt_sha256: str,
    notes: str = "",
    duration_basis: str = "EXACT_RECORDED_END",
    duration_uncertainty_seconds: float = 0.0,
) -> dict[str, Any]:
    """Append one exact Colab session without touching the Kaggle GPU ledger."""
    if not session_id or not _is_sha256(receipt_sha256):
        raise HarnessError("Colab session identity or receipt SHA is invalid")
    started = _parse_utc(started_utc)
    ended = _parse_utc(ended_utc)
    elapsed = (ended - started).total_seconds()
    if elapsed < 0:
        raise HarnessError("Colab session ended before it started")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_bytes = handle.read()
        try:
            raw_rows = existing_bytes.decode("ascii").splitlines()
            rows = [json.loads(line) for line in raw_rows if line.strip()]
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise HarnessError("Colab ledger is not valid ASCII JSONL") from exc
        rows = [_validate_ledger_row(row) for row in rows]
        starts = [
            row
            for row in rows
            if row.get("session_id") == session_id and row.get("event") == "SESSION_START"
        ]
        if len(starts) != 1:
            raise HarnessError("Colab session end lacks one exact SESSION_START row")
        start = starts[0]
        if (
            start["assigned_gpu_model"] != assigned_gpu_model
            or start["hardware_fingerprint_sha256"] != hardware_fingerprint_sha256
            or _parse_utc(start["started_utc"]) != started
        ):
            raise HarnessError("Colab session hardware or start time drifted")
        for row in rows:
            if row.get("session_id") == session_id and row.get("event") == "SESSION_END":
                candidate = dict(row)
                if candidate.get("receipt_sha256") != receipt_sha256:
                    raise HarnessError("Colab session ID conflicts with an existing receipt")
                return candidate
        wall_hours = elapsed / 3600.0
        bucket = gpu_model_bucket(assigned_gpu_model)
        prior_same_model = sum(
            float(row["session_wall_hours"])
            for row in rows
            if row.get("event") == "SESSION_END" and row.get("gpu_model_bucket") == bucket
        )
        record = {
            "schema": LEDGER_SCHEMA,
            "budget_scope": "colab_only_by_assigned_gpu_model",
            "provider": "Google Colab",
            "event": "SESSION_END",
            "session_id": session_id,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "elapsed_seconds": elapsed,
            "session_wall_hours": wall_hours,
            "cumulative_hours_same_gpu_model": prior_same_model + wall_hours,
            "assigned_gpu_model": assigned_gpu_model,
            "gpu_model_bucket": bucket,
            "hardware_fingerprint_sha256": hardware_fingerprint_sha256,
            "status": status,
            "duration_basis": duration_basis,
            "duration_uncertainty_seconds": float(duration_uncertainty_seconds),
            "receipt_sha256": receipt_sha256,
            "notes": notes,
            "kaggle_gpu_hours_included": 0.0,
        }
        _validate_ledger_row(record)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(record))
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def reconcile_interrupted_colab_sessions(
    path: Path,
    *,
    before_utc: str,
    exclude_session_id: str,
    latest_producer_session: Mapping[str, Any],
    checkpoint_cadence_seconds: float = 600.0,
    session_limit_seconds: float = 6.0 * 3600.0,
) -> list[dict[str, Any]]:
    """Close prior open sessions conservatively when a manual resume begins."""
    if checkpoint_cadence_seconds <= 0.0 or session_limit_seconds <= 0.0:
        raise HarnessError("interrupted-session accounting bounds are invalid")
    cutoff = _parse_utc(before_utc)
    rows = read_colab_ledger(path)
    ended_ids = {
        str(row["session_id"])
        for row in rows
        if row.get("event") == "SESSION_END"
    }
    producer_id = str(latest_producer_session.get("session_id", ""))
    checkpoint_at_raw = latest_producer_session.get("checkpoint_started_utc")
    receipts: list[dict[str, Any]] = []
    for start in rows:
        session_id = str(start.get("session_id", ""))
        if (
            start.get("event") != "SESSION_START"
            or session_id == exclude_session_id
            or session_id in ended_ids
        ):
            continue
        started = _parse_utc(str(start["started_utc"]))
        if started > cutoff:
            raise HarnessError("open Colab session begins after the resume cutoff")
        if session_id == producer_id and checkpoint_at_raw:
            checkpoint_at = _parse_utc(str(checkpoint_at_raw))
            bounded_end = checkpoint_at + timedelta(seconds=checkpoint_cadence_seconds)
            duration_basis = "UPPER_BOUND_LAST_CHECKPOINT_PLUS_CADENCE"
            uncertainty = checkpoint_cadence_seconds
        else:
            bounded_end = started + timedelta(seconds=session_limit_seconds)
            duration_basis = "UPPER_BOUND_SESSION_LIMIT_NO_ATTRIBUTABLE_CHECKPOINT"
            uncertainty = session_limit_seconds
        ended = min(cutoff, max(started, bounded_end))
        receipt_body = {
            "session_id": session_id,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "assigned_gpu_model": start["assigned_gpu_model"],
            "hardware_fingerprint_sha256": start["hardware_fingerprint_sha256"],
            "status": "INTERRUPTED_RECONCILED_CONSERVATIVE_UPPER_BOUND",
            "duration_basis": duration_basis,
            "duration_uncertainty_seconds": uncertainty,
        }
        receipt_sha = sha256_bytes(canonical_json_bytes(receipt_body))
        receipts.append(
            append_colab_session(
                path,
                session_id=session_id,
                started_utc=started.isoformat(),
                ended_utc=ended.isoformat(),
                assigned_gpu_model=str(start["assigned_gpu_model"]),
                hardware_fingerprint_sha256=str(start["hardware_fingerprint_sha256"]),
                status=str(receipt_body["status"]),
                receipt_sha256=receipt_sha,
                notes=(
                    "Disconnect prevented Cell 3; duration is a conservative upper bound "
                    "and remains in the exact assigned-GPU bucket."
                ),
                duration_basis=duration_basis,
                duration_uncertainty_seconds=uncertainty,
            )
        )
    return receipts


def validate_objective_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse scientific training until the post-taxonomy contract is frozen."""
    contract = dict(value)
    training = contract.get("training", {})
    gate = contract.get("deployment_gate", {})
    metrics = contract.get("metrics", {})
    evaluation = contract.get("evaluation", {})
    if contract.get("status") != OBJECTIVE_STATUS:
        raise HarnessError("training objective is pending the APNCC-vs-V32 taxonomy")
    governance = contract.get("governance", {})
    if (
        not isinstance(governance, Mapping)
        or governance.get("owner_scope_classification")
        != "NEW_LONGER_CONTEXT_ARCHITECTURE_OUTSIDE_WEIGHT_REFINEMENT_RULE"
        or governance.get("owner_scope_receipt_sha256") != OWNER_SCOPE_SHA256
    ):
        raise HarnessError("training objective lacks the owner new-architecture scope receipt")
    gate_source = contract.get("deployment_gate_preregistration", {})
    if (
        not isinstance(gate_source, Mapping)
        or gate_source.get("sha256") != DEPLOYMENT_GATE_SHA256
    ):
        raise HarnessError("training objective does not bind the frozen deployment gate")
    taxonomy = contract.get("taxonomy", {})
    if not isinstance(taxonomy, Mapping) or len(str(taxonomy.get("sha256", ""))) != 64:
        raise HarnessError("frozen objective does not bind the APNCC-vs-V32 taxonomy")
    if (
        training.get("target") != "parent_event_detection"
        or training.get("data_source") != "REAL_BIOHUB_COMPETITION_TRAIN_ONLY"
        or training.get("architecture_class") != "NEW_LONGER_TEMPORAL_CONTEXT"
        or int(training.get("temporal_context_frames", 0)) <= 2
        or float(training.get("division_loss_weight", 0.0)) <= 0.0
        or training.get("daughter_assignment") != "EXISTING_CONDITIONAL_PAIR_HEAD_FROZEN"
        or training.get("initialization") != "PARTIAL_SHAPE_COMPATIBLE_SPATIAL_ONLY"
        or training.get("v32_weights_trainable") is not False
        or training.get("resume_hardware_policy")
        != "ALLOW_LOGGED_GPU_TRANSITION_NO_BIT_IDENTICAL_CLAIM"
        or float(training.get("checkpoint_interval_seconds", 0.0)) != 600.0
        or not 4.0 <= float(training.get("session_target_hours", 0.0)) <= 6.0
        or not _is_sha256(training.get("entrypoint_sha256"))
    ):
        raise HarnessError("training contract does not encode the requested missing capability")
    if (
        int(gate.get("k", 0)) != 22
        or int(gate.get("minimum_tp", -1)) != 5
        or int(gate.get("maximum_fp", -1)) != 17
        or abs(float(gate.get("minimum_precision", 0.0)) - 5.0 / 22.0) > 1e-15
    ):
        raise HarnessError("deployment gate drifted from the measured top-22 gate")
    primary = metrics.get("primary", [])
    if (
        not isinstance(primary, list)
        or primary != ["precision_at_22", "average_precision", "ap_lift"]
        or float(metrics.get("minimum_ap_lift", 0.0)) != 8462.0
        or metrics.get("auroc_is_a_gate") is not False
    ):
        raise HarnessError("objective must score precision/AP and forbid AUROC gating")
    if (
        not isinstance(evaluation, Mapping)
        or not _is_sha256(evaluation.get("population_manifest_sha256"))
        or not _is_sha256(evaluation.get("scoring_code_sha256"))
        or evaluation.get("scores_sealed_before_labels") is not True
        or evaluation.get("post_score_threshold_or_top_k_sweep") is not False
    ):
        raise HarnessError("objective does not freeze an anti-selection evaluation boundary")
    return contract


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-checkpoint")
    verify.add_argument("--root", type=Path, required=True)
    objective = sub.add_parser("check-objective")
    objective.add_argument("--contract", type=Path, required=True)
    ledger = sub.add_parser("ledger-append")
    ledger.add_argument("--ledger", type=Path, required=True)
    ledger.add_argument("--session-id", required=True)
    ledger.add_argument("--started-utc", required=True)
    ledger.add_argument("--ended-utc", required=True)
    ledger.add_argument("--assigned-gpu-model", required=True)
    ledger.add_argument("--hardware-fingerprint-sha256", required=True)
    ledger.add_argument("--status", required=True)
    ledger.add_argument("--receipt-sha256", required=True)
    ledger.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.command == "verify-checkpoint":
        receipt = AtomicCheckpointStore(args.root).verify_latest()
    elif args.command == "check-objective":
        receipt = validate_objective_contract(_read_json(args.contract, "objective contract"))
    else:
        receipt = append_colab_session(
            args.ledger,
            session_id=args.session_id,
            started_utc=args.started_utc,
            ended_utc=args.ended_utc,
            assigned_gpu_model=args.assigned_gpu_model,
            hardware_fingerprint_sha256=args.hardware_fingerprint_sha256,
            status=args.status,
            receipt_sha256=args.receipt_sha256,
            notes=args.notes,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
