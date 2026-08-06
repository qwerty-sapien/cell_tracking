#!/usr/bin/env python3
"""Bounded, receipted, serially-locked Kaggle operations for the owner's cmux session.

Doctrine (see dag_prompts/2026-07-31-gpu-window/codex/C0_CMUX_OPS_MODULE.md):

    Never type ad-hoc Kaggle commands into the authenticated cmux shell. Invoke this
    module, whose COMPLETE external command surface is enumerated by ``command_plan()``
    and which writes a receipt for every invocation.

An enumerated plan is reviewable before it runs and reproducible after. Free-form typing
into an authenticated session is how a one-run authorization became two byte-identical
kernel versions and a 0.74 GPU-h debit.

Campaign-agnostic on purpose: G-02 is archived, G-03 is live, and both need this. It
complements ``reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py``
(static/launch kernel validation) rather than duplicating it -- that module validates what
you are about to push; this one binds its exact PASS receipt to the bytes being pushed.

Hard rules enforced in code, not in comments:
  * whitelist-only execution, argv lists, never ``shell=True``
  * exclusive serial lock for every mutating op
  * ``kernel_push`` refuses without a fresh eligible prelaunch receipt whose metadata and
    code hashes still match the package directory
  * ``kernel_push`` refuses while a version is RUNNING/QUEUED
  * ``competition_submit`` refuses without an ``eligible: true`` predicate receipt
  * credential-scrubbed receipts for every invocation
  * NO retry logic, ever
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import PurePath
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "BudgetLedgerError",
    "CmuxOpsError",
    "InsufficientGpuQuota",
    "KernelPredicateNotSatisfied",
    "LockHeld",
    "ProjectionApprovalRequired",
    "PredicateNotSatisfied",
    "PushRefused",
    "command_plan",
    "run",
    "serial_lock",
    "bracketed_launch",
    "authenticated_kaggle_environment",
    "reconcile_gpu_reservation",
    "semantic_mutation_error",
    "scrub",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical_shared_repo_root(local_root: Path) -> Path:
    """Resolve the main checkout so linked worktrees share one lock and ledger."""
    override = os.environ.get("CELLTRACK_SHARED_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    marker = local_root / ".git"
    if not marker.is_file():
        return local_root
    try:
        line = marker.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir:"):
            return local_root
        git_dir = Path(line.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (local_root / git_dir).resolve()
        if git_dir.parent.name == "worktrees":
            return git_dir.parent.parent.parent.resolve()
    except OSError:
        pass
    return local_root


SHARED_REPO_ROOT = _canonical_shared_repo_root(REPO_ROOT)
DEFAULT_LOCK = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/.cmux_serial.lock"
)
DEFAULT_GPU_LEDGER = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/KAGGLE_SHARED_GPU_LEDGER.json"
)
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
DEFAULT_SUBMISSION_CLAIM_DIR = (
    SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/.submission_claims/direct"
)

_KERNEL_STATUS = re.compile(
    r"(?<![A-Za-z0-9_])KernelWorkerStatus\."
    r"(RUNNING|QUEUED|COMPLETE|ERROR|CANCELLED)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _parse_kernel_status(text: str) -> str | None:
    """Return one exact Kaggle worker state, rejecting ambiguous prose."""

    matches = _KERNEL_STATUS.findall(text or "")
    if len(matches) != 1:
        return None
    return matches[0].upper()


def _status_proves_fresh_kernel_absence(receipt: dict[str, Any]) -> bool:
    """Accept Kaggle's exact new-slug denial only for an explicit v1 predicate."""

    stderr = str(receipt.get("stderr_scrubbed") or "")
    return (
        receipt.get("returncode") == 1
        and receipt.get("timed_out") is False
        and receipt.get("semantic_failure") is False
        and not str(receipt.get("stdout_scrubbed") or "").strip()
        and "Cannot access kernel " in stderr
        and "Permission 'kernels.get' was denied" in stderr
        and "most likely cause is a wrong kernel slug" in stderr
    )

#: Ops that mutate external state and therefore require the serial lock.
MUTATING_OPS = frozenset(
    {"dataset_create", "dataset_version", "kernel_push", "competition_submit"}
)


class CmuxOpsError(RuntimeError):
    """Base class for every refusal raised by this module."""


class LockHeld(CmuxOpsError):
    """Another holder owns the serial lock."""


class PredicateNotSatisfied(CmuxOpsError):
    """competition_submit was attempted without an eligible predicate receipt."""


class KernelPredicateNotSatisfied(CmuxOpsError):
    """kernel_push was attempted without a fresh eligible package receipt."""


class PushRefused(CmuxOpsError):
    """A kernel push was refused because a version is already in flight."""


class BudgetLedgerError(CmuxOpsError):
    """The single shared GPU ledger is absent, malformed, or unreconciled."""


class InsufficientGpuQuota(BudgetLedgerError):
    """Authenticated remaining quota cannot cover all live reservations."""


class ProjectionApprovalRequired(BudgetLedgerError):
    """The 8-hour owner projection gate is unsatisfied."""


class UnknownOp(CmuxOpsError):
    """The requested op is not in the bounded command plan."""


# --------------------------------------------------------------------------------------
# The bounded command plan -- the security boundary of this module.
# --------------------------------------------------------------------------------------


def command_plan(kaggle: str = "kaggle", **kw: Any) -> dict[str, list[str]]:
    """Return the COMPLETE set of external commands this module may ever execute.

    Every value is an argv list. Nothing here is ever passed through a shell, and no
    caller-supplied string is interpolated into a command word. Adding an entry is a
    reviewable change with a reason in the calling node's report -- not a convenience.
    """
    dataset_id = kw.get("dataset_id", "")
    kernel_id = kw.get("kernel_id", "")
    kernel_version = kw.get("kernel_version")
    if kernel_version is None:
        kernel_ref = kernel_id
    else:
        if (
            isinstance(kernel_version, bool)
            or not isinstance(kernel_version, int)
            or kernel_version <= 0
        ):
            raise ValueError("kernel_version must be a positive integer")
        kernel_ref = f"{kernel_id}/{kernel_version}"
    path = str(kw.get("path", ""))
    message = kw.get("message", "")
    competition = kw.get("competition", DEFAULT_COMPETITION)
    csv = str(kw.get("csv", ""))

    return {
        # --- read-only -------------------------------------------------------------
        "quota": [kaggle, "quota", "--format", "json"],
        "dataset_status": [kaggle, "datasets", "status", dataset_id, "--format", "json"],
        "dataset_search_mine": [
            kaggle, "datasets", "list", "--mine", "--search",
            dataset_id.split("/", 1)[-1], "--format", "json",
        ],
        "dataset_files": [
            kaggle, "datasets", "files", dataset_id,
            "--page-size", "200", "--format", "json",
        ],
        "dataset_download": [
            kaggle, "datasets", "download", "-d", dataset_id, "-p", path, "--unzip",
        ],
        "kernel_status": [kaggle, "kernels", "status", kernel_ref],
        "kernel_list_mine": [
            kaggle, "kernels", "list", "--mine", "--page-size", "100", "--format", "json",
        ],
        "kernel_files": [
            kaggle, "kernels", "files", kernel_ref,
            "--page-size", "200", "--format", "json",
        ],
        "kernel_pull": [
            kaggle, "kernels", "pull", kernel_ref, "-p", path, "--metadata",
        ],
        "kernel_output": [kaggle, "kernels", "output", kernel_ref, "-p", path],
        # Version-qualified logs are the only remaining read-only route for a
        # historical kernel traceback when ``output`` returns latest artifacts.
        "kernel_logs": [kaggle, "kernels", "logs", kernel_ref],
        "competition_submissions": [
            kaggle, "competitions", "submissions", competition,
            "--page-size", "50", "--format", "json",
        ],
        # --- mutating --------------------------------------------------------------
        "dataset_create": [
            kaggle, "datasets", "create", "-p", path, "--dir-mode", "zip",
        ],
        "dataset_version": [
            kaggle, "datasets", "version", "-p", path, "-m", message, "-r", "zip",
        ],
        "kernel_push": [kaggle, "kernels", "push", "-p", path],
        # --- irreversible ----------------------------------------------------------
        "competition_submit": [
            kaggle, "competitions", "submit", "-c", competition, "-f", csv, "-m", message,
        ],
    }


# --------------------------------------------------------------------------------------
# Credential scrubbing -- a receipt must never carry a secret.
# --------------------------------------------------------------------------------------

_SECRET_KEY = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization|bearer|cookie|"
    r"session|credential)\b\s*[:=]\s*\S+"
)
_LONG_OPAQUE = re.compile(r"\b[A-Za-z0-9+/_-]{32,}={0,2}\b")
_REDACTED = "[REDACTED]"
_HOURS = re.compile(r"^([0-9]+(?:\.[0-9]+)?)h$")


def scrub(text: str) -> str:
    """Remove anything that looks like a credential from *text*.

    Deliberately aggressive: a false positive costs a little readability in a receipt,
    a false negative writes the owner's token into a committed artifact.
    """
    if not text:
        return ""
    out = _SECRET_KEY.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)
    return _LONG_OPAQUE.sub(_REDACTED, out)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _kaggle_credentials_path() -> Path:
    """Return the one credential file the authenticated Kaggle CLI may use."""
    return Path.home() / ".kaggle" / "credentials.json"


def _authenticated_kaggle_environment(kaggle: str) -> dict[str, str] | None:
    """Supply the owner's API token to a real Kaggle CLI without exposing it.

    The current CLI requires ``KAGGLE_API_TOKEN`` but does not automatically
    consume the existing access-token credential file.  Test fixtures and
    arbitrary non-Kaggle executables receive no altered environment, which
    keeps the command-plan test surface credential-free.
    """
    if PurePath(kaggle).name not in {"kaggle", "kaggle.exe"}:
        return None
    candidate = Path(kaggle)
    if not candidate.is_absolute():
        resolved_text = shutil.which(kaggle)
        if resolved_text is None:
            return None
        candidate = Path(resolved_text)
    candidate = candidate.resolve()
    trusted: set[Path] = set()
    configured = os.environ.get("CELLTRACK_KAGGLE_EXECUTABLE")
    if configured:
        trusted.add(Path(configured).expanduser().resolve())
    discovered = shutil.which("kaggle")
    if discovered:
        trusted.add(Path(discovered).resolve())
    local_cli = REPO_ROOT / ".venv/bin/kaggle"
    if local_cli.is_file():
        trusted.add(local_cli.resolve())
    if candidate not in trusted:
        return None
    environment = os.environ.copy()
    if environment.get("KAGGLE_API_TOKEN"):
        return environment
    path = _kaggle_credentials_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        token = value["access_token"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CmuxOpsError(
            "authenticated Kaggle access token is unavailable; refusing external action"
        ) from exc
    if not isinstance(token, str) or len(token) < 32:
        raise CmuxOpsError(
            "authenticated Kaggle access token is malformed; refusing external action"
        )
    environment["KAGGLE_API_TOKEN"] = token
    return environment


def authenticated_kaggle_environment(kaggle: str) -> dict[str, str] | None:
    """Return the credential-safe environment for a bounded Kaggle command."""

    return _authenticated_kaggle_environment(kaggle)


# --------------------------------------------------------------------------------------
# Serial lock
# --------------------------------------------------------------------------------------


class serial_lock:
    """Exclusive lock guarding every mutating Kaggle op.

    The persistent path is only a diagnostic anchor. Ownership is an advisory OS lock
    held on an open descriptor, so process death releases it atomically and no supervisor
    ever deletes another supervisor's lock file. ``stale_after_s`` and ``status_probe``
    remain accepted for API compatibility but are intentionally not used to break locks.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_LOCK,
        *,
        node: str = "",
        target: str = "",
        stale_after_s: float = (5.0 + 1.0) * 3600.0,
        status_probe: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.node = node
        self.target = target
        self.stale_after_s = stale_after_s
        self.status_probe = status_probe
        self._fd: int | None = None
        self.broke_stale: dict[str, Any] | None = None

    def _payload(self) -> str:
        return json.dumps(
            {
                "pid": os.getpid(),
                "node": self.node,
                "target": self.target,
                "acquired_utc": _utc(),
                "acquired_monotonic": time.time(),
                "broke_stale": self.broke_stale,
            },
            indent=2,
        )

    def _try_create(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        body = self._payload().encode()
        os.ftruncate(fd, 0)
        os.write(fd, body)
        os.fsync(fd)
        self._fd = fd
        return True

    def acquire(self) -> "serial_lock":
        if self._try_create():
            return self
        raise LockHeld(
            f"serial lock held at {self.path}; refusing to wait, delete, or race"
        )

    def release(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self) -> "serial_lock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _write_receipt(receipt: dict[str, Any], receipt_dir: Path | str | None) -> dict[str, Any]:
    if receipt_dir is None:
        return receipt
    d = Path(receipt_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = f"{receipt['op']}_{receipt['started_utc'].replace(':', '').replace('.', '')}.json"
    p = d / name
    p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_path"] = str(p)
    receipt["receipt_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return receipt


def _validate_submission_predicate(
    path: Path | str,
    *,
    competition: str,
    csv: str,
) -> dict[str, Any]:
    """Validate a direct-file permit against the exact competition and bytes.

    A bare ``eligible: true`` flag is not mutation authority. Code-output
    submissions use the stricter version-bound lifecycle module instead.
    """

    predicate_path = Path(path).resolve()
    try:
        payload = json.loads(predicate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredicateNotSatisfied(f"submission predicate is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise PredicateNotSatisfied("submission predicate must be a JSON object")

    failures: list[str] = []
    schema = payload.get("schema")
    if not isinstance(schema, str) or re.fullmatch(
        r"celltrack-alpha\.[a-z0-9-]+-submission-predicate\.v[1-9][0-9]*",
        schema,
    ) is None:
        failures.append("unsupported predicate schema")
    if payload.get("status") != "PASS" or payload.get("eligible") is not True:
        failures.append("predicate is not PASS/eligible")
    if payload.get("competition") != competition:
        failures.append("competition binding drift")
    if payload.get("submission_mode") != "direct_file":
        failures.append("submission mode is not direct_file")
    if payload.get("revoked") is not False or payload.get("consumed") is not False:
        failures.append("predicate is revoked or consumed")
    if payload.get("failures") != []:
        failures.append("predicate failures are nonempty")

    try:
        artifact_path = Path(str(payload.get("artifact_path", ""))).resolve()
    except (OSError, RuntimeError, ValueError):
        artifact_path = Path()
    if not csv or artifact_path != Path(csv).resolve():
        failures.append("artifact path binding drift")
    expected_sha = payload.get("artifact_sha256")
    expected_bytes = payload.get("artifact_bytes")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        failures.append("artifact SHA is invalid")
    try:
        actual_bytes = artifact_path.stat().st_size
        actual_sha = _sha256_file(artifact_path)
    except OSError as exc:
        failures.append(f"artifact unreadable: {exc}")
    else:
        if expected_bytes != actual_bytes:
            failures.append("artifact byte-size drift")
        if expected_sha != actual_sha:
            failures.append("artifact SHA drift")
    if failures:
        raise PredicateNotSatisfied("; ".join(failures))
    return {
        "predicate_path": str(predicate_path),
        "predicate_sha256": _sha256_file(predicate_path),
        "schema": schema,
        "competition": competition,
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_sha,
        "artifact_bytes": expected_bytes,
    }


def _claim_direct_submission(predicate: dict[str, Any]) -> dict[str, Any]:
    """Consume one direct-file predicate globally before external mutation."""

    predicate_sha = str(predicate["predicate_sha256"])
    claim_path = DEFAULT_SUBMISSION_CLAIM_DIR / f"{predicate_sha}.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "celltrack-alpha.direct-submission-claim.v1",
        "status": "CLAIMED_BEFORE_EXTERNAL_MUTATION",
        "claimed_utc": _utc(),
        "predicate_sha256": predicate_sha,
        "competition": predicate["competition"],
        "artifact_sha256": predicate["artifact_sha256"],
        "retry_allowed": False,
    }
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(claim, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PredicateNotSatisfied(
            f"submission predicate was already consumed at {claim_path}"
        ) from exc
    return {"claim_path": str(claim_path), **claim}


def _validate_kernel_push_predicate(
    receipt_path: Path | str,
    *,
    kernel_id: str,
    kernel_dir: Path | str,
) -> dict[str, Any]:
    """Bind an attested prelaunch PASS to the exact local package bytes.

    The former launch path could bypass the prelaunch guard entirely. A stale
    receipt is also unsafe, so this check re-hashes the metadata and declared
    code file immediately before any status query or push.
    """
    receipt_file = Path(receipt_path)
    try:
        predicate = json.loads(receipt_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KernelPredicateNotSatisfied(
            f"kernel push predicate receipt unreadable: {exc}"
        ) from exc
    if not isinstance(predicate, dict):
        raise KernelPredicateNotSatisfied("kernel push predicate must be a JSON object")

    package = Path(kernel_dir).resolve()
    metadata_path = package / "kernel-metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise KernelPredicateNotSatisfied(
            f"kernel metadata unreadable at {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise KernelPredicateNotSatisfied("kernel metadata must be a JSON object")
    code_path = package / str(metadata.get("code_file", ""))
    if not code_path.is_file():
        raise KernelPredicateNotSatisfied(
            f"declared kernel code file is absent: {code_path}"
        )

    failures: list[str] = []
    if predicate.get("schema") != "celltrack-alpha.kaggle-prelaunch-guard-result.v1":
        failures.append("unexpected predicate schema")
    if predicate.get("status") != "PASS" or predicate.get("eligible") is not True:
        failures.append("predicate is not PASS and eligible:true")
    if predicate.get("phase") not in {"inline", "launch"}:
        failures.append("predicate phase is not push-eligible")
    if predicate.get("failures") != []:
        failures.append("predicate contains failures")
    if (
        not kernel_id
        or metadata.get("id") != kernel_id
        or predicate.get("kernel_id") != kernel_id
    ):
        failures.append("kernel ID drift")
    if predicate.get("kernel_dir") != str(package):
        failures.append("kernel directory drift")
    if predicate.get("metadata_sha256") != _sha256_file(metadata_path):
        failures.append("kernel metadata changed after prelaunch")
    if predicate.get("code_sha256") != _sha256_file(code_path):
        failures.append("kernel code changed after prelaunch")
    try:
        predicate_code_path = Path(str(predicate.get("code_file", ""))).resolve()
    except (OSError, RuntimeError, ValueError):
        predicate_code_path = Path()
    if predicate_code_path != code_path.resolve():
        failures.append("declared code path drift")
    for field in ("contract_sha256", "registry_sha256"):
        value = predicate.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            failures.append(f"{field} missing or malformed")
    expected_initial_version = predicate.get("expected_initial_kernel_version")
    if expected_initial_version not in {None, 1}:
        failures.append("expected_initial_kernel_version must be exactly 1 when present")
    if failures:
        raise KernelPredicateNotSatisfied(
            "kernel push predicate rejected: " + "; ".join(failures)
        )
    return {
        "receipt_path": str(receipt_file.resolve()),
        "receipt_sha256": _sha256_file(receipt_file),
        "kernel_id": kernel_id,
        "kernel_dir": str(package),
        "metadata_sha256": _sha256_file(metadata_path),
        "code_sha256": _sha256_file(code_path),
        "phase": predicate["phase"],
        "expected_initial_kernel_version": expected_initial_version,
    }


_MUTATION_ERROR_TEXT = re.compile(
    r"(?im)^(?:dataset creation|dataset versioning|kernel push|competition submission) "
    r"error:\s*(.+)$"
)


def _semantic_mutation_error(op: str, stdout: str, stderr: str) -> str | None:
    """Detect Kaggle CLI mutations that print failure text but exit zero.

    Kaggle CLI 2.x can report ``Dataset creation error: ...`` on stdout while
    returning process status 0.  Treating that as a successful mutation can
    advance a launch into a missing or partial input carrier.
    """
    if op not in MUTATING_OPS:
        return None
    match = _MUTATION_ERROR_TEXT.search(f"{stdout}\n{stderr}")
    if match is not None:
        return f"Kaggle CLI reported {match.group(0).strip()}"
    return None


def semantic_mutation_error(op: str, stdout: str, stderr: str) -> str | None:
    """Public semantic-error classifier for bounded lifecycle modules."""

    return _semantic_mutation_error(op, stdout, stderr)


def run(
    op: str,
    *,
    kaggle: str = "kaggle",
    receipt_dir: Path | str | None = None,
    timeout: float | None = 1800.0,
    submission_predicate_receipt: Path | str | None = None,
    kernel_push_predicate_receipt: Path | str | None = None,
    lock_path: Path | str = DEFAULT_LOCK,
    _held_lock: serial_lock | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Execute one bounded op and return its receipt.

    Never retries. A failure returns its receipt with the real returncode; deciding whether
    a retry is legitimate is a registry/human decision (KAGGLE_FAILURE_REGISTRY.json),
    never a loop in here.
    """
    plan = command_plan(kaggle=kaggle, **kw)
    if op not in plan:
        raise UnknownOp(f"{op!r} is not in the bounded command plan: {sorted(plan)}")

    if op == "kernel_push" and kernel_push_predicate_receipt is None:
        raise KernelPredicateNotSatisfied(
            "kernel_push requires kernel_push_predicate_receipt; there is no override"
        )
    if op == "kernel_push" and _held_lock is None:
        raise BudgetLedgerError(
            "direct kernel_push is forbidden; use bracketed_launch with a positive "
            "gpu_reservation_h so the shared weekly ledger is checked"
        )

    owned_lock: serial_lock | None = None
    if op in MUTATING_OPS:
        if _held_lock is not None:
            if _held_lock._fd is None:
                raise LockHeld("the supplied mutation lock is not held")
        else:
            owned_lock = serial_lock(
                lock_path,
                node=f"cmux_ops:{op}",
                target=str(
                    kw.get("kernel_id")
                    or kw.get("dataset_id")
                    or kw.get("competition", DEFAULT_COMPETITION)
                ),
            )
            owned_lock.acquire()

    submission_predicate = None
    submission_claim = None
    kernel_push_predicate = None
    kernel_push_status_guard = None
    try:
        # Predicate validation occurs under the same lock as the mutation,
        # closing the former validate-then-wait time-of-check/time-of-use gap.
        if op == "competition_submit":
            if submission_predicate_receipt is None:
                raise PredicateNotSatisfied(
                    "competition_submit requires submission_predicate_receipt; there is no override"
                )
            submission_predicate = _validate_submission_predicate(
                submission_predicate_receipt,
                competition=str(kw.get("competition", DEFAULT_COMPETITION)),
                csv=str(kw.get("csv", "")),
            )
            submission_claim = _claim_direct_submission(submission_predicate)

        if op == "kernel_push":
            if kernel_push_predicate_receipt is None:
                raise KernelPredicateNotSatisfied(
                    "kernel_push requires kernel_push_predicate_receipt; there is no override"
                )
            kernel_push_predicate = _validate_kernel_push_predicate(
                kernel_push_predicate_receipt,
                kernel_id=str(kw.get("kernel_id", "")),
                kernel_dir=str(kw.get("path", "")),
            )
            status_kw = dict(kw)
            status_kw["kernel_version"] = None
            status = run(
                "kernel_status",
                kaggle=kaggle,
                receipt_dir=receipt_dir,
                timeout=timeout,
                **status_kw,
            )
            parsed_status = _parse_kernel_status(
                str(status.get("stdout_scrubbed") or "")
            )
            expected_initial_version = kernel_push_predicate.get(
                "expected_initial_kernel_version"
            )
            if expected_initial_version == 1:
                if not _status_proves_fresh_kernel_absence(status):
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: predicate requires "
                        "a fresh exact Version 1, but the under-lock status did not prove "
                        "that the slug is absent"
                    )
                kernel_push_status_guard = {
                    "disposition": "FRESH_SLUG_ABSENT_EXPECT_VERSION_1",
                    "status_receipt": status.get("receipt_path"),
                    "status_receipt_sha256": status.get("receipt_sha256"),
                }
            else:
                if (
                    status.get("returncode") != 0
                    or status.get("timed_out") is True
                    or parsed_status is None
                ):
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: latest-version status "
                        "was unsuccessful or unparseable"
                    )
                if parsed_status in {"RUNNING", "QUEUED"}:
                    raise PushRefused(
                        f"refusing to push {kw.get('kernel_id')!r}: a version is already in flight "
                        f"({status.get('stdout_scrubbed', '').strip()!r}). This is the 0.74 GPU-h "
                        "launch-race guard."
                    )
                kernel_push_status_guard = {
                    "disposition": f"EXISTING_{parsed_status}",
                    "status_receipt": status.get("receipt_path"),
                    "status_receipt_sha256": status.get("receipt_sha256"),
                }

        argv = plan[op]
        started = _utc()
        t0 = time.time()
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, never shell
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env=_authenticated_kaggle_environment(kaggle),
            )
            process_rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
            semantic_error = _semantic_mutation_error(op, out, err)
            rc = 1 if process_rc == 0 and semantic_error else process_rc
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            rc, process_rc, timed_out, semantic_error = None, None, True, None
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        receipt = {
            "schema": "celltrack-alpha.cmux-op-receipt.v1",
            "op": op,
            "argv": argv,
            "mutating": op in MUTATING_OPS,
            "started_utc": started,
            "ended_utc": _utc(),
            "duration_s": round(time.time() - t0, 6),
            "returncode": rc,
            "process_returncode": process_rc,
            "semantic_failure": semantic_error is not None,
            "semantic_failure_reason": semantic_error,
            "timed_out": timed_out,
            "stdout_sha256": _sha256(out),
            "stderr_sha256": _sha256(err),
            "stdout_scrubbed": scrub(out),
            "stderr_scrubbed": scrub(err),
        }
        if kernel_push_predicate is not None:
            receipt["kernel_push_predicate"] = kernel_push_predicate
        if kernel_push_status_guard is not None:
            receipt["kernel_push_status_guard"] = kernel_push_status_guard
        if submission_predicate is not None:
            receipt["submission_predicate"] = submission_predicate
        if submission_claim is not None:
            receipt["submission_claim"] = submission_claim
        return _write_receipt(receipt, receipt_dir)
    finally:
        if owned_lock is not None:
            owned_lock.release()


def parse_authoritative_gpu_quota(payload: Any) -> dict[str, Any]:
    """Normalize only an authenticated, unambiguous Kaggle GPU quota response.

    The real CLI emits a list of resource rows using values such as ``"5.43h"``.
    A test fixture may use the already-normalized mapping form.  Any other shape is
    an integrity failure, never a reason to run the mutating operation anyway.
    """
    if isinstance(payload, list):
        rows = [
            row for row in payload
            if isinstance(row, dict) and str(row.get("resource", "")).upper() == "GPU"
        ]
        if len(rows) != 1:
            raise CmuxOpsError(
                f"expected exactly one GPU quota row, found {len(rows)}"
            )
        row = rows[0]

        def hours(name: str) -> float:
            match = _HOURS.fullmatch(str(row.get(name, "")).strip())
            if match is None:
                raise CmuxOpsError(f"malformed GPU quota {name!r}: {row.get(name)!r}")
            return float(match.group(1))

        result = {
            "used_h": hours("used"),
            "remaining_h": hours("remaining"),
            "total_h": hours("total"),
            "refresh_at": row.get("refreshAt"),
        }
        if not all(
            math.isfinite(result[key]) and result[key] >= 0
            for key in ("used_h", "remaining_h", "total_h")
        ):
            raise CmuxOpsError("GPU quota contains a non-finite or negative balance")
        if round(result["used_h"] + result["remaining_h"], 6) != round(result["total_h"], 6):
            raise CmuxOpsError(f"GPU quota arithmetic drifted: {result}")
        return result
    if isinstance(payload, dict):
        try:
            used = float(payload["used_h"])
            remaining = float(payload["remaining_h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CmuxOpsError("normalized quota fixture is malformed") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (used, remaining)):
            raise CmuxOpsError(
                "normalized quota fixture contains a non-finite or negative balance"
            )
        result = {"used_h": used, "remaining_h": remaining}
        if "total_h" in payload:
            total = float(payload["total_h"])
            if not math.isfinite(total) or total < 0:
                raise CmuxOpsError(
                    "normalized quota fixture contains a non-finite or negative total"
                )
            if round(used + remaining, 6) != round(total, 6):
                raise CmuxOpsError("normalized quota arithmetic drifted")
            result["total_h"] = total
        if "refresh_at" in payload:
            result["refresh_at"] = payload["refresh_at"]
        return result
    raise CmuxOpsError("quota response is neither a Kaggle resource list nor a normalized mapping")


GPU_LEDGER_SCHEMA = "celltrack-alpha.kaggle-shared-gpu-ledger.v1"


def _read_gpu_ledger(path: Path | str) -> dict[str, Any]:
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetLedgerError(
            f"shared GPU ledger is unavailable or malformed at {ledger_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != GPU_LEDGER_SCHEMA:
        raise BudgetLedgerError("shared GPU ledger has an unexpected schema")
    if not isinstance(payload.get("active_reservations"), list):
        raise BudgetLedgerError("shared GPU ledger active_reservations is malformed")
    if not isinstance(payload.get("events"), list):
        raise BudgetLedgerError("shared GPU ledger events is malformed")
    try:
        capacity = float(payload.get("capacity_h"))
        reservation_hours = [
            float(item["max_gpu_h"])
            for item in payload["active_reservations"]
            if isinstance(item, dict)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetLedgerError("shared GPU ledger hour values are malformed") from exc
    if len(reservation_hours) != len(payload["active_reservations"]):
        raise BudgetLedgerError("shared GPU ledger contains a non-object reservation")
    if not math.isfinite(capacity) or capacity <= 0 or any(
        not math.isfinite(value) or value <= 0 for value in reservation_hours
    ):
        raise BudgetLedgerError("shared GPU ledger contains non-finite/non-positive hours")
    return payload


def _write_gpu_ledger(path: Path | str, payload: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ledger_path)
    finally:
        temporary.unlink(missing_ok=True)


def _authoritative_snapshot(quota: dict[str, Any], *, receipt: str | None = None) -> dict[str, Any]:
    snapshot = {
        "used_h": float(quota["used_h"]),
        "remaining_h": float(quota["remaining_h"]),
        "total_h": float(quota.get("total_h", 30.0)),
        "refresh_at": quota.get("refresh_at"),
        "recorded_utc": _utc(),
    }
    if receipt:
        receipt_path = Path(receipt)
        snapshot["receipt"] = str(receipt_path)
        if receipt_path.is_file():
            snapshot["receipt_sha256"] = _sha256_file(receipt_path)
    return snapshot


def _reserve_gpu_budget(
    *,
    ledger_path: Path | str,
    quota: dict[str, Any],
    kernel_id: str,
    max_gpu_h: float,
    quota_receipt: str | None,
) -> dict[str, Any]:
    if not kernel_id:
        raise BudgetLedgerError("GPU reservation requires a kernel_id")
    if not math.isfinite(max_gpu_h) or max_gpu_h <= 0:
        raise BudgetLedgerError("kernel_push requires a positive --gpu-reservation-h")
    ledger = _read_gpu_ledger(ledger_path)
    capacity = float(ledger.get("capacity_h", 0.0))
    total = float(quota.get("total_h", capacity))
    if not all(
        math.isfinite(value) and value >= 0
        for value in (capacity, total, float(quota["remaining_h"]))
    ):
        raise BudgetLedgerError("quota/ledger hours must be finite and non-negative")
    if capacity != total:
        raise BudgetLedgerError(
            f"shared ledger capacity {capacity} differs from authenticated quota {total}"
        )
    refresh_at = quota.get("refresh_at")
    if ledger.get("refresh_at") != refresh_at:
        if ledger["active_reservations"]:
            raise BudgetLedgerError(
                "quota refresh boundary changed while reservations remain; reconcile first"
            )
        ledger["refresh_at"] = refresh_at
        ledger["events"].append(
            {"type": "weekly_rollover", "recorded_utc": _utc(), "refresh_at": refresh_at}
        )
    reservations = ledger["active_reservations"]
    if any(item.get("kernel_id") == kernel_id for item in reservations):
        raise BudgetLedgerError(f"kernel already has an active GPU reservation: {kernel_id}")
    reserved_h = sum(float(item["max_gpu_h"]) for item in reservations)
    required_h = reserved_h + max_gpu_h
    if required_h > float(quota["remaining_h"]) + 1e-9:
        raise InsufficientGpuQuota(
            f"authenticated remaining {quota['remaining_h']} h cannot cover existing "
            f"reservations {reserved_h} h plus new ceiling {max_gpu_h} h"
        )
    reservation = {
        "kernel_id": kernel_id,
        "max_gpu_h": max_gpu_h,
        "reserved_utc": _utc(),
        "status": "RESERVED_PRELAUNCH",
    }
    reservations.append(reservation)
    ledger["latest_authoritative_quota"] = _authoritative_snapshot(
        quota, receipt=quota_receipt
    )
    ledger["events"].append(
        {
            "type": "reserve",
            "recorded_utc": _utc(),
            "kernel_id": kernel_id,
            "max_gpu_h": max_gpu_h,
            "existing_reserved_h": reserved_h,
            "authenticated_remaining_h": float(quota["remaining_h"]),
        }
    )
    ledger["updated_utc"] = _utc()
    _write_gpu_ledger(ledger_path, ledger)
    return reservation


def _projection_gate(
    authenticated_projection_h: float | None,
    human_approval_receipt: Path | str | None,
) -> dict[str, Any]:
    """Enforce the owner-approved 8 h analysis gate without a runtime kill.

    The weekly ledger reservation remains independently mandatory.  This gate is
    intentionally based on the authenticated estimate, never on sharding or a
    synthetic per-process timeout.
    """

    if authenticated_projection_h is None:
        raise ProjectionApprovalRequired(
            "kernel_push requires authenticated_projection_h for the owner 8 h gate"
        )
    projection = float(authenticated_projection_h)
    if not math.isfinite(projection) or projection <= 0:
        raise ProjectionApprovalRequired(
            "authenticated_projection_h must be a finite positive number"
        )
    result: dict[str, Any] = {
        "authenticated_projection_h": projection,
        "threshold_h": 8.0,
        "runtime_wall_clock_kill": False,
    }
    if projection < 8.0:
        result["disposition"] = "AUTO_APPROVED_LT_8H"
        return result

    if human_approval_receipt is None:
        raise ProjectionApprovalRequired(
            "projection >= 8 h requires an explicit human approval receipt with a "
            "literature-cited pitch; do not shard to evade this gate"
        )
    path = Path(human_approval_receipt)
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionApprovalRequired(
            f"human approval receipt is unreadable: {exc}"
        ) from exc
    if not isinstance(approval, dict):
        raise ProjectionApprovalRequired("human approval receipt must be a JSON object")
    required_text = (
        "expected_effect_size",
        "compute_necessity",
        "failure_modes",
        "falsification_evidence",
        "approved_by",
    )
    missing = [key for key in required_text if not str(approval.get(key, "")).strip()]
    sources = approval.get("primary_sources")
    if (
        approval.get("status") != "APPROVED_BY_HUMAN"
        or not isinstance(sources, list)
        or not sources
        or missing
    ):
        raise ProjectionApprovalRequired(
            "human approval receipt must be APPROVED_BY_HUMAN and include primary_sources, "
            "effect size, compute necessity, failure modes, falsification evidence, and approver"
        )
    result.update(
        {
            "disposition": "HUMAN_APPROVED_GE_8H",
            "approval_receipt": str(path.resolve()),
            "approval_receipt_sha256": _sha256_file(path),
        }
    )
    return result


def reconcile_gpu_reservation(
    kernel_id: str,
    *,
    terminal_status: str,
    quota: dict[str, Any] | None = None,
    quota_receipt: str | None = None,
    terminal_receipt: str | None = None,
    ledger_path: Path | str = DEFAULT_GPU_LEDGER,
    lock_path: Path | str = DEFAULT_LOCK,
    _held_lock: serial_lock | None = None,
) -> dict[str, Any]:
    """Release one terminal reservation while holding the canonical mutation lock."""

    owned_lock: serial_lock | None = None
    if _held_lock is not None:
        if _held_lock._fd is None:
            raise LockHeld("the supplied reservation-reconcile lock is not held")
    else:
        owned_lock = serial_lock(
            lock_path, node="reconcile_gpu_reservation", target=kernel_id
        ).acquire()
    try:
        normalized_status = terminal_status.upper()
        if normalized_status not in {"COMPLETE", "ERROR", "CANCELLED"}:
            raise BudgetLedgerError(f"nonterminal reservation status: {terminal_status!r}")
        ledger = _read_gpu_ledger(ledger_path)
        matches = [
            item for item in ledger["active_reservations"]
            if item.get("kernel_id") == kernel_id
        ]
        if len(matches) != 1:
            raise BudgetLedgerError(
                f"expected one active reservation for {kernel_id}, found {len(matches)}"
            )
        reservation_status = str(matches[0].get("status", ""))
        internal_prelaunch_rollback = (
            normalized_status == "ERROR"
            and reservation_status == "RESERVED_PRELAUNCH"
            and _held_lock is not None
        )
        if not terminal_receipt and not internal_prelaunch_rollback:
            raise BudgetLedgerError(
                "terminal reservation release requires kernel-bound terminal evidence"
            )
        if terminal_receipt:
            terminal_path = Path(terminal_receipt)
            try:
                terminal_evidence = json.loads(terminal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BudgetLedgerError(f"terminal receipt is unreadable: {exc}") from exc
            if not isinstance(terminal_evidence, dict):
                raise BudgetLedgerError("terminal receipt must be a JSON object")
            evidence_op = terminal_evidence.get("op")
            evidence_argv = terminal_evidence.get("argv")
            parsed_evidence_status = _parse_kernel_status(
                str(terminal_evidence.get("stdout_scrubbed", ""))
            )
            exact_status = (
                terminal_evidence.get("schema")
                == "celltrack-alpha.cmux-op-receipt.v1"
                and evidence_op == "kernel_status"
                and terminal_evidence.get("returncode") == 0
                and isinstance(evidence_argv, list)
                and len(evidence_argv) >= 4
                and evidence_argv[1:3] == ["kernels", "status"]
                and evidence_argv[3] == kernel_id
                and parsed_evidence_status == normalized_status
            )
            push_predicate = terminal_evidence.get("kernel_push_predicate")
            push_rc = terminal_evidence.get("returncode")
            push_failed = (
                (type(push_rc) is int and push_rc != 0)
                or (push_rc is None and terminal_evidence.get("timed_out") is True)
            )
            failed_push = (
                normalized_status == "ERROR"
                and terminal_evidence.get("schema")
                == "celltrack-alpha.cmux-op-receipt.v1"
                and evidence_op == "kernel_push"
                and isinstance(evidence_argv, list)
                and len(evidence_argv) >= 3
                and evidence_argv[1:3] == ["kernels", "push"]
                and push_failed
                and isinstance(push_predicate, dict)
                and push_predicate.get("kernel_id") == kernel_id
            )
            if not (exact_status or failed_push):
                raise BudgetLedgerError(
                    "terminal receipt does not prove this kernel and terminal state"
                )
        if quota is not None:
            quota = parse_authoritative_gpu_quota(quota)
        ledger["active_reservations"] = [
            item for item in ledger["active_reservations"]
            if item.get("kernel_id") != kernel_id
        ]
        event = {
            "type": "terminal_reconcile",
            "recorded_utc": _utc(),
            "kernel_id": kernel_id,
            "terminal_status": normalized_status,
            "released_max_gpu_h": float(matches[0]["max_gpu_h"]),
        }
        if terminal_receipt:
            terminal_path = Path(terminal_receipt)
            event["terminal_receipt"] = str(terminal_path)
            if terminal_path.is_file():
                event["terminal_receipt_sha256"] = _sha256_file(terminal_path)
        if quota is not None:
            ledger["latest_authoritative_quota"] = _authoritative_snapshot(
                quota, receipt=quota_receipt
            )
        ledger["events"].append(event)
        ledger["updated_utc"] = _utc()
        _write_gpu_ledger(ledger_path, ledger)
        return event
    finally:
        if owned_lock is not None:
            owned_lock.release()


def bracketed_launch(
    op: str,
    *,
    kaggle: str = "kaggle",
    receipt_dir: Path | str | None = None,
    lock: serial_lock | None = None,
    gpu_reservation_h: float | None = None,
    authenticated_projection_h: float | None = None,
    human_approval_receipt: Path | str | None = None,
    gpu_ledger_path: Path | str = DEFAULT_GPU_LEDGER,
    **kw: Any,
) -> dict[str, Any]:
    """Quota-bracket a mutating op: query, act, query. The delta is the authoritative debit.

    If either quota query fails, the op does not run. Timing your own elapsed seconds is not
    an acceptable substitute -- this campaign already reconciled a 1.65 h accounting drift
    and a 0.74 h launch race that way.
    """
    def _quota() -> tuple[dict[str, Any], dict[str, Any] | None]:
        r = run("quota", kaggle=kaggle, receipt_dir=receipt_dir, **kw)
        if r["returncode"] != 0:
            return r, None
        try:
            return r, parse_authoritative_gpu_quota(json.loads(r["stdout_scrubbed"]))
        except (json.JSONDecodeError, CmuxOpsError):
            return r, None

    owned = lock is None
    lock = lock or serial_lock(node=f"bracketed_launch:{op}", target=str(kw.get("kernel_id") or kw.get("dataset_id") or ""))
    acquired_here = owned or lock._fd is None
    if acquired_here:
        lock.acquire()
    try:
        # The serial lock spans the entire authoritative bracket.  Otherwise another
        # mutation could fall between the pre-reading, action, and post-reading.
        pre_receipt, pre = _quota()
        if pre is None:
            raise CmuxOpsError(
                "quota pre-query failed or was unparseable; refusing to launch without an "
                f"authoritative pre-reading (returncode={pre_receipt['returncode']})"
            )
        reservation = None
        projection_gate = None
        if op == "kernel_push":
            reservation = _reserve_gpu_budget(
                ledger_path=gpu_ledger_path,
                quota=pre,
                kernel_id=str(kw.get("kernel_id", "")),
                max_gpu_h=float(gpu_reservation_h or 0.0),
                quota_receipt=pre_receipt.get("receipt_path"),
            )
            try:
                projection_gate = _projection_gate(
                    authenticated_projection_h, human_approval_receipt
                )
            except Exception:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
                raise
        try:
            action = run(
                op,
                kaggle=kaggle,
                receipt_dir=receipt_dir,
                _held_lock=lock,
                **kw,
            )
        except Exception:
            if reservation is not None:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
            raise
        post_receipt, post = _quota()
        if reservation is not None:
            ledger = _read_gpu_ledger(gpu_ledger_path)
            for item in ledger["active_reservations"]:
                if item.get("kernel_id") == kw.get("kernel_id"):
                    item["status"] = (
                        "LAUNCHED_ACTIVE" if action.get("returncode") == 0
                        else "LAUNCH_FAILED"
                    )
                    item["launch_receipt"] = action.get("receipt_path")
                    if action.get("receipt_sha256"):
                        item["launch_receipt_sha256"] = action["receipt_sha256"]
            if post is not None:
                ledger["latest_authoritative_quota"] = _authoritative_snapshot(
                    post, receipt=post_receipt.get("receipt_path")
                )
            ledger["updated_utc"] = _utc()
            _write_gpu_ledger(gpu_ledger_path, ledger)
            if action.get("returncode") != 0:
                reconcile_gpu_reservation(
                    str(kw.get("kernel_id", "")),
                    terminal_status="ERROR",
                    quota=post,
                    quota_receipt=post_receipt.get("receipt_path"),
                    terminal_receipt=action.get("receipt_path"),
                    ledger_path=gpu_ledger_path,
                    _held_lock=lock,
                )
    finally:
        if acquired_here:
            lock.release()

    delta = None if post is None else round(post["used_h"] - pre["used_h"], 6)

    return {
        "schema": "celltrack-alpha.cmux-bracketed-launch.v1",
        "op": op,
        "quota_pre": pre,
        "quota_post": post,
        "authoritative_debit_h": delta,
        "accounting_basis": "owner-authenticated Kaggle pre/post quota difference",
        "action_receipt": action,
        "quota_pre_receipt": pre_receipt,
        "quota_post_receipt": post_receipt,
        "gpu_reservation_h": gpu_reservation_h if op == "kernel_push" else None,
        "projection_gate": projection_gate if op == "kernel_push" else None,
        "gpu_ledger_path": str(gpu_ledger_path) if op == "kernel_push" else None,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("op", nargs="?", help="op name from command_plan()")
    ap.add_argument("--kaggle", default="kaggle")
    ap.add_argument("--receipt-dir", default=None)
    ap.add_argument("--dataset-id", dest="dataset_id", default="")
    ap.add_argument("--kernel-id", dest="kernel_id", default="")
    ap.add_argument("--kernel-version", dest="kernel_version", type=int, default=None)
    ap.add_argument("--path", default="")
    ap.add_argument("--message", default="")
    ap.add_argument("--competition", default=DEFAULT_COMPETITION)
    ap.add_argument("--csv", default="")
    ap.add_argument("--submission-predicate-receipt", default=None)
    ap.add_argument("--kernel-push-predicate-receipt", default=None)
    ap.add_argument("--bracketed", action="store_true", help="quota-bracket a mutating op")
    ap.add_argument(
        "--gpu-reservation-h",
        type=float,
        default=None,
        help="required worst-case GPU-hour reservation for a bracketed kernel_push",
    )
    ap.add_argument(
        "--authenticated-projection-h",
        type=float,
        default=None,
        help="authenticated whole-run projection for the owner 8 h approval gate",
    )
    ap.add_argument(
        "--human-approval-receipt",
        default=None,
        help="required approval/pitch receipt when the authenticated projection is >= 8 h",
    )
    ap.add_argument("--gpu-ledger", default=str(DEFAULT_GPU_LEDGER))
    ap.add_argument("--print-plan", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    kw = dict(
        dataset_id=a.dataset_id, kernel_id=a.kernel_id, kernel_version=a.kernel_version, path=a.path,
        message=a.message, competition=a.competition, csv=a.csv,
    )
    if a.print_plan:
        print(json.dumps(command_plan(a.kaggle, **kw), indent=2))
        return 0
    if not a.op:
        ap.error("op is required unless --print-plan is given")

    extra = {
        "submission_predicate_receipt": a.submission_predicate_receipt,
        "kernel_push_predicate_receipt": a.kernel_push_predicate_receipt,
    }
    if a.bracketed:
        result = bracketed_launch(
            a.op,
            kaggle=a.kaggle,
            receipt_dir=a.receipt_dir,
            gpu_reservation_h=a.gpu_reservation_h,
            authenticated_projection_h=a.authenticated_projection_h,
            human_approval_receipt=a.human_approval_receipt,
            gpu_ledger_path=a.gpu_ledger,
            **extra,
            **kw,
        )
    else:
        result = run(a.op, kaggle=a.kaggle, receipt_dir=a.receipt_dir, **extra, **kw)
    print(json.dumps(result, indent=2, sort_keys=True))
    rc = result.get("returncode", result.get("action_receipt", {}).get("returncode", 0))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
