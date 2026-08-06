#!/usr/bin/env python3
"""One-shot, version-bound lifecycle for Kaggle code-output submissions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import cmux_ops


CLAIM_DIR = (
    cmux_ops.SHARED_REPO_ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/.submission_claims/code"
)


class SubmissionLifecycleError(RuntimeError):
    """A submission could not satisfy the one-shot lifecycle contract."""


class PredicateConsumed(SubmissionLifecycleError):
    """The predicate already authorized an attempt and cannot be reused."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class CodeSubmissionSpec:
    competition: str
    kernel_id: str
    kernel_version: int
    output_file: str
    message: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.kernel_version, bool)
            or not isinstance(self.kernel_version, int)
            or self.kernel_version <= 0
        ):
            raise ValueError("kernel version must be a positive integer")
        for name, value in (
            ("competition", self.competition),
            ("kernel_id", self.kernel_id),
            ("output_file", self.output_file),
            ("message", self.message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def exact_kernel_ref(self) -> str:
        return f"{self.kernel_id}/{self.kernel_version}"

    def submit_argv(self, kaggle: str) -> list[str]:
        return [
            kaggle,
            "competitions",
            "submit",
            "-c",
            self.competition,
            "-k",
            self.kernel_id,
            "-v",
            str(self.kernel_version),
            "-f",
            self.output_file,
            "-m",
            self.message,
        ]

    def submissions_argv(self, kaggle: str) -> list[str]:
        return [
            kaggle,
            "competitions",
            "submissions",
            self.competition,
            "--page-size",
            "50",
            "--format",
            "json",
        ]


def parse_submission_rows(value: str | Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SubmissionLifecycleError(f"submission listing is not JSON: {exc}") from exc
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise SubmissionLifecycleError("submission listing must be a JSON array")
    rows: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, Mapping):
            raise SubmissionLifecycleError("submission listing contains a non-object row")
        reference = item.get("ref")
        if isinstance(reference, bool) or not isinstance(reference, int) or reference <= 0:
            raise SubmissionLifecycleError("submission listing contains an invalid ref")
        if reference in rows:
            raise SubmissionLifecycleError(f"duplicate submission ref {reference}")
        rows[reference] = dict(item)
    return rows


def classify_submission(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "")).rsplit(".", 1)[-1].upper()
    def score_present(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return value == value and value not in {float("inf"), float("-inf")}
        return isinstance(value, str) and bool(value.strip())

    scored = score_present(row.get("publicScore")) or score_present(
        row.get("privateScore")
    )
    if status == "COMPLETE" and scored:
        return "SCORED_SUCCESS"
    if status == "COMPLETE":
        return "SCORING_ERROR_BLANK_SCORE"
    if status in {"ERROR", "FAILED", "CANCELLED"}:
        return "SUBMISSION_ERROR"
    if status in {"PENDING", "RUNNING"}:
        return "PENDING"
    return "UNKNOWN"


def _run_command(argv: list[str], *, kaggle: str, timeout: float) -> dict[str, Any]:
    started = utc_now()
    started_monotonic = time.monotonic()
    try:
        process = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=cmux_ops.authenticated_kaggle_environment(kaggle),
        )
        process_returncode = process.returncode
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        process_returncode = None
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timed_out = True
    return {
        "argv": argv,
        "started_utc": started,
        "ended_utc": utc_now(),
        "duration_seconds": round(time.monotonic() - started_monotonic, 6),
        "process_returncode": process_returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def _prior_attempt(receipt_dir: Path, predicate_sha256: str) -> Path | None:
    if not receipt_dir.is_dir():
        return None
    for path in sorted(receipt_dir.glob("competition_code_submit_*.json")):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bound = receipt.get("code_submission_predicate", {})
        if isinstance(bound, Mapping) and bound.get("predicate_sha256") == predicate_sha256:
            return path
    return None


def _claim_path(predicate_sha256: str) -> Path:
    """Return the canonical cross-wrapper claim path for one predicate identity."""

    return CLAIM_DIR / f"competition_submit_claim_{predicate_sha256}.json"


def _write_receipt(receipt_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    stamp = receipt["started_utc"].replace(":", "").replace(".", "")
    path = receipt_dir / f"competition_code_submit_{stamp}.json"
    atomic_json(path, receipt)
    receipt["receipt_path"] = str(path.resolve())
    receipt["receipt_sha256"] = sha256_file(path)
    return receipt


PredicateValidator = Callable[..., dict[str, Any]]


def run_code_submission_once(
    *,
    spec: CodeSubmissionSpec,
    predicate_receipt: Path,
    predicate_validator: PredicateValidator,
    receipt_dir: Path,
    receipt_schema: str,
    node: str,
    kaggle: str = "kaggle",
    lock_path: Path = cmux_ops.DEFAULT_LOCK,
    timeout: float = 300.0,
    inventory_poll_attempts: int = 3,
    inventory_poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Submit once, bind exactly one new ref, and consume the predicate forever."""

    receipt_dir = receipt_dir.resolve()
    predicate = predicate_validator(
        predicate_receipt.resolve(), version=spec.kernel_version
    )
    predicate_sha = str(predicate.get("predicate_sha256", ""))
    if len(predicate_sha) != 64:
        raise SubmissionLifecycleError("validated predicate has no SHA-256 identity")

    lock = cmux_ops.serial_lock(
        lock_path, node=node, target=f"{spec.exact_kernel_ref}:{spec.competition}"
    )
    with lock:
        # Revalidate only after acquiring the mutation lock. The exact result
        # must remain stable between the caller's preliminary check and use.
        current = predicate_validator(
            predicate_receipt.resolve(), version=spec.kernel_version
        )
        if current != predicate:
            raise SubmissionLifecycleError("submission predicate changed before locked use")

        prior = _prior_attempt(receipt_dir, predicate_sha)
        if inventory_poll_attempts <= 0 or inventory_poll_interval_s < 0:
            raise ValueError("inventory polling bounds are invalid")
        claim_path = _claim_path(predicate_sha)
        if prior is not None or claim_path.exists():
            evidence = prior or claim_path
            raise PredicateConsumed(f"predicate already consumed by {evidence}")

        pre = _run_command(spec.submissions_argv(kaggle), kaggle=kaggle, timeout=timeout)
        if pre["process_returncode"] != 0 or pre["timed_out"]:
            raise SubmissionLifecycleError("pre-submit submission inventory failed")
        before = parse_submission_rows(pre["stdout"])

        claim = {
            "schema": "celltrack-alpha.kaggle-code-submit-claim.v1",
            "status": "CLAIMED_BEFORE_EXTERNAL_MUTATION",
            "claimed_utc": utc_now(),
            "predicate_sha256": predicate_sha,
            "kernel_ref": spec.exact_kernel_ref,
            "competition": spec.competition,
            "retry_allowed": False,
        }
        receipt_dir.mkdir(parents=True, exist_ok=True)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with claim_path.open("x", encoding="utf-8") as handle:
                json.dump(claim, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise PredicateConsumed(f"predicate already claimed at {claim_path}") from exc

        submit = _run_command(spec.submit_argv(kaggle), kaggle=kaggle, timeout=timeout)
        semantic_error = cmux_ops.semantic_mutation_error(
            "competition_submit", submit["stdout"], submit["stderr"]
        )
        process_returncode = submit["process_returncode"]
        accepted = process_returncode == 0 and semantic_error is None and not submit["timed_out"]

        post = None
        after: dict[int, dict[str, Any]] = {}
        new_refs: list[int] = []
        if accepted:
            for attempt in range(inventory_poll_attempts):
                post = _run_command(
                    spec.submissions_argv(kaggle), kaggle=kaggle, timeout=timeout
                )
                if post["process_returncode"] == 0 and not post["timed_out"]:
                    after = parse_submission_rows(post["stdout"])
                    new_refs = sorted(set(after) - set(before))
                    if new_refs:
                        break
                if attempt + 1 < inventory_poll_attempts:
                    time.sleep(inventory_poll_interval_s)

        bound_ref = new_refs[0] if len(new_refs) == 1 else None
        immediate_state = (
            classify_submission(after[bound_ref]) if bound_ref is not None else None
        )
        if not accepted:
            lifecycle_status = "ATTEMPT_FAILED_NO_RETRY"
        elif bound_ref is None:
            lifecycle_status = "CLI_ACCEPTED_REF_UNRESOLVED_NO_RETRY"
        else:
            lifecycle_status = "CLI_ACCEPTED_REF_BOUND"
        returncode = 0 if lifecycle_status == "CLI_ACCEPTED_REF_BOUND" else 1

        claim.update(
            {
                "status": lifecycle_status,
                "finalized_utc": utc_now(),
                "process_returncode": process_returncode,
                "semantic_failure": semantic_error is not None,
                "submission_ref": bound_ref,
                "immediate_submission_state": immediate_state,
            }
        )
        atomic_json(claim_path, claim)

        receipt = {
            "schema": receipt_schema,
            "op": "competition_code_submit",
            "mutating": True,
            "retry_attempted": False,
            "retry_allowed": False,
            "started_utc": submit["started_utc"],
            "ended_utc": utc_now(),
            "returncode": returncode,
            "process_returncode": process_returncode,
            "timed_out": submit["timed_out"],
            "semantic_failure": semantic_error is not None,
            "semantic_failure_reason": semantic_error,
            "stdout_sha256": hashlib.sha256(submit["stdout"].encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(submit["stderr"].encode()).hexdigest(),
            "stdout_scrubbed": cmux_ops.scrub(submit["stdout"]),
            "stderr_scrubbed": cmux_ops.scrub(submit["stderr"]),
            "serial_lock": str(lock_path.resolve()),
            "lifecycle_status": lifecycle_status,
            "submission_ref": bound_ref,
            "immediate_submission_state": immediate_state,
            "new_submission_refs": new_refs,
            "pre_submission_refs": sorted(before),
            "post_submission_refs": sorted(after),
            "claim_path": str(claim_path),
            "code_submission_predicate": predicate,
            "argv": spec.submit_argv(kaggle),
        }
        return _write_receipt(receipt_dir, receipt)


def reconcile_submission(
    reference: int, rows: str | Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    parsed = parse_submission_rows(rows)
    if reference not in parsed:
        raise SubmissionLifecycleError(f"submission ref {reference} is absent")
    row = parsed[reference]
    return {
        "submission_ref": reference,
        "state": classify_submission(row),
        "status": row.get("status"),
        "public_score": row.get("publicScore"),
        "private_score": row.get("privateScore"),
        "error_description": row.get("errorDescription"),
        "total_bytes": row.get("totalBytes"),
    }
