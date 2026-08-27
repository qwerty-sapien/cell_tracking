#!/usr/bin/env python3
"""Audit or execute the sole serially authorized CELLECT v7 Kaggle run."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Iterator


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
RUN = HERE / "run_final"
LOCK = ROOT / "reports/g02/screening/tiered_surface/.kaggle_gpu_serial.lock"
BUDGET = ROOT / "reports/g02/screening/tiered_surface/kaggle_budget.json"
KAGGLE = ROOT / ".venv313/bin/kaggle"
KERNEL_ID = "onebitbrain/g02-cellect-r7-closeout-final-v1"
# Every holder of the shared GPU lock must query all live GPU lanes.  The
# APNCC transport smoke uses a historical `bhap-` slug rather than `g02-`.
SHARED_GPU_KERNELS = {
    "onebitbrain/bhap-transport-smoke-v1",
    "onebitbrain/bhap-transport-smoke-v2",
}
AUTHORIZATION_ID = "CELLECT-R7-CLOSEOUT-FINAL-V1"
HARD_LIMIT_SECONDS = 18_000
POLL_SECONDS = 45
EXPECTED_TOTAL_H = 30.0
EXPECTED_POPULATION_SHA256 = (
    "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c"
)
PINS = {
    HERE / "kaggle_cellect_closeout_final.py":
        "2142e7b64aefb4eb9f263f32715c1634dd1bdbf6919b1212e6f2c6f680393386",
    HERE / "kernel-metadata.json":
        "4c595230e7f0ef4489d5a418e3ef62a325e8ff3ba11a5a59cba7aa4e8d26e6bc",
    HERE / "BUILD_RECEIPT.json":
        "2fb840371f495b5f3921e11175bfe16e1b9c911a9dabadd5ebbae1dc1726fcb8",
    HERE / "CONTRACT.json":
        "492ab662f8597ae3cb205f1ae19f764a27601ea5dacd917be57000c82abbfe42",
    HERE / "INLINE_GUARD.json":
        "4b9dd9aada19c16e1118693b1a377e70f80df11dc57ff560886c8b8d888cc025",
    HERE.parent / "PREREGISTRATION_v3_per_class_authenticated_weight.json":
        "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e",
    HERE.parent / "run_v3/B0_SEALED.json":
        "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    HERE.parent / "measured_gate_v4/SEPARATION_GATE_MEASURED_AMENDMENT_V4.json":
        "3526b54b29c6372488f15ef715ace4f76e4744efc1a7031215abc28cef64951d",
}


class LaunchError(RuntimeError):
    """A serial authorization, quota, or remote-state invariant failed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def command_text(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = run(command)
    if result.returncode:
        raise LaunchError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{command_text(result)}"
        )
    return result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def write_once(path: Path, value: object | str) -> None:
    if path.exists():
        raise LaunchError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value if value.endswith("\n") else value + "\n")
    else:
        atomic_json(path, value)


@contextmanager
def exclusive_lock(path: Path = LOCK) -> Iterator[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaunchError("another Kaggle launch supervisor holds the lock") from error
        owner = {
            "pid": os.getpid(),
            "acquired_utc": now(),
            "kernel_id": KERNEL_ID,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(owner, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield owner
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_hours(value: object) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)h", str(value).strip())
    if match is None:
        raise LaunchError(f"invalid quota hours: {value!r}")
    return float(match.group(1))


def parse_gpu_quota(payload: str) -> dict[str, object]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchError("Kaggle quota response is not JSON") from error
    gpu = [row for row in rows if row.get("resource") == "GPU"]
    if len(gpu) != 1:
        raise LaunchError(f"expected one GPU quota row, found {len(gpu)}")
    result = {
        "resource": "GPU",
        "used_h": parse_hours(gpu[0].get("used")),
        "remaining_h": parse_hours(gpu[0].get("remaining")),
        "total_h": parse_hours(gpu[0].get("total")),
        "refresh_at": gpu[0].get("refreshAt"),
        "queried_utc": now(),
        "source": "owner-authenticated Kaggle CLI `kaggle quota --format json`",
    }
    if result["total_h"] != EXPECTED_TOTAL_H:
        raise LaunchError(f"unexpected weekly GPU quota: {result['total_h']}h")
    if round(float(result["used_h"]) + float(result["remaining_h"]), 2) != 30.0:
        raise LaunchError("Kaggle used and remaining GPU hours do not sum to 30")
    return result


def query_gpu_quota() -> dict[str, object]:
    result = checked([str(KAGGLE), "quota", "--format", "json"])
    return parse_gpu_quota(result.stdout)


def reconcile_budget(
    path: Path,
    quota: dict[str, object],
    *,
    phase: str,
    debit_id: str | None = None,
    debit_h: float | None = None,
) -> dict:
    budget = json.loads(path.read_text())
    budget["accounting_authority"] = {
        "canonical_source": "owner-authenticated Kaggle quota",
        "local_file_role": "derived mirror plus named-run audit trail",
        "quota_command": "kaggle quota --format json",
    }
    budget["authoritative_quota"] = dict(quota)
    budget["used_kernel_h"] = quota["used_h"]
    budget["remaining_kernel_h"] = quota["remaining_h"]
    budget["updated_utc"] = quota["queried_utc"]
    history = budget.setdefault("authoritative_quota_history", [])
    history.append(
        {
            "phase": phase,
            "queried_utc": quota["queried_utc"],
            "used_h": quota["used_h"],
            "remaining_h": quota["remaining_h"],
            "source": quota["source"],
        }
    )
    basis = budget.setdefault("authenticated_capacity_basis", {})
    basis["basis"] = "latest owner-authenticated Kaggle quota; local balance is derived"
    basis["conservative_effective_used_gpu_h"] = quota["used_h"]
    basis["conservative_remaining_gpu_h"] = quota["remaining_h"]
    basis["capacity_gate_uses_authoritative_kaggle_quota"] = True
    if debit_id is not None:
        if debit_h is None or debit_h < 0:
            raise LaunchError("postrun debit must be nonnegative")
        if any(row.get("id") == debit_id for row in budget["debits"]):
            raise LaunchError(f"duplicate postrun debit: {debit_id}")
        budget["debits"].append(
            {
                "id": debit_id,
                "kernel_h": debit_h,
                "status": "authoritative_postrun_quota_delta",
                "accounting_basis":
                    "owner-authenticated Kaggle pre/post quota difference",
            }
        )
    atomic_json(path, budget)
    return budget


def parse_kernel_rows(payload: str) -> list[dict]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchError("Kaggle kernels list response is not JSON") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise LaunchError("Kaggle kernels list response has invalid shape")
    return rows


def list_mine(search: str) -> list[dict]:
    result = checked(
        [
            str(KAGGLE),
            "kernels",
            "list",
            "--mine",
            "--search",
            search,
            "--sort-by",
            "dateRun",
            "--page-size",
            "200",
            "--format",
            "json",
        ]
    )
    return parse_kernel_rows(result.stdout)


def exact_target_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("ref") == KERNEL_ID]


def recent_campaign_inflight(
    rows: list[dict],
    status_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run,
) -> list[dict[str, str]]:
    """Query today's G02 kernels and return any nonterminal versions."""
    today = datetime.now(timezone.utc).date().isoformat()
    candidates = {
        str(row["ref"])
        for row in rows
        if (
            # G02 jobs are scoped to today's launch campaign, while the
            # shared APNCC lane can cross UTC midnight under its 360-minute
            # ceiling and must remain fenced until it is terminal.
            str(row.get("ref", "")) in SHARED_GPU_KERNELS
            or (
                str(row.get("ref", "")).startswith("onebitbrain/g02-")
                and str(row.get("lastRunTime", "")).startswith(today)
            )
        )
    }
    inflight = []
    for ref in sorted(candidates):
        status = status_runner([str(KAGGLE), "kernels", "status", ref])
        text = command_text(status)
        lowered = text.lower()
        if status.returncode == 0 and any(
            word in lowered for word in ("running", "queued", "pending")
        ):
            inflight.append({"ref": ref, "status": text})
    return inflight


def parse_pushed_version(text: str) -> int:
    match = re.search(r"\bKernel version\s+([0-9]+)\s+successfully pushed\b", text)
    if match is None:
        raise LaunchError("successful push did not report a kernel version")
    return int(match.group(1))


def classify_terminal(
    remote_status: str, kernel_log: str, *, output_downloaded: bool
) -> str:
    if remote_status == "COMPLETE" and output_downloaded:
        return "FEATURES_COMPLETE_PENDING_FROZEN_GATE"
    del kernel_log
    return "INFRASTRUCTURE_FAILURE_PERMANENT_PARK_C_SCIENTIFIC_UNTESTED"


def assert_version_delta(
    pre_rows: list[dict], post_rows: list[dict], pushed_version: int
) -> dict[str, int]:
    pre_count = len(exact_target_rows(pre_rows))
    post_count = len(exact_target_rows(post_rows))
    if pre_count != 0:
        raise LaunchError("target kernel already exists; one-shot version count is not zero")
    if post_count - pre_count != 1 or pushed_version != 1:
        raise LaunchError(
            "post-push version invariant failed: expected a fresh slug and version 1"
        )
    return {
        "pre_version_count": pre_count,
        "post_version_count": post_count,
        "version_count_delta": post_count - pre_count,
        "pushed_version": pushed_version,
    }


def validate_local_contract(*, require_authorization: bool) -> dict:
    observed = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise LaunchError(f"frozen input hash mismatch: {path}: {actual}")
        observed[str(path.relative_to(ROOT))] = actual
    build = json.loads((HERE / "BUILD_RECEIPT.json").read_text())
    if (
        build.get("status") != "PASS_LOCAL_STEP0_EXPANDED_LAYOUT"
        or build.get("population_manifest_sha256") != EXPECTED_POPULATION_SHA256
        or build.get("request_count") != 1_391_458
        or build.get("request_file_count") != 77
        or build.get("external_bundle_manifest_required") is not False
        or build.get("external_request_manifest_required") is not False
        or build.get("hard_kill_seconds") != HARD_LIMIT_SECONDS
    ):
        raise LaunchError("CELLECT closeout build receipt drift")
    metadata = json.loads((HERE / "kernel-metadata.json").read_text())
    if (
        metadata.get("id") != KERNEL_ID
        or metadata.get("code_file") != "kaggle_cellect_closeout_final.py"
        or metadata.get("enable_gpu") is not True
        or metadata.get("machine_shape") != "NvidiaTeslaT4"
        or metadata.get("enable_internet") is not False
    ):
        raise LaunchError("CELLECT v7 metadata scope drift")
    budget = json.loads(BUDGET.read_text())
    if require_authorization:
        active = [
            row
            for row in budget.get("active_authorizations", [])
            if row.get("id") == AUTHORIZATION_ID
            and row.get("status") == "AUTHORIZED_NOT_YET_LAUNCHED"
        ]
        if len(active) != 1 or len(budget["active_authorizations"]) != 1:
            raise LaunchError("sole serial authorization is absent or ambiguous")
        authorization = active[0]
        if (
            authorization.get("maximum_runs") != 1
            or authorization.get("hard_kill_elapsed_h") != 5.0
            or authorization.get("projection_after_tile_jobs") != 64
            or authorization.get("projection_must_be_strictly_less_than_h") != 5.0
            or authorization.get("infrastructure_failure_disposition")
            != "PERMANENT_PARK_C_SCIENTIFIC_UNTESTED"
        ):
            raise LaunchError("serial authorization limits drift")
    else:
        authorization = None
    return {"pins": observed, "authorization": authorization}


def remote_audit(*, require_authorization: bool) -> dict:
    local = validate_local_contract(require_authorization=require_authorization)
    quota = query_gpu_quota()
    budget = reconcile_budget(BUDGET, quota, phase="prelaunch")
    if quota["remaining_h"] < 5.0:
        raise LaunchError("authoritative Kaggle quota cannot cover the five-hour ceiling")
    campaign_rows = list_mine("g02-")
    shared_lane_rows = list_mine("bhap-transport-smoke")
    target_rows = exact_target_rows(campaign_rows)
    if target_rows:
        raise LaunchError("target kernel already has an existing version; aborting")
    inflight = recent_campaign_inflight(campaign_rows + shared_lane_rows)
    if inflight:
        raise LaunchError(f"another G02 Kaggle kernel is in flight: {inflight}")
    return {
        "schema": "celltrack-alpha.cellect-v7-remote-prelaunch-audit.v1",
        "status": "PASS_EXCLUSIVE_ZERO_EXISTING_VERSION",
        "checked_utc": now(),
        "lock_path": str(LOCK),
        "kernel_id": KERNEL_ID,
        "target_version_count": 0,
        "campaign_inflight": [],
        "authoritative_quota": quota,
        "derived_budget_used_h": budget["used_kernel_h"],
        "derived_budget_remaining_h": budget["remaining_kernel_h"],
        "local_contract": local,
        "scope": "44b6 separation-population endpoint features only",
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "maximum_runs": 1,
        "stop_rule":
            "any infrastructure failure permanently parks CELLECT PARK-C with scientific status UNTESTED",
        "forbidden": [
            "training",
            "full-pool cache",
            "graph",
            "6bba",
            "CV57",
            "submission",
            "RunPod",
        ],
        "_pre_rows": campaign_rows,
    }


def update_authorization_status(status: str) -> None:
    budget = json.loads(BUDGET.read_text())
    active = budget.get("active_authorizations", [])
    matching = [row for row in active if row.get("id") == AUTHORIZATION_ID]
    if len(matching) != 1:
        raise LaunchError("cannot update absent serial authorization")
    matching[0]["status"] = status
    matching[0]["status_updated_utc"] = now()
    atomic_json(BUDGET, budget)


def wait_for_post_rows() -> list[dict]:
    deadline = time.monotonic() + 60
    latest: list[dict] = []
    while time.monotonic() < deadline:
        try:
            latest = list_mine("g02-cellect-r7-separation-v7-one-shot")
        except LaunchError:
            # The CLI can transiently return a non-JSON response immediately
            # after a successful push. Never turn that into another push.
            time.sleep(3)
            continue
        if len(exact_target_rows(latest)) == 1:
            return latest
        time.sleep(3)
    return latest


def resume_pushed_version() -> int:
    """Supervise the already-pushed v1 without issuing another push."""
    if not RUN.exists() or (RUN / "TERMINAL.json").exists():
        raise LaunchError("recoverable pushed run directory is absent or already terminal")
    if (RUN / "PUSH.json").exists():
        raise LaunchError("post-push receipt already exists; refusing ambiguous recovery")
    prelaunch = json.loads((RUN / "PRELAUNCH.json").read_text())
    if (
        prelaunch.get("status") != "PASS_EXCLUSIVE_ZERO_EXISTING_VERSION"
        or prelaunch.get("target_version_count") != 0
        or prelaunch.get("kernel_id") != KERNEL_ID
    ):
        raise LaunchError("prelaunch zero-version receipt drift")
    local = validate_local_contract(require_authorization=False)
    del local
    budget = json.loads(BUDGET.read_text())
    active = budget.get("active_authorizations", [])
    matching = [row for row in active if row.get("id") == AUTHORIZATION_ID]
    if (
        len(active) != 1
        or len(matching) != 1
        or matching[0].get("status") != "LOCK_HELD_PUSHING"
    ):
        raise LaunchError("pushed-run recovery authorization state is ambiguous")
    pushed_version = parse_pushed_version((RUN / "push.stdout.txt").read_text())
    status_probe = run(
        [str(KAGGLE), "kernels", "status", f"{KERNEL_ID}/{pushed_version}"]
    )
    if status_probe.returncode:
        raise LaunchError(
            "the server-assigned pushed version is not status-addressable: "
            + command_text(status_probe)
        )
    version = {
        "pre_version_count": 0,
        "post_version_count": 1,
        "version_count_delta": 1,
        "pushed_version": pushed_version,
    }
    write_once(
        RUN / "PUSH.json",
        {
            "schema": "celltrack-alpha.cellect-v7-push.v1",
            "status": "PASS_EXACTLY_ONE_REMOTE_VERSION_RECOVERED",
            "completed_utc": now(),
            "recovery_reason":
                "post-push list lag; server-assigned v1 is directly status-addressable; no second push issued",
            "direct_status_probe": command_text(status_probe),
            **version,
        },
    )
    update_authorization_status("CONSUMED_ONE_RUN_LOCK_HELD")
    started = time.monotonic()
    push_age = max(0.0, time.time() - (RUN / "push.stdout.txt").stat().st_mtime)
    remaining = max(0.0, HARD_LIMIT_SECONDS - push_age)
    deadline = time.monotonic() + remaining
    remote_status = "UNKNOWN"
    polls = 0
    while time.monotonic() < deadline:
        status = run([str(KAGGLE), "kernels", "status", KERNEL_ID])
        status_text = command_text(status)
        polls += 1
        write_once(
            RUN / f"status_recovery_{polls:04d}.json",
            {
                "observed_utc": now(),
                "returncode": status.returncode,
                "text": status_text,
                "push_age_seconds": push_age + time.monotonic() - started,
            },
        )
        lowered = status_text.lower()
        if status.returncode == 0 and "complete" in lowered:
            remote_status = "COMPLETE"
            break
        if any(word in lowered for word in ("error", "failed", "cancel")):
            remote_status = "ERROR"
            break
        time.sleep(POLL_SECONDS)
    if remote_status == "UNKNOWN":
        remote_status = "HARD_LIMIT_REACHED_SERVER_TIMEOUT_SET"
    logs = run([str(KAGGLE), "kernels", "logs", KERNEL_ID])
    write_once(RUN / "kernel.log", command_text(logs))
    output_returncode = None
    if remote_status == "COMPLETE":
        output = run(
            [
                str(KAGGLE),
                "kernels",
                "output",
                KERNEL_ID,
                "-p",
                str(RUN / "kaggle_output"),
                "-o",
            ]
        )
        output_returncode = output.returncode
        write_once(RUN / "output.stdout.txt", output.stdout)
        write_once(RUN / "output.stderr.txt", output.stderr)
    post_quota = query_gpu_quota()
    pre_quota = prelaunch["authoritative_quota"]
    debit = round(
        max(0.0, float(post_quota["used_h"]) - float(pre_quota["used_h"])), 4
    )
    budget = json.loads(BUDGET.read_text())
    budget["active_authorizations"] = []
    atomic_json(BUDGET, budget)
    reconcile_budget(
        BUDGET,
        post_quota,
        phase="postrun",
        debit_id=AUTHORIZATION_ID,
        debit_h=debit,
    )
    output_downloaded = remote_status == "COMPLETE" and output_returncode == 0
    terminal = {
        "schema": "celltrack-alpha.cellect-v7-terminal.v1",
        "status": remote_status,
        "terminal_utc": now(),
        "elapsed_seconds_since_recovery": time.monotonic() - started,
        "remote_version": pushed_version,
        "version_count_delta": 1,
        "post_push_supervision_recovered": True,
        "second_push_issued": False,
        "poll_count_since_recovery": polls,
        "logs_returncode": logs.returncode,
        "output_returncode": output_returncode,
        "outcome_classification": classify_terminal(
            remote_status, command_text(logs), output_downloaded=output_downloaded
        ),
        "authoritative_pre_quota": pre_quota,
        "authoritative_post_quota": post_quota,
        "authoritative_debit_h": debit,
        "authorization_consumed": True,
        "additional_run_allowed": False,
        "human_escalation_required_if_transport_failure": True,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
    }
    write_once(RUN / "TERMINAL.json", terminal)
    return 0 if remote_status == "COMPLETE" and output_returncode == 0 else 4


def launch() -> int:
    if RUN.exists():
        raise LaunchError(f"fresh one-shot run directory required: {RUN}")
    audit = remote_audit(require_authorization=True)
    pre_rows = audit.pop("_pre_rows")
    RUN.mkdir(parents=True)
    write_once(RUN / "PRELAUNCH.json", audit)
    update_authorization_status("LOCK_HELD_PUSHING")
    started = time.monotonic()
    push = run(
        [
            str(KAGGLE),
            "kernels",
            "push",
            "-p",
            str(HERE),
            "--timeout",
            str(HARD_LIMIT_SECONDS),
            "--accelerator",
            "NvidiaTeslaT4",
        ]
    )
    write_once(RUN / "push.stdout.txt", push.stdout)
    write_once(RUN / "push.stderr.txt", push.stderr)
    if push.returncode:
        update_authorization_status("CONSUMED_PUSH_FAILED_NO_RETRY")
        raise LaunchError(f"Kaggle push failed: {command_text(push)}")
    pushed_version = parse_pushed_version(command_text(push))
    post_rows = wait_for_post_rows()
    version = assert_version_delta(pre_rows, post_rows, pushed_version)
    write_once(
        RUN / "PUSH.json",
        {
            "schema": "celltrack-alpha.cellect-v7-push.v1",
            "status": "PASS_EXACTLY_ONE_REMOTE_VERSION",
            "completed_utc": now(),
            **version,
        },
    )
    update_authorization_status("CONSUMED_ONE_RUN_LOCK_HELD")

    polls = 0
    remote_status = "UNKNOWN"
    while time.monotonic() - started < HARD_LIMIT_SECONDS:
        result = run([str(KAGGLE), "kernels", "status", KERNEL_ID])
        text = command_text(result)
        polls += 1
        write_once(
            RUN / f"status_{polls:04d}.json",
            {
                "observed_utc": now(),
                "elapsed_seconds": time.monotonic() - started,
                "returncode": result.returncode,
                "text": text,
            },
        )
        lowered = text.lower()
        if result.returncode == 0 and "complete" in lowered:
            remote_status = "COMPLETE"
            break
        if any(word in lowered for word in ("error", "failed", "cancel")):
            remote_status = "ERROR"
            break
        time.sleep(POLL_SECONDS)
    if remote_status == "UNKNOWN":
        remote_status = "HARD_LIMIT_REACHED_SERVER_TIMEOUT_SET"

    logs = run([str(KAGGLE), "kernels", "logs", KERNEL_ID])
    write_once(RUN / "kernel.log", command_text(logs))
    output_returncode = None
    if remote_status == "COMPLETE":
        output = run(
            [
                str(KAGGLE),
                "kernels",
                "output",
                KERNEL_ID,
                "-p",
                str(RUN / "kaggle_output"),
                "-o",
            ]
        )
        output_returncode = output.returncode
        write_once(RUN / "output.stdout.txt", output.stdout)
        write_once(RUN / "output.stderr.txt", output.stderr)

    post_quota = query_gpu_quota()
    pre_quota = audit["authoritative_quota"]
    debit = round(
        max(0.0, float(post_quota["used_h"]) - float(pre_quota["used_h"])), 4
    )
    budget = json.loads(BUDGET.read_text())
    budget["active_authorizations"] = []
    atomic_json(BUDGET, budget)
    reconcile_budget(
        BUDGET,
        post_quota,
        phase="postrun",
        debit_id=AUTHORIZATION_ID,
        debit_h=debit,
    )
    output_downloaded = remote_status == "COMPLETE" and output_returncode == 0
    terminal = {
        "schema": "celltrack-alpha.cellect-v7-terminal.v1",
        "status": remote_status,
        "terminal_utc": now(),
        "elapsed_seconds": time.monotonic() - started,
        "remote_version": pushed_version,
        "version_count_delta": 1,
        "poll_count": polls,
        "logs_returncode": logs.returncode,
        "output_returncode": output_returncode,
        "outcome_classification": classify_terminal(
            remote_status, command_text(logs), output_downloaded=output_downloaded
        ),
        "authoritative_pre_quota": pre_quota,
        "authoritative_post_quota": post_quota,
        "authoritative_debit_h": debit,
        "authorization_consumed": True,
        "additional_run_allowed": False,
        "human_escalation_required_if_transport_failure": True,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
    }
    write_once(RUN / "TERMINAL.json", terminal)
    return 0 if remote_status == "COMPLETE" and output_returncode == 0 else 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="consume the one-run authorization; default is read-only remote audit",
    )
    parser.add_argument(
        "--resume-pushed",
        action="store_true",
        help="supervise the evidenced existing v1 after a post-push receipt interruption",
    )
    args = parser.parse_args()
    if args.launch and args.resume_pushed:
        parser.error("--launch and --resume-pushed are mutually exclusive")
    try:
        with exclusive_lock() as owner:
            if args.launch:
                return launch()
            if args.resume_pushed:
                return resume_pushed_version()
            audit = remote_audit(require_authorization=False)
            audit.pop("_pre_rows")
            audit["lock_owner"] = owner
            atomic_json(HERE / "REMOTE_PREFLIGHT_AUDIT.json", audit)
            print(json.dumps(audit, sort_keys=True))
            return 0
    except Exception as error:
        print(f"SERIAL_LAUNCH_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
