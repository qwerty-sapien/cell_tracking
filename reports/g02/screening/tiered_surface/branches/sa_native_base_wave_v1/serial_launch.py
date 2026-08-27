#!/usr/bin/env python3
"""Launch and supervise exactly one Trackastra native fit105 Kaggle version."""

from __future__ import annotations

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
from typing import Any, Iterator


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kaggle_kernel/g02-trackastra-native-fit-v1"
RUN = HERE / "run_v1"
LOCK = ROOT / "reports/g02/screening/tiered_surface/.kaggle_gpu_serial.lock"
BUDGET = ROOT / "reports/g02/screening/tiered_surface/kaggle_budget.json"
KAGGLE = ROOT / ".venv/bin/kaggle"
KERNEL_ID = "onebitbrain/g02-trackastra-native-fit-v1"
AUTHORIZATION_ID = "TRACKASTRA-NATIVE-BASE-FIT-V1"
HARD_LIMIT_SECONDS = 18_000
POLL_SECONDS = 45
EXPECTED_TOTAL_H = 30.0
PINS = {
    KERNEL_DIR / "g02_trackastra_native_fit_v1.py":
        "57117bf14dee2c3621bfa9c5f6f4d29f48e8e69fa780f6f913164a0650b4b731",
    KERNEL_DIR / "kernel-metadata.json":
        "209bb7b4433f2260d98a7b8de27d0bba502d86da78124cf7958479a0b915ec74",
    HERE / "PREREGISTRATION.md":
        "f1ab1f5b84c21e38ee0470606b79937661dceb1c76ebd166d79e8e731ca6c9c0",
    HERE / "PREREGISTRATION.json":
        "8fec69ec5bd607af82773f1a39b2c3449fd70c720ba78831299099d4f49571ef",
    HERE / "KAGGLE_PRELAUNCH_CONTRACT.json":
        "3db3727f495a8a774b61b7c659a4fa20dc9ee2568c2950a9acdb0babac26eec2",
    HERE / "PRELAUNCH_GUARD_STATIC.json":
        "325925dd9b2e691561b668d290d088a488a686d19ddaceab4bfc4779a9e05f02",
    HERE / "PRELAUNCH_GUARD_INLINE.json":
        "32f2bd8e05c9e6f6185be5fca7d4dd5a5298fb49f34ca3186d08585a0c157b1b",
}


class LaunchError(RuntimeError):
    """A serial, identity, quota, or version invariant failed closed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = run(command)
    if result.returncode:
        raise LaunchError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output(result)}"
        )
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def write_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise LaunchError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload if payload.endswith("\n") else payload + "\n")
    else:
        atomic_json(path, payload)


@contextmanager
def exclusive_lock() -> Iterator[dict[str, Any]]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaunchError("another Kaggle supervisor holds the serial lock") from error
        owner = {
            "pid": os.getpid(),
            "acquired_utc": now(),
            "kernel_id": KERNEL_ID,
            "authorization_id": AUTHORIZATION_ID,
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


def parse_hours(value: Any) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)h", str(value).strip())
    if match is None:
        raise LaunchError(f"invalid quota-hour field: {value!r}")
    return float(match.group(1))


def query_quota() -> dict[str, Any]:
    rows = json.loads(checked([str(KAGGLE), "quota", "--format", "json"]).stdout)
    gpu = [row for row in rows if row.get("resource") == "GPU"]
    if len(gpu) != 1:
        raise LaunchError(f"expected one GPU quota row, found {len(gpu)}")
    quota = {
        "resource": "GPU",
        "used_h": parse_hours(gpu[0].get("used")),
        "remaining_h": parse_hours(gpu[0].get("remaining")),
        "total_h": parse_hours(gpu[0].get("total")),
        "refresh_at": gpu[0].get("refreshAt"),
        "queried_utc": now(),
        "source": "owner-authenticated Kaggle CLI `kaggle quota --format json`",
    }
    if (
        quota["total_h"] != EXPECTED_TOTAL_H
        or round(quota["used_h"] + quota["remaining_h"], 2) != EXPECTED_TOTAL_H
    ):
        raise LaunchError(f"authoritative quota arithmetic drifted: {quota}")
    return quota


def reconcile_budget(
    quota: dict[str, Any], *, phase: str, debit_h: float | None = None
) -> None:
    budget = json.loads(BUDGET.read_text())
    budget["authoritative_quota"] = dict(quota)
    budget["used_kernel_h"] = quota["used_h"]
    budget["remaining_kernel_h"] = quota["remaining_h"]
    budget["updated_utc"] = quota["queried_utc"]
    budget.setdefault("authoritative_quota_history", []).append(
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
    active = [
        row for row in budget.get("active_authorizations", [])
        if row.get("authorization_id") == AUTHORIZATION_ID
    ]
    if len(active) != 1:
        raise LaunchError("native-base authorization is absent or ambiguous")
    active[0]["status"] = "LOCK_HELD_MONITORING" if debit_h is None else "TERMINAL"
    active[0]["status_updated_utc"] = now()
    if debit_h is not None:
        if debit_h < 0 or any(
            row.get("id") == AUTHORIZATION_ID for row in budget.get("debits", [])
        ):
            raise LaunchError("invalid or duplicate native-base quota debit")
        budget.setdefault("debits", []).append(
            {
                "id": AUTHORIZATION_ID,
                "kernel_h": debit_h,
                "status": "authoritative_postrun_quota_delta",
                "accounting_basis": "owner-authenticated Kaggle pre/post quota difference",
            }
        )
        budget["active_authorizations"] = [
            row for row in budget.get("active_authorizations", [])
            if row.get("authorization_id") != AUTHORIZATION_ID
        ]
    atomic_json(BUDGET, budget)


def validate_local() -> dict[str, str]:
    observed = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise LaunchError(f"pinned input drift: {path}: {actual}")
        observed[str(path)] = actual
    for name in ("PRELAUNCH_GUARD_STATIC.json", "PRELAUNCH_GUARD_INLINE.json"):
        guard = json.loads((HERE / name).read_text())
        if guard.get("status") != "PASS" or guard.get("failures") != []:
            raise LaunchError(f"guard is not PASS: {name}")
    metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    contract = json.loads((HERE / "KAGGLE_PRELAUNCH_CONTRACT.json").read_text())
    budget = json.loads(BUDGET.read_text())
    active = [
        row for row in budget.get("active_authorizations", [])
        if row.get("authorization_id") == AUTHORIZATION_ID
        and row.get("status") in {
            "AUTHORIZED_NOT_YET_LAUNCHED",
            "LOCK_HELD_MONITORING",
        }
    ]
    if (
        metadata.get("id") != KERNEL_ID
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("competition_sources") != []
        or metadata.get("dataset_sources") != contract.get("dataset_sources")
        or len(active) != 1
        or len(budget.get("active_authorizations", [])) != 1
    ):
        raise LaunchError("metadata, contract, or one-run authorization drifted")
    return observed


def list_kernels(search: str) -> list[dict[str, Any]]:
    result = checked(
        [
            str(KAGGLE), "kernels", "list", "--mine", "--search", search,
            "--sort-by", "dateRun", "--page-size", "200", "--format", "json",
        ]
    )
    raw = result.stdout.strip()
    if not raw or raw.lower() in {"not found", "no kernels found"}:
        return []
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise LaunchError("kernel listing response is not a list")
    return rows


def exact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("ref") == KERNEL_ID]


def inflight(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    today = datetime.now(timezone.utc).date().isoformat()
    refs = sorted(
        {
            str(row["ref"])
            for row in rows
            if str(row.get("lastRunTime", "")).startswith(today)
        }
    )
    live = []
    for ref in refs:
        status = checked([str(KAGGLE), "kernels", "status", ref])
        observed = output(status)
        if any(word in observed.lower() for word in ("running", "queued", "pending")):
            live.append({"ref": ref, "status": observed})
    return live


def parse_pushed_version(text: str) -> int:
    match = re.search(r"\bKernel version\s+([0-9]+)\s+successfully pushed\b", text)
    if match is None:
        raise LaunchError("successful push output omitted the assigned version")
    return int(match.group(1))


def wait_for_target_listing() -> list[dict[str, Any]]:
    deadline = time.monotonic() + 90
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = list_kernels("g02-trackastra-native-fit-v1")
        if len(exact(rows)) == 1:
            return rows
        time.sleep(3)
    return rows


def launch() -> int:
    retry_after_parser_stop = False
    if RUN.exists():
        existing = sorted(path.name for path in RUN.iterdir())
        if existing != ["PRELAUNCH.json"]:
            raise LaunchError(f"ambiguous pre-existing one-shot receipts: {existing}")
        retry_after_parser_stop = True
    else:
        RUN.mkdir(parents=True)
    pins = validate_local()
    pre_quota = query_quota()
    if pre_quota["remaining_h"] < 5.0:
        raise LaunchError("authoritative remaining quota cannot cover hard limit")
    campaign = list_kernels("g02-")
    if exact(campaign):
        raise LaunchError("target already has a remote version")
    live = inflight(campaign)
    if live:
        raise LaunchError(f"another campaign GPU kernel is in flight: {live}")
    prelaunch = {
        "schema": "celltrack-alpha.trackastra-native-fit-remote-prelaunch.v1",
        "status": "PASS_ZERO_INFLIGHT_ZERO_TARGET_VERSIONS",
        "checked_utc": now(),
        "kernel_id": KERNEL_ID,
        "target_version_count": 0,
        "campaign_inflight": [],
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "authoritative_quota": pre_quota,
        "pins": pins,
    }
    write_once(
        RUN / ("PRELAUNCH_RETRY_AFTER_NOT_FOUND_PARSER.json" if retry_after_parser_stop else "PRELAUNCH.json"),
        prelaunch,
    )
    reconcile_budget(pre_quota, phase="trackastra-native-fit-prelaunch")

    # Recheck the two serial invariants immediately before the sole push.
    if exact(list_kernels("g02-trackastra-native-fit-v1")):
        raise LaunchError("target version appeared before push")
    live = inflight(list_kernels("g02-"))
    if live:
        raise LaunchError(f"GPU lane became busy before push: {live}")
    push_started = time.monotonic()
    push = run(
        [
            str(KAGGLE), "kernels", "push", "-p", str(KERNEL_DIR),
            "--timeout", str(HARD_LIMIT_SECONDS), "--accelerator", "NvidiaTeslaT4",
        ]
    )
    write_once(RUN / "push.stdout.txt", push.stdout)
    write_once(RUN / "push.stderr.txt", push.stderr)
    if push.returncode:
        raise LaunchError(f"Kaggle push failed: {output(push)}")
    pushed_version = parse_pushed_version(output(push))
    post_rows = wait_for_target_listing()
    if pushed_version != 1 or len(exact(post_rows)) != 1:
        raise LaunchError("serial version invariant failed: expected fresh v1")
    write_once(
        RUN / "PUSH.json",
        {
            "schema": "celltrack-alpha.trackastra-native-fit-push.v1",
            "status": "PASS_EXACTLY_ONE_REMOTE_VERSION",
            "completed_utc": now(),
            "pre_version_count": 0,
            "post_version_count": 1,
            "version_count_delta": 1,
            "pushed_version": pushed_version,
        },
    )
    print(json.dumps({"LAUNCHED": KERNEL_ID, "version": 1}, sort_keys=True), flush=True)

    remote_status = "UNKNOWN"
    polls = 0
    while time.monotonic() - push_started < HARD_LIMIT_SECONDS + 120:
        status = run([str(KAGGLE), "kernels", "status", KERNEL_ID])
        observed = output(status)
        polls += 1
        atomic_json(
            RUN / "LATEST_STATUS.json",
            {
                "observed_utc": now(),
                "elapsed_seconds": time.monotonic() - push_started,
                "poll": polls,
                "returncode": status.returncode,
                "text": observed,
            },
        )
        print(
            json.dumps(
                {
                    "poll": polls,
                    "elapsed_seconds": round(time.monotonic() - push_started, 3),
                    "status": observed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        lowered = observed.lower()
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
    write_once(RUN / "kernel.log", output(logs))
    fetched = run(
        [str(KAGGLE), "kernels", "output", KERNEL_ID, "-p", str(RUN / "kaggle_output"), "-o"]
    )
    write_once(RUN / "output.stdout.txt", fetched.stdout)
    write_once(RUN / "output.stderr.txt", fetched.stderr)
    post_quota = query_quota()
    debit = round(max(0.0, post_quota["used_h"] - pre_quota["used_h"]), 4)
    reconcile_budget(post_quota, phase="trackastra-native-fit-postrun", debit_h=debit)
    report_path = RUN / "kaggle_output/TRACKASTRA_NATIVE_FIT_REPORT.json"
    terminal = {
        "schema": "celltrack-alpha.trackastra-native-fit-terminal.v1",
        "status": remote_status,
        "outcome": "OUTPUT_READY" if report_path.is_file() else "OUTPUT_MISSING_FAIL_CLOSED",
        "terminal_utc": now(),
        "elapsed_seconds": time.monotonic() - push_started,
        "remote_version": pushed_version,
        "version_count_delta": 1,
        "poll_count": polls,
        "logs_returncode": logs.returncode,
        "output_returncode": fetched.returncode,
        "authoritative_pre_quota": pre_quota,
        "authoritative_post_quota": post_quota,
        "authoritative_debit_h": debit,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "additional_run_authorized": False,
        "submission": False,
    }
    write_once(RUN / "TERMINAL.json", terminal)
    print(json.dumps({"TERMINAL": terminal}, sort_keys=True), flush=True)
    return 0 if remote_status == "COMPLETE" and fetched.returncode == 0 else 4


def main() -> int:
    try:
        with exclusive_lock():
            return launch()
    except Exception as error:
        print(f"SERIAL_LAUNCH_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
