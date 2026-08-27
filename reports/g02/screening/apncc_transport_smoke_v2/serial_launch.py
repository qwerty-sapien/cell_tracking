#!/usr/bin/env python3
"""Perform exactly one serialized APNCC transport-smoke kernel push."""

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
from typing import Iterator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KERNEL = HERE / "kernel"
CONTRACT = HERE / "CONTRACT.json"
BUILD = HERE / "BUILD_RECEIPT.json"
REGISTRY = ROOT / "reports/g02/screening/tiered_surface/kaggle_prelaunch/KAGGLE_FAILURE_REGISTRY.json"
GUARD = ROOT / "reports/g02/screening/tiered_surface/kaggle_prelaunch/kaggle_prelaunch_guard.py"
FRESHNESS = ROOT / "reports/g02/screening/tiered_surface/kaggle_prelaunch/check_registry_freshness.py"
LOCK = ROOT / "reports/g02/screening/tiered_surface/.kaggle_gpu_serial.lock"
KAGGLE = ROOT / ".venv313/bin/kaggle"
KERNEL_ID = "onebitbrain/bhap-transport-smoke-v2"
EXPECTED_REMAINING_H = 25.47
POLL_SECONDS = 30.0


class LaunchError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def write_once(path: Path, value: object) -> None:
    if path.exists():
        raise LaunchError(f"refusing to overwrite one-shot receipt: {path}")
    atomic_json(path, value)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def text(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = run(command)
    if result.returncode:
        raise LaunchError(f"command failed ({result.returncode}): {' '.join(command)}\n{text(result)}")
    return result


@contextmanager
def exclusive_lock(path: Path = LOCK) -> Iterator[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaunchError("another Kaggle launch supervisor holds the shared serial lock") from error
        owner = {"pid": os.getpid(), "acquired_utc": now(), "kernel_id": KERNEL_ID}
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
        raise LaunchError(f"invalid quota hour string: {value!r}")
    return float(match.group(1))


def parse_gpu_quota(payload: str) -> dict[str, object]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchError("Kaggle quota response is not JSON") from error
    gpu = [row for row in rows if isinstance(row, dict) and row.get("resource") == "GPU"]
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
    if result["total_h"] != 30.0 or round(float(result["used_h"]) + float(result["remaining_h"]), 2) != 30.0:
        raise LaunchError("unexpected Kaggle GPU quota total")
    return result


def query_quota() -> dict[str, object]:
    return parse_gpu_quota(checked([str(KAGGLE), "quota", "--format", "json"]).stdout)


def parse_kernel_rows(payload: str) -> list[dict]:
    if payload.strip() in ("", "Not found"):
        return []
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchError("Kaggle kernels-list response is not JSON") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise LaunchError("Kaggle kernels-list response has invalid shape")
    return rows


def list_mine(search: str) -> list[dict]:
    result = checked([
        str(KAGGLE), "kernels", "list", "--mine", "--search", search,
        "--sort-by", "dateRun", "--page-size", "200", "--format", "json",
    ])
    return parse_kernel_rows(result.stdout)


def inflight_campaign(rows: list[dict]) -> list[dict[str, str]]:
    inflight: list[dict[str, str]] = []
    for row in rows:
        ref = str(row.get("ref", ""))
        if not (ref.startswith("onebitbrain/g02-") or ref == KERNEL_ID):
            continue
        status = checked([str(KAGGLE), "kernels", "status", ref])
        observed = text(status)
        if any(word in observed.lower() for word in ("running", "queued", "pending")):
            inflight.append({"ref": ref, "status": observed})
    return inflight


def assert_exactly_one_version(pre: list[dict], pushed_version: int) -> dict[str, object]:
    """Prove the server accepted v1 for a slug verified fresh before push.

    Kaggle's status API deliberately reports only the *latest* session: its
    parsed version is not sent in ApiGetKernelSessionStatusRequest.  A v2
    status probe therefore aliases v1 and cannot be used as evidence.
    """
    before = sum(row.get("ref") == KERNEL_ID for row in pre)
    if before != 0 or pushed_version != 1:
        raise LaunchError("target must be a fresh slug with pushed version exactly 1")
    version_one = checked([str(KAGGLE), "kernels", "status", f"{KERNEL_ID}/1"])
    return {
        "pre_version_count": before,
        "version_count_delta": 1,
        "pushed_version": pushed_version,
        "version_one_status": text(version_one),
        "proof_method": "fresh-zero precondition plus Kaggle server push response assigning version 1",
    }


def parse_pushed_version(value: str) -> int:
    match = re.search(r"\bKernel version\s+([0-9]+)\s+successfully pushed\b", value)
    if match is None:
        raise LaunchError("Kaggle did not report a successfully pushed kernel version")
    return int(match.group(1))


def run_inline_launch_guard() -> dict:
    freshness = checked([sys.executable, str(FRESHNESS)])
    guard = checked([
        sys.executable, str(GUARD), "--kernel-dir", str(KERNEL), "--contract", str(CONTRACT),
        "--registry", str(REGISTRY), "--phase", "inline", "--output", str(HERE / "INLINE_LAUNCH_GUARD.json"),
    ])
    try:
        result = json.loads(guard.stdout)
    except json.JSONDecodeError as error:
        raise LaunchError("static guard did not emit JSON") from error
    if result.get("status") != "PASS":
        raise LaunchError("inline launch guard did not pass")
    return {
        "freshness": json.loads(freshness.stdout),
        "guard": result,
        "guard_sha256": sha256(HERE / "INLINE_LAUNCH_GUARD.json"),
        "mode": "inline_step0_before_heavy_work",
    }


def validate_local() -> dict[str, str]:
    contract = json.loads(CONTRACT.read_text())
    build = json.loads(BUILD.read_text())
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    if (
        build.get("status") != "PASS"
        or build.get("banked_identity_sha256") != contract.get("input_manifest_sha256")
        or contract.get("kernel_id") != KERNEL_ID
        or metadata.get("id") != KERNEL_ID
        or metadata.get("dataset_sources") != ["pilkwang/biohub-tracking-support-pack-50ep-v1"]
        or metadata.get("competition_sources") != ["biohub-cell-tracking-during-development"]
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or contract.get("remote_preflight_mode") != "inline_step0_before_heavy_work"
        or sha256(KERNEL / "kaggle_apncc_transport.py") != contract.get("code_sha256")
    ):
        raise LaunchError("local APNCC transport contract drift")
    return {
        "build_sha256": sha256(BUILD),
        "contract_sha256": sha256(CONTRACT),
        "code_sha256": contract["code_sha256"],
        "metadata_sha256": sha256(KERNEL / "kernel-metadata.json"),
    }


def remote_prelaunch(lock_owner: dict[str, object]) -> dict:
    local = validate_local()
    guard = run_inline_launch_guard()
    quota = query_quota()
    if float(quota["remaining_h"]) != EXPECTED_REMAINING_H:
        raise LaunchError(
            "authoritative prelaunch quota changed from the required 25.47 h: %s" % quota["remaining_h"]
        )
    target = list_mine("bhap-transport-smoke-v2")
    if any(row.get("ref") == KERNEL_ID for row in target):
        raise LaunchError("target kernel already exists; refusing a second version")
    campaign = list_mine("g02-")
    # Include this lane in the shared in-flight fence.  This is material after
    # a supervisor crash: the durable remote kernel, not a local lockfile,
    # must still block another campaign's GPU launch.
    inflight = inflight_campaign(campaign + target)
    if inflight:
        raise LaunchError(f"another campaign kernel is in flight: {inflight}")
    return {
        "schema": "celltrack-alpha.apncc-transport-prelaunch.v1",
        "status": "PASS_EXCLUSIVE_ZERO_EXISTING_VERSION",
        "checked_utc": now(),
        "lock_owner": lock_owner,
        "kernel_id": KERNEL_ID,
        "authoritative_quota": quota,
        "target_version_count": 0,
        "campaign_inflight": [],
        "local": local,
        "inline_launch_guard": guard,
        "submission_action": False,
        "_pre_rows": target,
    }


def wait_for_terminal_version() -> dict[str, object]:
    """Hold the shared launch lock until this one version is terminal."""
    transient_errors: list[dict[str, str]] = []
    while True:
        try:
            result = checked([str(KAGGLE), "kernels", "status", f"{KERNEL_ID}/1"])
        except LaunchError as error:
            # A post-push status outage must fail closed: retain the serial
            # lock and retry rather than allowing a second GPU launch.
            transient_errors.append({"observed_utc": now(), "error": str(error)})
            time.sleep(POLL_SECONDS)
            continue
        observed = text(result)
        lowered = observed.lower()
        if any(token in lowered for token in ("complete", "error", "failed", "cancelled")):
            return {
                "kernel_ref": f"{KERNEL_ID}/1",
                "status": observed,
                "terminal_utc": now(),
                "transient_status_errors": transient_errors,
            }
        time.sleep(POLL_SECONDS)


def execute() -> dict:
    with exclusive_lock() as owner:
        prelaunch = remote_prelaunch(owner)
        pre_rows = prelaunch.pop("_pre_rows")
        write_once(HERE / "PRELAUNCH.json", prelaunch)
        pushed = checked([str(KAGGLE), "kernels", "push", "-p", str(KERNEL)])
        pushed_text = text(pushed)
        pending = {
            "schema": "celltrack-alpha.apncc-transport-push-pending.v1",
            "status": "PUSH_ACCEPTED_PENDING_VERSION_PROOF",
            "pushed_utc": now(),
            "kernel_id": KERNEL_ID,
            "expected_version": 1,
            "push_output": pushed_text,
            "submission_action": False,
        }
        # This durable receipt means a process restart still observes the
        # remote APNCC lane as in flight through Kaggle before launching any
        # subsequent GPU work.
        write_once(HERE / "PUSH_PENDING.json", pending)
        recovery = []
        try:
            version = parse_pushed_version(pushed_text)
        except LaunchError as error:
            # Prelaunch proved the slug fresh, hence any successful push can
            # only create v1.  Preserve the parser anomaly for later proof,
            # but never release the lock while that v1 may be running.
            version = 1
            recovery.append({"attempt": 0, "error": str(error)})
        for attempt in range(4):
            try:
                version_delta = assert_exactly_one_version(pre_rows, version)
                break
            except LaunchError as error:
                recovery.append({"attempt": attempt + 1, "error": str(error)})
                time.sleep(1.0)
        else:
            # Do not release the serial lock while a possibly-running remote
            # version exists.  Continue to terminal observation; the final
            # receipt records the failed proof and no second push is attempted.
            version_delta = {"version_count_delta": None, "proof_error": recovery[-1]["error"]}
        receipt = {
            "schema": "celltrack-alpha.apncc-transport-push.v1",
            "status": "PUSHED_EXACTLY_ONE_VERSION",
            "pushed_utc": now(),
            "kernel_id": KERNEL_ID,
            "push_output": pushed_text,
            "version_delta": version_delta,
            "post_list_recovery": recovery,
            "submission_action": False,
        }
        write_once(HERE / "PUSH.json", receipt)
        # Keep the same exclusive lock across execution and reconciliation.
        # Importing here avoids a module cycle; finalize_remote imports the
        # shared Kaggle parsing helpers from this file.
        terminal = wait_for_terminal_version()
        import finalize_remote
        reconciled = finalize_remote.execute()
        return {"push": receipt, "terminal": terminal, "reconciled": reconciled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform the single authorized push")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to push without --execute")
    print(json.dumps(execute(), indent=2, sort_keys=True))
