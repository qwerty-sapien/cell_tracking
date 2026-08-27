#!/usr/bin/env python3
"""Provision and supervise the one authorized §6.2 PEFT Kaggle version."""

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
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kaggle_kernel/g02-peft-delete-v1"
ASSET_BUILD = Path("/private/tmp/g02-peft-delete-assets-v1-final-20260731")
UPLOAD_STAGE = Path("/private/tmp/g02-peft-delete-assets-v1-upload-20260731")
RUN = HERE / "run_v1"
LOCK = ROOT / "reports/g02/screening/tiered_surface/.kaggle_gpu_serial.lock"
BUDGET = ROOT / "reports/g02/screening/tiered_surface/kaggle_budget.json"
KAGGLE = ROOT / ".venv313/bin/kaggle"
REGISTRY = (
    ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
    "KAGGLE_FAILURE_REGISTRY.json"
)
GUARD = (
    ROOT
    / "reports/g02/screening/tiered_surface/kaggle_prelaunch/"
    "kaggle_prelaunch_guard.py"
)
CONTRACT = HERE / "KAGGLE_PRELAUNCH_CONTRACT.json"
AUTHORIZATION = HERE / "AUTHORIZATION.json"
KERNEL_ID = "onebitbrain/g02-peft-delete-v1"
DATASET_ID = "onebitbrain/g02-peft-delete-assets-v1"
AUTHORIZATION_ID = "PEFT-SECTION-6P2-DELETE-V1"
HARD_LIMIT_SECONDS = 18_000
POLL_SECONDS = 45
EXPECTED_TOTAL_H = 30.0
PROJECTED_T4_H = 2.25
CALIBRATION_MAX_T4_H = 3.0
EXPECTED_ASSET_MANIFEST_SHA256 = (
    "197ebbeb8fa5530ebe1df5b9e1888a34060ff8d4f897c24fcf24da3748e696e8"
)
EXPECTED_TAR_SHA256 = (
    "326a05a62be6afb1cbe9ab6307602e6132ae7aab9d5c75976582123edf8593b2"
)
EXPECTED_TAR_BYTES = 28_180_480
PINS = {
    KERNEL_DIR / "g02_peft_delete_v1.py":
        "f84f4622b280de4396a3f5d0f85523770225d97c03b0376fbb44b2823d7be9d0",
    KERNEL_DIR / "kernel-metadata.json":
        "4ce618ed5c2227b664dae1285c7da352b44033379a2c6cdda3128dc273255ca4",
    HERE / "RUNTIME_MARGIN_AMENDMENT_20260731.json":
        "a5b72dbcb7eb02b338fda004213b75076998228a2126be47950503522651093a",
    HERE / "REPROJECTION_20260731.json":
        "b99cee76da298a8543016b5a9f3d77f1f93c1cc4d0d02faedfc60f017fd7019a",
    CONTRACT:
        "32463024fd6b33055989a33e1b3f742550ce9b41c132751e14422f892c6a601c",
    HERE / "PRELAUNCH_GUARD_STATIC.json":
        "c642d20cfb026aaebfc0c7dc2a387b804d503d1967b2d15b0be71fa996123300",
    HERE / "PRELAUNCH_GUARD_INLINE.json":
        "f5ba90b054b2e813e148a6ce51ceb666aca2db6324fac5d9804e9ad6f58bf228",
    ASSET_BUILD / "asset_manifest.json": EXPECTED_ASSET_MANIFEST_SHA256,
    ASSET_BUILD / "peft_bundle.tar": EXPECTED_TAR_SHA256,
    ASSET_BUILD / "BUILD_RECEIPT.json":
        "46926dea159fe33d417cdeb79d1bb6a7a52627fc0b61f165f1e2802ae6d86467",
}
SHARED_GPU_KERNELS = {
    "onebitbrain/bhap-transport-smoke-v1",
    "onebitbrain/bhap-transport-smoke-v2",
}


class LaunchError(RuntimeError):
    """A serial, identity, quota, or remote-state gate failed closed."""


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


def text(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part for part in (result.stdout, result.stderr) if part
    ).strip()


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = run(command)
    if result.returncode:
        raise LaunchError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{text(result)}"
        )
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def write_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise LaunchError(f"refusing to overwrite receipt: {path}")
    if isinstance(payload, str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            payload if payload.endswith("\n") else payload + "\n",
            encoding="utf-8",
        )
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
        or round(quota["used_h"] + quota["remaining_h"], 2)
        != EXPECTED_TOTAL_H
    ):
        raise LaunchError(f"authoritative quota arithmetic drifted: {quota}")
    return quota


def reconcile_budget(
    quota: dict[str, Any],
    *,
    phase: str,
    debit_h: float | None = None,
) -> None:
    budget = json.loads(BUDGET.read_text())
    budget["accounting_authority"] = {
        "canonical_source": "owner-authenticated Kaggle quota",
        "local_file_role": "derived mirror plus named-run audit trail",
        "quota_command": "kaggle quota --format json",
    }
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
    if debit_h is not None:
        if debit_h < 0:
            raise LaunchError("postrun quota debit cannot be negative")
        if any(
            row.get("id") == AUTHORIZATION_ID
            for row in budget.get("debits", [])
        ):
            raise LaunchError("duplicate PEFT quota debit")
        budget.setdefault("debits", []).append(
            {
                "id": AUTHORIZATION_ID,
                "kernel_h": debit_h,
                "status": "authoritative_postrun_quota_delta",
                "accounting_basis":
                    "owner-authenticated Kaggle pre/post quota difference",
            }
        )
        budget["active_authorizations"] = []
    atomic_json(BUDGET, budget)


def update_authorization(status: str) -> None:
    budget = json.loads(BUDGET.read_text())
    matches = [
        row
        for row in budget.get("active_authorizations", [])
        if row.get("authorization_id") == AUTHORIZATION_ID
    ]
    if len(matches) != 1:
        raise LaunchError("PEFT serial authorization is absent or ambiguous")
    matches[0]["status"] = status
    matches[0]["status_updated_utc"] = now()
    atomic_json(BUDGET, budget)


def validate_local(*, require_authorization: bool) -> dict[str, Any]:
    observed = {}
    for path, expected in PINS.items():
        actual = sha256(path)
        if actual != expected:
            raise LaunchError(f"pinned local input drift: {path}: {actual}")
        observed[str(path)] = actual
    if (ASSET_BUILD / "peft_bundle.tar").stat().st_size != EXPECTED_TAR_BYTES:
        raise LaunchError("asset tar byte count drifted")
    amendment = json.loads(
        (HERE / "RUNTIME_MARGIN_AMENDMENT_20260731.json").read_text()
    )
    reprojection = json.loads((HERE / "REPROJECTION_20260731.json").read_text())
    if (
        amendment["runtime_budget"]["projected_t4_h"] != PROJECTED_T4_H
        or reprojection["conservative_projection"]["projected_t4_h"]
        != PROJECTED_T4_H
        or reprojection["conservative_projection"]["target_t4_h_max"]
        != CALIBRATION_MAX_T4_H
        or PROJECTED_T4_H > CALIBRATION_MAX_T4_H
        or amendment["runtime_budget"]["hard_kill_t4_h"] != 5.0
        or amendment["data"]["evaluation"]["population"] != 1_345_497
        or amendment["data"]["training"]["sample_total"] != 112_918
    ):
        raise LaunchError("survivable runtime/evaluation amendment drifted")
    for receipt_name in ("PRELAUNCH_GUARD_STATIC.json", "PRELAUNCH_GUARD_INLINE.json"):
        receipt = json.loads((HERE / receipt_name).read_text())
        if receipt.get("status") != "PASS" or receipt.get("failures") != []:
            raise LaunchError(f"prelaunch guard is not PASS: {receipt_name}")
    metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    if (
        metadata.get("id") != KERNEL_ID
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
        or metadata.get("dataset_sources") != [DATASET_ID]
        or metadata.get("competition_sources") != []
    ):
        raise LaunchError("kernel metadata scope drifted")
    if require_authorization:
        authorization = json.loads(AUTHORIZATION.read_text())
        budget = json.loads(BUDGET.read_text())
        active = [
            row
            for row in budget.get("active_authorizations", [])
            if row.get("authorization_id") == AUTHORIZATION_ID
            and row.get("status") == "AUTHORIZED_NOT_YET_LAUNCHED"
        ]
        if (
            authorization.get("status") != "AUTHORIZED_NOT_YET_LAUNCHED"
            or authorization.get("maximum_remote_kernel_versions") != 1
            or authorization.get("projected_t4_h") != PROJECTED_T4_H
            or authorization.get("hard_kill_t4_h") != 5.0
            or len(active) != 1
            or len(budget.get("active_authorizations", [])) != 1
        ):
            raise LaunchError("one-run PEFT authorization drifted")
    return {"pins": observed, "projected_t4_h": PROJECTED_T4_H}


def json_rows(result: subprocess.CompletedProcess[str], label: str) -> list[dict]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise LaunchError(f"{label} response is not JSON") from error
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise LaunchError(f"{label} response shape drifted")
    return payload


def list_kernels(search: str) -> list[dict]:
    return json_rows(
        checked(
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
        ),
        "kernel listing",
    )


def exact_kernel_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("ref") == KERNEL_ID]


def inflight(rows: list[dict]) -> list[dict[str, str]]:
    today = datetime.now(timezone.utc).date().isoformat()
    candidates = {
        str(row["ref"])
        for row in rows
        if (
            str(row.get("ref", "")) in SHARED_GPU_KERNELS
            or (
                str(row.get("ref", "")).startswith("onebitbrain/g02-")
                and str(row.get("lastRunTime", "")).startswith(today)
            )
        )
    }
    live = []
    for ref in sorted(candidates):
        result = run([str(KAGGLE), "kernels", "status", ref])
        observed = text(result)
        if result.returncode == 0 and any(
            word in observed.lower() for word in ("running", "queued", "pending")
        ):
            live.append({"ref": ref, "status": observed})
    return live


def exact_dataset_rows() -> list[dict]:
    result = checked(
        [
            str(KAGGLE),
            "datasets",
            "list",
            "--mine",
            "--search",
            DATASET_ID.split("/", 1)[1],
            "--format",
            "json",
        ]
    )
    if result.stdout.strip() == "No datasets found":
        rows = []
    else:
        rows = json_rows(result, "dataset listing")
    return [
        row
        for row in rows
        if row.get("ref") == DATASET_ID
        or row.get("id") == DATASET_ID
    ]


def dataset_status() -> dict[str, Any]:
    result = checked(
        [str(KAGGLE), "datasets", "status", DATASET_ID, "--format", "json"]
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise LaunchError("dataset status is not a JSON object")
    return payload


def status_version(payload: dict[str, Any]) -> int:
    for key in (
        "currentVersionNumber",
        "current_version_number",
        "versionNumber",
        "version_number",
    ):
        if key in payload:
            return int(payload[key])
    raise LaunchError(f"dataset status omits current version: {payload}")


def wait_dataset_ready() -> dict[str, Any]:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        payload = dataset_status()
        status = str(payload.get("status", "")).lower()
        if status == "ready":
            if status_version(payload) != 1:
                raise LaunchError("fresh PEFT dataset is not version 1")
            return payload
        if any(word in status for word in ("error", "failed")):
            raise LaunchError(f"dataset processing failed: {payload}")
        print(json.dumps({"dataset_status": payload}, sort_keys=True), flush=True)
        time.sleep(10)
    raise LaunchError("dataset did not become ready within ten minutes")


def verify_dataset_files() -> list[dict]:
    rows = json_rows(
        checked(
            [
                str(KAGGLE),
                "datasets",
                "files",
                DATASET_ID,
                "--page-size",
                "200",
                "--format",
                "json",
            ]
        ),
        "dataset file listing",
    )
    matching = [
        row
        for row in rows
        if row.get("name") == "peft_bundle.tar"
        or row.get("ref") == "peft_bundle.tar"
    ]
    if len(matching) != 1:
        raise LaunchError(f"uploaded tar inventory drifted: {rows}")
    size = matching[0].get("totalBytes", matching[0].get("size"))
    if int(size) != EXPECTED_TAR_BYTES:
        raise LaunchError(f"uploaded tar byte count drifted: {matching[0]}")
    return rows


def provision_dataset() -> dict[str, Any]:
    status_403_receipt = RUN / "DATASET_CREATE_STATUS_403.json"
    if status_403_receipt.is_file():
        receipt = json.loads(status_403_receipt.read_text())
        if (
            receipt.get("create_returncode") != 0
            or receipt.get("fresh_slug_precreate_listing_count") != 0
            or receipt.get("inferred_version") != 1
            or receipt.get("staged_transport", {}).get("sha256")
            != EXPECTED_TAR_SHA256
            or receipt.get("staged_transport", {}).get("bytes")
            != EXPECTED_TAR_BYTES
        ):
            raise LaunchError("private-dataset create/403 receipt drifted")
        return {
            "schema": "celltrack-alpha.peft-delete-dataset-provision.v1",
            "status": "CREATE_ACCEPTED_INLINE_REMOTE_ATTESTATION_REQUIRED",
            "dataset": DATASET_ID,
            "created_in_this_launch": True,
            "inferred_fresh_version": 1,
            "remote_status_endpoint": "HTTP_403_DO_NOT_RETRY",
            "transport_sha256": EXPECTED_TAR_SHA256,
            "transport_bytes": EXPECTED_TAR_BYTES,
            "member_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
            "same_kernel_inline_step0_required": True,
        }
    existing = exact_dataset_rows()
    created = False
    if len(existing) > 1:
        raise LaunchError("ambiguous PEFT dataset listing")
    if not existing:
        if UPLOAD_STAGE.exists():
            contents = sorted(path.name for path in UPLOAD_STAGE.iterdir())
            if contents != ["dataset-metadata.json", "peft_bundle.tar"]:
                raise LaunchError(f"upload staging directory drifted: {contents}")
        else:
            UPLOAD_STAGE.mkdir(parents=True)
            shutil.copyfile(
                ASSET_BUILD / "dataset-metadata.json",
                UPLOAD_STAGE / "dataset-metadata.json",
            )
            shutil.copyfile(
                ASSET_BUILD / "peft_bundle.tar",
                UPLOAD_STAGE / "peft_bundle.tar",
            )
        if (
            sha256(UPLOAD_STAGE / "peft_bundle.tar") != EXPECTED_TAR_SHA256
            or (UPLOAD_STAGE / "peft_bundle.tar").stat().st_size
            != EXPECTED_TAR_BYTES
        ):
            raise LaunchError("upload-staged tar identity drifted")
        create = checked(
            [
                str(KAGGLE),
                "datasets",
                "create",
                "-p",
                str(UPLOAD_STAGE),
                "-r",
                "skip",
            ]
        )
        created = True
        create_text = text(create)
    else:
        create_text = "pre-existing fresh lane dataset reused after exact validation"
    status = wait_dataset_ready()
    files = verify_dataset_files()
    return {
        "schema": "celltrack-alpha.peft-delete-dataset-provision.v1",
        "status": "PASS_READY_VERSION_1",
        "dataset": DATASET_ID,
        "created_in_this_launch": created,
        "create_output": create_text,
        "remote_status": status,
        "remote_files": files,
        "transport_sha256": EXPECTED_TAR_SHA256,
        "transport_bytes": EXPECTED_TAR_BYTES,
        "member_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
    }


def guard_again(phase: str) -> dict[str, Any]:
    result = checked(
        [
            str(ROOT / ".venv313/bin/python"),
            "-B",
            str(GUARD),
            "--kernel-dir",
            str(KERNEL_DIR),
            "--contract",
            str(CONTRACT),
            "--registry",
            str(REGISTRY),
            "--phase",
            phase,
        ]
    )
    payload = json.loads(result.stdout)
    if payload.get("status") != "PASS":
        raise LaunchError(f"{phase} guard failed: {payload}")
    return payload


def prelaunch(*, require_authorization: bool) -> dict[str, Any]:
    local = validate_local(require_authorization=require_authorization)
    quota = query_quota()
    reconcile_budget(quota, phase="peft-prelaunch")
    if quota["remaining_h"] < 5.0:
        raise LaunchError("authoritative quota cannot cover the five-hour hard limit")
    if PROJECTED_T4_H > CALIBRATION_MAX_T4_H:
        raise LaunchError("projection lacks the authorized <=3-hour margin")
    campaign = list_kernels("g02-")
    shared = list_kernels("bhap-transport-smoke")
    target = exact_kernel_rows(campaign)
    if target:
        raise LaunchError("PEFT target already has a remote kernel version")
    live = inflight(campaign + shared)
    if live:
        raise LaunchError(f"another campaign GPU kernel is in flight: {live}")
    return {
        "schema": "celltrack-alpha.peft-delete-remote-prelaunch.v1",
        "status": "PASS_ZERO_INFLIGHT_ZERO_TARGET_VERSIONS",
        "checked_utc": now(),
        "kernel_id": KERNEL_ID,
        "dataset_id": DATASET_ID,
        "target_version_count": 0,
        "campaign_inflight": [],
        "projected_t4_h": PROJECTED_T4_H,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "authoritative_quota": quota,
        "local": local,
        "static_guard": guard_again("static"),
        "inline_guard": guard_again("inline"),
        "_campaign_rows": campaign,
    }


def parse_pushed_version(output: str) -> int:
    match = re.search(r"\bKernel version\s+([0-9]+)\s+successfully pushed\b", output)
    if match is None:
        raise LaunchError("successful push output omitted the assigned version")
    return int(match.group(1))


def wait_post_version() -> list[dict]:
    deadline = time.monotonic() + 90
    latest: list[dict] = []
    while time.monotonic() < deadline:
        try:
            latest = list_kernels("g02-peft-delete-v1")
        except LaunchError:
            time.sleep(3)
            continue
        if len(exact_kernel_rows(latest)) == 1:
            return latest
        time.sleep(3)
    return latest


def launch() -> int:
    audit = prelaunch(require_authorization=True)
    pre_rows = audit.pop("_campaign_rows")
    if RUN.exists():
        existing = sorted(path.name for path in RUN.iterdir())
        permitted = {
            "PRELAUNCH.json",
            "PRELAUNCH_RETRY_AFTER_CLI_EMPTY_LIST.json",
            "DATASET_CREATE_STATUS_403.json",
        }
        if not set(existing).issubset(permitted):
            raise LaunchError(
                f"ambiguous pre-existing one-shot run directory: {existing}"
            )
        retry_path = RUN / "PRELAUNCH_RETRY_AFTER_PRIVATE_403.json"
        write_once(retry_path, audit)
    else:
        RUN.mkdir(parents=True)
        write_once(RUN / "PRELAUNCH.json", audit)
    dataset = provision_dataset()
    write_once(RUN / "DATASET.json", dataset)

    # Dataset provisioning can take time. Recheck both serial conditions and
    # local byte identities immediately before the sole kernel push.
    validate_local(require_authorization=True)
    campaign = list_kernels("g02-")
    shared = list_kernels("bhap-transport-smoke")
    if exact_kernel_rows(campaign):
        raise LaunchError("target version appeared during dataset provisioning")
    live = inflight(campaign + shared)
    if live:
        raise LaunchError(f"GPU lane became busy before push: {live}")
    update_authorization("LOCK_HELD_PUSHING")
    push_started = time.monotonic()
    push = run(
        [
            str(KAGGLE),
            "kernels",
            "push",
            "-p",
            str(KERNEL_DIR),
            "--timeout",
            str(HARD_LIMIT_SECONDS),
            "--accelerator",
            "NvidiaTeslaT4",
        ]
    )
    write_once(RUN / "push.stdout.txt", push.stdout)
    write_once(RUN / "push.stderr.txt", push.stderr)
    if push.returncode:
        update_authorization("CONSUMED_PUSH_FAILED_NO_RETRY_WITHOUT_REVIEW")
        raise LaunchError(f"Kaggle push failed: {text(push)}")
    pushed_version = parse_pushed_version(text(push))
    post_rows = wait_post_version()
    pre_count = len(exact_kernel_rows(pre_rows))
    post_count = len(exact_kernel_rows(post_rows))
    if pre_count != 0 or post_count - pre_count != 1 or pushed_version != 1:
        raise LaunchError(
            "serial version invariant failed: expected fresh v1 and delta exactly one"
        )
    write_once(
        RUN / "PUSH.json",
        {
            "schema": "celltrack-alpha.peft-delete-push.v1",
            "status": "PASS_EXACTLY_ONE_REMOTE_VERSION",
            "completed_utc": now(),
            "pre_version_count": pre_count,
            "post_version_count": post_count,
            "version_count_delta": post_count - pre_count,
            "pushed_version": pushed_version,
        },
    )
    update_authorization("CONSUMED_ONE_RUN_LOCK_HELD")
    print(
        json.dumps(
            {
                "LAUNCHED": KERNEL_ID,
                "version": pushed_version,
                "version_delta": 1,
                "authoritative_pre_quota": audit["authoritative_quota"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    remote_status = "UNKNOWN"
    polls = 0
    while time.monotonic() - push_started < HARD_LIMIT_SECONDS:
        status = run([str(KAGGLE), "kernels", "status", KERNEL_ID])
        observed = text(status)
        polls += 1
        write_once(
            RUN / f"status_{polls:04d}.json",
            {
                "observed_utc": now(),
                "elapsed_seconds": time.monotonic() - push_started,
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
    write_once(RUN / "kernel.log", text(logs))
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

    post_quota = query_quota()
    pre_quota = audit["authoritative_quota"]
    debit = round(
        max(0.0, float(post_quota["used_h"]) - float(pre_quota["used_h"])),
        4,
    )
    reconcile_budget(post_quota, phase="peft-postrun", debit_h=debit)
    result_path = RUN / "kaggle_output/PEFT_RESULT.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if result is not None:
        outcome = result.get("status")
    elif remote_status == "COMPLETE":
        outcome = "OUTPUT_MISSING_FAIL_CLOSED"
    else:
        outcome = "INFRASTRUCTURE_OR_RUNTIME_FAILURE"
    terminal = {
        "schema": "celltrack-alpha.peft-delete-terminal.v1",
        "status": remote_status,
        "outcome": outcome,
        "terminal_utc": now(),
        "elapsed_seconds": time.monotonic() - push_started,
        "remote_version": pushed_version,
        "version_count_delta": 1,
        "poll_count": polls,
        "logs_returncode": logs.returncode,
        "output_returncode": output_returncode,
        "authoritative_pre_quota": pre_quota,
        "authoritative_post_quota": post_quota,
        "authoritative_debit_h": debit,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "additional_run_authorized": False,
        "submission": False,
    }
    write_once(RUN / "TERMINAL.json", terminal)
    print(json.dumps({"TERMINAL": terminal}, sort_keys=True), flush=True)
    return (
        0
        if remote_status == "COMPLETE"
        and output_returncode == 0
        and result is not None
        else 4
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="consume the one-version authorization; default is read-only audit",
    )
    args = parser.parse_args()
    try:
        with exclusive_lock() as owner:
            if args.launch:
                return launch()
            audit = prelaunch(require_authorization=False)
            audit.pop("_campaign_rows")
            audit["lock_owner"] = owner
            atomic_json(HERE / "REMOTE_PREFLIGHT_AUDIT.json", audit)
            print(json.dumps(audit, sort_keys=True), flush=True)
            return 0
    except Exception as error:
        print(
            f"SERIAL_LAUNCH_FAIL: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
