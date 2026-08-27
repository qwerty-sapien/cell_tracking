#!/usr/bin/env python3
"""Reconcile the sole APNCC transport-smoke kernel version without submitting it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import serial_launch


HERE = Path(__file__).resolve().parent
KERNEL = HERE / "kernel"
CONTRACT = HERE / "CONTRACT.json"
PRELAUNCH = HERE / "PRELAUNCH.json"
PUSH = HERE / "PUSH.json"
TERMINAL = HERE / "TERMINAL.json"
REMOTE_OUTPUT = HERE / "remote_output"
LAUNCH_GUARD = HERE / "LAUNCH_GUARD.json"
KAGGLE = serial_launch.KAGGLE
KERNEL_REF = f"{serial_launch.KERNEL_ID}/1"


class FinalizeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalizeError(f"{path.name} must be a JSON object")
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        rendered = "\n".join(item for item in (result.stdout, result.stderr) if item).strip()
        raise FinalizeError(f"command failed ({result.returncode}): {' '.join(command)}\n{rendered}")
    return result


def one_named_file(root: Path, filename: str) -> Path:
    candidates = [path for path in root.rglob(filename) if path.is_file()]
    if len(candidates) != 1:
        raise FinalizeError(f"expected exactly one downloaded {filename}, found {len(candidates)}")
    return candidates[0]


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def verify_downloaded_output(contract: dict[str, Any]) -> dict[str, Any]:
    expected_output = contract["banked_output"]
    fragments: list[dict[str, Any]] = []
    for expected in expected_output["fragments"]:
        path = one_named_file(REMOTE_OUTPUT, expected["filename"])
        observed = {
            "filename": expected["filename"],
            "path": str(path.relative_to(HERE)),
            "rows": csv_rows(path),
            "sha256": sha256(path),
        }
        if observed["rows"] != expected["rows"] or observed["sha256"] != expected["sha256"]:
            raise FinalizeError(f"CV57-parity fragment mismatch: {expected['filename']}")
        fragments.append(observed)
    submission_expected = expected_output["submission"]
    submission_path = one_named_file(REMOTE_OUTPUT, "submission.csv")
    submission = {
        "path": str(submission_path.relative_to(HERE)),
        "rows": csv_rows(submission_path),
        "sha256": sha256(submission_path),
    }
    if submission["rows"] != submission_expected["rows"] or submission["sha256"] != submission_expected["sha256"]:
        raise FinalizeError("combined submission.csv row count or SHA-256 mismatch")
    return {"fragments": fragments, "submission": submission, "expected": submission_expected}


def attest_step0(contract: dict[str, Any], step0: dict[str, Any]) -> dict[str, Any]:
    if step0.get("status") != "PASS_TARGETED_DECLARED_INPUTS":
        raise FinalizeError(f"remote step 0 did not pass: {step0.get('status')!r}")
    attestation = {
        "status": "PASS",
        "dataset_source": step0.get("dataset_source"),
        "dataset_version": step0.get("dataset_version"),
        "dataset_status": step0.get("dataset_status"),
        "input_manifest_sha256": step0.get("input_manifest_sha256"),
        "checkpoint_sha256": step0.get("checkpoint_sha256"),
        "population_manifest_sha256": step0.get("population_manifest_sha256"),
        "request_count": step0.get("request_count"),
        "request_file_count": step0.get("request_file_count"),
        "population_file_count": step0.get("population_file_count"),
        "wheel_count": step0.get("wheel_count"),
        "competition_path_visits": step0.get("competition_path_visits"),
        "enable_gpu": contract["kernel_properties"]["enable_gpu"],
        "heavy_work_started": step0.get("heavy_work_started"),
        "step0_elapsed_seconds": step0.get("elapsed_seconds"),
    }
    contract["remote_preflight_attestation"] = attestation
    atomic_json(CONTRACT, contract)
    guard = checked([
        sys.executable, str(serial_launch.GUARD), "--kernel-dir", str(KERNEL),
        "--contract", str(CONTRACT), "--registry", str(serial_launch.REGISTRY),
        "--phase", "launch", "--output", str(LAUNCH_GUARD),
    ])
    result = json.loads(guard.stdout)
    if result.get("status") != "PASS":
        raise FinalizeError("remote step-0 launch attestation did not satisfy the shared guard")
    return attestation


def quota_reconciliation() -> dict[str, Any]:
    pre = load_json(PRELAUNCH)["authoritative_quota"]
    post = serial_launch.query_quota()
    return {
        "before": pre,
        "after": post,
        "used_delta_h": round(float(post["used_h"]) - float(pre["used_h"]), 2),
        "remaining_delta_h": round(float(pre["remaining_h"]) - float(post["remaining_h"]), 2),
    }


def terminal_status() -> str:
    status = checked([str(KAGGLE), "kernels", "status", KERNEL_REF])
    rendered = "\n".join(item for item in (status.stdout, status.stderr) if item).strip()
    lower = rendered.lower()
    if not any(token in lower for token in ("complete", "error", "failed", "cancelled")):
        raise FinalizeError("kernel is not terminal yet")
    return rendered


def execute(*, recover_existing_output: bool = False) -> dict[str, Any]:
    if TERMINAL.exists():
        raise FinalizeError("terminal receipt already exists; refusing to reconcile twice")
    if not PRELAUNCH.is_file() or not PUSH.is_file():
        raise FinalizeError("PRELAUNCH.json and PUSH.json are required before reconciliation")
    if REMOTE_OUTPUT.exists() and not recover_existing_output:
        raise FinalizeError("remote_output already exists; refusing to trust a stale download")
    remote_status = terminal_status()
    quota = quota_reconciliation()
    receipt: dict[str, Any] = {
        "schema": "celltrack-alpha.apncc-transport-terminal-reconcile.v2",
        "kernel_ref": KERNEL_REF,
        "kernel_status": remote_status,
        "submission_action": False,
        "quota": quota,
    }
    try:
        if not REMOTE_OUTPUT.exists():
            checked([str(KAGGLE), "kernels", "output", KERNEL_REF, "-p", str(REMOTE_OUTPUT), "--force"])
        else:
            receipt["output_acquisition"] = "read-only recovery of already-downloaded terminal v2 artifacts"
        contract = load_json(CONTRACT)
        step0 = load_json(one_named_file(REMOTE_OUTPUT, "APNCC_STEP0.json"))
        receipt["step0"] = step0
        if step0.get("status") != "PASS_TARGETED_DECLARED_INPUTS":
            raise FinalizeError(f"remote step 0 did not pass: {step0.get('status')!r}")
        transport = load_json(one_named_file(REMOTE_OUTPUT, "APNCC_TRANSPORT_RECEIPT.json"))
        receipt["transport"] = transport
        receipt["remote_preflight_attestation"] = attest_step0(contract, step0)
        if transport.get("status") != "PASS_VALID_SUBMISSION_TRANSPORT_ONLY":
            raise FinalizeError(f"banked runner did not produce a valid transport result: {transport.get('status')!r}")
        receipt["verified_output"] = verify_downloaded_output(contract)
        receipt["status"] = "PASS_VALID_SUBMISSION_TRANSPORT_ONLY"
    except BaseException as error:
        receipt["status"] = "FAIL_TRANSPORT_PATH"
        step0 = receipt.get("step0")
        receipt["failure_layer"] = (
            "remote step-0"
            if isinstance(step0, dict) and step0.get("status") != "PASS_TARGETED_DECLARED_INPUTS"
            else "remote execution/output"
        )
        receipt["error_type"] = type(error).__name__
        receipt["error_message"] = str(error)
    atomic_json(TERMINAL, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="download and reconcile the sole pushed version")
    parser.add_argument(
        "--recover-existing-output", action="store_true",
        help="adopt an already-downloaded terminal v2 artifact set after a supervisor interruption",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing to reconcile without --execute")
    print(json.dumps(execute(recover_existing_output=args.recover_existing_output), indent=2, sort_keys=True))
