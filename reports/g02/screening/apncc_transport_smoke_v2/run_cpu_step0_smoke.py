#!/usr/bin/env python3
"""Execute the generated APNCC v2 Step 0 locally, without its banked runner.

This is deliberately a transport-envelope smoke, not a scientific run.  It
uses the authentic local support package named in the signed attachment,
executes the generated kernel through its complete Step 0, and exits before
the first non-stdlib import or banked-runner invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KERNEL = HERE / "kernel/kaggle_apncc_transport.py"
CONTRACT = HERE / "CONTRACT.json"
LOCAL_SUPPORT = ROOT / "external/Tracking_Support"
BANKED_ATTACHMENT = ROOT / "kaggle_kernel/bhap-submit/bh45att.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label} anchor")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "CPU_STEP0_SMOKE.json")
    parser.add_argument("--support", type=Path, default=LOCAL_SUPPORT)
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    kernel_sha256 = sha256(KERNEL)
    if kernel_sha256 != contract["code_sha256"]:
        raise RuntimeError("kernel byte hash does not match the APNCC v2 contract")
    support = args.support.resolve()
    if not support.is_dir():
        raise RuntimeError(f"authentic local APNCC support package is absent: {support}")
    attachment = json.loads(BANKED_ATTACHMENT.read_text(encoding="utf-8"))
    expected_wheels = {
        item["path"] for item in attachment["identity"]["wheelhouse"]["files"]
    }
    actual_wheels = {path.name for path in (support / "wheels").glob("*.whl")}
    support_inventory = {
        "expected_wheel_count": len(expected_wheels),
        "actual_wheel_count": len(actual_wheels),
        "missing_wheels": sorted(expected_wheels - actual_wheels),
        "unexpected_wheels": sorted(actual_wheels - expected_wheels),
    }

    source = KERNEL.read_text(encoding="utf-8")
    if "/kaggle/input/competitions/" in source:
        raise RuntimeError("Step 0 source contains a forbidden competition-path visit")

    with tempfile.TemporaryDirectory(prefix="apncc-v2-cpu-step0-") as temporary:
        work = Path(temporary) / "working"
        candidate = replace_once(
            source,
            'WORK = Path("/kaggle/working")',
            f"WORK = Path({str(work)!r})",
            "working-directory",
        )
        candidate = replace_once(
            candidate,
            'SUPPORT_MOUNT = Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")',
            f"SUPPORT_MOUNT = Path({str(support)!r})",
            "support-mount",
        )
        candidate = replace_once(
            candidate,
            "STEP0 = run_step0()\n\n\ndef row_count",
            "STEP0 = run_step0()\nraise SystemExit(0)  # CPU smoke terminates before the banked runner.\n\n\ndef row_count",
            "post-Step-0 termination",
        )
        script = Path(temporary) / "cpu_step0_kernel.py"
        script.write_text(candidate, encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        receipt_path = work / "APNCC_STEP0.json"
        if not receipt_path.is_file():
            raise RuntimeError(
                "CPU-only Step 0 did not produce its fail-closed receipt: "
                + (completed.stdout + completed.stderr).strip()
            )
        step0 = json.loads(receipt_path.read_text(encoding="utf-8"))
        passed_step0 = step0.get("status") == "PASS_TARGETED_DECLARED_INPUTS"
        expected_inventory_failure = (
            step0.get("status") == "FAIL_CLOSED"
            and step0.get("error_type") == "Step0InputError"
            and step0.get("error_message") == "canonical PEP 427 wheel names drift"
            and bool(support_inventory["missing_wheels"] or support_inventory["unexpected_wheels"])
        )
        assertions = {
            "step0_passed": passed_step0,
            "complete_declared_input_byte_verification": step0.get("full_declared_byte_sha_verification") is True if passed_step0 else False,
            "all_62_wheels_verified": step0.get("wheel_count") == 62 if passed_step0 else False,
            "parallel_worker_count": step0.get("wheel_hash_workers") == 4 if passed_step0 else False,
            "no_competition_path_visit": step0.get("competition_path_visits") == 0 if passed_step0 else True,
            "no_heavy_work": step0.get("heavy_work_started") is False,
            "warning_envelope_respected": step0.get("elapsed_seconds", float("inf")) < 60.0,
            "banked_runner_not_executed": not (work / "APNCC_TRANSPORT_RECEIPT.json").exists(),
            "expected_inventory_failure_observed": expected_inventory_failure,
            "kernel_process_returned_nonzero_on_fail_closed": completed.returncode != 0 if expected_inventory_failure else completed.returncode == 0,
        }
        if passed_step0:
            status = "PASS_CPU_ONLY_STEP0_NO_RUNNER"
        elif expected_inventory_failure:
            status = "FAIL_CLOSED_LOCAL_SUPPORT_INVENTORY"
        else:
            status = "FAIL_UNEXPECTED_CPU_STEP0"
        result = {
            "schema": "celltrack-alpha.apncc-transport-cpu-step0-smoke.v1",
            "status": status,
            "kernel": {
                "path": "reports/g02/screening/apncc_transport_smoke_v2/kernel/kaggle_apncc_transport.py",
                "sha256": kernel_sha256,
            },
            "contract": {
                "path": "reports/g02/screening/apncc_transport_smoke_v2/CONTRACT.json",
                "sha256": sha256(CONTRACT),
            },
            "execution": {
                "mode": "LOCAL_CPU_ONLY_STEP0",
                "cuda_visible_devices": "",
                "kaggle_action": False,
                "gpu_action": False,
                "submission_action": False,
                "budget_debit": False,
                "banked_runner_executed": False,
            },
            "support_package": {
                "path": str(support),
                "dataset_ref": contract["exclusive_dataset_source"],
                "wheel_inventory": support_inventory,
            },
            "step0": step0,
            "assertions": assertions,
        }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_CPU_ONLY_STEP0_NO_RUNNER":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
