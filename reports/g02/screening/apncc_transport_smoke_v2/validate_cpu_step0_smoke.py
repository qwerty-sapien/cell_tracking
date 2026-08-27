#!/usr/bin/env python3
"""Verify the recorded APNCC v2 CPU-only Step-0 smoke without relaxing its failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SMOKE = HERE / "CPU_STEP0_SMOKE.json"
READINESS = HERE / "LOCAL_READINESS.json"
CONTRACT = HERE / "CONTRACT.json"
KERNEL = HERE / "kernel/kaggle_apncc_transport.py"
EXPECTED_MISSING = [
    "llvmlite-0.47.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "polars_runtime_32-1.42.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks = {
        "recorded_as_fail_closed_support_inventory": smoke["status"] == "FAIL_CLOSED_LOCAL_SUPPORT_INVENTORY",
        "smoke_kernel_byte_pinned": smoke["kernel"]["sha256"] == sha256(KERNEL) == contract["code_sha256"],
        "smoke_contract_byte_pinned": smoke["contract"]["sha256"] == sha256(CONTRACT),
        "no_kaggle_gpu_submission_or_debit": all(
            smoke["execution"][key] is False
            for key in ("kaggle_action", "gpu_action", "submission_action", "budget_debit", "banked_runner_executed")
        ),
        "cpu_visibility_forced_empty": smoke["execution"]["cuda_visible_devices"] == "",
        "no_heavy_work": smoke["step0"]["heavy_work_started"] is False,
        "expected_step0_fail_closed": (
            smoke["step0"]["status"] == "FAIL_CLOSED"
            and smoke["step0"]["error_type"] == "Step0InputError"
            and smoke["step0"]["error_message"] == "canonical PEP 427 wheel names drift"
        ),
        "exact_missing_wheels_recorded": smoke["support_package"]["wheel_inventory"]["missing_wheels"] == EXPECTED_MISSING,
        "local_readiness_byte_pins_smoke": (
            readiness["cpu_step0_smoke"]["sha256"] == sha256(SMOKE)
            and readiness["cpu_step0_smoke"]["status"] == smoke["status"]
        ),
        "local_readiness_is_not_overstated": readiness["status"] == "LOCAL_BUILD_GUARD_READY_CPU_SMOKE_FAIL_CLOSED_REMOTE_TERMINAL_FAIL",
        "source_has_no_competition_path": "/kaggle/input/competitions/" not in KERNEL.read_text(encoding="utf-8"),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
