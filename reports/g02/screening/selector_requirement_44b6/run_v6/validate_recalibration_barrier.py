#!/usr/bin/env python3
"""Fail closed if the recalibration barrier misstates a source or scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BARRIER = HERE / "RECALIBRATION_BARRIER_20260731.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    barrier = json.loads(BARRIER.read_text())
    sources = {
        "supplier_gate_amendment": barrier["supplier_gate_amendment"],
        "nodetopo_diagnosis": {"path": barrier["nodetopo"]["diagnosis_path"], "sha256": barrier["nodetopo"]["diagnosis_sha256"]},
        "nodetopo_preregistration": {"path": barrier["nodetopo"]["preregistration_path"], "sha256": barrier["nodetopo"]["preregistration_sha256"]},
        "measured_surface_rank": barrier["measured_surface_rank"],
        "apncc_readiness": {"path": barrier["apncc_transport_v2"]["readiness_path"], "sha256": barrier["apncc_transport_v2"]["readiness_sha256"]},
        "apncc_cpu_step0_smoke": barrier["apncc_transport_v2"]["cpu_step0_smoke"],
        "apncc_remote_prelaunch": {
            "path": barrier["apncc_transport_v2"]["remote_running_evidence"]["prelaunch_path"],
            "sha256": barrier["apncc_transport_v2"]["remote_running_evidence"]["prelaunch_sha256"],
        },
        "apncc_remote_push": {
            "path": barrier["apncc_transport_v2"]["remote_running_evidence"]["push_path"],
            "sha256": barrier["apncc_transport_v2"]["remote_running_evidence"]["push_sha256"],
        },
        "apncc_remote_status": {
            "path": barrier["apncc_transport_v2"]["remote_running_evidence"]["read_only_status_path"],
            "sha256": barrier["apncc_transport_v2"]["remote_running_evidence"]["read_only_status_sha256"],
        },
        "apncc_remote_terminal": barrier["apncc_transport_v2"]["terminal"],
        "division_44b6": barrier["division_44b6"],
        "tier_census": barrier["tier_census"],
    }
    checks = {f"{name}_byte_pinned": sha256(ROOT / value["path"]) == value["sha256"] for name, value in sources.items()}
    readiness = json.loads((ROOT / barrier["apncc_transport_v2"]["readiness_path"]).read_text())
    prelaunch = json.loads((ROOT / barrier["apncc_transport_v2"]["remote_running_evidence"]["prelaunch_path"]).read_text())
    push = json.loads((ROOT / barrier["apncc_transport_v2"]["remote_running_evidence"]["push_path"]).read_text())
    remote_status = json.loads((ROOT / barrier["apncc_transport_v2"]["remote_running_evidence"]["read_only_status_path"]).read_text())
    terminal = json.loads((ROOT / barrier["apncc_transport_v2"]["terminal"]["path"]).read_text())
    build_receipt = ROOT / readiness["build_receipt"]["path"]
    inline_guard = ROOT / readiness["inline_prelaunch_guard"]["path"]
    checks.update({
        "barrier_is_held": barrier["status"] == "HOLD_AT_BARRIER_REMOTE_TRANSPORT_FAIL",
        "no_tier_move": barrier["supplier_gate_amendment"]["tier_moves"] == 0 and barrier["tier_census"]["tier_moves"] == 0 and barrier["scope"]["tier_changes"] == 0,
        "no_supplier_resurrection": barrier["supplier_gate_amendment"]["supplier_resurrections"] == 0,
        "nodetopo_causality": barrier["nodetopo"]["conclusion"] == "GENERATOR_FAILURE_NOT_SELECTOR_FAILURE" and barrier["nodetopo"]["tier"] == "PARK_A",
        "apncc_remote_run_is_explicitly_held_after_terminal_failure": (
            barrier["apncc_transport_v2"]["remote_status"] == "FAIL_TRANSPORT_PATH"
            and barrier["apncc_transport_v2"]["no_remote_action_occurred"] is False
            and barrier["scope"]["remote_kaggle_launch"] is True
            and barrier["scope"]["gpu_budget_debit_h"] == 0.16
        ),
        "apncc_remote_evidence_is_consistent": (
            prelaunch["status"] == "PASS_EXCLUSIVE_ZERO_EXISTING_VERSION"
            and push["status"] == "PUSHED_EXACTLY_ONE_VERSION"
            and push["version_delta"]["pushed_version"] == 1
            and remote_status["observation"] == 'onebitbrain/bhap-transport-smoke-v2/1 has status "KernelWorkerStatus.RUNNING"'
        ),
        "apncc_build_receipt_byte_pinned": sha256(build_receipt) == readiness["build_receipt"]["sha256"],
        "apncc_inline_guard_byte_pinned": sha256(inline_guard) == readiness["inline_prelaunch_guard"]["sha256"],
        "apncc_inline_guard_passed": readiness["inline_prelaunch_guard"]["status"] == "PASS",
        "apncc_terminal_failure_is_evidence_bound": (
            terminal["status"] == "FAIL_TRANSPORT_PATH"
            and terminal["step0"]["status"] == "PASS_TARGETED_DECLARED_INPUTS"
            and terminal["step0"]["wheel_count"] == 62
            and terminal["step0"]["competition_path_visits"] == 0
            and terminal["step0"]["elapsed_seconds"] < 60.0
            and terminal["transport"]["banked_runner_returncode"] == 0
            and terminal["transport"]["status"] == "FAIL"
            and terminal["submission_action"] is False
            and terminal["quota"]["used_delta_h"] == 0.16
        ),
        "apncc_readiness_status": readiness["status"] == "LOCAL_BUILD_GUARD_READY_CPU_SMOKE_FAIL_CLOSED_REMOTE_TERMINAL_FAIL",
        "apncc_cpu_step0_is_honest_fail_closed": (
            barrier["apncc_transport_v2"]["cpu_step0_smoke"]["status"] == "FAIL_CLOSED_LOCAL_SUPPORT_INVENTORY"
            and readiness["cpu_step0_smoke"]["status"] == "FAIL_CLOSED_LOCAL_SUPPORT_INVENTORY"
            and readiness["cpu_step0_smoke"]["heavy_work_started"] is False
        ),
        "apncc_remote_scope_preserves_scientific_hard_stops": (
            readiness["scope"]["kaggle_action"] is True
            and readiness["scope"]["gpu_action"] is True
            and readiness["scope"]["submission_action"] is False
            and readiness["scope"]["budget_debit"] == 0.16
            and readiness["scope"]["graph"] is False
            and readiness["scope"]["family_6bba"] is False
            and readiness["scope"]["cv57"] is False
            and readiness["scope"]["tier_moves"] == 0
        ),
        "division_remains_terminal": barrier["division_44b6"]["status"] == "TERMINAL_NO_PASS_HOLD_44B6",
    })
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
