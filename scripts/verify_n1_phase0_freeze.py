#!/usr/bin/env python3
"""Verify the frozen N1 package and root-owned Phase-0 state."""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SURFACE = ROOT / "reports/g02/screening/tiered_surface"
N1_DIR = SURFACE / "n1_feasibility_compiler"
OUTPUT = N1_DIR / "phase0_final_verification_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite final verification")

    report_path = N1_DIR / "N1_FREEZE_REPORT.json"
    report = json.loads(report_path.read_text(encoding="ascii"))
    if report["status"] != "GO":
        raise SystemExit("N1 report is not GO")
    for item in report["pinned_inputs"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"frozen input mismatch: {item['path']}")
    for name, item in report["frozen_artifacts"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise SystemExit(f"frozen artifact mismatch: {name}")

    guard = json.loads(
        (N1_DIR / "n1_freeze_guard_v1.json").read_text(encoding="ascii")
    )
    if (
        guard["status"] != "PASS"
        or guard["swap_growth_mib"] != 0.0
        or guard["return_code"] != 0
    ):
        raise SystemExit("N1 freeze resource guard failed")

    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_action_compiler")
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not outcome.wasSuccessful() or outcome.testsRun != 10:
        raise SystemExit(f"focused N1 unit suite failed:\n{stream.getvalue()}")

    source_files = (
        "src/celltrack_alpha/action_compiler.py",
        "tests/test_action_compiler.py",
        "scripts/freeze_n1_feasibility_compiler.py",
        "scripts/build_tier_census.py",
        "scripts/verify_n1_phase0_freeze.py",
    )
    for relative in source_files:
        py_compile.compile(str(ROOT / relative), doraise=True)
        for line_number, line in enumerate(
            (ROOT / relative).read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if line != line.rstrip():
                raise SystemExit(f"trailing whitespace: {relative}:{line_number}")

    census_path = SURFACE / "tier_census.json"
    census = json.loads(census_path.read_text(encoding="ascii"))
    nodes = census["nodes"]
    rollup = {
        key: Counter(node["tier"] for node in nodes).get(key, 0)
        for key in ("LIVE", "PARK_A", "PARK_B", "PARK_C", "DEAD")
    }
    expected_rollup = {
        "LIVE": 0,
        "PARK_A": 14,
        "PARK_B": 22,
        "PARK_C": 8,
        "DEAD": 1,
    }
    if len(nodes) != 45 or rollup != expected_rollup or census["rollup"] != rollup:
        raise SystemExit("tier census rollup mismatch")
    if len({node["id"] for node in nodes}) != len(nodes):
        raise SystemExit("duplicate tier node ID")
    for node in nodes:
        evidence = ROOT / node["evidence_artifact"]
        if not evidence.is_file() or sha256(evidence) != node["evidence_sha256"]:
            raise SystemExit(f"tier evidence mismatch: {node['id']}")
        if node["tier"] in ("PARK_C", "DEAD") and not node["banked"]:
            raise SystemExit(f"unbanked terminal tier: {node['id']}")
        if node["tier"] == "DEAD" and (
            not node.get("human_authorized") or not node.get("named_mechanism")
        ):
            raise SystemExit(f"invalid DEAD row: {node['id']}")

    g1 = next(node for node in nodes if node["id"] == "G1")
    if (
        g1["tier"] != "PARK_A"
        or g1["compiler"]["interface_sha256"]
        != report["frozen_artifacts"]["interface"]["sha256"]
        or g1["compiler"]["tests_sha256"]
        != report["frozen_artifacts"]["tests"]["sha256"]
    ):
        raise SystemExit("G1 census pin mismatch")

    budget_path = SURFACE / "kaggle_budget.json"
    budget = json.loads(budget_path.read_text(encoding="ascii"))
    debit_sum = sum(item["kernel_h"] for item in budget["debits"])
    if (
        abs(debit_sum - budget["used_kernel_h"]) > 1e-12
        or abs(
            budget["budget_kernel_h"]
            - budget["used_kernel_h"]
            - budget["remaining_kernel_h"]
        )
        > 1e-12
        or budget["active_authorizations"]
        or budget["runpod"] != "FORBIDDEN"
    ):
        raise SystemExit("Kaggle budget reconciliation failed")

    live = (ROOT / "reports/g02/screening/LIVE_STATE.md").read_text(
        encoding="utf-8"
    )
    census_hash = sha256(census_path)
    if (
        census_hash not in live
        or "LIVE 0 / PARK-A 14 / PARK-B 22 / PARK-C 8 / DEAD 1" not in live
    ):
        raise SystemExit("LIVE_STATE census rendering mismatch")

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--check",
            "--",
            "src/celltrack_alpha/action_compiler.py",
            "tests/test_action_compiler.py",
            "scripts/freeze_n1_feasibility_compiler.py",
            "scripts/build_tier_census.py",
            "scripts/verify_n1_phase0_freeze.py",
            "reports/g02/screening/tiered_surface",
            "reports/g02/screening/LIVE_STATE.md",
            "reports/g02/screening/LEDGER.md",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode:
        raise SystemExit(f"git diff --check failed: {diff.stdout}{diff.stderr}")

    result = {
        "schema": "celltrack-alpha.n1-phase0-final-verification.v1",
        "status": "PASS",
        "n1_report_sha256": sha256(report_path),
        "n1_interface_sha256": report["frozen_artifacts"]["interface"]["sha256"],
        "n1_implementation_sha256": (
            report["frozen_artifacts"]["implementation"]["sha256"]
        ),
        "n1_tests_sha256": report["frozen_artifacts"]["tests"]["sha256"],
        "unit_tests": {"passed": outcome.testsRun, "total": 10},
        "tier_census": {
            "sha256": census_hash,
            "node_count": len(nodes),
            "rollup": rollup,
        },
        "kaggle_budget": {
            "sha256": sha256(budget_path),
            "used_kernel_h": budget["used_kernel_h"],
            "remaining_kernel_h": budget["remaining_kernel_h"],
            "active_authorizations": 0,
        },
        "scope": {
            "audit_or_held_or_test_used": False,
            "cv57_used": False,
            "external_action": False,
            "gpu_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
        "phase1_dispatched": False,
        "diff_check": "PASS",
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
