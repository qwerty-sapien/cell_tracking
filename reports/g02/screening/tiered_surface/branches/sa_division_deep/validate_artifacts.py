#!/usr/bin/env python3
"""Fail-closed validation for the SA-DIVISION-DEEP terminal package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/agent/Desktop/cell_tracking")
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str) -> dict:
    value = json.loads((HERE / name).read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise AssertionError(f"{name} is not a JSON object")
    return value


def main() -> int:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence: object) -> None:
        if not condition:
            raise AssertionError(f"{name}: {evidence}")
        checks.append({"name": name, "passed": True, "evidence": evidence})

    audit = load("oracle_audit_v2.json")
    receipts = load("INPUT_HASH_RECEIPTS.json")
    state = load("state_fragment.json")
    prereg = load("preregistration.json")
    preflight = load("preflight_retry4.json")
    audit_guard = load("audit_guard_retry4.json")
    test_guard = load("test_guard_retry1.json")

    counts = audit["counts"]
    check("cv57_denominator", counts["cv57_gt_divisions"] == 42, counts)
    check(
        "classification_partition",
        counts["PRESENT_FRAGMENTED"] == 38
        and counts["PRESENT_CONNECTED_NO_LOCAL_FORK"] == 1
        and counts["ROLE_MISSING"] == 3,
        counts,
    )
    check("strict_role_covered", counts["role_covered"] == 39, counts)
    check(
        "n1_compile_population",
        counts["n1_oracle_actions_compiled"] == 39
        and counts["n1_oracle_actions_failed"] == 0,
        counts,
    )
    check(
        "d1_suppression",
        counts["d1_persistent_actions_selected"] == 36
        and counts["missing_branch_persistence"] == 3
        and counts["duplicate_equivalence_class"] == 0
        and counts["one_to_one_node_conflict"] == 0
        and counts["duplicate_survivors"] == 0,
        counts,
    )
    action_hashes = [
        row["oracle_action"]["action_sha256"]
        for row in audit["events"]
        if row["oracle_action"] is not None
    ]
    check(
        "action_hash_uniqueness",
        len(action_hashes) == len(set(action_hashes)) == 39,
        len(action_hashes),
    )
    check(
        "dependency_closed",
        audit["dependency"]
        == {
            "cv57_fit": False,
            "fitting_locked": True,
            "held_graph_run": False,
            "qualifying_both_family_association_arm_count": 0,
            "scorer_run": False,
        },
        audit["dependency"],
    )
    check(
        "no_external_or_gpu_actions",
        not any(
            bool(audit["execution"][key])
            for key in (
                "budget_debit",
                "cellect_touched",
                "dead_verdict",
                "gpu",
                "kaggle",
                "quarantined_memo_numbers_used",
                "runpod",
                "submission",
            )
        ),
        audit["execution"],
    )
    check(
        "preflight",
        preflight["status"] == "PASS_ZERO_SWAP_GROWTH"
        and preflight["elapsed_seconds"] >= 45.0
        and preflight["swap_growth_mib"] == 0.0,
        preflight,
    )
    check(
        "audit_guard",
        audit_guard["status"] == "PASS"
        and audit_guard["returncode"] == 0
        and audit_guard["swap_growth_mib"] == 0.0,
        {
            key: audit_guard[key]
            for key in ("status", "returncode", "swap_growth_mib")
        },
    )
    check(
        "focused_tests",
        test_guard["status"] == "PASS"
        and test_guard["returncode"] == 0
        and test_guard["swap_growth_mib"] == 0.0
        and "Ran 7 tests" in test_guard["stdout"]
        and "\nOK\n" in test_guard["stdout"],
        {
            key: test_guard[key]
            for key in ("status", "returncode", "swap_growth_mib")
        },
    )
    failed_guards = [
        load("audit_guard.json"),
        load("audit_guard_retry1.json"),
        load("audit_guard_retry2.json"),
    ]
    check(
        "failed_attempts_closed",
        all(
            guard["status"] == "FAIL"
            and guard["returncode"] != 0
            and guard["swap_growth_mib"] == 0.0
            for guard in failed_guards
        ),
        [
            {
                key: guard[key]
                for key in ("status", "returncode", "swap_growth_mib")
            }
            for guard in failed_guards
        ],
    )
    for name, receipt in receipts["inputs"].items():
        observed = sha256(ROOT / receipt["path"])
        check(f"input_pin_{name}", observed == receipt["sha256"], observed)
    check(
        "fragment_receipts",
        len(audit["fragment_receipts"]) == 26
        and all(
            sha256(Path(row["path"])) == row["sha256"]
            for row in audit["fragment_receipts"]
        ),
        len(audit["fragment_receipts"]),
    )
    audit_hash = sha256(HERE / "oracle_audit_v2.json")
    check(
        "state_fragment_evidence",
        state["nodes"][0]["id"] == "DIV-F1"
        and state["nodes"][0]["tier"] == "PARK_A"
        and not state["nodes"][0]["tier_move_proposed"]
        and state["nodes"][0]["evidence_sha256"] == audit_hash,
        state["nodes"][0],
    )
    check(
        "prereg_boundary",
        prereg["status"] == "FROZEN_BEFORE_WAVE2_COMPONENT_AUDIT"
        and prereg["dependency"]["fitting_locked"]
        and prereg["forbidden"]["graph_run"]
        and prereg["forbidden"]["gpu"]
        and not prereg["kill_line"]["dead_verdict_authorized"],
        prereg["status"],
    )
    output = {
        "schema": "celltrack-alpha.sa-division-deep-validation.v1",
        "status": "PASS",
        "passed": len(checks),
        "total": len(checks),
        "checks": checks,
        "artifact_hashes": {
            name: sha256(HERE / name)
            for name in (
                "D1_PERSISTENT_FORK_PREREGISTRATION.md",
                "preregistration.json",
                "audit_division_oracle.py",
                "test_audit_division_oracle.py",
                "resource_guard.py",
                "oracle_audit_v2.json",
                "INPUT_HASH_RECEIPTS.json",
                "state_fragment.json",
                "PROPOSED_LEDGER_ROWS.md",
                "TERMINAL_REPORT.md",
            )
        },
    }
    path = HERE / "validation_v2.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "passed": len(checks), "total": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
