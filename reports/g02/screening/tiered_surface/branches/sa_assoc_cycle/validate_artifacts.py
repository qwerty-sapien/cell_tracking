#!/usr/bin/env python3
"""Validate the terminal association-cycle lane without changing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    lane = args.lane.resolve()
    checks: list[dict] = []

    def check(name: str, value: bool) -> None:
        checks.append({"name": name, "pass": bool(value)})
        if not value:
            raise RuntimeError(f"validation failed: {name}")

    prereg = load(lane / "PREREGISTRATION.json")
    result = load(lane / "AUDIT_RESULT.json")
    state = load(lane / "state_fragment.json")
    idle = load(lane / "IDLE_PREFLIGHT.json")
    tests = load(lane / "TEST_GUARD.json")
    guard = load(lane / "AUDIT_GUARD.json")

    check("prereg_frozen", prereg["status"].startswith("FROZEN_BEFORE"))
    check("expected_park", result["status"] == "PARK_NOT_READY_MISSING_AUTHENTIC_REVERSE")
    check("forward_complete", result["existing_cache"]["fit_only_44b6_video_count"] == 36)
    check("one_forward_margin", result["existing_cache"]["trackastra_feature_count"] == 1)
    check("reverse_missing", not result["capability_checks"]["reverse_inference_artifact_present"])
    check("endpoint_return_missing", not result["capability_checks"]["endpoint_return_contract_present"])
    check("separation_unrun", not result["separation"]["executed"])
    check("graph_unrun", not result["separation"]["graph_run"])
    check("no_tier_proposal", result["tier_proposal"] is None and state["tier_moves_proposed"] == 0)
    check("state_evidence_hash", all(row["evidence_sha256"] == sha256(lane / "AUDIT_RESULT.json") for row in state["nodes"]))
    check("tiers_retained", {row["tier"] for row in state["nodes"]} == {"PARK_B"})
    check("preflight_45s", idle["status"] == "PASS" and idle["elapsed_seconds"] >= 45.0)
    check("zero_swap", idle["swap_growth_mib"] == tests["swap_growth_mib"] == guard["swap_growth_mib"] == 0.0)
    check("tests_pass", tests["status"] == "PASS" and tests["returncode"] == 0)
    check("audit_pass", guard["status"] == "PASS" and guard["returncode"] == 0)
    check("no_forbidden_action", not any(result["forbidden_actions"].values()))

    repository = lane.parents[5]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", args.base],
        cwd=repository,
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repository,
        text=True,
    ).splitlines()
    paths = sorted(set(changed + untracked))
    prefix = "reports/g02/screening/tiered_surface/branches/sa_assoc_cycle/"
    check("path_allowlist", bool(paths) and all(path.startswith(prefix) for path in paths))

    output = {
        "schema": "g02.sa-assoc-cycle.validation.v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
        "changed_paths": paths,
        "base_commit": args.base,
        "audit_result_sha256": sha256(lane / "AUDIT_RESULT.json"),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
