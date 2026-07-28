#!/usr/bin/env python3
"""Validate SA-NODETOPO terminal artifacts and the git path fence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


BASE = "2c74eed32c6f080faff045ae22847b98bbde8db0"
PREFIX = "reports/g02/screening/tiered_surface/branches/sa_nodetopo/"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_lines(repo: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--branch-dir", type=Path, required=True)
    parser.add_argument("--validation-out", type=Path, required=True)
    parser.add_argument("--path-audit-out", type=Path, required=True)
    args = parser.parse_args()
    if args.validation_out.exists() or args.path_audit_out.exists():
        raise SystemExit("refusing to overwrite validation artifacts")

    repo = args.repo.resolve()
    branch = args.branch_dir.resolve()
    audit = load(branch / "PROVENANCE_AUDIT.json")
    prereg = load(branch / "PREREGISTRATION.json")
    state = load(branch / "state_fragment.json")
    n1 = load(branch / "N1_ACTION_VALIDITY_EVIDENCE.json")
    stress = load(branch / "SUPPORT_STRESS_RESULT.json")
    preflight = load(branch / "idle_preflight.json")
    guard = load(branch / "local_run_guard.json")
    attempt1 = load(branch / "local_run_guard_attempt1.json")

    checks: dict[str, bool] = {
        "prereg_schema": prereg.get("schema") == "celltrack-alpha.sa-nodetopo-preregistration.v1",
        "terminal_verdict": audit.get("verdict") == "PARK_NO_COMPLETE_AUTHENTIC_CANDIDATE_UNIVERSE",
        "audit_sha_matches_state": all(
            node.get("latest_audit_sha256") == digest(branch / "PROVENANCE_AUDIT.json")
            for node in state.get("nodes", [])
        ),
        "audit_sha_matches_n1": n1.get("source", {}).get("audit_sha256") == digest(branch / "PROVENANCE_AUDIT.json"),
        "audit_sha_matches_stress": stress.get("source", {}).get("audit_sha256") == digest(branch / "PROVENANCE_AUDIT.json"),
        "state_schema": state.get("schema_version") == 1 and len(state.get("nodes", [])) == 2,
        "state_tiers_legal": all(node.get("tier") in {"LIVE", "PARK_A", "PARK_B", "PARK_C", "DEAD"} for node in state.get("nodes", [])),
        "state_banked_paths_resolvable": all(
            node.get("banked") is True
            and (
                (repo / str(node.get("evidence_artifact"))).is_file()
                or (Path("/Users/agent/Desktop/cell_tracking") / str(node.get("evidence_artifact"))).is_file()
            )
            for node in state.get("nodes", [])
        ),
        "state_banked_hashes_match": all(
            digest(
                (repo / str(node.get("evidence_artifact")))
                if (repo / str(node.get("evidence_artifact"))).is_file()
                else (Path("/Users/agent/Desktop/cell_tracking") / str(node.get("evidence_artifact")))
            )
            == node.get("evidence_sha256")
            for node in state.get("nodes", [])
        ),
        "no_graph_result": all(node.get("graph_result") is None for node in state.get("nodes", [])),
        "no_tier_move": audit.get("decision", {}).get("tier_move_proposed") is False,
        "44b6_not_scored": audit.get("scope", {}).get("held_graph_scored") is False,
        "6bba_closed": audit.get("scope", {}).get("second_family_opened") is False,
        "cv57_closed": audit.get("scope", {}).get("cv57_used") is False,
        "quarantine_flags_false": all(value is False for value in audit.get("quarantine", {}).values()),
        "n1_complete_action_pass": n1.get("complete_restore_topology_action", {}).get("compiled") is True,
        "n1_isolated_action_rejected": n1.get("isolated_coordinate_only_control", {}).get("rejected") is True,
        "real_n1_actions_zero": n1.get("real_candidate_actions_compiled") == 0,
        "stress_not_fabricated": stress.get("stress_curve", {}).get("executed") is False and stress.get("stress_curve", {}).get("points") == [],
        "fresh_preflight_pass": preflight.get("status") == "PASS_ZERO_SWAP_GROWTH" and preflight.get("swap_growth_mib") == 0.0,
        "guarded_run_pass": guard.get("status") == "PASS" and guard.get("swap_growth_mib") == 0.0,
        "failed_attempt_fail_closed": attempt1.get("status") == "FAIL_COMMAND" and attempt1.get("swap_growth_mib") == 0.0,
        "no_gpu_request": not (branch / "GPU_REQUEST.json").exists(),
    }

    changed = sorted(set(git_lines(repo, "diff", "--name-only", BASE)))
    untracked = []
    for line in git_lines(repo, "status", "--porcelain"):
        path = line[3:]
        if path:
            untracked.append(path)
    touched = sorted(set(changed + untracked + [
        str(args.validation_out.resolve().relative_to(repo)),
        str(args.path_audit_out.resolve().relative_to(repo)),
    ]))
    outside = [path for path in touched if not path.startswith(PREFIX)]
    checks["path_allowlist"] = not outside
    checks["branch_name"] = git_lines(repo, "branch", "--show-current") == ["codex/sa-nodetopo"]

    root_dirty = git_lines(
        Path("/Users/agent/Desktop/cell_tracking"),
        "status",
        "--porcelain",
        "--",
        PREFIX,
    )
    checks["root_checkout_lane_clean"] = root_dirty == []

    path_audit = {
        "schema": "celltrack-alpha.sa-nodetopo-path-audit.v1",
        "status": "PASS" if checks["path_allowlist"] and checks["branch_name"] and checks["root_checkout_lane_clean"] else "FAIL",
        "git_base": BASE,
        "branch": "codex/sa-nodetopo",
        "allowed_prefix": PREFIX,
        "touched_paths": touched,
        "outside_allowlist": outside,
        "root_checkout_lane_status": root_dirty,
    }
    args.path_audit_out.write_text(
        json.dumps(path_audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    validation = {
        "schema": "celltrack-alpha.sa-nodetopo-validation.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifact_hashes": {
            path.name: digest(path)
            for path in sorted(branch.iterdir())
            if path.is_file() and path not in {args.validation_out, args.path_audit_out}
        },
    }
    args.validation_out.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": validation["status"], "failed": [key for key, ok in checks.items() if not ok]}))
    return 0 if validation["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
