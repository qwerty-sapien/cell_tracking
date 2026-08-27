#!/usr/bin/env python3
"""Validate SA-REVERSE-SUPPLIER terminal artifacts without opening outcomes."""

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
    contract = load(lane / "ENDPOINT_RETURN_CONTRACT.json")
    audit = load(lane / "ASSET_AUDIT.json")
    request = load(lane / "COMPUTE_REQUEST.json")
    state = load(lane / "state_fragment.json")
    idle = load(lane / "IDLE_PREFLIGHT.json")
    tests = load(lane / "TEST_GUARD.json")
    audit_guard = load(lane / "AUDIT_GUARD.json")

    check("prereg_frozen", prereg["status"].startswith("FROZEN_BEFORE"))
    check("expected_stop", audit["status"] == "STOP_COMPUTE_REQUEST_REQUIRED")
    check("forward_complete", audit["forward_cache"]["video_count"] == 36)
    check("forward_counts", (
        audit["forward_cache"]["inference_calls"],
        audit["forward_cache"]["proposal_count"],
        audit["forward_cache"]["row_count"],
    ) == (3539, 84883, 121534))
    check("sparse_cache_complete", (
        audit["sparse_detection_cache"]["bhc_count"],
        audit["sparse_detection_cache"]["json_count"],
    ) == (36, 36))
    check("source_authenticated", audit["source_authentication"]["all_pins_match"])
    check("checkpoint_absent", not audit["checkpoint"]["exact_local_checkpoint_present"])
    check("reverse_absent", not audit["reverse_supplier"]["immutable_authentic_reverse_cache_present"])
    check("genuine_inference_required", audit["reverse_supplier"]["genuine_reverse_inference_required"])
    check("synthetic_reverse_false", not audit["reverse_supplier"]["synthetic_reverse_used"])
    check("endpoint_return_exact", (
        contract["genuine_reverse_inference"]["returned_pair_score"]
        == "for canonical edge (u_t,v_t+1), R_return(u,v) = A_reverse[index(v),index(u)]"
    ))
    check("separate_model_pass", "new model forward pass" in contract["genuine_reverse_inference"]["execution"])
    check("no_link_exact", contract["no_link"]["formula"].startswith("N_return"))
    check("division_not_hard_veto", "never be a hard veto" in contract["division_and_boundary_behavior"]["division"])
    check("compute_not_launched", request["status"] == "REQUESTED_NOT_AUTHORIZED_NOT_LAUNCHED")
    check("cpu_bounded", (
        request["projection"]["cpu_inference_calls"] == 3638
        and request["projection"]["cpu_hard_wall_hours"] == 1.5
        and request["projection"]["gpu_hours"] == 0
    ))
    check("no_outcomes", (
        not audit["decision"]["separation_executed"]
        and not audit["decision"]["graph_executed"]
        and not state["family_execution"]["44b6_reverse_inference_ran"]
    ))
    check("no_tier_move", (
        audit["decision"]["tier_move_proposed"] is False
        and state["tier_moves_proposed"] == 0
        and {node["tier"] for node in state["nodes"]} == {"PARK_B"}
    ))
    check("audit_hash_cited", all(
        node["evidence_sha256"] == sha256(lane / "ASSET_AUDIT.json")
        for node in state["nodes"]
    ))
    check("contract_hash_cited", (
        state["artifacts"]["endpoint_return_contract"]["sha256"]
        == sha256(lane / "ENDPOINT_RETURN_CONTRACT.json")
    ))
    check("request_hash_cited", (
        state["artifacts"]["compute_request"]["sha256"]
        == sha256(lane / "COMPUTE_REQUEST.json")
    ))
    check("preflight_45s", idle["status"] == "PASS" and idle["elapsed_seconds"] >= 45.0)
    check("zero_swap", (
        idle["swap_growth_mib"]
        == tests["swap_growth_mib"]
        == audit_guard["swap_growth_mib"]
        == 0.0
    ))
    check("focused_tests", tests["status"] == "PASS" and tests["returncode"] == 0)
    check("guarded_audit", audit_guard["status"] == "PASS" and audit_guard["returncode"] == 0)
    check("no_forbidden_action", not any(audit["forbidden_actions"].values()))

    repository = lane.parents[5]
    prefix = "reports/g02/screening/tiered_surface/branches/sa_reverse/"
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", prefix],
        cwd=repository,
        text=True,
    ).splitlines()
    authored = sorted(line[3:] for line in status)
    check("lane_files_present", bool(authored))
    check("lane_path_allowlist", all(path.startswith(prefix) for path in authored))

    key_hashes = {
        name: sha256(lane / name)
        for name in (
            "PREREGISTRATION.json",
            "ENDPOINT_RETURN_CONTRACT.json",
            "ASSET_AUDIT.json",
            "COMPUTE_REQUEST.json",
            "state_fragment.json",
            "PROPOSED_LEDGER_ROWS.md",
            "TERMINAL_REPORT.md",
            "IDLE_PREFLIGHT.json",
            "TEST_GUARD.json",
            "AUDIT_GUARD.json",
        )
    }
    output = {
        "schema": "g02.sa-reverse.validation.v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
        "base_commit": args.base,
        "authored_lane_paths": authored,
        "key_artifact_sha256": key_hashes,
        "ambient_empty_checkout_deletions_excluded_from_authored_lane_scope": True
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
