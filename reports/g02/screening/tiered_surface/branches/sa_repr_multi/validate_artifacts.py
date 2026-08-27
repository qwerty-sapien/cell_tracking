#!/usr/bin/env python3
"""Strict self-validation for SA-REPR-MULTI branch artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/agent/Desktop/cell_tracking")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def check(condition: bool, name: str, checks: list[dict]) -> None:
    checks.append({"check": name, "pass": bool(condition)})
    if not condition:
        raise AssertionError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "VALIDATION.json")
    args = parser.parse_args()
    checks: list[dict] = []
    prereg = load("PREREGISTRATION_v1.json")
    sequence = load("PREREGISTRATION_v2_SEQUENCE_CLARIFICATION.json")
    pop = load("POPULATION_FREEZE.json")
    provenance = load("SUPPLIER_PROVENANCE_AUDIT.json")
    queue = load("READY_SAMPLE_QUEUE.json")
    summary = load("AUDIT_SUMMARY.json")
    state = load("state_fragment.json")
    resources = load("RESOURCE_REPORT.json")
    idle = load("IDLE_PREFLIGHT.json")
    tests = load("TEST_GUARD.json")
    audit = load("AUDIT_GUARD.json")
    allowlist = load("PATH_ALLOWLIST_AUDIT_v2.json")
    amendment = json.loads(
        (
            ROOT
            / "reports/g02/screening/tiered_surface/branches/sa_repr"
            / "phase2_separation_gate/PREREGISTRATION_v2_per_class.json"
        ).read_text()
    )

    check(prereg["status"] == "FROZEN_BEFORE_SUPPLIER_SCAN_OR_OUTCOME_OPEN", "prereg frozen", checks)
    check(
        sequence["clarification"]["effective_status_text"]
        == "FROZEN_BEFORE_NEW_SUPPLIER_FEATURE_OR_PROBE_OUTCOME",
        "sequence wording narrowed",
        checks,
    )
    check(
        sequence["clarification"]["scientific_contract_changed"] is False,
        "sequence clarification changes no scientific contract",
        checks,
    )
    check(
        sha256(HERE / "PREREGISTRATION_v1.json")
        == summary["preregistration"]["sha256"],
        "prereg hash exact",
        checks,
    )
    check(pop["status"] == "FROZEN_COMPLETE", "population frozen", checks)
    check(pop["inventory_file_count"] == 154, "154 shard files", checks)
    check(pop["inventory_total_bytes"] == 44729626, "population byte total", checks)
    check(
        all(
            value is True
            for key, value in pop["integrity"].items()
            if key != "duplicate_identity_class_pairs"
        )
        and pop["integrity"]["duplicate_identity_class_pairs"] == 0,
        "all population integrity checks pass",
        checks,
    )
    check(
        pop["aggregate_sha256"]
        == "ebde982b68abfa1359b0323b8699c31a0e303c0982cfcb01caa2b2fa0536130c",
        "population aggregate exact",
        checks,
    )
    frozen_thresholds = prereg["authoritative_comparison_rules_unchanged"][
        "every_fold_thresholds"
    ]
    upstream_gate = amendment["replacements"]["pass_gate"]
    check(
        all(upstream_gate[key] == value for key, value in frozen_thresholds.items()),
        "all authoritative numeric/integrity thresholds unchanged",
        checks,
    )
    check(summary["authoritative_comparison_rules_changed"] is False, "comparison unchanged", checks)
    check(provenance["authenticated_count"] == 1, "one authenticated supplier", checks)
    check(provenance["authenticated_non_r7_count"] == 0, "no new authenticated supplier", checks)
    check(queue["ready_sample_count"] == 0, "empty new queue", checks)
    check(queue["authorization_requested"] is False, "no authorization requested", checks)
    check(queue["authorization_held"] is False, "no authorization held", checks)
    check(queue["gpu_launched"] is False, "no GPU launched", checks)
    check(summary["tier_moves"] == [], "no tier moves", checks)
    check(summary["uncited_tier_moves"] == 0, "zero uncited tier moves", checks)
    check(summary["memo_quarantine_pass"] is True, "memo quarantine", checks)
    check(idle["status"] == "PASS" and idle["swap_growth_mib"] == 0.0, "idle preflight", checks)
    check(tests["status"] == "PASS" and tests["swap_growth_mib"] == 0.0, "tests guarded", checks)
    check(audit["status"] == "PASS" and audit["swap_growth_mib"] == 0.0, "audit guarded", checks)
    check(resources["external_resource_use"]["gpu_hours"] == 0.0, "zero GPU hours", checks)
    check(resources["external_resource_use"]["submissions"] == 0, "zero submissions", checks)
    check(resources["external_resource_use"]["runpod_runs"] == 0, "zero RunPod", checks)
    check(allowlist["status"] == "PASS", "path allowlist pass", checks)
    check(allowlist["violation_count"] == 0, "zero path violations", checks)
    check(
        allowlist["protected_single_writer_paths_modified"] == [],
        "protected single-writer paths untouched",
        checks,
    )
    check(len(state["nodes"]) == 9, "R2 through R10 represented", checks)
    check(all(node["tier"] == "PARK_B" for node in state["nodes"]), "all tiers unchanged PARK_B", checks)
    check(
        all(
            node["evidence_sha256"]
            == sha256(HERE / "SUPPLIER_PROVENANCE_AUDIT.json")
            for node in state["nodes"]
        ),
        "state evidence hashes exact",
        checks,
    )
    check(state["comparison_contract"]["changed"] is False, "state comparison unchanged", checks)

    result = {
        "schema": "g02.sa-repr-multi.validation.v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks_failed": 0,
        "checks": checks,
    }
    output = args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2)
