#!/usr/bin/env python3
"""Fail-closed verifier for the approved, non-retroactive gate recalibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
AMENDMENT = HERE / "GATE_RECALIBRATION_APPROVED_20260731.json"
RERANK = HERE / "MEASURED_SURFACE_RERANK_20260731.json"
REQUIREMENTS = HERE / "requirement_surface.json"
DINOV2 = ROOT / "reports/g02/screening/tiered_surface/branches/sa_dinov2/local_bakeoff_v3/TERMINAL_REPORT.json"
CENSUS = ROOT / "reports/g02/screening/tiered_surface/tier_census.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str, checks: list[dict[str, object]]) -> None:
    checks.append({"name": message, "pass": bool(condition)})


def main() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    rerank = json.loads(RERANK.read_text())
    requirements = json.loads(REQUIREMENTS.read_text())
    dinov2 = json.loads(DINOV2.read_text())
    census = json.loads(CENSUS.read_text())
    checks: list[dict[str, object]] = []
    expected_hash = "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b"
    check(sha256(REQUIREMENTS) == expected_hash, "run_v6 requirement surface remains byte-pinned", checks)
    check(amendment["status"] == "APPROVED_DIAGNOSTIC_RECALIBRATION", "human amendment recorded", checks)
    floors = amendment["supersedes_for_future_supplier_evaluation"]["absolute_floors"]
    check(floors == {"delete": {"auc": 0.85, "retention": 0.02}, "occupied_swap": {"auc": 0.9, "retention": 0.068}}, "approved class-specific floors exact", checks)
    check(amendment["nonretroactivity"]["previously_failed_supplier_results_are_immutable"], "historic results immutable", checks)
    check(amendment["nonretroactivity"]["supplier_resurrections"] == 0 and amendment["nonretroactivity"]["tier_moves"] == 0, "no resurrection or tier move", checks)
    observed = dinov2["results"]["delete"]["A1_fold_mean_retention"]
    target = floors["delete"]["retention"]
    check(sha256(DINOV2) == amendment["nonretroactivity"]["dinov2_delete"]["result_sha256"], "DINOv2 source receipt is byte-pinned", checks)
    check(observed < target and target / observed > 15.0, "DINOv2 delete remains more than 15x short", checks)
    measured_delete = requirements["requirements"]["delete"]["target_plus_0_005"]["mean_requirement"]
    measured_swap = requirements["requirements"]["occupied_swap"]["target_plus_0_005"]["mean_requirement"]
    check(measured_delete["target_auc"] == 0.85 and measured_delete["retention"]["mean"] > 0.020, "delete target comes from measured grid", checks)
    check(measured_swap["target_auc"] == 0.9 and measured_swap["retention"]["mean"] < 0.068, "swap target is conservatively rounded up", checks)
    ranked = rerank["ranking"]
    check([row["surface"] for row in ranked] == ["division", "delete supplier", "occupied_swap supplier"], "measured rank order retained", checks)
    check(rerank["tier_moves"] == 0, "rerank proposes no tier move", checks)
    nodes = {row["id"]: row for row in census["nodes"]}
    for node_id in ("AE-P6", "R8"):
        node = nodes[node_id]
        check(node["tier"] == "PARK_B", f"{node_id} remains PARK-B", checks)
        check("AUC>=0.85/retention>=0.020" in node["kill_line"] and "AUC>=0.90/retention>=0.068" in node["kill_line"], f"{node_id} future gate uses measured floors", checks)
        check("not re-evaluated by the amendment" in node["resume_condition"], f"{node_id} historic failure is non-retroactive", checks)
    result = {"status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
