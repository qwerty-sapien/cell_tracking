#!/usr/bin/env python3
"""Fail-closed validation for the division-value terminal artifact set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_ROOT = Path("/Users/agent/Desktop/cell_tracking")
GT = "GT-DEPENDENT"
NONGT = "NON-GT-IDENTIFIABLE"
DAG2_SHA = "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b"


def load(name: str) -> object:
    return json.loads((HERE / name).read_text())


def sha(name: str) -> str:
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def labeled_numbers(value: object, inherited: str | None = None) -> int:
    if isinstance(value, dict):
        own = value.get("information_label")
        label = own if own in (GT, NONGT) else inherited
        return sum(labeled_numbers(child, label) for child in value.values())
    if isinstance(value, list):
        return sum(labeled_numbers(child, inherited) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert inherited in (GT, NONGT)
        return 1
    return 0


def main() -> None:
    handoff = load("N4_REQUIREMENT_HANDOFF.json")
    verdict = load("N5_VERDICT.json")
    barrier = load("BARRIER_REPORT.json")
    n0n3 = load("N0_N3_RESULTS.json")
    sensitivity = load("FP_SENSITIVITY.json")

    assert handoff["schema"] == (
        "celltrack-alpha.selector-requirement-44b6.v1.requirement-surface"
    )
    assert set(handoff["requirements"]) == {
        "delete",
        "occupied_swap",
        "restore_topology",
        "division",
    }
    dag2_source_path = (
        SHARED_ROOT
        / "reports/g02/screening/selector_requirement_44b6/run_v6/requirement_surface.json"
    )
    assert hashlib.sha256(dag2_source_path.read_bytes()).hexdigest() == DAG2_SHA
    dag2_source = json.loads(dag2_source_path.read_text())
    assert {
        key: handoff[key]
        for key in ("grid_only", "interpolation", "schema", "status")
    } == {
        key: dag2_source[key]
        for key in ("grid_only", "interpolation", "schema", "status")
    }
    for name in ("delete", "occupied_swap", "restore_topology"):
        assert handoff["requirements"][name]["source_surface_sha256"] == DAG2_SHA
        assert handoff["requirements"][name]["information_label"] == GT
        copied_record = dict(handoff["requirements"][name])
        copied_record.pop("source_surface_sha256")
        copied_record.pop("information_label")
        assert copied_record == dag2_source["requirements"][name]
    division = handoff["requirements"]["division"]
    assert set(("breakeven_0", "frozen_gate_verdict", "target_plus_0_005")).issubset(
        division
    )
    assert division["frozen_gate_verdict"]["frozen_auc_floor"] == 0.65
    assert division["frozen_gate_verdict"]["frozen_retention_floor"] == 0.15
    assert division["frozen_gate_verdict"]["required_tested_auc"] is None
    assert division["frozen_gate_verdict"]["required_mean_retention"] is None
    assert division["frozen_gate_verdict"]["verdict"] == (
        "NOT_EVALUABLE_NO_DIVISION_SELECTOR"
    )
    assert division["target_plus_0_005"]["status"] == (
        "INFEASIBLE_AT_PERFECT_SELECTION_CURRENT_POOL"
    )
    assert division["target_plus_0_005"]["oracle_ceiling"] < 0.005
    assert division["representation_conditioned_diagnostic"][
        "selector_achievability"
    ] == "NOT_ESTABLISHED"

    assert verdict["decision"] == {
        "materially_large_realizable_subset": False,
        "required_selector_precision_achievable_per_dag2": False,
        "amendment_condition_satisfied": False,
        "amendment_proposal_emitted": False,
        "division_fitting": "LOCKED",
        "tier_moves": 0,
        "plan_amendments": 0,
        "dead_actions": 0,
    }
    assert verdict["current_pool"]["combined_score_delta"] == (
        n0n3["n2_realizable_range"]["a"]["pooled"]["delta"]["combined_score_delta"]
    )
    assert verdict["gt_ceiling"]["combined_score_delta"] == (
        n0n3["n0_ceiling"]["coalition"]["pooled"]["delta"]["combined_score_delta"]
    )
    assert verdict["stronger_representation"][
        "minimum_edge_neutral_precision_for_plus_0_005"
    ] == sensitivity["curves"]["ab"]["requirements"][
        "target_combined_delta_plus_0p005"
    ]["minimum_precision"]

    assert barrier["concurrent_barrier_pins"][
        "dag2_requirement_surface_sha256"
    ] == DAG2_SHA
    assert barrier["status"] == "COMPLETE"
    assert barrier["governance"]["division_fitting_locked"] is True
    assert barrier["governance"]["live_state_reconciled"] is True
    assert barrier["governance"]["ledger_reconciled"] is True
    assert barrier["governance"]["dag3_commit_merged"] is True
    assert barrier["governance"]["tier_moves"] == 0
    assert barrier["governance"]["plan_amendments"] == 0
    assert barrier["governance"]["dead_actions"] == 0
    assert not (HERE / "AMENDMENT_PROPOSAL.md").exists()

    numeric_leaves = (
        labeled_numbers(handoff)
        + labeled_numbers(verdict)
        + labeled_numbers(barrier)
        + labeled_numbers(load("state_fragment.json"))
    )
    assert numeric_leaves > 0

    receipt = {
        "schema": "celltrack-alpha.division-value-final-validation.v1",
        "passed": True,
        "checks": 39,
        "labeled_numeric_leaves": numeric_leaves,
        "information_label": NONGT,
        "artifact_sha256": {
            name: sha(name)
            for name in (
                "N0_N3_RESULTS.json",
                "GT_DEPENDENCY_TABLE.json",
                "FP_SENSITIVITY.json",
                "N4_REQUIREMENT_HANDOFF.json",
                "N5_VERDICT.json",
                "BARRIER_REPORT.json",
                "state_fragment.json",
                "PROPOSED_LEDGER_ROW.md",
                "FINAL_REPORT.md",
            )
        },
    }
    (HERE / "FINAL_VALIDATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
