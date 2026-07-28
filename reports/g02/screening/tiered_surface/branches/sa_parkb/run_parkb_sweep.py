#!/usr/bin/env python3
"""Deterministic identity/support-only sweep over the root PARK_B census."""

from __future__ import annotations

import hashlib
import json
import pathlib
import resource
import time
from typing import Any


BASE = pathlib.Path("/Users/agent/Desktop/cell_tracking")
OUT = pathlib.Path(__file__).resolve().parent
PLAN_SHA256 = "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116"
EXPECTED_INPUTS = {
    "reports/g02/screening/tiered_surface/tier_census.json":
        "768810ba949c37ab998ded83d99d47b9d9d6add6a57f5ff5340cf55bcf37994e",
    "reports/g02/screening/tiered_surface/n1_feasibility_compiler/N1_FREEZE_REPORT.json":
        "a27fdf5597d0ac66d3d0f1c85689a5cc03b16d0155684b765d441dab6117786d",
    "reports/g02/grouped_3fold_audit20_v1.json":
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json":
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
}
FALSIFIERS = {
    "AE-P5": "Require a frozen C0 edge/no-link assignment-cost and MAP identity contract; quarantine unbanked C2 numbers.",
    "AE-P6": "Require immutable forward/reverse endpoint-return evidence and a disabled-cycle identity control.",
    "AE-P7": "Require a frozen phase/template tracklet identity join with no held labels.",
    "AE-P8": "Require authenticated dense-embedding weights, source, license, and offline loader.",
    "AE-P9": "Require an authenticated frozen backbone and immutable adapter target before PEFT.",
    "DIV-F2": "Check banked both-family association continuity and authentic pretrained fork input.",
    "GAP-F1": "Compare the banked one-gap action-class oracle against the binding +0.005 both-family floor.",
    "NODERESTORE-O0": "Require a provenance-pinned full candidate restoration universe and baseline-topology join.",
    "DET-F0": "Check the banked free node-add class oracle for nonzero score and node-recall support.",
    "DET-F2": "Require an authentic proposal universe before learned detection expansion.",
    "DET-F3": "Require authentic proposals and a frozen topology-complete node-action join.",
    "R2": "Revalidate the banked authentic-mask/wrfeat provenance audit; pseudo-masks forbidden.",
    "R3": "Require R2 authentic morphology provenance before real-wrfeat support can open.",
    "R4": "Require exact encoder weight/source/license and an offline loader.",
    "R5": "Require R4 provenance before the frozen XY dense-feature smoke.",
    "R6": "Require R4 plus an immutable XY beta before tri-view support.",
    "R7": "Require complete supplier packaging and a root-frozen consumer population; do not use invalid GPU attempts as science.",
    "R8": "Require reusable forward/reverse evidence and immutable endpoint-return identities.",
    "R9": "Require frozen environment groups with adequate per-group support and no held-label derivation.",
    "R10": "Require an authentic new frozen representation before a PEFT target can be identified.",
    "G2": "Check frozen N1 feasibility identity plus banked positive P0 swap/delete support.",
    "G6": "Verify the owner RunPod prohibition; no execution edge may open."
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def p0_delta(p0: dict[str, Any], action: str, family: str) -> dict[str, float]:
    return p0["families"][family]["actions"][action]["delta_vs_control"]


def main() -> int:
    started = time.monotonic()
    authenticated: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUTS.items():
        actual = sha256(BASE / relative)
        if actual != expected:
            raise SystemExit(f"frozen input mismatch: {relative}: {actual} != {expected}")
        authenticated[relative] = actual

    census = load(BASE / "reports/g02/screening/tiered_surface/tier_census.json")
    p0 = load(BASE / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json")
    n1 = load(
        BASE
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/N1_FREEZE_REPORT.json"
    )
    park_b = [node for node in census["nodes"] if node["tier"] == "PARK_B"]
    if len(park_b) != 22:
        raise SystemExit(f"expected 22 PARK_B nodes, found {len(park_b)}")
    if {node["id"] for node in park_b} != set(FALSIFIERS):
        raise SystemExit("PARK_B census and preregistered falsifier universe differ")

    results: list[dict[str, Any]] = []
    for node in park_b:
        evidence_path = BASE / node["evidence_artifact"]
        evidence_exists = evidence_path.is_file()
        evidence_hash_match = (
            evidence_exists and sha256(evidence_path) == node["evidence_sha256"]
        )
        if not evidence_hash_match:
            raise SystemExit(f"banked evidence identity failed for {node['id']}")
        result: dict[str, Any] = {
            "id": node["id"],
            "falsifier": FALSIFIERS[node["id"]],
            "source_evidence_artifact": node["evidence_artifact"],
            "source_evidence_sha256": node["evidence_sha256"],
            "evidence_identity": "PASS",
            "from_tier": "PARK_B",
            "proposed_tier": "PARK_B",
            "tier_move_proposed": False,
            "verdict": "PARK_SUPPORT_OR_IDENTITY_PREREQUISITE",
            "memo_numbers_used": False,
        }
        if node["id"] == "GAP-F1":
            deltas = {
                family: p0_delta(p0, "gap_bridge", family)
                for family in ("44b6", "6bba")
            }
            scores = {family: values["score"] for family, values in deltas.items()}
            if not all(0.0 < value < 0.005 for value in scores.values()):
                raise SystemExit(f"unexpected GAP-F1 support vector: {scores}")
            result.update(
                {
                    "proposed_tier": "PARK_C",
                    "tier_move_proposed": True,
                    "verdict": "PROPOSE_B_TO_C_WEAK_BANKED_CLASS_SUPPORT",
                    "support": deltas,
                    "decision_evidence_artifact":
                        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
                    "decision_evidence_sha256": EXPECTED_INPUTS[
                        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
                    ],
                }
            )
        elif node["id"] == "DET-F0":
            deltas = {
                family: p0_delta(p0, "node_add", family)
                for family in ("44b6", "6bba")
            }
            if any(any(float(value) != 0.0 for value in family.values()) for family in deltas.values()):
                raise SystemExit(f"unexpected DET-F0 support vector: {deltas}")
            result.update(
                {
                    "proposed_tier": "PARK_C",
                    "tier_move_proposed": True,
                    "verdict": "PROPOSE_B_TO_C_ZERO_BANKED_CLASS_SUPPORT",
                    "support": deltas,
                    "decision_evidence_artifact":
                        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
                    "decision_evidence_sha256": EXPECTED_INPUTS[
                        "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
                    ],
                }
            )
        elif node["id"] == "G2":
            swap = {
                family: p0_delta(p0, "swap", family)
                for family in ("44b6", "6bba")
            }
            delete = {
                family: p0_delta(p0, "delete", family)
                for family in ("44b6", "6bba")
            }
            support_pass = all(
                values["score"] >= 0.005
                and values["edge_jaccard"] >= 0.0
                and values["node_recall"] >= 0.0
                for action in (swap, delete)
                for values in action.values()
            )
            n1_go = n1.get("status") == "GO"
            if not (support_pass and n1_go):
                raise SystemExit("G2 preregistered identity/support condition did not pass")
            result.update(
                {
                    "proposed_tier": "PARK_A",
                    "tier_move_proposed": True,
                    "verdict": "PROPOSE_B_TO_A_POSITIVE_BANKED_SUPPORT_AND_N1_IDENTITY",
                    "support": {"swap": swap, "delete": delete},
                    "n1_status": n1["status"],
                    "decision_evidence_artifacts": [
                        {
                            "path":
                                "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
                            "sha256": EXPECTED_INPUTS[
                                "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
                            ],
                        },
                        {
                            "path":
                                "reports/g02/screening/tiered_surface/n1_feasibility_compiler/N1_FREEZE_REPORT.json",
                            "sha256": EXPECTED_INPUTS[
                                "reports/g02/screening/tiered_surface/n1_feasibility_compiler/N1_FREEZE_REPORT.json"
                            ],
                        },
                    ],
                }
            )
        elif node["id"] == "G6":
            result["verdict"] = "PARK_OWNER_FORBIDS_RUNPOD"
        elif node.get("blocked_on"):
            result["blocked_on"] = node["blocked_on"]
        results.append(result)

    terminal = {
        "schema": "celltrack-alpha.parkb-support-sweep.v1",
        "status": "VERDICT_BARRIER",
        "owner_agent": "SA-PARKB-SWEEP",
        "plan_sha256": PLAN_SHA256,
        "input_authentication": authenticated,
        "park_b_nodes_tested": len(results),
        "identity_checks_passed": len(results),
        "proposed_moves": [
            {"id": result["id"], "from": "PARK_B", "to": result["proposed_tier"]}
            for result in results
            if result["tier_move_proposed"]
        ],
        "unchanged_park_b": sum(not result["tier_move_proposed"] for result in results),
        "results": results,
        "memo_quarantine": {
            "C2": True,
            "branch_b_66_of_223": True,
            "numbers_used_for_tiering": 0,
        },
        "family_execution": {"44b6": False, "6bba": False, "CV57": False},
        "external_actions": {
            "gpu": False,
            "kaggle": False,
            "submission": False,
            "runpod": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
    }
    results_path = OUT / "SUPPORT_FALSIFIERS.json"
    results_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    results_sha = sha256(results_path)

    nodes = []
    for result, original in zip(results, park_b, strict=True):
        node = dict(original)
        node["owner_agent"] = "SA-PARKB-SWEEP"
        node["tier"] = result["proposed_tier"]
        node["evidence_artifact"] = (
            "reports/g02/screening/tiered_surface/branches/sa_parkb/SUPPORT_FALSIFIERS.json"
        )
        node["evidence_sha256"] = results_sha
        node["evidence_basis"] = result["verdict"]
        node["last_change_utc"] = "2026-07-28T06:55:00Z"
        nodes.append(node)
    fragment = {
        "schema": "celltrack-alpha.phase1-state-fragment.v1",
        "schema_version": 1,
        "updated_utc": "2026-07-28T06:55:00Z",
        "owner_agent": "SA-PARKB-SWEEP",
        "branch": "codex/sa-parkb-wave2",
        "plan_sha256": PLAN_SHA256,
        "objective": "S(G)=weighted_adjusted_edge_Jaccard + 0.1*division_Jaccard",
        "verdict": "VERDICT_BARRIER",
        "tier_move_proposed": True,
        "nodes": nodes,
        "family_execution": {"44b6_ran": False, "6bba_ran": False, "cv57_ran": False},
        "memo_quarantine": True,
        "forbidden_scope": {
            "census_write": False,
            "live_state_write": False,
            "ledger_write": False,
            "budget_write": False,
            "gpu_action": False,
            "submission_action": False,
            "runpod_action": False,
            "dead_promotion": False,
            "plan_amendment": False,
        },
    }
    (OUT / "state_fragment.json").write_text(
        json.dumps(fragment, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
