#!/usr/bin/env python3
"""Build the root-owned G-02 tier census and weekly Kaggle budget."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "reports/g02/screening/tiered_surface"
CENSUS = SURFACE / "tier_census.json"
BUDGET = SURFACE / "kaggle_budget.json"
UPDATED = "2026-07-28T00:00:00Z"

PLAN_SHA = "a322724e77dd77f53c36e918e5345e86e8394a3848033a9a2225c21924e75116"
P0_PATH = "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
P0_SHA = "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06"

BRANCHES = {
    "association_expansion": (
        "reports/g02/screening/tiered_surface/branches/association_expansion.md",
        "37dd4b9a31b5b8cf420dc6744c97d5165bc29faa83cd3d7990ebdafffbc6f5be",
    ),
    "event_node_actions": (
        "reports/g02/screening/tiered_surface/branches/event_node_actions.md",
        "6df46ab0b3744a89b588d55d24e71a687f70e261667491adf6e2552c1b5fb5b3",
    ),
    "representation_and_governance": (
        "reports/g02/screening/tiered_surface/branches/"
        "representation_and_governance.md",
        "cdcc7214c7fe70c51a08d96a861a46920779107ad13b38dd002997ec5fb8a153",
    ),
}

NODE_TIERS = {
    "association_expansion": {
        "AE-P0": "PARK_A",
        "AE-P1": "PARK_A",
        "AE-P2": "PARK_A",
        "AE-P3": "PARK_A",
        "AE-P4": "PARK_C",
        "AE-P5": "PARK_B",
        "AE-P6": "PARK_B",
        "AE-P7": "PARK_B",
        "AE-P8": "PARK_B",
        "AE-P9": "PARK_B",
        "AE-P10": "PARK_C",
    },
    "event_node_actions": {
        "DIV-O0": "PARK_A",
        "DIV-F0": "PARK_C",
        "DIV-F1": "PARK_A",
        "DIV-F2": "PARK_B",
        "GAP-O0": "PARK_C",
        "GAP-F1": "PARK_B",
        "NODEADD-O0": "PARK_C",
        "NODERESTORE-O0": "PARK_B",
        "DET-F0": "PARK_B",
        "DET-F1": "PARK_C",
        "DET-F2": "PARK_B",
        "DET-F3": "PARK_B",
    },
    "representation_and_governance": {
        "R0": "DEAD",
        "R1": "PARK_C",
        "R2": "PARK_B",
        "R3": "PARK_B",
        "R4": "PARK_B",
        "R5": "PARK_B",
        "R6": "PARK_B",
        "R7": "PARK_B",
        "R8": "PARK_B",
        "R9": "PARK_B",
        "R10": "PARK_B",
        "G0": "PARK_A",
        "G1": "PARK_A",
        "G2": "PARK_B",
        "G3": "PARK_A",
        "G4": "PARK_A",
        "G5": "PARK_A",
        "G6": "PARK_B",
    },
    "atomic_p0": {
        "ADD-O0": "PARK_C",
        "DELETE-O0": "PARK_A",
        "SWAP-O0": "PARK_A",
        "NODETOPO-O0": "PARK_A",
    },
}

HEADROOM = {
    "ADD-O0": (0.0, 0.0),
    "DELETE-O0": (0.08877378690902493, 0.07483987337673481),
    "SWAP-O0": (0.075827918, 0.059735),
    "NODETOPO-O0": (0.009812493435202185, 0.024040),
    "DIV-O0": (0.014723031692599364, 0.02140505577004803),
    "DIV-F1": (0.014723031692599364, 0.02140505577004803),
    "AE-P1": (0.075827918, 0.059735),
    "AE-P2": (0.075827918, 0.059735),
    "AE-P3": (0.0, 0.0),
    "AE-P4": (0.003638840028941659, 0.004424175906171723),
    "GAP-O0": (0.003638840028941659, 0.004424175906171723),
    "NODEADD-O0": (0.0, 0.0),
}

BLOCKED_ON = {
    "AE-P0": ["COMPLETED_P0_ENABLER"],
    "AE-P1": [],
    "AE-P2": [],
    "AE-P3": ["AE-P2_BOTH_FAMILY_PASS"],
    "AE-P4": [],
    "DIV-O0": ["ASSOCIATION_CONTINUITY_PASS"],
    "DIV-F0": ["ASSOCIATION_CONTINUITY_PASS"],
    "DIV-F1": ["ASSOCIATION_CONTINUITY_PASS"],
    "DIV-F2": ["ASSOCIATION_CONTINUITY_PASS", "AUTHENTIC_PRETRAINED_FORK_INPUT"],
    "GAP-O0": [],
    "GAP-F1": ["POSITIVE_GAP_CLASS_GATE"],
    "NODEADD-O0": ["TOPOLOGY_REQUIRED"],
    "NODERESTORE-O0": ["AUTHENTIC_NODE_PROPOSAL_UNIVERSE"],
    "DET-F0": ["AUTHENTIC_NODE_PROPOSAL_UNIVERSE"],
    "DET-F2": ["AUTHENTIC_NODE_PROPOSAL_UNIVERSE"],
    "DET-F3": ["AUTHENTIC_NODE_PROPOSAL_UNIVERSE"],
    "R2": ["OWNER_PROVENANCE_COMPLETE_MASK_ARTIFACT"],
    "R3": ["R2"],
    "R4": ["AUTHENTIC_ENCODER_WEIGHTS"],
    "R5": ["R4"],
    "R6": ["R4", "R5"],
    "R7": ["AUTHENTIC_MICROSCOPY_ENCODER"],
    "R8": [],
    "R9": [],
    "R10": ["AUTHENTIC_NEW_REPRESENTATION"],
    "G0": ["COMPLETED_GOVERNANCE"],
    "G1": ["COMPLETED_N1_ENABLER"],
    "G2": [],
    "G3": ["SERIAL_GPU_AUTHORIZATION"],
    "G4": ["EXECUTION_DISCIPLINE_NOT_SCIENTIFIC_ARM"],
    "G5": ["CV57_PASS", "EXPLICIT_HUMAN_SUBMISSION_APPROVAL"],
    "G6": ["OWNER_FORBIDS_RUNPOD"],
    "ADD-O0": ["REALIZER_REQUIRED"],
    "DELETE-O0": ["RETENTION_REALIZER_REQUIRED"],
    "SWAP-O0": ["SWAP_REALIZER_REQUIRED"],
    "NODETOPO-O0": ["AUTHENTIC_NODE_PROPOSAL_UNIVERSE"],
}

RESUME = {
    "G1": "Frozen and complete; expose only this exact interface to Phase 1.",
    "DELETE-O0": "Fit a conservative true-edge retention/no-link beta.",
    "SWAP-O0": "Fit an occupied-conflict exact-regret alpha with explicit no-op.",
    "NODETOPO-O0": "Construct an authentic proposal universe and harmful-node budget.",
    "DIV-O0": "Wait for a both-family association-continuity pass.",
    "R2": "Supply a provenance-, license-, coverage-, and join-complete mask artifact.",
    "R4": "Authenticate frozen weights, source revision, license, and offline loader.",
    "G3": "Open only one root-recorded GPU authorization at a time.",
    "G5": "Require a passing CV57 package and explicit human approval.",
    "G6": "Do not open autonomously; RunPod requires explicit approval under LMT.md.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_for(branch: str, node_id: str) -> tuple[str, str, str]:
    if node_id == "G1":
        path = (
            "reports/g02/screening/tiered_surface/n1_feasibility_compiler/"
            "N1_FREEZE_REPORT.json"
        )
        return path, sha256(ROOT / path), "N1 GO: 100% structural validity"
    if node_id == "R0":
        return (
            "reports/g02/screening/ga_h1_zero_wrfeat_graph/44b6/report.json",
            "530117276bbea4911c4ed2422a6b5b2d6cfc76120dfb28bd8a0185f85b19619d",
            "human-promoted 44b6 graph kill",
        )
    if branch == "atomic_p0":
        return P0_PATH, P0_SHA, "exact fit117 P0 action-class oracle"
    path, digest = BRANCHES[branch]
    return path, digest, "P0-adjusted banked branch register"


def make_node(branch: str, node_id: str, tier: str) -> dict[str, Any]:
    evidence_path, evidence_hash, basis = evidence_for(branch, node_id)
    headroom = HEADROOM.get(node_id)
    node: dict[str, Any] = {
        "id": node_id,
        "branch": branch,
        "owner_agent": "UNASSIGNED_PHASE1",
        "tier": tier,
        "banked": True,
        "evidence_artifact": evidence_path,
        "evidence_sha256": evidence_hash,
        "evidence_basis": basis,
        "oracle_headroom": (
            {
                "family_44b6_dS": headroom[0],
                "family_6bba_dS": headroom[1],
                "control": "CG0_fit117",
            }
            if headroom is not None
            else {
                "family_44b6_dS": 0.0,
                "family_6bba_dS": 0.0,
                "control": "not_measured_or_governance",
            }
        ),
        "graph_result": None,
        "kill_line": (
            "44b6 adjusted>=+0.005, raw>=0, node>=0; then identical 6bba gate"
        ),
        "resume_condition": RESUME.get(
            node_id,
            "Follow the node-specific frozen branch contract before outcomes open.",
        ),
        "blocked_on": BLOCKED_ON.get(node_id, []),
        "priority": {"PARK_A": 1, "PARK_B": 2, "PARK_C": 3, "DEAD": 4}[tier],
        "last_change_utc": UPDATED,
    }
    if node_id == "R0":
        node.update(
            {
                "human_authorized": True,
                "named_mechanism": "H1-ZWRFEAT",
                "kill_line": "already fired at 44b6; 6bba and CV57 remain closed",
                "resume_condition": (
                    "Never re-screen zero-filled wrfeat; require a new named "
                    "representation."
                ),
                "blocked_on": ["NEW_NAMED_MECHANISM_REQUIRED"],
            }
        )
    if node_id == "G1":
        node["owner_agent"] = "ROOT"
        node["compiler"] = {
            "interface_sha256": (
                "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865"
            ),
            "implementation_sha256": (
                "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5"
            ),
            "tests_sha256": (
                "09107306fbdea2078d5a73006bde6215a242779ae9245015e365181dd48bee97"
            ),
            "unit_tests": "10/10 PASS",
            "single_action_kinds": "6/6 PASS",
            "batch_orderings": "2/2 PASS",
            "negative_fixtures": "7/7 REJECTED",
            "swap_growth_mib": 0.0,
        }
        node["kill_line"] = "any invalid or nondeterministic action fails closed"
    return node


def main() -> int:
    if CENSUS.exists() or BUDGET.exists():
        raise SystemExit("refusing to overwrite root-owned tier state")

    nodes = [
        make_node(branch, node_id, tier)
        for branch, entries in NODE_TIERS.items()
        for node_id, tier in entries.items()
    ]
    counts = Counter(node["tier"] for node in nodes)
    expected = {"LIVE": 0, "PARK_A": 14, "PARK_B": 22, "PARK_C": 8, "DEAD": 1}
    rollup = {key: counts.get(key, 0) for key in expected}
    if len(nodes) != 45 or rollup != expected:
        raise SystemExit(f"tier rollup mismatch: nodes={len(nodes)} rollup={rollup}")

    budget = {
        "schema": "celltrack-alpha.kaggle-weekly-budget.v1",
        "updated_utc": UPDATED,
        "week_start_utc": "2026-07-27T00:00:00Z",
        "week_end_utc_exclusive": "2026-08-03T00:00:00Z",
        "budget_kernel_h": 30.0,
        "used_kernel_h": 2.117,
        "remaining_kernel_h": 27.883,
        "autonomy_policy_authority": "LMT.md",
        "historical_policy": {
            "auto_approve_projected_under_h": 5.0,
            "hard_kill_elapsed_h": 5.0,
            "status": "SUPERSEDED_BY_LMT",
        },
        "active_authorizations": [],
        "debits": [
            {
                "id": "H1-ZWRFEAT-SMOKE3",
                "kernel_h": 0.0822,
                "status": "completed",
            },
            {
                "id": "H1-ZWRFEAT-FULL-CACHE",
                "kernel_h": 2.0348,
                "status": "completed",
            },
        ],
        "runpod": "FORBIDDEN",
        "public_submission": "EXPLICIT_HUMAN_APPROVAL_REQUIRED",
    }
    if abs(
        sum(item["kernel_h"] for item in budget["debits"])
        - budget["used_kernel_h"]
    ) > 1e-12:
        raise SystemExit("Kaggle debit reconciliation failed")

    census = {
        "schema_version": 1,
        "updated_utc": UPDATED,
        "plan_sha256": PLAN_SHA,
        "objective": (
            "S(G)=weighted_adjusted_edge_Jaccard + 0.1*division_Jaccard"
        ),
        "gate": (
            "44b6->6bba->CV57; adjusted>=+0.005 both families; raw,node>=0"
        ),
        "phase": "N1_GO_PHASE1_READY_UNDISPATCHED",
        "frozen_inputs": {
            "p0_sha256": P0_SHA,
            "n1_interface_sha256": (
                "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865"
            ),
            "n1_tests_sha256": (
                "09107306fbdea2078d5a73006bde6215a242779ae9245015e365181dd48bee97"
            ),
            "grouped_folds_sha256": (
                "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
            ),
            "fit_provenance_sha256": (
                "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f"
            ),
        },
        "kaggle_week": {
            "week_start_utc": budget["week_start_utc"],
            "budget_h": budget["budget_kernel_h"],
            "used_h": budget["used_kernel_h"],
            "remaining_h": budget["remaining_kernel_h"],
            "autonomy_policy_authority": budget["autonomy_policy_authority"],
            "historical_policy": budget["historical_policy"],
            "budget_artifact": str(BUDGET.relative_to(ROOT)),
        },
        "rollup": rollup,
        "fallback_order": ["PARK_A", "PARK_B", "PARK_C"],
        "fallback_sort": (
            "lowest priority, then descending oracle_headroom.family_44b6_dS, "
            "excluding nodes with blocked_on"
        ),
        "nodes": nodes,
    }
    BUDGET.write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    CENSUS.write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "census": str(CENSUS.relative_to(ROOT)),
                "census_sha256": sha256(CENSUS),
                "budget": str(BUDGET.relative_to(ROOT)),
                "budget_sha256": sha256(BUDGET),
                "rollup": rollup,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
