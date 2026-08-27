#!/usr/bin/env python3
"""Root-only deterministic reconciliation of the 2026-07-28 CPU wave."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[4]
SURFACE = ROOT / "reports/g02/screening/tiered_surface"
CENSUS = SURFACE / "tier_census.json"
KAGGLE_BUDGET = SURFACE / "kaggle_budget.json"
UPDATED_UTC = "2026-07-28T07:30:00Z"

LANES = [
    {
        "name": "SA-PARKB-SWEEP",
        "commit": "3bb73ef14260279fb00985dc73c4a5c710e9cf38",
        "scope": "reports/g02/screening/tiered_surface/branches/sa_parkb/",
        "fragment": "reports/g02/screening/tiered_surface/branches/sa_parkb/state_fragment.json",
    },
    {
        "name": "SA-REPR-MULTI",
        "commit": "cefd1063d42bfbeb476bc6bce85ab9d8ef8b9327",
        "scope": "reports/g02/screening/tiered_surface/branches/sa_repr_multi/",
        "fragment": "reports/g02/screening/tiered_surface/branches/sa_repr_multi/state_fragment.json",
    },
    {
        "name": "SA-DIVISION",
        "commit": "0bfa879ab6c8a289085c420e886190e7cc9cd2f4",
        "scope": "reports/g02/screening/tiered_surface/branches/sa_division/",
        "fragment": "reports/g02/screening/tiered_surface/branches/sa_division/state_fragment.json",
    },
    {
        "name": "SA-NODETOPO",
        "commit": "27b72209f6cf7128b9f8d6ee6c1018b2acb8ea42",
        "scope": "reports/g02/screening/tiered_surface/branches/sa_nodetopo/",
        "fragment": "reports/g02/screening/tiered_surface/branches/sa_nodetopo/state_fragment.json",
    },
]

REJECTIONS = {
    "GAP-F1": (
        "Reject PARK-B->PARK-C: its frozen rule demotes only after the first fixed "
        "new bridge mechanism fails 44b6. This lane ran no mechanism or graph; P0 "
        "gap-oracle headroom below +0.005 is not a realizer result."
    ),
    "DET-F0": (
        "Reject PARK-B->PARK-C: the detection-family parent remains PARK-B when an "
        "individual child fails, and no authentic node-add/node-restore universe "
        "completed. SA-NODETOPO consumed zero videos and no metric."
    ),
    "G2": (
        "Reject PARK-B->PARK-A: promotion requires G1 plus a common both-family "
        "graph pass and positive exact regret versus no-op. Identity/support checks "
        "and P0 action headroom do not satisfy that transition."
    ),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    census = json.loads(CENSUS.read_text())
    by_id = {node["id"]: node for node in census["nodes"]}
    lane_receipts = []
    uncited_moves = []
    proposal_decisions = []

    for lane in LANES:
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", lane["commit"]],
            cwd=ROOT,
            text=True,
        ).splitlines()
        assert changed and all(path.startswith(lane["scope"]) for path in changed)
        fragment_path = ROOT / lane["fragment"]
        fragment = json.loads(fragment_path.read_text())
        assert fragment.get("memo_quarantine") is True
        lane_receipts.append(
            {
                **lane,
                "changed_paths": len(changed),
                "all_paths_scoped": True,
                "fragment_sha256": sha256(fragment_path),
                "verdict": fragment.get("verdict") or fragment.get("lane_verdict"),
            }
        )
        for candidate in fragment.get("nodes", []):
            node_id = candidate["id"]
            assert node_id in by_id
            current = by_id[node_id]
            proposed_tier = candidate["tier"]
            if proposed_tier != current["tier"]:
                reason = REJECTIONS.get(node_id)
                if reason is None:
                    uncited_moves.append(
                        {
                            "lane": lane["name"],
                            "node": node_id,
                            "from": current["tier"],
                            "to": proposed_tier,
                        }
                    )
                    continue
                proposal_decisions.append(
                    {
                        "lane": lane["name"],
                        "node": node_id,
                        "proposed_from": current["tier"],
                        "proposed_to": proposed_tier,
                        "decision": "REJECT",
                        "reason": reason,
                    }
                )
                continue
            artifact = ROOT / candidate["evidence_artifact"]
            assert artifact.is_file()
            assert sha256(artifact) == candidate["evidence_sha256"]
            by_id[node_id] = candidate

    assert not uncited_moves

    # The independent serial R7 lane is deliberately outside this CPU wave and
    # may advance while these fragments run. Its newer terminal state always
    # wins over SA-REPR-MULTI's dispatch-time snapshot.
    r7_terminal = (
        SURFACE
        / "branches/sa_repr/phase2_separation_gate/TERMINAL_REPORT_V3.md"
    )
    if r7_terminal.is_file():
        r7 = by_id["R7"]
        r7.update(
            {
                "blocked_on": [
                    "TERMINAL_INFRASTRUCTURE_HOLD",
                    "NO_FURTHER_AUTHORIZED_RETRY",
                ],
                "evidence_artifact": str(r7_terminal.relative_to(ROOT)),
                "evidence_basis": (
                    "The official CELLECT checkpoint and v3 package authenticated, "
                    "but the one authorized serial kernel observed no mounted file "
                    "under the exact weight name and stopped before load/features. "
                    "No A1/A0 metric or scientific verdict exists."
                ),
                "evidence_sha256": sha256(r7_terminal),
                "last_change_utc": "2026-07-28T07:29:28Z",
                "owner_agent": "ROOT_SERIAL_R7",
                "resume_condition": (
                    "Remain PARK-B; no further launch is authorized without a new "
                    "explicit human instruction and a new frozen contract."
                ),
            }
        )
        by_id["R7"] = r7

    census["nodes"] = [by_id[node["id"]] for node in census["nodes"]]
    counts: dict[str, int] = {}
    for node in census["nodes"]:
        counts[node["tier"]] = counts.get(node["tier"], 0) + 1
    assert counts == {"PARK_A": 13, "PARK_B": 22, "PARK_C": 9, "DEAD": 1}
    census["rollup"] = {key: counts[key] for key in ("DEAD", "PARK_A", "PARK_B", "PARK_C")}
    census["updated_utc"] = UPDATED_UTC
    census["active_agents"] = []
    budget = json.loads(KAGGLE_BUDGET.read_text())
    census["kaggle_week"].update(
        {
            "remaining_h": budget["remaining_kernel_h"],
            "used_h": budget["used_kernel_h"],
            "budget_artifact_sha256": sha256(KAGGLE_BUDGET),
        }
    )
    CENSUS.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")

    decisions = {
        "schema": "celltrack-alpha.wave2-root-barrier.v1",
        "updated_utc": UPDATED_UTC,
        "status": "PASS",
        "lanes": lane_receipts,
        "proposal_decisions": proposal_decisions,
        "accepted_tier_moves": [],
        "uncited_tier_moves": uncited_moves,
        "memo_c2_numbers_used": False,
        "memo_branch_b_numbers_used": False,
        "partial_metrics_consumed": False,
        "comparison_rules_changed": False,
        "independent_serial_r7_preserved": True,
        "kaggle_budget_sha256": sha256(KAGGLE_BUDGET),
        "kaggle_budget_used_h": budget["used_kernel_h"],
        "kaggle_budget_remaining_h": budget["remaining_kernel_h"],
        "forbidden_actions": {
            "gpu_or_kaggle": False,
            "runpod": False,
            "submission": False,
            "dead_promotion": False,
            "plan_amendment": False,
        },
        "final_rollup": counts,
        "census_sha256": sha256(CENSUS),
    }
    decisions_path = SURFACE / "wave2_barrier_decisions.json"
    decisions_path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n")
    validation = {
        "schema": "celltrack-alpha.wave2-root-validation.v1",
        "status": "PASS",
        "checks": {
            "lane_count": len(LANES),
            "all_commits_scope_clean": all(item["all_paths_scoped"] for item in lane_receipts),
            "all_evidence_hashes_resolved": True,
            "all_proposed_moves_adjudicated": len(proposal_decisions) == 3,
            "uncited_tier_moves": len(uncited_moves),
            "memo_quarantine": True,
            "rollup_unchanged": counts == {
                "PARK_A": 13,
                "PARK_B": 22,
                "PARK_C": 9,
                "DEAD": 1,
            },
        },
        "decisions_sha256": sha256(decisions_path),
        "census_sha256": sha256(CENSUS),
    }
    (SURFACE / "wave2_barrier_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
