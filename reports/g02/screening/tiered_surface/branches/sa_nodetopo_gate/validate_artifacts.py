#!/usr/bin/env python3
"""Validate terminal SA-NODETOPO-GATE artifacts without rerunning the scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite validation output")
    result = load(HERE / "RESULT_44B6.json")
    preflight = load(HERE / "idle_preflight_retry.json")
    guard = load(HERE / "local_run_guard_retry.json")
    state = load(HERE / "state_fragment.json")
    checks: dict[str, bool] = {}
    checks["verdict"] = result["status"] == "KILL_44B6_STOP"
    checks["family_44b6_only"] = (
        result["family"] == "44b6"
        and result["scope"]["second_family_opened"] is False
        and result["scope"]["cv57_used"] is False
    )
    population = result["population"]
    checks["population_50_24_36"] = (
        population["actions"] == 50
        and population["affected_videos"] == 24
        and population["scored_videos"] == 36
        and population["single_n1_replays"] == 50
        and population["batch_n1_replays"] == 24
    )
    per_video = result["per_video"]
    checks["per_video_coverage"] = (
        len(per_video) == 36
        and sum(int(row["action_count"]) for row in per_video) == 50
        and sum(int(row["action_count"]) > 0 for row in per_video) == 24
        and sum(row["batch_receipt"] is not None for row in per_video) == 24
    )
    checks["affected_graphs_observation_distinct"] = all(
        row["batch_receipt"]["before_graph_sha256"]
        != row["batch_receipt"]["after_graph_sha256"]
        for row in per_video
        if row["batch_receipt"] is not None
    )
    baseline = result["baseline"]
    candidate = result["candidate"]
    decision = result["decision"]
    adjusted = float(candidate["adj_edge_jaccard"]) - float(
        baseline["adj_edge_jaccard"]
    )
    raw = float(candidate["edge_jaccard"]) - float(baseline["edge_jaccard"])
    node = float(candidate["node_recall"]) - float(baseline["node_recall"])
    checks["delta_recomputed"] = (
        math.isclose(
            adjusted,
            float(decision["adjusted_edge_jaccard_delta"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and raw == float(decision["raw_edge_jaccard_delta"])
        and node == float(decision["node_recall_delta"])
    )
    checks["gate_recomputed"] = (
        adjusted < 0.005
        and raw >= 0.0
        and node >= 0.0
        and decision["checks"]
        == {
            "adjusted_ge_0p005": False,
            "raw_ge_0": True,
            "node_ge_0": True,
        }
    )
    checks["edge_counts_exact"] = all(
        int(baseline[key]) == expected and int(candidate[key]) == expected
        for key, expected in (("edge_tp", 10341), ("edge_fp", 1064), ("edge_fn", 673))
    )
    checks["preflight"] = (
        preflight["status"] == "PASS_ZERO_SWAP_GROWTH"
        and float(preflight["elapsed_seconds"]) >= 45.0
        and float(preflight["swap_growth_mib"]) == 0.0
    )
    checks["guard"] = (
        guard["status"] == "PASS"
        and int(guard["returncode"]) == 0
        and float(guard["swap_growth_mib"]) == 0.0
        and guard["preflight_sha256"] == sha256(HERE / "idle_preflight_retry.json")
    )
    checks["instantiation_only"] = (
        decision["instantiation_tier_proposal"]["id"] == "BLOCKPEAK-RESTORE-V1"
        and decision["instantiation_tier_proposal"]["tier"] == "PARK_C"
        and decision["instantiation_tier_proposal"]["broader_family_tier_changes"] == []
        and state["tier_move_proposed"] is False
        and state["nodes"][0]["tier"] == "PARK_A"
        and state["nodes"][1]["tier"] == "PARK_B"
    )
    checks["no_downstream_unblocks"] = (
        decision["pass_unblocks_proposed"] == []
        and all(value is False for value in state["downstream_unblocks"].values())
    )
    forbidden = result["scope"]
    checks["forbidden_scope"] = all(
        forbidden[key] is False
        for key in (
            "cv57_used",
            "dead_proposed",
            "model_fitted",
            "plan_amended",
            "preexisting_untracked_root_graph_gate_outputs_used",
            "root_single_writer_files_written",
            "second_family_opened",
            "sibling_live_outputs_used",
            "threshold_selected",
        )
    )
    artifact_names = (
        "PREREGISTRATION.json",
        "PRE_OUTCOME_MANIFEST.json",
        "EXECUTION_REPAIR.json",
        "RESULT_44B6.json",
        "idle_preflight.json",
        "local_run_guard.json",
        "idle_preflight_retry.json",
        "local_run_guard_retry.json",
        "RESOURCE_RECEIPTS.json",
        "TERMINAL_REPORT.md",
        "PROPOSED_LEDGER_ROWS.md",
        "state_fragment.json",
        "evaluate_graph_gate.py",
        "run_preflight.py",
        "run_guarded.py",
        "test_gate.py",
        "validate_artifacts.py",
    )
    artifact_hashes = {name: sha256(HERE / name) for name in artifact_names}
    checks["all_artifacts_present"] = len(artifact_hashes) == len(artifact_names)
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "celltrack-alpha.sa-nodetopo-gate-validation.v1",
        "status": status,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "artifact_hashes": artifact_hashes,
        "scope": {
            "write_root": "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gate/",
            "root_single_writer_files_written": False,
            "external_action": False,
            "gpu_action": False,
            "kaggle_action": False,
            "budget_debit": False,
            "runpod_action": False,
            "submission_action": False,
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({"status": status, "passed": report["passed"], "total": report["total"]}))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
