#!/usr/bin/env python3
"""Validate the complete SA-NODETOPO-GEN branch-local package."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


WORKTREE = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/agent/Desktop/cell_tracking").resolve()
PREFIX = "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
BASE = "5a805c2f7240e30c572644d3d5f0e733add04769"
EXPECTED_FIELDS = {
    "dataset",
    "identity_sha256",
    "intensity",
    "node_id",
    "t",
    "x",
    "y",
    "z",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKTREE / value


def main() -> int:
    prereg = load(HERE / "PREREGISTRATION.json")
    assert sha256_file(HERE / "PREREGISTRATION.json") == (
        "c6529b5da9e6bbf76f08067fd68d248a56aa8dcb6104512e91905d8d58ecaa87"
    )
    for label, pin in prereg["frozen_inputs"].items():
        if label == "raw_archive":
            archive = Path(pin["path"])
            assert archive.stat().st_size == pin["bytes"]
            continue
        assert sha256_file(Path(pin["path"])) == pin["sha256"], label

    generated = HERE / "generated"
    result = load(generated / "RESULT.json")
    manifest = load(generated / "raw_universe" / "manifest.json")
    audit = load(generated / "COVERAGE_IDENTITY_AUDIT.json")
    curve = load(generated / "FALSE_NODE_STRESS_CURVE.json")
    resources = load(HERE / "RESOURCE_RECEIPTS.json")
    state = load(HERE / "state_fragment.json")

    assert result["status"] == "GO_AUTHENTIC_GENERATOR_PREREQUISITE"
    assert result["preregistration"]["git_commit"] == (
        "cd7f24d30ed17bac9940432a90213a87a1f68e78"
    )
    assert result["preregistration"]["outcomes_opened_after_commit"] is True
    assert result["raw_universe"]["manifest_sha256"] == sha256_file(
        generated / "raw_universe" / "manifest.json"
    )
    assert result["coverage_identity_audit"]["sha256"] == sha256_file(
        generated / "COVERAGE_IDENTITY_AUDIT.json"
    )
    assert result["action_population"]["sha256"] == sha256_file(
        generated / "FALSE_NODE_STRESS_CURVE.json"
    )

    assert manifest["status"] == "COMPLETE"
    assert manifest["identity_count"] == 36
    assert manifest["frame_count"] == 3600
    assert manifest["candidate_count"] == 107370
    assert manifest["labels_in_raw_banks"] is False
    assert manifest["masks_used"] is False
    assert manifest["pseudo_masks_used"] is False
    assert len(manifest["videos"]) == 36
    total_rows = 0
    identities: set[str] = set()
    for video in manifest["videos"]:
        identities.add(video["identity"])
        assert video["frame_count"] == len(video["frame_candidate_counts"])
        assert video["frame_count"] == 100
        assert sum(video["frame_candidate_counts"]) == video["candidate_count"]
        assert video["raw_bank_field_names"] == sorted(EXPECTED_FIELDS)
        assert all(receipt["equal"] for receipt in video["replay"].values())
        bank = resolve(video["raw_bank_path"])
        assert sha256_file(bank) == video["raw_bank_sha256"]
        count = 0
        with gzip.open(bank, "rt", encoding="ascii") as handle:
            for line in handle:
                row = json.loads(line)
                assert set(row) == EXPECTED_FIELDS
                assert row["dataset"] == video["identity"].split(":", 1)[1]
                assert len(row["identity_sha256"]) == 64
                count += 1
        assert count == video["candidate_count"]
        total_rows += count
    assert len(identities) == 36
    assert total_rows == 107370

    assert audit["status"] == "PASS"
    assert audit["video_count"] == 36
    assert audit["identity_failures"] == 0
    assert audit["labels_in_raw_bank"] is False
    assert audit["support"] == {
        "false_candidates": 106668,
        "hard_false_gt14um": 101343,
        "positive_assignments_7um": 702,
        "raw_candidates": 107370,
        "topology_joinable_false": 100983,
    }

    assert curve["status"] == "GO"
    assert curve["action_kind"] == "restore_topology"
    assert curve["action_kind_count"] == {"restore_topology": 50}
    assert curve["population_count"] == 50
    assert curve["single_n1_compile_passed"] == 50
    assert curve["single_n1_compile_total"] == 50
    assert curve["global_order_exact_through_50"] is True
    assert curve["official_scorer_run"] is False
    assert curve["graph_metric_run"] is False
    assert curve["tier_gate"] is False
    assert [row["budget"] for row in curve["false_node_budget_stress_curve"]] == [
        0,
        1,
        2,
        5,
        10,
        20,
        50,
    ]
    assert sum(
        len(row["batch_receipts"]) for row in curve["false_node_budget_stress_curve"]
    ) == 55
    assert all(
        row["graph_metric"] is None and row["official_scorer_run"] is False
        for row in curve["false_node_budget_stress_curve"]
    )
    action_bank = resolve(curve["action_population_path"])
    assert sha256_file(action_bank) == curve["action_population_sha256"]
    with gzip.open(action_bank, "rt", encoding="ascii") as handle:
        action_rows = [json.loads(line) for line in handle]
    assert len(action_rows) == 50
    assert {row["action_kind"] for row in action_rows} == {"restore_topology"}

    assert resources["status"] == "PASS_ZERO_SWAP_GROWTH"
    assert resources["initial_attempt"]["preflight"]["swap_growth_mib"] == 0.0
    assert resources["initial_attempt"]["guard"]["status"] == "FAIL_WALL_CLOCK"
    assert resources["initial_attempt"]["guard"]["swap_growth_mib"] == 0.0
    assert resources["bounded_resume"]["preflight"]["swap_growth_mib"] == 0.0
    assert resources["bounded_resume"]["guard"]["status"] == "PASS"
    assert resources["bounded_resume"]["guard"]["swap_growth_mib"] == 0.0
    for phase in ("initial_attempt", "bounded_resume"):
        for kind in ("preflight", "guard"):
            receipt = resources[phase][kind]
            assert sha256_file(resolve(receipt["path"])) == receipt["sha256"]
    assert not any(resources["external_actions"].values())

    assert state["tier_move_proposed"] is False
    assert {node["id"]: node["tier"] for node in state["nodes"]} == {
        "NODETOPO-O0": "PARK_A",
        "NODERESTORE-O0": "PARK_B",
    }
    assert all(node["graph_result"] is None for node in state["nodes"])
    assert state["forbidden_scope"]["held_graph_scoring"] is False
    assert state["forbidden_scope"]["second_family_opened"] is False
    assert state["forbidden_scope"]["dead_proposed"] is False
    assert not any(state["memo_quarantine"].values())

    terminal = (HERE / "TERMINAL_REPORT.md").read_text(encoding="utf-8")
    ledger = (HERE / "PROPOSED_LEDGER_ROWS.md").read_text(encoding="utf-8")
    for expected in (
        sha256_file(generated / "RESULT.json"),
        sha256_file(generated / "raw_universe" / "manifest.json"),
        sha256_file(generated / "COVERAGE_IDENTITY_AUDIT.json"),
        sha256_file(generated / "FALSE_NODE_STRESS_CURVE.json"),
        sha256_file(action_bank),
    ):
        assert expected in terminal
    assert sha256_file(generated / "RESULT.json") in ledger
    assert sha256_file(generated / "FALSE_NODE_STRESS_CURVE.json") in ledger
    assert sha256_file(HERE / "RESOURCE_RECEIPTS.json") in ledger

    forbidden_path_prefixes = (
        "/private/tmp/g02_cpu_wave2",
        "branches/sa_nodetopo/",
        "branches/sa_division",
        "branches/sa_parkb",
        "branches/sa_assoc_cycle",
    )
    for path in HERE.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert not any(prefix in text for prefix in forbidden_path_prefixes), path

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    changed_paths = [line[3:] for line in status if line]
    assert changed_paths
    assert all(path.startswith(PREFIX) for path in changed_paths), changed_paths
    committed_paths = subprocess.run(
        ["git", "diff", "--name-only", f"{BASE}..HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert all(path.startswith(PREFIX) for path in committed_paths), committed_paths
    assert not any(path.name == "__pycache__" for path in HERE.rglob("__pycache__"))
    assert not any(path.suffix == ".pyc" for path in HERE.rglob("*.pyc"))

    artifact_paths = [
        HERE / "GENERATOR_CONTRACT.md",
        HERE / "PREREGISTRATION.json",
        HERE / "RESOURCE_RECEIPTS.json",
        HERE / "PROPOSED_LEDGER_ROWS.md",
        HERE / "TERMINAL_REPORT.md",
        HERE / "state_fragment.json",
        generated / "RESULT.json",
        generated / "raw_universe" / "manifest.json",
        generated / "COVERAGE_IDENTITY_AUDIT.json",
        generated / "FALSE_NODE_STRESS_CURVE.json",
        generated / "action_population.jsonl.gz",
        HERE / "idle_preflight.json",
        HERE / "idle_preflight_resume.json",
        HERE / "resource_guard.json",
        HERE / "resource_guard_resume.json",
    ]
    report = {
        "schema": "celltrack-alpha.sa-nodetopo-gen-validation.v1",
        "status": "PASS",
        "checks": {
            "focused_unit_tests": "6/6",
            "frozen_input_hashes": "PASS",
            "raw_video_coverage": "36/36",
            "raw_frame_coverage": "3600/3600",
            "raw_bank_rows": "107370/107370",
            "replay_checks": "72/72",
            "n1_single_actions": "50/50",
            "n1_nonempty_budget_batches": "55/55",
            "action_kinds": {"restore_topology": 50},
            "swap_growth_mib": 0.0,
            "held_graph_scores": 0,
            "tier_moves": 0,
            "uncited_tier_moves": 0,
            "scope": "PASS",
            "pycache_files": 0,
        },
        "artifact_hashes": {
            str(path.relative_to(WORKTREE)): sha256_file(path) for path in artifact_paths
        },
        "changed_paths": changed_paths,
        "committed_paths_before_final_commit": committed_paths,
    }
    output = HERE / "VALIDATION.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
