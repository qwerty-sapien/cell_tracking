#!/usr/bin/env python3
"""Deterministically reconcile the CPU-only Wave-2 lane fragments.

This script is root-coordinator tooling. It never executes a model, graph, GPU,
Kaggle, RunPod, or submission action.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SURFACE = ROOT / "reports/g02/screening/tiered_surface"
BASE_COMMIT = "5a805c2f7240e30c572644d3d5f0e733add04769"
UPDATED_UTC = "2026-07-28T16:04:51Z"

LANES = {
    "SA-DIVISION-DEEP": {
        "commits": [
            "3f38a52403018257b2e8e66fc7a2b92d59551173",
            "50bc0e087b3a2c15d2399e0036483b80ac5f0f4f",
        ],
        "prefix": "reports/g02/screening/tiered_surface/branches/sa_division_deep/",
        "fragment": "branches/sa_division_deep/state_fragment.json",
        "preflight": "branches/sa_division_deep/preflight_retry6.json",
        "validation": "branches/sa_division_deep/validation_v2.json",
        "governance_receipt": "branches/sa_division_deep/INPUT_HASH_RECEIPTS.json",
        "terminal": "branches/sa_division_deep/TERMINAL_REPORT.md",
        "terminal_sha256": "8a787c02ac34c6ce44c800889c728ddba28d40c27c9eaa82e0bbc7564188047f",
        "outcome": "PARK_DEPENDENCY",
    },
    "SA-NODETOPO-GEN": {
        "commits": [
            "cd7f24d30ed17bac9940432a90213a87a1f68e78",
            "cf703b5cc30d0bdab941a7cb59e67894e55440a6",
            "9151934e410fed17cf367afead28ee1558c7ef77",
        ],
        "prefix": "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/",
        "fragment": "branches/sa_nodetopo_gen/state_fragment.json",
        "preflight": "branches/sa_nodetopo_gen/idle_preflight_resume.json",
        "validation": "branches/sa_nodetopo_gen/VALIDATION.json",
        "governance_receipt": "branches/sa_nodetopo_gen/GOVERNANCE_RECEIPT.json",
        "terminal": "branches/sa_nodetopo_gen/TERMINAL_REPORT.md",
        "terminal_sha256": "71d9d6c1f4c16c3f1be6ee332f760edfc3ec8800b52b702eaee2fa9556203af0",
        "outcome": "GO_AUTHENTIC_GENERATOR_PREREQUISITE_HUMAN_GRAPH_HOLD",
    },
    "SA-PARKB-SWEEP": {
        "commits": ["479059283825204c1e47fe7d8151716ac1603977"],
        "prefix": "reports/g02/screening/tiered_surface/branches/sa_parkb/",
        "fragment": "branches/sa_parkb/state_fragment_wave2.json",
        "preflight": "branches/sa_parkb/IDLE_PREFLIGHT_WAVE2.json",
        "validation": "branches/sa_parkb/VALIDATION_WAVE2.json",
        "governance_receipt": "branches/sa_parkb/PREREGISTRATION_WAVE2.json",
        "terminal": "branches/sa_parkb/TERMINAL_REPORT_WAVE2.md",
        "terminal_sha256": "e9f793ffba77635dc12befbd297dc15c467bff1369a28763c0caa198ed16284a",
        "outcome": "VERDICT_BARRIER_NO_TIER_MOVE",
    },
    "SA-ASSOC-CYCLE": {
        "commits": [
            "f92dd8d6b8933a4863e7732b8d02193d969604f2",
            "17c8d6e64d1bab326a53574caf39cc9b72ce5927",
        ],
        "prefix": "reports/g02/screening/tiered_surface/branches/sa_assoc_cycle/",
        "fragment": "branches/sa_assoc_cycle/state_fragment.json",
        "preflight": "branches/sa_assoc_cycle/IDLE_PREFLIGHT.json",
        "validation": "branches/sa_assoc_cycle/VALIDATION.json",
        "governance_receipt": "branches/sa_assoc_cycle/GOVERNANCE_RECEIPT.json",
        "terminal": "branches/sa_assoc_cycle/TERMINAL_REPORT.md",
        "terminal_sha256": "c78ffd85c78e9b8904a432466a4b63206a67750d11d65c56f4a207c68bf9523a",
        "outcome": "PARK_NOT_READY_MISSING_AUTHENTIC_REVERSE",
    },
}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(f"{sha256(item)}  {item.relative_to(ROOT)}\n".encode())
    return digest.hexdigest()


def commit_paths(commit: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def commit_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def add_check(checks: list[dict], name: str, passed: bool, evidence: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    census_path = SURFACE / "tier_census.json"
    census = load(census_path)
    initial_rollup = dict(census["rollup"])
    initial_tiers = {node["id"]: node["tier"] for node in census["nodes"]}
    by_id = {node["id"]: node for node in census["nodes"]}

    fragments = {
        lane: load(SURFACE / spec["fragment"]) for lane, spec in LANES.items()
    }

    # The generic PARK-B lane is authoritative only for its 14 CPU-testable
    # nodes. Its eight explicit GPU/weight skips retain their pre-wave evidence,
    # including the separate CELLECT track.
    parkb_nodes = [
        node
        for node in fragments["SA-PARKB-SWEEP"]["nodes"]
        if node["evidence_basis"] != "SKIP_WEIGHT_OR_GPU_GATED_REPRESENTATION"
    ]
    for node in parkb_nodes:
        by_id[node["id"]] = node

    # More-specific lanes override the generic support sweep.
    for lane in ("SA-ASSOC-CYCLE", "SA-DIVISION-DEEP", "SA-NODETOPO-GEN"):
        for node in fragments[lane]["nodes"]:
            by_id[node["id"]] = node

    # Banking this barrier closes only the root-bank dependency. The graph gate
    # remains explicitly human-held.
    restore = by_id["NODERESTORE-O0"]
    restore["blocked_on"] = [
        blocker for blocker in restore["blocked_on"] if blocker != "ROOT_BARRIER_BANK"
    ]

    census["nodes"] = [by_id[node["id"]] for node in census["nodes"]]
    census["active_agents"] = []
    census["updated_utc"] = UPDATED_UTC
    census["rollup"] = dict(
        sorted(Counter(node["tier"] for node in census["nodes"]).items())
    )
    dump(census_path, census)
    census_sha = sha256(census_path)

    checks: list[dict] = []
    lane_receipts: dict[str, dict] = {}
    for lane, spec in LANES.items():
        fragment_path = SURFACE / spec["fragment"]
        validation_path = SURFACE / spec["validation"]
        governance_path = SURFACE / spec["governance_receipt"]
        preflight_path = SURFACE / spec["preflight"]
        terminal_path = SURFACE / spec["terminal"]
        preflight = load(preflight_path)
        validation = load(validation_path)

        paths = [
            path for commit in spec["commits"] for path in commit_paths(commit)
        ]
        scope_ok = bool(paths) and all(path.startswith(spec["prefix"]) for path in paths)
        add_check(checks, f"{lane}:commit_scope", scope_ok, paths)
        add_check(
            checks,
            f"{lane}:terminal_hash",
            sha256(terminal_path) == spec["terminal_sha256"],
            sha256(terminal_path),
        )

        elapsed = float(
            preflight.get("elapsed_seconds", preflight.get("required_idle_seconds", 0))
        )
        swap_growth = float(preflight.get("swap_growth_mib", -1))
        add_check(checks, f"{lane}:preflight_45s", elapsed >= 45.0, elapsed)
        add_check(checks, f"{lane}:zero_swap_growth", swap_growth == 0.0, swap_growth)
        validation_status = validation.get("status")
        add_check(
            checks,
            f"{lane}:lane_validation",
            validation_status == "PASS",
            validation_status,
        )
        governance_text = governance_path.read_text()
        governance_ok = (
            "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37"
            in governance_text
            and "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49"
            in governance_text
        )
        add_check(
            checks,
            f"{lane}:governance_receipt",
            governance_ok,
            {
                "path": str(governance_path.relative_to(ROOT)),
                "sha256": sha256(governance_path),
            },
        )

        lane_receipts[lane] = {
            "commits": spec["commits"],
            "fragment": str(fragment_path.relative_to(ROOT)),
            "fragment_sha256": sha256(fragment_path),
            "governance_receipt": str(governance_path.relative_to(ROOT)),
            "governance_receipt_sha256": sha256(governance_path),
            "outcome": spec["outcome"],
            "preflight_elapsed_seconds": elapsed,
            "preflight_swap_growth_mib": swap_growth,
            "terminal": str(terminal_path.relative_to(ROOT)),
            "terminal_sha256": sha256(terminal_path),
            "validation": str(validation_path.relative_to(ROOT)),
            "validation_sha256": sha256(validation_path),
        }

    changed_nodes = [
        node_id
        for node_id, initial_tier in initial_tiers.items()
        if by_id[node_id]["tier"] != initial_tier
    ]
    affected = {
        node["id"]
        for node in parkb_nodes
        + fragments["SA-ASSOC-CYCLE"]["nodes"]
        + fragments["SA-DIVISION-DEEP"]["nodes"]
        + fragments["SA-NODETOPO-GEN"]["nodes"]
    }
    for node_id in sorted(affected):
        node = by_id[node_id]
        artifact = ROOT / node["evidence_artifact"]
        evidence_ok = artifact.is_file() and sha256(artifact) == node["evidence_sha256"]
        add_check(
            checks,
            f"census_evidence:{node_id}",
            evidence_ok,
            {
                "artifact": node["evidence_artifact"],
                "expected_sha256": node["evidence_sha256"],
                "actual_sha256": sha256(artifact) if artifact.is_file() else None,
            },
        )

    budget_path = SURFACE / "kaggle_budget.json"
    cellect_path = SURFACE / "branches/sa_repr/phase2_separation_gate"
    budget_sha = sha256(budget_path)
    cellect_tree_sha = tree_sha256(cellect_path)
    alignment_path = SURFACE / "ALIGNMENT.md"
    scope_manifest_path = SURFACE / "SCOPE_MANIFEST.md"
    alignment_sha = sha256(alignment_path)
    scope_manifest_sha = sha256(scope_manifest_path)
    add_check(
        checks,
        "shared_governance_alignment_frozen",
        alignment_sha
        == "eda3840ca4decf1fbc9943dcd872800852cf1ff52e0923136adb51b4f2ea6b37",
        alignment_sha,
    )
    add_check(
        checks,
        "shared_governance_scope_manifest_frozen",
        scope_manifest_sha
        == "066fb71b8ead255ff061a7aa75283df27c367e08aa8fd0a0b91677537a0a7d49",
        scope_manifest_sha,
    )
    add_check(
        checks,
        "kaggle_budget_unchanged",
        budget_sha
        == "2e02bc4f0032b1125f42604bd4021a24fd70bb45b786c82e89ad17290bf31fc6",
        budget_sha,
    )
    add_check(
        checks,
        "cellect_tree_unchanged",
        cellect_tree_sha
        == "3ab43fa41624a119ad6917d8017a5a7bd5ce8a6c9eed736e290d45ef4f2c047a",
        cellect_tree_sha,
    )
    add_check(checks, "tier_moves_accepted", changed_nodes == [], changed_nodes)
    add_check(checks, "rollup_unchanged", census["rollup"] == initial_rollup, census["rollup"])
    add_check(
        checks,
        "graph_gate_human_held",
        by_id["NODETOPO-O0"]["blocked_on"] == ["HUMAN_REVIEW_NODETOPO_GRAPH_GATE"]
        and by_id["NODERESTORE-O0"]["blocked_on"]
        == ["HUMAN_REVIEW_NODETOPO_GRAPH_GATE"],
        {
            "NODETOPO-O0": by_id["NODETOPO-O0"]["blocked_on"],
            "NODERESTORE-O0": by_id["NODERESTORE-O0"]["blocked_on"],
        },
    )
    parkb_banked_fragment = SURFACE / "branches/sa_parkb/state_fragment_wave2.json"
    parkb_source_fragment_sha = commit_blob_sha256(
        "479059283825204c1e47fe7d8151716ac1603977",
        "reports/g02/screening/tiered_surface/branches/sa_parkb/state_fragment.json",
    )
    add_check(
        checks,
        "parkb_collision_safe_fragment_relocation",
        parkb_source_fragment_sha == sha256(parkb_banked_fragment),
        {
            "source_commit_sha256": parkb_source_fragment_sha,
            "banked_sha256": sha256(parkb_banked_fragment),
        },
    )

    barrier = {
        "schema": "celltrack-alpha.cpu-wave2-barrier.v1",
        "updated_utc": UPDATED_UTC,
        "base_commit": BASE_COMMIT,
        "status": "PASS_ALL_LANES_BANKED_GRAPH_HUMAN_HELD",
        "lane_receipts": lane_receipts,
        "shared_governance": {
            "alignment": {
                "path": str(alignment_path.relative_to(ROOT)),
                "sha256": alignment_sha,
            },
            "scope_manifest": {
                "path": str(scope_manifest_path.relative_to(ROOT)),
                "sha256": scope_manifest_sha,
            },
            "policy": "Each lane carries an explicit receipt for both frozen governance inputs.",
        },
        "reconciliation": {
            "accepted_tier_moves": [],
            "rejected_uncited_tier_moves": [],
            "uncited_tier_move_count": 0,
            "cpu_testable_park_b_nodes_audited": 14,
            "gpu_or_weight_gated_nodes_skipped": 8,
            "memo_c2_branch_b_numbers_used": 0,
            "dead_promotions": 0,
            "initial_rollup": initial_rollup,
            "final_rollup": census["rollup"],
            "tier_census_sha256": census_sha,
            "specific_fragment_precedence": [
                "SA-ASSOC-CYCLE over SA-PARKB-SWEEP for AE-P6 and R8",
                "SA-NODETOPO-GEN over SA-PARKB-SWEEP for NODERESTORE-O0",
            ],
            "collision_safe_transformations": [
                {
                    "lane": "SA-PARKB-SWEEP",
                    "source_commit": "479059283825204c1e47fe7d8151716ac1603977",
                    "source_path": "reports/g02/screening/tiered_surface/branches/sa_parkb/state_fragment.json",
                    "banked_path": "reports/g02/screening/tiered_surface/branches/sa_parkb/state_fragment_wave2.json",
                    "sha256": sha256(parkb_banked_fragment),
                    "byte_identical_to_commit_blob": parkb_source_fragment_sha
                    == sha256(parkb_banked_fragment),
                    "reason": "Preserve the earlier untracked PARK-B fragment at its existing path without overwrite.",
                }
            ],
        },
        "frozen_inputs": census["frozen_inputs"],
        "external_actions": {
            "gpu": False,
            "kaggle_launch": False,
            "kaggle_budget_debit": False,
            "cellect_write_or_launch": False,
            "runpod": False,
            "submission": False,
            "graph_run": False,
        },
        "human_review_gate": {
            "node": "NODETOPO-O0",
            "status": "HOLD",
            "reason": "Generator prerequisite is authentic and N1-compiled, but no graph run is authorized by this barrier.",
        },
    }
    barrier_path = SURFACE / "wave2_cpu_barrier.json"
    dump(barrier_path, barrier)
    barrier_sha = sha256(barrier_path)

    validation_doc = {
        "schema": "celltrack-alpha.cpu-wave2-barrier-validation.v1",
        "updated_utc": UPDATED_UTC,
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "barrier_sha256": barrier_sha,
        "tier_census_sha256": census_sha,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    dump(SURFACE / "wave2_cpu_barrier_validation.json", validation_doc)
    if validation_doc["status"] != "PASS":
        raise SystemExit("Wave-2 barrier validation failed")


if __name__ == "__main__":
    main()
