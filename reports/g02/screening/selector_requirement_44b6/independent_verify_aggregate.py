#!/usr/bin/env python3
"""Aggregate isolated class reviews and audit the requirement surface."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import independent_verify as review  # noqa: E402
import selector_requirement as target  # noqa: E402


CLASSES = ("delete", "occupied_swap", "restore_topology")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise review.ReviewError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delete-review", type=Path, required=True)
    parser.add_argument("--occupied-swap-review", type=Path, required=True)
    parser.add_argument("--restore-topology-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise SystemExit("binding local zero-swap guard is required")
    if args.output.exists():
        raise SystemExit("refusing to overwrite aggregate review output")

    run_dir = args.run_dir.resolve()
    dose = load(run_dir / "dose_response.json")
    noise = load(run_dir / "noise_surface.json")
    requirements = load(run_dir / "requirement_surface.json")
    barrier = load(run_dir / "BARRIER.json")
    result = load(run_dir / "RESULT.json")
    review_paths = {
        "delete": args.delete_review / "CLASS_REVIEW.json",
        "occupied_swap": args.occupied_swap_review / "CLASS_REVIEW.json",
        "restore_topology": args.restore_topology_review / "CLASS_REVIEW.json",
    }
    class_reviews = {name: load(path) for name, path in review_paths.items()}
    checks: list[str] = []
    for name, value in class_reviews.items():
        if value.get("status") != "PASS" or value.get("action_class") != name:
            raise review.ReviewError(f"class review failed or drifted: {name}")
        checks.append(f"{name}_oracle_best_direct_official_match")

    for name in CLASSES:
        raw = noise["classes"][name]
        beneficial = int(dose["classes"][name]["beneficial"])
        harmful = int(dose["classes"][name]["harmful"])
        if beneficial == 0 or harmful == 0:
            if raw["status"] != "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE":
                raise review.ReviewError(f"{name} degenerate status drifted")
            continue
        runs = raw["seed_runs"]
        if len(runs) != len(target.TARGET_AUCS) * target.SEED_COUNT:
            raise review.ReviewError(f"{name} seed coverage drifted")
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        lattice = 0.5 / (beneficial * harmful)
        for row in runs:
            auc = float(row["target_auc"])
            grouped[auc].append(row)
            if abs(float(row["empirical_auc"]) - auc) > max(
                target.AUC_TOLERANCE, lattice + 1e-15
            ):
                raise review.ReviewError(f"{name} empirical AUC drifted")
            if any(
                int(video["selected_harmful"]) > int(video["budget"])
                for video in row["retention"]["videos"]
            ):
                raise review.ReviewError(f"{name} CG0 budget violation")
        for auc, rows in grouped.items():
            if len(rows) != target.SEED_COUNT:
                raise review.ReviewError(f"{name} AUC {auc} seed count drifted")
            aggregate = next(
                row
                for row in noise["aggregates"][name]["auc_results"]
                if float(row["target_auc"]) == auc
            )
            observed = float(np.mean([row["empirical_auc"] for row in rows]))
            if not math.isclose(
                observed,
                float(aggregate["empirical_auc"]["mean"]),
                abs_tol=1e-12,
            ):
                raise review.ReviewError(f"{name} AUC aggregate drifted")
        checks.append(f"{name}_noise_coverage_auc_budget")

        recomputed = {
            "target_plus_0_005": target.requirement_for(
                dose["classes"][name]["curve"],
                noise["aggregates"][name],
                target.GRAPH_TARGET,
            ),
            "breakeven_0": target.requirement_for(
                dose["classes"][name]["curve"],
                noise["aggregates"][name],
                0.0,
            ),
        }
        recomputed["frozen_gate_verdict"] = target.threshold_verdict(
            recomputed["target_plus_0_005"]
        )
        if recomputed != requirements["requirements"][name]:
            raise review.ReviewError(f"{name} requirement recomputation drifted")
        checks.append(f"{name}_requirement_recomputed")

    forbidden = (
        "thresholds_changed",
        "ledger_written",
        "plan_changed",
        "gpu_action",
        "kaggle_action",
        "runpod_action",
        "submission_action",
        "family_6bba_used",
        "cv57_used",
        "external_action",
    )
    if any(barrier.get(key) for key in forbidden):
        raise review.ReviewError("a governance hard-stop flag is true")
    if barrier.get("tier_moves") != 0 or barrier.get("dead_actions") != 0:
        raise review.ReviewError("tier or DEAD mutation reported")
    too_lenient = sorted(
        name
        for name, value in requirements["requirements"].items()
        if value["frozen_gate_verdict"]["verdict"] == "TOO_LENIENT"
    )
    expected_status = (
        "HUMAN_REVIEW_REQUIRED" if too_lenient else "DIAGNOSTIC_COMPLETE"
    )
    if (
        sorted(barrier["amendment_proposed_for"]) != too_lenient
        or barrier["status"] != expected_status
        or result["status"] != expected_status
    ):
        raise review.ReviewError("terminal barrier is inconsistent")
    checks.append("governance_hard_stops_and_barrier")

    args.output.mkdir(parents=True)
    direct = {
        name: {
            key: value
            for key, value in class_reviews[name].items()
            if key
            in (
                "candidate_prefix_N",
                "applied_support",
                "direct_official_delta",
                "reported_delta",
            )
        }
        for name in CLASSES
    }
    review.write(
        args.output / "INDEPENDENT_REVIEW.json",
        {
            "schema": "celltrack-alpha.selector-requirement-independent-review.v2",
            "status": "PASS",
            "method": (
                "Three fresh class-isolated direct official-score checks, "
                "independent occupied-swap greedy replay, complete seed/AUC/CG0 "
                "audit, requirement recomputation, and governance review."
            ),
            "direct_points": direct,
            "findings": [],
            "hard_stops_preserved": True,
        },
    )
    review.write(
        args.output / "VALIDATION.json",
        {
            "schema": "celltrack-alpha.selector-requirement-validation.v2",
            "status": "PASS",
            "checks_passed": len(checks),
            "checks": checks,
            "class_count": 3,
            "target_auc_count": len(target.TARGET_AUCS),
            "seed_count_per_auc": target.SEED_COUNT,
            "human_review_required": expected_status == "HUMAN_REVIEW_REQUIRED",
            "tier_moves": 0,
            "threshold_changes": 0,
            "external_actions": 0,
        },
    )
    print(json.dumps({"status": "PASS", "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
