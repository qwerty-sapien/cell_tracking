#!/usr/bin/env python3
"""Fail-closed integrity validation for the assignment-evidence screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.manifest.read_text())
    result = json.loads(args.result.read_text())
    verdict = json.loads(args.verdict.read_text())
    checks: dict[str, bool] = {}

    checks["contract_frozen"] = (
        contract["status"] == "FROZEN_BEFORE_SUPPLIER_FEATURES_OR_SCORES"
    )
    checks["all_contract_pins_match"] = all(
        sha256(Path(pin["path"])) == pin["sha256"]
        for pin in contract["pins"].values()
    )
    checks["manifest_contract_identity"] = (
        manifest["contract"]["sha256"] == sha256(args.contract)
        and manifest["population_coverage"] == 1.0
        and manifest["integrity"]["sampled"] is False
        and manifest["integrity"]["oracle_label_read"] is False
        and manifest["integrity"]["oracle_delta_read"] is False
    )

    action_counts = {"delete": 0, "occupied_swap": 0}
    finite = True
    shapes = True
    hashes = True
    for row in manifest["feature_files"]:
        path = Path(row["path"])
        hashes &= path.is_file() and sha256(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            expected = (int(row["actions"]), 8)
            shapes &= (
                set(data.files) == {"a1", "a0"}
                and data["a1"].shape == expected
                and data["a0"].shape == expected
            )
            finite &= bool(np.isfinite(data["a1"]).all())
            finite &= bool(np.isfinite(data["a0"]).all())
        action_counts[row["action_class"]] += int(row["actions"])
    checks["feature_file_hashes"] = hashes
    checks["feature_shapes_and_finite"] = shapes and finite
    checks["full_action_counts"] = action_counts == {
        "delete": 1_345_497,
        "occupied_swap": 45_961,
    }

    checks["sealed_result_identity"] = (
        result["schema"] == "celltrack-alpha.cellect-separation-result.v1"
        and result["status"] == "COMPLETE"
        and result["feature_manifest"]["sha256"] == sha256(args.manifest)
        and result["sealed_B0"]["sha256"]
        == contract["pins"]["sealed_b0"]["sha256"]
        and result["both_class_reporting_complete"] is True
        and result["cross_class_conjunction"] is False
    )
    checks["per_video_budgets"] = all(
        video["budget_pass"]
        for class_row in result["results"].values()
        for fold in class_row["folds"]
        for video in fold["arms"]["A1"]["videos"]
    )
    checks["measured_verdict_identity"] = (
        verdict["status"] == "COMPLETE"
        and verdict["separation_result"]["sha256"] == sha256(args.result)
        and verdict["amendment"]["sha256"]
        == contract["pins"]["measured_requirement_amendment"]["sha256"]
        and verdict["verdicts"]
        == {"delete": "FAIL_PARK_C", "occupied_swap": "FAIL_PARK_C"}
    )
    checks["barrier_closed"] = (
        verdict["graph_run"] is False
        and verdict["family_6bba_run"] is False
        and verdict["cv57_run"] is False
        and verdict["tier_move"] is False
        and verdict["submission"] is False
    )
    if not all(checks.values()):
        raise SystemExit(
            "validation failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    output = {
        "schema": "celltrack-alpha.assignment-evidence-validation.v1",
        "status": "PASS",
        "checks": checks,
        "artifacts": {
            "contract": {"path": str(args.contract), "sha256": sha256(args.contract)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "result": {"path": str(args.result), "sha256": sha256(args.result)},
            "verdict": {"path": str(args.verdict), "sha256": sha256(args.verdict)},
        },
        "action_counts": action_counts,
        "verdicts": verdict["verdicts"],
        "barrier": verdict["barrier"],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "checks": len(checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
