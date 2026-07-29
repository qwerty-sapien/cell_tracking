#!/usr/bin/env python3
"""Pin both complete class shards, then run the unchanged frozen evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import constants


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_requirement_cross_read(
    auc_values: list[float],
    retention_values: list[float],
    requirement: dict[str, float],
) -> dict[str, Any]:
    auc_array = np.asarray(auc_values, dtype=np.float64)
    retention_array = np.asarray(retention_values, dtype=np.float64)
    if (
        auc_array.shape != (5,)
        or retention_array.shape != (5,)
        or not np.isfinite(auc_array).all()
        or not np.isfinite(retention_array).all()
    ):
        raise RuntimeError("measured-requirement cross-read requires five finite folds")
    observed = {
        "fold_mean_auc": float(auc_array.mean()),
        "fold_mean_retention": float(retention_array.mean()),
        "fold_empirical_p05_auc": float(
            np.quantile(auc_array, 0.05, method="linear")
        ),
        "fold_empirical_p05_retention": float(
            np.quantile(retention_array, 0.05, method="linear")
        ),
    }
    checks = {
        "mean_auc_clears_mean_requirement": (
            observed["fold_mean_auc"] >= requirement["mean_auc"]
        ),
        "mean_retention_clears_mean_requirement": (
            observed["fold_mean_retention"] >= requirement["mean_retention"]
        ),
        "p05_auc_clears_robust_requirement": (
            observed["fold_empirical_p05_auc"] >= requirement["robust_p05_auc"]
        ),
        "p05_retention_clears_robust_requirement": (
            observed["fold_empirical_p05_retention"]
            >= requirement["robust_p05_retention"]
        ),
    }
    checks["mean_clears_measured_requirements"] = (
        checks["mean_auc_clears_mean_requirement"]
        and checks["mean_retention_clears_mean_requirement"]
    )
    checks["p05_clears_robust_requirements"] = (
        checks["p05_auc_clears_robust_requirement"]
        and checks["p05_retention_clears_robust_requirement"]
    )
    return {
        "requirements": requirement,
        "observed_A1_across_folds": observed,
        "checks": checks,
        "reporting_only": True,
        "frozen_gate_controls_verdict": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_DINO_GUARD_ACTIVE") != "1":
        raise SystemExit("binding aggregation guard absent")
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    frozen = constants.verify_pins()
    class_manifests = {}
    files = []
    for action_class in constants.CLASSES:
        path = args.feature_root / action_class / "class_manifest.json"
        manifest = json.loads(path.read_text(encoding="ascii"))
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("action_class") != action_class
            or manifest.get("action_count") != constants.EXPECTED_ACTIONS[action_class]
        ):
            raise SystemExit(f"incomplete class manifest: {action_class}")
        for row in manifest["files"]:
            feature = args.feature_root / action_class / row["output"]
            if constants.sha256(feature) != row["output_sha256"]:
                raise SystemExit(f"feature shard drift: {feature}")
            files.append({**row, "output": f"{action_class}/{row['output']}"})
        class_manifests[action_class] = {
            "path": str(path.resolve()),
            "sha256": constants.sha256(path),
        }
    aggregate_manifest = {
        "schema": "celltrack-alpha.dinov2-xy-action-features.v2",
        "status": "COMPLETE",
        "supplier_id": "F0-DINOV2-VITS14-XY-V1",
        "amendment_sha256": constants.PINS[constants.AMENDMENT],
        "sealed_B0_sha256": constants.PINS[constants.SEALED_B0],
        "population_manifest_sha256": constants.PINS[constants.POPULATION_MANIFEST],
        "class_manifests": class_manifests,
        "frozen_inputs": frozen,
        "file_count": len(files),
        "files": files,
        "population_actions": constants.EXPECTED_ACTIONS,
        "per_class_subprocesses": True,
        "per_video_load_release": True,
        "full_pool_cache": False,
        "training": False,
        "graph": False,
        "family_6bba": False,
        "cv57": False,
        "submission": False,
        "runpod": False,
    }
    args.output.mkdir(parents=True)
    manifest_path = args.output / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(aggregate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    evaluator = load_module(constants.EVALUATOR, "pinned_separation_evaluator")
    results = {}
    sealed = json.loads(constants.SEALED_B0.read_text(encoding="ascii"))
    for action_class in constants.CLASSES:
        data = evaluator.load_class(
            constants.POPULATION_ROOT, args.feature_root, action_class
        )
        results[action_class] = evaluator.evaluate_class(
            data, action_class, sealed["results"][action_class]
        )
        fold_aucs = [
            row["arms"]["A1"]["auc"] for row in results[action_class]["folds"]
        ]
        fold_retentions = [
            row["arms"]["A1"]["retention"]
            for row in results[action_class]["folds"]
        ]
        results[action_class][
            "measured_requirement_comparison"
        ] = measured_requirement_cross_read(
            fold_aucs,
            fold_retentions,
            constants.REQUIREMENTS[action_class],
        )
    result = {
        "schema": "celltrack-alpha.dinov2-separation-result.v2",
        "status": "COMPLETE",
        "supplier_id": "F0-DINOV2-VITS14-XY-V1",
        "amendment_sha256": constants.PINS[constants.AMENDMENT],
        "sealed_B0_sha256": constants.PINS[constants.SEALED_B0],
        "population_manifest_sha256": constants.PINS[constants.POPULATION_MANIFEST],
        "requirement_surface_sha256": constants.PINS[constants.REQUIREMENT_SURFACE],
        "execution_order": ["B0_SEALED", "A1", "A0"],
        "feature_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": constants.sha256(manifest_path),
        },
        "results": results,
        "both_class_reporting_complete": True,
        "cross_class_conjunction": False,
        "barrier": "HOLD_FOR_HUMAN_REVIEW_BEFORE_ANY_GRAPH_PREREGISTRATION",
        "execution": {
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "submission": False,
            "runpod": False,
        },
    }
    (args.output / "SEPARATION_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps({key: value["verdict"] for key, value in results.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
