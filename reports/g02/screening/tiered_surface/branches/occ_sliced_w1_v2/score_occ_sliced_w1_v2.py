#!/usr/bin/env python3
"""Authenticate and score OCC-SLICED-W1-v2 on the frozen 44b6 swap gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/Users/agent/Desktop/cell_tracking")
BRANCH = ROOT / "reports/g02/screening/tiered_surface/branches/occ_sliced_w1_v2"
RUN = BRANCH / "run_v1/occ_sliced_w1_v2"
FEATURE_ROOT = RUN / "features"
FEATURE_MANIFEST = RUN / "FEATURE_MANIFEST.json"
SEMANTIC_RECEIPT = RUN / "SEMANTIC_RECEIPT.json"
POPULATION_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
    / "run_v1/population/occupied_swap"
)
POPULATION_MANIFEST = POPULATION_ROOT.parent.parent / "population_manifest.json"
SEALED_B0 = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
    / "run_v3/B0_SEALED.json"
)
EVALUATOR = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/phase2_separation_gate"
    / "evaluate_cellect_separation.py"
)

EXPECTED = {
    "population_manifest_sha256": "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    "sealed_b0_sha256": "a9ce8c0be5afe4c2d2bfb37f3878691956ea458964788776af2ae9358f6b4ff2",
    "evaluator_sha256": "548b68185ecd1bb063bc9dfbc23f3acbfc2b322b3e387867f56cab7237bb30ca",
    "semantic_receipt_sha256": "b698efb7df3c2943ba5c4908c018b5c47b7d7298839599d9635e4a6608614723",
    "population_actions": 45_961,
    "beneficial_actions": 134,
    "files": 36,
    "feature_columns": 14,
}

GATE = {
    "auc_absolute_floor": 0.90,
    "auc_min_lift_over_B0": 0.05,
    "auc_min_lift_over_A0": 0.03,
    "retention_absolute_floor": 0.06791044776119404,
    "retention_min_lift_over_B0": 0.05,
    "retention_min_lift_over_A0": 0.03,
    "decision_rule": "all conditions in all five identity-grouped OOF folds",
}

ROBUST_REPORTING = {
    "mean_auc": 0.90,
    "mean_retention": 0.06791044776119404,
    "empirical_p05_auc": 0.95,
    "empirical_p05_retention": 0.14738805970149255,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def semantic_array_sha256(array: np.ndarray) -> str:
    array = np.asarray(array)
    descriptor = canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(descriptor + np.ascontiguousarray(array).tobytes()).hexdigest()


def load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_occ_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_and_load() -> tuple[dict[str, Any], dict[str, Any]]:
    pins = {
        "population_manifest_sha256": sha256(POPULATION_MANIFEST),
        "sealed_b0_sha256": sha256(SEALED_B0),
        "evaluator_sha256": sha256(EVALUATOR),
        "semantic_receipt_sha256": sha256(SEMANTIC_RECEIPT),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
    }
    for key in (
        "population_manifest_sha256",
        "sealed_b0_sha256",
        "evaluator_sha256",
        "semantic_receipt_sha256",
    ):
        if pins[key] != EXPECTED[key]:
            raise RuntimeError(f"frozen pin drift: {key}: {pins[key]}")

    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="ascii"))
    receipt = json.loads(SEMANTIC_RECEIPT.read_text(encoding="ascii"))
    if manifest.get("status") != "COMPLETE" or receipt.get("status") != "COMPLETE":
        raise RuntimeError("remote output is not COMPLETE")
    if manifest.get("semantic_receipt_sha256") != pins["semantic_receipt_sha256"]:
        raise RuntimeError("feature manifest does not bind the semantic receipt")
    if manifest.get("population_manifest_sha256") != EXPECTED["population_manifest_sha256"]:
        raise RuntimeError("remote population pin drift")
    if len(manifest.get("files", [])) != EXPECTED["files"]:
        raise RuntimeError("remote feature file count drift")

    manifest_rows = {row["output"]: row for row in manifest["files"]}
    receipt_rows = {
        row["identity"].replace(":", "__", 1) + ".npz": row for row in receipt["files"]
    }
    pieces: list[dict[str, np.ndarray]] = []
    budgets: dict[str, int] = {}
    feature_shards: list[dict[str, Any]] = []

    population_paths = sorted(POPULATION_ROOT.glob("*.npz"))
    if len(population_paths) != EXPECTED["files"]:
        raise RuntimeError("local population file count drift")
    for population_path in population_paths:
        name = population_path.name
        identity = population_path.stem.replace("__", ":", 1)
        feature_path = FEATURE_ROOT / name
        if name not in manifest_rows or name not in receipt_rows or not feature_path.is_file():
            raise RuntimeError(f"missing bound feature shard: {name}")
        manifest_row = manifest_rows[name]
        receipt_row = receipt_rows[name]
        if sha256(feature_path) != manifest_row["output_sha256"]:
            raise RuntimeError(f"feature file hash drift: {name}")
        with np.load(population_path, allow_pickle=False) as pop, np.load(
            feature_path, allow_pickle=False
        ) as feature:
            if set(feature.files) != {"a1", "a0"}:
                raise RuntimeError(f"feature schema drift: {name}")
            count = len(pop["oracle_label"])
            expected_shape = (count, EXPECTED["feature_columns"])
            if feature["a1"].shape != expected_shape or feature["a0"].shape != expected_shape:
                raise RuntimeError(f"feature shape drift: {name}")
            if not np.isfinite(feature["a1"]).all() or not np.isfinite(feature["a0"]).all():
                raise RuntimeError(f"nonfinite feature value: {name}")
            if semantic_array_sha256(feature["a1"]) != receipt_row["a1_sha256"]:
                raise RuntimeError(f"A1 semantic hash drift: {name}")
            if semantic_array_sha256(feature["a0"]) != receipt_row["a0_sha256"]:
                raise RuntimeError(f"A0 semantic hash drift: {name}")
            if count != manifest_row["actions"] or count != receipt_row["actions"]:
                raise RuntimeError(f"action count drift: {name}")
            pieces.append(
                {
                    "identity": np.full(count, identity, dtype=f"<U{len(identity)}"),
                    "fold": pop["fold"],
                    "y": pop["oracle_label"],
                    "b0": pop["b0"],
                    "a1": feature["a1"],
                    "a0": feature["a0"],
                }
            )
            budgets[identity] = int(pop["fp_budget"][0])
            feature_shards.append(
                {
                    "identity": identity,
                    "actions": count,
                    "feature_sha256": manifest_row["output_sha256"],
                    "a1_semantic_sha256": receipt_row["a1_sha256"],
                    "a0_semantic_sha256": receipt_row["a0_sha256"],
                }
            )

    data = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    data["budgets"] = budgets
    if len(data["y"]) != EXPECTED["population_actions"]:
        raise RuntimeError("aggregate action count drift")
    if int(data["y"].sum()) != EXPECTED["beneficial_actions"]:
        raise RuntimeError("aggregate beneficial count drift")
    integrity = {
        "status": "PASS",
        "pins": pins,
        "files": len(feature_shards),
        "actions": len(data["y"]),
        "beneficial": int(data["y"].sum()),
        "harmful": int((data["y"] == 0).sum()),
        "feature_columns": EXPECTED["feature_columns"],
        "nonfinite_features": 0,
        "duplicate_or_invalid_actions": 0,
        "population_coverage": 1.0,
        "feature_shards": feature_shards,
    }
    return data, integrity


def metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "empirical_p05": float(np.quantile(array, 0.05, method="linear")),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    data, integrity = authenticate_and_load()
    evaluator = load_evaluator()
    sealed = json.loads(SEALED_B0.read_text(encoding="ascii"))["results"]["occupied_swap"]
    measured = evaluator.evaluate_class(data, "occupied_swap", sealed)

    fold_rows = []
    for row in measured["folds"]:
        a1, b0, a0 = row["arms"]["A1"], row["arms"]["B0"], row["arms"]["A0"]
        checks = {
            "auc_absolute_floor": a1["auc"] >= GATE["auc_absolute_floor"],
            "auc_lift_over_B0": a1["auc"] - b0["auc"] >= GATE["auc_min_lift_over_B0"],
            "auc_lift_over_A0": a1["auc"] - a0["auc"] >= GATE["auc_min_lift_over_A0"],
            "retention_absolute_floor": a1["retention"] >= GATE["retention_absolute_floor"],
            "retention_lift_over_B0": (
                a1["retention"] - b0["retention"] >= GATE["retention_min_lift_over_B0"]
            ),
            "retention_lift_over_A0": (
                a1["retention"] - a0["retention"] >= GATE["retention_min_lift_over_A0"]
            ),
            "per_video_fp_budget": all(video["budget_pass"] for video in a1["videos"]),
        }
        fold_rows.append(
            {
                "held_fold": row["held_fold"],
                "held_actions": row["held_actions"],
                "beneficial": row["beneficial"],
                "harmful": row["harmful"],
                "arms": row["arms"],
                "lifts": row["lifts"],
                "binding_checks": checks,
                "pass": all(checks.values()),
            }
        )

    auc_values = [row["arms"]["A1"]["auc"] for row in fold_rows]
    retention_values = [row["arms"]["A1"]["retention"] for row in fold_rows]
    auc_summary = metric_summary(auc_values)
    retention_summary = metric_summary(retention_values)
    reporting_checks = {
        "mean_auc": auc_summary["mean"] >= ROBUST_REPORTING["mean_auc"],
        "mean_retention": retention_summary["mean"] >= ROBUST_REPORTING["mean_retention"],
        "empirical_p05_auc": (
            auc_summary["empirical_p05"] >= ROBUST_REPORTING["empirical_p05_auc"]
        ),
        "empirical_p05_retention": (
            retention_summary["empirical_p05"]
            >= ROBUST_REPORTING["empirical_p05_retention"]
        ),
    }
    gate_pass = all(row["pass"] for row in fold_rows)
    selected_beneficial = sum(row["arms"]["A1"]["selected_beneficial"] for row in fold_rows)
    selected_harmful = sum(row["arms"]["A1"]["selected_harmful"] for row in fold_rows)
    result = {
        "schema": "celltrack-alpha.occ-sliced-w1-local-score.v1",
        "status": "COMPLETE",
        "supplier": "OCC-SLICED-W1-v2",
        "population": "fit-only 44b6 occupied_swap",
        "integrity": integrity,
        "probe": {
            "implementation": str(EVALUATOR),
            "implementation_sha256": sha256(EVALUATOR),
            "kind": "frozen class-balanced standardized ridge least-squares OOF rank probe",
            "ridge_lambda": 0.01,
            "folds": 5,
            "hyperparameter_search": False,
            "feature_selection": False,
            "execution_order": ["sealed_B0", "A1", "A0"],
        },
        "gate": GATE,
        "folds": fold_rows,
        "A1_summary": {
            "auc": auc_summary,
            "retention": retention_summary,
            "selected_beneficial": selected_beneficial,
            "selected_harmful": selected_harmful,
        },
        "measured_requirement_reporting": {
            "requirements": ROBUST_REPORTING,
            "checks": reporting_checks,
            "pass": all(reporting_checks.values()),
        },
        "verdict": "PASS_UNLOCK_GRAPH_PREREG_REVIEW" if gate_pass else "FAIL_PARK_C",
        "gate_pass": gate_pass,
        "graph_preregistration_unlocked": gate_pass,
        "graph_run": False,
        "official_graph_delta": None,
        "gate_implied_selected_action_count": selected_beneficial + selected_harmful,
        "gate_implied_noop_adjusted_delta": 0.0 if not selected_beneficial and not selected_harmful else None,
        "family_6bba_run": False,
        "cv57_run": False,
        "submission_recommended": False,
        "submission": False,
        "barrier": "PARK_SUPPLIER_NO_GRAPH_NO_CV57_NO_SUBMISSION" if not gate_pass else "HOLD_FOR_GRAPH_PREREGISTRATION",
    }
    if not all(math.isfinite(value) for value in auc_values + retention_values):
        raise RuntimeError("nonfinite gate metric")
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "mean_auc": auc_summary["mean"],
                "mean_retention": retention_summary["mean"],
                "selected_actions": selected_beneficial + selected_harmful,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
