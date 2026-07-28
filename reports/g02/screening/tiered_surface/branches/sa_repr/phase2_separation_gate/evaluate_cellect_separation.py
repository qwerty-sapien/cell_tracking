#!/usr/bin/env python3
"""Evaluate frozen B0/A1/A0 CELLECT separation arms and issue per-class verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import rankdata

AMENDMENT_SHA = "9fb030bdcffbf95e242c71aa8b2c14bf8254b42730acbb489d66e2fe35a3310e"
LAMBDA = 0.01


class EvaluateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_score(train_x: np.ndarray, train_y: np.ndarray, held_x: np.ndarray) -> np.ndarray:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.concatenate((z, np.ones((len(z), 1))), axis=1)
    positive = y == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise EvaluateError("training fold lacks a class")
    weights = np.empty(len(y), dtype=np.float64)
    weights[positive] = 0.5 / positive.sum()
    weights[negative] = 0.5 / negative.sum()
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    lhs = design.T @ (weights[:, None] * design) + LAMBDA * penalty
    rhs = design.T @ (weights * y)
    solution = np.linalg.solve(lhs, rhs)
    held = (np.asarray(held_x, dtype=np.float64) - mean) / scale
    scores = held @ solution[:-1] + solution[-1]
    if not np.isfinite(scores).all():
        raise EvaluateError("nonfinite held score")
    return scores


def auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise EvaluateError("held fold lacks a class")
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def retained(y: np.ndarray, score: np.ndarray, video: np.ndarray, budgets: dict[str, int]) -> tuple[float, int, int, list[dict[str, Any]]]:
    selected_beneficial = 0
    selected_harmful = 0
    total_beneficial = int(y.sum())
    rows = []
    for identity in sorted(set(video.tolist())):
        index = np.flatnonzero(video == identity)
        order = index[np.lexsort((np.arange(len(index)), -score[index]))]
        budget = budgets[identity]
        chosen = []
        harmful = 0
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and score[order[end]] == score[order[position]]:
                end += 1
            block = order[position:end]
            block_harmful = int((y[block] == 0).sum())
            if harmful + block_harmful > budget:
                break
            chosen.extend(block.tolist())
            harmful += block_harmful
            position = end
        chosen_array = np.asarray(chosen, dtype=np.int64)
        beneficial = int(y[chosen_array].sum()) if len(chosen_array) else 0
        selected_beneficial += beneficial
        selected_harmful += harmful
        rows.append(
            {
                "identity": identity,
                "budget": budget,
                "selected_beneficial": beneficial,
                "selected_harmful": harmful,
                "budget_pass": harmful <= budget,
            }
        )
    value = selected_beneficial / total_beneficial if total_beneficial else float("nan")
    return value, selected_beneficial, selected_harmful, rows


def load_class(
    root: Path,
    feature_root: Path | None,
    action_class: str,
) -> dict[str, Any]:
    pieces = []
    budgets = {}
    for path in sorted((root / "population" / action_class).glob("*.npz")):
        identity = path.stem.replace("__", ":", 1)
        feature_path = feature_root / action_class / path.name if feature_root else None
        if feature_path is not None and not feature_path.is_file():
            raise EvaluateError(f"missing CELLECT feature file {feature_path}")
        with np.load(path, allow_pickle=False) as pop:
            count = len(pop["oracle_label"])
            piece = {
                "identity": np.full(count, identity, dtype=f"<U{len(identity)}"),
                "fold": pop["fold"],
                "y": pop["oracle_label"],
                "b0": pop["b0"],
            }
            if feature_path is not None:
                with np.load(feature_path, allow_pickle=False) as feat:
                    if feat["a1"].shape != (count, 8) or feat["a0"].shape != (count, 8):
                        raise EvaluateError("action feature shape mismatch")
                    piece["a1"] = feat["a1"]
                    piece["a0"] = feat["a0"]
            pieces.append(piece)
            budgets[identity] = int(pop["fp_budget"][0])
    if not pieces:
        raise EvaluateError(f"empty class {action_class}")
    return {
        key: np.concatenate([piece[key] for piece in pieces])
        for key in pieces[0]
    } | {"budgets": budgets}


def arm_metrics(
    data: dict[str, Any],
    features: np.ndarray,
    held_fold: int,
) -> dict[str, Any]:
    held = data["fold"] == held_fold
    fit = ~held
    score = fit_score(features[fit], data["y"][fit], features[held])
    retention, selected_beneficial, selected_harmful, videos = retained(
        data["y"][held], score, data["identity"][held], data["budgets"]
    )
    return {
        "auc": auc(data["y"][held], score),
        "retention": retention,
        "selected_beneficial": selected_beneficial,
        "selected_harmful": selected_harmful,
        "videos": videos,
    }


def evaluate_b0_class(data: dict[str, Any], action_class: str) -> dict[str, Any]:
    folds = []
    for held_fold in range(5):
        held = data["fold"] == held_fold
        fit = ~held
        if not held.any() or not fit.any():
            raise EvaluateError(f"empty fold {held_fold}")
        folds.append(
            {
                "held_fold": held_fold,
                "held_actions": int(held.sum()),
                "beneficial": int(data["y"][held].sum()),
                "harmful": int((data["y"][held] == 0).sum()),
                "B0": arm_metrics(data, data["b0"], held_fold),
            }
        )
    return {
        "action_class": action_class,
        "actions": len(data["y"]),
        "beneficial": int(data["y"].sum()),
        "harmful": int((data["y"] == 0).sum()),
        "folds": folds,
    }


def evaluate_class(
    data: dict[str, Any],
    action_class: str,
    sealed_b0: dict[str, Any],
) -> dict[str, Any]:
    for key in ("action_class", "actions", "beneficial", "harmful"):
        observed = (
            action_class
            if key == "action_class"
            else len(data["y"])
            if key == "actions"
            else int(data["y"].sum())
            if key == "beneficial"
            else int((data["y"] == 0).sum())
        )
        if sealed_b0[key] != observed:
            raise EvaluateError(f"sealed B0 {key} identity mismatch")
    b0_folds = {row["held_fold"]: row for row in sealed_b0["folds"]}
    if set(b0_folds) != set(range(5)):
        raise EvaluateError("sealed B0 fold coverage mismatch")
    arms = {
        "A1": np.concatenate((data["b0"], data["a1"]), axis=1),
        "A0": np.concatenate((data["b0"], data["a0"]), axis=1),
    }
    folds = []
    for held_fold in range(5):
        held = data["fold"] == held_fold
        fit = ~held
        if not held.any() or not fit.any():
            raise EvaluateError(f"empty fold {held_fold}")
        sealed = b0_folds[held_fold]
        expected_counts = (
            int(held.sum()),
            int(data["y"][held].sum()),
            int((data["y"][held] == 0).sum()),
        )
        observed_counts = (
            sealed["held_actions"],
            sealed["beneficial"],
            sealed["harmful"],
        )
        if observed_counts != expected_counts:
            raise EvaluateError("sealed B0 held-fold identity mismatch")
        metrics = {
            "B0": sealed["B0"],
            "A1": arm_metrics(data, arms["A1"], held_fold),
            "A0": arm_metrics(data, arms["A0"], held_fold),
        }
        a1 = metrics["A1"]
        b0 = metrics["B0"]
        a0 = metrics["A0"]
        checks = {
            "auc_absolute_floor": a1["auc"] >= 0.65,
            "auc_lift_over_B0": a1["auc"] - b0["auc"] >= 0.05,
            "auc_lift_over_A0": a1["auc"] - a0["auc"] >= 0.03,
            "retention_absolute_floor": a1["retention"] >= 0.15,
            "retention_lift_over_B0": a1["retention"] - b0["retention"] >= 0.05,
            "retention_lift_over_A0": a1["retention"] - a0["retention"] >= 0.03,
            "per_video_fp_budget": all(row["budget_pass"] for row in a1["videos"]),
        }
        folds.append(
            {
                "held_fold": held_fold,
                "held_actions": int(held.sum()),
                "beneficial": int(data["y"][held].sum()),
                "harmful": int((data["y"][held] == 0).sum()),
                "arms": metrics,
                "lifts": {
                    "A1_minus_B0_auc": a1["auc"] - b0["auc"],
                    "A1_minus_A0_auc": a1["auc"] - a0["auc"],
                    "A1_minus_B0_retention": a1["retention"] - b0["retention"],
                    "A1_minus_A0_retention": a1["retention"] - a0["retention"],
                },
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    passed = all(row["pass"] for row in folds)
    return {
        "action_class": action_class,
        "actions": len(data["y"]),
        "beneficial": int(data["y"].sum()),
        "harmful": int((data["y"] == 0).sum()),
        "folds": folds,
        "verdict": "PASS_UNLOCK_GRAPH_PREREG_REVIEW" if passed else "FAIL_PARK_C",
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("b0", "a1a0"), required=True)
    parser.add_argument("--amendment-sha", required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--b0-sealed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.amendment_sha != AMENDMENT_SHA:
        raise SystemExit("requested amendment identity mismatch")
    started = time.perf_counter()
    if args.stage == "b0":
        results = {
            action_class: evaluate_b0_class(
                load_class(args.population_root, None, action_class), action_class
            )
            for action_class in ("delete", "occupied_swap")
        }
        result = {
            "schema": "celltrack-alpha.cellect-separation-b0-sealed.v1",
            "status": "SEALED_BEFORE_A1_A0",
            "amendment_sha256": AMENDMENT_SHA,
            "results": results,
            "both_class_reporting_complete": True,
            "execution_order": ["B0_SEALED"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result["status"], "classes": sorted(results)}))
        return 0
    if args.feature_root is None or args.feature_manifest is None or args.b0_sealed is None:
        raise SystemExit("A1/A0 stage requires feature root, feature manifest, and sealed B0")
    feature_manifest = json.loads(args.feature_manifest.read_text())
    if feature_manifest["amendment_sha256"] != AMENDMENT_SHA:
        raise SystemExit("feature amendment identity mismatch")
    sealed_b0 = json.loads(args.b0_sealed.read_text())
    if (
        sealed_b0.get("status") != "SEALED_BEFORE_A1_A0"
        or sealed_b0.get("amendment_sha256") != AMENDMENT_SHA
    ):
        raise SystemExit("sealed B0 identity mismatch")
    results = {
        action_class: evaluate_class(
            load_class(args.population_root, args.feature_root, action_class),
            action_class,
            sealed_b0["results"][action_class],
        )
        for action_class in ("delete", "occupied_swap")
    }
    result = {
        "schema": "celltrack-alpha.cellect-separation-result.v1",
        "status": "COMPLETE",
        "amendment_sha256": AMENDMENT_SHA,
        "execution_order": ["B0_SEALED", "A1", "A0"],
        "sealed_B0": {
            "path": str(args.b0_sealed),
            "sha256": sha256(args.b0_sealed),
        },
        "feature_manifest": {
            "path": str(args.feature_manifest),
            "sha256": sha256(args.feature_manifest),
        },
        "results": results,
        "both_class_reporting_complete": True,
        "cross_class_conjunction": False,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "full_pool_cache": False,
            "graph": False,
            "family_6bba": False,
            "cv57": False,
            "submission": False,
            "runpod": False,
        },
        "barrier": "HOLD_FOR_HUMAN_REVIEW_BEFORE_ANY_GRAPH_PREREGISTRATION_EXECUTION",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["verdict"] for key, value in results.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
