#!/usr/bin/env python3
"""Retrospective CPU screen for two-frame DeepCenter path persistence.

The input was produced by the earlier frozen CV57 R1/DeepCenter redundancy
probe.  This script aggregates the two missing-frame observations per bridge
and evaluates whether persistence is materially more discriminative than a
single point.  Because the population is CV57, the result is diagnostic only:
it may kill or park the mechanism but cannot select a deployable configuration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr


FEATURES = ("minimum", "mean", "maximum", "absolute_delta", "negative_cost")


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def fold(dataset: str) -> int:
    return int(hashlib.sha256(dataset.encode("ascii")).hexdigest()[:8], 16) % 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--trials-per-fold", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["tail"]), int(row["head"]))].append(row)
    bridges = []
    for (dataset, tail, head), values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: int(row["t"]))
        if len(values) != 2:
            raise ValueError(f"bridge {dataset}:{tail}:{head} does not have exactly two frames")
        scores = np.asarray([float(row["deepcenter_score"]) for row in values])
        bridges.append(
            {
                "dataset": dataset,
                "tail": tail,
                "head": head,
                "matched_any": bool(values[0]["matched_any"]),
                "matched_all": bool(values[0]["matched_all"]),
                "features": [
                    float(scores.min()),
                    float(scores.mean()),
                    float(scores.max()),
                    float(abs(scores[1] - scores[0])),
                    -float(values[0]["cost"]),
                ],
            }
        )
    matrix = np.asarray([row["features"] for row in bridges], dtype=np.float64)
    labels = np.asarray([row["matched_any"] for row in bridges], dtype=bool)
    folds = np.asarray([fold(row["dataset"]) for row in bridges], dtype=np.int64)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    oof = np.zeros(len(bridges), dtype=np.float64)
    fold_receipts = []
    for held in range(5):
        training = folds != held
        validation = folds == held
        center = np.median(matrix[training], axis=0)
        scale = np.percentile(matrix[training], 75, axis=0) - np.percentile(matrix[training], 25, axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        standardized = (matrix - center[None, :]) / scale[None, :]

        def objective(trial: Any) -> float:
            weights = np.asarray(
                [trial.suggest_float(f"weight_{name}", -3.0, 3.0) for name in FEATURES],
                dtype=np.float64,
            )
            return auc(labels[training], standardized[training] @ weights)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed + held, multivariate=True),
        )
        study.optimize(objective, n_trials=args.trials_per_fold, show_progress_bar=False)
        weights = np.asarray([study.best_params[f"weight_{name}"] for name in FEATURES])
        oof[validation] = standardized[validation] @ weights
        fold_receipts.append(
            {
                "fold": held,
                "train_bridges": int(training.sum()),
                "validation_bridges": int(validation.sum()),
                "train_positives": int(labels[training].sum()),
                "validation_positives": int(labels[validation].sum()),
                "best_train_auc": float(study.best_value),
                "weights": {name: float(value) for name, value in zip(FEATURES, weights)},
                "validation_auc": auc(labels[validation], oof[validation]),
            }
        )

    univariate = {name: auc(labels, matrix[:, index]) for index, name in enumerate(FEATURES)}
    oof_auc = auc(labels, oof)
    mean_score = matrix[:, FEATURES.index("mean")]
    rho, rho_p = spearmanr(mean_score, -matrix[:, FEATURES.index("negative_cost")])
    gate = "GO" if oof_auc >= 0.70 and abs(float(rho)) <= 0.35 else ("PARK" if oof_auc >= 0.65 else "DEAD")
    payload = {
        "schema": "celltrack-alpha.deepcenter-temporal-persistence-screen.v1",
        "status": "COMPLETE_RETROSPECTIVE_DIAGNOSTIC",
        "candidate": "TEMP-CENTERLINE-v1",
        "population": "CV57_R1_GEOMETRY_CANDIDATES",
        "launch_eligible": False,
        "launch_ineligibility": (
            "CV57 was used for this retrospective mechanism screen; a fresh fit105 exact-action "
            "bank would be required to freeze a deployable rule"
        ),
        "counts": {
            "bridges": len(bridges),
            "matched_any": int(labels.sum()),
            "matched_all": sum(bool(row["matched_all"]) for row in bridges),
            "videos": len({row["dataset"] for row in bridges}),
        },
        "univariate_auc": univariate,
        "nested_grouped_oof": {
            "auc": oof_auc,
            "fold_rule": "sha256(dataset)[0:8] modulo 5",
            "folds": fold_receipts,
            "optimizer": {
                "library": "optuna",
                "version": optuna.__version__,
                "sampler": "TPESampler(multivariate=True)",
                "trials_per_fold": args.trials_per_fold,
            },
        },
        "redundancy": {
            "spearman_mean_deepcenter_vs_gap_cost": float(rho),
            "p_value": float(rho_p),
            "maximum_absolute_rho_for_go": 0.35,
        },
        "gate": {
            "verdict": gate,
            "minimum_oof_auc_for_go": 0.70,
            "minimum_oof_auc_for_park": 0.65,
            "observed_oof_auc": oof_auc,
        },
        "interpretation": (
            "The test asks whether requiring coherent support at both missing frames improves "
            "ranking; it does not authorize a threshold or a kernel from CV57."
        ),
    }
    if not math.isfinite(oof_auc):
        raise ValueError("OOF AUC is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": payload["gate"], "univariate_auc": univariate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
