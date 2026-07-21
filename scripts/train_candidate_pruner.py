#!/usr/bin/env python3
"""Train a tiny fold-safe node-pruning calibration from candidate tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.localization import sha256_file  # noqa: E402
from celltrack_alpha.scoring import load_scoring_calibration  # noqa: E402


DEFAULT_FEATURES = (
    "feature__log1p_intensity",
    "feature__log1p_local_density",
    "feature__has_tile_raw",
    "feature__has_tile_contrast",
    "feature__has_cover",
    "feature__has_cover_contrast",
    "feature__has_cover_dog_bright",
    "feature__has_cover_dog_dark",
    "feature__has_dog_bright",
    "feature__has_dog_dark",
    "feature__has_zshift",
    "feature__has_zshift_dog_dark_m5",
    "feature__has_zshift_dog_bright_m3",
    "feature__source_family_count",
    "feature__rescue_source_count",
    "feature__tile_only",
    "feature__cover_only",
    "feature__dog_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-csv", type=Path, required=True)
    parser.add_argument("--table-report", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/general_model_split_manifest.json"))
    parser.add_argument("--fold-id", default="0")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--group-by", choices=("route", "prefix", "route_prefix"), default="route_prefix")
    parser.add_argument("--min-positive-recall", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--keep-top-k-per-t", type=int, default=None)
    parser.add_argument("--min-keep-per-t", type=int, default=0)
    parser.add_argument("--replace-node-score", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/general_model_candidate_pruner_fold0.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/general_model_candidate_pruner_fold0_report.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def float_value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def group_key(row: dict[str, str], group_by: str) -> tuple[str, str]:
    route = str(row.get("route", ""))
    prefix = str(row.get("prefix", ""))
    if group_by == "route":
        return "by_route", route
    if group_by == "prefix":
        return "by_prefix", prefix
    return "by_route_prefix", f"{route}:{prefix}"


def fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int, lr: float, l2: float) -> tuple[np.ndarray, float]:
    n, p = x.shape
    weights = np.zeros(p, dtype=np.float64)
    pos = float(np.mean(y))
    bias = math.log(max(pos, 1e-4) / max(1.0 - pos, 1e-4))
    for _ in range(max(1, int(steps))):
        logits = x @ weights + bias
        logits = np.clip(logits, -50.0, 50.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        err = pred - y
        grad_w = (x.T @ err) / n + float(l2) * weights
        grad_b = float(np.mean(err))
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b
    return weights, float(bias)


def threshold_metrics(logits: np.ndarray, y: np.ndarray, min_recall: float) -> dict[str, float]:
    candidates = sorted({float(v) for v in logits})
    if candidates:
        candidates = [min(candidates) - 1e-6] + candidates + [max(candidates) + 1e-6]
    else:
        candidates = [0.0]
    best: dict[str, float] | None = None
    for threshold in candidates:
        pred = logits >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        beta2 = 4.0
        f2 = (1 + beta2) * precision * recall / max(1e-9, beta2 * precision + recall)
        row = {
            "threshold": float(threshold),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f2": float(f2),
            "predicted_positive": float(np.sum(pred)),
        }
        if recall + 1e-12 < min_recall:
            continue
        key = (row["precision"], row["f2"], row["threshold"])
        if best is None or key > (best["precision"], best["f2"], best["threshold"]):
            best = row
    if best is not None:
        return best
    # Fall back to the best F2 threshold if the recall target is impossible.
    for threshold in candidates:
        pred = logits >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        beta2 = 4.0
        f2 = (1 + beta2) * precision * recall / max(1e-9, beta2 * precision + recall)
        row = {
            "threshold": float(threshold),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f2": float(f2),
            "predicted_positive": float(np.sum(pred)),
        }
        if best is None or row["f2"] > best["f2"]:
            best = row
    assert best is not None
    return best


def load_split(split_manifest_path: Path, fold_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise ValueError("split manifest root must be an object")
    fold = split.get("folds", {}).get(str(fold_id))
    if not isinstance(fold, dict):
        raise ValueError(f"fold {fold_id!r} is not present in {split_manifest_path}")
    return split, fold, sha256_file(split_manifest_path)


def main() -> int:
    args = parse_args()
    features = parse_csv_strings(args.features)
    if not features:
        raise ValueError("at least one feature is required")
    if not 0.0 <= args.min_positive_recall <= 1.0:
        raise ValueError("--min-positive-recall must be in [0, 1]")
    rows = read_rows(args.node_csv)
    table_report = json.loads(args.table_report.read_text(encoding="utf-8"))
    split, fold, split_hash = load_split(args.split_manifest, str(args.fold_id))
    source_datasets = [str(v) for v in table_report.get("source_datasets", [])]
    calibration_train = [str(v) for v in fold.get("calibration_train", [])]
    validation = [str(v) for v in fold.get("validation", [])]
    excluded = [
        str(v)
        for v in split.get("excluded_from_general_score", {}).get("duplicate_public_ids", [])
    ]
    if set(source_datasets) - set(calibration_train):
        raise ValueError("table source_datasets are outside fold calibration_train")
    if set(source_datasets) & set(excluded):
        raise ValueError("table source_datasets include excluded duplicate public IDs")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    usable_status = {"positive_7um_assignment", "hard_negative_near_sparse_gt"}
    for row in rows:
        if row.get("label_status") not in usable_status:
            continue
        if row.get("label") not in {"0", "1", 0, 1}:
            continue
        groups[group_key(row, args.group_by)].append(row)

    policy: dict[str, Any] = {"default": {}}
    model_reports: list[dict[str, Any]] = []
    for (section, key), group_rows in sorted(groups.items()):
        y = np.asarray([float(row["label"]) for row in group_rows], dtype=np.float64)
        if len(set(int(v) for v in y)) < 2:
            model_reports.append({"section": section, "key": key, "status": "skipped_one_class", "rows": len(group_rows)})
            continue
        x = np.asarray([[float_value(row, name) for name in features] for row in group_rows], dtype=np.float64)
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std < 1e-6] = 1.0
        x_std = (x - mean) / std
        weights_std, bias_std = fit_logistic(x_std, y, steps=args.steps, lr=args.learning_rate, l2=args.l2)
        logits = x_std @ weights_std + bias_std
        threshold = threshold_metrics(logits, y, args.min_positive_recall)
        weights_raw = weights_std / std
        bias_raw = float(bias_std - np.sum(weights_std * mean / std))
        row_policy: dict[str, Any] = {
            "weights": {
                name.replace("feature__", ""): float(weight)
                for name, weight in zip(features, weights_raw, strict=True)
            },
            "bias": bias_raw,
            "min_score": float(threshold["threshold"]),
            "min_keep_per_t": int(args.min_keep_per_t),
            "replace_node_score": bool(args.replace_node_score),
            "feature_names": [name.replace("feature__", "") for name in features],
            "training_label_counts": dict(Counter(str(int(v)) for v in y)),
        }
        if args.keep_top_k_per_t is not None:
            row_policy["keep_top_k_per_t"] = int(args.keep_top_k_per_t)
        policy.setdefault(section, {})[key] = row_policy
        model_reports.append(
            {
                "section": section,
                "key": key,
                "status": "trained",
                "rows": len(group_rows),
                "positive_rows": int(np.sum(y == 1)),
                "negative_rows": int(np.sum(y == 0)),
                "threshold_metrics": threshold,
                "weights": row_policy["weights"],
                "bias": bias_raw,
            }
        )

    payload = {
        "kind": "candidate_graph_scoring",
        "version": 1,
        "created_at_unix": time.time(),
        "training_table": {
            "node_csv": str(args.node_csv),
            "node_csv_sha256": sha256_file(args.node_csv),
            "table_report": str(args.table_report),
            "table_report_sha256": sha256_file(args.table_report),
        },
        "provenance": {
            "status": "fold_safe",
            "fold_id": str(args.fold_id),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": split_hash,
            "source_datasets": source_datasets,
            "calibration_train": calibration_train,
            "validation_datasets": validation,
            "excluded_duplicate_public_ids": excluded,
        },
        "node_policy": policy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Validate the emitted calibration before reporting success.
    load_scoring_calibration(args.output, split_manifest_path=args.split_manifest, expected_fold_id=str(args.fold_id))
    report = {
        "kind": "candidate_pruner_training_report",
        "created_at_unix": time.time(),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "node_csv": str(args.node_csv),
        "features": features,
        "group_by": args.group_by,
        "min_positive_recall": float(args.min_positive_recall),
        "keep_top_k_per_t": args.keep_top_k_per_t,
        "model_reports": model_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trained_models": sum(1 for row in model_reports if row["status"] == "trained")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
