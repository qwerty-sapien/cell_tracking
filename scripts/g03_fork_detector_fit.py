#!/usr/bin/env python3
"""Fit and directly evaluate the preregistered raw-crop fork detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


PREREG_SHA256 = "1c75dcc2a5c989b51e8d9d4e86fed7bdbd15c036ab96c48e912adc999184f6ab"
MIN_RECALL = 25.0 / 38.0
MIN_PRECISION = 25.0 / 35.0


class ForkFitError(RuntimeError):
    """The trained-detector direct screen failed its immutable contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise ForkFitError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    if path.exists():
        raise ForkFitError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return sha256(path)


def load_crop_data(path: Path, *, development: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        required = {"x", "y", "identity", "node_id", "frame"}
        if development:
            required.add("split")
        if not required <= set(value.files):
            raise ForkFitError(f"crop payload schema drifted: {path}")
        result = {key: np.asarray(value[key]) for key in required}
    if result["x"].ndim != 5 or tuple(result["x"].shape[1:]) != (3, 8, 16, 16):
        raise ForkFitError("crop tensor shape drifted")
    if len(result["x"]) != len(result["y"]) or not np.isfinite(result["x"]).all():
        raise ForkFitError("crop payload is malformed")
    if set(np.unique(result["y"]).tolist()) != {0, 1}:
        raise ForkFitError("crop payload lacks a class")
    return result


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = score >= threshold
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def select_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float | None, list[dict[str, Any]]]:
    rows = []
    selected = None
    for threshold in sorted(set(np.asarray(score, dtype=np.float64).tolist()), reverse=True):
        metrics = classification_metrics(y, score, threshold)
        metrics["pass"] = (
            metrics["recall"] >= MIN_RECALL
            and metrics["precision"] >= MIN_PRECISION
        )
        rows.append(metrics)
        if selected is None and metrics["pass"]:
            selected = float(threshold)
    return selected, rows


def fit(development: Path, model_output: Path, result_output: Path) -> dict[str, Any]:
    data = load_crop_data(development, development=True)
    split = data["split"].astype(str)
    if set(split.tolist()) != {"train", "validation"}:
        raise ForkFitError("development splits drifted")
    train = split == "train"
    validation = split == "validation"
    x = data["x"].reshape(len(data["x"]), -1).astype(np.float64)
    y = data["y"].astype(np.uint8)
    model = LogisticRegression(
        penalty="l2",
        dual=False,
        tol=1.0e-4,
        C=0.01,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight="balanced",
        random_state=20260803,
        solver="liblinear",
        max_iter=1000,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    model.fit(x[train], y[train])
    if int(model.n_iter_[0]) >= 1000:
        raise ForkFitError("liblinear failed to converge")
    validation_score = model.predict_proba(x[validation])[:, 1]
    threshold, threshold_rows = select_threshold(y[validation], validation_score)
    pass_gate = threshold is not None
    if threshold is None:
        # Diagnostic metrics at 0.5 only; final population remains unopened.
        metrics = classification_metrics(y[validation], validation_score, 0.5)
    else:
        metrics = classification_metrics(y[validation], validation_score, threshold)
    model_digest = atomic_npz(
        model_output,
        coef=np.asarray(model.coef_, dtype=np.float64),
        intercept=np.asarray(model.intercept_, dtype=np.float64),
        classes=np.asarray(model.classes_, dtype=np.int64),
        threshold=np.asarray([np.nan if threshold is None else threshold], dtype=np.float64),
        n_features=np.asarray([model.n_features_in_], dtype=np.int64),
    )
    result = {
        "schema": "celltrack-alpha.g03-trained-fork-cross-family-result.v1",
        "status": (
            "PASS_CROSS_FAMILY_UNLOCK_FINAL_DIRECT_EVALUATION"
            if pass_gate
            else "FAIL_CROSS_FAMILY_STOP_BEFORE_FINAL_38"
        ),
        "lane_preregistration_sha256": PREREG_SHA256,
        "development_path": str(development),
        "development_sha256": sha256(development),
        "model": {
            "path": str(model_output),
            "sha256": model_digest,
            "type": "class-balanced L2 logistic raw-voxel probe",
            "C": 0.01,
            "solver": "liblinear",
            "random_state": 20260803,
            "iterations": int(model.n_iter_[0]),
            "features": int(model.n_features_in_),
        },
        "train": {
            "family": "6bba",
            "samples": int(train.sum()),
            "positives": int(y[train].sum()),
            "negatives": int((y[train] == 0).sum()),
        },
        "validation": {
            "family": "44b6",
            "samples": int(validation.sum()),
            "positives": int(y[validation].sum()),
            "negatives": int((y[validation] == 0).sum()),
            "selected_threshold": threshold,
            "metrics": metrics,
            "minimum_recall": MIN_RECALL,
            "minimum_precision": MIN_PRECISION,
            "pass": pass_gate,
            "threshold_candidates": threshold_rows,
        },
        "final_38_access": False,
        "graph_integration": False,
        "gpu_hours": 0.0,
    }
    atomic_json(result_output, result)
    return result


def load_model(path: Path) -> tuple[np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as value:
        coef = np.asarray(value["coef"], dtype=np.float64)
        intercept = float(np.asarray(value["intercept"])[0])
        threshold = float(np.asarray(value["threshold"])[0])
        n_features = int(np.asarray(value["n_features"])[0])
    if coef.shape != (1, n_features) or not np.isfinite(coef).all() or not np.isfinite(threshold):
        raise ForkFitError("model is not final-evaluation eligible")
    return coef[0], intercept, threshold


def frontier(y: np.ndarray, score: np.ndarray, budgets: tuple[int, ...] = (0, 5, 10)) -> list[dict[str, Any]]:
    thresholds = [float("inf")] + sorted(set(score.tolist()), reverse=True)
    rows = []
    for budget in budgets:
        best = {"fp_budget": budget, "tp": 0, "fp": 0, "threshold": None}
        for threshold in thresholds:
            predicted = score >= threshold
            tp = int(np.sum((y == 1) & predicted))
            fp = int(np.sum((y == 0) & predicted))
            if fp <= budget and (tp > best["tp"] or (tp == best["tp"] and fp < best["fp"])):
                best = {
                    "fp_budget": budget,
                    "tp": tp,
                    "fp": fp,
                    "threshold": None if np.isinf(threshold) else threshold,
                }
        rows.append(best)
    return rows


def evaluate_final(
    final_crops: Path,
    model_path: Path,
    cross_family_result: Path,
    output: Path,
) -> dict[str, Any]:
    prior = json.loads(cross_family_result.read_text(encoding="ascii"))
    if (
        prior.get("status") != "PASS_CROSS_FAMILY_UNLOCK_FINAL_DIRECT_EVALUATION"
        or prior.get("model", {}).get("sha256") != sha256(model_path)
        or prior.get("validation", {}).get("pass") is not True
    ):
        raise ForkFitError("cross-family gate did not unlock final evaluation")
    data = load_crop_data(final_crops, development=False)
    y = data["y"].astype(np.uint8)
    if int(y.sum()) != 38 or int((y == 0).sum()) != 38:
        raise ForkFitError("final direct population count drifted")
    coef, intercept, threshold = load_model(model_path)
    x = data["x"].reshape(len(y), -1).astype(np.float64)
    logits = x @ coef + intercept
    score = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    metrics = classification_metrics(y, score, threshold)
    passed = metrics["tp"] >= 25 and metrics["fp"] <= 10
    result = {
        "schema": "celltrack-alpha.g03-trained-fork-final-direct-result.v1",
        "status": (
            "PASS_DIRECT_GATE_UNLOCK_GRAPH_PREREG_REVIEW"
            if passed
            else "FAIL_DIRECT_GATE_NO_GRAPH_INTEGRATION"
        ),
        "lane_preregistration_sha256": PREREG_SHA256,
        "final_crops": {"path": str(final_crops), "sha256": sha256(final_crops)},
        "model": {"path": str(model_path), "sha256": sha256(model_path)},
        "cross_family_result": {
            "path": str(cross_family_result),
            "sha256": sha256(cross_family_result),
        },
        "population": {"positives": 38, "matched_negatives": 38},
        "frozen_validation_threshold_metrics": metrics,
        "tie_safe_diagnostic_frontier": frontier(y, score),
        "gate": {"minimum_tp": 25, "maximum_fp": 10, "pass": passed},
        "banked_additive_values": [
            {"tp": 25, "fp": 0, "score_addition": 0.0595},
            {"tp": 25, "fp": 5, "score_addition": 0.0532},
            {"tp": 25, "fp": 10, "score_addition": 0.0481},
        ],
        "graph_integration_executed": False,
        "gpu_hours": 0.0,
        "submission": False,
        "runpod": False,
    }
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--development", type=Path, required=True)
    fit_parser.add_argument("--model-output", type=Path, required=True)
    fit_parser.add_argument("--result-output", type=Path, required=True)
    final_parser = subparsers.add_parser("evaluate-final")
    final_parser.add_argument("--final-crops", type=Path, required=True)
    final_parser.add_argument("--model", type=Path, required=True)
    final_parser.add_argument("--cross-family-result", type=Path, required=True)
    final_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        result = fit(args.development, args.model_output, args.result_output)
    else:
        result = evaluate_final(
            args.final_crops, args.model, args.cross_family_result, args.output
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"validation"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
