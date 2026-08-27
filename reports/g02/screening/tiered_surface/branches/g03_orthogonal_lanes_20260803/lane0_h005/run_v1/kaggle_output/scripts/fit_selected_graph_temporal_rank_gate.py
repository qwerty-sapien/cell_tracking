#!/usr/bin/env python3
"""Evaluate temporal gates as proposal rankers inside the deployed video cap."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    SCHEMA as PARENT_SCHEMA,
    _fit_model,
    _fold,
    _predict,
    _sha256,
    _video_data,
    proposal_confidence,
)


SCHEMA = "celltrack-alpha.selected-graph-temporal-rank-gate.v1"
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_temporal_rank_gate_v1"
CONFIGURATIONS = (
    {"name": "linear-bg005", "hidden": 0, "background_weight": 0.05},
    {"name": "mlp16-bg005", "hidden": 16, "background_weight": 0.05},
)
THRESHOLD_GRID_SIZE = 257


class TemporalRankError(RuntimeError):
    """The capped temporal-ranker contract failed."""


def capped_video_metrics(
    labeled_scores: np.ndarray,
    labels: np.ndarray,
    background_scores: np.ndarray,
    *,
    threshold: float,
    budget: int,
) -> dict[str, int | float]:
    """Apply one deterministic cap to labeled and sampled background proposals."""

    candidates = [
        (float(score), 1, index, int(label))
        for index, (score, label) in enumerate(
            zip(labeled_scores, labels, strict=True)
        )
        if float(score) >= threshold
    ]
    # Background wins exact ties, making the sampled competition conservative.
    candidates.extend(
        (float(score), 0, index, -1)
        for index, score in enumerate(background_scores)
        if float(score) >= threshold
    )
    selected = sorted(
        candidates, key=lambda row: (-row[0], row[1], row[2])
    )[:budget]
    correct = sum(row[3] == 1 for row in selected)
    harmful = sum(row[3] == 0 for row in selected)
    background = sum(row[3] == -1 for row in selected)
    return {
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "background_accepted": background,
        "selected": len(selected),
        "budget": budget,
    }


def _feature_scores(
    videos: Sequence[Mapping[str, Any]], feature_index: int
) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        labeled = proposal_confidence(
            video["labeled_x"][:, feature_index], video["labeled_groups"]
        )
        background = (
            proposal_confidence(
                video["background_x"][:, feature_index],
                video["background_groups"],
            )
            if video["background_groups"]
            else np.empty(0, dtype=np.float64)
        )
        result.append(
            {
                "identity": video["identity"],
                "selected_edges": video["selected_edges"],
                "labeled_confidence": labeled,
                "labels": video["proposal_labels"],
                "background_confidence": background,
            }
        )
    return result


def _model_scores(model: Any, mean: np.ndarray, scale: np.ndarray, videos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        labeled_probability = _predict(model, mean, scale, video["labeled_x"])
        background_probability = _predict(
            model, mean, scale, video["background_x"]
        )
        result.append(
            {
                "identity": video["identity"],
                "selected_edges": video["selected_edges"],
                "labeled_confidence": proposal_confidence(
                    labeled_probability, video["labeled_groups"]
                ),
                "labels": video["proposal_labels"],
                "background_confidence": (
                    proposal_confidence(
                        background_probability, video["background_groups"]
                    )
                    if video["background_groups"]
                    else np.empty(0, dtype=np.float64)
                ),
            }
        )
    return result


def _metrics(scored: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    correct = harmful = background = selected = 0
    videos = []
    for video in scored:
        budget = max(16, int(math.ceil(0.005 * int(video["selected_edges"]))))
        metrics = capped_video_metrics(
            np.asarray(video["labeled_confidence"]),
            np.asarray(video["labels"]),
            np.asarray(video["background_confidence"]),
            threshold=threshold,
            budget=budget,
        )
        correct += int(metrics["correct_rescues"])
        harmful += int(metrics["harmful_swaps"])
        background += int(metrics["background_accepted"])
        selected += int(metrics["selected"])
        videos.append({"identity": video["identity"], **metrics})
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
        "background_accepted": background,
        "total_selected": selected,
        "videos": videos,
    }


def _select(
    scored: Sequence[Mapping[str, Any]],
    *,
    minimum_accepted: int,
    minimum_precision: float,
) -> tuple[float, dict[str, Any]]:
    positives = np.concatenate(
        [
            np.asarray(video["labeled_confidence"])[
                np.asarray(video["labels"]) == 1
            ]
            for video in scored
            if np.any(np.asarray(video["labels"]) == 1)
        ]
    )
    threshold_grid = np.unique(
        np.quantile(
            positives,
            np.linspace(0.0, 1.0, THRESHOLD_GRID_SIZE),
        )
    )
    candidates = []
    for threshold in threshold_grid:
        metrics = _metrics(scored, float(threshold))
        if (
            metrics["accepted_labeled"] >= minimum_accepted
            and metrics["precision"] >= minimum_precision
        ):
            candidates.append((float(threshold), metrics))
    if not candidates:
        return float("inf"), _metrics(scored, float("inf"))
    return max(
        candidates,
        key=lambda item: (
            item[1]["conservative_utility"],
            item[1]["correct_rescues"],
            -item[1]["harmful_swaps"],
            item[0],
        ),
    )


def _oof_control(
    train: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    totals = {"correct_rescues": 0, "harmful_swaps": 0, "background_accepted": 0}
    folds = []
    for fold in range(args.fold_count):
        fit_videos = [v for v in train if _fold(v["identity"], args.fold_count) != fold]
        held_videos = [v for v in train if _fold(v["identity"], args.fold_count) == fold]
        threshold, fit_metrics = _select(
            _feature_scores(fit_videos, 44),
            minimum_accepted=args.minimum_train_accepted,
            minimum_precision=args.minimum_precision,
        )
        held_metrics = _metrics(_feature_scores(held_videos, 44), threshold)
        for key in totals:
            totals[key] += int(held_metrics[key])
        folds.append({"fold": fold, "threshold": threshold, "fit": fit_metrics, "held": held_metrics})
    totals["accepted_labeled"] = totals["correct_rescues"] + totals["harmful_swaps"]
    totals["precision"] = totals["correct_rescues"] / max(totals["accepted_labeled"], 1)
    totals["conservative_utility"] = totals["correct_rescues"] - 2 * totals["harmful_swaps"]
    return {"oof": totals, "folds": folds}


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TemporalRankError(f"refusing to overwrite output: {args.output}")
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    if prepare.get("schema") != "celltrack-alpha.selected-graph-temporal-background-prepare.v1":
        raise TemporalRankError("unexpected prepare schema")
    scope = prepare["scope"]
    videos = [_video_data(row) for row in prepare["videos"]]
    train = [v for v in videos if v["identity"] in scope["train_video_ids"]]
    validation = [v for v in videos if v["identity"] in scope["validation_video_ids"]]
    started = time.perf_counter()
    control = _oof_control(train, args)
    grid = []
    for config_index, config in enumerate(CONFIGURATIONS):
        totals = {"correct_rescues": 0, "harmful_swaps": 0, "background_accepted": 0}
        folds = []
        for fold in range(args.fold_count):
            fit_videos = [v for v in train if _fold(v["identity"], args.fold_count) != fold]
            held_videos = [v for v in train if _fold(v["identity"], args.fold_count) == fold]
            model, mean, scale, optimization = _fit_model(
                fit_videos,
                hidden=int(config["hidden"]),
                background_weight=float(config["background_weight"]),
                epochs=args.epochs,
                seed=args.seed + 100 * config_index + fold,
            )
            threshold, fit_metrics = _select(
                _model_scores(model, mean, scale, fit_videos),
                minimum_accepted=args.minimum_train_accepted,
                minimum_precision=args.minimum_precision,
            )
            held_metrics = _metrics(
                _model_scores(model, mean, scale, held_videos), threshold
            )
            for key in totals:
                totals[key] += int(held_metrics[key])
            folds.append({"fold": fold, "threshold": threshold, "fit": fit_metrics, "held": held_metrics, "optimization": optimization})
        totals["accepted_labeled"] = totals["correct_rescues"] + totals["harmful_swaps"]
        totals["precision"] = totals["correct_rescues"] / max(totals["accepted_labeled"], 1)
        totals["conservative_utility"] = totals["correct_rescues"] - 2 * totals["harmful_swaps"]
        admissible = (
            totals["accepted_labeled"] >= args.minimum_oof_accepted
            and totals["precision"] >= args.minimum_precision
            and totals["conservative_utility"] >= control["oof"]["conservative_utility"] + args.minimum_oof_utility_gain
        )
        grid.append({"configuration": dict(config), "oof": totals, "admissible": admissible, "folds": folds})
    admissible = [row for row in grid if row["admissible"]]
    selected = max(admissible, key=lambda row: row["oof"]["conservative_utility"]) if admissible else max(grid, key=lambda row: row["oof"]["conservative_utility"])
    config = selected["configuration"]
    model, mean, scale, optimization = _fit_model(
        train,
        hidden=int(config["hidden"]),
        background_weight=float(config["background_weight"]),
        epochs=args.epochs,
        seed=args.seed + 999,
    )
    threshold, train_metrics = _select(
        _model_scores(model, mean, scale, train),
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_precision,
    )
    validation_metrics = _metrics(
        _model_scores(model, mean, scale, validation), threshold
    )
    control_threshold, control_train = _select(
        _feature_scores(train, 44),
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_precision,
    )
    control_validation = _metrics(
        _feature_scores(validation, 44), control_threshold
    )
    validation_gain = (
        validation_metrics["conservative_utility"]
        - control_validation["conservative_utility"]
    )
    validation_passed = (
        bool(selected["admissible"])
        and validation_metrics["precision"] >= args.minimum_precision
        and validation_metrics["accepted_labeled"] >= args.minimum_validation_accepted
        and validation_gain >= args.minimum_validation_utility_gain
    )
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "gate.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "feature_count": 48,
            "hidden": int(config["hidden"]),
            "mean": mean,
            "scale": scale,
            "threshold": threshold,
            "state_dict": model.state_dict(),
            "configuration": config,
        },
        artifact,
    )
    result = {
        "schema": SCHEMA,
        "status": "BANK" if validation_passed else "PARK",
        "scope": dict(scope),
        "selection_contract": {
            "policy": "threshold then joint labeled/background top per-video cap",
            "cap": "max(16, ceil(0.005 * selected edges))",
            "threshold_grid": (
                f"{THRESHOLD_GRID_SIZE} equally spaced positive-score quantiles"
            ),
            "minimum_precision": args.minimum_precision,
            "minimum_oof_utility_gain": args.minimum_oof_utility_gain,
            "minimum_validation_utility_gain": args.minimum_validation_utility_gain,
            "validation_used_for_selection": False,
        },
        "control": control,
        "grid": grid,
        "selected": selected,
        "final": {
            "configuration": config,
            "threshold": threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "control_threshold": control_threshold,
            "control_train": control_train,
            "control_validation": control_validation,
            "validation_utility_gain": validation_gain,
            "validation_passed": validation_passed,
            "optimization": optimization,
        },
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "provenance": {"prepare": str(args.prepare), "prepare_sha256": _sha256(args.prepare), "parent_fit_schema": PARENT_SCHEMA},
        "execution": {"device": "cpu", "elapsed_seconds": time.perf_counter() - started, "gpu_action": False, "kaggle_action": False},
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--fold-count", type=int, default=5)
    value.add_argument("--minimum-train-accepted", type=int, default=50)
    value.add_argument("--minimum-oof-accepted", type=int, default=50)
    value.add_argument("--minimum-validation-accepted", type=int, default=10)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--minimum-oof-utility-gain", type=int, default=5)
    value.add_argument("--minimum-validation-utility-gain", type=int, default=5)
    value.add_argument("--seed", type=int, default=907)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(json.dumps({"status": report["status"], "selected": report["selected"], "final": report["final"]}, sort_keys=True))
