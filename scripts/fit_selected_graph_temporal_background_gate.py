#!/usr/bin/env python3
"""Fit a grouped-OOF temporal selected-graph gate with background abstention."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from prepare_extended_radius_shell import _sha256  # noqa: E402


SCHEMA = "celltrack-alpha.selected-graph-temporal-background-gate.v1"
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_temporal_background_gate_v1"
CONFIGURATIONS = (
    {"name": "linear-bg005", "hidden": 0, "background_weight": 0.05},
    {"name": "mlp16-bg005", "hidden": 16, "background_weight": 0.05},
    {"name": "mlp16-bg020", "hidden": 16, "background_weight": 0.20},
    {"name": "mlp32-bg020", "hidden": 32, "background_weight": 0.20},
)


class TemporalBackgroundFitError(RuntimeError):
    """The grouped temporal/background fit contract failed."""


class TemporalGate(torch.nn.Module):
    """Small deployable row scorer."""

    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.hidden = int(hidden)
        self.network = (
            torch.nn.Linear(feature_count, 1)
            if hidden == 0
            else torch.nn.Sequential(
                torch.nn.Linear(feature_count, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise TemporalBackgroundFitError(f"expected JSON object: {path}")
    return value


def proposal_confidence(
    probabilities: np.ndarray,
    groups: Sequence[Sequence[Sequence[int]]],
) -> np.ndarray:
    """Aggregate max-over-ambiguous-options then min-over-conflicts."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = []
    for group in groups:
        if not group or any(not options for options in group):
            raise TemporalBackgroundFitError("proposal group is empty")
        result.append(
            min(
                max(float(probabilities[int(row)]) for row in options)
                for options in group
            )
        )
    value = np.asarray(result, dtype=np.float64)
    if not np.isfinite(value).all():
        raise TemporalBackgroundFitError("proposal confidence is non-finite")
    return value


def _video_data(row: Mapping[str, Any]) -> dict[str, Any]:
    value = torch.load(
        ROOT / str(row["path"]), map_location="cpu", weights_only=False
    )
    labeled_x = np.asarray(value["labeled_features"], dtype=np.float32)
    labeled_y = np.asarray(value["labeled_labels"], dtype=np.int8)
    pairs = np.asarray(value["labeled_challenger_pairs"], dtype=np.int64)
    background_x = np.asarray(value["background_features"], dtype=np.float32)
    background_groups = value["background_groups"]
    if (
        labeled_x.shape != (len(labeled_y), 48)
        or pairs.shape != (len(labeled_y), 2)
        or background_x.ndim != 2
        or background_x.shape[1:] != (48,)
        or np.any(~np.isin(labeled_y, (0, 1)))
    ):
        raise TemporalBackgroundFitError("prepared video arrays disagree")
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[(int(pair[0]), int(pair[1]))].append(index)
    labeled_groups = [[[index] for index in indices] for indices in grouped.values()]
    labels = []
    for indices in grouped.values():
        observed = {int(labeled_y[index]) for index in indices}
        if len(observed) != 1:
            raise TemporalBackgroundFitError("labeled proposal has mixed labels")
        labels.append(observed.pop())
    return {
        "identity": str(row["identity"]),
        "selected_edges": int(row["graph"]["raw_selected_edges"]),
        "labeled_x": labeled_x,
        "labeled_y": labeled_y,
        "labeled_groups": labeled_groups,
        "proposal_labels": np.asarray(labels, dtype=np.int8),
        "background_x": background_x,
        "background_groups": background_groups,
    }


def _fit_model(
    videos: Sequence[Mapping[str, Any]],
    *,
    hidden: int,
    background_weight: float,
    epochs: int,
    seed: int,
) -> tuple[TemporalGate, np.ndarray, np.ndarray, dict[str, Any]]:
    labeled = np.concatenate([video["labeled_x"] for video in videos])
    labels = np.concatenate([video["labeled_y"] for video in videos])
    backgrounds = [video["background_x"] for video in videos if len(video["background_x"])]
    background = (
        np.concatenate(backgrounds)
        if backgrounds
        else np.empty((0, labeled.shape[1]), dtype=np.float32)
    )
    if int(labels.sum()) < 2 or int(np.sum(labels == 0)) < 2:
        raise TemporalBackgroundFitError("training labels are insufficient")
    all_x = np.concatenate((labeled, background)).astype(np.float32, copy=False)
    all_y = np.concatenate(
        (labels.astype(np.float32), np.zeros(len(background), dtype=np.float32))
    )
    weights = np.concatenate(
        (
            np.ones(len(labeled), dtype=np.float32),
            np.full(len(background), background_weight, dtype=np.float32),
        )
    )
    mean = all_x.mean(axis=0)
    scale = all_x.std(axis=0)
    scale[scale < 1.0e-5] = 1.0
    x = torch.from_numpy((all_x - mean) / scale)
    y = torch.from_numpy(all_y)
    weight = torch.from_numpy(weights)
    positive_weight = min(
        20.0, max(1.0, float(np.sum(labels == 0) / np.sum(labels == 1)))
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TemporalGate(all_x.shape[1], hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005 if hidden == 0 else 0.002,
        weight_decay=1.0e-4,
    )
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(
            len(y), generator=torch.Generator().manual_seed(seed + epoch)
        )
        batch_losses = []
        model.train()
        for start in range(0, len(order), 512):
            rows = order[start : start + 512]
            optimizer.zero_grad(set_to_none=True)
            raw = F.binary_cross_entropy_with_logits(
                model(x[rows]),
                y[rows],
                pos_weight=torch.tensor(positive_weight),
                reduction="none",
            )
            loss = (raw * weight[rows]).sum() / weight[rows].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        losses.append(float(np.mean(batch_losses)))
    return model, mean, scale, {
        "final_loss": losses[-1],
        "positive_weight": positive_weight,
        "rows": len(all_y),
        "labeled_rows": len(labels),
        "background_rows": len(background),
    }


def _predict(
    model: TemporalGate,
    mean: np.ndarray,
    scale: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    if not len(features):
        return np.empty(0, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        return (
            torch.sigmoid(
                model(torch.from_numpy((features - mean) / scale))
            )
            .numpy()
            .astype(np.float64)
        )


def _scores(
    model: TemporalGate,
    mean: np.ndarray,
    scale: np.ndarray,
    videos: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for video in videos:
        labeled_probability = _predict(
            model, mean, scale, video["labeled_x"]
        )
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
                "background_confidence": proposal_confidence(
                    background_probability, video["background_groups"]
                )
                if video["background_groups"]
                else np.empty(0, dtype=np.float64),
            }
        )
    return result


def _metrics(scored: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    correct = harmful = background = 0
    video_support = []
    for video in scored:
        accepted = np.asarray(video["labeled_confidence"]) >= threshold
        labels = np.asarray(video["labels"])
        correct += int(np.sum(accepted & (labels == 1)))
        harmful += int(np.sum(accepted & (labels == 0)))
        accepted_background = int(
            np.sum(np.asarray(video["background_confidence"]) >= threshold)
        )
        background += accepted_background
        budget = max(16, int(math.ceil(0.005 * int(video["selected_edges"]))))
        video_support.append(
            {
                "identity": video["identity"],
                "accepted": accepted_background,
                "budget": budget,
                "passed": accepted_background <= budget,
            }
        )
    accepted = correct + harmful
    return {
        "accepted": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
        "background_accepted": background,
        "background_support_passed": all(
            row["passed"] for row in video_support
        ),
        "background_video_max_ratio": max(
            (
                row["accepted"] / max(row["budget"], 1)
                for row in video_support
            ),
            default=0.0,
        ),
        "video_support": video_support,
    }


def _select_threshold(
    scored: Sequence[Mapping[str, Any]],
    *,
    minimum_accepted: int,
    minimum_precision: float,
) -> tuple[float, dict[str, Any]]:
    positive = np.concatenate(
        [
            np.asarray(video["labeled_confidence"])[
                np.asarray(video["labels"]) == 1
            ]
            for video in scored
            if np.any(np.asarray(video["labels"]) == 1)
        ]
    )
    candidates = np.unique(positive)
    admissible = []
    for threshold in candidates:
        metrics = _metrics(scored, float(threshold))
        if (
            metrics["accepted"] >= minimum_accepted
            and metrics["precision"] >= minimum_precision
            and metrics["background_support_passed"]
        ):
            admissible.append((float(threshold), metrics))
    if not admissible:
        return 1.0, _metrics(scored, 1.0)
    return max(
        admissible,
        key=lambda item: (
            item[1]["conservative_utility"],
            item[1]["correct_rescues"],
            -item[1]["harmful_swaps"],
            item[0],
        ),
    )


def _fold(identity: str, count: int) -> int:
    return int.from_bytes(
        hashlib.sha256(identity.encode("ascii")).digest()[:8], "big"
    ) % count


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TemporalBackgroundFitError(
            f"refusing to overwrite output: {args.output}"
        )
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
    ):
        raise TemporalBackgroundFitError("prepare input is not exact fit-only 105/12")
    videos = [_video_data(row) for row in prepare["videos"]]
    train = [video for video in videos if video["identity"] in scope["train_video_ids"]]
    validation = [
        video for video in videos if video["identity"] in scope["validation_video_ids"]
    ]
    started = time.perf_counter()
    grid = []
    for config_index, config in enumerate(CONFIGURATIONS):
        totals = {
            "accepted": 0,
            "correct_rescues": 0,
            "harmful_swaps": 0,
            "background_accepted": 0,
        }
        folds = []
        support = True
        for fold in range(args.fold_count):
            fit_videos = [
                video
                for video in train
                if _fold(video["identity"], args.fold_count) != fold
            ]
            held_videos = [
                video
                for video in train
                if _fold(video["identity"], args.fold_count) == fold
            ]
            model, mean, scale, optimization = _fit_model(
                fit_videos,
                hidden=int(config["hidden"]),
                background_weight=float(config["background_weight"]),
                epochs=args.epochs,
                seed=args.seed + 100 * config_index + fold,
            )
            fit_scores = _scores(model, mean, scale, fit_videos)
            threshold, fit_metrics = _select_threshold(
                fit_scores,
                minimum_accepted=args.minimum_train_accepted,
                minimum_precision=args.minimum_precision,
            )
            held_metrics = _metrics(
                _scores(model, mean, scale, held_videos), threshold
            )
            for key in totals:
                totals[key] += int(held_metrics[key])
            support = support and bool(held_metrics["background_support_passed"])
            folds.append(
                {
                    "fold": fold,
                    "fit_videos": len(fit_videos),
                    "held_videos": len(held_videos),
                    "threshold": threshold,
                    "fit": {
                        key: fit_metrics[key]
                        for key in (
                            "accepted",
                            "correct_rescues",
                            "harmful_swaps",
                            "precision",
                            "conservative_utility",
                            "background_accepted",
                            "background_support_passed",
                        )
                    },
                    "held": {
                        key: held_metrics[key]
                        for key in (
                            "accepted",
                            "correct_rescues",
                            "harmful_swaps",
                            "precision",
                            "conservative_utility",
                            "background_accepted",
                            "background_support_passed",
                            "background_video_max_ratio",
                        )
                    },
                    "optimization": optimization,
                }
            )
        totals["precision"] = totals["correct_rescues"] / max(
            totals["accepted"], 1
        )
        totals["conservative_utility"] = (
            totals["correct_rescues"] - 2 * totals["harmful_swaps"]
        )
        admissible = (
            totals["accepted"] >= args.minimum_oof_accepted
            and totals["precision"] >= args.minimum_precision
            and support
        )
        grid.append(
            {
                "configuration": dict(config),
                "oof": totals,
                "oof_background_support_passed": support,
                "admissible": admissible,
                "folds": folds,
            }
        )
    admissible = [row for row in grid if row["admissible"]]
    selected = (
        max(
            admissible,
            key=lambda row: (
                row["oof"]["conservative_utility"],
                row["oof"]["correct_rescues"],
                -row["oof"]["harmful_swaps"],
            ),
        )
        if admissible
        else max(
            grid,
            key=lambda row: (
                row["oof"]["precision"],
                row["oof"]["conservative_utility"],
            ),
        )
    )
    config = selected["configuration"]
    model, mean, scale, optimization = _fit_model(
        train,
        hidden=int(config["hidden"]),
        background_weight=float(config["background_weight"]),
        epochs=args.epochs,
        seed=args.seed + 999,
    )
    train_scores = _scores(model, mean, scale, train)
    threshold, train_metrics = _select_threshold(
        train_scores,
        minimum_accepted=args.minimum_train_accepted,
        minimum_precision=args.minimum_precision,
    )
    validation_metrics = _metrics(
        _scores(model, mean, scale, validation), threshold
    )
    validation_passed = (
        bool(selected["admissible"])
        and validation_metrics["accepted"] >= args.minimum_validation_accepted
        and validation_metrics["precision"] >= args.minimum_precision
        and validation_metrics["conservative_utility"]
        >= args.minimum_validation_utility
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
            "fold_assignment": "sha256(video identity)[0:8] modulo 5",
            "minimum_oof_accepted": args.minimum_oof_accepted,
            "minimum_train_accepted": args.minimum_train_accepted,
            "minimum_validation_accepted": args.minimum_validation_accepted,
            "minimum_precision": args.minimum_precision,
            "minimum_validation_utility": args.minimum_validation_utility,
            "background_budget": "max(16, ceil(0.005 * selected edges)) per video",
            "validation_used_for_selection": False,
        },
        "grid": grid,
        "selected": selected,
        "final": {
            "configuration": config,
            "threshold": threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_passed": validation_passed,
            "optimization": optimization,
        },
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
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
    value.add_argument("--minimum-validation-utility", type=int, default=153)
    value.add_argument("--seed", type=int, default=607)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"]["configuration"],
                "oof": report["selected"]["oof"],
                "final": report["final"],
            },
            sort_keys=True,
        )
    )
