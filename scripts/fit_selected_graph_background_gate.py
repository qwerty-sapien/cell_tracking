#!/usr/bin/env python3
"""Fine-tune the selected-graph gate with a low-weight background prior."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
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
from evaluate_selected_graph_conflict_gate import _load_gate  # noqa: E402
from fit_selected_graph_conflict_gate import (  # noqa: E402
    _load_partition,
    _proposal_metrics,
    _select_threshold,
    _sha256,
)


SCHEMA = "celltrack-alpha.selected-graph-conflict-gate.v2"
DEFAULT_LABELED = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_BACKGROUND = (
    ROOT / "reports/g02/selected_graph_background_prepare_v1/prepare.json"
)
DEFAULT_V1_GATE = ROOT / "reports/g02/selected_graph_conflict_gate_v1/gate.pt"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_background_gate_v1"


class BackgroundFitError(RuntimeError):
    """The background-gate fit contract was violated."""


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise BackgroundFitError(f"expected JSON object: {path}")
    return value


def _load_background(
    report: Mapping[str, Any],
) -> tuple[torch.Tensor, list[np.ndarray], dict[str, int]]:
    videos = report.get("videos")
    if not isinstance(videos, list):
        raise BackgroundFitError("background report lacks video records")
    features: list[np.ndarray] = []
    proposal_rows: list[np.ndarray] = []
    offset = 0
    proposal_count = 0
    for row in videos:
        path = ROOT / str(row["path"])
        value = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(value, Mapping)
            or "relative_features" not in value
            or "proposal_lengths" not in value
        ):
            raise BackgroundFitError(f"invalid background artifact: {path}")
        part = np.asarray(value["relative_features"], dtype=np.float32)
        lengths = np.asarray(value["proposal_lengths"], dtype=np.int64)
        if (
            part.ndim != 2
            or part.shape[1:] != (44,)
            or lengths.ndim != 1
            or np.any((lengths < 1) | (lengths > 2))
            or int(lengths.sum()) != len(part)
            or not np.isfinite(part).all()
        ):
            raise BackgroundFitError(f"invalid background shapes: {path}")
        cursor = 0
        for length in lengths.tolist():
            proposal_rows.append(
                np.arange(offset + cursor, offset + cursor + length, dtype=np.int64)
            )
            cursor += length
        features.append(part)
        offset += len(part)
        proposal_count += len(lengths)
    if not features or offset == 0:
        raise BackgroundFitError("background preparation contains no rows")
    return (
        torch.from_numpy(np.concatenate(features)),
        proposal_rows,
        {"rows": offset, "proposals": proposal_count, "videos": len(videos)},
    )


def _background_metrics(
    probabilities: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    confidence = np.asarray(
        [float(np.min(probabilities[rows])) for rows in groups], dtype=np.float64
    )
    return {
        "proposals": len(groups),
        "accepted": int(np.sum(confidence >= threshold)),
        "acceptance_rate": float(np.mean(confidence >= threshold)),
        "confidence_quantiles": {
            str(quantile): float(np.quantile(confidence, quantile))
            for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
    }


def _weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32),
        reduction="none",
    )
    return (raw * weights).sum() / weights.sum().clamp(min=1.0)


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BackgroundFitError(f"refusing to overwrite output: {args.output}")
    if not 0.0 < args.background_weight < 1.0:
        raise BackgroundFitError("background weight must be strictly between zero and one")
    labeled = _load(args.labeled)
    background = _load(args.background)
    labeled_scope = labeled.get("scope")
    background_scope = background.get("scope")
    if (
        labeled.get("schema") != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or background.get("schema")
        != "celltrack-alpha.selected-graph-background-prepare.v1"
        or not isinstance(labeled_scope, Mapping)
        or not isinstance(background_scope, Mapping)
        or len(labeled_scope.get("train_video_ids", [])) != 105
        or len(labeled_scope.get("validation_video_ids", [])) != 12
        or background_scope.get("train_video_ids") != labeled_scope.get("train_video_ids")
        or any(
            labeled_scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
        or any(
            background_scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise BackgroundFitError("fit inputs are not exact fit-only 105/12")
    by_id = {
        str(row["identity"]): ROOT / str(row["path"])
        for row in labeled["videos"]
    }
    train_ids = tuple(str(value) for value in labeled_scope["train_video_ids"])
    validation_ids = tuple(
        str(value) for value in labeled_scope["validation_video_ids"]
    )
    train_x, train_y, train_groups, train_totals = _load_partition(train_ids, by_id)
    validation_x, validation_y, validation_groups, validation_totals = _load_partition(
        validation_ids, by_id
    )
    background_x, background_groups, background_totals = _load_background(background)

    model, _v1_threshold = _load_gate(args.v1_gate)
    positives = int(train_y.sum().item())
    negatives = len(train_y) - positives
    positive_weight = max(
        1.0, min(float(negatives / positives), args.max_positive_weight)
    )
    all_x = torch.cat((train_x, background_x), dim=0)
    all_y = torch.cat((train_y, torch.zeros(len(background_x))), dim=0)
    all_weights = torch.cat(
        (
            torch.ones(len(train_x)),
            torch.full((len(background_x),), args.background_weight),
        ),
        dim=0,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float | int, ...] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(0, args.epochs + 1):
        if epoch:
            order = torch.randperm(
                len(all_y),
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            model.train()
            losses: list[float] = []
            for start in range(0, len(order), args.batch_size):
                rows = order[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = _weighted_loss(
                    model(all_x[rows]),
                    all_y[rows],
                    all_weights[rows],
                    positive_weight=positive_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))
            train_loss = float(np.mean(losses))
        else:
            train_loss = None

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x)
            validation_loss = float(
                F.binary_cross_entropy_with_logits(
                    validation_logits,
                    validation_y,
                    pos_weight=torch.tensor(positive_weight),
                ).item()
            )
            validation_probabilities = torch.sigmoid(validation_logits).numpy()
            background_probabilities = torch.sigmoid(model(background_x)).numpy()
        threshold, _sweep = _select_threshold(
            validation_probabilities, validation_groups
        )
        validation_metrics = _proposal_metrics(
            validation_probabilities,
            validation_groups,
            threshold=threshold,
        )
        background_metrics = _background_metrics(
            background_probabilities,
            background_groups,
            threshold=threshold,
        )
        admissible = (
            int(validation_metrics["accepted"]) >= 10
            and float(validation_metrics["precision"]) >= 0.95
        )
        key = (
            int(admissible),
            -float(background_metrics["acceptance_rate"]),
            int(validation_metrics["conservative_utility"]),
            int(validation_metrics["correct_rescues"]),
            -int(validation_metrics["harmful_swaps"]),
            -validation_loss,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "threshold": threshold,
                "validation": validation_metrics,
                "background": background_metrics,
                "admissible": admissible,
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise BackgroundFitError("fit produced no model state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(train_x)).numpy()
        validation_probabilities = torch.sigmoid(model(validation_x)).numpy()
        background_probabilities = torch.sigmoid(model(background_x)).numpy()
    threshold, sweep = _select_threshold(validation_probabilities, validation_groups)
    train_metrics = _proposal_metrics(
        train_probabilities, train_groups, threshold=threshold
    )
    validation_metrics = _proposal_metrics(
        validation_probabilities, validation_groups, threshold=threshold
    )
    background_metrics = _background_metrics(
        background_probabilities, background_groups, threshold=threshold
    )

    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "gate.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "hidden": model.hidden,
            "threshold": threshold,
            "state_dict": model.state_dict(),
            "background_weight": args.background_weight,
        },
        artifact,
    )
    result = {
        "schema": SCHEMA,
        "status": "fitted",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "initialization": "selected-graph gate v1",
            "labeled_supervision": "rescue/control",
            "background_supervision": "unlabeled low-weight abstention prior",
            "global_rescore": False,
            "gnn": False,
        },
        "data": {
            "labeled_train": train_totals,
            "labeled_validation": validation_totals,
            "background_train": background_totals,
        },
        "optimization": {
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "positive_weight": positive_weight,
            "background_weight": args.background_weight,
            "seed": args.seed,
            "parameters": model.parameter_count,
            "selection": (
                "labeled validation admissibility first, then lower fit-train "
                "background acceptance, then labeled utility and validation loss"
            ),
        },
        "threshold": {
            "selected": threshold,
            "selection_rule": (
                "validation accepted>=10 and precision>=0.95; maximize "
                "correct-2*harmful, then rescues, fewer harms, higher threshold"
            ),
            "sweep": sweep,
        },
        "metrics": {
            "labeled_train": train_metrics,
            "labeled_validation": validation_metrics,
            "background_train": background_metrics,
        },
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "inputs": {
            "labeled_prepare": str(args.labeled),
            "labeled_prepare_sha256": _sha256(args.labeled),
            "background_prepare": str(args.background),
            "background_prepare_sha256": _sha256(args.background),
            "v1_gate": str(args.v1_gate),
            "v1_gate_sha256": _sha256(args.v1_gate),
        },
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    value.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    value.add_argument("--v1-gate", type=Path, default=DEFAULT_V1_GATE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--epochs", type=int, default=60)
    value.add_argument("--batch-size", type=int, default=256)
    value.add_argument("--lr", type=float, default=5.0e-4)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--background-weight", type=float, default=0.05)
    value.add_argument("--max-positive-weight", type=float, default=20.0)
    value.add_argument("--seed", type=int, default=149)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "threshold": report["threshold"]["selected"],
                "validation": report["metrics"]["labeled_validation"],
                "background": report["metrics"]["background_train"],
            },
            sort_keys=True,
        )
    )
