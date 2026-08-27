#!/usr/bin/env python3
"""Fit an abstaining gate on selected-graph rescue/control comparisons."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha.relative_geometry_gate import (  # noqa: E402
    RelativeGeometryGate,
    relative_features,
)


SCHEMA = "celltrack-alpha.selected-graph-conflict-gate.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/selected_graph_conflict_gate_v1"


class SelectedGraphFitError(RuntimeError):
    """The selected-graph fit contract was violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectedGraphFitError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SelectedGraphFitError(f"expected JSON object: {path}")
    return value


def _load_partition(
    identities: Sequence[str],
    by_id: Mapping[str, Path],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    groups: list[dict[str, Any]] = []
    totals = {"rows": 0, "rescue_rows": 0, "control_rows": 0, "proposals": 0}
    offset = 0
    for identity in identities:
        path = by_id[identity]
        value = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "challenger_features",
            "selected_features",
            "challenger_logits",
            "selected_logits",
            "challenger_geometry_um",
            "selected_geometry_um",
            "labels",
            "challenger_pairs",
            "selected_pairs",
        }
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise SelectedGraphFitError(f"prepared examples are invalid: {path}")
        labels = np.asarray(value["labels"], dtype=np.int8)
        count = len(labels)
        relative = relative_features(
            np.asarray(value["challenger_features"], dtype=np.float32),
            np.asarray(value["selected_features"], dtype=np.float32),
            np.asarray(value["challenger_geometry_um"], dtype=np.float32),
            np.asarray(value["selected_geometry_um"], dtype=np.float32),
            challenger_logit=np.asarray(value["challenger_logits"], dtype=np.float32),
            winner_logit=np.asarray(value["selected_logits"], dtype=np.float32),
            dummy_logit=0.0,
            candidate_count=np.full(count, 2, dtype=np.float32),
        )
        challengers = np.asarray(value["challenger_pairs"], dtype=np.int64)
        selected = np.asarray(value["selected_pairs"], dtype=np.int64)
        if (
            labels.shape != (count,)
            or challengers.shape != (count, 2)
            or selected.shape != (count, 2)
            or np.any(~np.isin(labels, (0, 1)))
        ):
            raise SelectedGraphFitError("prepared labels or pairs are inconsistent")
        feature_parts.append(relative)
        label_parts.append(labels.astype(np.float32, copy=False))
        local: dict[tuple[int, int], list[int]] = {}
        for row, pair in enumerate(challengers):
            local.setdefault((int(pair[0]), int(pair[1])), []).append(row)
        for pair, rows in sorted(local.items()):
            observed = set(int(labels[row]) for row in rows)
            if len(observed) != 1:
                raise SelectedGraphFitError("one challenger proposal has mixed labels")
            groups.append(
                {
                    "identity": identity,
                    "challenger_pair": list(pair),
                    "rows": [offset + row for row in rows],
                    "label": observed.pop(),
                    "occupied_endpoints": len(
                        {
                            (int(selected[row, 0]), int(selected[row, 1]))
                            for row in rows
                        }
                    ),
                }
            )
        totals["rows"] += count
        totals["rescue_rows"] += int(labels.sum())
        totals["control_rows"] += int(count - labels.sum())
        totals["proposals"] += len(local)
        offset += count
    if not feature_parts or offset == 0:
        raise SelectedGraphFitError("partition has no selected-graph comparisons")
    return (
        torch.from_numpy(np.concatenate(feature_parts)),
        torch.from_numpy(np.concatenate(label_parts)),
        groups,
        totals,
    )


def _proposal_metrics(
    probabilities: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, float | int]:
    accepted = correct = harmful = available = 0
    occupied_correct = occupied_harmful = 0
    for group in groups:
        rows = np.asarray(group["rows"], dtype=np.int64)
        confidence = float(np.min(probabilities[rows]))
        label = int(group["label"])
        available += label
        if confidence < threshold:
            continue
        accepted += 1
        occupied = int(group["occupied_endpoints"])
        if label == 1:
            correct += 1
            occupied_correct += occupied
        else:
            harmful += 1
            occupied_harmful += occupied
    return {
        "proposals": len(groups),
        "available_rescues": available,
        "accepted": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "occupied_correct_edges": occupied_correct,
        "occupied_harmful_edges": occupied_harmful,
        "conservative_utility": correct - 2 * harmful,
        "precision": correct / max(accepted, 1),
        "rescue_recall": correct / max(available, 1),
    }


def _select_threshold(
    probabilities: np.ndarray, groups: Sequence[Mapping[str, Any]]
) -> tuple[float, list[dict[str, float | int]]]:
    rows = [
        {
            "threshold": float(threshold),
            **_proposal_metrics(probabilities, groups, threshold=float(threshold)),
        }
        for threshold in np.linspace(0.5, 0.995, 100)
    ]
    admissible = [
        row
        for row in rows
        if int(row["accepted"]) >= 10 and float(row["precision"]) >= 0.95
    ]
    if not admissible:
        return 1.0, rows
    selected = max(
        admissible,
        key=lambda row: (
            int(row["conservative_utility"]),
            int(row["correct_rescues"]),
            -int(row["occupied_harmful_edges"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), rows


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SelectedGraphFitError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    videos = prepare.get("videos")
    if (
        prepare.get("schema") != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
    ):
        raise SelectedGraphFitError("prepare report is not exact fit-only 105/12")
    by_id = {str(row["identity"]): ROOT / str(row["path"]) for row in videos}
    train_ids = tuple(str(value) for value in scope["train_video_ids"])
    validation_ids = tuple(str(value) for value in scope["validation_video_ids"])
    if set(by_id) != set((*train_ids, *validation_ids)):
        raise SelectedGraphFitError("prepared video coverage is incomplete")
    train_x, train_y, train_groups, train_totals = _load_partition(train_ids, by_id)
    validation_x, validation_y, validation_groups, validation_totals = _load_partition(
        validation_ids, by_id
    )
    positives = int(train_y.sum().item())
    negatives = len(train_y) - positives
    if positives < 2 or negatives < 2:
        raise SelectedGraphFitError("training classes are insufficient")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RelativeGeometryGate(hidden=args.hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    positive_weight = max(
        1.0, min(float(negatives / positives), args.max_positive_weight)
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32)
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(
            len(train_y), generator=torch.Generator().manual_seed(args.seed + epoch)
        )
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_x[rows]), train_y[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_x), validation_y).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise SelectedGraphFitError("fit did not produce a finite model")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(train_x)).numpy()
        validation_probabilities = torch.sigmoid(model(validation_x)).numpy()
    threshold, sweep = _select_threshold(validation_probabilities, validation_groups)
    train_metrics = _proposal_metrics(
        train_probabilities, train_groups, threshold=threshold
    )
    validation_metrics = _proposal_metrics(
        validation_probabilities, validation_groups, threshold=threshold
    )
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "gate.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "hidden": args.hidden,
            "threshold": threshold,
            "state_dict": model.state_dict(),
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
            "decision_unit": "challenger proposal against every occupied selected endpoint",
            "acceptance": "minimum row probability clears threshold",
            "controls": "GT-correct selected graph edges versus known non-edge challengers",
            "global_rescore": False,
            "gnn": False,
        },
        "data": {"train": train_totals, "validation": validation_totals},
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "positive_weight": positive_weight,
            "seed": args.seed,
            "best_validation_loss": best_loss,
            "parameters": model.parameter_count,
        },
        "threshold": {
            "selected": threshold,
            "selection_rule": (
                "validation accepted>=10 and precision>=0.95; maximize "
                "correct-2*harmful, then rescues, fewer occupied harms, higher threshold"
            ),
            "sweep": sweep,
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "inputs": {"prepare": str(args.prepare), "prepare_sha256": _sha256(args.prepare)},
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--hidden", type=int, default=16)
    value.add_argument("--lr", type=float, default=2.0e-3)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--max-positive-weight", type=float, default=20.0)
    value.add_argument("--seed", type=int, default=79)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "threshold": report["threshold"]["selected"],
                "validation": report["metrics"]["validation"],
            },
            sort_keys=True,
        )
    )
