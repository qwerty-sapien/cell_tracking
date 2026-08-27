#!/usr/bin/env python3
"""Fit an abstaining competitor-relative swap gate on frozen P4 fit records.

Unlike the archived conflict residual, this model never globally rescales edge
costs.  It learns whether one source-local challenger should replace the frozen
pairwise winner and is trained with both rescue examples and correct-winner
controls.  Sources for which the explicit no-link logit wins are excluded: the
gate is not authorized to create a link.
"""

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
    choose_swap,
    relative_features,
)


SCHEMA = "celltrack-alpha.relative-geometry-swap-gate.v1"
DEFAULT_PREPARE = ROOT / "reports/g02/p4_edge_attention_prototype_v1/prepare.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/relative_geometry_swap_gate_v1"


class SwapGateFitError(RuntimeError):
    """The fit-only swap-gate contract was violated."""


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
        raise SwapGateFitError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SwapGateFitError(f"expected JSON object: {path}")
    return value


def source_examples(
    record: Mapping[str, Any], *, dummy_logit: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert one frozen frame record into challenger/control source groups."""

    features = np.asarray(record["features"], dtype=np.float32)
    logits = np.asarray(record["base_logits"], dtype=np.float32)
    geometry = np.asarray(record["geometry_um"], dtype=np.float32)
    source_index = np.asarray(record["source_index"], dtype=np.int64)
    supervised = np.asarray(record["supervised_sources"], dtype=np.int64)
    target_rows = np.asarray(
        record.get("target_rows", record.get("positive_rows")), dtype=np.int64
    )
    groups: list[dict[str, Any]] = []
    counts = {
        "supervised_sources": 0,
        "dummy_winner_sources": 0,
        "unsupported_unlink_sources": 0,
        "correct_winner_controls": 0,
        "wrong_winner_rescues": 0,
        "challenger_rows": 0,
        "positive_challengers": 0,
    }
    for source, target_row in zip(supervised, target_rows, strict=True):
        rows = np.flatnonzero(source_index == source)
        if len(rows) < 1 or (int(target_row) >= 0 and int(target_row) not in rows):
            raise SwapGateFitError("prepared source supervision is inconsistent")
        counts["supervised_sources"] += 1
        best_edge = int(rows[int(np.argmax(logits[rows]))])
        winner = -1 if float(dummy_logit) >= float(logits[best_edge]) else best_edge
        if winner < 0:
            counts["dummy_winner_sources"] += 1
        if int(target_row) < 0 and winner >= 0:
            # This gate only selects edge challengers.  Learning an unlink action
            # requires a separate topology-safe control and is intentionally
            # outside this first CPU proxy.
            counts["unsupported_unlink_sources"] += 1
            continue
        challengers = rows if winner < 0 else rows[rows != winner]
        labels = (challengers == int(target_row)).astype(np.float32)
        if winner < 0:
            winner_features = np.zeros((len(challengers), features.shape[1]), dtype=np.float32)
            source_position = geometry[challengers, :3]
            winner_geometry = np.column_stack((source_position, source_position)).astype(
                np.float32, copy=False
            )
            winner_logits = np.full(len(challengers), dummy_logit, dtype=np.float32)
            winner_dummy = np.ones(len(challengers), dtype=np.float32)
        else:
            winner_features = np.repeat(features[winner][None], len(challengers), axis=0)
            winner_geometry = np.repeat(geometry[winner][None], len(challengers), axis=0)
            winner_logits = np.full(len(challengers), logits[winner], dtype=np.float32)
            winner_dummy = np.zeros(len(challengers), dtype=np.float32)
        relative = relative_features(
            features[challengers],
            winner_features,
            geometry[challengers],
            winner_geometry,
            challenger_logit=logits[challengers],
            winner_logit=winner_logits,
            dummy_logit=dummy_logit,
            candidate_count=np.full(len(challengers), len(rows) + 1, dtype=np.float32),
            winner_is_dummy=winner_dummy,
        )
        rescue = winner != int(target_row)
        if rescue:
            counts["wrong_winner_rescues"] += 1
        else:
            counts["correct_winner_controls"] += 1
        counts["challenger_rows"] += len(challengers)
        counts["positive_challengers"] += int(labels.sum())
        groups.append(
            {
                "features": relative,
                "labels": labels,
                "challenger_rows": challengers.astype(np.int64, copy=False),
                "winner_row": winner,
                "positive_row": int(target_row),
                "rescue": rescue,
            }
        )
    return groups, counts


def _load_partition(
    paths: Sequence[Path], *, dummy_logit: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for path in paths:
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise SwapGateFitError(f"prepared frame file is invalid: {path}")
        for record in records:
            local, counts = source_examples(record, dummy_logit=dummy_logit)
            groups.extend(local)
            for name, value in counts.items():
                totals[name] = totals.get(name, 0) + int(value)
    return groups, totals


def _flatten(groups: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not groups:
        raise SwapGateFitError("partition has no gate examples")
    features = np.concatenate([np.asarray(group["features"]) for group in groups])
    labels = np.concatenate([np.asarray(group["labels"]) for group in groups])
    return torch.from_numpy(features), torch.from_numpy(labels)


def _source_metrics(
    model: RelativeGeometryGate,
    groups: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, float | int]:
    model.eval()
    accepted = correct = harmful = available = 0
    with torch.no_grad():
        for group in groups:
            probabilities = torch.sigmoid(
                model(torch.as_tensor(group["features"], dtype=torch.float32))
            ).numpy()
            decision = choose_swap(
                np.asarray(group["challenger_rows"]),
                probabilities,
                threshold=threshold,
            )
            available += int(bool(group["rescue"]))
            if decision.challenger_row is None:
                continue
            accepted += 1
            if decision.challenger_row == int(group["positive_row"]):
                correct += 1
            else:
                harmful += 1
    return {
        "sources": len(groups),
        "available_rescues": available,
        "accepted": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "net_rescues": correct - harmful,
        "precision": correct / max(accepted, 1),
        "rescue_recall": correct / max(available, 1),
    }


def _select_threshold(
    model: RelativeGeometryGate, groups: Sequence[Mapping[str, Any]]
) -> tuple[float, list[dict[str, float | int]]]:
    candidates = tuple(float(value) for value in np.linspace(0.5, 0.99, 50))
    rows = [
        {"threshold": threshold, **_source_metrics(model, groups, threshold=threshold)}
        for threshold in candidates
    ]
    # Require at least 80% observed precision.  Among admissible thresholds,
    # maximize net rescues, then correct rescues, then choose the higher threshold.
    admissible = [
        row
        for row in rows
        if int(row["accepted"]) > 0 and float(row["precision"]) >= 0.8
    ]
    if not admissible:
        return 1.0, rows
    selected = max(
        admissible,
        key=lambda row: (
            int(row["net_rescues"]),
            int(row["correct_rescues"]),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), rows


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SwapGateFitError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    videos = prepare.get("videos")
    if (
        not isinstance(scope, Mapping)
        or not isinstance(videos, list)
        or scope.get("held_out_used") is not False
        or scope.get("audit_used") is not False
        or scope.get("test_used") is not False
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
    ):
        raise SwapGateFitError("prepared records do not preserve the exact fit-only 105/12 split")
    by_id = {str(row["identity"]): ROOT / str(row["path"]) for row in videos}
    train_paths = [by_id[str(value)] for value in scope["train_video_ids"]]
    validation_paths = [by_id[str(value)] for value in scope["validation_video_ids"]]
    if any(not path.is_file() for path in (*train_paths, *validation_paths)):
        raise SwapGateFitError("prepared frame files are incomplete")
    dummy_logit = float(prepare["frozen_dummy_logit"])
    train_groups, train_counts = _load_partition(train_paths, dummy_logit=dummy_logit)
    validation_groups, validation_counts = _load_partition(
        validation_paths, dummy_logit=dummy_logit
    )
    train_features, train_labels = _flatten(train_groups)
    validation_features, validation_labels = _flatten(validation_groups)
    positive_count = int(train_labels.sum().item())
    negative_count = len(train_labels) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise SwapGateFitError("training classes are insufficient")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RelativeGeometryGate(hidden=args.hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    positive_weight = max(
        1.0, min(float(negative_count / positive_count), args.max_positive_weight)
    )
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32)
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_labels), generator=generator)
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_features[rows]), train_labels[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_features), validation_labels).item()
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
        raise SwapGateFitError("training did not produce a finite model")
    model.load_state_dict(best_state, strict=True)
    threshold, threshold_sweep = _select_threshold(model, validation_groups)
    train_metrics = _source_metrics(model, train_groups, threshold=threshold)
    validation_metrics = _source_metrics(model, validation_groups, threshold=threshold)

    args.output.mkdir(parents=True, exist_ok=False)
    artifact_path = args.output / "gate.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "hidden": args.hidden,
            "threshold": threshold,
            "dummy_logit": dummy_logit,
            "state_dict": model.state_dict(),
        },
        artifact_path,
    )
    result = {
        "schema": SCHEMA,
        "status": "fitted",
        "scope": {
            "train_video_ids": list(scope["train_video_ids"]),
            "validation_video_ids": list(scope["validation_video_ids"]),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "mechanism": {
            "decision": "source-local challenger versus frozen winner",
            "controls": "all challengers against a correct frozen winner are negative",
            "no_link_policy": "exclude sources where frozen dummy wins; never create a link",
            "topology": "frozen cap-8 candidate graph",
            "global_rescore": False,
            "gnn": False,
        },
        "data": {"train": train_counts, "validation": validation_counts},
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
            "selection_rule": "validation precision >=0.8; max net rescue; max rescue; highest threshold",
            "sweep": threshold_sweep,
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
        "inputs": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
        },
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--epochs", type=int, default=80)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--hidden", type=int, default=16)
    value.add_argument("--lr", type=float, default=2.0e-3)
    value.add_argument("--weight-decay", type=float, default=1.0e-4)
    value.add_argument("--max-positive-weight", type=float, default=20.0)
    value.add_argument("--seed", type=int, default=73)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.hidden <= 0
        or args.lr <= 0.0
        or args.weight_decay < 0.0
        or args.max_positive_weight < 1.0
    ):
        raise SwapGateFitError("optimization arguments are invalid")
    result = fit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "threshold": result["threshold"]["selected"],
                "validation": result["metrics"]["validation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
