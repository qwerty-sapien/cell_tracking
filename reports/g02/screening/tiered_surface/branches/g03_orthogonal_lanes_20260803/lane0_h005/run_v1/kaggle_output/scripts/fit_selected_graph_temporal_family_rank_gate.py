#!/usr/bin/env python3
"""Fit family-tagged temporal rankers with within-family grouped OOF."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    _fit_model,
    _fold,
    _sha256,
    _video_data,
)
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _feature_scores,
    _metrics,
    _model_scores,
    _select,
)


SCHEMA = "celltrack-alpha.selected-graph-temporal-family-rank-gate.v1"
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_temporal_background_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/selected_graph_temporal_family_rank_gate_v1"
)
FAMILIES = ("44b6", "6bba")
CONFIGURATIONS = (
    {"name": "linear-bg005", "hidden": 0, "background_weight": 0.05},
    {"name": "mlp16-bg005", "hidden": 16, "background_weight": 0.05},
)


class TemporalFamilyRankError(RuntimeError):
    """The family-tagged ranker contract failed."""


def family(identity: str) -> str:
    """Extract the preregistered dataset-family prefix."""

    try:
        value = identity.split(":", 1)[1].split("_", 1)[0]
    except (AttributeError, IndexError) as exc:
        raise TemporalFamilyRankError(f"invalid video identity: {identity}") from exc
    if value not in FAMILIES:
        raise TemporalFamilyRankError(f"unknown video family: {identity}")
    return value


def _sum_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct_rescues"]) for row in rows)
    harmful = sum(int(row["harmful_swaps"]) for row in rows)
    background = sum(int(row["background_accepted"]) for row in rows)
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "background_accepted": background,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
    }


def _control(
    train: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    by_family = {}
    all_held = []
    for family_name in FAMILIES:
        family_videos = [v for v in train if family(v["identity"]) == family_name]
        folds = []
        held_rows = []
        for fold in range(args.fold_count):
            fit_videos = [
                v
                for v in family_videos
                if _fold(v["identity"], args.fold_count) != fold
            ]
            held_videos = [
                v
                for v in family_videos
                if _fold(v["identity"], args.fold_count) == fold
            ]
            threshold, fit_metrics = _select(
                _feature_scores(fit_videos, 44),
                minimum_accepted=args.minimum_train_accepted,
                minimum_precision=args.minimum_precision,
            )
            held_metrics = _metrics(
                _feature_scores(held_videos, 44), threshold
            )
            held_rows.append(held_metrics)
            all_held.append(held_metrics)
            folds.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "fit": fit_metrics,
                    "held": held_metrics,
                }
            )
        by_family[family_name] = {
            "oof": _sum_metrics(held_rows),
            "folds": folds,
        }
    return {"families": by_family, "oof": _sum_metrics(all_held)}


def _configuration_oof(
    train: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    config_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_family = {}
    all_held = []
    for family_index, family_name in enumerate(FAMILIES):
        family_videos = [v for v in train if family(v["identity"]) == family_name]
        folds = []
        held_rows = []
        for fold in range(args.fold_count):
            fit_videos = [
                v
                for v in family_videos
                if _fold(v["identity"], args.fold_count) != fold
            ]
            held_videos = [
                v
                for v in family_videos
                if _fold(v["identity"], args.fold_count) == fold
            ]
            model, mean, scale, optimization = _fit_model(
                fit_videos,
                hidden=int(config["hidden"]),
                background_weight=float(config["background_weight"]),
                epochs=args.epochs,
                seed=(
                    args.seed
                    + 1000 * config_index
                    + 100 * family_index
                    + fold
                ),
            )
            threshold, fit_metrics = _select(
                _model_scores(model, mean, scale, fit_videos),
                minimum_accepted=args.minimum_train_accepted,
                minimum_precision=args.minimum_precision,
            )
            held_metrics = _metrics(
                _model_scores(model, mean, scale, held_videos), threshold
            )
            held_rows.append(held_metrics)
            all_held.append(held_metrics)
            folds.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "fit": fit_metrics,
                    "held": held_metrics,
                    "optimization": optimization,
                }
            )
        by_family[family_name] = {
            "oof": _sum_metrics(held_rows),
            "folds": folds,
        }
    return {"families": by_family, "oof": _sum_metrics(all_held)}


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TemporalFamilyRankError(
            f"refusing to overwrite output: {args.output}"
        )
    prepare = json.loads(args.prepare.read_text(encoding="ascii"))
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
    ):
        raise TemporalFamilyRankError("unexpected prepare schema")
    scope = prepare["scope"]
    videos = [_video_data(row) for row in prepare["videos"]]
    train = [v for v in videos if v["identity"] in scope["train_video_ids"]]
    validation = [
        v for v in videos if v["identity"] in scope["validation_video_ids"]
    ]
    started = time.perf_counter()
    control = _control(train, args)
    grid = []
    for config_index, config in enumerate(CONFIGURATIONS):
        evidence = _configuration_oof(
            train, config, config_index=config_index, args=args
        )
        family_pass = all(
            evidence["families"][name]["oof"]["precision"]
            >= args.minimum_precision
            and evidence["families"][name]["oof"]["accepted_labeled"]
            >= args.minimum_family_oof_accepted
            for name in FAMILIES
        )
        utility_gain = (
            evidence["oof"]["conservative_utility"]
            - control["oof"]["conservative_utility"]
        )
        grid.append(
            {
                "configuration": dict(config),
                **evidence,
                "utility_gain_vs_control": utility_gain,
                "admissible": (
                    family_pass
                    and evidence["oof"]["precision"] >= args.minimum_precision
                    and utility_gain >= args.minimum_oof_utility_gain
                ),
            }
        )
    admissible = [row for row in grid if row["admissible"]]
    selected = (
        max(admissible, key=lambda row: row["utility_gain_vs_control"])
        if admissible
        else max(grid, key=lambda row: row["oof"]["precision"])
    )
    config = selected["configuration"]
    artifacts = {}
    validation_rows = []
    control_validation_rows = []
    final_families = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for family_index, family_name in enumerate(FAMILIES):
        train_family = [v for v in train if family(v["identity"]) == family_name]
        validation_family = [
            v for v in validation if family(v["identity"]) == family_name
        ]
        model, mean, scale, optimization = _fit_model(
            train_family,
            hidden=int(config["hidden"]),
            background_weight=float(config["background_weight"]),
            epochs=args.epochs,
            seed=args.seed + 9000 + family_index,
        )
        threshold, train_metrics = _select(
            _model_scores(model, mean, scale, train_family),
            minimum_accepted=args.minimum_train_accepted,
            minimum_precision=args.minimum_precision,
        )
        validation_metrics = _metrics(
            _model_scores(model, mean, scale, validation_family), threshold
        )
        control_threshold, control_train = _select(
            _feature_scores(train_family, 44),
            minimum_accepted=args.minimum_train_accepted,
            minimum_precision=args.minimum_precision,
        )
        control_validation = _metrics(
            _feature_scores(validation_family, 44), control_threshold
        )
        validation_rows.append(validation_metrics)
        control_validation_rows.append(control_validation)
        artifact = args.output / f"gate_{family_name}.pt"
        torch.save(
            {
                "schema": SCHEMA,
                "family": family_name,
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
        artifacts[family_name] = {
            "path": str(artifact),
            "sha256": _sha256(artifact),
        }
        final_families[family_name] = {
            "threshold": threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "control_threshold": control_threshold,
            "control_train": control_train,
            "control_validation": control_validation,
            "optimization": optimization,
        }
    validation_metrics = _sum_metrics(validation_rows)
    control_validation = _sum_metrics(control_validation_rows)
    validation_gain = (
        validation_metrics["conservative_utility"]
        - control_validation["conservative_utility"]
    )
    validation_passed = (
        bool(selected["admissible"])
        and validation_metrics["precision"] >= args.minimum_precision
        and all(
            final_families[name]["validation"]["precision"]
            >= args.minimum_precision
            for name in FAMILIES
        )
        and validation_gain >= args.minimum_validation_utility_gain
    )
    result = {
        "schema": SCHEMA,
        "status": "BANK" if validation_passed else "PARK",
        "scope": dict(scope),
        "selection_contract": {
            "families": list(FAMILIES),
            "family_source": "dataset identity prefix only",
            "fold_assignment": "sha256(video identity)[0:8] modulo 5 within family",
            "minimum_precision": args.minimum_precision,
            "minimum_family_oof_accepted": args.minimum_family_oof_accepted,
            "minimum_oof_utility_gain": args.minimum_oof_utility_gain,
            "minimum_validation_utility_gain": args.minimum_validation_utility_gain,
            "validation_used_for_selection": False,
        },
        "control": control,
        "grid": grid,
        "selected": selected,
        "final": {
            "families": final_families,
            "validation": validation_metrics,
            "control_validation": control_validation,
            "validation_utility_gain": validation_gain,
            "validation_passed": validation_passed,
        },
        "artifacts": artifacts,
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
    value.add_argument("--minimum-train-accepted", type=int, default=25)
    value.add_argument("--minimum-family-oof-accepted", type=int, default=25)
    value.add_argument("--minimum-precision", type=float, default=0.95)
    value.add_argument("--minimum-oof-utility-gain", type=int, default=5)
    value.add_argument("--minimum-validation-utility-gain", type=int, default=5)
    value.add_argument("--seed", type=int, default=1207)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": {
                    "configuration": report["selected"]["configuration"],
                    "families": {
                        name: report["selected"]["families"][name]["oof"]
                        for name in FAMILIES
                    },
                    "oof": report["selected"]["oof"],
                    "admissible": report["selected"]["admissible"],
                },
                "final": {
                    "validation": report["final"]["validation"],
                    "control_validation": report["final"][
                        "control_validation"
                    ],
                    "validation_utility_gain": report["final"][
                        "validation_utility_gain"
                    ],
                    "validation_passed": report["final"]["validation_passed"],
                },
            },
            sort_keys=True,
        )
    )
