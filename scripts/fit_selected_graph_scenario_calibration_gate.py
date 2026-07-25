#!/usr/bin/env python3
"""Screen fit-only per-video competition calibration for pair-NCC proposals."""

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


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from fit_selected_graph_temporal_background_gate import (  # noqa: E402
    _fold,
    _sha256,
    _video_data,
)
from fit_selected_graph_temporal_rank_gate import (  # noqa: E402
    _feature_scores,
    _metrics,
    _select,
)


SCHEMA = "celltrack-alpha.selected-graph-scenario-calibration-gate.v1"
CONTRACT_SCHEMA = (
    "celltrack-alpha.selected-graph-scenario-calibration-contract.v1"
)
DEFAULT_CONTRACT = (
    ROOT / "reports/g02/selected_graph_scenario_calibration_contract_v1.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "037dccf60cbae4000d2ba67590d8bf42d84b117d9d79a7d24cc96e829b9269a1"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/selected_graph_scenario_calibration_gate_v1"
)
METHODS = {"raw", "robust50_90", "robust75_95", "empirical_cdf", "cap_rank"}


class ScenarioCalibrationError(RuntimeError):
    """The fit-only scenario-calibration contract failed closed."""


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
        raise ScenarioCalibrationError("contract SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScenarioCalibrationError("contract cannot be loaded") from exc
    sections = (
        "scope",
        "mechanism",
        "configurations",
        "selection",
        "validation_gate",
        "promotion",
    )
    if (
        not isinstance(value, dict)
        or value.get("schema") != CONTRACT_SCHEMA
        or value.get("status") != "locked"
        or any(section not in value for section in sections)
        or not isinstance(value["configurations"], list)
        or len(value["configurations"]) != 11
    ):
        raise ScenarioCalibrationError("contract is malformed")
    scope = value["scope"]
    mechanism = value["mechanism"]
    promotion = value["promotion"]
    if (
        scope.get("fit_train_video_count") != 105
        or scope.get("fit_validation_video_count") != 12
        or scope.get("outer_fold_count") != 5
        or scope.get("validation_used_for_selection") is not False
        or any(
            scope.get(name) is not False
            for name in ("cv57_access", "kaggle_action", "gpu_action", "runpod_action")
        )
        or mechanism.get("base_score")
        != "prepared full 3-D pair-NCC proposal confidence at feature index 44"
        or mechanism.get("robust_scale_floor") != 0.05
        or "dataset family prefix"
        not in mechanism.get("forbidden_router_inputs", ())
        or promotion.get("full_cv57_authorized") is not False
    ):
        raise ScenarioCalibrationError("contract scope is not fail-closed")
    names = set()
    for configuration in value["configurations"]:
        if (
            not isinstance(configuration, dict)
            or set(configuration) != {"name", "method", "density_penalty"}
            or configuration["method"] not in METHODS
            or not isinstance(configuration["name"], str)
            or configuration["name"] in names
            or type(configuration["density_penalty"]) not in (int, float)
            or not 0.0 <= float(configuration["density_penalty"]) <= 0.5
            or (
                configuration["method"] in {"raw", "cap_rank"}
                and float(configuration["density_penalty"]) != 0.0
            )
        ):
            raise ScenarioCalibrationError("configuration grid is invalid")
        names.add(configuration["name"])
    if value["configurations"][0] != {
        "name": "raw-control",
        "method": "raw",
        "density_penalty": 0.0,
    }:
        raise ScenarioCalibrationError("raw control identity drifted")
    return value


def _budget(selected_edges: int) -> int:
    if type(selected_edges) is not int or selected_edges <= 0:
        raise ScenarioCalibrationError("selected-edge count is invalid")
    return max(16, int(math.ceil(0.005 * selected_edges)))


def transform_video_scores(
    labeled: np.ndarray,
    background: np.ndarray,
    *,
    budget: int,
    method: str,
    density_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Normalize one video's unlabeled score distribution deterministically."""

    labeled = np.asarray(labeled, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if (
        labeled.ndim != 1
        or background.ndim != 1
        or budget <= 0
        or method not in METHODS
        or not np.isfinite(labeled).all()
        or not np.isfinite(background).all()
        or not 0.0 <= float(density_penalty) <= 0.5
        or (
            method in {"raw", "cap_rank"}
            and float(density_penalty) != 0.0
        )
    ):
        raise ScenarioCalibrationError("score distribution is invalid")
    combined = np.concatenate((labeled, background))
    if not len(combined):
        return (
            labeled.copy(),
            background.copy(),
            {
                "proposal_count": 0,
                "budget": budget,
                "competition": 0.0,
                "center": 0.0,
                "scale": 1.0,
            },
        )
    if method == "raw":
        transformed = combined.copy()
        center = 0.0
        scale = 1.0
    elif method in {"robust50_90", "robust75_95"}:
        lower, upper = (
            (0.50, 0.90) if method == "robust50_90" else (0.75, 0.95)
        )
        center, high = np.quantile(combined, (lower, upper))
        scale = max(float(high - center), 0.05)
        transformed = (combined - float(center)) / scale
    elif method == "empirical_cdf":
        ordered = np.sort(combined)
        denominator = max(len(ordered) - 1, 1)
        transformed = (
            np.searchsorted(ordered, combined, side="left") / denominator
        )
        center = 0.0
        scale = 1.0
    else:
        ordered = np.sort(combined)
        greater = len(ordered) - np.searchsorted(
            ordered, combined, side="right"
        )
        transformed = -greater.astype(np.float64) / budget
        center = 0.0
        scale = float(budget)
    competition = math.log1p(len(combined) / budget)
    transformed = transformed - float(density_penalty) * competition
    if not np.isfinite(transformed).all():
        raise ScenarioCalibrationError("normalized scores are non-finite")
    return (
        transformed[: len(labeled)],
        transformed[len(labeled) :],
        {
            "proposal_count": len(combined),
            "budget": budget,
            "competition": competition,
            "center": float(center),
            "scale": float(scale),
        },
    )


def normalize_scored(
    scored: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for video in scored:
        budget = _budget(int(video["selected_edges"]))
        labeled, background, normalization = transform_video_scores(
            np.asarray(video["labeled_confidence"]),
            np.asarray(video["background_confidence"]),
            budget=budget,
            method=str(configuration["method"]),
            density_penalty=float(configuration["density_penalty"]),
        )
        result.append(
            {
                "identity": str(video["identity"]),
                "selected_edges": int(video["selected_edges"]),
                "labeled_confidence": labeled,
                "labels": np.asarray(video["labels"], dtype=np.int8),
                "background_confidence": background,
                "normalization": normalization,
            }
        )
    return result


def _sum_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct_rescues"]) for row in rows)
    harmful = sum(int(row["harmful_swaps"]) for row in rows)
    background = sum(int(row["background_accepted"]) for row in rows)
    selected = sum(int(row["total_selected"]) for row in rows)
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / max(accepted, 1),
        "conservative_utility": correct - 2 * harmful,
        "background_accepted": background,
        "total_selected": selected,
    }


def _oof(
    train: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
    *,
    fold_count: int,
    minimum_fit_accepted: int,
    minimum_precision: float,
) -> dict[str, Any]:
    folds = []
    for fold in range(fold_count):
        fit_videos = [
            value
            for value in train
            if _fold(value["identity"], fold_count) != fold
        ]
        held_videos = [
            value
            for value in train
            if _fold(value["identity"], fold_count) == fold
        ]
        threshold, fit_metrics = _select(
            normalize_scored(_feature_scores(fit_videos, 44), configuration),
            minimum_accepted=minimum_fit_accepted,
            minimum_precision=minimum_precision,
        )
        held_metrics = _metrics(
            normalize_scored(_feature_scores(held_videos, 44), configuration),
            threshold,
        )
        folds.append(
            {
                "fold": fold,
                "threshold": threshold,
                "fit": fit_metrics,
                "held": held_metrics,
            }
        )
    return {
        "oof": _sum_metrics([row["held"] for row in folds]),
        "folds": folds,
    }


def _family(identity: str) -> str:
    try:
        value = identity.split(":", 1)[1].split("_", 1)[0]
    except (AttributeError, IndexError) as exc:
        raise ScenarioCalibrationError("video identity is invalid") from exc
    if value not in {"44b6", "6bba"}:
        raise ScenarioCalibrationError("video family is unknown")
    return value


def _family_metrics(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("44b6", "6bba"):
        rows = [
            row
            for row in metrics["videos"]
            if _family(str(row["identity"])) == name
        ]
        result[name] = _sum_metrics(
            [
                {
                    **row,
                    "total_selected": row["selected"],
                }
                for row in rows
            ]
        )
    return result


def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ScenarioCalibrationError(
            f"refusing to overwrite output: {args.output}"
        )
    contract = load_contract(args.contract, args.contract_sha256)
    prepare_path = ROOT / contract["scope"]["prepare_report"]
    if (
        not prepare_path.is_file()
        or _sha256(prepare_path)
        != contract["scope"]["prepare_report_sha256"]
    ):
        raise ScenarioCalibrationError("prepare report identity drifted")
    prepare = json.loads(prepare_path.read_text(encoding="ascii"))
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-temporal-background-prepare.v1"
    ):
        raise ScenarioCalibrationError("prepare report schema drifted")
    scope = prepare["scope"]
    if (
        len(scope["train_video_ids"])
        != contract["scope"]["fit_train_video_count"]
        or len(scope["validation_video_ids"])
        != contract["scope"]["fit_validation_video_count"]
    ):
        raise ScenarioCalibrationError("fit partition identity drifted")
    videos = [_video_data(row) for row in prepare["videos"]]
    train = [
        value
        for value in videos
        if value["identity"] in scope["train_video_ids"]
    ]
    validation = [
        value
        for value in videos
        if value["identity"] in scope["validation_video_ids"]
    ]
    selection = contract["selection"]
    validation_gate = contract["validation_gate"]
    fold_count = int(contract["scope"]["outer_fold_count"])
    started = time.perf_counter()
    grid = []
    for configuration in contract["configurations"]:
        evidence = _oof(
            train,
            configuration,
            fold_count=fold_count,
            minimum_fit_accepted=int(selection["minimum_fit_accepted"]),
            minimum_precision=float(selection["minimum_oof_precision"]),
        )
        grid.append(
            {
                "configuration": dict(configuration),
                **evidence,
            }
        )
    control = grid[0]
    for row in grid:
        row["utility_gain_over_raw"] = (
            row["oof"]["conservative_utility"]
            - control["oof"]["conservative_utility"]
        )
        row["admissible"] = (
            row["configuration"]["method"] != "raw"
            and row["oof"]["accepted_labeled"]
            >= int(selection["minimum_oof_accepted"])
            and row["oof"]["precision"]
            >= float(selection["minimum_oof_precision"])
            and min(
                float(fold["held"]["precision"]) for fold in row["folds"]
            )
            >= float(selection["minimum_each_fold_precision"])
            and row["utility_gain_over_raw"]
            >= int(selection["minimum_oof_utility_gain_over_raw"])
        )
    admissible = [row for row in grid[1:] if row["admissible"]]
    selected = (
        max(
            admissible,
            key=lambda row: (
                row["utility_gain_over_raw"],
                row["oof"]["precision"],
            ),
        )
        if admissible
        else max(
            grid[1:],
            key=lambda row: (
                row["oof"]["precision"],
                row["utility_gain_over_raw"],
            ),
        )
    )
    configuration = selected["configuration"]
    threshold, train_metrics = _select(
        normalize_scored(_feature_scores(train, 44), configuration),
        minimum_accepted=int(selection["minimum_fit_accepted"]),
        minimum_precision=float(selection["minimum_oof_precision"]),
    )
    validation_metrics = _metrics(
        normalize_scored(_feature_scores(validation, 44), configuration),
        threshold,
    )
    control_threshold, control_train = _select(
        _feature_scores(train, 44),
        minimum_accepted=int(selection["minimum_fit_accepted"]),
        minimum_precision=float(selection["minimum_oof_precision"]),
    )
    control_validation = _metrics(
        _feature_scores(validation, 44), control_threshold
    )
    candidate_families = _family_metrics(validation_metrics)
    control_families = _family_metrics(control_validation)
    family_gate = {}
    for name in ("44b6", "6bba"):
        candidate = candidate_families[name]
        baseline = control_families[name]
        family_gate[name] = {
            "candidate": candidate,
            "control": baseline,
            "utility_delta": (
                candidate["conservative_utility"]
                - baseline["conservative_utility"]
            ),
            "passed": (
                candidate["conservative_utility"]
                >= baseline["conservative_utility"]
                and (
                    candidate["accepted_labeled"] < 10
                    or candidate["precision"]
                    >= float(
                        validation_gate[
                            "minimum_family_precision_when_at_least_10_accepted"
                        ]
                    )
                )
            ),
        }
    validation_gain = (
        validation_metrics["conservative_utility"]
        - control_validation["conservative_utility"]
    )
    passed = (
        bool(selected["admissible"])
        and validation_metrics["accepted_labeled"]
        >= int(validation_gate["minimum_accepted"])
        and validation_metrics["precision"]
        >= float(validation_gate["minimum_precision"])
        and validation_gain
        >= int(validation_gate["minimum_utility_gain_over_raw"])
        and all(value["passed"] for value in family_gate.values())
    )
    result = {
        "schema": SCHEMA,
        "status": "BANK" if passed else "PARK",
        "scope": {
            **dict(scope),
            "cv57_access": False,
            "validation_used_for_selection": False,
        },
        "control": control,
        "grid": grid,
        "selected": selected,
        "final": {
            "configuration": configuration,
            "threshold": threshold,
            "train": train_metrics,
            "validation": validation_metrics,
            "control_threshold": control_threshold,
            "control_train": control_train,
            "control_validation": control_validation,
            "validation_utility_gain": validation_gain,
            "families": family_gate,
            "passed": passed,
        },
        "decision": {
            "status": "BANK" if passed else "PARK",
            "next_action": (
                contract["promotion"]["on_pass"]
                if passed
                else contract["promotion"]["on_fail"]
            ),
            "full_cv57_authorized": False,
        },
        "provenance": {
            "contract": str(args.contract),
            "contract_sha256": args.contract_sha256,
            "prepare": str(prepare_path),
            "prepare_sha256": _sha256(prepare_path),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    run_bh.atomic_create_json(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    value.add_argument(
        "--contract-sha256", default=DEFAULT_CONTRACT_SHA256
    )
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = fit(parser().parse_args())
    print(json.dumps(report["decision"], sort_keys=True))
