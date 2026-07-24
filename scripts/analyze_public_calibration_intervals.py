#!/usr/bin/env python3
"""Audit affine local-to-public calibration using displayed-score intervals."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports/g02/screening/p0_public_calibration_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/g02/screening/p0_interval_feasibility_v1.json"
SCHEMA = "celltrack-alpha.p0-interval-feasibility.v1"
# The open-upper shrink is deliberately much larger than the numerical
# half-space tolerance, while remaining seven orders below the 0.001 display
# bin. This preserves strict interval semantics without losing vertices in the
# ill-conditioned near-tied pair.
OPEN_UPPER_EPSILON = 1e-10
FEASIBILITY_TOLERANCE = 1e-12
EXPECTED_PAIRS = (
    (
        "backward motion c=0, divisions off",
        0.8483945155545333,
        0.860,
    ),
    (
        "backward motion c=2, divisions off",
        0.8721499474601118,
        0.878,
    ),
    (
        "backward motion c=0.25, divisions off",
        0.8731495935289003,
        0.877,
    ),
)
APNCC_DIRECT_LOCAL = 0.8815220699597315


class CalibrationIntervalError(RuntimeError):
    """The frozen calibration input or interval program is invalid."""


def _constraint_vertices(
    points: Sequence[dict[str, Any]],
    *,
    expansion: float = 0.0,
    minimum_slope: float | None = None,
) -> list[tuple[float, float]]:
    """Return feasible (intercept, slope) vertices for interval constraints."""

    if len(points) < 2 or expansion < 0:
        raise CalibrationIntervalError(
            "at least two points and nonnegative expansion are required"
        )
    constraints: list[tuple[float, float, float]] = []
    for point in points:
        x = float(point["local_cv57"])
        low, high = map(float, point["public_interval"])
        if not low < high:
            raise CalibrationIntervalError("public interval must be nonempty")
        # A * intercept + B * slope <= C.
        constraints.append((-1.0, -x, -(low - expansion)))
        constraints.append(
            (1.0, x, high + expansion - OPEN_UPPER_EPSILON)
        )
    if minimum_slope is not None:
        constraints.append((0.0, -1.0, -minimum_slope))

    vertices: list[tuple[float, float]] = []
    for first, second in itertools.combinations(constraints, 2):
        a1, b1, c1 = first
        a2, b2, c2 = second
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) <= 1e-18:
            continue
        intercept = (c1 * b2 - c2 * b1) / determinant
        slope = (a1 * c2 - a2 * c1) / determinant
        if all(
            a * intercept + b * slope <= c + FEASIBILITY_TOLERANCE
            for a, b, c in constraints
        ):
            vertices.append((intercept, slope))

    unique: list[tuple[float, float]] = []
    for vertex in sorted(vertices):
        if not any(
            abs(vertex[0] - existing[0]) <= 1e-11
            and abs(vertex[1] - existing[1]) <= 1e-11
            for existing in unique
        ):
            unique.append(vertex)
    return unique


def affine_envelope(
    points: Sequence[dict[str, Any]],
    *,
    expansion: float = 0.0,
    minimum_slope: float | None = None,
    predict_at: float | None = None,
) -> dict[str, Any]:
    vertices = _constraint_vertices(
        points,
        expansion=expansion,
        minimum_slope=minimum_slope,
    )
    result: dict[str, Any] = {
        "feasible": bool(vertices),
        "minimum_slope_constraint": minimum_slope,
        "uniform_interval_expansion": expansion,
        "vertex_count": len(vertices),
    }
    if not vertices:
        return result
    result["intercept_range"] = [
        min(vertex[0] for vertex in vertices),
        max(vertex[0] for vertex in vertices),
    ]
    result["slope_range"] = [
        min(vertex[1] for vertex in vertices),
        max(vertex[1] for vertex in vertices),
    ]
    if predict_at is not None:
        predictions = [
            intercept + slope * predict_at for intercept, slope in vertices
        ]
        result["prediction_at_local"] = predict_at
        result["prediction_range"] = [min(predictions), max(predictions)]
    return result


def minimum_uniform_expansion(
    points: Sequence[dict[str, Any]],
    *,
    minimum_slope: float | None,
) -> float:
    if affine_envelope(points, minimum_slope=minimum_slope)["feasible"]:
        return 0.0
    lower = 0.0
    upper = 0.001
    while not affine_envelope(
        points, expansion=upper, minimum_slope=minimum_slope
    )["feasible"]:
        upper *= 2.0
        if upper > 1.0:
            raise CalibrationIntervalError(
                "could not bound minimum interval expansion"
            )
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if affine_envelope(
            points, expansion=middle, minimum_slope=minimum_slope
        )["feasible"]:
            upper = middle
        else:
            lower = middle
    return upper


def interval_miss(
    prediction_range: Sequence[float],
    observed_interval: Sequence[float],
) -> float:
    predicted_low, predicted_high = map(float, prediction_range)
    observed_low, observed_high = map(float, observed_interval)
    if predicted_high < observed_low:
        return observed_low - predicted_high
    if predicted_low >= observed_high:
        return predicted_low - observed_high
    return 0.0


def point_ols(points: Sequence[dict[str, Any]]) -> dict[str, float]:
    if len(points) < 2:
        raise CalibrationIntervalError("OLS requires at least two points")
    xs = [float(point["local_cv57"]) for point in points]
    ys = [float(point["public_displayed"]) for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise CalibrationIntervalError("OLS local scores are degenerate")
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)
    ) / denominator
    return {"intercept": y_mean - slope * x_mean, "slope": slope}


def pair_slope_range(
    first: dict[str, Any], second: dict[str, Any]
) -> list[float]:
    if second["local_cv57"] < first["local_cv57"]:
        first, second = second, first
    delta_x = float(second["local_cv57"]) - float(first["local_cv57"])
    if delta_x <= 0:
        raise CalibrationIntervalError("pair local scores must differ")
    first_low, first_high = map(float, first["public_interval"])
    second_low, second_high = map(float, second["public_interval"])
    return [
        (second_low - first_high) / delta_x,
        (second_high - first_low) / delta_x,
    ]


def validate_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("artifact") != "p0_public_calibration_v1":
        raise CalibrationIntervalError("unexpected calibration artifact")
    if payload.get("public_score_display_resolution") != 0.001:
        raise CalibrationIntervalError("display resolution drifted")
    points = payload.get("pairs")
    if not isinstance(points, list) or len(points) != len(EXPECTED_PAIRS):
        raise CalibrationIntervalError("expected exactly three frozen pairs")
    for point, expected in zip(points, EXPECTED_PAIRS, strict=True):
        configuration, local, public = expected
        observed = (
            point.get("configuration"),
            point.get("local_cv57"),
            point.get("public_displayed"),
        )
        if observed != expected:
            raise CalibrationIntervalError(
                f"calibration pair identity drifted: {observed!r}"
            )
        expected_interval = [public - 0.0005, public + 0.0005]
        if any(
            abs(float(left) - right) > 1e-15
            for left, right in zip(
                point.get("public_interval", []),
                expected_interval,
                strict=True,
            )
        ):
            raise CalibrationIntervalError(
                f"public interval drifted for {configuration}"
            )
    return points


def build_report(
    payload: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    points = validate_input(payload)
    pairwise = []
    prospective = []
    for first, second in itertools.combinations(points, 2):
        label = (
            f"{first['configuration']} ↔ {second['configuration']}"
        )
        pairwise.append(
            {
                "pair": label,
                "implied_slope_range": pair_slope_range(first, second),
            }
        )
        prospective.append(
            {
                "fit_pair": label,
                "unconstrained": affine_envelope(
                    [first, second],
                    predict_at=APNCC_DIRECT_LOCAL,
                ),
                "monotone_nonnegative": affine_envelope(
                    [first, second],
                    minimum_slope=0.0,
                    predict_at=APNCC_DIRECT_LOCAL,
                ),
            }
        )

    loo = []
    for held_index, held in enumerate(points):
        training = [
            point for index, point in enumerate(points) if index != held_index
        ]
        ols = point_ols(training)
        prediction = (
            ols["intercept"] + ols["slope"] * held["local_cv57"]
        )
        unconstrained = affine_envelope(
            training, predict_at=float(held["local_cv57"])
        )
        monotone = affine_envelope(
            training,
            minimum_slope=0.0,
            predict_at=float(held["local_cv57"]),
        )
        for envelope in (unconstrained, monotone):
            if envelope["feasible"]:
                envelope["held_interval_miss"] = interval_miss(
                    envelope["prediction_range"],
                    held["public_interval"],
                )
                envelope["held_interval_covered"] = (
                    envelope["held_interval_miss"] == 0.0
                )
        loo.append(
            {
                "held_configuration": held["configuration"],
                "held_public_interval": held["public_interval"],
                "center_ols": {
                    **ols,
                    "prediction": prediction,
                    "residual_displayed_minus_prediction": (
                        held["public_displayed"] - prediction
                    ),
                    "interval_miss": interval_miss(
                        [prediction, prediction],
                        held["public_interval"],
                    ),
                },
                "unconstrained_training_interval_envelope": unconstrained,
                "monotone_training_interval_envelope": monotone,
            }
        )

    unconstrained_expansion = minimum_uniform_expansion(
        points, minimum_slope=None
    )
    monotone_expansion = minimum_uniform_expansion(
        points, minimum_slope=0.0
    )
    point_model = point_ols(points)
    point_predictions = [
        point_model["intercept"] + point_model["slope"] * point["local_cv57"]
        for point in points
    ]
    residuals = [
        point["public_displayed"] - prediction
        for point, prediction in zip(points, point_predictions, strict=True)
    ]
    return {
        "schema": SCHEMA,
        "status": "completed",
        "verdict": "STILL-UNRESOLVED",
        "scope": {
            "pair_count": len(points),
            "mechanisms_observed": ["graph cost/solver"],
            "observed_local_range": [
                min(point["local_cv57"] for point in points),
                max(point["local_cv57"] for point in points),
            ],
            "display_resolution": 0.001,
            "public_interval_convention": "[displayed-0.0005, displayed+0.0005)",
            "open_upper_epsilon": OPEN_UPPER_EPSILON,
            "kaggle_action": False,
            "gpu_action": False,
            "runpod_action": False,
        },
        "provenance": {
            "source": str(source_path),
            "source_pairs_sha256": hashlib.sha256(
                json.dumps(
                    points,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest(),
            "source_binding": (
                "canonical pairs only; audit metadata may be appended to the "
                "source document without changing this identity"
            ),
        },
        "descriptive_center_ols": {
            **point_model,
            "predictions": point_predictions,
            "residuals": residuals,
            "prospective_apncc_direct_prediction": (
                point_model["intercept"]
                + point_model["slope"] * APNCC_DIRECT_LOCAL
            ),
            "decision_grade": False,
        },
        "all_pair_interval_feasibility": {
            "unconstrained": affine_envelope(points),
            "monotone_nonnegative": affine_envelope(
                points, minimum_slope=0.0
            ),
            "minimum_uniform_expansion_unconstrained": (
                unconstrained_expansion
            ),
            "minimum_uniform_expansion_monotone_nonnegative": (
                monotone_expansion
            ),
        },
        "pairwise_implied_slopes": pairwise,
        "leave_one_pair_out": loo,
        "prospective_apncc_direct_local": APNCC_DIRECT_LOCAL,
        "prospective_apncc_direct_pair_envelopes": prospective,
        "identifiability": {
            "global_affine": (
                "descriptive only; no affine line intersects all three "
                "display intervals"
            ),
            "mechanism_intercept": (
                "not estimable: only one observed mechanism"
            ),
            "per_mechanism_slope": (
                "not estimable: no repeated score-separated public anchors "
                "outside graph cost/solver"
            ),
            "covariate_regression": (
                "not admissible: three pairs do not exceed the effective "
                "parameter count"
            ),
        },
        "required_next_measurement": {
            "candidate": "APNCC-direct",
            "local_cv57": APNCC_DIRECT_LOCAL,
            "reason": (
                "only observation-distinct, package-ready anchor above local "
                "0.88; it extends both score range and mechanism coverage"
            ),
            "submission_count_if_approved": 1,
            "explicit_human_approval_required": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        raise CalibrationIntervalError(
            f"refusing to overwrite output: {args.output}"
        )
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(payload, source_path=args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": report["verdict"],
                "pair_count": report["scope"]["pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
