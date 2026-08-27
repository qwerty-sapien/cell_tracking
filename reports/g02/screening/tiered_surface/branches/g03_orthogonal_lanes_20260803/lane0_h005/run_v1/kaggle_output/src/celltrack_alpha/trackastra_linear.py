"""Deterministic linear head over frozen Trackastra/APNCC conflict features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import appearance_swap


class TrackastraLinearError(ValueError):
    """A frozen-feature linear-head identity or metric contract failed."""


@dataclass(frozen=True)
class LinearHead:
    """A standardized ridge least-squares binary ranker."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    ridge_lambda: float

    def validate(self, feature_count: int = 49) -> None:
        if (
            self.mean.shape != (feature_count,)
            or self.scale.shape != (feature_count,)
            or self.weights.shape != (feature_count,)
            or np.any(self.scale <= 0.0)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.intercept)
            or not np.isfinite(self.ridge_lambda)
            or self.ridge_lambda <= 0.0
        ):
            raise TrackastraLinearError("linear head is malformed")

    def score(self, features: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(features, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.mean)
            or not np.isfinite(values).all()
        ):
            raise TrackastraLinearError("score features are malformed")
        return (
            ((values - self.mean) / self.scale) @ self.weights
            + self.intercept
        )


def proposal_features(payload: Mapping[str, Any]) -> np.ndarray:
    """Return the pre-registered limiting 49-D row for every proposal."""

    rows = np.asarray(payload.get("row_features"), dtype=np.float64)
    offsets = np.asarray(payload.get("proposal_offsets"), dtype=np.int64)
    challengers = np.asarray(payload.get("challengers"), dtype=np.int64)
    if (
        rows.ndim != 2
        or rows.shape[1] != 49
        or challengers.ndim != 2
        or challengers.shape[1] != 2
        or offsets.shape != (len(challengers) + 1,)
        or len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != len(rows)
        or np.any(np.diff(offsets) <= 0)
        or not np.isfinite(rows).all()
    ):
        raise TrackastraLinearError("cached proposal rows are malformed")
    limiting = np.empty((len(challengers), 49), dtype=np.float64)
    for index, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        choices = range(int(start), int(end))
        row = min(choices, key=lambda value: (float(rows[value, 44]), value))
        limiting[index] = rows[row]
    return limiting


def challenger_labels(
    pairs: np.ndarray, labels: np.ndarray
) -> dict[tuple[int, int], int]:
    """Collapse prepared row labels to unanimous challenger labels."""

    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        pairs.shape != (len(labels), 2)
        or np.any(~np.isin(labels, (0, 1)))
    ):
        raise TrackastraLinearError("prepared labels are malformed")
    grouped: dict[tuple[int, int], set[int]] = {}
    for pair, label in zip(pairs, labels, strict=True):
        key = (int(pair[0]), int(pair[1]))
        grouped.setdefault(key, set()).add(int(label))
    if any(len(values) != 1 for values in grouped.values()):
        raise TrackastraLinearError(
            "one challenger has inconsistent prepared labels"
        )
    return {key: next(iter(values)) for key, values in grouped.items()}


def labeled_subset(
    payload: Mapping[str, Any],
    known: Mapping[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join full-pool features to fit-only labels by exact challenger pair."""

    challengers = np.asarray(payload.get("challengers"), dtype=np.int64)
    features = proposal_features(payload)
    if challengers.shape != (len(features), 2):
        raise TrackastraLinearError("challenger feature coverage disagrees")
    indices = np.asarray(
        [
            index
            for index, pair in enumerate(challengers)
            if (int(pair[0]), int(pair[1])) in known
        ],
        dtype=np.int64,
    )
    if len(indices) == 0:
        raise TrackastraLinearError("feature cache has no prepared labels")
    labels = np.asarray(
        [
            known[
                (
                    int(challengers[index, 0]),
                    int(challengers[index, 1]),
                )
            ]
            for index in indices
        ],
        dtype=np.int8,
    )
    return features[indices], labels, indices


def fit_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float,
    standard_deviation_floor: float,
) -> LinearHead:
    """Fit the fixed closed-form standardized ridge head."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] != 49
        or y.shape != (len(x),)
        or len(x) < 50
        or not np.isfinite(x).all()
        or np.any(~np.isin(y, (0.0, 1.0)))
        or len(np.unique(y)) != 2
        or not np.isfinite(ridge_lambda)
        or ridge_lambda <= 0.0
        or not np.isfinite(standard_deviation_floor)
        or standard_deviation_floor <= 0.0
    ):
        raise TrackastraLinearError("ridge fitting inputs are malformed")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < standard_deviation_floor] = 1.0
    z = (x - mean) / scale
    design = np.concatenate(
        (z, np.ones((len(z), 1), dtype=np.float64)), axis=1
    )
    regularization = np.eye(design.shape[1], dtype=np.float64)
    regularization[-1, -1] = 0.0
    lhs = design.T @ design / len(design)
    lhs += ridge_lambda * regularization
    rhs = design.T @ y / len(design)
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise TrackastraLinearError(
            "ridge normal equations are singular"
        ) from exc
    head = LinearHead(
        mean=mean,
        scale=scale,
        weights=solution[:-1],
        intercept=float(solution[-1]),
        ridge_lambda=float(ridge_lambda),
    )
    head.validate()
    return head


def precision_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        scores.shape != labels.shape
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or np.any(~np.isin(labels, (0, 1)))
        or not np.isfinite(threshold)
    ):
        raise TrackastraLinearError("precision metric inputs are malformed")
    accepted = scores >= threshold
    count = int(accepted.sum())
    correct = int(labels[accepted].sum())
    harmful = count - correct
    return {
        "eligible_labeled": len(labels),
        "accepted_labeled": count,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / count if count else 0.0,
        "threshold": float(threshold),
    }


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    minimum_accepted: int,
    minimum_precision: float,
) -> tuple[float, dict[str, Any]]:
    """Select maximum support under the fixed calibration precision floor."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if (
        minimum_accepted < 1
        or not 0.0 < minimum_precision <= 1.0
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
    ):
        raise TrackastraLinearError(
            "threshold-selection inputs are malformed"
        )
    admissible = []
    for threshold in np.unique(scores)[::-1]:
        metrics = precision_metrics(scores, labels, float(threshold))
        if (
            metrics["accepted_labeled"] >= minimum_accepted
            and metrics["precision"] >= minimum_precision
        ):
            admissible.append((metrics["accepted_labeled"], threshold, metrics))
    if not admissible:
        raise TrackastraLinearError(
            "no calibration threshold satisfies the locked gate"
        )
    _support, threshold, metrics = max(
        admissible, key=lambda value: (value[0], value[1])
    )
    return float(threshold), metrics


def aggregate_precision(
    values: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible = sum(int(value["eligible_labeled"]) for value in values)
    accepted = sum(int(value["accepted_labeled"]) for value in values)
    correct = sum(int(value["correct_rescues"]) for value in values)
    harmful = sum(int(value["harmful_swaps"]) for value in values)
    return {
        "eligible_labeled": eligible,
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "precision": correct / accepted if accepted else 0.0,
    }


def propose_swaps(
    payload: Mapping[str, Any],
    scores: np.ndarray,
    *,
    threshold: float,
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Threshold and greedily realize cached proposals without label access."""

    challengers = np.asarray(payload.get("challengers"), dtype=np.int64)
    conflicts = np.asarray(payload.get("conflicts"), dtype=np.int64)
    offsets = np.asarray(payload.get("proposal_offsets"), dtype=np.int64)
    rows = np.asarray(payload.get("row_features"), dtype=np.float64)
    geometry = np.asarray(
        payload.get("geometry_confidences"), dtype=np.float64
    )
    row_probabilities = np.asarray(
        payload.get("geometry_probabilities"), dtype=np.float64
    )
    scores = np.asarray(scores, dtype=np.float64)
    if (
        challengers.shape != (len(scores), 2)
        or conflicts.shape != (len(rows), 2)
        or offsets.shape != (len(scores) + 1,)
        or geometry.shape != (len(scores),)
        or row_probabilities.shape != (len(rows),)
        or not np.isfinite(scores).all()
        or not np.isfinite(threshold)
        or type(maximum) is not int
        or maximum < 1
    ):
        raise TrackastraLinearError("cached deployment arrays are malformed")
    raw = []
    for index, confidence in enumerate(scores):
        if confidence < threshold:
            continue
        start, end = int(offsets[index]), int(offsets[index + 1])
        raw.append(
            {
                "challenger": challengers[index].tolist(),
                "conflicts": conflicts[start:end].tolist(),
                "confidence": float(confidence),
                "appearance_margins": rows[start:end, 44].tolist(),
                "geometry_confidence": float(geometry[index]),
                "row_probabilities": row_probabilities[start:end].tolist(),
            }
        )
    accepted_before_cap = appearance_swap._greedy_nonconflicting(raw)
    accepted = accepted_before_cap[:maximum]
    return accepted, {
        "full_pool_proposals": len(scores),
        "threshold_proposals": len(raw),
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": True,
        "cap_removed": len(accepted_before_cap) - len(accepted),
    }


__all__ = [
    "LinearHead",
    "TrackastraLinearError",
    "aggregate_precision",
    "challenger_labels",
    "fit_ridge",
    "labeled_subset",
    "precision_metrics",
    "proposal_features",
    "propose_swaps",
    "select_threshold",
]
