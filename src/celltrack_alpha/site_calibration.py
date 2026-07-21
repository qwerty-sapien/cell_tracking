"""Small, auditable corrections for local-to-Kaggle score calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationModel:
    name: str
    intercept: float
    slope: float

    def predict(self, local_score: float) -> float:
        return self.intercept + self.slope * local_score


def fit_bias(local_scores: Iterable[float], kaggle_scores: Iterable[float]) -> CalibrationModel:
    pairs = list(zip(local_scores, kaggle_scores, strict=True))
    if not pairs:
        raise ValueError("at least one calibration pair is required")
    bias = sum(kaggle - local for local, kaggle in pairs) / len(pairs)
    return CalibrationModel("bias", bias, 1.0)


def fit_affine(local_scores: Iterable[float], kaggle_scores: Iterable[float]) -> CalibrationModel:
    pairs = list(zip(local_scores, kaggle_scores, strict=True))
    if len(pairs) < 2:
        raise ValueError("at least two calibration pairs are required")
    mean_local = sum(local for local, _ in pairs) / len(pairs)
    mean_kaggle = sum(kaggle for _, kaggle in pairs) / len(pairs)
    variance = sum((local - mean_local) ** 2 for local, _ in pairs)
    if variance <= 1e-12:
        raise ValueError("affine calibration requires non-identical local scores")
    covariance = sum(
        (local - mean_local) * (kaggle - mean_kaggle)
        for local, kaggle in pairs
    )
    slope = covariance / variance
    return CalibrationModel("affine", mean_kaggle - slope * mean_local, slope)


def residuals(
    model: CalibrationModel,
    local_scores: Iterable[float],
    kaggle_scores: Iterable[float],
) -> list[float]:
    return [
        model.predict(local) - kaggle
        for local, kaggle in zip(local_scores, kaggle_scores, strict=True)
    ]


def select_calibration_model(
    local_scores: Iterable[float],
    kaggle_scores: Iterable[float],
    *,
    tolerance: float = 0.02,
) -> CalibrationModel:
    local = list(local_scores)
    kaggle = list(kaggle_scores)
    if len(local) != len(kaggle) or not local:
        raise ValueError("matching non-empty local and Kaggle score lists are required")
    identity = CalibrationModel("identity", 0.0, 1.0)
    if max(abs(value) for value in residuals(identity, local, kaggle)) <= tolerance:
        return identity
    bias = fit_bias(local, kaggle)
    if max(abs(value) for value in residuals(bias, local, kaggle)) <= tolerance:
        return bias
    return fit_affine(local, kaggle)
