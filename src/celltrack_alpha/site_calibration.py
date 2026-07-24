"""Small, auditable corrections for local-to-Kaggle score calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CalibrationModel:
    name: str
    intercept: float
    slope: float

    def predict(self, local_score: float) -> float:
        return self.intercept + self.slope * local_score


@dataclass(frozen=True)
class BinaryProbabilityCalibration:
    """A monotone calibration map from a binary classifier logit."""

    name: str
    intercept: float
    slope: float
    calibration_metrics_before: dict[str, float]
    calibration_metrics_after: dict[str, float]

    def predict_logits(self, logits: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(logits), dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("calibration logits must be a finite vector")
        if not np.isfinite(self.intercept) or not 0.0 < self.slope:
            raise ValueError("binary calibration must be monotone increasing")
        return self.intercept + self.slope * values

    def predict_probabilities(self, logits: Iterable[float]) -> np.ndarray:
        calibrated = self.predict_logits(logits)
        return _sigmoid(calibrated)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _binary_arrays(
    logits: Iterable[float], labels: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(logits), dtype=np.float64)
    targets = np.asarray(list(labels), dtype=np.int8)
    if (
        values.ndim != 1
        or targets.shape != values.shape
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(~np.isin(targets, (0, 1)))
        or not np.any(targets == 0)
        or not np.any(targets == 1)
    ):
        raise ValueError("binary calibration requires finite logits and both classes")
    return values, targets.astype(np.float64)


def binary_calibration_metrics(
    logits: Iterable[float], labels: Iterable[int]
) -> dict[str, float]:
    """Return deterministic NLL and Brier diagnostics."""

    values, targets = _binary_arrays(logits, labels)
    probabilities = _sigmoid(values)
    return {
        "negative_log_likelihood": float(
            np.mean(np.logaddexp(0.0, values) - targets * values)
        ),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
    }


def fit_temperature_calibration(
    logits: Iterable[float], labels: Iterable[int]
) -> BinaryProbabilityCalibration:
    """Fit one positive temperature on a disjoint binary calibration set."""

    values, targets = _binary_arrays(logits, labels)

    def objective(parameters: np.ndarray) -> float:
        temperature = float(np.exp(parameters[0]))
        calibrated = values / temperature
        return float(
            np.mean(
                np.logaddexp(0.0, calibrated) - targets * calibrated
            )
        )

    result = minimize(
        objective,
        np.asarray((0.0,), dtype=np.float64),
        method="L-BFGS-B",
        bounds=((np.log(0.05), np.log(20.0)),),
        options={"ftol": 1.0e-12, "gtol": 1.0e-10, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"temperature calibration failed: {result.message}")
    temperature = float(np.exp(result.x[0]))
    slope = 1.0 / temperature
    calibrated = slope * values
    return BinaryProbabilityCalibration(
        name="temperature",
        intercept=0.0,
        slope=slope,
        calibration_metrics_before=binary_calibration_metrics(values, targets),
        calibration_metrics_after=binary_calibration_metrics(
            calibrated, targets
        ),
    )


def fit_platt_calibration(
    logits: Iterable[float], labels: Iterable[int]
) -> BinaryProbabilityCalibration:
    """Fit a monotone two-parameter Platt map on a disjoint calibration set."""

    values, targets = _binary_arrays(logits, labels)

    def objective(parameters: np.ndarray) -> float:
        intercept = float(parameters[0])
        log_slope = float(parameters[1])
        slope = float(np.exp(log_slope))
        calibrated = intercept + slope * values
        # A negligible ridge prevents an unbounded optimum on separated small data.
        return float(
            np.mean(
                np.logaddexp(0.0, calibrated) - targets * calibrated
            )
            + 1.0e-6 * (intercept * intercept + log_slope * log_slope)
        )

    result = minimize(
        objective,
        np.asarray((0.0, 0.0), dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-20.0, 20.0), (np.log(1.0e-3), np.log(1.0e3))),
        options={"ftol": 1.0e-12, "gtol": 1.0e-10, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Platt calibration failed: {result.message}")
    intercept = float(result.x[0])
    slope = float(np.exp(result.x[1]))
    calibrated = intercept + slope * values
    return BinaryProbabilityCalibration(
        name="platt",
        intercept=intercept,
        slope=slope,
        calibration_metrics_before=binary_calibration_metrics(values, targets),
        calibration_metrics_after=binary_calibration_metrics(
            calibrated, targets
        ),
    )


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
