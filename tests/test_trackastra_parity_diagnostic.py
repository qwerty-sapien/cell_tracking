from __future__ import annotations

import numpy as np
import pytest

from scripts.trackastra_parity_diagnostic import (
    DiagnosticError,
    PARITY_IDENTITY,
    assert_authorized_scope,
    classify_column,
    compute_columnwise_stats,
)


def test_columnwise_statistics_use_locked_formulas() -> None:
    m5 = np.asarray(
        (
            (1.0, 0.0, -2.0),
            (2.0, 0.0, 4.0),
            (3.0, 0.0, -8.0),
            (4.0, 0.0, 16.0),
        ),
        dtype=np.float32,
    )
    t4 = m5.copy()
    t4[:, 0] += np.asarray((0.0, 1.0e-6, -2.0e-6, 4.0e-6))
    t4[:, 1] += np.asarray((1.0e-8, -2.0e-8, 3.0e-8, -4.0e-8))
    t4[:, 2] += np.asarray((1.0e-5, -2.0e-5, 3.0e-5, -4.0e-5))

    result = compute_columnwise_stats(m5, t4)
    abs_diff = np.abs(m5 - t4)

    np.testing.assert_allclose(result["max"], abs_diff.max(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(
        result["mean"], abs_diff.mean(axis=0), rtol=0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        result["p99"],
        np.quantile(abs_diff, 0.99, axis=0),
        rtol=0,
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(
        result["scale"], np.max(np.abs(m5.astype(np.float64)), axis=0)
    )
    assert result["allclose"] == np.allclose(
        m5, t4, rtol=1.0e-5, atol=1.0e-7, equal_nan=True
    )


def test_classifier_separates_benign_fp_from_structural_drift() -> None:
    reference = np.asarray((0.25, 0.5, 1.0, 2.0), dtype=np.float32)
    benign = np.nextafter(reference, np.float32(np.inf))
    classification, reason = classify_column(reference, benign)
    assert classification == "BENIGN_FP"
    assert reason in {
        "passes initial FP probe",
        "bounded by float reduction envelope",
    }

    offset = (reference + np.float32(0.01)).astype(np.float32)
    classification, reason = classify_column(reference, offset)
    assert classification == "STRUCTURAL"
    assert reason == "whole-column offset"

    classification, reason = classify_column(
        reference, reference.astype(np.float64)
    )
    assert classification == "STRUCTURAL"
    assert reason == "dtype mismatch"

    with_nan = reference.copy()
    with_nan[0] = np.nan
    classification, reason = classify_column(reference, with_nan)
    assert classification == "STRUCTURAL"
    assert reason == "NaN present"


@pytest.mark.parametrize(
    "value",
    (
        "train:44b6_edf14583",
        "/kaggle/input/holdout/6bba_edf14583.zarr",
        "reports/g02/CV57/features.npy",
        "validation12-cache",
    ),
)
def test_scope_guard_fails_closed_on_44b6_or_holdout_paths(value: str) -> None:
    if value.startswith("train:"):
        with pytest.raises(DiagnosticError, match="restricted"):
            assert_authorized_scope(value)
    else:
        with pytest.raises(DiagnosticError, match="forbidden"):
            assert_authorized_scope(PARITY_IDENTITY, value)


def test_scope_guard_accepts_only_the_parity_video() -> None:
    assert_authorized_scope(
        PARITY_IDENTITY,
        "/kaggle/input/biohub/train/6bba_edf14583.zarr",
    )
