from __future__ import annotations

from pathlib import Path

import pytest

from scripts import analyze_public_calibration_intervals as calibration


def point(x: float, low: float, high: float) -> dict[str, object]:
    return {
        "local_cv57": x,
        "public_interval": [low, high],
        "public_displayed": (low + high) / 2.0,
        "configuration": f"x={x}",
    }


def test_affine_envelope_recovers_feasible_parameter_ranges() -> None:
    points = [point(0.0, 0.0, 0.1), point(1.0, 1.0, 1.1)]
    result = calibration.affine_envelope(
        points, minimum_slope=0.0, predict_at=2.0
    )
    assert result["feasible"] is True
    assert result["slope_range"][0] == pytest.approx(0.9)
    assert result["slope_range"][1] == pytest.approx(1.1)
    assert result["prediction_range"][0] == pytest.approx(1.9)
    assert result["prediction_range"][1] == pytest.approx(2.2)


def test_monotone_envelope_rejects_a_strict_rank_reversal() -> None:
    points = [point(0.0, 0.9, 1.0), point(1.0, 0.8, 0.9)]
    assert calibration.affine_envelope(
        points, minimum_slope=0.0
    )["feasible"] is False
    assert calibration.affine_envelope(points)["feasible"] is True


def test_minimum_expansion_repairs_incompatible_intervals() -> None:
    points = [
        point(0.0, 0.0, 0.01),
        point(1.0, 1.0, 1.01),
        point(2.0, 3.0, 3.01),
    ]
    expansion = calibration.minimum_uniform_expansion(
        points, minimum_slope=None
    )
    assert expansion > 0.0
    assert calibration.affine_envelope(
        points, expansion=expansion + 1e-10
    )["feasible"]


def test_interval_miss_handles_overlap_and_separation() -> None:
    assert calibration.interval_miss([0.8, 0.9], [0.85, 0.95]) == 0.0
    assert calibration.interval_miss([0.7, 0.8], [0.85, 0.95]) == pytest.approx(
        0.05
    )
    assert calibration.interval_miss([1.0, 1.1], [0.85, 0.95]) == pytest.approx(
        0.05
    )


def test_frozen_report_is_unresolved_and_requests_apncc() -> None:
    source = (
        calibration.ROOT
        / "reports/g02/screening/p0_public_calibration_v1.json"
    )
    payload = calibration.json.loads(source.read_text(encoding="utf-8"))
    report = calibration.build_report(payload, source_path=source)
    assert report["verdict"] == "STILL-UNRESOLVED"
    assert (
        report["all_pair_interval_feasibility"]["unconstrained"]["feasible"]
        is False
    )
    assert report["required_next_measurement"]["candidate"] == "APNCC-direct"
    assert report["scope"]["kaggle_action"] is False


def test_appended_audit_metadata_does_not_change_pair_binding() -> None:
    source = (
        calibration.ROOT
        / "reports/g02/screening/p0_public_calibration_v1.json"
    )
    payload = calibration.json.loads(source.read_text(encoding="utf-8"))
    first = calibration.build_report(payload, source_path=source)
    payload["interval_aware_audit"] = {"report_sha256": "different"}
    second = calibration.build_report(payload, source_path=source)
    assert first == second


def test_input_identity_is_fail_closed() -> None:
    source = Path("calibration.json")
    with pytest.raises(
        calibration.CalibrationIntervalError, match="unexpected calibration"
    ):
        calibration.build_report({}, source_path=source)
