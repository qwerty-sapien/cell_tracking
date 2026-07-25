from __future__ import annotations

import numpy as np
import pytest

from scripts import fit_selected_graph_scenario_calibration_gate as gate


def test_contract_is_hash_locked_and_forbids_family_routing() -> None:
    contract = gate.load_contract(
        gate.DEFAULT_CONTRACT, gate.DEFAULT_CONTRACT_SHA256
    )
    assert contract["scope"]["cv57_access"] is False
    assert contract["promotion"]["full_cv57_authorized"] is False
    assert (
        "dataset family prefix"
        in contract["mechanism"]["forbidden_router_inputs"]
    )


def test_raw_transform_is_exact_and_preserves_ties() -> None:
    labeled, background, summary = gate.transform_video_scores(
        np.asarray([0.2, 0.8, 0.8]),
        np.asarray([0.1, 0.8]),
        budget=16,
        method="raw",
        density_penalty=0.0,
    )
    np.testing.assert_array_equal(labeled, [0.2, 0.8, 0.8])
    np.testing.assert_array_equal(background, [0.1, 0.8])
    assert labeled[1] == background[1]
    assert summary["proposal_count"] == 5


@pytest.mark.parametrize("method", ["robust50_90", "robust75_95"])
def test_robust_transform_is_affine_invariant(method: str) -> None:
    first = gate.transform_video_scores(
        np.asarray([0.2, 0.4, 0.7]),
        np.asarray([0.1, 0.5]),
        budget=16,
        method=method,
        density_penalty=0.0,
    )
    second = gate.transform_video_scores(
        np.asarray([12.0, 14.0, 17.0]),
        np.asarray([11.0, 15.0]),
        budget=16,
        method=method,
        density_penalty=0.0,
    )
    np.testing.assert_allclose(first[0], second[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(first[1], second[1], rtol=0.0, atol=1e-12)


def test_empirical_cdf_preserves_equal_score_competition() -> None:
    labeled, background, _summary = gate.transform_video_scores(
        np.asarray([0.8, 0.4]),
        np.asarray([0.8, 0.2]),
        budget=16,
        method="empirical_cdf",
        density_penalty=0.0,
    )
    assert labeled[0] == background[0]
    assert labeled[0] > labeled[1] > background[1]


def test_density_penalty_lowers_only_the_more_competitive_video() -> None:
    sparse, _, sparse_summary = gate.transform_video_scores(
        np.asarray([0.9, 0.8]),
        np.asarray([0.7]),
        budget=16,
        method="robust50_90",
        density_penalty=0.5,
    )
    dense, _, dense_summary = gate.transform_video_scores(
        np.asarray([0.9, 0.8]),
        np.tile(np.asarray([0.9, 0.8, 0.7, 0.6]), 25),
        budget=16,
        method="robust50_90",
        density_penalty=0.5,
    )
    assert dense_summary["competition"] > sparse_summary["competition"]
    assert dense[0] < sparse[0]


def test_cap_rank_uses_budget_and_keeps_ties() -> None:
    labeled, background, _summary = gate.transform_video_scores(
        np.asarray([0.9, 0.7, 0.7]),
        np.asarray([0.8, 0.7]),
        budget=2,
        method="cap_rank",
        density_penalty=0.0,
    )
    assert labeled[0] == 0.0
    assert background[0] == -0.5
    assert labeled[1] == labeled[2] == background[1]


def test_background_only_and_empty_videos_are_valid() -> None:
    labeled, background, summary = gate.transform_video_scores(
        np.empty(0),
        np.asarray([0.2, 0.5]),
        budget=16,
        method="empirical_cdf",
        density_penalty=0.0,
    )
    assert len(labeled) == 0
    assert background.tolist() == [0.0, 1.0]
    assert summary["proposal_count"] == 2

    labeled, background, summary = gate.transform_video_scores(
        np.empty(0),
        np.empty(0),
        budget=16,
        method="robust50_90",
        density_penalty=0.5,
    )
    assert len(labeled) == len(background) == 0
    assert summary["proposal_count"] == 0
    assert summary["competition"] == 0.0


def test_invalid_distribution_fails_closed() -> None:
    with pytest.raises(gate.ScenarioCalibrationError):
        gate.transform_video_scores(
            np.asarray([np.nan]),
            np.empty(0),
            budget=16,
            method="raw",
            density_penalty=0.0,
        )
