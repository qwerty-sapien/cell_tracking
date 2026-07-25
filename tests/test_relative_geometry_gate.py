"""Focused tests for the competitor-relative abstaining swap gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from celltrack_alpha.relative_geometry_gate import (
    RELATIVE_FEATURE_COUNT,
    RelativeGeometryGate,
    RelativeGeometryGateError,
    choose_swap,
    relative_features,
)
from scripts.fit_relative_geometry_swap_gate import source_examples


def _features(count: int = 2) -> np.ndarray:
    return np.arange(count * 16, dtype=np.float32).reshape(count, 16) / 10.0


def _geometry(count: int = 2) -> np.ndarray:
    values = np.zeros((count, 6), dtype=np.float32)
    values[:, 3] = np.arange(1, count + 1)
    return values


def test_relative_features_are_directional_and_include_no_link_margins() -> None:
    challenger = _features()
    winner = challenger[::-1].copy()
    values = relative_features(
        challenger,
        winner,
        _geometry(),
        _geometry()[::-1].copy(),
        challenger_logit=np.asarray([2.0, 1.0], dtype=np.float32),
        winner_logit=np.asarray([1.5, 1.5], dtype=np.float32),
        dummy_logit=0.5,
        candidate_count=np.asarray([2, 3]),
    )
    reverse = relative_features(
        winner,
        challenger,
        _geometry()[::-1].copy(),
        _geometry(),
        challenger_logit=np.asarray([1.5, 1.5], dtype=np.float32),
        winner_logit=np.asarray([2.0, 1.0], dtype=np.float32),
        dummy_logit=0.5,
        candidate_count=np.asarray([2, 3]),
    )
    assert values.shape == (2, RELATIVE_FEATURE_COUNT)
    np.testing.assert_allclose(values[:, :16], -reverse[:, :16])
    np.testing.assert_allclose(values[:, 39], np.asarray([0.5, -0.5]))
    np.testing.assert_allclose(values[:, 40], np.asarray([1.5, 0.5]))
    np.testing.assert_allclose(values[:, 41], np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(values[:, 43], 0.0)


def test_choose_swap_is_single_and_fail_closed_at_threshold() -> None:
    assert choose_swap(np.asarray([4, 7]), np.asarray([0.4, 0.8]), threshold=0.9).challenger_row is None
    decision = choose_swap(np.asarray([4, 7]), np.asarray([0.4, 0.8]), threshold=0.75)
    assert decision.challenger_row == 7
    assert decision.probability == pytest.approx(0.8)


def test_gate_rejects_wrong_width_and_returns_one_logit_per_pair() -> None:
    model = RelativeGeometryGate(hidden=8)
    output = model(torch.zeros((3, RELATIVE_FEATURE_COUNT)))
    assert output.shape == (3,)
    with pytest.raises(RelativeGeometryGateError, match="wrong shape"):
        model(torch.zeros((3, RELATIVE_FEATURE_COUNT - 1)))


def test_source_examples_include_rescue_and_correct_winner_controls() -> None:
    record = {
        "features": np.vstack((_features(3), _features(3) + 1.0)),
        "base_logits": np.asarray([3.0, 2.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float32),
        "geometry_um": np.vstack((_geometry(3), _geometry(3) + 1.0)),
        "source_index": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        "supervised_sources": np.asarray([0, 1], dtype=np.int64),
        "positive_rows": np.asarray([1, 3], dtype=np.int64),
    }
    groups, counts = source_examples(record, dummy_logit=0.0)
    assert counts["wrong_winner_rescues"] == 1
    assert counts["correct_winner_controls"] == 1
    assert counts["positive_challengers"] == 1
    assert len(groups) == 2
    np.testing.assert_array_equal(groups[0]["labels"], np.asarray([1.0, 0.0]))
    np.testing.assert_array_equal(groups[1]["labels"], np.asarray([0.0, 0.0]))


def test_source_examples_calibrate_edge_challenger_against_dummy_winner() -> None:
    record = {
        "features": _features(2),
        "base_logits": np.asarray([-1.0, -2.0], dtype=np.float32),
        "geometry_um": _geometry(2),
        "source_index": np.asarray([0, 0], dtype=np.int64),
        "supervised_sources": np.asarray([0], dtype=np.int64),
        "positive_rows": np.asarray([1], dtype=np.int64),
    }
    groups, counts = source_examples(record, dummy_logit=0.0)
    assert len(groups) == 1
    assert counts["dummy_winner_sources"] == 1
    np.testing.assert_array_equal(groups[0]["labels"], np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(groups[0]["features"][:, 43], 1.0)


def test_source_examples_include_true_no_link_control() -> None:
    record = {
        "features": _features(2),
        "base_logits": np.asarray([-1.0, -2.0], dtype=np.float32),
        "geometry_um": _geometry(2),
        "source_index": np.asarray([0, 0], dtype=np.int64),
        "supervised_sources": np.asarray([0], dtype=np.int64),
        "target_rows": np.asarray([-1], dtype=np.int64),
    }
    groups, counts = source_examples(record, dummy_logit=0.0)
    assert len(groups) == 1
    assert counts["correct_winner_controls"] == 1
    np.testing.assert_array_equal(groups[0]["labels"], np.asarray([0.0, 0.0]))
