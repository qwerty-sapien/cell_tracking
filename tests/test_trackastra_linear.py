from __future__ import annotations

import numpy as np

from celltrack_alpha.trackastra_linear import (
    challenger_labels,
    fit_ridge,
    labeled_subset,
    precision_metrics,
    proposal_features,
    propose_swaps,
    select_threshold,
)


def _payload() -> dict[str, object]:
    rows = np.zeros((4, 49), dtype=np.float32)
    rows[:, 0] = (0.0, 1.0, 2.0, 3.0)
    rows[:, 44] = (0.4, 0.2, 0.3, 0.1)
    return {
        "challengers": np.asarray(((1, 2), (3, 4), (5, 6))),
        "conflicts": np.asarray(((1, 7), (8, 2), (3, 9), (5, 10))),
        "row_features": rows,
        "proposal_offsets": np.asarray((0, 2, 3, 4)),
        "geometry_confidences": np.asarray((0.8, 0.9, 0.7)),
        "geometry_probabilities": np.asarray((0.8, 0.7, 0.9, 0.6)),
    }


def test_limiting_features_and_exact_label_join() -> None:
    payload = _payload()
    features = proposal_features(payload)
    assert features[:, 0].tolist() == [1.0, 2.0, 3.0]

    labels = challenger_labels(
        np.asarray(((1, 2), (1, 2), (5, 6))),
        np.asarray((1, 1, 0)),
    )
    joined, joined_y, indices = labeled_subset(payload, labels)
    assert indices.tolist() == [0, 2]
    assert joined[:, 0].tolist() == [1.0, 3.0]
    assert joined_y.tolist() == [1, 0]


def test_ridge_threshold_and_metrics_are_deterministic() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(200, 49))
    labels = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(np.int8)
    head = fit_ridge(
        features,
        labels,
        ridge_lambda=0.01,
        standard_deviation_floor=1.0e-6,
    )
    scores = head.score(features)
    threshold, calibration = select_threshold(
        scores,
        labels,
        minimum_accepted=40,
        minimum_precision=0.95,
    )
    observed = precision_metrics(scores, labels, threshold)
    assert calibration == observed
    assert observed["accepted_labeled"] >= 40
    assert observed["precision"] >= 0.95


def test_cached_proposals_threshold_greedily_and_cap() -> None:
    proposals, counts = propose_swaps(
        _payload(),
        np.asarray((0.8, 0.9, 0.7)),
        threshold=0.75,
        maximum=1,
    )
    assert counts == {
        "full_pool_proposals": 3,
        "threshold_proposals": 2,
        "accepted_before_cap": 2,
        "accepted_proposals": 1,
        "proposal_cap": 1,
        "cap_active": True,
        "cap_removed": 1,
    }
    assert proposals[0]["challenger"] == [3, 4]
