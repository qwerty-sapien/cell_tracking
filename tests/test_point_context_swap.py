from __future__ import annotations

import numpy as np
import torch

from celltrack_alpha.point_context import PointContextTransformer
from celltrack_alpha.point_context_swap import (
    PointContextSwapError,
    RuntimeProposalWindow,
    propose_swaps,
)


def _runtime_window() -> RuntimeProposalWindow:
    first_rows = np.zeros((1, 48), dtype=np.float32)
    first_rows[0, 44] = 0.3
    second_rows = np.zeros((2, 48), dtype=np.float32)
    second_rows[:, 44] = [0.4, 0.2]
    return RuntimeProposalWindow(
        identity="train:6bba_unit",
        family=1,
        target_t=1,
        features=np.stack((first_rows[0], second_rows[1])),
        target_ids=np.asarray([10, 10], dtype=np.int64),
        challengers=((1, 10), (2, 10)),
        conflicts=(((1, 11),), ((2, 12), (3, 10))),
        row_features=(first_rows, second_rows),
        geometry_confidences=np.asarray([0.9, 0.8], dtype=np.float32),
        geometry_probabilities=(
            np.asarray([0.9], dtype=np.float32),
            np.asarray([0.8, 0.85], dtype=np.float32),
        ),
    )


def test_runtime_window_retains_lossless_graph_identities() -> None:
    window = _runtime_window()
    window.validate()
    model_window = window.model_window()
    assert model_window.target_ids.tolist() == [10, 10]
    assert model_window.labels.tolist() == [-1, -1]
    assert window.challengers[1] == (2, 10)
    assert window.conflicts[1] == ((2, 12), (3, 10))


def test_runtime_window_rejects_nonlimiting_collapsed_feature() -> None:
    window = _runtime_window()
    malformed = RuntimeProposalWindow(
        **{
            **window.__dict__,
            "features": np.stack(
                (window.features[0], window.row_features[1][0])
            ),
        }
    )
    try:
        malformed.validate()
    except PointContextSwapError as exc:
        assert "limiting conflict" in str(exc)
    else:
        raise AssertionError("nonlimiting token was accepted")


def test_propose_swaps_preserves_score_to_identity_alignment() -> None:
    torch.manual_seed(4)
    model = PointContextTransformer(
        width=16, heads=4, layers=1, dropout=0.0
    )
    accepted, counts = propose_swaps(
        model,
        [_runtime_window()],
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        threshold=0.0,
        maximum=4,
    )
    assert counts["full_pool_proposals"] == 2
    assert counts["threshold_proposals"] == 2
    # Both proposals compete for target 10, so node-disjoint greediness accepts one.
    assert len(accepted) == 1
    assert tuple(accepted[0]["challenger"]) in {(1, 10), (2, 10)}
    assert accepted[0]["conflicts"]


def test_propose_swaps_accepts_a_valid_empty_deployment_pool() -> None:
    model = PointContextTransformer(
        width=16, heads=4, layers=1, dropout=0.0
    )
    accepted, counts = propose_swaps(
        model,
        [],
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        threshold=0.5,
        maximum=3,
    )
    assert accepted == []
    assert counts["full_pool_proposals"] == 0
    assert counts["accepted_proposals"] == 0
