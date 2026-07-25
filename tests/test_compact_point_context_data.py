from __future__ import annotations

import numpy as np

from scripts import audit_compact_point_context_data as audit


def test_group_labeled_preserves_pair_order_and_rejects_mixed_labels() -> None:
    pairs, labels, groups = audit._group_labeled(
        np.asarray([[1, 3], [1, 3], [2, 4]], dtype=np.int64),
        np.asarray([1, 1, 0], dtype=np.int8),
    )
    assert pairs.tolist() == [[1, 3], [2, 4]]
    assert labels.tolist() == [1, 0]
    assert groups == [[[0], [1]], [[2]]]
    empty_pairs, empty_labels, empty_groups = audit._group_labeled(
        np.empty((0, 2), dtype=np.int64),
        np.empty(0, dtype=np.int8),
    )
    assert empty_pairs.shape == (0, 2)
    assert empty_labels.shape == (0,)
    assert empty_groups == []


def test_window_stats_uses_conservative_cap_and_parental_groups() -> None:
    windows, totals = audit._window_stats(
        np.asarray([[1, 3], [2, 3], [1, 4]], dtype=np.int64),
        np.asarray([1, -1, 0], dtype=np.int8),
        np.asarray([0.8, 0.9, 0.7], dtype=np.float64),
        {1: 0, 2: 0, 3: 1, 4: 1},
        token_cap=2,
    )
    assert windows == [
        {
            "target_t": 1,
            "tokens": 3,
            "positive_tokens": 1,
            "positive_tokens_retained": 1,
            "over_cap": True,
        }
    ]
    assert totals["parental_groups"] == 2
    assert totals["no_parent_groups"] == 1
    assert totals["maximum_parent_candidates"] == 2
    assert totals["positive_tokens_retained"] == 1
