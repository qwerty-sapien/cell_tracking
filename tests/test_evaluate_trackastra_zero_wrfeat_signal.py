from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_trackastra_zero_wrfeat_signal import (
    TrackastraSignalError,
    average_precision,
    grouped_rows,
    roc_auc,
    select_window_nodes,
)


def test_grouped_rows_preserves_first_challenger_order() -> None:
    result = grouped_rows(
        np.asarray(((1, 2), (3, 4), (1, 2))),
        np.asarray(((1, 5), (6, 4), (7, 2))),
        np.asarray((1, 0, 1)),
    )

    assert [row["challenger"] for row in result] == [(1, 2), (3, 4)]
    assert result[0]["rows"] == (0, 2)
    assert result[0]["incumbents"] == ((1, 5), (7, 2))


def test_grouped_rows_rejects_contradictory_labels() -> None:
    with pytest.raises(TrackastraSignalError, match="contradictory"):
        grouped_rows(
            np.asarray(((1, 2), (1, 2))),
            np.asarray(((1, 5), (7, 2))),
            np.asarray((1, 0)),
        )


def test_select_window_nodes_crops_deterministically() -> None:
    by_node = {
        node: (
            node // 3,
            np.asarray((0.0, float(node), 0.0), dtype=np.float32),
        )
        for node in range(12)
    }
    frame_nodes = {
        frame: tuple(range(frame * 3, frame * 3 + 3))
        for frame in range(4)
    }
    nodes, cropped = select_window_nodes(
        by_node=by_node,
        frame_nodes=frame_nodes,
        source_frame=1,
        required={4, 7},
        window_frames=4,
        max_tokens=5,
    )

    assert cropped is True
    assert set((4, 7)) <= set(nodes)
    assert len(nodes) == 5
    assert nodes == tuple(sorted(nodes, key=lambda node: (node // 3, node)))


def test_rank_metrics_handle_ties() -> None:
    labels = np.asarray((0, 0, 1, 1))
    assert roc_auc(labels, np.asarray((0.0, 0.0, 1.0, 1.0))) == 1.0
    assert average_precision(
        labels, np.asarray((0.0, 0.0, 1.0, 1.0))
    ) == 1.0
