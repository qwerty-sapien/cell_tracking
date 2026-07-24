from __future__ import annotations

import numpy as np

from scripts.audit_trackastra_full_pool_feasibility import (
    transition_coverage,
)


def _fixture():
    by_node = {
        node: (
            node // 4,
            np.asarray((0.0, float(node), 0.0), dtype=np.float32),
        )
        for node in range(16)
    }
    frame_nodes = {
        frame: tuple(range(frame * 4, frame * 4 + 4))
        for frame in range(4)
    }
    return by_node, frame_nodes


def test_transition_coverage_retains_all_endpoints_when_cropped() -> None:
    by_node, frame_nodes = _fixture()
    result = transition_coverage(
        groups=(
            {"challenger": (4, 8), "conflicts": ((4, 9),)},
            {"challenger": (5, 10), "conflicts": ((6, 10),)},
        ),
        by_node=by_node,
        frame_nodes=frame_nodes,
        window_frames=4,
        max_tokens=8,
    )

    assert len(result) == 1
    assert result[0]["proposal_count"] == 2
    assert result[0]["required_node_count"] == 6
    assert result[0]["selected_token_count"] == 8
    assert result[0]["cropped"] is True
    assert result[0]["missing_endpoint_count"] == 0


def test_transition_coverage_reports_required_node_overflow() -> None:
    by_node, frame_nodes = _fixture()
    result = transition_coverage(
        groups=(
            {
                "challenger": (4, 8),
                "conflicts": ((5, 9), (6, 10)),
            },
        ),
        by_node=by_node,
        frame_nodes=frame_nodes,
        window_frames=4,
        max_tokens=5,
    )

    assert result[0]["required_node_count"] == 6
    assert result[0]["token_cap_pass"] is False
    assert result[0]["selected_token_count"] == 0
    assert result[0]["missing_endpoint_count"] == 6
