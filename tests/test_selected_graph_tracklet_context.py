from __future__ import annotations

import numpy as np

from scripts import audit_selected_graph_tracklet_context as tracklet
from celltrack_alpha.edge_data import SOURCE_CONFLICT, TARGET_CONFLICT


def _unit(index: int) -> np.ndarray:
    value = np.zeros(4, dtype=np.float32)
    value[index] = 1.0
    return value


def test_source_context_prefers_candidate_tracklet() -> None:
    detections = {
        1: (0, np.asarray((0.0, 0.0, 0.0))),
        2: (1, np.asarray((0.0, 1.0, 0.0))),
        3: (2, np.asarray((0.0, 2.0, 0.0))),
        4: (2, np.asarray((0.0, 5.0, 0.0))),
    }
    vectors = {1: _unit(0), 2: _unit(1), 3: _unit(0), 4: _unit(2)}
    context_ncc, motion, available = tracklet.context_features(
        challenger=(2, 3),
        winner=(2, 4),
        conflict_kind=SOURCE_CONFLICT,
        selected_in={2: 1},
        selected_out={},
        detections=detections,
        vectors=vectors,
    )
    assert available
    assert context_ncc == 1.0
    assert motion > 0.0


def test_target_context_prefers_candidate_tracklet() -> None:
    detections = {
        1: (0, np.asarray((0.0, -2.0, 0.0))),
        2: (0, np.asarray((0.0, 0.0, 0.0))),
        3: (1, np.asarray((0.0, 1.0, 0.0))),
        4: (2, np.asarray((0.0, 2.0, 0.0))),
    }
    vectors = {1: _unit(2), 2: _unit(0), 3: _unit(1), 4: _unit(0)}
    context_ncc, motion, available = tracklet.context_features(
        challenger=(2, 3),
        winner=(1, 3),
        conflict_kind=TARGET_CONFLICT,
        selected_in={},
        selected_out={3: 4},
        detections=detections,
        vectors=vectors,
    )
    assert available
    assert context_ncc == 1.0
    assert motion > 0.0


def test_missing_context_is_neutral() -> None:
    detections = {
        2: (1, np.asarray((0.0, 1.0, 0.0))),
        3: (2, np.asarray((0.0, 2.0, 0.0))),
        4: (2, np.asarray((0.0, 5.0, 0.0))),
    }
    vectors = {2: _unit(1), 3: _unit(0), 4: _unit(2)}
    assert tracklet.context_features(
        challenger=(2, 3),
        winner=(2, 4),
        conflict_kind=SOURCE_CONFLICT,
        selected_in={},
        selected_out={},
        detections=detections,
        vectors=vectors,
    ) == (0.0, 0.0, False)
