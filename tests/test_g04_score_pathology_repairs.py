from __future__ import annotations

import numpy as np

from celltrack_alpha.bh_model import FrameDet
from scripts import g04_score_pathology_repairs as repair


def frame(frame_index: int, node_ids, coords) -> FrameDet:
    coords = np.asarray(coords, dtype=np.int64)
    return FrameDet(
        frame_index=frame_index,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        coords_zyx=coords,
        scores=np.full(len(coords), 0.9, dtype=np.float32),
        grid_coords_zyx=np.minimum(coords, 63),
        grid_shape_zyx=(64, 64, 64),
    )


def fragment_rows():
    rows = []
    for node_id, t, coord in (
        (1, 0, (2, 3, 4)),
        (2, 0, (5, 6, 7)),
        (3, 1, (2, 3, 4)),
        (4, 1, (5, 6, 7)),
    ):
        rows.append(
            {
                "id": str(len(rows)),
                "dataset": "fixture",
                "row_type": "node",
                "node_id": str(node_id),
                "t": str(t),
                "z": str(coord[0]),
                "y": str(coord[1]),
                "x": str(coord[2]),
                "source_id": "-1",
                "target_id": "-1",
            }
        )
    for source, target in ((1, 4), (2, 3)):
        rows.append(
            {
                "id": str(len(rows)),
                "dataset": "fixture",
                "row_type": "edge",
                "node_id": "-1",
                "t": "-1",
                "z": "-1",
                "y": "-1",
                "x": "-1",
                "source_id": str(source),
                "target_id": str(target),
            }
        )
    return rows


def test_frozen_repair_replaces_conflicts_with_exact_pairs() -> None:
    rows, counts = repair.apply_frozen(
        fragment_rows(), {0: {(1, 3), (2, 4)}}
    )
    assert repair.edge_pairs(rows) == {(1, 3), (2, 4)}
    assert counts == {
        "installed_pairs": 2,
        "retained_existing_pairs": 0,
        "novel_pairs": 2,
        "removed_conflicting_edges": 2,
        "skipped_pruned_pairs": 0,
    }


def test_cumulative_registration_applies_only_pool_breaking_steps() -> None:
    detections = (
        frame(0, [1], [[3, 4, 5]]),
        frame(1, [2], [[3, 4, 5]]),
        frame(2, [3], [[3, 4, 5]]),
    )
    transitions = [
        {
            "frame": 0,
            "pool_breaking": True,
            "shift_registered_zyx": [-2, 3, -4],
        },
        {
            "frame": 1,
            "pool_breaking": False,
            "shift_registered_zyx": [9, 9, 9],
        },
    ]
    transforms, offset = repair.cumulative_frame_transforms(
        transitions, detections
    )
    np.testing.assert_array_equal(transforms[0], [0, 0, 0])
    np.testing.assert_array_equal(transforms[1], [-2, 12, -16])
    np.testing.assert_array_equal(transforms[2], [-2, 12, -16])
    np.testing.assert_array_equal(offset, [0, 0, 11])


def test_registered_repair_changes_only_raw_edges_and_conflicts() -> None:
    rows, counts = repair.apply_registered(
        fragment_rows(), {0: {(1, 3), (2, 4)}}, {1, 2, 3, 4}
    )
    assert repair.edge_pairs(rows) == {(1, 3), (2, 4)}
    assert counts == {
        "removed_raw_edges": 2,
        "added_registered_edges": 2,
        "removed_conflicting_synthetic_edges": 0,
        "skipped_pruned_pairs": 0,
    }


def test_gt_audit_keeps_incompatible_annotations_but_counts_them() -> None:
    gt = {
        "nodes": {
            "node_id": np.asarray([10, 11, 12]),
            "t": np.asarray([0, 1, 1]),
            "z": np.asarray([0.0, 0.0, 0.0]),
            "y": np.asarray([0.0, 0.0, 28.0]),
            "x": np.asarray([0.0, 0.0, 0.0]),
        },
        "edges": np.asarray([[10, 11], [10, 12]], dtype=np.int64),
    }
    observed = repair.gt_transition_audit(
        gt=gt,
        transitions=[
            {
                "frame": 0,
                "shift_um_zyx": [0.0, -11.375, 0.0],
            }
        ],
        frozen={0},
        flagged={0},
    )
    assert observed == {
        "frozen_gt_edges": 2,
        "frozen_gt_edges_exact_zero": 1,
        "frozen_gt_edges_nonzero": 1,
        "frozen_gt_edges_over_10um": 1,
        "jump_gt_edges": 2,
        "jump_gt_edges_raw_over_10um": 1,
        "jump_gt_edges_corrected_inside_10um": 1,
        "jump_gt_edges_rescued_inside_10um": 1,
    }
