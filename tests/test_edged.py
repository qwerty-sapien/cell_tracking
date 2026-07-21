"""Sparse metric-aligned node matching for edge calibration examples."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.bh_model import FrameDet
from celltrack_alpha.edge_data import match_detections_to_gt


def _frame(frame: int, ids: list[int], coords: list[tuple[int, int, int]]) -> FrameDet:
    count = len(ids)
    return FrameDet(
        frame_index=frame,
        node_ids=np.asarray(ids, dtype=np.int64),
        coords_zyx=np.asarray(coords, dtype=np.int64),
        scores=np.ones(count, dtype=np.float32),
        grid_coords_zyx=np.asarray(coords, dtype=np.int64),
        grid_shape_zyx=(20, 40, 40),
    )


class SparseMetricMatchTests(unittest.TestCase):
    def test_physical_radius_and_sparse_cardinality_matching(self) -> None:
        detections = (
            _frame(0, [1, 2], [(0, 10, 10), (0, 12, 10)]),
            _frame(1, [3], [(3, 10, 10)]),
        )
        gt = {
            "nodes": {
                "node_id": np.asarray([101, 102, 103]),
                "t": np.asarray([0, 0, 1]),
                "z": np.asarray([0, 0, 3]),
                "y": np.asarray([10, 11, 10]),
                "x": np.asarray([10, 10, 10]),
            },
            "edges": np.asarray([[101, 103]], dtype=np.int64),
        }
        matches, edges = match_detections_to_gt(detections, gt)
        self.assertEqual(matches, {1: 101, 2: 102, 3: 103})
        self.assertEqual(edges, {(101, 103)})

    def test_outside_seven_um_is_unmatched(self) -> None:
        detections = (_frame(0, [1], [(5, 10, 10)]),)
        gt = {
            "nodes": {
                "node_id": np.asarray([101]),
                "t": np.asarray([0]),
                "z": np.asarray([0]),
                "y": np.asarray([10]),
                "x": np.asarray([10]),
            },
            "edges": np.empty((0, 2), dtype=np.int64),
        }
        matches, _edges = match_detections_to_gt(detections, gt)
        self.assertEqual(matches, {1: None})
