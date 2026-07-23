from __future__ import annotations

import importlib.util
import json
import math
import numpy as np
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_bh_graph_weights.py"
SPEC = importlib.util.spec_from_file_location("sweep_bh_graph_weights", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SweepTopologyContractTests(unittest.TestCase):
    def test_legacy_setting_name_is_backward_compatible(self) -> None:
        self.assertEqual(
            MODULE._setting_name(1.0, 0.05, "legacy"),
            "learned_bonus_1__raw_distance_weight_0p05",
        )

    def test_scored_partial_setting_has_distinct_identity(self) -> None:
        self.assertEqual(
            MODULE._setting_name(1.0, 0.05, "scored_partial"),
            "learned_bonus_1__raw_distance_weight_0p05__topology_scored_partial",
        )

    def test_zero_bonus_is_an_explicit_geometry_control(self) -> None:
        self.assertEqual(MODULE._validate_values([0.0, 0.5], 0.05), (0.0, 0.5))
        self.assertEqual(
            MODULE._setting_name(0.0, 0.05, "legacy"),
            "learned_bonus_0__raw_distance_weight_0p05",
        )

    def test_node_threshold_filters_detections_and_pair_features_together(self) -> None:
        frame0 = MODULE.bh_model.FrameDet(
            frame_index=0,
            node_ids=np.asarray([1, 2]),
            coords_zyx=np.asarray([[0, 0, 0], [0, 1, 0]]),
            scores=np.asarray([0.98, 0.99]),
            grid_coords_zyx=np.asarray([[0, 0, 0], [0, 1, 0]]),
            grid_shape_zyx=(1, 2, 1),
        )
        frame1 = MODULE.bh_model.FrameDet(
            frame_index=1,
            node_ids=np.asarray([3, 4]),
            coords_zyx=np.asarray([[0, 0, 0], [0, 1, 0]]),
            scores=np.asarray([0.97, 0.995]),
            grid_coords_zyx=np.asarray([[0, 0, 0], [0, 1, 0]]),
            grid_shape_zyx=(1, 2, 1),
        )
        context = MODULE.bh_model.PairContext(
            source_frame=0,
            target_frame=1,
            source_ids=np.asarray([1, 2]),
            target_ids=np.asarray([3, 4]),
            source_feat=np.zeros((2, 32), dtype=np.float32),
            target_feat=np.ones((2, 32), dtype=np.float32),
            grid_shape_zyx=(1, 2, 1),
        )
        filtered, audit = MODULE._filter_cache_by_node_threshold(
            type("Cache", (), {"detections": (frame0, frame1), "contexts": (context,)})(),
            0.985,
        )
        self.assertEqual(filtered.detections[0].node_ids.tolist(), [2])
        self.assertEqual(filtered.detections[1].node_ids.tolist(), [4])
        self.assertEqual(filtered.contexts[0].source_ids.tolist(), [2])
        self.assertEqual(filtered.contexts[0].target_ids.tolist(), [4])
        self.assertEqual(audit["removed_nodes"], 2)
        self.assertEqual(
            MODULE._setting_name(1.0, 0.05, "legacy", 0.985),
            "learned_bonus_1__raw_distance_weight_0p05__node_threshold_0p985",
        )

    def test_parser_requires_explicit_valid_topology(self) -> None:
        args = MODULE.parser().parse_args(["--topology-mode", "scored_partial"])
        self.assertEqual(args.topology_mode, "scored_partial")
        with self.assertRaises(SystemExit):
            MODULE.parser().parse_args(["--topology-mode", "unknown"])

    def test_subset_nan_metrics_are_strict_json_nulls(self) -> None:
        safe = MODULE._json_safe({"division_jaccard": math.nan, "nested": [math.inf]})
        self.assertEqual(
            json.dumps(safe, allow_nan=False),
            '{"division_jaccard": null, "nested": [null]}',
        )


if __name__ == "__main__":
    unittest.main()
