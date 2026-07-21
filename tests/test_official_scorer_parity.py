from __future__ import annotations

import math
import sys

import numpy as np
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.official_eval import official_edge_metric_row, official_metric_row  # noqa: E402
from celltrack_alpha.bh_graph import GraphEdge, GraphNode, GraphResult  # noqa: E402
from celltrack_alpha.scorer_parity import (  # noqa: E402
    CASES,
    LINE_EDGES,
    LINE_NODES,
    SCALE_ZYX,
    _gt,
    _pred_edges,
    _pred_nodes,
    run_scorer_parity,
)


class OfficialScorerParityTests(unittest.TestCase):
    def test_scorer_parity_cases_pass(self) -> None:
        report = run_scorer_parity()
        self.assertTrue(report["overall_pass"])
        self.assertGreaterEqual(len(report["cases"]), 7)

    def test_adjusted_node_count_penalty_can_exceed_one(self) -> None:
        report = run_scorer_parity()
        case = next(
            row
            for row in report["cases"]
            if row["name"] == "gt_self_with_adjusted_node_count_penalty"
        )
        self.assertEqual(case["actual"]["edge_jaccard"], 1.0)
        self.assertAlmostEqual(case["actual"]["total_node_ratio"], -0.7)
        self.assertAlmostEqual(case["actual"]["adj_edge_jaccard"], 1.07)
        self.assertTrue(math.isnan(float(case["actual"]["division_jaccard"])))


    def test_empty_gt_clip_returns_skipped_nan_row(self) -> None:
        gt = {
            "nodes": {
                "node_id": np.asarray([], dtype=np.int64),
                "t": np.asarray([], dtype=np.int64),
                "z": np.asarray([], dtype=np.int64),
                "y": np.asarray([], dtype=np.int64),
                "x": np.asarray([], dtype=np.int64),
            },
            "edges": np.empty((0, 2), dtype=np.int64),
            "estimated_number_of_nodes": 0.0,
        }
        row = official_metric_row(
            [{"dataset": "empty", "node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0}],
            [],
            gt,
            (1.625, 0.40625, 0.40625),
        )
        self.assertTrue(math.isnan(float(row["edge_tp"])))
        self.assertEqual(row["num_pred_nodes"], 1.0)

    def test_division_false_positive_fixture_is_active(self) -> None:
        report = run_scorer_parity()
        case = next(
            row
            for row in report["cases"]
            if row["name"] == "division_extra_fork_counts_division_fp"
        )
        self.assertEqual(case["actual"]["division_tp"], 1.0)
        self.assertEqual(case["actual"]["division_fp"], 1.0)
        self.assertEqual(case["actual"]["division_jaccard"], 0.5)

    def test_edge_only_scorer_matches_full_edge_metrics(self) -> None:
        edge_keys = (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "num_pred_nodes",
            "node_recall",
            "total_node_ratio",
            "edge_jaccard",
            "adj_edge_jaccard",
            "estimated_number_of_nodes",
        )
        for case in CASES:
            with self.subTest(case=case["name"]):
                gt = _gt(case["gt_nodes"], case["gt_edges"], case["estimated_nodes"])
                nodes = _pred_nodes(case["pred_nodes"])
                edges = _pred_edges(case["pred_edges"])
                full = official_metric_row(nodes, edges, gt, SCALE_ZYX)
                edge_only = official_edge_metric_row(nodes, edges, gt, SCALE_ZYX)
                for key in edge_keys:
                    self.assertAlmostEqual(float(edge_only[key]), float(full[key]))
                self.assertEqual(edge_only["division_tp"], 0.0)
                self.assertEqual(edge_only["division_fp"], 0.0)
                self.assertEqual(edge_only["division_fn"], 0.0)
                self.assertTrue(math.isnan(float(edge_only["division_jaccard"])))

    def test_edge_only_scorer_handles_predictions_with_no_edges(self) -> None:
        row = official_edge_metric_row(
            _pred_nodes(LINE_NODES),
            [],
            _gt(LINE_NODES, LINE_EDGES, 10.0),
            SCALE_ZYX,
        )
        self.assertEqual(row["edge_tp"], 0.0)
        self.assertEqual(row["edge_fp"], 0.0)
        self.assertEqual(row["edge_fn"], 2.0)
        self.assertEqual(row["num_pred_nodes"], 3.0)
        self.assertEqual(row["node_recall"], 1.0)
        self.assertEqual(row["edge_jaccard"], 0.0)
        self.assertEqual(row["adj_edge_jaccard"], 0.0)

    def test_official_scorer_accepts_image_support_corrected_rows(self) -> None:
        from scripts import run_bh

        raw = GraphResult(
            (
                GraphNode(1, 0, 0.0, 0.0, -0.5000001),
                GraphNode(2, 1, 0.0, 0.0, 1.0),
            ),
            (GraphEdge(1, 2, 1.0, 1.0),),
            {},
            {"full_frame_dense_matrix": False},
        )
        corrected = run_bh.project_graph_result_to_image_support(
            "official", raw, (2, 1, 1, 2)
        )
        public = run_bh.graph_rows("official", corrected, (2, 1, 1, 2))
        nodes = [row for row in public if row["row_type"] == "node"]
        edges = [row for row in public if row["row_type"] == "edge"]
        result = official_metric_row(
            nodes,
            edges,
            _gt([(1, 0, 0, 0, 0), (2, 1, 0, 0, 1)], [(1, 2)], 2.0),
            SCALE_ZYX,
        )
        self.assertEqual(result["edge_tp"], 1.0)
        self.assertEqual(result["edge_fp"], 0.0)
        self.assertEqual(result["edge_fn"], 0.0)
        self.assertEqual(len(corrected.nodes), len(raw.nodes))
        self.assertEqual(corrected.edges, raw.edges)


if __name__ == "__main__":
    unittest.main()
