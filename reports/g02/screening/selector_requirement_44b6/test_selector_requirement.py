#!/usr/bin/env python3
"""Focused tests for the selector-requirement diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "selector_requirement_under_test", HERE / "selector_requirement.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load selector requirement module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SelectorRequirementTests(unittest.TestCase):
    def test_support_grid_is_fixed_log2_plus_population(self) -> None:
        self.assertEqual(MODULE.support_grid(1), [0, 1])
        self.assertEqual(MODULE.support_grid(8), [0, 1, 2, 4, 8])
        self.assertEqual(MODULE.support_grid(10), [0, 1, 2, 4, 8, 10])

    def test_empirical_auc_exact_extremes(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        self.assertEqual(
            MODULE.empirical_auc(labels, np.asarray([0.0, 1.0, 2.0, 3.0])),
            1.0,
        )
        self.assertEqual(
            MODULE.empirical_auc(labels, np.asarray([3.0, 2.0, 1.0, 0.0])),
            0.0,
        )

    def test_controlled_scores_are_deterministic_and_hit_auc(self) -> None:
        labels = np.asarray([1] * 200 + [0] * 800, dtype=np.uint8)
        first = MODULE.controlled_scores(labels, "fixture", 3, 0.85)
        second = MODULE.controlled_scores(labels, "fixture", 3, 0.85)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertLessEqual(abs(first[1] - 0.85), 0.5 / (200 * 800) + 1e-15)
        self.assertGreaterEqual(first[2], 0.0)

    def test_retention_enforces_each_video_budget(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        scores = np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        video = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int16)
        fold = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
        result = MODULE.retention_metrics(
            labels, scores, video, fold, np.asarray([0, 1], dtype=np.int64)
        )
        self.assertEqual(result["selected_harmful"], 1)
        self.assertEqual(result["selected_beneficial"], 2)
        self.assertAlmostEqual(result["retention"], 2 / 3)
        self.assertTrue(
            all(
                row["selected_harmful"] <= row["budget"]
                for row in result["videos"]
            )
        )

    def test_edge_summary_matches_official_weighting(self) -> None:
        rows = [
            {
                "edge_tp": 8,
                "edge_fp": 1,
                "edge_fn": 1,
                "edge_jaccard": 0.8,
                "adj_edge_jaccard": 0.72,
                "node_recall": 1.0,
            },
            {
                "edge_tp": 2,
                "edge_fp": 1,
                "edge_fn": 1,
                "edge_jaccard": 0.5,
                "adj_edge_jaccard": 0.45,
                "node_recall": 0.8,
            },
        ]
        summary = MODULE.edge_summary(rows)
        self.assertAlmostEqual(summary["edge_jaccard"], 10 / 14)
        self.assertAlmostEqual(summary["adj_edge_jaccard"], (10 * 0.72 + 4 * 0.45) / 14)
        self.assertAlmostEqual(summary["node_recall"], 0.9)

    def test_requirement_detects_oracle_infeasibility(self) -> None:
        curve = [
            {
                "candidate_prefix_N": 0,
                "graph_delta": {"adjusted_edge_jaccard": 0.0},
            },
            {
                "candidate_prefix_N": 4,
                "graph_delta": {"adjusted_edge_jaccard": 0.001},
            },
        ]
        result = MODULE.requirement_for(curve, None, 0.005)
        self.assertEqual(result["status"], "INFEASIBLE_ON_FROZEN_POPULATION")

    def test_class_degenerate_auc_fails_closed(self) -> None:
        labels = np.ones(10, dtype=np.uint8)
        with self.assertRaises(MODULE.DiagnosticError):
            MODULE.controlled_scores(labels, "fixture", 0, 0.65)


if __name__ == "__main__":
    unittest.main()
