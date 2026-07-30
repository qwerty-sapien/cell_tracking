from __future__ import annotations

import unittest

from celltrack_alpha.kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)
from scripts.analyze_action_class_oracle import (
    _copy_with,
    _replace_incident,
    edge_fp_budget,
    select_oracle_row,
)


class ActionClassOracleTest(unittest.TestCase):
    def test_replace_incident_removes_complete_endpoint_conflicts(self) -> None:
        edges = {(1, 2), (1, 3), (4, 5), (6, 5), (7, 8)}
        result, removed = _replace_incident(
            edges,
            sources={1},
            targets={5},
            additions={(1, 5)},
        )
        self.assertEqual(result, {(1, 5), (7, 8)})
        self.assertEqual(removed, 4)

    def test_copy_with_rejects_merge(self) -> None:
        graph = SubmissionGraph(
            "fixture",
            {
                1: SubmissionNode(1, 0, 0, 0, 0),
                2: SubmissionNode(2, 0, 0, 1, 0),
                3: SubmissionNode(3, 1, 0, 0, 0),
            },
            [SubmissionEdge(1, 3)],
        )
        with self.assertRaises(ValueError):
            _copy_with(graph, edges={(1, 3), (2, 3)})

    def test_edge_fp_budget_is_zero_when_oracle_misses_target(self) -> None:
        summary = {
            "score": 0.90,
            "edge_jaccard": 0.90,
            "adj_edge_jaccard": 0.90,
            "division_jaccard": 0.0,
            "edge_tp": 90,
            "edge_fp": 5,
            "edge_fn": 5,
        }
        budget = edge_fp_budget(summary, 0.9255)
        self.assertFalse(budget["reaches_target_before_extra_fp"])
        self.assertEqual(budget["additional_edge_fp_allowed"], 0)

    def test_edge_fp_budget_counts_additional_false_edges(self) -> None:
        summary = {
            "score": 1.0,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 1.0,
            "division_jaccard": 0.0,
            "edge_tp": 100,
            "edge_fp": 0,
            "edge_fn": 0,
        }
        budget = edge_fp_budget(summary, 0.90)
        self.assertEqual(budget["additional_edge_fp_allowed"], 11)

    def test_oracle_selects_noop_when_intervention_hurts_score(self) -> None:
        action = {"score": 0.8, "edge_jaccard": 0.9}
        baseline = {"score": 0.85, "edge_jaccard": 0.8}
        selected, no_op = select_oracle_row(action, baseline)
        self.assertTrue(no_op)
        self.assertEqual(selected, baseline)

    def test_oracle_keeps_intervention_when_score_improves(self) -> None:
        action = {"score": 0.9, "edge_jaccard": 0.9}
        baseline = {"score": 0.85, "edge_jaccard": 0.8}
        selected, no_op = select_oracle_row(action, baseline)
        self.assertFalse(no_op)
        self.assertEqual(selected, action)


if __name__ == "__main__":
    unittest.main()
