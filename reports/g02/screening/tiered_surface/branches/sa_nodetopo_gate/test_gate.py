#!/usr/bin/env python3
"""Synthetic checks for the preregistered graph-gate evaluator."""

from __future__ import annotations

import math
import unittest

from evaluate_graph_gate import gate_decision, graph_action, json_safe


GATE = {
    "minimum_adjusted_edge_jaccard_delta": 0.005,
    "minimum_raw_edge_jaccard_delta": 0.0,
    "minimum_node_recall_delta": 0.0,
    "pass_verdict": "PASS_44B6",
    "fail_verdict": "KILL_44B6_STOP",
}


class GateTests(unittest.TestCase):
    def test_graph_action_field_mapping(self) -> None:
        action = graph_action(
            {
                "action_id": "fixture.restore",
                "action_kind": "restore_topology",
                "add_nodes": [{"node_id": 9, "t": 1, "z": 2, "y": 3, "x": 4}],
                "add_edges": [[1, 9], [9, 2]],
                "replace_outgoing_from": [1],
                "replace_incoming_to": [2],
                "expected_conflict_edges": [[1, 2]],
            }
        )
        self.assertEqual(action.add_nodes[0].coord_zyx, (2, 3, 4))
        self.assertEqual(action.add_edges, ((1, 9), (9, 2)))

    def test_gate_pass_at_exact_floor(self) -> None:
        baseline = {
            "adj_edge_jaccard": 0.8,
            "edge_jaccard": 0.7,
            "node_recall": 0.9,
        }
        candidate = {
            "adj_edge_jaccard": 0.805,
            "edge_jaccard": 0.7,
            "node_recall": 0.9,
        }
        decision = gate_decision(baseline, candidate, GATE)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["pass_unblocks_proposed"], ["DET-F0", "DET-F2", "DET-F3", "NODERESTORE-O0"])

    def test_adjusted_miss_parks_instantiation_only(self) -> None:
        baseline = {
            "adj_edge_jaccard": 0.8,
            "edge_jaccard": 0.7,
            "node_recall": 0.9,
        }
        candidate = {
            "adj_edge_jaccard": 0.804999,
            "edge_jaccard": 0.7,
            "node_recall": 0.9,
        }
        decision = gate_decision(baseline, candidate, GATE)
        self.assertFalse(decision["passed"])
        self.assertEqual(
            decision["instantiation_tier_proposal"],
            {
                "id": "BLOCKPEAK-RESTORE-V1",
                "tier": "PARK_C",
                "scope": "this exact 50-action instantiation only",
                "broader_family_tier_changes": [],
            },
        )

    def test_raw_or_node_regression_kills(self) -> None:
        baseline = {
            "adj_edge_jaccard": 0.8,
            "edge_jaccard": 0.7,
            "node_recall": 0.9,
        }
        raw_fail = {
            "adj_edge_jaccard": 0.81,
            "edge_jaccard": 0.699,
            "node_recall": 0.9,
        }
        node_fail = {
            "adj_edge_jaccard": 0.81,
            "edge_jaccard": 0.7,
            "node_recall": 0.899,
        }
        self.assertFalse(gate_decision(baseline, raw_fail, GATE)["passed"])
        self.assertFalse(gate_decision(baseline, node_fail, GATE)["passed"])

    def test_nonfinite_diagnostic_is_serialized_as_null(self) -> None:
        self.assertEqual(
            json_safe({"metric": math.nan, "nested": [math.inf, 1.0]}),
            {"metric": None, "nested": [None, 1.0]},
        )


if __name__ == "__main__":
    unittest.main()
