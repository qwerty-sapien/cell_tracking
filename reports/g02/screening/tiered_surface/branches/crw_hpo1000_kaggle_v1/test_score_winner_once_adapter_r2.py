#!/usr/bin/env python3
"""Outcome-free differential tests for the bounded winner-score adapter."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import score_winner_once_adapter_r2 as adapter
from celltrack_alpha.kaggle_calibration import SubmissionNode


a0 = adapter.a0
SubmissionEdge = a0.SubmissionEdge
SubmissionGraph = a0.SubmissionGraph


def graph(edges: set[tuple[int, int]]) -> SubmissionGraph:
    nodes = {
        node_id: SubmissionNode(node_id, (node_id - 1) // 2, 0, node_id * 3, 0)
        for node_id in range(1, 9)
    }
    return SubmissionGraph(
        "synthetic",
        nodes,
        [SubmissionEdge(*pair) for pair in sorted(edges)],
    )


class BoundedCompilerTests(unittest.TestCase):
    def assert_batch_equal(self, expected: object, observed: object) -> None:
        for field in (
            "actions",
            "before_graph_sha256",
            "after_graph_sha256",
            "edge_set_sha256",
            "batch_sha256",
            "action_sha256s",
            "retained_anchors",
        ):
            self.assertEqual(getattr(expected, field), getattr(observed, field))
        self.assertEqual(a0.edge_set(expected.graph), a0.edge_set(observed.graph))

    def test_empty_batch_matches_reference(self) -> None:
        baseline = graph({(1, 3), (2, 4)})
        self.assert_batch_equal(
            adapter._REFERENCE_COMPILE(baseline, []),
            adapter.compile_replace_components_bounded(baseline, []),
        )

    def test_valid_multi_action_batch_matches_reference_in_both_orders(self) -> None:
        baseline_edges = {(1, 4), (2, 3), (3, 6), (4, 5)}
        oracle_edges = {(1, 3), (2, 4), (3, 5), (4, 6)}
        actions = a0.reconstruct_components(
            "train:synthetic", baseline_edges, set(), oracle_edges
        )
        for ordered in (actions, list(reversed(actions))):
            baseline = graph(baseline_edges)
            self.assert_batch_equal(
                adapter._REFERENCE_COMPILE(baseline, ordered),
                adapter.compile_replace_components_bounded(baseline, ordered),
            )

    def test_retained_anchor_batch_matches_reference(self) -> None:
        baseline_edges = {(1, 3), (2, 4)}
        actions = a0.reconstruct_components(
            "train:synthetic", baseline_edges, {(1, 3)}, {(1, 3), (1, 4)}
        )
        baseline = graph(baseline_edges)
        self.assert_batch_equal(
            adapter._REFERENCE_COMPILE(baseline, actions),
            adapter.compile_replace_components_bounded(baseline, actions),
        )

    def test_cross_action_overlap_rejection_matches_reference(self) -> None:
        baseline = graph({(1, 3), (2, 4)})
        first = a0.ReplaceComponent(
            "train:synthetic", 0, ((1, 4),), ((1, 4),),
            ((1, 3), (2, 4)), (1,), (4,),
        )
        second = a0.ReplaceComponent(
            "train:synthetic", 1, ((2, 3),), ((2, 3),),
            ((1, 3), (2, 4)), (2,), (3,),
        )
        for compiler in (
            adapter._REFERENCE_COMPILE,
            adapter.compile_replace_components_bounded,
        ):
            with self.assertRaisesRegex(a0.ScientificMismatch, "removed by both"):
                compiler(baseline, [first, second])


if __name__ == "__main__":
    unittest.main()
