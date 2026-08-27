from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from celltrack_alpha.kaggle_calibration import (
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)

from reports.g02.screening.selector_requirement_44b6 import (
    selector_requirement_class as isolated,
)


class IsolatedClassRunnerTests(unittest.TestCase):
    def test_lightweight_runner_does_not_import_torch(self) -> None:
        source = Path(isolated.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("torch", imported)

    def test_truth_topology_matches_direct_sets(self) -> None:
        truth, outgoing, incoming = isolated.truth_topology(
            {
                "edges": [(1, 3), (2, 4), (3, 5)],
            }
        )
        self.assertEqual(truth, {(1, 3), (2, 4), (3, 5)})
        self.assertEqual(outgoing, {1: {3}, 2: {4}, 3: {5}})
        self.assertEqual(incoming, {3: {1}, 4: {2}, 5: {3}})

    def test_edge_class_preserves_true_and_valid_semantics(self) -> None:
        pred_to_gt = {10: 1, 20: 2, 30: 3}
        truth = {(1, 3)}
        outgoing = {1: {3}}
        incoming = {3: {1}}
        self.assertEqual(
            isolated.edge_class(
                (10, 30), pred_to_gt, truth, outgoing, incoming
            ),
            (True, True),
        )
        self.assertEqual(
            isolated.edge_class(
                (10, 20), pred_to_gt, truth, outgoing, incoming
            ),
            (False, True),
        )
        self.assertEqual(
            isolated.edge_class(
                (20, 20), pred_to_gt, truth, outgoing, incoming
            ),
            (False, False),
        )

    def test_edge_sets_use_exact_source_target_pairs(self) -> None:
        graph = SubmissionGraph(
            "fixture",
            {
                1: SubmissionNode(1, 0, 0, 0, 0),
                2: SubmissionNode(2, 1, 0, 0, 1),
                3: SubmissionNode(3, 1, 0, 0, 2),
            },
            [SubmissionEdge(1, 2), SubmissionEdge(1, 3)],
        )
        self.assertEqual(isolated.edge_sets(graph), {(1, 2), (1, 3)})

    def test_bulk_swap_replay_matches_atomic_batch_union(self) -> None:
        graph = SubmissionGraph(
            "fixture",
            {
                node_id: SubmissionNode(
                    node_id, 0 if node_id <= 4 else 1, 0, 0, node_id
                )
                for node_id in range(1, 9)
            },
            [
                SubmissionEdge(1, 5),
                SubmissionEdge(2, 6),
                SubmissionEdge(3, 7),
                SubmissionEdge(4, 8),
            ],
        )
        population = isolated.base.EdgePopulation(
            "occupied_swap",
            ("44b6:fixture",),
            (graph,),
            ({},),
            np.asarray([0, 0]),
            np.asarray([1, 3]),
            np.asarray([6, 8]),
            np.asarray([1.0, 0.5]),
            np.asarray([1, 1], dtype=np.uint8),
            np.asarray([0, 0], dtype=np.int8),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int8),
            np.asarray([0, 0], dtype=np.int8),
            np.asarray([0, 0], dtype=np.int8),
            (((1, 5), (2, 6)), ((3, 7), (4, 8))),
        )
        selected = np.asarray([0, 1], dtype=np.int64)
        atomic = isolated.base.replay_compiler(population, selected)
        bulk = isolated.replay_swap_compiler(population, selected)
        self.assertEqual(
            atomic["receipts"][0]["after_graph_sha256"],
            bulk["receipts"][0]["after_graph_sha256"],
        )
        self.assertEqual(bulk["compiled_actions"], 2)
        self.assertEqual(bulk["compiler_transactions"], 1)


if __name__ == "__main__":
    unittest.main()
