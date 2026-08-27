from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
    prune_long_edges,
    prune_short_components,
    read_submission,
    rescue_divisions,
    thin_edges,
    validate_graphs,
    write_submission,
)


def node(node_id: int, t: int, z: int, y: int, x: int) -> SubmissionNode:
    return SubmissionNode(node_id, t, z, y, x)


class KaggleCalibrationTests(unittest.TestCase):
    def test_round_trip_preserves_graph(self) -> None:
        graph = SubmissionGraph(
            "sample",
            {1: node(1, 0, 0, 0, 0), 2: node(2, 1, 0, 1, 0)},
            [SubmissionEdge(1, 2)],
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "submission.csv"
            write_submission({"sample": graph}, path)
            loaded = read_submission(path)["sample"]
        self.assertEqual(loaded.nodes, graph.nodes)
        self.assertEqual(loaded.edges, graph.edges)

    def test_higher_degree_parent_requires_explicit_parser_policy(self) -> None:
        graph = SubmissionGraph(
            "sample",
            {
                1: node(1, 0, 0, 0, 0),
                2: node(2, 1, 0, 1, 0),
                3: node(3, 1, 0, 2, 0),
                4: node(4, 1, 0, 3, 0),
            },
            [SubmissionEdge(1, 2), SubmissionEdge(1, 3), SubmissionEdge(1, 4)],
        )
        with self.assertRaisesRegex(ValueError, "out-degree greater than 2"):
            validate_graphs({"sample": graph})
        validate_graphs({"sample": graph}, max_outdegree=None)

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "external.csv"
            with self.assertRaisesRegex(ValueError, "out-degree greater than 2"):
                write_submission({"sample": graph}, path)
            write_submission({"sample": graph}, path, max_outdegree=None)
            loaded = read_submission(path, max_outdegree=None)["sample"]
        self.assertEqual(loaded.edges, graph.edges)

    def test_edge_thinning_is_deterministic(self) -> None:
        nodes = {index: node(index, index, 0, index, 0) for index in range(1, 102)}
        graph = SubmissionGraph(
            "sample",
            nodes,
            [SubmissionEdge(index, index + 1) for index in range(1, 101)],
        )
        first, first_stats = thin_edges(graph, fraction=0.10)
        second, second_stats = thin_edges(graph, fraction=0.10)
        self.assertEqual(first.edges, second.edges)
        self.assertEqual(first_stats, second_stats)
        self.assertGreater(first_stats["removed_edges"], 0)

    def test_long_edge_pruning_uses_physical_scale(self) -> None:
        graph = SubmissionGraph(
            "sample",
            {
                1: node(1, 0, 0, 0, 0),
                2: node(2, 1, 0, 5, 0),
                3: node(3, 1, 0, 20, 0),
            },
            [SubmissionEdge(1, 2), SubmissionEdge(1, 3)],
        )
        pruned, stats = prune_long_edges(graph, max_distance_um=5.0)
        self.assertEqual(pruned.edges, [SubmissionEdge(1, 2)])
        self.assertEqual(stats["removed_edges"], 1)

    def test_short_component_pruning_removes_one_edge_track(self) -> None:
        graph = SubmissionGraph(
            "sample",
            {index: node(index, index - 1, 0, index, 0) for index in range(1, 6)},
            [SubmissionEdge(1, 2), SubmissionEdge(3, 4), SubmissionEdge(4, 5)],
        )
        pruned, stats = prune_short_components(graph, max_component_edges=1)
        self.assertEqual(set(pruned.nodes), {3, 4, 5})
        self.assertEqual(stats["removed_components"], 1)

    def test_division_rescue_links_supported_birth_track(self) -> None:
        graph = SubmissionGraph(
            "sample",
            {
                1: node(1, 0, 10, 10, 10),
                2: node(2, 1, 10, 11, 10),
                3: node(3, 2, 10, 12, 6),
                4: node(4, 2, 10, 12, 16),
                5: node(5, 3, 10, 13, 5),
                6: node(6, 3, 10, 13, 17),
            },
            [
                SubmissionEdge(1, 2),
                SubmissionEdge(2, 3),
                SubmissionEdge(3, 5),
                SubmissionEdge(4, 6),
            ],
        )
        rescued, stats = rescue_divisions(graph)
        self.assertIn(SubmissionEdge(2, 4), rescued.edges)
        self.assertEqual(stats["added_division_edges"], 1)


if __name__ == "__main__":
    unittest.main()
