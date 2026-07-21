from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_fixed_node_association_oracle import oracle_graph  # noqa: E402
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    SubmissionEdge,
    SubmissionGraph,
    SubmissionNode,
)


class FixedNodeAssociationOracleTests(unittest.TestCase):
    def test_oracle_preserves_nodes_and_recovers_supported_gt_edges(self) -> None:
        graph = SubmissionGraph(
            "tiny",
            {
                10: SubmissionNode(10, 0, 0, 0, 0),
                20: SubmissionNode(20, 1, 0, 1, 0),
                30: SubmissionNode(30, 2, 0, 2, 0),
                40: SubmissionNode(40, 1, 0, 100, 100),
            },
            [SubmissionEdge(10, 40)],
        )
        gt = {
            "nodes": {
                "node_id": np.asarray([1, 2, 3], dtype=np.int64),
                "t": np.asarray([0, 1, 2], dtype=np.int64),
                "z": np.asarray([0, 0, 0], dtype=np.float64),
                "y": np.asarray([0, 1, 2], dtype=np.float64),
                "x": np.asarray([0, 0, 0], dtype=np.float64),
            },
            "edges": np.asarray([[1, 2], [2, 3]], dtype=np.int64),
        }
        result, stats = oracle_graph(graph, gt, (1.625, 0.40625, 0.40625))
        self.assertEqual(result.nodes, graph.nodes)
        self.assertEqual(result.edges, [SubmissionEdge(10, 20), SubmissionEdge(20, 30)])
        self.assertEqual(stats["matched_nodes"], 3)
        self.assertEqual(stats["recoverable_gt_edges"], 2)


if __name__ == "__main__":
    unittest.main()
