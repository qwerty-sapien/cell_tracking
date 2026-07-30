from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import evaluate_c1_hoct_lite_graph_gate as gate


class C1GraphGateTests(unittest.TestCase):
    def test_locked_contracts_authenticate(self) -> None:
        prereg, n2 = gate._authenticate_contracts()
        self.assertEqual(prereg["self_sha256"], gate.PREREGISTRATION_SHA256)
        self.assertEqual(n2["verdict"], "GO")

    def test_auc_with_ties(self) -> None:
        value = gate._auc(
            np.asarray([0.0, 0.5, 0.5, 1.0]),
            np.asarray([0, 0, 1, 1]),
        )
        self.assertAlmostEqual(value, 0.875)

    def test_dead_at_nonpositive_adjusted_delta(self) -> None:
        graph = self._graph(candidate_adjusted=0.8)
        self.assertEqual(gate._decision(graph, "44b6")["verdict"], "DEAD")

    def test_go_requires_alpha_margin_and_distinct_observation(self) -> None:
        graph = self._graph(candidate_adjusted=0.806)
        decision = gate._decision(graph, "44b6")
        self.assertEqual(decision["verdict"], "GO")
        graph["observation_identity"]["observation_distinct"] = False
        self.assertEqual(gate._decision(graph, "44b6")["verdict"], "PARK")

    @staticmethod
    def _graph(candidate_adjusted: float) -> dict:
        return {
            "apncc": {
                "summary": {
                    "adj_edge_jaccard": 0.8,
                    "edge_jaccard": 0.8,
                    "node_recall": 0.9,
                }
            },
            "c0": {
                "summary": {
                    "adj_edge_jaccard": 0.8,
                    "edge_jaccard": 0.8,
                    "node_recall": 0.9,
                }
            },
            "c1": {
                "summary": {
                    "adj_edge_jaccard": candidate_adjusted,
                    "edge_jaccard": 0.806,
                    "node_recall": 0.9,
                }
            },
            "null": {"epsilon_null": 1.0e-12},
            "observation_identity": {"observation_distinct": True},
        }


if __name__ == "__main__":
    unittest.main()
