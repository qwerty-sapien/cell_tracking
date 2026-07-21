from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v45gdiag", ROOT / "scripts" / "v45gdiag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V45GraphDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_sparse_diagnostic_separates_reachable_and_selected_edges(self) -> None:
        nodes = [
            {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
            {"node_id": 2, "t": 1, "z": 0, "y": 0, "x": 2},
            {"node_id": 3, "t": 1, "z": 0, "y": 8, "x": 0},
        ]
        gt = {
            "nodes": {
                "node_id": [101, 102, 103],
                "t": [0, 1, 1],
                "z": [0, 0, 0],
                "y": [0, 0, 8],
                "x": [0, 2, 0],
            },
            "edges": [(101, 102), (101, 103)],
        }
        result = self.mod.diagnose(nodes, {(1, 2)}, gt, (1.0, 1.0, 1.0))
        headroom = result["edge_headroom"]
        self.assertEqual(headroom["matched_endpoint_edges"], 2)
        self.assertEqual(headroom["reachable_at_10um"], 2)
        self.assertEqual(headroom["current_selected_true_positives"], 1)
        self.assertEqual(headroom["reachable_but_not_selected"], 1)
        self.assertEqual(headroom["raw_distance_rank_histogram"], {"1": 1, "2": 1})
        self.assertEqual(result["posthoc_local_cost_probe"]["source_conflicts"], 1)
        self.assertEqual(result["posthoc_local_cost_probe"]["source_conflicts_where_true_cost_lower"], 0)
        self.assertFalse(result["candidate_graph"]["full_frame_dense_matrix"])

    def test_candidate_discovery_uses_a_physical_radius(self) -> None:
        sources = [{"node_id": 1, "t": 0, "z": 1, "y": 0, "x": 0}]
        targets = [
            {"node_id": 2, "t": 1, "z": 0, "y": 0, "x": 0},
            {"node_id": 3, "t": 1, "z": 0, "y": 3, "x": 0},
        ]
        pairs = self.mod._radius_pairs(sources, targets, radius_um=2.0, scale_zyx=(2.0, 1.0, 1.0))
        self.assertEqual([(source, target) for source, target, _ in pairs], [(1, 2)])

    def test_source_never_builds_a_dense_pair_matrix(self) -> None:
        source = (ROOT / "scripts" / "v45gdiag.py").read_text(encoding="ascii")
        self.assertIn("cKDTree", source)
        self.assertNotIn("linear_sum_assignment", source)
        self.assertNotIn("[:, None", source)


if __name__ == "__main__":
    unittest.main()
