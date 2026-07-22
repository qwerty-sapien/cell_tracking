from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha import bh_graph


def _node_um(node_id: int, t: int, x_um: float) -> bh_graph.GraphNode:
    return bh_graph.GraphNode(
        node_id,
        t,
        0.0,
        0.0,
        x_um / bh_graph.PHYSICAL_SCALE_ZYX[2],
    )


def _provenance() -> dict[str, object]:
    return {
        "schema": bh_graph.EXPERIMENTAL_SWEEP_SCHEMA,
        "experiment_id": "phase1-e1-topk-unit-test",
        "purpose": "offline-cached-cv57-sweep",
        "varied_fields": ["candidate_top_k"],
        "production_eligible": False,
        "submit_eligible": False,
    }


class TopKPairTests(unittest.TestCase):
    def test_distance_then_node_id_breaks_boundary_tie(self) -> None:
        source = _node_um(0, 0, 0.0)
        targets = (
            _node_um(11, 1, -1.0),
            _node_um(10, 1, 1.0),
            _node_um(12, 1, 2.0),
        )
        pairs = bh_graph._top_k_pairs(
            (source,), targets, 1, bh_graph.PHYSICAL_SCALE_ZYX
        )
        self.assertEqual([(row[0].node_id, row[1].node_id) for row in pairs], [(0, 10)])
        self.assertAlmostEqual(pairs[0][2], 1.0)

    def test_top_k_includes_targets_outside_metric_radius(self) -> None:
        nodes = (
            _node_um(0, 0, 0.0),
            _node_um(10, 1, 2.0),
            _node_um(11, 1, 20.0),
        )
        _, default_stats = bh_graph.build_adjacent_edges(nodes)
        _, topk_stats = bh_graph.build_adjacent_edges(
            nodes,
            _experimental_sweep_provenance=_provenance(),
            _experimental_candidate_top_k=2,
        )
        self.assertEqual(default_stats["adjacent_candidate_pairs"], 1)
        self.assertEqual(default_stats["radius_candidate_pairs_diagnostic"], 1)
        self.assertEqual(default_stats["candidate_top_k"], 0)
        self.assertEqual(topk_stats["adjacent_candidate_pairs"], 2)
        self.assertEqual(topk_stats["radius_candidate_pairs_diagnostic"], 1)
        self.assertEqual(topk_stats["candidate_top_k"], 2)

    def test_override_requires_matching_offline_provenance(self) -> None:
        nodes = (_node_um(0, 0, 0.0), _node_um(10, 1, 1.0))
        with self.assertRaises(bh_graph.GraphCalibrationError):
            bh_graph.build_adjacent_edges(
                nodes,
                _experimental_candidate_top_k=10,
            )
        wrong = {**_provenance(), "varied_fields": ["motion_error_weight"]}
        with self.assertRaises(bh_graph.GraphCalibrationError):
            bh_graph.build_adjacent_edges(
                nodes,
                _experimental_sweep_provenance=wrong,
                _experimental_candidate_top_k=10,
            )

    def test_calibration_records_top_k_and_keeps_divisions_off(self) -> None:
        result = bh_graph.calibrate_v45_graph(
            (_node_um(0, 0, 0.0), _node_um(10, 1, 1.0)),
            division_enabled=False,
            _experimental_sweep_provenance=_provenance(),
            _experimental_candidate_top_k=10,
        )
        self.assertEqual(result.stats["candidate_top_k"], 10)
        self.assertEqual(result.provenance["candidate_top_k"], 10)
        self.assertIs(result.provenance["division_enabled"], False)
        self.assertIs(result.provenance["production_eligible"], False)


if __name__ == "__main__":
    unittest.main()
