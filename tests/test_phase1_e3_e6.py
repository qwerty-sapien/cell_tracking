from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

from celltrack_alpha.bh_graph import (
    EXPERIMENTAL_SWEEP_SCHEMA,
    PHYSICAL_SCALE_ZYX,
    GraphCalibrationError,
    GraphNode,
    build_adjacent_edges,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from phase1_e6_affine_motion_residual_cv57 import (  # noqa: E402
    _fit_affine,
    _predict_affine,
)


def _node_um(node_id: int, t: int, x_um: float) -> GraphNode:
    return GraphNode(
        node_id,
        t,
        0.0,
        0.0,
        x_um / PHYSICAL_SCALE_ZYX[2],
    )


def _provenance() -> dict[str, object]:
    return {
        "schema": EXPERIMENTAL_SWEEP_SCHEMA,
        "experiment_id": "phase1-e3-unit-test",
        "purpose": "offline-cached-cv57-sweep",
        "varied_fields": ["motion_error_weight"],
        "production_eligible": False,
        "submit_eligible": False,
    }


class MotionCoefficientTests(unittest.TestCase):
    def test_weight_one_is_production_exact_and_zero_changes_assignment(self) -> None:
        nodes = [
            _node_um(0, 0, -6.0),
            _node_um(1, 0, 16.0),
            _node_um(10, 1, 2.0),
            _node_um(11, 1, 8.0),
            _node_um(20, 2, 4.0),
            _node_um(21, 2, 6.0),
        ]
        production, _ = build_adjacent_edges(nodes)
        experimental_one, stats_one = build_adjacent_edges(
            nodes,
            _experimental_sweep_provenance=_provenance(),
            _experimental_motion_error_weight=1.0,
        )
        experimental_zero, stats_zero = build_adjacent_edges(
            nodes,
            _experimental_sweep_provenance=_provenance(),
            _experimental_motion_error_weight=0.0,
        )
        self.assertEqual(production, experimental_one)
        self.assertEqual(stats_one["motion_error_weight"], 1.0)
        self.assertEqual(stats_zero["motion_error_weight"], 0.0)
        production_last = {
            (edge.source_id, edge.target_id)
            for edge in production
            if edge.source_id in {10, 11}
        }
        zero_last = {
            (edge.source_id, edge.target_id)
            for edge in experimental_zero
            if edge.source_id in {10, 11}
        }
        self.assertEqual(production_last, {(10, 21), (11, 20)})
        self.assertEqual(zero_last, {(10, 20), (11, 21)})

    def test_weight_override_requires_offline_provenance(self) -> None:
        with self.assertRaises(GraphCalibrationError):
            build_adjacent_edges(
                [_node_um(0, 0, 0.0), _node_um(1, 1, 1.0)],
                _experimental_motion_error_weight=0.0,
            )


class AffineFitTests(unittest.TestCase):
    def test_recovers_full_12_parameter_affine(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, -1.0, 3.0],
                [-2.0, 4.0, 1.0],
            ]
        )
        linear = np.asarray(
            [[1.1, 0.2, -0.1], [0.1, 0.9, 0.3], [-0.2, 0.4, 1.2]]
        )
        translation = np.asarray([3.0, -2.0, 1.5])
        target = source @ linear + translation
        center, coefficient, rank, condition = _fit_affine(source, target)
        self.assertEqual(rank, 4)
        self.assertTrue(np.isfinite(condition))
        observed = np.asarray(
            [_predict_affine(point, center, coefficient) for point in source]
        )
        np.testing.assert_allclose(observed, target, atol=1.0e-12)

    def test_rank_deficient_fit_fails(self) -> None:
        source = np.asarray([[float(index), 0.0, 0.0] for index in range(6)])
        with self.assertRaisesRegex(RuntimeError, "rank deficient"):
            _fit_affine(source, source + 1.0)


if __name__ == "__main__":
    unittest.main()
