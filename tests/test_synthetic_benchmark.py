from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.synthetic import (  # noqa: E402
    EXPECTED_EDGE_DIGEST_SHA256,
    EXPECTED_FIXTURE_DIGEST_SHA256,
    SYNTHETIC_SCALE_ZYX,
    _benchmark_config,
    assert_synthetic_benchmark,
    make_synthetic_4d_case,
    run_synthetic_benchmark,
)
from celltrack_alpha.solver import build_edge_candidates, select_edges  # noqa: E402


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_fixed_detection_4d_benchmark_meets_software_gate(self) -> None:
        result = run_synthetic_benchmark(repetitions=3)

        assert_synthetic_benchmark(result)
        self.assertEqual(result["scope"], "fixed detections; segmentation bypassed")
        self.assertEqual(result["shape_tzyx"], [6, 3, 16, 16])
        self.assertEqual(result["edge_precision"], 1.0)
        self.assertEqual(result["edge_recall"], 1.0)
        self.assertTrue(result["deterministic"])
        self.assertEqual(result["fixture_digest_sha256"], EXPECTED_FIXTURE_DIGEST_SHA256)
        self.assertEqual(result["edge_digest_sha256"], EXPECTED_EDGE_DIGEST_SHA256)

    def test_anisotropic_z_scale_changes_near_gate_assignment(self) -> None:
        _, nodes_by_t, truth_edges = make_synthetic_4d_case()
        cfg = _benchmark_config()

        correct_candidates = build_edge_candidates(nodes_by_t, SYNTHETIC_SCALE_ZYX, cfg)
        correct, _ = select_edges(nodes_by_t, correct_candidates, SYNTHETIC_SCALE_ZYX, cfg)
        wrong_candidates = build_edge_candidates(nodes_by_t, (1.0, 1.0, 1.0), cfg)
        wrong, _ = select_edges(nodes_by_t, wrong_candidates, (1.0, 1.0, 1.0), cfg)

        self.assertEqual({(edge.source_id, edge.target_id) for edge in correct}, truth_edges)
        self.assertNotEqual({(edge.source_id, edge.target_id) for edge in wrong}, truth_edges)

    def test_invalid_repetition_count_is_rejected(self) -> None:
        for repetitions in (0, -1, True, 1.5):
            with self.subTest(repetitions=repetitions):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    run_synthetic_benchmark(repetitions)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
