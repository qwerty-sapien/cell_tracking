import importlib.util
import json
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_fidelity.py")
SPEC = importlib.util.spec_from_file_location("validate_fidelity", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EstimatorTests(unittest.TestCase):
    def test_design_weighted_official_score(self):
        rows = [
            {
                "edge_tp": 8, "edge_fp": 1, "edge_fn": 1,
                "adj_edge_jaccard": 0.72,
                "division_tp": 1, "division_fp": 0, "division_fn": 1,
            },
            {
                "edge_tp": 15, "edge_fp": 3, "edge_fn": 2,
                "adj_edge_jaccard": 0.64,
                "division_tp": 2, "division_fp": 1, "division_fn": 0,
            },
        ]
        observed = MODULE.score_from_rows(rows, [3.0, 2.0])
        expected_adj = (3 * 10 * 0.72 + 2 * 20 * 0.64) / (3 * 10 + 2 * 20)
        expected_div = (3 * 1 + 2 * 2) / (
            3 * (1 + 0 + 1) + 2 * (2 + 1 + 0)
        )
        self.assertAlmostEqual(observed, expected_adj + 0.1 * expected_div)

    def test_division_term_drops_at_zero_denominator(self):
        row = {
            "edge_tp": 8, "edge_fp": 1, "edge_fn": 1,
            "adj_edge_jaccard": 0.72,
            "division_tp": 0, "division_fp": 0, "division_fn": 0,
        }
        self.assertEqual(MODULE.score_from_rows([row]), 0.72)


class BankedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("/Users/agent/Desktop/cell_tracking")
        cls.candidates, cls.universe = MODULE.audit_sources(cls.root)
        cls.by_id = {
            candidate["candidate_id"]: candidate for candidate in cls.candidates
        }

    def test_all_six_candidates_explicit(self):
        self.assertEqual(
            set(self.by_id),
            {"APNCC", "C1", "H1-ZWRFEAT", "SA-DELETE", "SA-SWAP", "NODETOPO"},
        )

    def test_three_exact_reconstructions(self):
        for candidate_id in ("APNCC", "SA-DELETE", "NODETOPO"):
            reconstruction = self.by_id[candidate_id]["reconstruction"]
            self.assertTrue(reconstruction["passes"], candidate_id)
            self.assertLessEqual(reconstruction["delta_abs_error"], 1e-12)

    def test_three_missing_per_video_sufficient_statistics(self):
        for candidate_id in ("C1", "H1-ZWRFEAT", "SA-SWAP"):
            self.assertFalse(
                self.by_id[candidate_id]["per_video_sufficient_statistics_banked"]
            )

    def test_universes_fail_closed(self):
        self.assertFalse(self.universe["all_six_same_universe"])
        self.assertTrue(
            self.universe["apncc_zero_overlap_with_all_44b6_gate_universes"]
        )
        self.assertTrue(self.universe["c1_h1_swap_nodetopo_same_36_ids"])
        self.assertTrue(self.universe["sa_delete_is_exact_36_plus_five"])
        self.assertEqual(len(self.universe["sa_delete_extra_ids"]), 5)
        self.assertEqual(self.universe["common_eligible_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
