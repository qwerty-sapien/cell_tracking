from __future__ import annotations

from decimal import Decimal
import unittest

from scripts import g04_division_permissive_sweep as MODULE


class TestPermissiveSweep(unittest.TestCase):
    def test_15_tp_15_fp_exact_combined_score(self) -> None:
        row = MODULE.edge_neutral_case(15, 15)
        self.assertEqual(row["predicted_forks"], 30)
        self.assertEqual(
            Decimal(row["division_jaccard_exact"]), Decimal(15) / Decimal(57)
        )
        self.assertEqual(
            Decimal(row["division_addition_exact"]), Decimal(1) / Decimal(38)
        )
        self.assertEqual(
            Decimal(row["combined_score_exact"]),
            Decimal("0.8815220699597315") + Decimal(1) / Decimal(38),
        )

    def test_banked_values_are_exact(self) -> None:
        for fp, expected in (
            (0, Decimal(25) / Decimal(420)),
            (5, Decimal(25) / Decimal(470)),
            (10, Decimal(25) / Decimal(520)),
        ):
            with self.subTest(fp=fp):
                row = MODULE.edge_neutral_case(25, fp)
                self.assertEqual(Decimal(row["division_addition_exact"]), expected)

    def test_oracle_best_curve_peaks_at_42_predictions(self) -> None:
        before = Decimal(MODULE.oracle_best_for_predicted_count(41)["combined_score_exact"])
        peak = Decimal(MODULE.oracle_best_for_predicted_count(42)["combined_score_exact"])
        after = Decimal(MODULE.oracle_best_for_predicted_count(43)["combined_score_exact"])
        self.assertLess(before, peak)
        self.assertLess(after, peak)
        self.assertEqual(MODULE.oracle_best_for_predicted_count(43)["tp"], 42)
        self.assertEqual(MODULE.oracle_best_for_predicted_count(43)["fp"], 1)

    def test_fixed_adjusted_edge_tp38_plus_0p005_limit(self) -> None:
        threshold = Decimal("0.8815220699597315") + Decimal("0.005")
        self.assertGreaterEqual(
            Decimal(MODULE.edge_neutral_case(38, 718)["combined_score_exact"]),
            threshold,
        )
        self.assertLess(
            Decimal(MODULE.edge_neutral_case(38, 719)["combined_score_exact"]),
            threshold,
        )

    def test_invalid_counts_fail_closed(self) -> None:
        with self.assertRaises(MODULE.PermissiveSweepError):
            MODULE.edge_neutral_case(43, 0)
        with self.assertRaises(MODULE.PermissiveSweepError):
            MODULE.edge_neutral_case(1, -1)

    def test_nondominated_frontier_keeps_high_tp_high_fp_tradeoff(self) -> None:
        rows = [
            {
                "config_name": "a",
                "predicted_forks": 10,
                "tp": 2,
                "fp": 8,
                "division_jaccard": 0.04,
                "combined_score": 0.80,
            },
            {
                "config_name": "dominated",
                "predicted_forks": 11,
                "tp": 1,
                "fp": 10,
                "division_jaccard": 0.02,
                "combined_score": 0.79,
            },
            {
                "config_name": "permissive",
                "predicted_forks": 30,
                "tp": 5,
                "fp": 25,
                "division_jaccard": 0.08,
                "combined_score": 0.81,
            },
        ]
        names = {row["config_name"] for row in MODULE.nondominated_frontier(rows)}
        self.assertEqual(names, {"a", "permissive"})

    def test_default_panel_explicitly_contains_large_cap_and_all(self) -> None:
        panel = MODULE.default_permissive_panel()
        self.assertIn("t035_g100_l0_dp0_gp0_n0_c08", panel)
        self.assertIn("t035_g100_l0_dp0_gp0_n0_call", panel)
        self.assertEqual(len(panel), len(set(panel)))

    def test_cv57_cache_panel_reaches_unfiltered_cap_eight(self) -> None:
        panel = MODULE.cv57_cache_panel()
        names = {row["name"] for row in panel}
        self.assertEqual(len(panel), 20)
        self.assertIn("unfiltered_mass_cap8", names)
        self.assertIn("unfiltered_geometry_cap8", names)
        unfiltered = next(row for row in panel if row["name"] == "unfiltered_mass_cap8")
        self.assertIsNone(unfiltered["ratio_max"])

    def test_appearance_features_match_explicit_mass_formula(self) -> None:
        candidate = {
            "features": {"geometry_cost": 1.25},
            "intensity_summaries": {
                "daughter_1": {"mass_above_background": 30.0},
                "daughter_2": {"mass_above_background": 20.0},
                "parent": {"mass_above_background": 40.0, "xy_roundness": 0.75},
            },
        }
        features = MODULE.appearance_features(candidate)
        self.assertAlmostEqual(features["daughter_mass_balance"], 2.0 / 3.0)
        self.assertAlmostEqual(features["daughter_parent_mass_ratio"], 1.25)
        self.assertAlmostEqual(features["parent_xy_roundness"], 0.75)


if __name__ == "__main__":
    unittest.main()
