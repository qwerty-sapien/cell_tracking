import math
import unittest

from scripts.analyze_h0_proxy_validity import (
    PRIMARY_CANDIDATES,
    REPEATED_OUTCOMES,
    SEED,
    analyze_rows,
    average_ranks,
    bootstrap_ci,
    build_report,
    decide,
    kendall_tau_b,
    pearson,
    sign_agreement,
    spearman,
)


class CorrelationLogicTest(unittest.TestCase):
    def test_average_ranks_are_tie_aware(self):
        self.assertEqual(average_ranks([10, 20, 20, 40]), [1.0, 2.5, 2.5, 4.0])

    def test_correlations_on_monotone_and_reversed_inputs(self):
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(pearson(x, x), 1.0)
        self.assertAlmostEqual(spearman(x, list(reversed(x))), -1.0)
        self.assertAlmostEqual(kendall_tau_b(x, list(reversed(x))), -1.0)

    def test_constant_input_is_not_fabricated_as_zero_correlation(self):
        self.assertIsNone(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_bootstrap_is_deterministic_and_records_valid_replicates(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 1.0, 3.0, 2.0]
        first = bootstrap_ci(x, y, spearman, seed=SEED, replicates=500)
        second = bootstrap_ci(x, y, spearman, seed=SEED, replicates=500)
        self.assertEqual(first, second)
        self.assertGreater(first["valid_replicates"], 0)
        self.assertLessEqual(first["valid_replicates"], 500)

    def test_sign_agreement_uses_graph_delta_sign(self):
        self.assertEqual(sign_agreement([True, True, False], [0.1, -0.2, -0.3]), 2 / 3)


class DecisionLogicTest(unittest.TestCase):
    def test_primary_corpus_is_candidate_level(self):
        self.assertEqual(len(PRIMARY_CANDIDATES), 4)
        self.assertTrue(all(row["unit"] == "candidate" for row in PRIMARY_CANDIDATES))
        self.assertEqual(
            len({row["mechanism_cluster"] for row in PRIMARY_CANDIDATES}),
            4,
        )

    def test_preregistered_decision_uses_spearman_and_bans(self):
        report = build_report()
        self.assertEqual(report["scope"]["rho_for_preregistered_decision"], "Spearman rank correlation")
        self.assertEqual(report["decision"]["verdict"], "PROPOSAL_ONLY_SCREENING_BANNED")
        self.assertEqual(report["decision"]["clearing_proxies"], [])
        for proxy in ("precision", "support"):
            rho = report["primary_analysis"][proxy]["spearman_rho"]["estimate"]
            self.assertLess(abs(rho), 0.5)

    def test_raw_scorer_outcome_is_separate_and_does_not_change_verdict(self):
        report = build_report()
        raw = report["raw_outcome_analysis"]
        self.assertEqual(raw["available_candidate_count"], 4)
        self.assertEqual(raw["unavailable_candidates"], [])
        self.assertIn("TP/(TP+FP+FN)", raw["definition"])
        for proxy in ("precision", "support"):
            rho = raw["analysis"][proxy]["spearman_rho"]["estimate"]
            self.assertLess(abs(rho), 0.5)
        self.assertEqual(report["decision"]["verdict"], "PROPOSAL_ONLY_SCREENING_BANNED")

    def test_raw_deltas_match_exact_banked_confusion_derivations(self):
        for candidate in PRIMARY_CANDIDATES + REPEATED_OUTCOMES:
            if "graph_raw_derivation" not in candidate:
                continue
            derivation = candidate["graph_raw_derivation"]
            control = derivation["control_tp_fp_fn"]
            challenger = derivation["candidate_tp_fp_fn"]
            expected = challenger[0] / sum(challenger) - control[0] / sum(control)
            self.assertAlmostEqual(candidate["graph_raw_delta"], expected, places=15)

    def test_complete_inventory_clusters_repeated_outcomes(self):
        report = build_report()
        inventory = report["complete_outcome_inventory"]
        self.assertEqual(inventory["row_count"], 8)
        self.assertEqual(inventory["mechanism_cluster_count"], 4)
        outcome_ids = {row["outcome_id"] for row in inventory["rows"]}
        self.assertIn("GA-TEMPHYBRID-20260724/full-cv57", outcome_ids)
        self.assertIn("GA-APNCC-GLOBAL-20260724/full-cv57", outcome_ids)
        self.assertIn("GA-APNCC-GLOBAL-20260724/top10", outcome_ids)
        self.assertIn("GA-APNCC-DIRECT-20260724/top10", outcome_ids)

    def test_complete_row_crossing_does_not_override_cluster_primary(self):
        report = build_report()
        complete = report["complete_outcome_inventory"]["adjusted_analysis"]
        self.assertGreater(abs(complete["precision"]["spearman_rho"]["estimate"]), 0.5)
        self.assertFalse(complete["precision"]["eligible_for_preregistered_decision"])
        self.assertEqual(complete["precision"]["cluster_n"], 4)
        self.assertEqual(report["decision"]["verdict"], "PROPOSAL_ONLY_SCREENING_BANNED")

    def test_insufficient_metrics_cannot_clear_gate(self):
        analysis = analyze_rows(PRIMARY_CANDIDATES)
        self.assertEqual(analysis["utility_gain"]["status"], "INSUFFICIENT_N")
        self.assertEqual(analysis["auc"]["n"], 0)
        self.assertEqual(decide(analysis)["verdict"], "PROPOSAL_ONLY_SCREENING_BANNED")

    def test_fold_sensitivity_cannot_enter_decision(self):
        report = build_report()
        sensitivity = report["temporal_fold_sensitivity"]["analysis"]
        self.assertTrue(
            math.isclose(
                sensitivity["utility_gain"]["spearman_rho"]["estimate"],
                0.6,
                abs_tol=1e-12,
            )
        )
        self.assertFalse(sensitivity["utility_gain"]["eligible_for_preregistered_decision"])
        raw_sensitivity = report["temporal_fold_sensitivity"]["raw_analysis"]
        self.assertFalse(raw_sensitivity["utility_gain"]["eligible_for_preregistered_decision"])


if __name__ == "__main__":
    unittest.main()
