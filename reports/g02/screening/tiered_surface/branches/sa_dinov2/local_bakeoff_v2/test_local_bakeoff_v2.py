from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import constants
import aggregate_evaluate
import resource_guard
import run_class


class LocalBakeoffRepairTests(unittest.TestCase):
    def test_all_frozen_pins_are_exact(self):
        observed = constants.verify_pins()
        self.assertEqual(len(observed), len(constants.PINS))

    def test_contract_is_exactly_two_classes(self):
        self.assertEqual(constants.CLASSES, ("delete", "occupied_swap"))
        self.assertEqual(constants.EXPECTED_ACTIONS["delete"], 1_345_497)
        self.assertEqual(constants.EXPECTED_ACTIONS["occupied_swap"], 45_961)

    def test_requirement_surface_is_run_v6(self):
        self.assertEqual(
            constants.PINS[constants.REQUIREMENT_SURFACE],
            "6e62a29ce3effdd2a3748716a9843f10f5f5954a322d10547f7a4cafc4d6c71b",
        )
        self.assertEqual(constants.REQUIREMENTS["delete"]["mean_auc"], 0.85)
        self.assertEqual(constants.REQUIREMENTS["occupied_swap"]["robust_p05_auc"], 0.95)
        self.assertEqual(
            constants.REQUIREMENTS["delete"]["mean_retention"],
            0.02039355992844365,
        )
        self.assertEqual(
            constants.REQUIREMENTS["occupied_swap"]["robust_p05_retention"],
            0.14738805970149255,
        )

    def test_requirement_cross_read_uses_mean_and_empirical_p05(self):
        result = aggregate_evaluate.measured_requirement_cross_read(
            [0.84, 0.90, 0.92, 0.93, 0.94],
            [0.01, 0.03, 0.04, 0.05, 0.06],
            {
                "mean_auc": 0.85,
                "mean_retention": 0.02,
                "robust_p05_auc": 0.85,
                "robust_p05_retention": 0.015,
            },
        )
        observed = result["observed_A1_across_folds"]
        self.assertAlmostEqual(observed["fold_mean_auc"], 0.906)
        self.assertAlmostEqual(observed["fold_empirical_p05_auc"], 0.852)
        self.assertAlmostEqual(observed["fold_mean_retention"], 0.038)
        self.assertAlmostEqual(observed["fold_empirical_p05_retention"], 0.014)
        self.assertTrue(result["checks"]["mean_clears_measured_requirements"])
        self.assertFalse(result["checks"]["p05_clears_robust_requirements"])
        self.assertNotIn("all_folds_clear", result["checks"])

    def test_preflight_rejects_less_than_45_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(resource_guard.GuardError):
                resource_guard.preflight(Path(directory) / "preflight.json", 44.99)

    def test_guard_rejects_stale_or_nonzero_swap_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(
                '{"status":"FAIL_SWAP_GROWTH","swap_growth_mib":1,'
                '"elapsed_seconds":45}\n',
                encoding="ascii",
            )
            with self.assertRaises(resource_guard.GuardError):
                resource_guard.run_guarded(path, Path(directory) / "guard.json", ["true"])

    def test_class_plan_never_reads_sibling_class(self):
        class FakeMaterializer:
            @staticmethod
            def load_request(path):
                self.assertEqual(path.parent.name, "delete")
                return {
                    "source_t": np.asarray([0]),
                    "source_zyx": np.asarray([[1, 2, 3]]),
                    "target_t": np.asarray([1]),
                    "target_zyx": np.asarray([[4, 5, 6]]),
                }

            @staticmethod
            def endpoint_rows(request, role):
                return np.column_stack((request[f"{role}_t"], request[f"{role}_zyx"]))

        with mock.patch.object(Path, "glob", return_value=[]):
            rows, endpoints = run_class.class_plan(FakeMaterializer(), "delete")
        self.assertEqual(rows, [])
        self.assertEqual(endpoints, 0)

    def test_base_evaluator_is_loadable_by_hash(self):
        spec = importlib.util.spec_from_file_location("frozen_eval_test", constants.EVALUATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)


if __name__ == "__main__":
    unittest.main()
