from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.site_calibration import (  # noqa: E402
    fit_affine,
    fit_bias,
    residuals,
    select_calibration_model,
)


class SiteCalibrationTests(unittest.TestCase):
    def test_identity_is_preferred_when_within_tolerance(self) -> None:
        model = select_calibration_model([0.8, 0.9], [0.81, 0.89], tolerance=0.02)
        self.assertEqual(model.name, "identity")

    def test_bias_is_preferred_before_affine(self) -> None:
        model = select_calibration_model([0.7, 0.8, 0.9], [0.75, 0.85, 0.95])
        self.assertEqual(model.name, "bias")
        self.assertAlmostEqual(model.intercept, 0.05)

    def test_affine_fit_recovers_linear_mapping(self) -> None:
        model = fit_affine([0.6, 0.8, 1.0], [0.5, 0.8, 1.1])
        self.assertAlmostEqual(model.slope, 1.5)
        self.assertAlmostEqual(model.intercept, -0.4)
        self.assertTrue(all(abs(value) < 1e-12 for value in residuals(model, [0.6, 0.8, 1.0], [0.5, 0.8, 1.1])))

    def test_bias_requires_pairs(self) -> None:
        with self.assertRaises(ValueError):
            fit_bias([], [])


if __name__ == "__main__":
    unittest.main()
