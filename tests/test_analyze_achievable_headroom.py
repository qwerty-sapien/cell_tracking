from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_achievable_headroom import (  # noqa: E402
    AchievableHeadroomError,
    assert_safe_fit_only_path,
    precision_recall_curve,
    select_p95_operating_point,
)
from celltrack_alpha.site_calibration import (  # noqa: E402
    fit_platt_calibration,
    fit_temperature_calibration,
)


class AchievableHeadroomTests(unittest.TestCase):
    def test_binary_calibrators_are_monotone_increasing(self) -> None:
        logits = np.asarray((-3.0, -1.0, -0.2, 0.4, 1.2, 3.0))
        labels = np.asarray((0, 0, 0, 1, 1, 1))
        probe = np.linspace(-8.0, 8.0, 1001)
        for calibrator in (
            fit_temperature_calibration(logits, labels),
            fit_platt_calibration(logits, labels),
        ):
            probabilities = calibrator.predict_probabilities(probe)
            self.assertGreater(calibrator.slope, 0.0)
            self.assertTrue(np.all(np.diff(probabilities) >= 0.0))

    def test_p95_counting_uses_net_correct_swaps(self) -> None:
        scored = [
            {
                "identity": "train:44b6_fixture",
                "selected_edges": 100,
                "labeled_confidence": np.asarray((0.90, 0.80, 0.70)),
                "labels": np.asarray((1, 1, 0), dtype=np.int8),
                "background_confidence": np.asarray((0.75,)),
            }
        ]
        curve = precision_recall_curve(scored, sampled_points=257)
        selected = select_p95_operating_point(curve, minimum_precision=0.95)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(float(selected["threshold"]), 0.80)
        self.assertEqual(selected["correct_swaps"], 2)
        self.assertEqual(selected["harmful_swaps"], 0)
        self.assertEqual(selected["net_correct_swaps"], 2)
        self.assertAlmostEqual(float(selected["precision"]), 1.0)
        self.assertAlmostEqual(float(selected["recall"]), 1.0)
        self.assertEqual(selected["background_accepted_unlabeled"], 0)

    def test_holdout_guard_fails_before_read(self) -> None:
        for path in (
            Path("reports/g02/cv57/report.json"),
            Path("reports/g02/holdout/scores.json"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    AchievableHeadroomError, "forbidden"
                ):
                    assert_safe_fit_only_path(path)


if __name__ == "__main__":
    unittest.main()
