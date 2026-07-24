from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from fit_motion_gated_shell_recall import (  # noqa: E402
    MotionGatedShellError,
    _load_json,
    motion_gate_mask,
    precision_counts,
    recall_accounting,
)


class MotionGatedShellRecallTest(unittest.TestCase):
    def test_motion_gate_reduces_candidates(self) -> None:
        candidate = np.asarray(
            ((1.1, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        )
        predicted = np.asarray(
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        )
        history = np.asarray((True, True, False))
        retained = motion_gate_mask(
            candidate, predicted, history, tolerance_um=0.25
        )
        np.testing.assert_array_equal(retained, (True, False, False))
        self.assertLess(int(retained.sum()), len(candidate))

    def test_recall_accounting(self) -> None:
        metrics = recall_accounting(
            np.asarray((1, 1, 0, -1), dtype=np.int8),
            np.asarray((True, False, True, True)),
            expected_positive_total=2,
        )
        self.assertEqual(metrics["positive_retained"], 1)
        self.assertEqual(metrics["positive_removed"], 1)
        self.assertEqual(metrics["recall"], 0.5)

    def test_holdout_guard_fails_before_read(self) -> None:
        with self.assertRaisesRegex(MotionGatedShellError, "forbidden"):
            _load_json(Path("/definitely/missing/holdout/report.json"))
        with self.assertRaisesRegex(MotionGatedShellError, "forbidden"):
            _load_json(Path("/definitely/missing/CV57/report.json"))

    def test_precision_counting(self) -> None:
        metrics = precision_counts(
            np.asarray((1, 1, 0, -1), dtype=np.int8),
            np.asarray((True, True, True, True)),
        )
        self.assertEqual(metrics["correct_edges"], 2)
        self.assertEqual(metrics["known_negative_edges"], 1)
        self.assertEqual(metrics["accepted_unlabeled"], 1)
        self.assertEqual(metrics["fit_precision"], 0.5)
        self.assertEqual(metrics["definite_precision"], 2 / 3)
        self.assertEqual(metrics["capped_utility_delta"], -2)


if __name__ == "__main__":
    unittest.main()
