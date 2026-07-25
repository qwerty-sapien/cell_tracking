"""Focused tests for shell appearance-support threshold selection."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.audit_extended_radius_shell_appearance_support import (
    ShellSupportError,
    select_threshold,
)


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_high_precision_zero_unknown_prefix(self) -> None:
        result = select_threshold(
            np.asarray((0.9, 0.8, 0.7, 0.6)),
            np.asarray((1, 1, -1, 0)),
            np.ones(4, dtype=bool),
            minimum_correct=2,
            minimum_precision=0.95,
            maximum_unknown=0,
        )
        self.assertTrue(result["admissible"])
        self.assertEqual(result["selected"]["correct_links"], 2)
        self.assertEqual(result["selected"]["accepted"], 2)

    def test_reports_best_supported_diagnostic_when_inadmissible(self) -> None:
        result = select_threshold(
            np.asarray((0.9, 0.85, 0.8, 0.7)),
            np.asarray((-1, 1, 1, 0)),
            np.ones(4, dtype=bool),
            minimum_correct=2,
            minimum_precision=0.95,
            maximum_unknown=0,
        )
        self.assertFalse(result["admissible"])
        self.assertEqual(result["selected"]["correct_links"], 2)
        self.assertEqual(result["selected"]["sampled_unknown_background"], 1)

    def test_eligibility_mask_is_enforced(self) -> None:
        result = select_threshold(
            np.asarray((0.9, 0.8, 0.7)),
            np.asarray((-1, 1, 1)),
            np.asarray((False, True, True)),
            minimum_correct=2,
            minimum_precision=0.95,
            maximum_unknown=0,
        )
        self.assertTrue(result["admissible"])

    def test_invalid_shapes_fail_closed(self) -> None:
        with self.assertRaisesRegex(ShellSupportError, "arrays are invalid"):
            select_threshold(
                np.asarray((0.9,)),
                np.asarray((1, 0)),
                np.ones(1, dtype=bool),
                minimum_correct=1,
                minimum_precision=0.95,
                maximum_unknown=0,
            )


if __name__ == "__main__":
    unittest.main()
