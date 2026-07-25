"""Focused tests for r10--12 um shell appearance augmentation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.augment_extended_radius_shell_appearance import (
    ShellAppearanceError,
    appearance_row_features,
)


class AppearanceRowFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = {
            1: np.asarray((1.0, 0.0), dtype=np.float32),
            2: np.asarray((0.8, 0.6), dtype=np.float32),
            3: np.asarray((0.0, 1.0), dtype=np.float32),
            4: np.asarray((0.6, 0.8), dtype=np.float32),
        }

    def test_free_pair_uses_absolute_ncc_as_margin(self) -> None:
        value = appearance_row_features((1, 2), (), self.vectors)
        np.testing.assert_allclose(value, (0.8, 0.0, 0.0, 0.8, 0.0, 0.0))

    def test_two_conflicts_use_strictest_incumbent(self) -> None:
        value = appearance_row_features(
            (1, 2), ((1, 3), (4, 2)), self.vectors
        )
        self.assertAlmostEqual(float(value[0]), 0.8)
        self.assertEqual(float(value[1]), 1.0)
        self.assertAlmostEqual(float(value[2]), 0.96)
        self.assertAlmostEqual(float(value[3]), -0.16)
        self.assertAlmostEqual(float(value[4]), 0.0)
        self.assertAlmostEqual(float(value[5]), 0.96)

    def test_missing_vector_fails_closed(self) -> None:
        with self.assertRaisesRegex(ShellAppearanceError, "missing pair vector"):
            appearance_row_features((1, 99), (), self.vectors)

    def test_more_than_two_conflicts_is_rejected(self) -> None:
        with self.assertRaisesRegex(ShellAppearanceError, "invalid pair/conflict"):
            appearance_row_features(
                (1, 2), ((1, 3), (4, 2), (1, 2)), self.vectors
            )


if __name__ == "__main__":
    unittest.main()
