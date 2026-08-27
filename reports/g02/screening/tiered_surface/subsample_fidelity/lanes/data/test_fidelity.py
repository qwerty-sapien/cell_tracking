#!/usr/bin/env python3
"""Focused deterministic tests for the banked-statistics fidelity estimator."""

from __future__ import annotations

import math
import unittest

import numpy as np

from run_fidelity import (
    allocate_largest_remainder,
    average_ranks,
    pair_seed,
    score_rows,
    spearman,
)


def row(
    *,
    tp: float,
    fp: float,
    fn: float,
    adj: float,
    dtp: float = 0,
    dfp: float = 0,
    dfn: float = 0,
) -> dict[str, float]:
    return {
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "adj_edge_jaccard": adj,
        "division_tp": dtp,
        "division_fp": dfp,
        "division_fn": dfn,
    }


class EstimatorTests(unittest.TestCase):
    def test_official_adjusted_weighting(self) -> None:
        rows = [row(tp=8, fp=1, fn=1, adj=0.8), row(tp=1, fp=0, fn=0, adj=1.0)]
        self.assertAlmostEqual(score_rows(rows), (10 * 0.8 + 1 * 1.0) / 11)

    def test_design_weight_expansion_and_division(self) -> None:
        rows = [
            row(tp=8, fp=1, fn=1, adj=0.8, dtp=1, dfn=1),
            row(tp=1, fp=0, fn=0, adj=1.0, dtp=1),
        ]
        observed = score_rows(rows, np.asarray([2.0, 1.0]))
        expected_adjusted = (2 * 10 * 0.8 + 1 * 1 * 1.0) / 21
        expected_division = (2 * 1 + 1 * 1) / (2 * 2 + 1 * 1)
        self.assertAlmostEqual(observed, expected_adjusted + 0.1 * expected_division)

    def test_largest_remainder_is_deterministic_and_capacity_safe(self) -> None:
        observed = allocate_largest_remainder({"A": 10, "B": 26}, 5)
        self.assertEqual(observed, {"A": 1, "B": 4})
        self.assertEqual(sum(observed.values()), 5)
        self.assertIsNone(allocate_largest_remainder({"A": 10, "B": 26}, 1))

    def test_average_ranks_ties(self) -> None:
        np.testing.assert_allclose(
            average_ranks(np.asarray([3.0, 1.0, 1.0, 2.0])),
            np.asarray([4.0, 1.5, 1.5, 3.0]),
        )

    def test_spearman_direction(self) -> None:
        self.assertTrue(
            math.isclose(
                spearman(
                    np.asarray([1.0, 2.0, 3.0]),
                    np.asarray([3.0, 2.0, 1.0]),
                ),
                -1.0,
            )
        )

    def test_pair_seed_frozen(self) -> None:
        self.assertEqual(pair_seed(3, "simple_random"), 6788270665500927273)
        self.assertNotEqual(
            pair_seed(3, "simple_random"), pair_seed(5, "simple_random")
        )


if __name__ == "__main__":
    unittest.main()
