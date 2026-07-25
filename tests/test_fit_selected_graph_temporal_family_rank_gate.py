"""Focused tests for family-tagged temporal ranking."""

from __future__ import annotations

import unittest

from scripts.fit_selected_graph_temporal_family_rank_gate import (
    TemporalFamilyRankError,
    family,
)


class FamilyTests(unittest.TestCase):
    def test_known_families(self) -> None:
        self.assertEqual(family("train:44b6_deadbeef"), "44b6")
        self.assertEqual(family("train:6bba_deadbeef"), "6bba")

    def test_unknown_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(TemporalFamilyRankError, "unknown"):
            family("train:ffff_deadbeef")

    def test_invalid_identity_fails_closed(self) -> None:
        with self.assertRaises(TemporalFamilyRankError):
            family("malformed")


if __name__ == "__main__":
    unittest.main()
