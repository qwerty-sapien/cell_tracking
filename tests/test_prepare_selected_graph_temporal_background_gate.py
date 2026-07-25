"""Focused tests for exact temporal/background proposal preparation."""

from __future__ import annotations

import unittest

from scripts.prepare_selected_graph_temporal_background_gate import (
    SOURCE_CONFLICT,
    TARGET_CONFLICT,
    TemporalBackgroundPrepareError,
    conflict_kind,
)


class ConflictKindTests(unittest.TestCase):
    def test_source_conflict(self) -> None:
        self.assertEqual(conflict_kind((1, 2), (1, 3)), SOURCE_CONFLICT)

    def test_target_conflict(self) -> None:
        self.assertEqual(conflict_kind((1, 3), (2, 3)), TARGET_CONFLICT)

    def test_identical_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TemporalBackgroundPrepareError, "not a one-endpoint conflict"
        ):
            conflict_kind((1, 2), (1, 2))

    def test_unrelated_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TemporalBackgroundPrepareError, "not a one-endpoint conflict"
        ):
            conflict_kind((1, 2), (3, 4))


if __name__ == "__main__":
    unittest.main()
