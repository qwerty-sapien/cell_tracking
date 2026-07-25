"""Focused runtime tests for temporal family swaps."""

from __future__ import annotations

import unittest

from celltrack_alpha.temporal_rank_swap import (
    TemporalRankSwapError,
    _kind,
)
from celltrack_alpha.edge_data import SOURCE_CONFLICT, TARGET_CONFLICT


class ConflictKindTests(unittest.TestCase):
    def test_kinds(self) -> None:
        self.assertEqual(_kind((1, 2), (1, 3)), SOURCE_CONFLICT)
        self.assertEqual(_kind((1, 3), (2, 3)), TARGET_CONFLICT)

    def test_invalid_conflict_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            TemporalRankSwapError, "one-endpoint conflict"
        ):
            _kind((1, 2), (3, 4))


if __name__ == "__main__":
    unittest.main()
