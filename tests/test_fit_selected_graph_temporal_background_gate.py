"""Focused tests for grouped temporal/background proposal aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.fit_selected_graph_temporal_background_gate import (
    TemporalBackgroundFitError,
    proposal_confidence,
)


class ProposalConfidenceTests(unittest.TestCase):
    def test_min_conflict_max_ambiguous_option(self) -> None:
        value = proposal_confidence(
            np.asarray((0.2, 0.8, 0.6, 0.4)),
            (((0, 1), (2, 3)),),
        )
        np.testing.assert_allclose(value, (0.6,))

    def test_multiple_proposals_remain_separate(self) -> None:
        value = proposal_confidence(
            np.asarray((0.1, 0.9, 0.7)),
            (((0,),), ((1, 2),)),
        )
        np.testing.assert_allclose(value, (0.1, 0.9))

    def test_empty_conflict_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            TemporalBackgroundFitError, "group is empty"
        ):
            proposal_confidence(np.asarray((0.5,)), (((),),))


if __name__ == "__main__":
    unittest.main()
