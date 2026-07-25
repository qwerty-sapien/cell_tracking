"""Focused tests for joint capped temporal proposal ranking."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.fit_selected_graph_temporal_rank_gate import capped_video_metrics


class CappedVideoMetricsTests(unittest.TestCase):
    def test_background_competes_for_same_cap(self) -> None:
        value = capped_video_metrics(
            np.asarray((0.9, 0.8)),
            np.asarray((1, 0)),
            np.asarray((0.95,)),
            threshold=0.0,
            budget=2,
        )
        self.assertEqual(value["correct_rescues"], 1)
        self.assertEqual(value["harmful_swaps"], 0)
        self.assertEqual(value["background_accepted"], 1)

    def test_background_wins_exact_tie_conservatively(self) -> None:
        value = capped_video_metrics(
            np.asarray((0.9,)),
            np.asarray((1,)),
            np.asarray((0.9,)),
            threshold=0.0,
            budget=1,
        )
        self.assertEqual(value["correct_rescues"], 0)
        self.assertEqual(value["background_accepted"], 1)

    def test_threshold_precedes_cap(self) -> None:
        value = capped_video_metrics(
            np.asarray((0.9, 0.7)),
            np.asarray((1, 1)),
            np.asarray((0.6,)),
            threshold=0.8,
            budget=3,
        )
        self.assertEqual(value["selected"], 1)
        self.assertEqual(value["correct_rescues"], 1)


if __name__ == "__main__":
    unittest.main()
