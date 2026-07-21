from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_fit_node_threshold import harden_report  # noqa: E402


def threshold_row(threshold: float, score: float, recall: float) -> dict:
    return {
        "threshold": threshold,
        "adj_edge_jaccard": score,
        "edge_jaccard": score,
        "node_recall": recall,
        "estimated_number_of_nodes": 100.0,
        "num_pred_nodes": 100.0,
    }


class FitNodeThresholdTests(unittest.TestCase):
    def test_hardened_policy_rejects_mean_safe_but_per_video_unsafe_threshold(self) -> None:
        report = {
            "schema": "celltrack-alpha.fit-node-threshold.v1",
            "scope": {
                "fit_only": True,
                "held_out_used": False,
                "audit_used": False,
                "test_used": False,
            },
            "method": {"thresholds": [0.97, 0.995]},
            "samples": [
                {
                    "identity": "train:44b6_a",
                    "scenario": "44b6",
                    "thresholds": [
                        threshold_row(0.97, 0.80, 0.99),
                        threshold_row(0.995, 0.82, 0.99),
                    ],
                },
                {
                    "identity": "train:6bba_a",
                    "scenario": "6bba",
                    "thresholds": [
                        threshold_row(0.97, 0.80, 0.99),
                        threshold_row(0.995, 0.90, 0.97),
                    ],
                },
            ],
        }
        hardened = harden_report(report, recall_tolerance=0.002)
        self.assertEqual(hardened["schema"], "celltrack-alpha.fit-node-threshold.v2")
        self.assertEqual(hardened["policy"]["44b6"]["selected_threshold"], 0.995)
        self.assertEqual(hardened["policy"]["6bba"]["selected_threshold"], 0.97)
        self.assertEqual(hardened["policy"]["all"]["selected_threshold"], 0.97)
        unsafe = hardened["aggregate"]["6bba"][1]
        self.assertEqual(unsafe["videos_exceeding_recall_loss"], 1)


if __name__ == "__main__":
    unittest.main()
