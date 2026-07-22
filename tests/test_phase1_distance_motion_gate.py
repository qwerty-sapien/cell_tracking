from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_phase1_distance_motion_gate as diagnostic  # noqa: E402


class DistanceRankTests(unittest.TestCase):
    def test_rank_uses_physical_distance(self) -> None:
        rank, distance = diagnostic.distance_rank(
            (0.0, 0.0, 0.0),
            20,
            np.asarray([10, 20, 30]),
            np.asarray([(1, 0, 0), (0, 3, 0), (0, 1, 0)]),
            (2.0, 1.0, 1.0),
        )
        self.assertEqual(rank, 3)
        self.assertEqual(distance, 3.0)

    def test_rank_tie_break_matches_gate_target_id_then_input(self) -> None:
        rank, _ = diagnostic.distance_rank(
            (0.0, 0.0, 0.0),
            20,
            np.asarray([30, 20, 10]),
            np.asarray([(0, 1, 0), (0, 1, 0), (0, 1, 0)]),
            (1.0, 1.0, 1.0),
        )
        self.assertEqual(rank, 2)


class MotionPredictionTests(unittest.TestCase):
    def test_no_parent_predicts_source_exactly(self) -> None:
        result = diagnostic.motion_predicted_coord((2, 4, 6), None, 0.5)
        np.testing.assert_array_equal(result, np.asarray((2.0, 4.0, 6.0)))

    def test_parent_uses_locked_half_velocity(self) -> None:
        result = diagnostic.motion_predicted_coord((2, 4, 6), (0, 2, 4), 0.5)
        np.testing.assert_array_equal(result, np.asarray((3.0, 5.0, 7.0)))


class DecisionRuleTests(unittest.TestCase):
    def event(
        self,
        *,
        rank: int,
        is_fn: bool = False,
        above: bool = False,
        predicted_distance: float = 0.0,
        unmatched_source: bool = True,
    ) -> dict[str, object]:
        return {
            "distance_rank": rank,
            "is_fn": is_fn,
            "above_10um_fn": above,
            "motion_predicted_distance_um": predicted_distance,
            "motion_history_present": True,
            "source_unmatched_after_current_adjacent_assignment": unmatched_source,
        }

    def test_e1_requires_both_overall_and_above_gate_at_ninety_percent(self) -> None:
        events = [self.event(rank=1) for _ in range(90)] + [
            self.event(rank=11) for _ in range(10)
        ]
        events += [
            self.event(rank=1, is_fn=True, above=True, predicted_distance=9.0)
            for _ in range(8)
        ]
        events += [
            self.event(rank=11, is_fn=True, above=True, predicted_distance=9.0)
            for _ in range(2)
        ]
        per_video = [
            {
                "edge_tp": 100,
                "edge_fp": 0,
                "edge_fn": 10,
                "ineligible_gt_edges": 0,
            }
        ]
        report = diagnostic.build_report(
            events=events,
            per_video=per_video,
            provenance={},
            elapsed_seconds=0.0,
        )
        self.assertFalse(report["e1_distance_rank"]["pass"])

    def test_e2_primary_threshold_and_actionable_subset_are_explicit(self) -> None:
        events = [
            self.event(
                rank=1,
                is_fn=True,
                above=True,
                predicted_distance=9.0 if index < 3 else 11.0,
                unmatched_source=index != 1,
            )
            for index in range(4)
        ]
        per_video = [
            {
                "edge_tp": 0,
                "edge_fp": 0,
                "edge_fn": 4,
                "ineligible_gt_edges": 0,
            }
        ]
        report = diagnostic.build_report(
            events=events,
            per_video=per_video,
            provenance={},
            elapsed_seconds=0.0,
        )
        e2 = report["e2_motion_recentring"]
        self.assertTrue(e2["pass"])
        self.assertEqual(e2["predicted_distance_le_10um_fraction"], 0.75)
        self.assertEqual(e2["proposed_unmatched_source_pass"]["denominator"], 3)
        self.assertEqual(
            e2["proposed_unmatched_source_pass"]["predicted_distance_le_10um"], 2
        )

    def test_candidate_absent_gt_edges_fail_the_primary_rank_denominator(self) -> None:
        events = [self.event(rank=1) for _ in range(9)]
        events.append(
            self.event(rank=1, is_fn=True, above=True, predicted_distance=9.0)
        )
        per_video = [
            {
                "edge_tp": 9,
                "edge_fp": 0,
                "edge_fn": 2,
                "ineligible_gt_edges": 1,
            }
        ]
        report = diagnostic.build_report(
            events=events,
            per_video=per_video,
            provenance={},
            elapsed_seconds=0.0,
        )
        overall = report["e1_distance_rank"]["all_official_gt_edges"]
        self.assertEqual(overall["denominator"], 11)
        self.assertEqual(overall["candidate_absent_or_unrankable"], 1)
        self.assertEqual(overall["rank_le_10"], 10)
        self.assertTrue(report["e1_distance_rank"]["pass"])

        per_video[0]["edge_fn"] = 3
        report = diagnostic.build_report(
            events=events,
            per_video=per_video,
            provenance={},
            elapsed_seconds=0.0,
        )
        self.assertFalse(report["e1_distance_rank"]["pass"])


class FrozenReportTests(unittest.TestCase):
    ARTIFACT = ROOT / "reports/g02/phase1_distance_motion_gate_holdout57_v1.json"

    @unittest.skipUnless(ARTIFACT.is_file(), "requires completed CV57 E1/E2 report")
    def test_report_reproduces_exact_baseline_and_decision_cohorts(self) -> None:
        report = json.loads(self.ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["cohort"]["official_edge_totals"],
            {"edge_tp": 34174, "edge_fp": 2736, "edge_fn": 2094},
        )
        self.assertEqual(report["cohort"]["official_gt_temporal_edges"], 36268)
        self.assertEqual(report["cohort"]["above_10um_fn_edges"], 317)
        overall = report["e1_distance_rank"]["all_official_gt_edges"]
        self.assertEqual(overall["candidate_absent_or_unrankable"], 803)
        self.assertEqual(overall["rank_le_10"], 35445)
        self.assertTrue(report["e1_distance_rank"]["pass"])
        self.assertFalse(report["e2_motion_recentring"]["pass"])


if __name__ == "__main__":
    unittest.main()
