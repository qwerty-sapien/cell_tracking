#!/usr/bin/env python3
"""Focused tests for the SA-NODETOPO fail-closed audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/agent/Desktop/cell_tracking")
SPEC = importlib.util.spec_from_file_location(
    "audit_candidate_universe", HERE / "audit_candidate_universe.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateUniverseAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = MODULE.audit(ROOT)

    def test_all_frozen_pins_match(self) -> None:
        self.assertTrue(all(row["ok"] for row in self.result["pins"].values()))

    def test_candidate_generation_precedes_gt_join(self) -> None:
        self.assertTrue(
            self.result["proposal_origin"]["generation_precedes_label_join"]
        )
        self.assertTrue(self.result["proposal_origin"]["artifact_is_later_gt_annotated"])

    def test_partial_bank_fails_complete_44b6_contract(self) -> None:
        coverage = self.result["candidate_identity_audit"][
            "fit105_partial_dataset_coverage"
        ]["44b6"]
        self.assertLess(coverage["covered"], coverage["required"])
        self.assertFalse(
            self.result["candidate_identity_audit"]["complete_44b6_coverage"]
        )

    def test_background_and_real_join_fail_closed(self) -> None:
        self.assertFalse(
            self.result["background_contract"]["exact_unlabeled_background_available"]
        )
        self.assertFalse(
            self.result["candidate_identity_audit"][
                "exact_candidate_to_baseline_join"
            ]
        )

    def test_n1_requires_complete_topology(self) -> None:
        evidence = self.result["n1_contract_evidence"]
        self.assertTrue(evidence["valid_complete_action"]["compiled"])
        self.assertTrue(evidence["valid_complete_action"]["observation_distinct"])
        self.assertTrue(evidence["isolated_coordinate_only_control"]["rejected"])
        self.assertEqual(evidence["real_candidate_actions_compiled"], 0)

    def test_scope_and_terminal_verdict(self) -> None:
        scope = self.result["scope"]
        self.assertFalse(scope["audit_used"])
        self.assertFalse(scope["test_split_used"])
        self.assertFalse(scope["cv57_used"])
        self.assertFalse(scope["held_graph_scored"])
        self.assertFalse(scope["second_family_opened"])
        self.assertEqual(
            self.result["verdict"],
            "PARK_NO_COMPLETE_AUTHENTIC_CANDIDATE_UNIVERSE",
        )

    def test_quarantined_values_absent(self) -> None:
        self.assertFalse(self.result["quarantine"]["memo_c2_values_used"])
        self.assertFalse(self.result["quarantine"]["memo_branch_b_66_223_used"])


if __name__ == "__main__":
    unittest.main()
