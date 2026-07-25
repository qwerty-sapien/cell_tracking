from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "celltrack_alpha"))
sys.path.insert(0, str(ROOT / "scripts"))

from model_candidate_gate import (  # noqa: E402
    CandidateGateError,
    validate_candidate,
)
import run_local_training_guard as guard  # noqa: E402


POLICY = {
    "schema": "celltrack-alpha.compute-governance.v1",
    "status": "locked",
    "promotion": {"hard_baseline_score_improvement_percent": 3.5},
    "exploration_strategy": {
        "local_parallel_lane": {
            "maximum_training_memory_gib": 18,
            "minimum_background_reserve_gib": 6,
            "minimum_system_memory_free_percent": 25,
            "maximum_unattended_stage_hours": 7,
        }
    },
}


def valid_candidate() -> dict[str, object]:
    return {
        "schema": "celltrack-alpha.model-candidate.v1",
        "candidate_id": "lite-001",
        "model_family": "tiny-temporal-conv",
        "literature_evidence": [
            {
                "kind": "primary_peer_reviewed",
                "citation": "Author et al. (2025)",
                "url": "https://example.test/paper",
                "reported_result": "Improved association accuracy by 6%.",
                "task_relevance": "Targets the measured fit-only recall bottleneck.",
            }
        ],
        "hypothesis": {
            "observed_failure_mode": "Missed short-range continuations.",
            "proposed_mechanism": "Small temporal kernels recover local motion.",
            "falsification_test": "Reject if fit validation recall does not rise.",
            "expected_score_improvement_percent": 3.5,
        },
        "local_fast_fail": {
            "m5_compatible": True,
            "projected_peak_memory_gib": 12,
            "background_reserve_gib": 6,
            "max_wall_minutes": 60,
            "training_scope": "fit_only",
            "holdout_access": False,
            "swap_growth_allowed": False,
            "early_stop_conditions": ["No learning by the preregistered checkpoint."],
            "success_criteria": ["Fit validation recall rises without precision collapse."],
        },
        "decision": "PASS",
    }


class CandidateGateTests(unittest.TestCase):
    def test_valid_literature_first_candidate_passes(self) -> None:
        result = validate_candidate(valid_candidate(), POLICY)
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["maximum_training_memory_gib"], 18.0)
        self.assertFalse(result["holdout_access"])

    def test_candidate_without_primary_literature_fails(self) -> None:
        candidate = valid_candidate()
        candidate["literature_evidence"][0]["kind"] = "blog"  # type: ignore[index]
        with self.assertRaisesRegex(CandidateGateError, "primary_peer_reviewed"):
            validate_candidate(candidate, POLICY)

    def test_candidate_below_uplift_gate_fails(self) -> None:
        candidate = valid_candidate()
        candidate["hypothesis"]["expected_score_improvement_percent"] = 3.49  # type: ignore[index]
        with self.assertRaisesRegex(CandidateGateError, "below"):
            validate_candidate(candidate, POLICY)

    def test_memory_swap_holdout_and_seven_hour_gates_fail_closed(self) -> None:
        mutations = (
            (("local_fast_fail", "projected_peak_memory_gib"), 18.01, "exceeds"),
            (("local_fast_fail", "background_reserve_gib"), 5.99, "below"),
            (("local_fast_fail", "max_wall_minutes"), 421, "exceeds"),
            (("local_fast_fail", "holdout_access"), True, "holdout_access"),
            (("local_fast_fail", "swap_growth_allowed"), True, "swap_growth_allowed"),
        )
        for keys, value, message in mutations:
            with self.subTest(field=keys[-1]):
                candidate = valid_candidate()
                candidate[keys[0]][keys[1]] = value  # type: ignore[index]
                with self.assertRaisesRegex(CandidateGateError, message):
                    validate_candidate(candidate, POLICY)

    def test_cgl_d14_goal_owner_swap_amendment_is_exact_and_bounded(self) -> None:
        candidate = valid_candidate()
        candidate["candidate_id"] = "cgl-temporalunet-tail-rank-head-v1"
        local = candidate["local_fast_fail"]  # type: ignore[assignment]
        local["swap_growth_allowed"] = True
        local["maximum_swap_growth_gib"] = 1.5
        local["runtime_rss_ceiling_gib"] = 12
        local["max_wall_minutes"] = 360
        candidate["runtime_guard_amendment"] = {
            "authorized_by": "goal_owner",
            "authorized_on": "2026-07-26",
            "scope": "CGL D-14 fast-fail only",
            "prior_maximum_swap_growth_gib": 1.5,
            "maximum_swap_growth_gib": 1.5,
            "rss_ceiling_gib": 12,
            "max_wall_minutes": 360,
            "outputs_deterministic_wrt_swap": True,
        }
        result = validate_candidate(candidate, POLICY)
        self.assertTrue(result["swap_growth_allowed"])
        self.assertEqual(result["maximum_swap_growth_gib"], 1.5)
        self.assertEqual(result["maximum_training_memory_gib"], 12.0)
        self.assertEqual(result["policy_maximum_training_memory_gib"], 18.0)

        local["maximum_swap_growth_gib"] = 1.51
        with self.assertRaisesRegex(CandidateGateError, "exactly 1.5 GiB"):
            validate_candidate(candidate, POLICY)

    def test_swap_parser_is_unit_aware_and_fail_closed(self) -> None:
        self.assertEqual(
            guard.parse_swap_used_bytes("total = 4096.00M  used = 12.50M  free = 4083.50M"),
            int(12.5 * guard.MIB),
        )
        self.assertEqual(
            guard.parse_swap_used_bytes("total = 4.00G used = 1.25G free = 2.75G"),
            int(1.25 * guard.GIB),
        )
        with self.assertRaises(guard.LocalTrainingGuardError):
            guard.parse_swap_used_bytes("unknown")
        self.assertEqual(
            guard.parse_memory_free_percent("System-wide memory free percentage: 43%"),
            43.0,
        )
        with self.assertRaises(guard.LocalTrainingGuardError):
            guard.parse_memory_free_percent("unknown")
        self.assertFalse(
            guard.swap_growth_exceeds_ceiling(2 * guard.GIB, 3 * guard.GIB, guard.GIB)
        )
        self.assertTrue(
            guard.swap_growth_exceeds_ceiling(
                2 * guard.GIB, 3 * guard.GIB + 1, guard.GIB
            )
        )

    def test_repository_template_fails_closed_until_reviewed(self) -> None:
        candidate = json.loads(
            (ROOT / "reports" / "g02" / "model_candidate_template_v1.json").read_text()
        )
        policy = json.loads(
            (ROOT / "reports" / "g02" / "compute_governance_v1.json").read_text()
        )
        with self.assertRaisesRegex(CandidateGateError, "decision"):
            validate_candidate(candidate, policy)

    def test_report_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guard.json"
            guard.write_report(path, {"stop_reason": "completed"})
            self.assertEqual(json.loads(path.read_text())["stop_reason"], "completed")
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
