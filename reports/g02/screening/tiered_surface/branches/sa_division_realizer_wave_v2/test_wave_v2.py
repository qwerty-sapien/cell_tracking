"""Focused contract and pure-function tests for division wave v2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREDICT = load("division_wave_predict", "predict_arm.py")
EVALUATE = load("division_wave_evaluate", "evaluate_wave.py")


class DivisionWaveV2Tests(unittest.TestCase):
    def test_scope_is_full_population(self) -> None:
        contract = json.loads((ROOT / "SHARED_CONTRACT.json").read_text())
        self.assertEqual(contract["binding_scope"]["target_counts"], {
            "44b6": 9, "6bba": 29, "total": 38
        })
        self.assertEqual(contract["binding_scope"]["fragment_counts"]["total"], 57)
        self.assertEqual(contract["reporting"]["per_family"], "generalization_diagnostic_only")

    def test_association_policy_is_unchanged(self) -> None:
        amendment = json.loads((ROOT / "SCOPE_AMENDMENT.json").read_text())
        self.assertFalse(amendment["association_policy"]["changed"])
        self.assertIn("44b6-first", amendment["association_policy"]["rule"])

    def test_direct_action_owns_all_incident_conflicts(self) -> None:
        edges = {(1, 2), (9, 3), (1, 8), (7, 6)}
        action = PREDICT.direct_action(edges, 1, (2, 3))
        self.assertEqual(action["add_edges"], [[1, 2], [1, 3]])
        self.assertEqual(
            action["expected_conflict_edges"], [[1, 2], [1, 8], [9, 3]]
        )

    def test_exact_target_matching_accepts_late_equivalence(self) -> None:
        event = {
            "event_key": "e",
            "identity": "train:44b6_x",
            "family": "44b6",
            "direct_repair": {"required_edges": [[1, 2], [1, 3]]},
            "late_fork_equivalence": {
                "available": True,
                "preferred": {"late_fork": {"required_edges": [[2, 4], [2, 5]]}},
                "alternatives": [],
            },
        }
        row = {
            "candidate_key": "c",
            "identity": "train:44b6_x",
            "family": "44b6",
            "action": {"add_edges": [[2, 4], [2, 5]]},
        }
        binding, families, matches = EVALUATE.exact_target_metrics([row], [event])
        self.assertEqual(binding["tp"], 1)
        self.assertEqual(binding["fp"], 0)
        self.assertEqual(binding["fn"], 0)
        self.assertEqual(families["44b6"]["tp"], 1)
        self.assertEqual(len(matches), 1)

    def test_duplicate_target_match_fails_closed(self) -> None:
        event = {
            "event_key": "e",
            "identity": "train:44b6_x",
            "family": "44b6",
            "direct_repair": {"required_edges": [[1, 2], [1, 3]]},
            "late_fork_equivalence": {"available": False},
        }
        row = {
            "candidate_key": "c",
            "identity": "train:44b6_x",
            "family": "44b6",
            "action": {"add_edges": [[1, 2], [1, 3]]},
        }
        with self.assertRaises(EVALUATE.EvaluationError):
            EVALUATE.exact_target_metrics([row, dict(row, candidate_key="d")], [event])

    def test_preregistrations_forbid_label_features(self) -> None:
        for directory in (
            "arm_a_geometry_motion",
            "arm_b_persistence",
            "arm_c_appearance",
            "arm_d_consensus",
        ):
            payload = json.loads((ROOT / directory / "PREREGISTRATION.json").read_text())
            self.assertFalse(payload["firewall"]["gt_bundle_features"])

    def test_peft_is_not_full_finetune(self) -> None:
        path = ROOT.parent / "sa_peft_adapter_v1" / "PREREGISTRATION.json"
        payload = json.loads(path.read_text())
        self.assertFalse(payload["model"]["full_finetune"])
        self.assertLess(payload["budget"]["projected_t4_h"], 5.0)
        self.assertLess(
            payload["model"]["trainable_parameter_fraction_strictly_less_than"], 0.01 + 1e-12
        )


if __name__ == "__main__":
    unittest.main()
