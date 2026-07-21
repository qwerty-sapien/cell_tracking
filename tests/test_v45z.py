from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("build_v45z", ROOT / "scripts" / "build_v45z.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V45ZeroControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    @staticmethod
    def _parent() -> dict:
        return {
            "status": "completed",
            "completed_steps": 1000,
            "checkpoint_sha256": "a" * 64,
            "identity": {
                "fold": 0,
                "registry_sha256": "b" * 64,
                "split_sha256": "c" * 64,
                "provenance_sha256": "d" * 64,
                "target_contract_sha256": self_target_contract(),
                "selected_sources": [
                    {"group_id": "group-1", "stem": "video", "video_id": "train:video"}
                ],
                "student_views": [0],
                "division_emission": False,
            },
        }

    def test_control_identity_has_no_training_claim(self) -> None:
        parent = self._parent()
        result = self.mod.control_identity(parent, "e" * 64)
        self.assertEqual(result["control"], "one-view-zero-correction-and-auxiliary")
        self.assertEqual(result["fold"], 0)
        self.assertTrue(result["image_or_geff_data_opened"] is False)
        self.assertEqual(result["selected_sources"], parent["identity"]["selected_sources"])
        self.assertEqual(result["parent_checkpoint_sha256"], "a" * 64)

    def test_parent_identity_rejects_non_one_view_or_trainingless_parent(self) -> None:
        parent = self._parent()
        parent["identity"]["student_views"] = [0, 1]
        with self.assertRaisesRegex(self.mod.V45ZeroControlError, "one-view"):
            self.mod.parent_identity(parent)
        parent = self._parent()
        parent["completed_steps"] = 0
        with self.assertRaisesRegex(self.mod.V45ZeroControlError, "positive"):
            self.mod.parent_identity(parent)

    def test_zero_head_requires_exact_zero_correction_and_auxiliary_rows(self) -> None:
        state = {
            "corr_hid.weight": torch.ones((1, 1, 1, 1, 1)),
            "corr_hid.bias": torch.ones((1,)),
            "corr_out.weight": torch.zeros((1, 1, 1, 1, 1)),
            "corr_out.bias": torch.zeros((1,)),
            "aux_head.weight": torch.zeros((5, 1, 1, 1, 1)),
            "aux_head.bias": torch.zeros((5,)),
        }
        self.mod.assert_zero_state(state, 1_254)
        broken = {name: value.clone() for name, value in state.items()}
        broken["corr_out.bias"][0] = 1.0
        with self.assertRaisesRegex(self.mod.V45ZeroControlError, "corr_out.bias"):
            self.mod.assert_zero_state(broken, 1_254)
        with self.assertRaisesRegex(self.mod.V45ZeroControlError, "accounting"):
            self.mod.assert_zero_state(state, 1)


def self_target_contract() -> str:
    from celltrack_alpha.t_adapter import TARGET_CONTRACT_SHA256

    return TARGET_CONTRACT_SHA256


if __name__ == "__main__":
    unittest.main()
