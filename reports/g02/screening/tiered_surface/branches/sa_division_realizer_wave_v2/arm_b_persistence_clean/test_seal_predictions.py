#!/usr/bin/env python3
"""Focused tests for target-free Arm B interface sealing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from seal_predictions import SealError, event_key, seal, validate_actions


ARM_DIR = Path(__file__).resolve().parent
WAVE_DIR = ARM_DIR.parent


def source_args(output: Path) -> dict[str, Path]:
    return {
        "actions_path": ARM_DIR / "ACTIONS.json",
        "run_receipt_path": ARM_DIR / "RUN_RECEIPT.json",
        "structural_verification_path": ARM_DIR / "STRUCTURAL_VERIFICATION.json",
        "preregistration_path": ARM_DIR / "PREREGISTRATION.json",
        "implementation_freeze_path": ARM_DIR / "IMPLEMENTATION_FREEZE.json",
        "execution_guard_amendment_path": ARM_DIR / "EXECUTION_GUARD_AMENDMENT.json",
        "predictor_path": ARM_DIR / "predict.py",
        "shared_contract_path": WAVE_DIR / "SHARED_CONTRACT.json",
        "output_path": output,
    }


class SealPredictionsTests(unittest.TestCase):
    def test_frozen_event_key_vector(self) -> None:
        self.assertEqual(
            event_key("train:44b6_267148e4", 20390, 20642, 20636),
            "8889f5134c1f547f624c290ba1b70b698ba219fd95b9f2cc1f1714fb54136c7d",
        )

    def test_real_sources_seal_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first" / "PREDICTIONS.json"
            second = root / "second" / "PREDICTIONS.json"
            receipt = seal(**source_args(first))
            seal(**source_args(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(payload["selection"]["count"], 48)
            self.assertEqual(payload["selection"]["by_family"], {"44b6": 19, "6bba": 29})
            self.assertEqual(
                payload["source_authentication"]["authenticated_input_count"], 57
            )
            self.assertTrue(payload["scope"]["all_upstream_inputs_authenticated"])
            self.assertEqual(receipt["selected_count"], 48)

    def test_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "PREDICTIONS.json"
            seal(**source_args(output))
            with self.assertRaisesRegex(SealError, "refusing to overwrite"):
                seal(**source_args(output))

    def test_actions_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actions = json.loads((ARM_DIR / "ACTIONS.json").read_text(encoding="utf-8"))
            actions["selected_fork_count"] = 47
            changed = root / "ACTIONS.json"
            changed.write_text(json.dumps(actions), encoding="utf-8")
            args = source_args(root / "PREDICTIONS.json")
            args["actions_path"] = changed
            with self.assertRaisesRegex(SealError, "does not match run receipt"):
                seal(**args)

    def test_duplicate_dataset_fails_closed(self) -> None:
        actions = json.loads((ARM_DIR / "ACTIONS.json").read_text(encoding="utf-8"))
        changed = copy.deepcopy(actions)
        changed["videos"][1]["dataset"] = changed["videos"][0]["dataset"]
        expected = {f"train__{row['dataset']}.csv" for row in actions["videos"]}
        with self.assertRaisesRegex(SealError, "duplicate dataset"):
            validate_actions(changed, expected)


if __name__ == "__main__":
    unittest.main()
