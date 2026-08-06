"""Focused fail-closed tests for division realizer Arm C."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArmCTest(unittest.TestCase):
    def test_preregistration_is_44b6_only_and_cache_only(self) -> None:
        value = json.loads((HERE / "PREREGISTRATION.json").read_text())
        self.assertEqual(value["scope"]["family"], "44b6")
        self.assertEqual(value["scope"]["fragment_count"], 21)
        self.assertFalse(value["scope"]["full_pool_cache_creation_or_extension"])
        self.assertTrue(value["hard_stops"]["six_bba"])
        self.assertTrue(value["hard_stops"]["cv57"])

    def test_prediction_cli_has_no_target_argument(self) -> None:
        source = (HERE / "generate_predictions.py").read_text()
        self.assertNotIn("argparse", source)
        self.assertNotIn("continuity_target_spec", source)
        self.assertNotIn("oracle_audit", source)
        self.assertNotIn("DIVISION_REPAIR_BUNDLES", source)

    def test_closed_gate_never_creates_predictions(self) -> None:
        module = load_module("arm_c_generate_predictions", HERE / "generate_predictions.py")
        if (HERE / "CACHE_AUDIT.json").is_file():
            self.assertEqual(
                module.prediction_gate(),
                "FAIL_CLOSED_NO_ADEQUATE_FROZEN_INTENSITY_CACHE",
            )
        self.assertFalse((HERE / "PREDICTIONS.json").exists())


if __name__ == "__main__":
    unittest.main()
