from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluation = load_script("evaluate_organizer_f0", "scripts/evaluate_organizer_f0.py")
builder = load_script("build_organizer_f0_eval_kaggle", "scripts/build_organizer_f0_eval_kaggle.py")


class OrganizerF0EvaluationTests(unittest.TestCase):
    def test_patched_scorer_is_hash_and_parity_locked(self) -> None:
        evidence = evaluation.validate_patched_scorer(
            ROOT / "reports/g02/scorer_parity_v1.json"
        )
        self.assertEqual(evidence["fixture_count"], 7)
        self.assertEqual(
            evidence["official_checkout"],
            "075fc5f5a52d11077f9dc2b074644618f26939e2",
        )

    def test_holdout_gate_rejects_incomplete_training_before_data_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(evaluation.HoldoutGateError, "missing or empty"):
                evaluation.validate_converged_training(Path(temporary))

    def test_evaluation_builder_is_private_scoped_and_nonproduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "package"
            report = builder.build(output, "onebitbrain/org-f0-03")
            self.assertTrue(report["private"])
            self.assertEqual(report["selection"], "frozen_fold0_holdout57")
            self.assertTrue(report["patched_scorer_required"])
            self.assertFalse(report["production_merge"])
            metadata = json.loads((output / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["kernel_sources"], ["onebitbrain/org-f0-03"])
            launch = (output / "launch.py").read_text(encoding="ascii")
            self.assertIn('value.get("status") == "converged"', launch)
            self.assertIn("expected exactly one converged organiser training output", launch)


if __name__ == "__main__":
    unittest.main()
