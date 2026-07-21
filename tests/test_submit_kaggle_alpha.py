from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_kaggle_alpha.py"


def load_module():
    spec = importlib.util.spec_from_file_location("submit_kaggle_alpha", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SubmitKaggleAlphaTests(unittest.TestCase):
    def test_classifies_file_submission_403(self) -> None:
        module = load_module()
        result = {
            "exit_status": 1,
            "output": "403 Client Error: Forbidden for url: .../CreateSubmission",
        }
        self.assertEqual(
            module.classify_submit_result(result),
            "blocked_kaggle_file_submission_permission_403",
        )

    def test_classifies_code_submission_403(self) -> None:
        module = load_module()
        result = {
            "exit_status": 1,
            "output": "403 Client Error: Forbidden for url: .../CreateCodeSubmission",
        }
        self.assertEqual(
            module.classify_submit_result(result),
            "blocked_kaggle_code_submission_permission_403",
        )

    def test_success_is_submitted(self) -> None:
        module = load_module()
        self.assertEqual(module.classify_submit_result({"exit_status": 0, "output": ""}), "submitted")


if __name__ == "__main__":
    unittest.main()
