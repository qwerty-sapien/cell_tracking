from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import cmux_ops
from scripts import g04_division_hidden_guard as GUARD
from scripts import g04_hidden_kaggle_code_submit as HIDDEN_SUBMIT


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "reports/g04/lanes/division_permissive/promoted_unfiltered_geometry_cap5"
    / "public_parent/submission.csv"
)
CANDIDATE = (
    ROOT
    / "reports/g04/lanes/division_permissive/promoted_unfiltered_geometry_cap5"
    / "submission.csv"
)
RAW_RECEIPT = GUARD.RAW_PARITY


def write_output_receipt(path: Path, output_dir: Path, *, version: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "celltrack-alpha.cmux-op-receipt.v1",
                "op": "kernel_output",
                "mutating": False,
                "returncode": 0,
                "process_returncode": 0,
                "timed_out": False,
                "semantic_failure": False,
                "argv": [
                    "kaggle",
                    "kernels",
                    "output",
                    f"{GUARD.KERNEL_ID}/{version}",
                    "-p",
                    str(output_dir.resolve()),
                ],
            }
        )
    )
    return path


class HiddenTransportGuardTests(unittest.TestCase):
    def test_prelaunch_is_cmux_compatible_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            receipt = Path(temporary_text) / "prelaunch.json"
            result = GUARD.prelaunch(receipt)
            self.assertTrue(result["eligible"])
            self.assertEqual(result["failures"], [])
            bound = cmux_ops._validate_kernel_push_predicate(
                receipt,
                kernel_id=GUARD.KERNEL_ID,
                kernel_dir=GUARD.PACKAGE,
            )
            self.assertEqual(bound["code_sha256"], GUARD.CODE_SHA256)

    def test_terminal_public_output_yields_version_bound_predicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            prelaunch = temporary / "prelaunch.json"
            GUARD.prelaunch(prelaunch)
            output_dir = temporary / "download"
            output_dir.mkdir()
            (output_dir / "g04_apncc_parent.csv").symlink_to(PARENT)
            (output_dir / "submission.csv").symlink_to(CANDIDATE)
            (output_dir / "G04_DIVISION_POSTPROCESS_RECEIPT.json").write_bytes(
                RAW_RECEIPT.read_bytes()
            )
            predicate_path = temporary / "predicate.json"
            output_receipt = write_output_receipt(
                temporary / "kernel-output.json", output_dir, version=3
            )
            predicate = GUARD.validate_terminal(
                output_dir,
                kernel_version=3,
                kernel_output_receipt=output_receipt,
                prelaunch_receipt=prelaunch,
                predicate_path=predicate_path,
            )
            self.assertTrue(predicate["eligible"])
            self.assertEqual(predicate["kernel_version"], 3)
            self.assertEqual(predicate["output_sha256"], GUARD.OUTPUT_SHA256)
            self.assertTrue(predicate["hidden_rerun_compatible"])
            self.assertTrue(predicate["dynamic_test_discovery"])
            self.assertTrue(predicate["fresh_submission_generation"])
            self.assertEqual(json.loads(predicate_path.read_text())["failures"], [])
            validated = HIDDEN_SUBMIT.validate_predicate(predicate_path, version=3)
            self.assertEqual(validated["kernel_version"], 3)
            self.assertEqual(
                validated["kernel_output_receipt_sha256"],
                predicate["kernel_output_receipt_sha256"],
            )

    def test_terminal_rejects_output_receipt_for_latest_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            prelaunch = temporary / "prelaunch.json"
            GUARD.prelaunch(prelaunch)
            output_dir = temporary / "download"
            output_dir.mkdir()
            (output_dir / "g04_apncc_parent.csv").symlink_to(PARENT)
            (output_dir / "submission.csv").symlink_to(CANDIDATE)
            (output_dir / "G04_DIVISION_POSTPROCESS_RECEIPT.json").write_bytes(
                RAW_RECEIPT.read_bytes()
            )
            output_receipt = write_output_receipt(
                temporary / "kernel-output.json", output_dir, version=3
            )
            receipt = json.loads(output_receipt.read_text())
            receipt["argv"][3] = GUARD.KERNEL_ID
            output_receipt.write_text(json.dumps(receipt))
            predicate = GUARD.validate_terminal(
                output_dir,
                kernel_version=3,
                kernel_output_receipt=output_receipt,
                prelaunch_receipt=prelaunch,
                predicate_path=temporary / "predicate.json",
            )
            self.assertFalse(predicate["eligible"])
            self.assertTrue(
                any("exact version" in item for item in predicate["failures"])
            )

    def test_terminal_output_drift_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            prelaunch = temporary / "prelaunch.json"
            GUARD.prelaunch(prelaunch)
            output_dir = temporary / "download"
            output_dir.mkdir()
            (output_dir / "g04_apncc_parent.csv").symlink_to(PARENT)
            (output_dir / "submission.csv").symlink_to(PARENT)
            (output_dir / "G04_DIVISION_POSTPROCESS_RECEIPT.json").write_bytes(
                RAW_RECEIPT.read_bytes()
            )
            output_receipt = write_output_receipt(
                temporary / "kernel-output.json", output_dir, version=1
            )
            predicate = GUARD.validate_terminal(
                output_dir,
                kernel_version=1,
                kernel_output_receipt=output_receipt,
                prelaunch_receipt=prelaunch,
                predicate_path=temporary / "predicate.json",
            )
            self.assertFalse(predicate["eligible"])
            self.assertTrue(any("output byte identity drift" in item for item in predicate["failures"]))


if __name__ == "__main__":
    unittest.main()
