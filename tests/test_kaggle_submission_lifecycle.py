from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import kaggle_submission_lifecycle as lifecycle


def validator(path: Path, *, version: int) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value["kernel_version"] != version:
        raise ValueError("version drift")
    return {
        "predicate_path": str(path.resolve()),
        "predicate_sha256": lifecycle.sha256_file(path),
        "kernel_version": version,
    }


def command_result(stdout: str, *, returncode: int = 0) -> dict[str, object]:
    return {
        "argv": [],
        "started_utc": "2026-08-03T00:00:00Z",
        "ended_utc": "2026-08-03T00:00:01Z",
        "duration_seconds": 1.0,
        "process_returncode": returncode,
        "timed_out": False,
        "stdout": stdout,
        "stderr": "",
    }


class KaggleSubmissionLifecycleTests(unittest.TestCase):
    def test_submission_state_classification_distinguishes_blank_complete(self) -> None:
        self.assertEqual(
            lifecycle.classify_submission({"status": "complete", "publicScore": "0.881"}),
            "SCORED_SUCCESS",
        )
        self.assertEqual(
            lifecycle.classify_submission({"status": "complete", "publicScore": ""}),
            "SCORING_ERROR_BLANK_SCORE",
        )
        self.assertEqual(
            lifecycle.classify_submission({"status": "pending", "publicScore": ""}),
            "PENDING",
        )
        self.assertEqual(
            lifecycle.classify_submission(
                {"status": "complete", "publicScore": None, "privateScore": None}
            ),
            "SCORING_ERROR_BLANK_SCORE",
        )

    def test_one_attempt_binds_exactly_one_new_ref_and_consumes_predicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            predicate = temporary / "predicate.json"
            predicate.write_text(json.dumps({"kernel_version": 7}))
            receipts = temporary / "receipts"
            rows = [
                command_result(json.dumps([{"ref": 10, "status": "complete", "publicScore": "0.8"}])),
                command_result("Successfully submitted"),
                command_result(
                    json.dumps(
                        [
                            {"ref": 10, "status": "complete", "publicScore": "0.8"},
                            {"ref": 11, "status": "pending", "publicScore": ""},
                        ]
                    )
                ),
            ]
            spec = lifecycle.CodeSubmissionSpec(
                competition="competition",
                kernel_id="owner/kernel",
                kernel_version=7,
                output_file="submission.csv",
                message="test",
            )
            with mock.patch.object(
                lifecycle, "_run_command", side_effect=rows
            ) as run_command, mock.patch.object(
                lifecycle, "CLAIM_DIR", temporary / "claims"
            ):
                receipt = lifecycle.run_code_submission_once(
                    spec=spec,
                    predicate_receipt=predicate,
                    predicate_validator=validator,
                    receipt_dir=receipts,
                    receipt_schema="test.receipt.v1",
                    node="test",
                    lock_path=temporary / "serial.lock",
                )
            self.assertEqual(receipt["returncode"], 0)
            self.assertEqual(receipt["submission_ref"], 11)
            self.assertEqual(receipt["immediate_submission_state"], "PENDING")
            submit_argv = run_command.call_args_list[1].args[0]
            self.assertEqual(submit_argv[submit_argv.index("-v") + 1], "7")
            with self.assertRaises(lifecycle.PredicateConsumed):
                with mock.patch.object(lifecycle, "CLAIM_DIR", temporary / "claims"):
                    lifecycle.run_code_submission_once(
                        spec=spec,
                        predicate_receipt=predicate,
                        predicate_validator=validator,
                        receipt_dir=temporary / "different-receipts",
                        receipt_schema="test.receipt.v1",
                        node="test",
                        lock_path=temporary / "serial.lock",
                    )

    def test_cli_success_without_one_new_ref_fails_closed_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            predicate = temporary / "predicate.json"
            predicate.write_text(json.dumps({"kernel_version": 2}))
            unchanged = json.dumps([{"ref": 10, "status": "complete", "publicScore": "0.8"}])
            with mock.patch.object(
                lifecycle,
                "_run_command",
                side_effect=[command_result(unchanged), command_result("ok"), command_result(unchanged)],
            ), mock.patch.object(lifecycle, "CLAIM_DIR", temporary / "claims"):
                receipt = lifecycle.run_code_submission_once(
                    spec=lifecycle.CodeSubmissionSpec(
                        competition="competition",
                        kernel_id="owner/kernel",
                        kernel_version=2,
                        output_file="submission.csv",
                        message="test",
                    ),
                    predicate_receipt=predicate,
                    predicate_validator=validator,
                    receipt_dir=temporary / "receipts",
                    receipt_schema="test.receipt.v1",
                    node="test",
                    lock_path=temporary / "serial.lock",
                    inventory_poll_attempts=1,
                )
            self.assertEqual(receipt["returncode"], 1)
            self.assertEqual(
                receipt["lifecycle_status"], "CLI_ACCEPTED_REF_UNRESOLVED_NO_RETRY"
            )
            self.assertFalse(receipt["retry_allowed"])


if __name__ == "__main__":
    unittest.main()
