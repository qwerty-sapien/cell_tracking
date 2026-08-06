"""Local integrity tests for the G-03 checkpoint handoff boundary."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from g03_checkpoint_resume import (  # noqa: E402
    INVENTORY_FILE,
    MANIFEST_FILE,
    READY_FILE,
    ResumeProofError,
    SCHEMA,
    publish_handoff,
    verify_handoff,
    write_idempotent_json,
)


def manifest() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "handoff_id": "g03-phase1-a-to-b-v1",
        "producer_kernel": "onebitbrain/g03-phase1-resume-a-v1",
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "state": {"global_step": 17, "scheduler_epoch": 17},
    }


class CheckpointResumeTests(unittest.TestCase):
    def test_publish_verify_and_identical_restart_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "handoff"
            first = publish_handoff(
                root,
                manifest=manifest(),
                payloads={"state/checkpoint.bin": b"payload", "state/rng.json": b"{}"},
            )
            second = publish_handoff(
                root,
                manifest=manifest(),
                payloads={"state/checkpoint.bin": b"payload", "state/rng.json": b"{}"},
            )
            self.assertEqual(first, second)
            self.assertEqual(verify_handoff(root)["handoff_id"], "g03-phase1-a-to-b-v1")

    def test_tampered_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "handoff"
            publish_handoff(root, manifest=manifest(), payloads={"state/checkpoint.bin": b"payload"})
            (root / "state/checkpoint.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ResumeProofError, "hash mismatch"):
                verify_handoff(root)

    def test_incomplete_or_undeclared_handoff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "handoff"
            root.mkdir()
            (root / MANIFEST_FILE).write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(ResumeProofError, "cannot read handoff ready marker"):
                verify_handoff(root)
            root = Path(directory) / "complete"
            publish_handoff(root, manifest=manifest(), payloads={"state/checkpoint.bin": b"payload"})
            (root / "unexpected.bin").write_bytes(b"drift")
            with self.assertRaisesRegex(ResumeProofError, "undeclared"):
                verify_handoff(root)

    def test_conflicting_restart_and_result_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "handoff"
            publish_handoff(root, manifest=manifest(), payloads={"state/checkpoint.bin": b"payload"})
            with self.assertRaisesRegex(ResumeProofError, "payload conflicts"):
                publish_handoff(root, manifest=manifest(), payloads={"state/checkpoint.bin": b"other"})
            result = Path(directory) / "result.json"
            self.assertEqual(write_idempotent_json(result, {"ok": True})[0], True)
            self.assertEqual(write_idempotent_json(result, {"ok": True})[0], False)
            with self.assertRaisesRegex(ResumeProofError, "conflicts"):
                write_idempotent_json(result, {"ok": False})


if __name__ == "__main__":
    unittest.main()
