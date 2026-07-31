"""Static package and launch-contract tests for the G-03 Phase 1 proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_g03_phase1_resume_proof as builder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase1PackageTests(unittest.TestCase):
    def test_builder_matches_the_locked_launch_contract(self) -> None:
        locked = json.loads(
            (ROOT / "reports/g03/screening/PHASE1_LAUNCH_CONTRACT_20260731.json").read_text(
                encoding="ascii"
            )
        )
        self.assertEqual(locked["status"], "LOCKED_PRE_GPU")
        for relative, expected in locked["source_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packages"
            self.assertEqual(builder.main(["--output", str(output)]), 0)
            receipt = json.loads((output / "BUILD_RECEIPT.json").read_text(encoding="ascii"))
            self.assertEqual(
                receipt["source_bundle_sha256"], locked["runtime_source_bundle_sha256"]
            )
            for role in ("a", "b"):
                package = output / role
                expected = locked["kernels"][role]
                metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="ascii"))
                self.assertEqual(metadata["id"], expected["id"])
                self.assertEqual(metadata["dataset_sources"], expected["dataset_sources"])
                self.assertEqual(
                    sha256(package / "kernel-metadata.json"), expected["metadata_sha256"]
                )
                self.assertEqual(
                    sha256(package / "PHASE1_CONTRACT.json"),
                    expected["package_contract_sha256"],
                )

    def test_builder_never_replaces_a_package_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packages"
            output.mkdir()
            with self.assertRaisesRegex(builder.BuildError, "refusing to replace"):
                builder.main(["--output", str(output)])


if __name__ == "__main__":
    unittest.main()
