"""Static and isolated-runtime tests for the G-03 single-file Kaggle package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_g03_phase1_resume_proof as builder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase1PackageTests(unittest.TestCase):
    def test_builder_materializes_self_contained_v2_packages(self) -> None:
        locked = json.loads(
            (
                ROOT
                / "reports/g03/screening/PHASE1_TRANSPORT_REPAIR_CONTRACT_20260801.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(locked["status"], "LOCKED_PRE_GPU_TRANSPORT_REPAIR")
        for relative, expected in locked["source_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packages"
            receipt = builder.build(output)
            self.assertEqual(
                receipt["schema"], "celltrack-alpha.g03-phase1-single-file-package-build.v2"
            )
            self.assertEqual(receipt["source_hashes"], builder.source_hashes())
            self.assertEqual(
                receipt["source_bundle_sha256"], locked["runtime_source_bundle_sha256"]
            )
            for role in ("a", "b"):
                package = output / role
                metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="ascii"))
                script = package / builder.SINGLE_FILE_NAME
                expected = locked["kernels"][role]
                self.assertEqual(metadata["code_file"], builder.SINGLE_FILE_NAME)
                self.assertEqual(metadata["id"], expected["id"])
                self.assertEqual(metadata["dataset_sources"], expected["dataset_sources"])
                self.assertTrue(script.is_file())
                self.assertFalse((package / "g03_resume_torch.py").exists())
                self.assertFalse((package / "g03_checkpoint_resume.py").exists())
                source = script.read_text(encoding="utf-8")
                self.assertNotIn("from g03_checkpoint_resume import", source)
                self.assertNotIn("from g03_resume_torch import", source)
                self.assertEqual(
                    receipt["roles"][0 if role == "a" else 1]["selftest"]["status"],
                    "PASS_G03_SINGLE_FILE_SELFTEST",
                )
                self.assertEqual(sha256(script), expected["code_file_sha256"])
                self.assertEqual(
                    sha256(package / "kernel-metadata.json"), expected["metadata_sha256"]
                )
                self.assertEqual(
                    receipt["roles"][0 if role == "a" else 1]["selftest"]["contract_sha256"],
                    expected["embedded_contract_sha256"],
                )

    def test_generated_scripts_pass_from_an_isolated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packages"
            builder.build(output)
            for role in ("A", "B"):
                receipt = builder._assert_isolated_selftest(
                    output / role.lower() / builder.SINGLE_FILE_NAME, role
                )
                self.assertEqual(receipt["imports_project_modules"], False)

    def test_builder_never_replaces_a_package_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packages"
            output.mkdir()
            with self.assertRaisesRegex(builder.BuildError, "refusing to replace"):
                builder.build(output)

    def test_role_contract_is_versioned_and_role_specific(self) -> None:
        hashes = builder.source_hashes()
        contract_a = builder.role_contract("A", hashes)
        contract_b = builder.role_contract("B", hashes)
        self.assertEqual(contract_a["schema"], "celltrack-alpha.g03-phase1-contract.v2")
        self.assertEqual(contract_a["handoff_id"], "g03-phase1-a-to-b-v2")
        self.assertEqual(contract_b["producer_kernel"], contract_a["kernel_id"])
        self.assertEqual(contract_a["source_hashes"], contract_b["source_hashes"])

    def test_builder_fails_closed_on_any_project_local_import(self) -> None:
        hashes = builder.source_hashes()
        with mock.patch.object(
            builder, "_component_source", return_value="import g03_checkpoint_resume\n"
        ):
            with self.assertRaisesRegex(builder.BuildError, "project-local imports"):
                builder._single_file_source("A", hashes)


if __name__ == "__main__":
    unittest.main()
