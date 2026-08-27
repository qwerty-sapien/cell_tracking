from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
KERNEL = HERE / "kernel/kaggle_apncc_transport.py"


def load_builder():
    """Load v2 explicitly so a v1 sibling on PYTHONPATH cannot mask it."""
    spec = importlib.util.spec_from_file_location(
        "apncc_transport_v2_builder", HERE / "build_transport_kernel.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load APNCC v2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_builder()


class ApnccTransportTests(unittest.TestCase):
    def test_build_pins_requested_banked_identity_and_all_fragment_references(self):
        receipt = json.loads((HERE / "BUILD_RECEIPT.json").read_text())
        contract = json.loads((HERE / "CONTRACT.json").read_text())
        output = build.expected_output()
        self.assertEqual(receipt["banked_identity_sha256"], build.BANKED_IDENTITY_SHA256)
        self.assertEqual(contract["input_manifest_sha256"], build.BANKED_IDENTITY_SHA256)
        self.assertEqual(contract["wheel_count"], 62)
        self.assertEqual(contract["request_count"], 264378)
        self.assertEqual(len(output["fragments"]), 4)
        self.assertEqual(sum(row["rows"] for row in output["fragments"]), 263282)
        self.assertEqual(
            output["submission"]["sha256"],
            build.PUBLIC_SUBMISSION_SHA256,
        )
        self.assertEqual(
            output["reference"]["archive_sha256"],
            build.PUBLIC_ARCHIVE_SHA256,
        )

    def test_terminal_receipt_prevents_generated_artifact_overwrite(self):
        self.assertTrue((HERE / "TERMINAL.json").is_file())
        with self.assertRaisesRegex(RuntimeError, "one-shot terminal receipt"):
            build.build()

    def test_public_archive_matches_terminal_remote_csvs_byte_for_byte(self):
        remote = HERE / "remote_output"
        names = [
            "submission.csv",
            "fragments/test__44b6_0113de3b.csv",
            "fragments/test__44b6_0b24845f.csv",
            "fragments/test__6bba_05b6850b.csv",
            "fragments/test__6bba_05db0fb1.csv",
        ]
        with zipfile.ZipFile(build.PUBLIC_ARCHIVE) as archive:
            for name in names:
                self.assertEqual(archive.read(name), (remote / name).read_bytes())

    def test_kernel_keeps_step0_before_heavy_imports_and_uses_one_support_dataset(self):
        source = KERNEL.read_text(encoding="utf-8")
        tree = ast.parse(source)
        step0 = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "STEP0" for target in node.targets)
        )
        heavy = {"numpy", "scipy", "torch", "zarr", "tracksdata"}
        for node in tree.body:
            if node.lineno >= step0.lineno:
                break
            names = set()
            if isinstance(node, ast.Import):
                names = {item.name.split(".", 1)[0] for item in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0]}
            self.assertFalse(names & heavy)
        metadata = json.loads((HERE / "kernel/kernel-metadata.json").read_text())
        self.assertEqual(metadata["dataset_sources"], [build.SUPPORT_DATASET])
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertIn('"--no-index", "--find-links"', source)
        self.assertIn("ThreadPoolExecutor", source)
        self.assertIn("full_declared_byte_sha_verification", source)
        self.assertIn('"competition_path_visits": 0', source)
        self.assertNotIn("/kaggle/input/competitions/", source)
        self.assertNotIn('"test").iterdir()', source)

    def test_missing_support_mount_fails_closed_before_heavy_work_in_under_five_seconds(self):
        source = KERNEL.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "working"
            missing = root / "missing_support"
            candidate = source.replace(
                'WORK = Path("/kaggle/working")', f'WORK = Path({str(work)!r})'
            ).replace(
                'SUPPORT_MOUNT = Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")',
                f'SUPPORT_MOUNT = Path({str(missing)!r})',
            )
            script = root / "kernel.py"
            script.write_text(candidate, encoding="utf-8")
            started = time.monotonic()
            completed = subprocess.run(["/Users/agent/Desktop/cell_tracking/.venv313/bin/python", str(script)], text=True, capture_output=True)
            elapsed = time.monotonic() - started
            self.assertNotEqual(completed.returncode, 0)
            receipt = json.loads((work / "APNCC_STEP0.json").read_text())
            self.assertEqual(receipt["status"], "FAIL_CLOSED")
            self.assertFalse(receipt["heavy_work_started"])
            self.assertLess(receipt["elapsed_seconds"], 5.0)
            self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
