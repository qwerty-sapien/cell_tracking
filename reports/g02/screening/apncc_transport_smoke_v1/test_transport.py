from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import build_transport_kernel as build


HERE = Path(__file__).resolve().parent
KERNEL = HERE / "kernel/kaggle_apncc_transport.py"


class ApnccTransportTests(unittest.TestCase):
    def test_build_pins_requested_banked_identity_and_all_fragment_references(self):
        receipt = build.build()
        contract = json.loads((HERE / "CONTRACT.json").read_text())
        self.assertEqual(receipt["banked_identity_sha256"], build.BANKED_IDENTITY_SHA256)
        self.assertEqual(contract["input_manifest_sha256"], build.BANKED_IDENTITY_SHA256)
        self.assertEqual(contract["wheel_count"], 62)
        self.assertEqual(contract["request_count"], 264378)
        self.assertEqual(len(contract["banked_output"]["fragments"]), 4)
        self.assertEqual(sum(row["rows"] for row in contract["banked_output"]["fragments"]), 264378)

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
