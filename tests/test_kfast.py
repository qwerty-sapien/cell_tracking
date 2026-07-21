from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kfast/run.py"
METADATA = ROOT / "kfast/kernel-metadata.json"
CHECKPOINT = ROOT / "reports/g02/t03-v6.pt"


def load_module():
    spec = importlib.util.spec_from_file_location("kfast_run", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load_module()


class KfastTests(unittest.TestCase):
    def test_embedded_package_identity(self) -> None:
        package = base64.b85decode(run.EMBED.encode("ascii"))
        self.assertEqual(hashlib.sha256(package).hexdigest(), run.PACKAGE_SHA)
        self.assertLess(len(SOURCE.read_bytes()), 1_000_000)

    def test_noncanonical_mount_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mounted = root / "t03-output-v6/t03"
            mounted.mkdir(parents=True)
            (mounted / "checkpoint.pt").symlink_to(CHECKPOINT)
            deps = mounted / "deps-v1"
            deps.mkdir()
            for name, version in run.REQ_DISTS.items():
                (deps / f"{name}-{version}.dist-info").mkdir()
            competition = root / "competitions/biohub-cell-tracking-during-development"
            (competition / "train").mkdir(parents=True)
            (competition / "test").mkdir()
            checkpoint = run.exact_checkpoint(root)
            self.assertEqual(checkpoint, mounted / "checkpoint.pt")
            self.assertEqual(run.exact_dependencies(checkpoint), deps)
            self.assertEqual(run.competition_root(root), competition)

    def test_private_kernel_sources(self) -> None:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["kernel_sources"], ["onebitbrain/biohub-t03-f0"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")

    def test_preflight_requires_ckdtree_and_reports_scipy(self) -> None:
        self.assertIn("from scipy.spatial import cKDTree", run.PREFLIGHT_CODE)
        self.assertIn("assert callable(cKDTree)", run.PREFLIGHT_CODE)
        self.assertIn("scipy=", run.PREFLIGHT_CODE)

    def test_wrapper_deadline_terminates_subprocess(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "deadline exceeded"):
            run._run_deadlined(
                [sys.executable, "-B", "-c", "import time; time.sleep(10)"],
                dict(os.environ), time.time() + 0.1,
            )
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
