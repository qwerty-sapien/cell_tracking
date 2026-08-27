from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path


KERNEL = Path(__file__).parent / "kaggle_cpu_kernel" / "kaggle_dinov2_cpu.py"


def load_step0_prefix() -> dict:
    source = KERNEL.read_text(encoding="utf-8")
    prefix = source.split("\nSTARTED = time.time()", 1)[0]
    namespace: dict = {"__name__": "dinov2_step0_test"}
    exec(compile(prefix, str(KERNEL), "exec"), namespace)
    return namespace


class DINOv2Step0FailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_step0_prefix()

    def assert_fast_failure(self, root: Path) -> str:
        started = time.perf_counter()
        with self.assertRaises(self.module["Step0Error"]) as caught:
            self.module["run_step0"](root)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0)
        return str(caught.exception)

    def test_missing_request_manifest_aborts_under_five_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            message = self.assert_fast_failure(Path(temp))
        self.assertEqual(message, "exact DINOv2 private-dataset manifest missing")

    def test_empty_request_manifest_aborts_under_five_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = (
                root / "datasets" / "onebitbrain" / "g02-dinov2-r5-cpu-input-v1"
            )
            bundle = dataset / "dinov2_cpu_bundle"
            bundle.mkdir(parents=True)
            (dataset / "dinov2_cpu_bundle_manifest.json").write_text(
                json.dumps({}), encoding="ascii"
            )
            (bundle / "dinov2_cpu_bundle_inventory.json").write_text(
                json.dumps({"files": []}), encoding="ascii"
            )
            message = self.assert_fast_failure(root)
        self.assertEqual(message, "DINOv2 required-input contract drifted")


if __name__ == "__main__":
    unittest.main()
