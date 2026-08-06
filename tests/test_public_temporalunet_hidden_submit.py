from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_temporalunet_hidden_submit",
    ROOT / "scripts/public_temporalunet_hidden_submit.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicTemporalUNetHiddenSubmitTests(unittest.TestCase):
    def test_revalidation_preserves_recorded_relative_paths_after_binding_resolution(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            package_dir = root / "package"
            output_dir = root / "output"
            package_dir.mkdir()
            output_dir.mkdir()
            relative_package = package_dir.relative_to(ROOT)
            relative_output = output_dir.relative_to(ROOT)
            expected = {
                "package_manifest": {
                    "path": str(relative_package / "PACKAGE_MANIFEST.json")
                },
                "submission": {"path": str(relative_output / "submission.csv")},
            }

            def fake_validate(args):
                self.assertEqual(args.package_dir, relative_package)
                self.assertEqual(args.output_dir, relative_output)
                return expected

            with mock.patch.object(
                MODULE.baseline, "validate_hidden_output", side_effect=fake_validate
            ):
                MODULE.revalidate_hidden(
                    package_dir=package_dir.resolve(),
                    output_dir=output_dir.resolve(),
                    expected=expected,
                )
                with self.assertRaisesRegex(
                    MODULE.HiddenSubmitError, "do not resolve to the predicate inputs"
                ):
                    MODULE.revalidate_hidden(
                        package_dir=package_dir.resolve(),
                        output_dir=(root / "wrong").resolve(),
                        expected=expected,
                    )

    def test_exact_terminal_status_requires_complete_exact_slug(self) -> None:
        receipt = {
            "schema": "celltrack-alpha.cmux-op-receipt.v1",
            "op": "kernel_status",
            "returncode": 0,
            "process_returncode": 0,
            "timed_out": False,
            "semantic_failure": False,
            "argv": ["kaggle", "kernels", "status", MODULE.KERNEL_ID],
            "stdout_scrubbed": 'onebitbrain/g02-v32-submit has status "KernelWorkerStatus.COMPLETE"',
        }
        self.assertTrue(MODULE.exact_status_complete(receipt, kernel_id=MODULE.KERNEL_ID))
        receipt["stdout_scrubbed"] = "KernelWorkerStatus.RUNNING"
        self.assertFalse(MODULE.exact_status_complete(receipt, kernel_id=MODULE.KERNEL_ID))

    def test_exact_output_receipt_binds_version_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            receipt = {
                "schema": "celltrack-alpha.cmux-op-receipt.v1",
                "op": "kernel_output",
                "returncode": 0,
                "process_returncode": 0,
                "timed_out": False,
                "semantic_failure": False,
                "argv": [
                    "kaggle",
                    "kernels",
                    "output",
                    f"{MODULE.KERNEL_ID}/1",
                    "-p",
                    str(output),
                ],
            }
            self.assertTrue(
                MODULE.exact_output_download(
                    receipt,
                    kernel_id=MODULE.KERNEL_ID,
                    version=1,
                    output_dir=output,
                )
            )
            receipt["argv"][3] = f"{MODULE.KERNEL_ID}/2"
            self.assertFalse(
                MODULE.exact_output_download(
                    receipt,
                    kernel_id=MODULE.KERNEL_ID,
                    version=1,
                    output_dir=output,
                )
            )


if __name__ == "__main__":
    unittest.main()
