from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import mkvs45  # noqa: E402


class ViewDiagnosticPackageTests(unittest.TestCase):
    def test_package_is_bounded_and_pins_the_frozen_held_out_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "vs45pkg"
            report = mkvs45.build_package(
                ROOT,
                output,
                registry=ROOT / "reports/g02/dataset_registry_v1.json",
                support_root=ROOT / "external/Tracking_Support",
            )
            self.assertTrue(report["strictly_under_budget"])
            self.assertFalse(report["submission_generation"])
            self.assertLess(report["aggregate_bytes"], 1_000_000)

            config_bytes = (output / mkvs45.CONFIG).read_bytes()
            config = json.loads(config_bytes)
            self.assertFalse(config["submission_generation"])
            self.assertFalse(config["competition_submission"])
            self.assertEqual([row["dataset"] for row in config["targets"]], list(mkvs45.TARGETS))
            self.assertEqual(
                [(row["name"], tuple(row["views"])) for row in config["view_sets"]],
                list(mkvs45.VIEW_SETS),
            )

            manifest = json.loads((output / mkvs45.MANIFEST).read_text(encoding="ascii"))
            identity = manifest["identity"]
            self.assertEqual(
                manifest["identity_sha256"],
                hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
                ).hexdigest(),
            )
            self.assertEqual(identity["diagnostic"]["config_sha256"], hashlib.sha256(config_bytes).hexdigest())
            self.assertEqual(identity["diagnostic"]["device"], "cuda:0")
            self.assertEqual(identity["diagnostic"]["graph_variant"], "a0")

            expected_files = {
                "bh45.py",
                "bh45att.json",
                "bh45man.json",
                "bh45src.zip",
                "kernel-metadata.json",
                "vs45.py",
                "vs45cfg.json",
                "vs45man.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)

            notebook_dir = Path(temporary) / "vs45nb"
            notebook = mkvs45.build_notebook(output, notebook_dir)
            self.assertTrue(notebook["strictly_under_budget"])
            self.assertFalse(notebook["submission_generation"])
            self.assertLess(notebook["notebook_bytes"], 1_000_000)
            metadata = json.loads((notebook_dir / "kernel-metadata.json").read_text(encoding="ascii"))
            self.assertEqual(metadata["code_file"], mkvs45.NOTEBOOK)
            self.assertEqual(metadata["kernel_type"], "notebook")
            decoded = json.loads((notebook_dir / mkvs45.NOTEBOOK).read_text(encoding="ascii"))
            cell = "".join(decoded["cells"][0]["source"])
            self.assertIn("vs45pkg", cell)
            self.assertIn("vs45 inner return code", cell)

    def test_batch_profile_is_private_bounded_and_records_execution_width(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "b45pkg"
            report = mkvs45.build_package(
                ROOT,
                output,
                registry=ROOT / "reports/g02/dataset_registry_v1.json",
                support_root=ROOT / "external/Tracking_Support",
                view_sets=mkvs45.BATCH_PROFILE_VIEWS,
                kernel_slug=mkvs45.BATCH_PROFILE_SLUG,
                deadline_seconds=mkvs45.BATCH_PROFILE_DEADLINE_SECONDS,
            )
            self.assertTrue(report["strictly_under_budget"])
            self.assertFalse(report["submission_generation"])
            self.assertEqual(report["kernel"], "onebitbrain/bh-v45b8")
            self.assertEqual(report["deadline_seconds"], mkvs45.BATCH_PROFILE_DEADLINE_SECONDS)
            self.assertEqual(
                report["view_sets"],
                [
                    {"name": "v8b1", "views": list(range(8)), "view_batch_size": 1},
                    {"name": "v8b8", "views": list(range(8)), "view_batch_size": 8},
                ],
            )
            config = json.loads((output / mkvs45.CONFIG).read_text(encoding="ascii"))
            self.assertEqual(config["view_sets"], report["view_sets"])
            manifest = json.loads((output / mkvs45.MANIFEST).read_text(encoding="ascii"))
            self.assertEqual(manifest["identity"]["kernel_slug"], mkvs45.BATCH_PROFILE_SLUG)
            self.assertEqual(manifest["identity"]["diagnostic"]["view_sets"], report["view_sets"])
            self.assertEqual(
                manifest["identity"]["diagnostic"]["deadline_seconds"],
                mkvs45.BATCH_PROFILE_DEADLINE_SECONDS,
            )
            metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="ascii"))
            self.assertEqual(metadata["id"], report["kernel"])
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])

            notebook_dir = Path(temporary) / "b45nb"
            notebook = mkvs45.build_notebook(output, notebook_dir, kernel_slug=mkvs45.BATCH_PROFILE_SLUG)
            self.assertEqual(notebook["kernel"], report["kernel"])
            notebook_metadata = json.loads((notebook_dir / "kernel-metadata.json").read_text(encoding="ascii"))
            self.assertEqual(notebook_metadata["id"], report["kernel"])


if __name__ == "__main__":
    unittest.main()
