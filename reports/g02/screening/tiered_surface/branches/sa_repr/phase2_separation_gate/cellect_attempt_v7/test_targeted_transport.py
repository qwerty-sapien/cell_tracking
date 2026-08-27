from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
KERNEL = HERE / "kaggle_cellect_v7.py"
BUNDLE = PARENT / "CONSOLIDATED_DATASET_V6_MANIFEST.json"
EXPANDED = Path(
    "/private/tmp/cellect-r7-v6-cmux-roundtrip.20260729/download"
)
RAW = Path("/private/tmp/cellect-r7-v6-consolidated.UQi4ls/assets")


def load_kernel():
    spec = importlib.util.spec_from_file_location("cellect_targeted_v7", KERNEL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generated CELLECT v7 kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link_declared(source: Path, destination: Path, *, raw_tar: bool) -> None:
    names = [
        "action_request_manifest.json",
        "population_manifest.json",
        "donfig-0.8.1.post1-py3-none-any.whl",
        "numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "zarr-3.2.1-py3-none-any.whl",
    ]
    if raw_tar:
        names.append("cellect_uext_x3rd_149_0_4p6540.tar")
    for name in names:
        os.link(source / name, destination / name)
    if not raw_tar:
        checkpoint = destination / "cellect_uext_x3rd_149_0_4p6540"
        checkpoint.mkdir()
        os.link(
            source
            / "cellect_uext_x3rd_149_0_4p6540"
            / "cellect_uext_x3rd_149_0_4p6540.pth",
            checkpoint / "cellect_uext_x3rd_149_0_4p6540.pth",
        )


class TargetedTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = load_kernel()
        cls.bundle = json.loads(BUNDLE.read_text())

    def test_every_wheel_filename_is_pep427_and_bytes_match_manifest(self):
        records = [
            row for row in self.bundle["files"] if row.get("role") == "offline_wheel"
        ]
        self.assertEqual(len(records), 3)
        for record in records:
            filename = Path(record["path"]).name
            try:
                parse_wheel_filename(filename)
            except InvalidWheelFilename as error:
                self.fail(f"invalid wheel filename {filename}: {error}")
            payload = (EXPANDED / filename).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
            self.assertNotRegex(filename, r"^wheel-[0-9a-f]{64}\.whl$")

    def test_generated_install_uses_no_index_find_links(self):
        source = KERNEL.read_text()
        self.assertIn('"--no-index"', source)
        self.assertIn('"--find-links"', source)
        self.assertIn('str(STEP0["wheel_directory"])', source)

    def test_expanded_checkpoint_member_targeted_step0_is_under_five_seconds(self):
        with tempfile.TemporaryDirectory() as work:
            with mock.patch.object(
                Path, "rglob", side_effect=AssertionError("recursive walk forbidden")
            ):
                started = time.monotonic()
                result = self.kernel.run_cellect_step0(EXPANDED, Path(work))
                wall = time.monotonic() - started
        self.assertLess(wall, 5.0)
        self.assertLess(result["elapsed_seconds"], 5.0)
        self.assertGreater(result["request_count"], 0)
        self.assertEqual(result["request_file_count"], 77)
        self.assertEqual(result["competition_paths_visited"], 0)
        self.assertEqual(result["declared_paths_hashed"], 6)
        self.assertEqual(
            result["population_manifest_sha256"],
            "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
        )
        self.assertTrue(result["checkpoint_member_sha256_verified"])
        self.assertFalse(result["checkpoint_container_sha256_verified"])

    def test_raw_uncompressed_tar_uses_member_sha_not_container_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as work:
            root = Path(directory)
            link_declared(RAW, root, raw_tar=True)
            result = self.kernel.run_cellect_step0(root, Path(work))
        self.assertEqual(result["checkpoint_representation"], "raw_uncompressed_tar")
        self.assertTrue(result["checkpoint_member_sha256_verified"])
        self.assertFalse(result["checkpoint_container_sha256_verified"])

    def test_genuine_checkpoint_member_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as work:
            root = Path(directory)
            link_declared(EXPANDED, root, raw_tar=False)
            checkpoint = (
                root
                / "cellect_uext_x3rd_149_0_4p6540"
                / "cellect_uext_x3rd_149_0_4p6540.pth"
            )
            payload = bytearray(checkpoint.read_bytes())
            payload[0] ^= 1
            checkpoint.unlink()
            checkpoint.write_bytes(payload)
            with self.assertRaisesRegex(
                self.kernel.Step0InputError, "checkpoint member SHA-256 mismatch"
            ):
                self.kernel.run_cellect_step0(root, Path(work))

    def test_request_payload_hash_is_checked_immediately_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.npz"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(
                self.kernel.FeatureError, "request payload SHA-256 mismatch"
            ):
                self.kernel.verify_request_record(
                    {
                        "resolved_path": str(path),
                        "bytes": len(b"corrupt"),
                        "sha256": "0" * 64,
                        "action_class": "delete",
                        "identity": "train:44b6_fixture",
                    }
                )


if __name__ == "__main__":
    unittest.main()
