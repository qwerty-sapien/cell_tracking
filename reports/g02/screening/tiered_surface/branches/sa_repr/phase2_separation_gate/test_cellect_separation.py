#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


step0 = module("kaggle_step0_preflight", HERE / "kaggle_step0_preflight.py")
kg = module("cellect_kernel", HERE / "kaggle_cellect_action_features.py")
ev = module("cellect_evaluate", HERE / "evaluate_cellect_separation.py")
mat = module("cellect_materialize", HERE / "materialize_cellect_gate.py")
asset = module("build_cellect_asset_v4", HERE / "build_cellect_asset_v4.py")
bundle = module(
    "cellect_bundle_v6", HERE / "build_cellect_consolidated_bundle_v6.py"
)
roundtrip = module(
    "verify_cellect_asset_roundtrip", HERE / "verify_cellect_asset_roundtrip.py"
)
cmux_roundtrip = module(
    "cellect_cmux_roundtrip_v4", HERE / "run_cellect_cmux_roundtrip.py"
)
selfcontained = module(
    "build_cellect_selfcontained_kernel",
    HERE / "build_cellect_selfcontained_kernel.py",
)
model = module(
    "cellect_safe_model",
    HERE.parents[6] / "_LATER/cellect/src/celltrack_alpha/cellect_model.py",
)


class CellectSeparationTest(unittest.TestCase):
    def test_selfcontained_kernel_has_no_sibling_step0_dependency(self):
        payload = selfcontained.build(
            HERE / "kaggle_cellect_action_features.py",
            HERE / "kaggle_step0_preflight.py",
        )
        source = payload.decode("utf-8")
        self.assertNotIn(selfcontained.STEP0_IMPORT, source)
        self.assertIn("def run_cellect_step0(", source)
        self.assertIn("expected_request_manifest_sha256=REQUEST_MANIFEST_SHA", source)
        self.assertIn('record["action_class"]', source)
        self.assertIn('record["identity"]', source)

    def test_kernel_metadata_uses_one_consolidated_dataset(self):
        for name in ("kaggle_kernel_metadata.json", "kernel-metadata.json"):
            metadata = json.loads((HERE / name).read_text())
            self.assertEqual(
                metadata["dataset_sources"],
                ["onebitbrain/g02-cellect-r7-published-assets-v1"],
            )
            self.assertNotIn(
                "onebitbrain/g02-cellect-r7-action-requests-v1",
                metadata["dataset_sources"],
            )

    def test_all_input_step0_precedes_heavy_work(self):
        source = (HERE / "kaggle_cellect_action_features.py").read_text()
        step0_call = source.index("STEP0 = (")
        self.assertLess(step0_call, source.index("import numpy as np"))
        self.assertLess(step0_call, source.index("import torch"))
        self.assertLess(step0_call, source.index("subprocess.check_call"))
        self.assertLess(step0_call, source.index("load_published_backbone"))

    def test_kernel_scopes_content_inventory_to_private_cellect_dataset(self):
        self.assertEqual(
            kg.CELLECT_INPUT_ROOT,
            Path(
                "/kaggle/input/datasets/onebitbrain/"
                "g02-cellect-r7-published-assets-v1"
            ),
        )
        source = (HERE / "kaggle_cellect_action_features.py").read_text()
        self.assertIn("run_cellect_step0(\n        CELLECT_INPUT_ROOT,", source)
        self.assertNotIn(
            "run_cellect_step0(\n        Path(\"/kaggle/input\"),",
            source,
        )

    def test_missing_request_manifest_aborts_in_under_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            with self.assertRaisesRegex(
                step0.Step0InputError, "bundle manifest content resolution"
            ):
                step0.run_cellect_step0(
                    root,
                    root / "work",
                    expected_bundle_manifest_sha256="0" * 64,
                    expected_request_manifest_sha256="3" * 64,
                    expected_checkpoint_sha256="1" * 64,
                    expected_population_manifest_sha256="2" * 64,
                    expected_wheels={},
                )
            self.assertLess(time.monotonic() - started, 5.0)

    def test_empty_request_manifest_aborts_in_under_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / bundle.REQUEST_MANIFEST_NAME
            request_path.write_text(
                json.dumps(
                    {
                        "request_count": 0,
                        "request_file_count": 0,
                        "requests": [],
                    }
                )
            )
            population_path = root / bundle.POPULATION_MANIFEST_NAME
            population_path.write_text("{}")
            manifest = {
                "status": "LOCAL_CONSOLIDATED_BYTE_EXACT",
                "files": [
                    {
                        "path": request_path.name,
                        "bytes": request_path.stat().st_size,
                        "sha256": step0.sha256(request_path),
                        "role": "request_manifest",
                    },
                    {
                        "path": population_path.name,
                        "bytes": population_path.stat().st_size,
                        "sha256": step0.sha256(population_path),
                        "role": "population_manifest",
                    },
                ],
            }
            manifest_path = root / step0.BUNDLE_MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest))
            started = time.monotonic()
            with self.assertRaisesRegex(step0.Step0InputError, "empty or has invalid"):
                step0.run_cellect_step0(
                    root,
                    root / "work",
                    expected_bundle_manifest_sha256=step0.sha256(manifest_path),
                    expected_request_manifest_sha256=step0.sha256(request_path),
                    expected_checkpoint_sha256="1" * 64,
                    expected_population_manifest_sha256=step0.sha256(population_path),
                    expected_wheels={},
                )
            self.assertLess(time.monotonic() - started, 5.0)

    def test_step0_discovers_opaque_request_content_and_returns_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "opaque" / "blob-a"
            payload.parent.mkdir()
            payload.write_bytes(b"request-payload")
            population = root / "other" / "blob-b"
            population.parent.mkdir()
            population.write_text("{}")
            loader = root / "loader" / "blob-c"
            loader.parent.mkdir()
            loader.write_text("# safe loader\n")
            checkpoint = root / "weights" / "checkpoint.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            request_manifest = root / "manifest" / "blob-d"
            request_manifest.parent.mkdir()
            request_manifest.write_text(
                json.dumps(
                    {
                        "status": "FROZEN_COMPLETE",
                        "request_count": 1,
                        "request_file_count": 1,
                        "population_manifest_sha256": step0.sha256(population),
                        "requests": [
                            {
                                "path": "transport/name-that-does-not-exist.npz",
                                "bytes": payload.stat().st_size,
                                "sha256": step0.sha256(payload),
                                "action_class": "delete",
                                "identity": "train:44b6_opaque",
                                "action_count": 1,
                            }
                        ],
                    }
                )
            )
            bundle_manifest = root / "control" / "blob-e"
            bundle_manifest.parent.mkdir()
            bundle_manifest.write_text(
                json.dumps(
                    {
                        "status": "LOCAL_CONSOLIDATED_BYTE_EXACT",
                        "files": [
                            {
                                "path": "canonical/request.json",
                                "bytes": request_manifest.stat().st_size,
                                "sha256": step0.sha256(request_manifest),
                                "role": "request_manifest",
                            },
                            {
                                "path": "canonical/population.json",
                                "bytes": population.stat().st_size,
                                "sha256": step0.sha256(population),
                                "role": "population_manifest",
                            },
                            {
                                "path": "canonical/request.npz",
                                "bytes": payload.stat().st_size,
                                "sha256": step0.sha256(payload),
                                "role": "feature_request",
                            },
                            {
                                "path": "canonical/loader.py",
                                "bytes": loader.stat().st_size,
                                "sha256": step0.sha256(loader),
                                "role": "safe_loader",
                            },
                        ],
                    }
                )
            )
            result = step0.run_cellect_step0(
                root,
                root / "work",
                expected_bundle_manifest_sha256=step0.sha256(bundle_manifest),
                expected_request_manifest_sha256=step0.sha256(request_manifest),
                expected_checkpoint_sha256=step0.sha256(checkpoint),
                expected_population_manifest_sha256=step0.sha256(population),
                expected_wheels={},
            )
            self.assertEqual(result["request_file_count"], 1)
            self.assertEqual(result["request_count"], 1)
            self.assertEqual(result["request_records"][0]["action_class"], "delete")
            self.assertEqual(
                result["request_records"][0]["identity"], "train:44b6_opaque"
            )
            self.assertEqual(
                Path(result["request_records"][0]["resolved_path"]), payload
            )

    def test_cmux_command_plan_has_no_kernel_or_submission_path(self):
        plan = cmux_roundtrip.command_plan(
            "/trusted/cmux/kaggle",
            Path("/staging"),
            Path("/download"),
        )
        self.assertEqual(set(plan), {"upload", "status", "files", "download"})
        flattened = " ".join(part for command in plan.values() for part in command)
        self.assertNotIn("kernels", flattened)
        self.assertNotIn("submit", flattened)
        self.assertIn("datasets version", flattened)
        self.assertIn("--format json", flattened)

    def test_remote_receipt_requires_exact_cli_schema_and_bytes(self):
        payload = b"uncompressed tar fixture"
        digest = __import__("hashlib").sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = root / "download"
            download.mkdir()
            (download / "checkpoint.tar").write_bytes(payload)
            manifest = {
                "dataset": {
                    "id": cmux_roundtrip.DATASET_ID,
                    "target_version": 4,
                    "version_before": 3,
                    "visibility": "private",
                },
                "control_metadata": {
                    "path": "dataset-metadata.json",
                    "bytes": 0,
                    "sha256": "unused-for-remote-download",
                },
                "files": [
                    {
                        "path": "checkpoint.tar",
                        "bytes": len(payload),
                        "sha256": digest,
                    }
                ],
                "checkpoint_transport": {
                    "container_path": "checkpoint.tar",
                    "container_bytes": len(payload),
                    "container_sha256": digest,
                    "member": kg.SAFE_WEIGHT_NAME,
                    "member_bytes": 123,
                    "member_sha256": kg.WEIGHT_SHA,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            status_path = root / "status.json"
            status_path.write_text(
                json.dumps({"status": "ready", "current_version_number": 4})
            )
            files_path = root / "files.json"
            files_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "checkpoint.tar",
                            "size": len(payload),
                            "creationDate": "fixture",
                        }
                    ]
                )
            )
            download_path = root / "download.json"
            download_path.write_text(
                json.dumps(
                    {
                        "dataset_id": cmux_roundtrip.DATASET_ID,
                        "version": 4,
                        "method": "authenticated_kaggle_sdk_cmux",
                        "success": True,
                    }
                )
            )
            with mock.patch.object(
                roundtrip,
                "verify_container",
                return_value={
                    "container_path": "checkpoint.tar",
                    "container_bytes": len(payload),
                    "container_sha256": digest,
                    "member": kg.SAFE_WEIGHT_NAME,
                    "member_bytes": 123,
                    "member_sha256": kg.WEIGHT_SHA,
                    "compression": "none",
                },
            ):
                result = roundtrip.verify_download(
                    download,
                    manifest_path,
                    "authenticated_kaggle_cli_cmux",
                    status_path,
                    files_path,
                    None,
                    download_path,
                )
                self.assertEqual(result["status"], "BYTE_EXACT_ROUNDTRIP_PASS")
                self.assertEqual(
                    result["authenticated_cli_receipts"]["version"], 4
                )
                files_path.write_text(
                    json.dumps(
                        [{"name": "checkpoint.tar", "size": len(payload) - 1}]
                    )
                )
                with self.assertRaises(roundtrip.RoundtripError):
                    roundtrip.verify_download(
                        download,
                        manifest_path,
                        "authenticated_kaggle_cli_cmux",
                        status_path,
                        files_path,
                        None,
                        download_path,
                    )

    def test_remote_expanded_tar_member_is_exact_and_403_is_banked(self):
        payload = b"authenticated checkpoint bytes"
        digest = __import__("hashlib").sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            download = root / "download"
            member = download / "checkpoint" / kg.SAFE_WEIGHT_NAME
            member.parent.mkdir(parents=True)
            member.write_bytes(payload)
            manifest = {
                "dataset": {
                    "id": cmux_roundtrip.DATASET_ID,
                    "target_version": 4,
                    "version_before": 3,
                    "visibility": "private",
                },
                "control_metadata": {
                    "path": "dataset-metadata.json",
                    "bytes": 0,
                    "sha256": "unused-for-remote-download",
                },
                "files": [
                    {
                        "path": "checkpoint.tar",
                        "bytes": 10240,
                        "sha256": "local-container-only",
                    }
                ],
                "checkpoint_transport": {
                    "container_path": "checkpoint.tar",
                    "container_bytes": 10240,
                    "container_sha256": "local-container-only",
                    "member": kg.SAFE_WEIGHT_NAME,
                    "member_bytes": len(payload),
                    "member_sha256": digest,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            status_path = root / "status.json"
            status_path.write_text(
                json.dumps({"status": "ready", "current_version_number": 4})
            )
            files_error_path = root / "files.error.txt"
            files_error_path.write_text(
                "403 Client Error: Forbidden for url: "
                "DatasetApiService/ListDatasetFiles"
            )
            download_path = root / "download.json"
            download_path.write_text(
                json.dumps(
                    {
                        "dataset_id": cmux_roundtrip.DATASET_ID,
                        "version": 4,
                        "method": "authenticated_kaggle_sdk_cmux",
                        "success": True,
                    }
                )
            )
            result = roundtrip.verify_download(
                download,
                manifest_path,
                "authenticated_kaggle_cli_cmux",
                status_path,
                None,
                files_error_path,
                download_path,
            )
            self.assertEqual(result["status"], "BYTE_EXACT_ROUNDTRIP_PASS")
            self.assertEqual(
                result["checkpoint_transport"]["materialization"],
                "kaggle_expanded_uncompressed_tar",
            )
            member.write_bytes(payload + b"drift")
            with self.assertRaises(roundtrip.RoundtripError):
                roundtrip.verify_download(
                    download,
                    manifest_path,
                    "authenticated_kaggle_cli_cmux",
                    status_path,
                    None,
                    files_error_path,
                    download_path,
                )

    def test_asset_v4_container_is_uncompressed_deterministic_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"x")
            with mock.patch.object(asset, "CHECKPOINT_BYTES", 1), mock.patch.object(
                asset,
                "CHECKPOINT_SHA256",
                __import__("hashlib").sha256(b"x").hexdigest(),
            ):
                container = root / asset.CONTAINER_NAME
                info = tarfile.TarInfo(asset.SAFE_MEMBER_NAME)
                info.size = 1
                info.mtime = 0
                with tarfile.open(container, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
                    archive.addfile(info, io.BytesIO(b"x"))
                result = asset.verify_container(container)
            self.assertEqual(result["compression"], "none")
            self.assertEqual(result["member"], asset.SAFE_MEMBER_NAME)
            self.assertEqual(result["member_bytes"], 1)

    def test_weight_resolution_from_uncompressed_tar_is_content_addressed(self):
        payload = b"authenticated checkpoint bytes"
        expected = __import__("hashlib").sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            work_root = root / "work"
            input_root.mkdir()
            with tarfile.open(input_root / "checkpoint.tar", mode="w:") as archive:
                info = tarfile.TarInfo(kg.SAFE_WEIGHT_NAME)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            resolved = kg.resolve_weight_by_content(input_root, work_root, expected)
            self.assertEqual(resolved.name, kg.SAFE_WEIGHT_NAME)
            self.assertEqual(resolved.read_bytes(), payload)

    def test_weight_resolution_ignores_wrong_direct_file_and_fails_without_match(self):
        expected = __import__("hashlib").sha256(b"expected").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            (input_root / "plausible.pth").write_bytes(b"transformed")
            with self.assertRaisesRegex(kg.FeatureError, '"match_count": 0'):
                kg.resolve_weight_by_content(input_root, root / "work", expected)

    def test_weight_resolution_rejects_unsafe_matching_tar_member(self):
        payload = b"authenticated checkpoint bytes"
        expected = __import__("hashlib").sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            with tarfile.open(input_root / "checkpoint.tar", mode="w:") as archive:
                info = tarfile.TarInfo(f"../{kg.SAFE_WEIGHT_NAME}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(kg.FeatureError, "unsafe or unexpected"):
                kg.resolve_weight_by_content(input_root, root / "work", expected)

    def test_weight_resolution_fails_on_duplicate_exact_matches(self):
        payload = b"authenticated checkpoint bytes"
        expected = __import__("hashlib").sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            (input_root / "first.bin").write_bytes(payload)
            (input_root / "second.pth").write_bytes(payload)
            with self.assertRaisesRegex(kg.FeatureError, '"match_count": 2'):
                kg.resolve_weight_by_content(input_root, root / "work", expected)

    def test_weight_resolution_rejects_compressed_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            (input_root / "checkpoint.tar").write_bytes(b"not a tar")
            with self.assertRaisesRegex(kg.FeatureError, "invalid uncompressed tar"):
                kg.resolve_weight_by_content(
                    input_root,
                    root / "work",
                    __import__("hashlib").sha256(b"expected").hexdigest(),
                )

    def test_action_features_identity(self):
        value = np.arange(128, dtype=np.float32).reshape(2, 64)
        result = kg.action_features(value, value)
        self.assertEqual(result.shape, (2, 8))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.allclose(result[:, 0], 1.0))
        self.assertTrue(np.allclose(result[:, 1:3], 0.0))

    def test_permutation_is_deterministic_and_node_consistent(self):
        values = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
        nodes = np.asarray([9, 2, 9, 7])
        first = kg.permutation(values, nodes, "train:44b6_fixture", 3)
        second = kg.permutation(values, nodes, "train:44b6_fixture", 3)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first[0], first[2]))

    def test_tile_grouping_matches_unique_job_count_and_coverage(self):
        request = {
            "source_t": np.asarray([0, 0, 1, 1], dtype=np.int16),
            "target_t": np.asarray([1, 1, 2, 2], dtype=np.int16),
            "source_zyx": np.asarray(
                [[0, 0, 0], [15, 63, 63], [16, 64, 64], [17, 65, 65]]
            ),
            "target_zyx": np.asarray(
                [[0, 0, 0], [16, 64, 64], [16, 64, 64], [32, 128, 128]]
            ),
        }
        source = kg.group_endpoint_indices(
            request["source_t"], request["source_zyx"], target_role=False
        )
        target = kg.group_endpoint_indices(
            request["target_t"], request["target_zyx"], target_role=True
        )
        self.assertEqual(
            len(source.keys() | target.keys()), kg.count_tile_jobs(request)
        )
        self.assertEqual(sum(len(value) for value in source.values()), 4)
        self.assertEqual(sum(len(value) for value in target.values()), 4)

    def test_auc_midrank(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        self.assertEqual(ev.auc(y, np.asarray([0.0, 0.0, 1.0, 1.0])), 1.0)
        self.assertEqual(ev.auc(y, np.zeros(4)), 0.5)

    def test_retention_respects_budget(self):
        y = np.asarray([1, 0, 1, 0], dtype=np.uint8)
        score = np.asarray([4.0, 3.0, 2.0, 1.0])
        video = np.asarray(["v", "v", "v", "v"])
        retention, _good, harmful, rows = ev.retained(y, score, video, {"v": 1})
        self.assertEqual(retention, 1.0)
        self.assertEqual(harmful, 1)
        self.assertTrue(rows[0]["budget_pass"])

    def test_fold_is_stable(self):
        identity = "train:44b6_fixture"
        self.assertEqual(mat.fold(identity), mat.fold(identity))
        self.assertIn(mat.fold(identity), range(5))

    def test_b0_is_sealed_before_a1_a0(self):
        fold = np.repeat(np.arange(5, dtype=np.int8), 4)
        y = np.tile(np.asarray([0, 1, 0, 1], dtype=np.uint8), 5)
        base = np.arange(len(y), dtype=np.float64)
        data = {
            "identity": np.full(len(y), "v"),
            "fold": fold,
            "y": y,
            "b0": np.column_stack((base, base % 3)),
            "a1": np.column_stack((y, base % 5)),
            "a0": np.column_stack((np.roll(y, 1), base % 7)),
            "budgets": {"v": 2},
        }
        sealed = ev.evaluate_b0_class(data, "delete")
        result = ev.evaluate_class(data, "delete", sealed)
        self.assertEqual(len(sealed["folds"]), 5)
        self.assertEqual(len(result["folds"]), 5)
        for sealed_fold, result_fold in zip(sealed["folds"], result["folds"]):
            self.assertEqual(sealed_fold["B0"], result_fold["arms"]["B0"])

    def test_numeric_loader_has_no_unsafe_pickle_fallback(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            with mock.patch.object(
                model.torch,
                "load",
                side_effect=TypeError("weights_only unsupported"),
            ) as loader:
                with self.assertRaises(model.CellectModelError):
                    model._load_numeric_state(Path(checkpoint.name))
        loader.assert_called_once_with(
            Path(checkpoint.name), map_location="cpu", weights_only=True
        )


if __name__ == "__main__":
    unittest.main()
