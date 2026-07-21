from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
import zipfile

from scripts import pkgbh


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class KagglePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.support = self.repo / "external" / "Tracking_Support"
        for relative in pkgbh.SOURCE_PATHS:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {relative}\n", encoding="ascii")

        records = []
        for index in range(4):
            dataset = f"d{index}"
            records.append(
                {
                    "split": "test",
                    "dataset": dataset,
                    "video_id": f"test:{dataset}",
                    "exact_identity_sha256": hashlib.sha256(dataset.encode("ascii")).hexdigest(),
                    "compact_metadata": {"image_shape": [100, 8, 16, 16], "ignored": "large"},
                    "member_vector": [{"ignored": True}],
                }
            )
        records.append(
            {
                "split": "train",
                "dataset": "train0",
                "video_id": "train:train0",
                "exact_identity_sha256": "0" * 64,
                "compact_metadata": {"image_shape": [100, 8, 16, 16]},
            }
        )
        self.registry = self.repo / "reports" / "g02" / "dataset_registry_v1.json"
        self.registry.parent.mkdir(parents=True, exist_ok=True)
        self.registry.write_text(
            json.dumps({"artifact": "dataset_registry_v1", "records": records}),
            encoding="utf-8",
        )

        fixture_values = {
            "ARTIFACT_MANIFEST.json": b"manifest\n",
            "weights/unet_transformer/split_0/config.json": b"{}\n",
            "weights/unet_transformer/split_0/edge_predictor_best.pth": b"checkpoint\x00",
            "repo/src/biohub_tracking/models/temporal_unet.py": b"class TemporalUNet3D: pass\n",
            "repo/src/biohub_tracking/models/simple_node_transformer.py": b"class SimpleNodeTransformer: pass\n",
            "repo/scripts/train_unet_transformer.py": b"# training provenance\n",
        }
        for relative, value in fixture_values.items():
            path = self.support / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        wheels = self.support / "wheels"
        wheels.mkdir(parents=True)
        (wheels / "zarr-3.2.1-py3-none-any.whl").write_bytes(b"offline-wheel-fixture")
        self.expected = {
            "manifest_sha256": _sha(fixture_values["ARTIFACT_MANIFEST.json"]),
            "config_sha256": _sha(fixture_values["weights/unet_transformer/split_0/config.json"]),
            "source_sha256": pkgbh.source_tree_sha256(self.support),
            "weight_sha256": _sha(fixture_values["weights/unet_transformer/split_0/edge_predictor_best.pth"]),
        }

    def build(self, output_name: str = "pkg1") -> tuple[Path, dict]:
        output = self.root / output_name
        report = pkgbh.build_package(
            self.repo,
            output,
            registry=self.registry,
            support_root=self.support,
            expected_artifact_hashes=self.expected,
        )
        return output, report

    def test_compact_registry_contains_only_runner_test_identity(self) -> None:
        compact = json.loads(pkgbh.compact_registry(self.registry))
        self.assertEqual(compact["artifact"], "dataset_registry_v1")
        self.assertEqual(compact["counts"], {"test_videos": 4, "videos": 4})
        self.assertEqual([row["video_id"] for row in compact["records"]], [f"test:d{i}" for i in range(4)])
        self.assertEqual(
            set(compact["records"][0]),
            {"compact_metadata", "dataset", "exact_identity_sha256", "split", "video_id"},
        )
        self.assertNotIn("member_vector", pkgbh.canonical_json(compact).decode("ascii"))

    def test_build_is_deterministic_private_and_strictly_under_budget(self) -> None:
        first, first_report = self.build("pkg1")
        second, second_report = self.build("pkg2")
        first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
        self.assertEqual(first_files, second_files)
        self.assertEqual(
            set(first_files),
            {
                pkgbh.ENTRYPOINT_NAME,
                pkgbh.ARCHIVE_NAME,
                pkgbh.ATTACHMENT_NAME,
                pkgbh.MANIFEST_NAME,
                pkgbh.METADATA_NAME,
            },
        )
        self.assertFalse(any(path.is_dir() for path in first.iterdir()))
        self.assertEqual(first_report["aggregate_bytes"], second_report["aggregate_bytes"])
        self.assertLess(first_report["aggregate_bytes"], 1_000_000)
        self.assertTrue(first_report["strictly_under_budget"])
        self.assertEqual(first_report["file_count"], 5)

        metadata = json.loads(first_files[pkgbh.METADATA_NAME])
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["code_file"], pkgbh.ENTRYPOINT_NAME)
        self.assertEqual(metadata["dataset_sources"], [pkgbh.SUPPORT_REF])
        self.assertLessEqual(len(pkgbh.KERNEL_SLUG), 15)
        self.assertEqual(pkgbh.SUPPORT_REF, "pilkwang/biohub-tracking-support-pack-50ep-v1")
        for name in first_files:
            if name != pkgbh.METADATA_NAME:
                self.assertLessEqual(len(name), 15)

    def test_manifests_cover_hashes_bytes_archive_and_attachment(self) -> None:
        output, report = self.build()
        package = json.loads((output / pkgbh.MANIFEST_NAME).read_text(encoding="ascii"))
        identity = package["identity"]
        self.assertEqual(package["identity_sha256"], _sha(pkgbh.canonical_json(identity)))
        self.assertEqual(identity["package_bytes"], sum(path.stat().st_size for path in output.iterdir() if path.is_file()))
        self.assertEqual(identity["runtime"]["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(identity["runtime"]["deadline_minutes"], 120)
        self.assertEqual(identity["runtime"]["default_view_batch_size"], 1)
        self.assertTrue(identity["runtime"]["accelerated_score_gate_required"])
        for item in identity["files"]:
            path = output / item["path"]
            self.assertEqual(item["bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], pkgbh.sha256_file(path))

        attachment = json.loads((output / pkgbh.ATTACHMENT_NAME).read_text(encoding="ascii"))
        attach_identity = attachment["identity"]
        self.assertEqual(attachment["identity_sha256"], _sha(pkgbh.canonical_json(attach_identity)))
        self.assertFalse(attach_identity["publish"])
        self.assertTrue(attach_identity["attachment_only"])
        self.assertTrue(attach_identity["external_dataset"])
        self.assertEqual(attach_identity["mount_candidates"], list(pkgbh.SUPPORT_MOUNT_CANDIDATES))
        self.assertFalse(attach_identity["wheelhouse"]["included_in_kernel_source"])
        self.assertEqual(attach_identity["checkpoint"]["sha256"], self.expected["weight_sha256"])
        self.assertEqual(report["attachment_manifest_sha256"], pkgbh.sha256_file(output / pkgbh.ATTACHMENT_NAME))

    def test_archive_is_compact_safe_and_contains_no_wheels(self) -> None:
        output, report = self.build()
        with zipfile.ZipFile(output / pkgbh.ARCHIVE_NAME) as archive:
            names = sorted(item.filename for item in archive.infolist() if not item.is_dir())
            self.assertEqual(names, sorted(item["path"] for item in report["archive"]["members"]))
            self.assertIn("scripts/run_bh.py", names)
            self.assertIn("src/celltrack_alpha/bh_model.py", names)
            self.assertIn("src/celltrack_alpha/bh_edge.py", names)
            self.assertIn("bhreg.json", names)
            self.assertFalse(any(name.endswith(".whl") for name in names))
            for item in report["archive"]["members"]:
                value = archive.read(item["path"])
                self.assertEqual(len(value), item["bytes"])
                self.assertEqual(_sha(value), item["sha256"])
        compile((output / pkgbh.ENTRYPOINT_NAME).read_text(encoding="ascii"), pkgbh.ENTRYPOINT_NAME, "exec")

    def test_dependency_plan_installs_only_missing_or_incompatible(self) -> None:
        statuses = {distribution: {"compatible": True} for distribution, *_rest in pkgbh.DEPENDENCIES}
        statuses["zarr"] = {"compatible": False}
        statuses["donfig"] = {"compatible": False}
        specs = pkgbh.dependency_install_specs(statuses)
        self.assertEqual(specs, ["zarr>=3.0.10,<4", "donfig>=0.8,<1"])
        self.assertFalse(any(spec.startswith(("torch", "numpy", "scipy")) for spec in specs))
        command = pkgbh.pip_command("python", "/private/wheels", specs)
        self.assertEqual(
            command[:9],
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--find-links",
                "/private/wheels",
                "zarr>=3.0.10,<4",
            ],
        )
        self.assertNotIn("--index-url", command)

    def test_runner_command_is_exact_dual_gpu_and_gates_acceleration(self) -> None:
        command = pkgbh.runner_command("python", "/src", "/competition", "/support", "/work")
        self.assertEqual(command[command.index("--devices") + 1], "cuda:0,cuda:1")
        self.assertEqual(command[command.index("--pair-batch-size") + 1], "1")
        self.assertEqual(command[command.index("--view-batch-size") + 1], "1")
        self.assertEqual(command[command.index("--deadline-min") + 1], "120")
        self.assertNotIn("--allow-cpu", command)
        self.assertNotIn("--accelerated-score-gate", command)

        accelerated = pkgbh.runner_command(
            "python", "/src", "/competition", "/support", "/work",
            view_batch_size=8,
            score_gate="/private/gate.json",
        )
        self.assertEqual(accelerated[accelerated.index("--view-batch-size") + 1], "8")
        self.assertEqual(accelerated[-2:], ["--accelerated-score-gate", "/private/gate.json"])
        for bad in (0, 9):
            with self.subTest(bad=bad), self.assertRaises(pkgbh.PackageError):
                pkgbh.runner_command("python", "/src", "/competition", "/support", "/work", view_batch_size=bad)
        with self.assertRaises(pkgbh.PackageError):
            pkgbh.runner_command("python", "/src", "/competition", "/support", "/work", view_batch_size=2)
        with self.assertRaises(pkgbh.PackageError):
            pkgbh.runner_command(
                "python", "/src", "/competition", "/support", "/work",
                view_batch_size=1,
                score_gate="/private/gate.json",
            )

    def test_checkpoint_hash_mismatch_fails_before_writing_package(self) -> None:
        expected = dict(self.expected)
        expected["weight_sha256"] = "f" * 64
        output = self.root / "badpkg"
        with self.assertRaisesRegex(pkgbh.PackageError, "checkpoint SHA-256"):
            pkgbh.build_package(
                self.repo,
                output,
                registry=self.registry,
                support_root=self.support,
                expected_artifact_hashes=expected,
            )
        self.assertFalse(output.exists())

    def test_public_support_mount_candidates_fail_closed(self) -> None:
        for mounts in (
            (),
            ("relative/support",),
            ("/kaggle/input/wrong-slug",),
            (pkgbh.SUPPORT_MOUNT_CANDIDATES[0], pkgbh.SUPPORT_MOUNT_CANDIDATES[0]),
        ):
            with self.subTest(mounts=mounts), self.assertRaisesRegex(
                pkgbh.PackageError, "support mount candidates"
            ):
                pkgbh.build_package(
                    self.repo,
                    self.root / "badmount",
                    registry=self.registry,
                    support_root=self.support,
                    expected_artifact_hashes=self.expected,
                    support_mounts=mounts,
                )

    def test_embedded_launcher_discovers_second_exact_public_mount(self) -> None:
        mounted = self.root / "mounted" / pkgbh.SUPPORT_DATASET_SLUG
        shutil.copytree(self.support, mounted)
        missing = self.root / "missing" / pkgbh.SUPPORT_DATASET_SLUG
        output = self.root / "launcher"
        pkgbh.build_package(
            self.repo,
            output,
            registry=self.registry,
            support_root=self.support,
            expected_artifact_hashes=self.expected,
            support_mounts=(str(missing), str(mounted)),
        )
        module = types.ModuleType("bh45_launcher_fixture")
        module.__file__ = str(output / pkgbh.ENTRYPOINT_NAME)
        exec(pkgbh.ENTRYPOINT_SOURCE, module.__dict__)
        diagnostics = {}
        selected, wheels = module._validate_attachment(diagnostics)
        self.assertEqual(selected, mounted)
        self.assertEqual(wheels, mounted / "wheels")
        self.assertEqual(diagnostics["support_selected"], str(mounted))
        self.assertEqual([row["ok"] for row in diagnostics["support_candidates"]], [False, True])


if __name__ == "__main__":
    unittest.main()
