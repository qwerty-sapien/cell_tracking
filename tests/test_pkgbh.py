from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from scripts import pkgbh


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class KagglePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.support = self.repo / "external" / "Tracking_Support"
        for relative in (*pkgbh.SOURCE_PATHS, *pkgbh.EDGE_SOURCE_PATHS):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {relative}\n", encoding="ascii")
        for relative in (pkgbh.CV57_EDGE_ARTIFACT, pkgbh.CV57_AUXILIARY_CHECKPOINT):
            source = ROOT / relative
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

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

    def test_cv57_selection_is_the_frozen_fold0_minus_terminal_exclusion(self) -> None:
        selected = pkgbh.cv57_video_ids()
        self.assertEqual(len(selected), 57)
        self.assertNotIn(pkgbh.CV57_EXCLUDED_IDENTITY, selected)
        self.assertEqual(
            _sha(pkgbh.canonical_json(list(selected))),
            pkgbh.CV57_SELECTION_SHA256,
        )

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
        self.assertLess(first_report["submitted_code_bytes"], 1_000_000)
        self.assertTrue(first_report["strictly_under_budget"])
        self.assertEqual(first_report["file_count"], 5)

        metadata = json.loads(first_files[pkgbh.METADATA_NAME])
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["machine_shape"], pkgbh.KAGGLE_ACCELERATOR_ID)
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
        logical_files = {
            path.name: path.read_bytes()
            for path in output.iterdir()
            if path.is_file()
        }
        logical_files[pkgbh.ENTRYPOINT_NAME] = pkgbh.ENTRYPOINT_SOURCE.encode("ascii")
        self.assertEqual(identity["package_bytes"], sum(map(len, logical_files.values())))
        self.assertEqual(
            (output / pkgbh.ENTRYPOINT_NAME).read_bytes(),
            pkgbh.monolithic_entrypoint(logical_files),
        )
        wrapper_tree = ast.parse((output / pkgbh.ENTRYPOINT_NAME).read_text(encoding="ascii"))
        payload_node = next(
            node.value
            for node in wrapper_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_PAYLOADS" for target in node.targets)
        )
        embedded = {
            name: base64.b64decode(value, validate=True)
            for name, value in ast.literal_eval(payload_node).items()
        }
        self.assertEqual(embedded, logical_files)
        self.assertEqual(identity["runtime"]["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(
            identity["runtime"]["deadline_minutes"],
            pkgbh.SUBMISSION_DEADLINE_MINUTES,
        )
        self.assertEqual(identity["runtime"]["default_view_batch_size"], 1)
        self.assertTrue(identity["runtime"]["accelerated_score_gate_required"])
        self.assertEqual(identity["runtime"]["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            identity["runtime"]["gpu_attestation_classifications"],
            [
                "no_cuda_devices",
                "unexpected_accelerator_family",
                "honored_t4_wrong_device_count",
                "exact_t4x2",
            ],
        )
        self.assertEqual(identity["runtime"]["run"], pkgbh.submit_run_spec())
        for item in identity["files"]:
            value = logical_files[item["path"]]
            self.assertEqual(item["bytes"], len(value))
            self.assertEqual(item["sha256"], _sha(value))

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
            self.assertIn(pkgbh.CV57_EDGE_ARTIFACT, names)
            self.assertIn(pkgbh.CV57_AUXILIARY_CHECKPOINT, names)
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
        self.assertEqual(
            command[command.index("--deadline-min") + 1],
            str(pkgbh.SUBMISSION_DEADLINE_MINUTES),
        )
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

    def test_cv57_runner_command_is_exact_released_detector_configuration(self) -> None:
        run = pkgbh.cv57_run_spec()
        command = pkgbh.runner_command(
            "python", "/src", "/competition", "/support", "/work", run_spec=run
        )
        self.assertEqual(command[command.index("--mode") + 1], "cv")
        self.assertEqual(command[command.index("--split") + 1], "train")
        self.assertEqual(
            command[command.index("--graph-variant") + 1],
            pkgbh.RELEASED_GRAPH_VARIANT,
        )
        self.assertEqual(run["topology_mode"], "legacy")
        self.assertEqual(command[command.index("--edge-artifact") + 1], f"/src/{pkgbh.CV57_EDGE_ARTIFACT}")
        self.assertEqual(
            command[command.index("--edge-artifact-sha256") + 1],
            pkgbh.CV57_EDGE_ARTIFACT_SHA256,
        )
        self.assertEqual(command[command.index("--datasets") + 1].split(","), list(pkgbh.cv57_video_ids()))
        self.assertNotIn("--test-only-one-view", command)
        self.assertNotIn("--allow-cpu", command)

    def test_submit_runner_command_matches_scored_configuration(self) -> None:
        run = pkgbh.submit_run_spec()
        command = pkgbh.runner_command(
            "python", "/src", "/competition", "/support", "/work", run_spec=run
        )
        self.assertEqual(command[command.index("--mode") + 1], "submit")
        self.assertEqual(command[command.index("--split") + 1], "test")
        self.assertEqual(
            command[command.index("--graph-variant") + 1],
            pkgbh.RELEASED_GRAPH_VARIANT,
        )
        self.assertEqual(command[command.index("--edge-artifact") + 1], f"/src/{pkgbh.CV57_EDGE_ARTIFACT}")
        self.assertEqual(
            command[command.index("--edge-artifact-sha256") + 1],
            pkgbh.CV57_EDGE_ARTIFACT_SHA256,
        )
        self.assertNotIn("--datasets", command)
        self.assertNotIn("--test-only-one-view", command)
        self.assertFalse(run["division_enabled"])
        self.assertEqual(run["topology_mode"], "legacy")

    def test_appearance_probe_package_is_hash_pinned_and_fail_closed(self) -> None:
        for relative in (
            "src/celltrack_alpha/relative_geometry_gate.py",
            "src/celltrack_alpha/appearance_swap.py",
            pkgbh.APPEARANCE_CONTRACT,
            pkgbh.APPEARANCE_GATE,
        ):
            source = ROOT / relative
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        output = self.root / "bhap-submit"
        report = pkgbh.build_package(
            self.repo,
            output,
            registry=self.registry,
            support_root=self.support,
            expected_artifact_hashes=self.expected,
            profile="appearance-submit",
        )
        self.assertEqual(report["profile"], "appearance-submit")
        self.assertEqual(
            report["run"], pkgbh.appearance_run_spec("submit")
        )
        self.assertLess(report["aggregate_bytes"], pkgbh.SOURCE_LIMIT)
        with zipfile.ZipFile(output / pkgbh.ARCHIVE_NAME) as archive:
            names = set(archive.namelist())
            self.assertIn(pkgbh.APPEARANCE_CONTRACT, names)
            self.assertIn(pkgbh.APPEARANCE_GATE, names)
            self.assertIn("src/celltrack_alpha/appearance_swap.py", names)
            self.assertEqual(
                _sha(archive.read(pkgbh.APPEARANCE_CONTRACT)),
                pkgbh.APPEARANCE_CONTRACT_SHA256,
            )
            self.assertEqual(
                _sha(archive.read(pkgbh.APPEARANCE_GATE)),
                pkgbh.APPEARANCE_GATE_SHA256,
            )
        command = pkgbh.runner_command(
            "python",
            "/src",
            "/competition",
            "/support",
            "/work",
            run_spec=report["run"],
        )
        self.assertEqual(
            command[command.index("--appearance-swap-contract") + 1],
            f"/src/{pkgbh.APPEARANCE_CONTRACT}",
        )
        self.assertEqual(
            command[command.index("--appearance-swap-contract-sha256") + 1],
            pkgbh.APPEARANCE_CONTRACT_SHA256,
        )

        (self.repo / pkgbh.APPEARANCE_GATE).write_bytes(b"corrupt")
        with self.assertRaisesRegex(
            pkgbh.PackageError, "appearance input identity mismatch"
        ):
            pkgbh.build_package(
                self.repo,
                self.root / "bad-appearance",
                registry=self.registry,
                support_root=self.support,
                expected_artifact_hashes=self.expected,
                profile="appearance-submit",
            )

    def test_embedded_gpu_diagnostics_classify_wrong_family_and_t4_count(self) -> None:
        output, _report = self.build("gpu")
        module = types.ModuleType("bh45_gpu_fixture")
        module.__file__ = str(output / pkgbh.ENTRYPOINT_NAME)
        exec(pkgbh.ENTRYPOINT_SOURCE, module.__dict__)

        class FakeCuda:
            def __init__(self, count: int, name: str) -> None:
                self.count = count
                self.name = name

            def device_count(self) -> int:
                return self.count

            def get_device_name(self, _index: int) -> str:
                return self.name

            @staticmethod
            def get_device_capability(_index: int) -> tuple[int, int]:
                return (7, 5)

            @staticmethod
            def get_device_properties(_index: int) -> SimpleNamespace:
                return SimpleNamespace(total_memory=16_000_000_000, multi_processor_count=40)

            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def get_arch_list() -> list[str]:
                return ["sm_75"]

        result = SimpleNamespace(returncode=0, stdout="diagnostic\n", stderr="")
        for count, name, classification, mismatches in (
            (0, "Tesla T4", "no_cuda_devices", ["device_count"]),
            (1, "Tesla T4", "honored_t4_wrong_device_count", ["device_count"]),
            (2, "Tesla P100", "unexpected_accelerator_family", ["device_name"]),
        ):
            with self.subTest(classification=classification):
                fake_torch = SimpleNamespace(
                    __version__="2.9.0+cu126",
                    version=SimpleNamespace(cuda="12.6"),
                    cuda=FakeCuda(count, name),
                    backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
                )
                module.GPU_DIAG = None
                with (
                    mock.patch.object(module.importlib, "import_module", return_value=fake_torch),
                    mock.patch.object(module.subprocess, "run", return_value=result),
                    self.assertRaises(module.LaunchError),
                ):
                    module.gpu()
                self.assertEqual(module.GPU_DIAG["observed"]["classification"], classification)
                self.assertEqual(module.GPU_DIAG["observed"]["mismatches"], mismatches)
                self.assertEqual(
                    module.GPU_DIAG["requested"]["kaggle_machine_shape"],
                    pkgbh.KAGGLE_ACCELERATOR_ID,
                )

        exact_torch = SimpleNamespace(
            __version__="2.9.0+cu126",
            version=SimpleNamespace(cuda="12.6"),
            cuda=FakeCuda(2, "Tesla T4"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
        )
        module.GPU_DIAG = None
        with (
            mock.patch.object(module.importlib, "import_module", return_value=exact_torch),
            mock.patch.object(module.subprocess, "run", return_value=result),
        ):
            self.assertEqual(module.gpu(), ["Tesla T4", "Tesla T4"])
        self.assertEqual(module.GPU_DIAG["observed"]["classification"], "exact_t4x2")
        self.assertEqual(module.GPU_DIAG["observed"]["mismatches"], [])

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
