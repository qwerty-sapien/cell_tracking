from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from scripts import pkgbh9


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_prerelease_wheel(path: Path) -> None:
    version = "0.1.0rc6.dev3+g980c2d30a"
    dist_info = f"tracksdata-{version}.dist-info"
    members = {
        "tracksdata/__init__.py": f'__version__ = "{version}"\n'.encode("ascii"),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: tracksdata\n"
            f"Version: {version}\n\n"
        ).encode("ascii"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: bh9-test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("ascii"),
        f"{dist_info}/RECORD": b"",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)


class Bh9PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        local_import_stubs = {
            "scripts/bh9gb.py": "import run_bh\n",
            "scripts/run_bh.py": (
                "import celltrack_alpha.bh_model\n"
                "import celltrack_alpha.bh_graph\n"
                "import celltrack_alpha.bh_edge\n"
                "import celltrack_alpha.bh_route\n"
            ),
            "scripts/scoreall.py": (
                "import celltrack_alpha.kaggle_calibration\n"
                "import celltrack_alpha.official_eval\n"
                "import celltrack_baseline.rule_based\n"
            ),
            "src/celltrack_alpha/bh_edge.py": (
                "import celltrack_alpha.bh_graph\n"
                "import celltrack_alpha.bh_model\n"
                "import celltrack_alpha.v45_head\n"
            ),
            "src/celltrack_alpha/v45_head.py": "import celltrack_alpha.bh_model\n",
            "src/celltrack_alpha/official_eval.py": "import tracking_cellmot.metrics\n",
            "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py": (
                "import tracking_cellmot.division_metrics\n"
            ),
        }
        for relative in pkgbh9.SOURCE_PATHS:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                local_import_stubs.get(relative, f"# fixture {relative}\n"),
                encoding="ascii",
            )

        records = []
        for dataset in ("a", "b"):
            records.append({
                "split": "train", "dataset": dataset, "video_id": f"train:{dataset}",
                "exact_identity_sha256": hashlib.sha256(dataset.encode("ascii")).hexdigest(),
                "compact_metadata": {"image_shape": [100, 2, 3, 4], "ignored": "bulk"},
                "member_vector": [{"ignored": True}],
            })
        self.registry = self.repo / "reports/g02/dataset_registry_v1.json"
        self.registry.parent.mkdir(parents=True)
        self.registry.write_text(json.dumps({
            "artifact": "dataset_registry_v1",
            "counts": {"train_videos": 2, "videos": 2},
            "source": {"zip_central_directory_sha256": "c" * 64, "zip_entry_count": 9},
            "records": records,
        }), encoding="utf-8")
        source_hashes = {
            relative: pkgbh9.sha256_file(self.repo / relative)
            for relative in pkgbh9.bh9gb.SCORER_FILES
        }
        self.split = self.repo / "reports/g02/grouped_3fold_audit20_v1.json"
        self.split.write_text(json.dumps({
            "dataset_registry_sha256": pkgbh9.sha256_file(self.registry),
            "source_hashes": source_hashes,
        }), encoding="utf-8")
        self.archive = self.root / "data.zip"
        self.archive.write_bytes(b"fixture; inventory is mocked")
        self.inventory = {
            "compressed_bytes": {"train:a": 5_000_000_000, "train:b": 5_000_000_000},
            "member_counts": {"train:a": 2, "train:b": 2},
            "train_image_bytes": 10_000_000_000,
            "central_sha256": "c" * 64,
            "entry_count": 9,
        }

        self.support = self.repo / "external/Tracking_Support"
        fixture = {
            "ARTIFACT_MANIFEST.json": b"manifest\n",
            "weights/unet_transformer/split_0/config.json": b"{}\n",
            "weights/unet_transformer/split_0/edge_predictor_best.pth": b"checkpoint\x00",
            "repo/src/biohub_tracking/__init__.py": b'"""fixture package."""\n',
            "repo/src/biohub_tracking/models/__init__.py": (
                b"from .temporal_unet import TemporalUNet3D\n"
                b"from .simple_node_transformer import SimpleNodeTransformer\n"
            ),
            "repo/src/biohub_tracking/models/temporal_unet.py": b"class TemporalUNet3D: pass\n",
            "repo/src/biohub_tracking/models/simple_node_transformer.py": b"class SimpleNodeTransformer: pass\n",
            "repo/scripts/train_unet_transformer.py": b"# provenance\n",
        }
        for relative, value in fixture.items():
            path = self.support / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        wheels = self.support / "wheels"
        wheels.mkdir(parents=True)
        base = {
            pkgbh9.normalized_distribution_name(item)
            for item in pkgbh9.BASE_ENVIRONMENT_DISTRIBUTIONS
        }
        for distribution, _module, minimum, _maximum, _policy in pkgbh9.DEPENDENCIES:
            normalized = pkgbh9.normalized_distribution_name(distribution)
            if normalized in base:
                continue
            version = (
                "0.1.0rc6.dev3+g980c2d30a"
                if normalized == "tracksdata"
                else minimum
            )
            filename_distribution = normalized.replace("-", "_")
            name = f"{filename_distribution}-{version}-py3-none-any.whl"
            (wheels / name).write_bytes(f"wheel:{distribution}:{version}".encode("ascii"))
        self.expected = {
            "manifest_sha256": _sha(fixture["ARTIFACT_MANIFEST.json"]),
            "config_sha256": _sha(fixture["weights/unet_transformer/split_0/config.json"]),
            "source_sha256": pkgbh9.support_source_tree_sha256(self.support),
            "weight_sha256": _sha(fixture["weights/unet_transformer/split_0/edge_predictor_best.pth"]),
        }

    def build(self, name: str = "bh9pkg") -> tuple[Path, dict]:
        output = self.root / name
        with mock.patch.object(pkgbh9, "_origin_inventory", return_value=self.inventory):
            report = pkgbh9.build_package(
                self.repo, output, archive=self.archive, registry=self.registry,
                split=self.split, support_root=self.support,
                expected_artifact_hashes=self.expected, expected_train_count=2,
            )
        return output, report

    def extracted_runtime(self, output: Path, name: str) -> tuple[dict, Path]:
        source_path = output / pkgbh9.ENTRYPOINT_NAME
        namespace = {"__file__": str(source_path), "__name__": f"bh9_{name}_test"}
        exec(compile(source_path.read_bytes(), str(source_path), "exec"), namespace)
        work = self.root / name
        work.mkdir()
        package = work / "bh9pkg"
        namespace.update({
            "WORK": work,
            "PKG": package,
            "MAN": package / pkgbh9.MANIFEST_NAME,
            "ATT": package / pkgbh9.ATTACHMENT_NAME,
            "SRC": package / pkgbh9.ARCHIVE_NAME,
            "ROOT": work / "bh9src",
            "OK": work / "bh9.json",
            "FAIL": work / "bh9fail.json",
            "OUT": work / "bh9out",
        })
        namespace["materialize"]()
        namespace["verify_pkg"]()
        namespace["extract"]()
        return namespace, work

    def test_build_is_deterministic_private_offline_and_under_budget(self) -> None:
        first, first_report = self.build("pkg1")
        second, second_report = self.build("pkg2")
        first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
        self.assertEqual(first_files, second_files)
        self.assertEqual(set(first_files), {
            pkgbh9.ENTRYPOINT_NAME, pkgbh9.ARCHIVE_NAME, pkgbh9.ATTACHMENT_NAME,
            pkgbh9.MANIFEST_NAME, pkgbh9.METADATA_NAME,
        })
        self.assertEqual(first_report["package_sha256"], second_report["package_sha256"])
        self.assertLess(first_report["aggregate_bytes"], 1_000_000)
        self.assertLess(len(first_files[pkgbh9.ENTRYPOINT_NAME]), 1_000_000)
        metadata = json.loads(first_files[pkgbh9.METADATA_NAME])
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], pkgbh9.KAGGLE_ACCELERATOR_ID)
        self.assertEqual(metadata["competition_sources"], [pkgbh9.COMPETITION_SLUG])
        self.assertEqual(metadata["dataset_sources"], [pkgbh9.SUPPORT_REF])
        for name in first_files:
            if name != pkgbh9.METADATA_NAME:
                self.assertLess(len(name), 15)

    def test_archive_freezes_byte_only_selection_and_exact_runtime_sources(self) -> None:
        output, report = self.build()
        with zipfile.ZipFile(output / pkgbh9.ARCHIVE_NAME) as archive:
            names = {item.filename for item in archive.infolist() if not item.is_dir()}
            self.assertTrue(set(pkgbh9.SOURCE_PATHS).issubset(names))
            self.assertIn("src/celltrack_alpha/bh_edge.py", names)
            self.assertIn("src/celltrack_alpha/v45_head.py", names)
            self.assertTrue({pkgbh9.FROZEN_NAME, pkgbh9.REGISTRY_NAME, pkgbh9.SPLIT_NAME}.issubset(names))
            registry = json.loads(archive.read(pkgbh9.REGISTRY_NAME))
            frozen = json.loads(archive.read(pkgbh9.FROZEN_NAME))
        self.assertEqual(registry["counts"]["train_videos"], 2)
        self.assertNotIn("member_vector", json.dumps(registry))
        identity = frozen["identity"]
        self.assertEqual(frozen["identity_sha256"], _sha(pkgbh9.canonical_json(identity)))
        self.assertEqual(identity["seed"], 20_260_718)
        self.assertEqual(identity["target_compressed_bytes"], 9_000_000_000)
        self.assertEqual(identity["selection"], report["selection"])
        self.assertEqual(identity["selection"]["selected_compressed_bytes"], 10_000_000_000)

    def test_source_paths_follow_recursive_local_imports(self) -> None:
        dependency = self.repo / "src/celltrack_alpha/future_aux.py"
        dependency.write_text("class FutureAux: pass\n", encoding="ascii")
        edge = self.repo / "src/celltrack_alpha/bh_edge.py"
        edge.write_text(
            edge.read_text(encoding="utf-8")
            + "\nfrom .future_aux import FutureAux\n",
            encoding="utf-8",
        )
        derived = pkgbh9.derive_source_paths(self.repo)
        self.assertIn("src/celltrack_alpha/future_aux.py", derived)
        output, _report = self.build("importwalk")
        with zipfile.ZipFile(output / pkgbh9.ARCHIVE_NAME) as archive:
            self.assertIn("src/celltrack_alpha/future_aux.py", archive.namelist())

    def test_source_paths_reject_missing_local_imports(self) -> None:
        edge = self.repo / "src/celltrack_alpha/bh_edge.py"
        edge.write_text(
            edge.read_text(encoding="utf-8")
            + "\nfrom .missing_aux import MissingAux\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(pkgbh9.PackageError, "missing local module"):
            pkgbh9.derive_source_paths(self.repo)

    def test_manifests_attest_package_checkpoint_sources_wheels_and_policy(self) -> None:
        output, _report = self.build()
        package = json.loads((output / pkgbh9.MANIFEST_NAME).read_text(encoding="ascii"))
        identity = package["identity"]
        self.assertEqual(package["identity_sha256"], _sha(pkgbh9.canonical_json(identity)))
        self.assertEqual(identity["deadline_seconds"], 3000)
        self.assertEqual(identity["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(identity["accelerator"], "T4x2")
        self.assertEqual(identity["trials"], 1)
        self.assertEqual(identity["minimum_score"], 0.90)
        self.assertEqual(identity["graph_variant"], pkgbh9.bh9gb.run_bh.A0_GRAPH_VARIANT)
        self.assertEqual(identity["selected_count"], 22)
        self.assertEqual(identity["adjacent_pair_count"], 2_178)
        self.assertEqual(identity["runtime_attestation_level"], 6)
        self.assertEqual(identity["hermetic_child_import_layout"], list(pkgbh9.CHILD_IMPORT_LAYOUT))
        self.assertEqual(
            identity["preserved_environment_keys"],
            list(pkgbh9.PRESERVED_ENVIRONMENT_KEYS),
        )
        self.assertEqual(identity["child_cwd"], "materialized_source_root")
        self.assertTrue(identity["python_safe_path"])
        self.assertFalse(identity["inherits_pythonpath"])
        self.assertTrue(identity["self_contained_code_file"])
        self.assertEqual(
            identity["embedded_files"],
            sorted((pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME)),
        )
        self.assertTrue(identity["sparse_graph"])
        self.assertEqual(
            identity["package_bytes"],
            sum(
                (output / name).stat().st_size
                for name in (pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME)
            ),
        )
        self.assertEqual(
            {row["path"] for row in identity["files"]},
            {pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME},
        )
        attachment = json.loads((output / pkgbh9.ATTACHMENT_NAME).read_text(encoding="ascii"))
        attached = attachment["identity"]
        self.assertEqual(attachment["identity_sha256"], _sha(pkgbh9.canonical_json(attached)))
        self.assertEqual(attached["checkpoint"]["sha256"], self.expected["weight_sha256"])
        self.assertEqual(attached["source_tree"]["sha256"], self.expected["source_sha256"])
        self.assertEqual(attached["source_tree"]["paths"], list(pkgbh9.SUPPORT_SOURCE_PATHS))
        self.assertIn("repo/src/biohub_tracking/__init__.py", attached["source_tree"]["paths"])
        self.assertIn(
            "repo/src/biohub_tracking/models/__init__.py", attached["source_tree"]["paths"]
        )
        self.assertEqual(attached["dependencies"], [list(row) for row in pkgbh9.DEPENDENCIES])
        self.assertEqual(
            attached["base_environment_dependencies"],
            sorted(pkgbh9.BASE_ENVIRONMENT_DISTRIBUTIONS),
        )
        wheels = attached["wheelhouse"]["files"]
        self.assertEqual(attached["wheelhouse"]["manifest_sha256"], _sha(pkgbh9.canonical_json(wheels)))
        plan = attached["dependency_wheels"]
        self.assertEqual(plan, pkgbh9.dependency_wheel_plan(wheels))
        self.assertEqual(attached["dependency_wheels_sha256"], _sha(pkgbh9.canonical_json(plan)))
        self.assertEqual(len(plan), len(pkgbh9.DEPENDENCIES) - 1)
        tracksdata = next(row for row in plan if row["distribution"] == "tracksdata")
        self.assertEqual(tracksdata["version"], "0.1.0rc6.dev3+g980c2d30a")
        self.assertTrue(
            pkgbh9.version_compatible(
                tracksdata["version"], tracksdata["minimum"], tracksdata["maximum"]
            )
        )

    def test_launcher_is_exact_t4x2_offline_one_trial_and_writes_evidence(self) -> None:
        output, _report = self.build()
        source = (output / pkgbh9.ENTRYPOINT_NAME).read_text(encoding="ascii")
        compile(source, pkgbh9.ENTRYPOINT_NAME, "exec")
        for token in (
            "count!=2", '"T4" not in name.upper()', '"CUDA_VISIBLE_DEVICES":"0,1"',
            'ROOT/"scripts/bh9gb.py"', 'ROOT/"bh9m.json"', 'WORK/"bh9.json"',
            'WORK/"bh9fail.json"', 'OUT/"run.json"', 'OUT/"fragments"',
            '"--no-index"', '"--no-deps"', '"--force-reinstall"', "one-trial contract",
            "embedded runtime hash mismatch", "materialized runtime hash mismatch",
            "runtime_attestation_level", "level-6 runtime attestation",
            'PKG=WORK/"bh9pkg"', "score_strictly_above_0_90",
            'i.get("graph_variant")!="a0"', 'audit.get("pair_count")!=2178',
            "exactly_two_a0_scorer_instances", "checkpoint_weight_identity",
            "selected exact wheel failed immediate basename/size/SHA verification",
            "installed versions differ from attested exact wheels",
            "required source import failed before data reads", "child_import_layout",
            '"PYTHONPATH":os.pathsep.join', 'cmd=[str(python_exe()),"-P",str(script),',
            'cwd=str(ROOT)', "PYTHONSAFEPATH", '"biohub_tracking"',
            '"tracking_cellmot.metrics"', '"celltrack_alpha"',
            'KSHAPE="NvidiaTeslaT4"', '"gpu_attestation"',
            'classification="honored_t4_wrong_device_count"',
            'classification="unexpected_accelerator_family"',
            '"nvidia_smi"', '"traceback":traceback.format_exc()',
        ):
            self.assertIn(token, source)
        self.assertNotIn("--find-links", source)
        self.assertNotIn("HERE=", source)
        self.assertNotIn('/kaggle/src/bh9man.json', source)
        self.assertNotIn("dict(os.environ", source)
        for mount in (*pkgbh9.COMPETITION_MOUNTS, *pkgbh9.SUPPORT_MOUNTS):
            self.assertIn(mount, source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)
        self.assertIn('("celltrack_alpha.bh_edge","embedded_src")', source)
        self.assertIn('("celltrack_alpha.v45_head","embedded_src")', source)
        namespace = {"__file__": str(output / pkgbh9.ENTRYPOINT_NAME), "__name__": "imports"}
        exec(compile(source, pkgbh9.ENTRYPOINT_NAME, "exec"), namespace)
        self.assertEqual(
            {name for name, _root in namespace["REQ"]},
            {
                "celltrack_alpha",
                "celltrack_alpha.bh_model",
                "celltrack_alpha.bh_graph",
                "celltrack_alpha.bh_edge",
                "celltrack_alpha.v45_head",
                "celltrack_alpha.bh_route",
                "celltrack_alpha.kaggle_calibration",
                "celltrack_alpha.official_eval",
                "celltrack_baseline.rule_based",
                "tracking_cellmot",
                "tracking_cellmot.metrics",
                "tracking_cellmot.division_metrics",
                "biohub_tracking",
                "biohub_tracking.models.temporal_unet",
                "biohub_tracking.models.simple_node_transformer",
                "run_bh",
                "scoreall",
                "bh9gb",
            },
        )

    def test_gpu_diagnostics_distinguish_t4_count_from_wrong_shape(self) -> None:
        output, _report = self.build()
        source = (output / pkgbh9.ENTRYPOINT_NAME).read_text(encoding="ascii")
        namespace = {"__file__": str(output / pkgbh9.ENTRYPOINT_NAME), "__name__": "gpu_diag"}
        exec(compile(source, pkgbh9.ENTRYPOINT_NAME, "exec"), namespace)

        class FakeCuda:
            def __init__(self, name: str) -> None:
                self.name = name

            @staticmethod
            def device_count() -> int:
                return 1

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
                return ["sm_70", "sm_75", "sm_80"]

        command_result = SimpleNamespace(
            returncode=0, stdout="0, Tesla T4, 7.5, 16000, 570.00\n", stderr=""
        )
        for name, classification, mismatches in (
            ("Tesla T4", "honored_t4_wrong_device_count", ["device_count"]),
            (
                "Tesla P100-PCIE-16GB",
                "unexpected_accelerator_family",
                ["device_count", "device_name"],
            ),
        ):
            with self.subTest(name=name):
                fake_torch = SimpleNamespace(
                    __version__="2.9.0+cu126",
                    version=SimpleNamespace(cuda="12.6"),
                    cuda=FakeCuda(name),
                    backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
                )
                namespace["GPU_DIAG"] = None
                with (
                    mock.patch.object(namespace["importlib"], "import_module", return_value=fake_torch),
                    mock.patch.object(namespace["subprocess"], "run", return_value=command_result),
                ):
                    with self.assertRaises(namespace["LaunchError"]) as raised:
                        namespace["gpu"]()
                diagnostics = namespace["GPU_DIAG"]
                self.assertEqual(diagnostics["requested"]["kaggle_machine_shape"], "NvidiaTeslaT4")
                self.assertEqual(diagnostics["observed"]["classification"], classification)
                self.assertEqual(diagnostics["observed"]["mismatches"], mismatches)
                self.assertEqual(diagnostics["observed"]["devices"][0]["name"], name)
                self.assertEqual(diagnostics["nvidia_smi"]["query"]["returncode"], 0)
                self.assertIn(classification, str(raised.exception))

        failure = self.root / "gpu-failure.json"
        namespace["FAIL"] = failure
        namespace["OK"] = self.root / "gpu-success.json"
        fake_torch = SimpleNamespace(
            __version__="2.9.0+cu126",
            version=SimpleNamespace(cuda="12.6"),
            cuda=FakeCuda("Tesla P100-PCIE-16GB"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
        )
        with (
            mock.patch.object(namespace["importlib"], "import_module", return_value=fake_torch),
            mock.patch.object(namespace["subprocess"], "run", return_value=command_result),
        ):
            try:
                namespace["gpu"]()
            except namespace["LaunchError"] as error:
                namespace["fail"](error, "gpu_attestation", namespace["time"].monotonic())
        payload = json.loads(failure.read_text(encoding="ascii"))
        self.assertEqual(payload["stage"], "gpu_attestation")
        self.assertEqual(
            payload["diagnostics"]["gpu_attestation"]["observed"]["classification"],
            "unexpected_accelerator_family",
        )
        self.assertIn("LaunchError", payload["traceback"])

    def test_archive_has_exact_curated_sources_and_no_secret_like_names(self) -> None:
        output, _report = self.build()
        with zipfile.ZipFile(output / pkgbh9.ARCHIVE_NAME) as archive:
            names = sorted(item.filename for item in archive.infolist() if not item.is_dir())
        expected = sorted(
            (*pkgbh9.SOURCE_PATHS, pkgbh9.FROZEN_NAME, pkgbh9.REGISTRY_NAME,
             pkgbh9.SPLIT_NAME, "src/celltrack_alpha/__init__.py",
             "src/celltrack_baseline/__init__.py",
             "external/kaggle-cell-tracking-competition/src/tracking_cellmot/__init__.py")
        )
        self.assertEqual(names, expected)
        for name in names:
            self.assertIsNone(pkgbh9._SECRET_NAME.search(name), name)
            self.assertNotIn(".env", name.lower())
            self.assertNotIn("credential", name.lower())

    def test_standalone_materializes_then_verifies_and_enforces_level_6_success(self) -> None:
        output, _report = self.build()
        source_path = output / pkgbh9.ENTRYPOINT_NAME
        namespace = {
            "__file__": str(source_path),
            "__name__": "bh9_standalone_test",
        }
        exec(compile(source_path.read_bytes(), str(source_path), "exec"), namespace)
        work = self.root / "work"
        work.mkdir()
        package = work / "bh9pkg"
        namespace.update({
            "WORK": work,
            "PKG": package,
            "MAN": package / pkgbh9.MANIFEST_NAME,
            "ATT": package / pkgbh9.ATTACHMENT_NAME,
            "SRC": package / pkgbh9.ARCHIVE_NAME,
            "OK": work / "bh9.json",
        })
        namespace["materialize"]()
        for name in (pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME):
            self.assertEqual((package / name).read_bytes(), (output / name).read_bytes())
            self.assertLess(len(name), 15)
        policy = namespace["verify_pkg"]()
        gates = {
            name: True
            for name in (
                "dual_gpu", "no_dense_nxn", "complete_score_coverage",
                "graph_validity", "deadline_met", "score_strictly_above_0_90",
                "graph_variant_a0", "exactly_two_a0_scorer_instances",
                "exact_22_video_a0_audit", "exact_2178_adjacent_pairs",
                "all_cache_contexts_covered",
                "finite_unique_original_coo_probabilities",
                "full_frame_dense_matrix_false",
                "balanced_attention_fallback_accounting",
                "checkpoint_weight_identity",
            )
        }
        weight = "c" * 64
        success = {
            "passed": True,
            "gates": gates,
            "runtime_attestation": {
                "level": 6, "passed": True, "maximum_seconds": 3000,
            },
            "official_score": {"summary": {"score": 0.900001}},
            "selection": {
                "selected_count": 22,
                "selected_compressed_bytes": 9_337_408_312,
            },
            "runner": {
                "graph_variant": "a0",
                "a0_validation": {
                    name: True
                    for name in (
                        "graph_variant_a0", "exactly_two_a0_scorer_instances",
                        "exact_22_video_a0_audit", "exact_2178_adjacent_pairs",
                        "all_cache_contexts_covered",
                        "finite_unique_original_coo_probabilities",
                        "full_frame_dense_matrix_false",
                        "balanced_attention_fallback_accounting",
                        "checkpoint_weight_identity",
                    )
                },
                "a0_audit": {
                    "scope": "run",
                    "video_count": 22,
                    "frame_count": 2200,
                    "pair_count": 2178,
                    "workers": {"worker_count": 2, "scorer_instances": 2},
                    "gate": {"full_frame_dense_matrix": False},
                    "probabilities": {
                        "candidate_count": 4,
                        "unique_pair_count": 4,
                        "finite_unit_interval": True,
                    },
                },
                "checkpoint_weight_identity": {
                    "preflight_weight_sha256": weight,
                    "frozen_cache_weight_sha256": weight,
                    "a0_audit_weight_sha256": weight,
                    "matched": True,
                },
            },
        }
        namespace["OK"].write_text(json.dumps(success), encoding="ascii")
        self.assertEqual(namespace["verify_ok"](policy), success)
        success["official_score"]["summary"]["score"] = 0.90
        namespace["OK"].write_text(json.dumps(success), encoding="ascii")
        with self.assertRaisesRegex(namespace["LaunchError"], "strictly above"):
            namespace["verify_ok"](policy)
        success["official_score"]["summary"]["score"] = 0.900001
        success["gates"]["exact_2178_adjacent_pairs"] = False
        namespace["OK"].write_text(json.dumps(success), encoding="ascii")
        with self.assertRaisesRegex(namespace["LaunchError"], "gates are incomplete"):
            namespace["verify_ok"](policy)

    def test_hermetic_import_probe_ignores_unrelated_cwd_and_hostile_pythonpath(self) -> None:
        output, _report = self.build()
        namespace, work = self.extracted_runtime(output, "runtime")
        hostile = self.root / "hostile"
        (hostile / "celltrack_alpha").mkdir(parents=True)
        (hostile / "celltrack_alpha/__init__.py").write_text(
            'raise RuntimeError("hostile celltrack_alpha imported")\n', encoding="ascii"
        )
        (hostile / "biohub_tracking").mkdir()
        (hostile / "biohub_tracking/__init__.py").write_text(
            'raise RuntimeError("hostile biohub_tracking imported")\n', encoding="ascii"
        )
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        prior_cwd = Path.cwd()
        try:
            os.chdir(unrelated)
            without_pythonpath = dict(os.environ)
            without_pythonpath.pop("PYTHONPATH", None)
            with mock.patch.dict(os.environ, without_pythonpath, clear=True):
                paths = namespace["child_paths"](self.support)
                environment = namespace["child_env"](paths)
                absent_observed = namespace["probe_required"](
                    paths, environment, namespace["time"].monotonic() + 60.0
                )
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(hostile)}, clear=False):
                paths = namespace["child_paths"](self.support)
                environment = namespace["child_env"](paths)
                hostile_observed = namespace["probe_required"](
                    paths, environment, namespace["time"].monotonic() + 60.0
                )
        finally:
            os.chdir(prior_cwd)
        expected_pythonpath = os.pathsep.join(
            str(paths[key])
            for key in ("embedded_src", "official_src", "support_src", "scripts")
        )
        self.assertEqual(environment["PYTHONPATH"], expected_pythonpath)
        self.assertNotIn(str(hostile), environment["PYTHONPATH"])
        self.assertEqual(environment["HOME"], str(work))
        self.assertEqual(environment["TMPDIR"], str(work))
        self.assertEqual(environment["PYTHONSAFEPATH"], "1")
        self.assertEqual(absent_observed, hostile_observed)
        self.assertIn("celltrack_alpha", hostile_observed)
        self.assertIn("biohub_tracking", hostile_observed)
        self.assertTrue(
            all(
                str(paths["embedded_src"]) in path
                for path in hostile_observed["celltrack_alpha"]
            )
        )
        self.assertTrue(
            all(
                str(paths["support_src"]) in path
                for path in hostile_observed["biohub_tracking"]
            )
        )

    def test_missing_or_symlink_escaped_import_path_fails_before_spawn(self) -> None:
        output, _report = self.build()
        namespace, _work = self.extracted_runtime(output, "tamper")
        official = namespace["ROOT"] / "external/kaggle-cell-tracking-competition/src"
        hidden = official.with_name("hidden")
        official.rename(hidden)
        with mock.patch.object(namespace["subprocess"], "run") as spawned:
            with self.assertRaisesRegex(namespace["LaunchError"], "missing or unresolved"):
                namespace["child_paths"](self.support)
            spawned.assert_not_called()
        hidden.rename(official)

        support_src = self.support / "repo/src"
        support_real = self.support / "repo/real"
        support_src.rename(support_real)
        outside = self.root / "outside"
        outside.mkdir()
        support_src.symlink_to(outside, target_is_directory=True)
        with mock.patch.object(namespace["subprocess"], "run") as spawned:
            with self.assertRaisesRegex(namespace["LaunchError"], "allowed root"):
                namespace["child_paths"](self.support)
            spawned.assert_not_called()

    def test_dependency_plan_audits_every_required_exact_wheel(self) -> None:
        wheel_dir = self.support / "wheels"
        rows = [pkgbh9.file_record(path, relative_to=wheel_dir) for path in sorted(wheel_dir.glob("*.whl"))]
        plan = pkgbh9.dependency_wheel_plan(rows)
        expected = {
            distribution
            for distribution, *_rest in pkgbh9.DEPENDENCIES
            if distribution not in pkgbh9.BASE_ENVIRONMENT_DISTRIBUTIONS
        }
        self.assertEqual({row["distribution"] for row in plan}, expected)
        self.assertTrue(all(
            pkgbh9.version_compatible(row["version"], row["minimum"], row["maximum"])
            for row in plan
        ))
        paths = [str((wheel_dir / row["path"]).resolve()) for row in plan[:2]]
        command = pkgbh9.exact_wheel_command(sys.executable, paths)
        self.assertIn("--no-index", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--force-reinstall", command)
        self.assertNotIn("--find-links", command)
        self.assertNotIn("--index-url", command)
        self.assertEqual(command[-2:], paths)

        with self.assertRaisesRegex(pkgbh9.PackageError, "exactly cover"):
            pkgbh9.dependency_wheel_plan(rows[1:])

    def test_runtime_rechecks_selected_wheel_and_pip_receives_only_exact_path(self) -> None:
        output, _report = self.build()
        source_path = output / pkgbh9.ENTRYPOINT_NAME
        namespace = {"__file__": str(source_path), "__name__": "bh9_runtime_plan_test"}
        exec(compile(source_path.read_bytes(), str(source_path), "exec"), namespace)
        attachment = json.loads((output / pkgbh9.ATTACHMENT_NAME).read_text(encoding="ascii"))
        plan = namespace["wheel_plan"](attachment["identity"]["wheelhouse"]["files"])
        self.assertEqual(plan, attachment["identity"]["dependency_wheels"])
        selected = next(row for row in plan if row["distribution"] == "tracksdata")

        before = {
            distribution: {"compatible": True, "version": minimum, "error": None}
            for distribution, _module, minimum, _maximum, _policy in pkgbh9.DEPENDENCIES
        }
        before["tracksdata"] = {"compatible": False, "version": None, "error": "missing"}
        after = {name: dict(status) for name, status in before.items()}
        after["tracksdata"] = {
            "compatible": True,
            "version": selected["version"],
            "error": None,
        }
        probes = iter((before, after))
        namespace["probe"] = lambda _end: next(probes)
        with mock.patch.object(namespace["subprocess"], "run") as run:
            result = namespace["repair"](self.support / "wheels", plan, 10**20)
        command = run.call_args.args[0]
        exact_path = str((self.support / "wheels" / selected["path"]).resolve())
        self.assertEqual(command[-1], exact_path)
        self.assertNotIn("--find-links", command)
        self.assertFalse(any(">=" in token or "<" in token for token in command))
        self.assertEqual(result["install_plan"][0]["absolute_path"], exact_path)

        (self.support / "wheels" / selected["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(namespace["LaunchError"], "basename/size/SHA"):
            namespace["verify_selected"](self.support / "wheels", [selected])

    def test_pep440_spec_rejects_attested_prerelease_but_exact_wheel_installs(self) -> None:
        wheel_dir = self.root / "adversarial-wheels"
        wheel_dir.mkdir()
        wheel = wheel_dir / "tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl"
        _make_prerelease_wheel(wheel)
        environment = dict(
            os.environ,
            PIP_NO_INDEX="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            PYTHONNOUSERSITE="1",
        )
        rejected_target = self.root / "rejected"
        rejected = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--ignore-installed", "--target", str(rejected_target),
                "--find-links", str(wheel_dir), "tracksdata>=0.1,<1",
            ],
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0, rejected.stdout)
        self.assertIn("No matching distribution", rejected.stdout)

        exact_target = self.root / "exact"
        exact = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--ignore-installed", "--target", str(exact_target), str(wheel.resolve()),
            ],
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(exact.returncode, 0, exact.stdout)
        smoke_environment = dict(environment, PYTHONPATH=str(exact_target))
        smoke = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib,importlib.metadata; "
                "m=importlib.import_module('tracksdata'); "
                "assert importlib.metadata.version('tracksdata') == "
                "'0.1.0rc6.dev3+g980c2d30a'; assert m.__version__.endswith('g980c2d30a')",
            ],
            check=False,
            env=smoke_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(smoke.returncode, 0, smoke.stdout)

    def test_checkpoint_mismatch_fails_before_output_creation(self) -> None:
        expected = dict(self.expected)
        expected["weight_sha256"] = "f" * 64
        output = self.root / "badpkg"
        with mock.patch.object(pkgbh9, "_origin_inventory", return_value=self.inventory):
            with self.assertRaisesRegex(pkgbh9.PackageError, "checkpoint SHA-256"):
                pkgbh9.build_package(
                    self.repo, output, archive=self.archive, registry=self.registry,
                    split=self.split, support_root=self.support,
                    expected_artifact_hashes=expected, expected_train_count=2,
                )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
