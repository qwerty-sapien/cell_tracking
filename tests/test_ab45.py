from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts import mkab45, pkgbh


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class V45ABPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.support = self.root / "external" / "Tracking_Support"
        for relative in pkgbh.SOURCE_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {relative}\n", encoding="ascii")
        values = {
            "ARTIFACT_MANIFEST.json": b"manifest\n",
            "weights/unet_transformer/split_0/config.json": b"{}\n",
            "weights/unet_transformer/split_0/edge_predictor_best.pth": b"checkpoint\x00",
            "repo/src/biohub_tracking/models/temporal_unet.py": b"class TemporalUNet3D: pass\n",
            "repo/src/biohub_tracking/models/simple_node_transformer.py": b"class SimpleNodeTransformer: pass\n",
            "repo/scripts/train_unet_transformer.py": b"# training provenance\n",
        }
        for relative, value in values.items():
            path = self.support / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        wheels = self.support / "wheels"
        wheels.mkdir(parents=True)
        (wheels / "zarr-3.2.1-py3-none-any.whl").write_bytes(b"wheel")
        registry = self.root / "reports" / "g02" / "dataset_registry_v1.json"
        registry.parent.mkdir(parents=True)
        rows = []
        for index, name in enumerate((mkab45.BENCHMARK_DATASET, "d1", "d2", "d3")):
            rows.append(
                {
                    "split": "test",
                    "dataset": name,
                    "video_id": f"test:{name}",
                    "exact_identity_sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
                    "compact_metadata": {"image_shape": [100, 8, 16, 16]},
                }
            )
        registry.write_text(json.dumps({"artifact": "dataset_registry_v1", "records": rows}), encoding="utf-8")
        self.registry = registry
        self.expected = {
            "manifest_sha256": _sha(values["ARTIFACT_MANIFEST.json"]),
            "config_sha256": _sha(values["weights/unet_transformer/split_0/config.json"]),
            "source_sha256": pkgbh.source_tree_sha256(self.support),
            "weight_sha256": _sha(values["weights/unet_transformer/split_0/edge_predictor_best.pth"]),
        }

    def build(self, name: str) -> tuple[Path, dict]:
        output = Path(self.temporary.name) / name
        report = mkab45.build_package(
            self.root,
            output,
            registry=self.registry,
            support_root=self.support,
            expected_artifact_hashes=self.expected,
        )
        return output, report

    def test_private_ab_package_is_deterministic_and_under_budget(self) -> None:
        first, first_report = self.build("pkg1")
        second, second_report = self.build("pkg2")
        first_files = {path.name: path.read_bytes() for path in first.iterdir()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir()}
        self.assertEqual(first_files, second_files)
        self.assertEqual(
            set(first_files),
            {"ab45.py", "ab45man.json", "bh45.py", "bh45att.json", "bh45man.json", "bh45src.zip", "kernel-metadata.json"},
        )
        self.assertLess(first_report["aggregate_bytes"], 1_000_000)
        self.assertTrue(first_report["strictly_under_budget"])
        self.assertFalse(first_report["submission_generation"])
        self.assertEqual(first_report["benchmark_dataset"], mkab45.BENCHMARK_DATASET)
        metadata = json.loads(first_files[pkgbh.METADATA_NAME])
        self.assertEqual(metadata["id"], mkab45.KERNEL_REF)
        self.assertEqual(metadata["code_file"], "ab45.py")
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(first_report["aggregate_bytes"], second_report["aggregate_bytes"])

    def test_both_manifests_cover_the_rewritten_private_package(self) -> None:
        output, _report = self.build("pkg")
        ab_manifest = json.loads((output / "ab45man.json").read_text(encoding="ascii"))
        identity = ab_manifest["identity"]
        self.assertEqual(ab_manifest["identity_sha256"], _sha(mkab45._canonical(identity)))
        self.assertFalse(identity["submission_generation"])
        self.assertEqual(identity["benchmark"]["view_batch_sizes"], [1, 4])
        self.assertTrue(identity["benchmark"]["requires_exact_feature_logit_and_detection_equality"])
        self.assertEqual(
            identity["package_bytes"],
            sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        )
        for item in identity["files"]:
            path = output / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(_sha(path.read_bytes()), item["sha256"])

        spec = importlib.util.spec_from_file_location("bh45_ab_fixture", output / "bh45.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        try:
            verified = module._verify_package()
        finally:
            sys.modules.pop(spec.name, None)
        self.assertEqual(verified["identity"]["kernel_slug"], pkgbh.KERNEL_SLUG)
        compile((output / "ab45.py").read_text(encoding="ascii"), "ab45.py", "exec")

    def test_target_must_exist_in_exact_registry(self) -> None:
        payload = json.loads(self.registry.read_text(encoding="utf-8"))
        payload["records"][0]["dataset"] = "wrong"
        self.registry.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(mkab45.PackageError, "absent"):
            self.build("bad")

    def test_compile_mode_signs_the_distinct_kernel_and_runtime_contract(self) -> None:
        output = Path(self.temporary.name) / "cmp"
        report = mkab45.build_package(
            self.root,
            output,
            registry=self.registry,
            support_root=self.support,
            expected_artifact_hashes=self.expected,
            kernel_slug="biohub-v45cmp",
            mode="compile",
        )
        self.assertEqual(report["kernel"], "onebitbrain/biohub-v45cmp")
        self.assertEqual(report["mode"], "compile")
        manifest = json.loads((output / "ab45man.json").read_text(encoding="ascii"))
        benchmark = manifest["identity"]["benchmark"]
        self.assertEqual(benchmark["mode"], "compile")
        self.assertEqual(benchmark["view_batch_sizes"], [1])
        source = (output / "ab45.py").read_text(encoding="ascii")
        self.assertIn("fullgraph=True", source)
        self.assertIn("mode=\"reduce-overhead\"", source)


if __name__ == "__main__":
    unittest.main()
