"""Private package invariants for the bounded one-pass V45 student fit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_v45f


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract_runtime(output: Path, root: Path):
    """Replay Kaggle script mode: upload only ``code_file`` and metadata."""

    script_root = root / "script"
    script_root.mkdir()
    (script_root / build_v45f.ENTRY).write_bytes((output / build_v45f.ENTRY).read_bytes())
    wrapper_name = f"v45f_wrap_{len(sys.modules)}"
    with mock.patch.dict(os.environ, {"V45F_WRAP_WORK": str(root / "work")}, clear=False):
        wrapper = _load_module(wrapper_name, script_root / build_v45f.ENTRY)
        with mock.patch.object(wrapper.runpy, "run_path") as run_path:
            self_result = wrapper.main()
        if self_result != 0:
            raise RuntimeError("wrapper did not return success")
    expected_run = root / "work" / build_v45f.RUNTIME_ROOT / build_v45f.RUN
    run_path.assert_called_once_with(str(expected_run), run_name="__main__")
    return expected_run.parent


class BuildV45FastTests(unittest.TestCase):
    def test_rejects_unsupported_armed_cuda_architecture(self) -> None:
        trainer = _load_module("v45f_device", ROOT / "scripts" / "train_v45f.py")
        with (
            mock.patch.object(trainer.torch.cuda, "is_available", return_value=True),
            mock.patch.object(trainer.torch.cuda, "get_device_capability", return_value=(6, 0)),
        ):
            with self.assertRaisesRegex(trainer.V45FastTrain, "capability >= 7.0"):
                trainer._device(armed=True)
        with mock.patch.object(trainer.torch.cuda, "is_available", return_value=False):
            self.assertEqual(trainer._device(armed=False), "cpu")

    def test_builds_bounded_package_with_exact_fit_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "v45fpkg"
            report = build_v45f.build(output)
            self.assertLess(report["aggregate_bytes"], build_v45f.SOURCE_LIMIT)
            self.assertFalse(report["network_or_kaggle_actions"])
            self.assertEqual(len(report["selected_sources"]), 12)
            for path in output.iterdir():
                self.assertTrue(path.name == build_v45f.METADATA or len(path.name) <= 15)
            self.assertEqual({path.name for path in output.iterdir()}, {build_v45f.ENTRY, build_v45f.METADATA})
            source = (output / build_v45f.ENTRY).read_text(encoding="ascii")
            self.assertTrue(source.isascii())
            self.assertNotIn("import bh45", source)
            compile(source, build_v45f.ENTRY, "exec")

            runtime = _extract_runtime(output, root)
            self.assertEqual(sorted(path.name for path in runtime.iterdir()), report["runtime_members"])
            manifest = json.loads((runtime / build_v45f.MANIFEST).read_text(encoding="ascii"))
            identity = manifest["identity"]
            canonical = json.dumps(identity, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")
            self.assertEqual(manifest["identity_sha256"], hashlib.sha256(canonical).hexdigest())
            self.assertEqual(hashlib.sha256((runtime / build_v45f.MANIFEST).read_bytes()).hexdigest(), report["package_sha256"])
            self.assertEqual(identity["fit"]["teacher"], "exact-sequential-d4-8-logit-mean")
            self.assertEqual(identity["fit"]["student_views"], [0])
            self.assertEqual(identity["fit"]["accelerator"], build_v45f.ACCELERATOR)
            self.assertFalse(identity["fit"]["division_emission"])

            wrapper = _load_module("v45f_guard", output / build_v45f.ENTRY)
            for unsafe in ("", "../escape", "/absolute", "C:drive", "nested\\escape", "./local"):
                self.assertFalse(wrapper._safe_member(unsafe))
            with mock.patch.object(wrapper, "BUNDLE_SHA256", "0" * 64):
                with self.assertRaisesRegex(wrapper.WrapperError, "identity mismatch"):
                    wrapper._decode_bundle()

            launcher = _load_module("v45f_boot", runtime / build_v45f.BOOT)
            launcher.MANIFEST = runtime / build_v45f.MANIFEST
            launcher.ATTACHMENT = runtime / build_v45f.ATTACHMENT
            launcher.SOURCE_ARCHIVE = runtime / build_v45f.ARCHIVE
            launcher.SOURCE_ROOT = root / "v45fsrc"
            package = launcher._verify_package()
            launcher._extract_source(package)
            projection = launcher.SOURCE_ROOT / "fitrec.json"
            self.assertTrue(projection.is_file())
            isolated = subprocess.run(
                [
                    sys.executable,
                    str(launcher.SOURCE_ROOT / "scripts" / "train_v45f.py"),
                    "--fold",
                    "0",
                    "--dry-run",
                    "--support-root",
                    str(ROOT / "external" / "Tracking_Support"),
                    "--split-manifest",
                    str(launcher.SOURCE_ROOT / "split.json"),
                    "--provenance-manifest",
                    str(launcher.SOURCE_ROOT / "prov.json"),
                    "--record-projection",
                    str(projection),
                    "--output-dir",
                    str(root / "dryout"),
                ],
                cwd=launcher.SOURCE_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        (str(launcher.SOURCE_ROOT / "src"), str(ROOT / "external" / "Tracking_Support" / "repo" / "src"))
                    ),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(isolated.returncode, 0, isolated.stderr[-4000:])
            with zipfile.ZipFile(runtime / build_v45f.ARCHIVE) as archive:
                self.assertEqual(sorted(archive.namelist()), [item["path"] for item in identity["archive"]["members"]])
                self.assertIn("scripts/train_v45f.py", archive.namelist())
                self.assertIn("src/celltrack_alpha/fast_data.py", archive.namelist())

            trainer = _load_module("v45f_train", launcher.SOURCE_ROOT / "scripts" / "train_v45f.py")
            args = trainer._args(
                [
                    "--fold", "0", "--output-dir", str(root / "out"), "--dry-run",
                    "--split-manifest", str(launcher.SOURCE_ROOT / "split.json"),
                    "--provenance-manifest", str(launcher.SOURCE_ROOT / "prov.json"),
                    "--record-projection", str(projection),
                ]
            )
            records, _split, _provenance, source = trainer._scope(args)
            self.assertEqual([record.json() for record in records], report["selected_sources"])
            self.assertEqual(source["projection_sha256"], report["projection_sha256"])

    def test_rejects_existing_output_and_tampered_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "v45fpkg"
            build_v45f.build(output)
            with self.assertRaisesRegex(build_v45f.PackageError, "existing package"):
                build_v45f.build(output)
            runtime = _extract_runtime(output, root)
            with zipfile.ZipFile(runtime / build_v45f.ARCHIVE) as archive:
                projection = json.loads(archive.read("fitrec.json"))
                split = root / "split.json"
                provenance = root / "prov.json"
                split.write_bytes(archive.read("split.json"))
                provenance.write_bytes(archive.read("prov.json"))
            projection["selected_sources"][0]["stem"] = "44b6_00000000"
            altered = root / "fitrec.json"
            altered.write_bytes(json.dumps(projection, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
            trainer = _load_module("v45f_tamper", ROOT / "scripts" / "train_v45f.py")
            args = trainer._args(
                [
                    "--fold", "0", "--output-dir", str(root / "out"), "--dry-run",
                    "--split-manifest", str(split), "--provenance-manifest", str(provenance),
                    "--record-projection", str(altered),
                ]
            )
            with self.assertRaisesRegex(trainer.V45FastTrain, "normalization changed"):
                trainer._scope(args)


if __name__ == "__main__":
    unittest.main()
