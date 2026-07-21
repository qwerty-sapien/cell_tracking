from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import mknb9, pkgbh9


def make_package(root: Path) -> dict[str, bytes]:
    files = {
        pkgbh9.ARCHIVE_NAME: b"source-archive",
        pkgbh9.ATTACHMENT_NAME: b"{}\n",
        pkgbh9.METADATA_NAME: json.dumps({
            "is_private": True, "enable_internet": False, "enable_gpu": True,
            "competition_sources": [pkgbh9.COMPETITION_SLUG],
            "dataset_sources": [pkgbh9.SUPPORT_REF],
        }, sort_keys=True).encode("ascii") + b"\n",
    }
    rows = [
        {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in sorted(files.items())
        if name in mknb9.RUNTIME_FILES
    ]
    package_bytes = 0
    while True:
        identity = {
            "private": True, "internet": False, "trials": 1,
            "accelerator": "T4x2", "deadline_seconds": 3000,
            "runtime_attestation_level": 6,
            "self_contained_code_file": True,
            "offline_dependency_repair": "attested_exact_wheel_paths_no_index_no_deps",
            "embedded_files": sorted(
                (pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME)
            ),
            "hermetic_child_import_layout": list(pkgbh9.CHILD_IMPORT_LAYOUT),
            "preserved_environment_keys": list(pkgbh9.PRESERVED_ENVIRONMENT_KEYS),
            "child_cwd": "materialized_source_root",
            "python_safe_path": True,
            "inherits_pythonpath": False,
            "files": rows, "package_bytes": package_bytes,
        }
        manifest = {
            "schema": "bh9-kaggle-package.v1", "identity": identity,
            "identity_sha256": pkgbh9.sha256_bytes(pkgbh9.canonical_json(identity)),
        }
        raw = pkgbh9.canonical_json(manifest) + b"\n"
        observed = sum(len(files[name]) for name in mknb9.RUNTIME_FILES) + len(raw)
        if observed == package_bytes:
            files[pkgbh9.MANIFEST_NAME] = raw
            break
        package_bytes = observed
    files[pkgbh9.ENTRYPOINT_NAME] = pkgbh9._entrypoint_payload({
        name: files[name]
        for name in (pkgbh9.MANIFEST_NAME, pkgbh9.ATTACHMENT_NAME, pkgbh9.ARCHIVE_NAME)
    })
    root.mkdir()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return files


class Bh9NotebookTests(unittest.TestCase):
    def test_notebook_embeds_only_hash_attested_package_in_one_cell(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            package = root / "bh9pkg"
            files = make_package(package)
            output = root / "bh9.ipynb"
            report = mknb9.build_notebook(package, output)
            payload = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(len(payload["cells"]), 1)
            self.assertEqual(payload["metadata"]["kaggle"]["accelerator"], "T4x2")
            self.assertFalse(payload["metadata"]["kaggle"]["enable_internet"])
            self.assertTrue(payload["metadata"]["kaggle"]["is_private"])
            self.assertEqual(payload["metadata"]["kaggle"]["trials"], 1)
            source = "".join(payload["cells"][0]["source"])
            line = next(line for line in source.splitlines() if line.startswith("ENVELOPE = json.loads("))
            encoded = json.loads(line[len("ENVELOPE = json.loads("):-1])
            envelope = json.loads(encoded)
            self.assertEqual(set(envelope), {pkgbh9.ENTRYPOINT_NAME})
            for name in envelope:
                value = files[name]
                row = envelope[name]
                self.assertEqual(base64.b64decode(row["base64"]), value)
                self.assertEqual(row["sha256"], hashlib.sha256(value).hexdigest())
            self.assertIn("embedded code_file hash mismatch", source)
            self.assertIn("bh9fail.json", source)
            self.assertIn("bh9out/run.json", source)
            self.assertIn('"-P"', source)
            self.assertIn("cwd=str(WORK)", source)
            self.assertIn("PYTHONSAFEPATH", source)
            self.assertNotIn("dict(os.environ", source)
            self.assertLess(report["bytes"], 1_000_000)
            self.assertEqual(report["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_build_is_deterministic_and_rejects_tampering_or_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            package = root / "bh9pkg"
            make_package(package)
            first = root / "one.ipynb"
            second = root / "two.ipynb"
            self.assertEqual(
                mknb9.build_notebook(package, first)["sha256"],
                mknb9.build_notebook(package, second)["sha256"],
            )
            (package / pkgbh9.ENTRYPOINT_NAME).write_bytes(b"tampered")
            with self.assertRaisesRegex(mknb9.NotebookError, "entrypoint"):
                mknb9.notebook_payload(package)

        with tempfile.TemporaryDirectory() as raw:
            package = Path(raw) / "bh9pkg"
            make_package(package)
            (package / "extra.txt").write_bytes(b"x")
            with self.assertRaisesRegex(mknb9.NotebookError, "member set mismatch"):
                mknb9.notebook_payload(package)

    def test_entrypoint_envelope_matches_every_runtime_sibling_and_fails_on_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            package = Path(raw) / "bh9pkg"
            files = make_package(package)
            source = files[pkgbh9.ENTRYPOINT_NAME]
            embedded = mknb9._embedded_files(source)
            self.assertEqual(
                embedded,
                {
                    name: files[name]
                    for name in (
                        pkgbh9.MANIFEST_NAME,
                        pkgbh9.ATTACHMENT_NAME,
                        pkgbh9.ARCHIVE_NAME,
                    )
                },
            )
            marker = b'\\"b85\\":\\"'
            index = source.index(marker) + len(marker)
            replacement = b"0" if source[index:index + 1] != b"0" else b"1"
            tampered = source[:index] + replacement + source[index + 1:]
            with self.assertRaisesRegex(mknb9.NotebookError, "entrypoint"):
                mknb9._embedded_files(tampered)

    def test_created_notebook_name_must_be_short(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            package = root / "bh9pkg"
            make_package(package)
            with self.assertRaisesRegex(mknb9.NotebookError, "under 15"):
                mknb9.build_notebook(package, root / "benchmark9.ipynb")


if __name__ == "__main__":
    unittest.main()
