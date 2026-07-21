from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import mknbab


class V45ABNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "pkg"
        self.package.mkdir()
        for name in (
            "ab45.py",
            "ab45man.json",
            "bh45.py",
            "bh45att.json",
            "bh45man.json",
            "bh45src.zip",
            "kernel-metadata.json",
        ):
            (self.package / name).write_bytes((name + "\n").encode("ascii"))

    def test_notebook_embeds_all_sidecars_under_kaggle_limit(self) -> None:
        output = self.root / "out"
        report = mknbab.build_notebook(self.package, output)
        self.assertLess(report["notebook_bytes"], 1_000_000)
        self.assertTrue(report["strictly_under_budget"])
        payload = json.loads((output / "ab45.ipynb").read_text(encoding="ascii"))
        self.assertEqual(payload["metadata"]["kernelspec"]["name"], "python3")
        source = "".join(payload["cells"][0]["source"])
        self.assertIn("base64.b64decode", source)
        self.assertIn("ab45.py", source)
        self.assertIn("v45ab.json", source)
        metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="ascii"))
        self.assertEqual(metadata["code_file"], "ab45.ipynb")
        self.assertEqual(metadata["kernel_type"], "notebook")
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])

    def test_missing_or_extra_package_members_are_rejected(self) -> None:
        (self.package / "unexpected.txt").write_text("bad", encoding="ascii")
        with self.assertRaisesRegex(mknbab.NotebookError, "file set changed"):
            mknbab.build_notebook(self.package, self.root / "out")

    def test_compile_mode_is_forwarded_to_the_private_launcher(self) -> None:
        output = self.root / "out"
        report = mknbab.build_notebook(
            self.package,
            output,
            kernel_slug="biohub-v45cmp",
            mode="compile",
        )
        self.assertEqual(report["kernel"], "onebitbrain/biohub-v45cmp")
        self.assertEqual(report["mode"], "compile")
        payload = json.loads((output / "ab45.ipynb").read_text(encoding="ascii"))
        source = "".join(payload["cells"][0]["source"])
        self.assertIn("--mode", source)
        self.assertIn("compile", source)


if __name__ == "__main__":
    unittest.main()
