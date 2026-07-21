from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

from scripts import mknb45


class NotebookPackageTests(unittest.TestCase):
    def test_notebook_embeds_exact_package_under_source_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "pkg"
            package.mkdir()
            expected = {}
            for index, name in enumerate(mknb45.PACKAGE_FILES):
                value = f"fixture-{index}-{name}".encode("ascii")
                (package / name).write_bytes(value)
                expected[name] = base64.b64encode(value).decode("ascii")
            output = root / "bh45.ipynb"
            report = mknb45.build_notebook(package, output)
            payload = json.loads(output.read_text(encoding="ascii"))
            source = "".join(payload["cells"][0]["source"])
            marker = "FILES = json.loads("
            encoded_line = next(line for line in source.splitlines() if line.startswith(marker))
            encoded_json = json.loads(encoded_line[len(marker):-1])
            self.assertEqual(json.loads(encoded_json), expected)
            self.assertLess(report["bytes"], 1_000_000)
            self.assertNotIn(str(package), source)
            self.assertIn("subprocess.run", source)
            self.assertIn("bhdiag.json", source)

    def test_unexpected_package_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            for name in mknb45.PACKAGE_FILES:
                (package / name).write_bytes(b"x")
            (package / "extra.txt").write_bytes(b"x")
            with self.assertRaisesRegex(mknb45.NotebookPackageError, "unexpected"):
                mknb45.notebook_payload(package)


if __name__ == "__main__":
    unittest.main()
