from __future__ import annotations

import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "dag_prompts/2026-08-16-five-kernel-validation-v1/ascent/implementation/"
    "ascent_safe_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("ascent_safe_smoke", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def make_archive(root: Path, *, link_name: str, link_target: str, hardlink: bool = False) -> Path:
    source = root / "source"
    (source / "pkg/config/backbone").mkdir(parents=True)
    (source / "pkg/config/backbone/value.txt").write_text("bound\n", encoding="utf-8")
    archive = root / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source / "pkg", arcname="pkg", recursive=True)
        link = tarfile.TarInfo(link_name)
        link.type = tarfile.LNKTYPE if hardlink else tarfile.SYMTYPE
        link.linkname = link_target
        handle.addfile(link)
    return archive


class AscentSafeExtractTests(unittest.TestCase):
    def test_internal_directory_alias_becomes_physical_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = make_archive(
                root,
                link_name="pkg/config/meta/backbone",
                link_target="../backbone",
            )
            output = root / "output"
            extracted = RUNNER.safe_extract(
                archive,
                output,
                RUNNER.sha256_file(archive),
                allowed_directory_aliases={
                    "pkg/config/meta/backbone": "../backbone"
                },
            )
            alias = extracted / "config/meta/backbone"
            self.assertTrue(alias.is_dir())
            self.assertFalse(alias.is_symlink())
            self.assertEqual((alias / "value.txt").read_text(), "bound\n")

    def test_absolute_escape_unlisted_missing_and_hardlink_fail(self) -> None:
        cases = (
            ("pkg/config/meta/backbone", "/etc", False, {"pkg/config/meta/backbone": "/etc"}),
            ("pkg/config/meta/backbone", "../../../../escape", False, {"pkg/config/meta/backbone": "../../../../escape"}),
            ("pkg/config/meta/backbone", "../backbone", False, {}),
            ("pkg/config/meta/backbone", "../missing", False, {"pkg/config/meta/backbone": "../missing"}),
            ("pkg/config/meta/backbone", "../backbone", True, {"pkg/config/meta/backbone": "../backbone"}),
        )
        for link_name, link_target, hardlink, allowlist in cases:
            with (
                self.subTest(target=link_target, hardlink=hardlink),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                archive = make_archive(
                    root,
                    link_name=link_name,
                    link_target=link_target,
                    hardlink=hardlink,
                )
                with self.assertRaises(RUNNER.SmokeError):
                    RUNNER.safe_extract(
                        archive,
                        root / "output",
                        RUNNER.sha256_file(archive),
                        allowed_directory_aliases=allowlist,
                    )

    def test_exact_official_channelvit_alias_is_root_confined_and_physical(self) -> None:
        archive = RUNNER.CHANNELVIT_ARCHIVE
        with tempfile.TemporaryDirectory() as directory:
            extracted = RUNNER.safe_extract(
                archive,
                Path(directory) / "output",
                RUNNER.CHANNELVIT_SHA256,
            )
            alias = extracted / "channelvit/config/meta_arch/backbone"
            source = extracted / "channelvit/config/backbone"
            self.assertTrue(alias.is_dir())
            self.assertFalse(alias.is_symlink())
            self.assertEqual(
                sorted(path.relative_to(alias) for path in alias.rglob("*") if path.is_file()),
                sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file()),
            )


if __name__ == "__main__":
    unittest.main()
