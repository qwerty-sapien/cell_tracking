from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHIPPED_ROOTS = (
    ROOT / "src/celltrack_alpha",
    ROOT / "kaggle_kernel/celltrack_alpha",
)
FORBIDDEN_MODULE_PREFIXES = (
    "external",
    "unetext3Dn_con7",
    "unetext3Dn_con7s",
    "recoloss",
)
FORBIDDEN_DYNAMIC_LOADERS = {
    "SourceFileLoader",
    "spec_from_file_location",
    "run_path",
}


def forbidden_reference_imports(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_MODULE_PREFIXES):
                    findings.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(FORBIDDEN_MODULE_PREFIXES):
                findings.append(f"line {node.lineno}: from {module}")
        elif isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if name in FORBIDDEN_DYNAMIC_LOADERS:
                findings.append(f"line {node.lineno}: dynamic source loader {name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized = node.value.replace("\\", "/").lower()
            if "external/cellect" in normalized or "unetext3dn_con7" in normalized:
                findings.append(f"line {node.lineno}: forbidden source path/module literal")
    return findings


class CellectPackagingGuardTests(unittest.TestCase):
    def test_shipped_modules_never_import_or_load_reference_source(self) -> None:
        findings: dict[str, list[str]] = {}
        for root in SHIPPED_ROOTS:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                violations = forbidden_reference_imports(path)
                if violations:
                    findings[str(path.relative_to(ROOT))] = violations
        self.assertEqual(findings, {}, f"GPL reference dependency reached shipped code: {findings}")

    def test_guard_detects_direct_and_dynamic_reference_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            direct = Path(temporary) / "direct.py"
            direct.write_text("from external.CELLECT import recoloss\n", encoding="utf-8")
            dynamic = Path(temporary) / "dynamic.py"
            dynamic.write_text(
                "import importlib.util\n"
                "importlib.util.spec_from_file_location('x', 'external/CELLECT/model.py')\n",
                encoding="utf-8",
            )
            self.assertTrue(forbidden_reference_imports(direct))
            self.assertTrue(forbidden_reference_imports(dynamic))


if __name__ == "__main__":
    unittest.main()
