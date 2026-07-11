from __future__ import annotations

import builtins
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_ultrack.core import UltrackConfig, _configure_ultrack  # noqa: E402


class UltrackBackendTests(unittest.TestCase):
    def test_missing_optional_backend_has_actionable_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'ultrack'", name="ultrack")
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "optional backend"):
                _configure_ultrack(UltrackConfig())

    def test_type_annotation_import_crash_is_wrapped(self) -> None:
        real_import = builtins.__import__

        def broken_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "ultrack":
                raise TypeError(
                    "unsupported operand type(s) for |: 'builtin_function_or_method' and 'NoneType'"
                )
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=broken_import):
            with self.assertRaisesRegex(RuntimeError, "incompatible Ultrack/Python"):
                _configure_ultrack(UltrackConfig())

    def test_missing_nested_dependency_is_not_misreported_as_missing_ultrack(self) -> None:
        missing = ModuleNotFoundError("No module named 'nested_dependency'", name="nested_dependency")
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError) as caught:
                _configure_ultrack(UltrackConfig())
        self.assertIs(caught.exception, missing)


if __name__ == "__main__":
    unittest.main()
