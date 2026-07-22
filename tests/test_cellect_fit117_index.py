from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_cellect_fit117_index.py"
SPEC = importlib.util.spec_from_file_location("build_cellect_fit117_index", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CellectFitIndexTests(unittest.TestCase):
    def test_internal_split_is_deterministic_stratified_and_group_safe(self) -> None:
        identities = [
            *(f"train:44b6_{index:08x}" for index in range(10)),
            *(f"train:6bba_{index:08x}" for index in range(15)),
        ]
        train, validation = MODULE.stratified_video_split(identities, 0.2)
        self.assertEqual((len(train), len(validation)), (20, 5))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(identities))
        self.assertEqual(
            (train, validation),
            MODULE.stratified_video_split(reversed(identities), 0.2),
        )
        self.assertEqual(sum("train:44b6_" in value for value in validation), 2)
        self.assertEqual(sum("train:6bba_" in value for value in validation), 3)


if __name__ == "__main__":
    unittest.main()
