#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "separation_features", HERE / "materialize_separation_features.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SeparationFeatureTests(unittest.TestCase):
    def test_permutation_is_transition_local_deterministic_and_capacity_matched(self) -> None:
        values = np.arange(32, dtype=np.float32).reshape(4, 8)
        source = np.asarray([1, 2, 3, 4])
        target = np.asarray([11, 12, 13, 14])
        frames = np.asarray([5, 5, 6, 6])
        first = MODULE.permute_within_transition(
            values, source, target, frames, "train:44b6_fixture"
        )
        second = MODULE.permute_within_transition(
            values, source, target, frames, "train:44b6_fixture"
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(
            sorted(map(tuple, first.tolist())), sorted(map(tuple, values.tolist()))
        )
        self.assertEqual(
            sorted(map(tuple, first[:2].tolist())),
            sorted(map(tuple, values[:2].tolist())),
        )
        self.assertEqual(
            sorted(map(tuple, first[2:].tolist())),
            sorted(map(tuple, values[2:].tolist())),
        )


if __name__ == "__main__":
    unittest.main()
