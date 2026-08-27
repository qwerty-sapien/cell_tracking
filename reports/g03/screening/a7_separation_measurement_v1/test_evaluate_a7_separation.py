from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_a7_separation.py")
SPEC = importlib.util.spec_from_file_location("a7_separation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_a0_is_deterministic_exact_within_frame_permutation() -> None:
    supplier = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
    source_t = np.asarray([0, 0, 0, 1, 1])
    source_id = np.asarray([10, 11, 12, 13, 14])
    target_id = np.asarray([20, 21, 22, 23, 24])
    kwargs = {
        "identity": "train:synthetic",
        "source_t": source_t,
        "source_id": source_id,
        "target_id": target_id,
    }
    first, stats = MODULE.a0_permutation(supplier, **kwargs)
    second, second_stats = MODULE.a0_permutation(supplier, **kwargs)
    assert np.array_equal(first, second)
    assert stats == second_stats == {"moved_rows": 5, "singleton_rows": 0}
    assert np.array_equal(np.sort(first[:3, 0]), np.sort(supplier[:3, 0]))
    assert np.array_equal(np.sort(first[3:, 0]), np.sort(supplier[3:, 0]))
    assert not np.array_equal(first, supplier)


def test_a0_preserves_singletons() -> None:
    supplier = np.asarray([[7.0], [8.0], [9.0]])
    a0, stats = MODULE.a0_permutation(
        supplier,
        identity="train:synthetic",
        source_t=np.asarray([0, 1, 2]),
        source_id=np.asarray([1, 2, 3]),
        target_id=np.asarray([4, 5, 6]),
    )
    assert np.array_equal(a0, supplier)
    assert stats == {"moved_rows": 0, "singleton_rows": 3}
