#!/usr/bin/env python3
"""Focused local validation for the self-contained OCC sliced-W1 kernel."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from occ_mass_transport_core import (  # noqa: E402
    extract_reflect_patch as reference_extract,
    sliced_pair_features as reference_features,
)


KERNEL_DIR = ROOT / "reports/g02/screening/tiered_surface/branches/occ_sliced_w1_v2/kaggle_kernel"
CODE = KERNEL_DIR / "kaggle_occ_sliced_w1_v2.py"
METADATA = KERNEL_DIR / "kernel-metadata.json"


class DummyArray:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.shape = value.shape

    def __getitem__(self, key: object) -> np.ndarray:
        return self.value[key]


def kernel_functions() -> dict[str, object]:
    tree = ast.parse(CODE.read_text(encoding="utf-8"), filename=str(CODE))
    wanted = {
        "extract_reflect_patch",
        "normalized_mass",
        "histogram_w1",
        "binned_mass",
        "sliced_features",
        "permutation_indices",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])

    class KernelError(RuntimeError):
        pass

    directions = np.asarray(
        ((1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)),
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    namespace = {
        "np": np,
        "zarr": zarr,
        "hashlib": __import__("hashlib"),
        "KernelError": KernelError,
        "CROP_RADIUS_ZYX": (3, 12, 12),
        "PROJECTION_DIRECTIONS": directions,
        "FEATURE_NAMES": tuple(range(14)),
    }
    exec(compile(module, str(CODE), "exec"), namespace)
    return namespace


def main() -> int:
    source = CODE.read_text(encoding="utf-8")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "onebitbrain/g02-occ-sliced-w1-repair2-v1"
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert metadata["dataset_sources"] == ["onebitbrain/g02-occ-sliced-w1-input-v1"]
    for forbidden in (
        "submission" + ".csv",
        "oracle_" + "label",
        "oracle_" + "delta",
        ".rglob(",
        "os.walk(",
        "g02-cellect-r7-published-assets-v1",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "PROJECTION_LIMIT_SECONDS",
        "SOFT_STOP_SECONDS",
        "worker projection exceeds 1.0 h",
    ):
        assert forbidden not in source

    functions = kernel_functions()
    rng = np.random.default_rng(20260809)
    source_patches = rng.integers(0, 65535, size=(7, 7, 25, 25), dtype=np.uint16)
    target_patches = rng.integers(0, 65535, size=(7, 7, 25, 25), dtype=np.uint16)
    observed = functions["sliced_features"](source_patches, target_patches)
    expected = reference_features(source_patches, target_patches)
    assert observed.dtype == expected.dtype == np.float32
    assert np.array_equal(observed, expected)

    raw = rng.integers(0, 65535, size=(2, 8, 27, 27), dtype=np.uint16)
    dummy = DummyArray(raw)
    for center in (np.asarray((0, 0, 0)), np.asarray((7, 26, 26)), np.asarray((3, 13, 13))):
        observed_patch = functions["extract_reflect_patch"](dummy, 0, center)
        expected_patch = reference_extract(dummy, 0, center)
        assert np.array_equal(observed_patch, expected_patch)

    node_ids = np.asarray((11, 11, 12, 13, 21, 22), dtype=np.int64)
    frames = np.asarray((1, 1, 1, 1, 2, 2), dtype=np.int16)
    first = functions["permutation_indices"](node_ids, frames, "train:44b6_test")
    second = functions["permutation_indices"](node_ids, frames, "train:44b6_test")
    assert np.array_equal(first, second)
    assert np.all(frames[first] == frames)

    print("14 focused OCC sliced-W1 v2 kernel checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
