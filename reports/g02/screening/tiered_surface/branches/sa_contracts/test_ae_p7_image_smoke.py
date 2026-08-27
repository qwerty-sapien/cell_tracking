from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ae_p7_image_smoke", HERE / "ae_p7_image_smoke.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeArray:
    shape = (2, 5, 5, 5)

    def __init__(self) -> None:
        self.value = np.arange(250, dtype=np.uint16).reshape(self.shape)

    def __getitem__(self, key):
        return self.value[key]


def test_center_crop_replicates_borders() -> None:
    crop = MODULE.center_crop(FakeArray(), 0, (0, 0, 0), (3, 3, 3))
    assert crop.shape == (3, 3, 3)
    assert crop[0, 0, 0] == 0
    assert crop[2, 2, 2] == 31


def test_phase_features_are_deterministic_and_finite() -> None:
    source = np.zeros((9, 9, 9), dtype=np.float32)
    source[4, 4, 4] = 1.0
    destination = np.roll(source, shift=(1, -2, 3), axis=(0, 1, 2))
    first = MODULE.phase_features(source, destination)
    second = MODULE.phase_features(source, destination)
    assert first == second
    assert np.isfinite([first["peak"], first["second_peak"], first["peak_ratio"]]).all()


def test_normalized_rejects_degenerate_quantiles() -> None:
    try:
        MODULE.normalized(np.ones((2, 2, 2), dtype=np.uint16), {"0.01": 1, "0.99": 1})
    except RuntimeError as error:
        assert "quantiles" in str(error)
    else:
        raise AssertionError("degenerate quantiles accepted")
