from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1_e6_affine_tvl1_prefix52 import (  # noqa: E402
    _fit_affine,
    _flow_at,
    _predict_affine,
)


def test_affine_fit_recovers_known_12_dof_transform() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    linear = np.asarray(
        [[1.0, 0.2, 0.0], [0.0, 0.9, -0.1], [0.1, 0.0, 1.1]]
    )
    offset = np.asarray([4.0, -2.0, 3.0])
    target = source @ linear + offset
    center, coefficient, rank, condition = _fit_affine(source, target)
    assert rank == 4
    assert np.isfinite(condition)
    for point, expected in zip(source, target, strict=True):
        np.testing.assert_allclose(
            _predict_affine(point, center, coefficient), expected, atol=1.0e-12
        )


def test_flow_sampling_uses_zyx_component_order() -> None:
    flow = np.zeros((3, 4, 5, 6), dtype=np.float32)
    flow[0, ...] = 1.0
    flow[1, ...] = -2.0
    flow[2, ...] = 3.5
    observed = _flow_at(
        flow,
        np.asarray([2.0, 8.0, 12.0]),
        np.asarray([1.0, 4.0, 4.0]),
    )
    np.testing.assert_allclose(observed, [1.0, -2.0, 3.5])
