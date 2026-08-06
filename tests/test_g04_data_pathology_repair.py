from __future__ import annotations

import numpy as np

from scripts import g04_data_pathology_repair as repair


def test_evaluator_stride_is_literal_subsample_not_block_mean() -> None:
    frame = np.arange(np.prod(repair.SHAPE_TZYX[1:]), dtype=np.uint16).reshape(
        repair.SHAPE_TZYX[1:]
    )
    observed = repair.evaluator_stride(frame)
    expected = frame[:, ::4, ::4]
    np.testing.assert_array_equal(observed, expected)
    assert observed.shape == (64, 64, 64)
    assert not np.array_equal(
        observed,
        frame.reshape(64, 64, 4, 64, 4).mean(axis=(2, 4)).astype(np.uint16),
    )


def test_integer_phase_shift_recovers_moving_to_reference_translation() -> None:
    rng = np.random.default_rng(20260803)
    reference = rng.normal(size=(64, 64, 64)).astype(np.float32)
    moving = np.roll(reference, shift=(3, -2, 5), axis=(0, 1, 2))
    observed = repair.integer_phase_shift(reference, moving)
    assert observed == (-3, 2, -5)
    np.testing.assert_allclose(
        repair.overlap_ncc(reference, moving, observed), 1.0, rtol=0, atol=1e-6
    )


def test_shift_magnitude_uses_isotropic_registered_grid() -> None:
    assert repair.REGISTERED_SCALE_ZYX_UM == (1.625, 1.625, 1.625)
    np.testing.assert_allclose(
        repair.shift_magnitude_um((3, 4, 0)), 8.125, rtol=0, atol=1e-12
    )


def test_overlap_slices_move_target_coordinates_by_reported_shift() -> None:
    reference, moving = repair.overlap_slices((8, 9, 10), (-2, 3, 0))
    assert reference == (slice(0, 6), slice(3, 9), slice(0, 10))
    assert moving == (slice(2, 8), slice(0, 6), slice(0, 10))
