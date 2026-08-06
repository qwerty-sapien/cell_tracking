from __future__ import annotations

import numpy as np
import pytest

from celltrack_alpha.frame_registration import (
    FrameRegistrationError,
    build_registration_plan,
    coordinate_views,
)


def transitions(*, frames: int = 4, jump_frame: int | None = 1):
    rows = []
    for frame in range(frames - 1):
        flagged = frame == jump_frame
        rows.append(
            {
                "frame": frame,
                "pool_breaking": flagged,
                "shift_magnitude_um": 11.5 if flagged else 1.625,
                "shift_registered_zyx": [1, -2, 3] if flagged else [0, 1, 0],
            }
        )
    return rows


def test_cumulative_plan_uses_only_pool_breaking_step():
    plan = build_registration_plan(
        identity="train:fixture",
        transitions=transitions(),
        census_sha256="a" * 64,
        frame_count=4,
    )
    np.testing.assert_array_equal(
        plan.cumulative_shift_raw_zyx,
        np.asarray([[0, 0, 0], [0, 0, 0], [1, -8, 12], [1, -8, 12]]),
    )
    assert plan.pool_breaking_frames == (1,)


def test_coordinate_views_preserve_original_and_are_order_invariant():
    plan = build_registration_plan(
        identity="train:fixture",
        transitions=transitions(),
        census_sha256="b" * 64,
        frame_count=4,
    )
    times = np.asarray([3, 0, 2, 1])
    coords = np.asarray([[2, 4, 6], [1, 1, 1], [5, 6, 7], [8, 9, 10]], np.float32)
    views = coordinate_views(plan, times, coords, nonnegative_candidate_storage=True)
    np.testing.assert_array_equal(views.original_zyx, coords)
    order = np.asarray([1, 3, 0, 2])
    permuted = coordinate_views(
        plan, times[order], coords[order], nonnegative_candidate_storage=True
    )
    inverse = np.argsort(order)
    np.testing.assert_array_equal(permuted.candidate_zyx[inverse], views.candidate_zyx)
    np.testing.assert_array_equal(coords, views.original_zyx)
    assert views.receipt["coordinate_contract"]["applications_per_video"] == 1


def test_no_jump_is_exact_identity_and_double_application_fails():
    plan = build_registration_plan(
        identity="train:fixture",
        transitions=transitions(jump_frame=None),
        census_sha256="c" * 64,
        frame_count=4,
    )
    coords = np.asarray([[1, 2, 3], [4, 5, 6]], np.float32)
    views = coordinate_views(plan, np.asarray([0, 3]), coords)
    np.testing.assert_array_equal(views.original_zyx, views.candidate_zyx)
    with pytest.raises(FrameRegistrationError, match="twice"):
        coordinate_views(plan, np.asarray([0, 3]), views)  # type: ignore[arg-type]


def test_declared_threshold_drift_fails_closed():
    rows = transitions()
    rows[1]["shift_magnitude_um"] = 9.0
    with pytest.raises(FrameRegistrationError, match=">10-um"):
        build_registration_plan(
            identity="train:fixture",
            transitions=rows,
            census_sha256="d" * 64,
            frame_count=4,
        )
