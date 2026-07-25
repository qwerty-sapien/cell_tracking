from __future__ import annotations

import numpy as np
import pytest

from scripts import audit_extended_radius_reachability as reach


def test_deterministic_rank_breaks_distance_ties_by_target_id() -> None:
    source = np.asarray((0.0, 0.0, 0.0))
    targets = np.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 2.0, 0.0),
        )
    )
    ids = np.asarray((20, 10, 30))
    distance, rank = reach.deterministic_rank(source, targets, ids, 0)
    assert distance == 1.0
    assert rank == 2


@pytest.mark.parametrize(
    ("distance", "rank", "radius", "cap", "expected"),
    (
        (10.0, 8, 10.0, 8, True),
        (10.01, 8, 10.0, 8, False),
        (9.0, 9, 10.0, 8, False),
        (12.0, 16, 12.0, 16, True),
    ),
)
def test_policy_reachability(
    distance: float,
    rank: int,
    radius: float,
    cap: int,
    expected: bool,
) -> None:
    assert reach.reachable(distance, rank, radius, cap) is expected
