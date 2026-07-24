from __future__ import annotations

import numpy as np

from scripts.build_trackastra_full_pool_feature_cache import flatten_windows
from celltrack_alpha.point_context_swap import RuntimeProposalWindow


def _window() -> RuntimeProposalWindow:
    rows = (
        np.vstack(
            (
                np.arange(48, dtype=np.float32),
                np.arange(48, dtype=np.float32) + 1,
            )
        ),
        np.arange(48, dtype=np.float32)[None],
    )
    return RuntimeProposalWindow(
        identity="train:6bba_fixture",
        family=1,
        target_t=2,
        features=np.stack((rows[0][0], rows[1][0])),
        target_ids=np.asarray((20, 21), dtype=np.int64),
        challengers=((10, 20), (11, 21)),
        conflicts=(((10, 22), (12, 20)), ((11, 23),)),
        row_features=rows,
        geometry_confidences=np.asarray((0.8, 0.9), dtype=np.float32),
        geometry_probabilities=(
            np.asarray((0.8, 0.7), dtype=np.float32),
            np.asarray((0.9,), dtype=np.float32),
        ),
    )


def test_flatten_windows_preserves_lossless_row_identities() -> None:
    result = flatten_windows(
        (_window(),),
        (
            (
                np.asarray((0.25, -0.5), dtype=np.float32),
                np.asarray((0.75,), dtype=np.float32),
            ),
        ),
    )

    assert result["challengers"].tolist() == [[10, 20], [11, 21]]
    assert result["conflicts"].tolist() == [
        [10, 22],
        [12, 20],
        [11, 23],
    ]
    assert result["proposal_offsets"].tolist() == [0, 2, 3]
    assert result["transition_offsets"].tolist() == [0, 2]
    assert result["transition_target_t"].tolist() == [2]
    assert result["row_features"].shape == (3, 49)
    assert result["row_features"][:, -1].tolist() == [0.25, -0.5, 0.75]
