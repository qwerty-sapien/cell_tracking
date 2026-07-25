from __future__ import annotations

import numpy as np

from scripts import audit_crop_linker_data as audit
from scripts import prepare_crop_linker_dataset as prepare


def test_exact_pair_label_masks_unmatched_nodes() -> None:
    mapping = {10: 1, 20: 2, 30: 3}
    gt_edges = {(1, 2)}
    assert audit.exact_pair_label((10, 20), mapping, gt_edges) == "positive"
    assert (
        audit.exact_pair_label((10, 30), mapping, gt_edges)
        == "guaranteed_negative"
    )
    assert (
        audit.exact_pair_label((10, 40), mapping, gt_edges)
        == "uncertain_masked"
    )


def test_calibration_ids_are_deterministic_and_family_balanced() -> None:
    train_ids = [
        *(f"train:44b6_{index:02d}" for index in range(8)),
        *(f"train:6bba_{index:02d}" for index in range(12)),
    ]
    first = audit.calibration_ids(
        train_ids, count_44b6=2, count_6bba=3
    )
    second = audit.calibration_ids(
        list(reversed(train_ids)), count_44b6=2, count_6bba=3
    )
    assert first == second
    assert sum("44b6_" in identity for identity in first) == 2
    assert sum("6bba_" in identity for identity in first) == 3


def test_pair_table_deduplicates_background_and_preserves_roles() -> None:
    labeled = np.zeros((3, 48), dtype=np.float32)
    labeled[:, 44] = [0.8, 0.7, 0.6]
    background = np.zeros((2, 48), dtype=np.float32)
    background[:, 44] = [0.5, 0.4]
    payload = {
        "labeled_features": labeled,
        "labeled_labels": np.asarray([1, 1, 0], dtype=np.int8),
        "labeled_challenger_pairs": np.asarray(
            [[1, 3], [1, 3], [2, 4]], dtype=np.int64
        ),
        "background_features": background,
        "background_challenger_pairs": np.asarray(
            [[1, 3], [5, 6]], dtype=np.int64
        ),
        "background_groups": [[[0]], [[1]]],
    }
    pairs, features, roles = prepare._pair_table(payload)
    assert pairs.tolist() == [[1, 3], [2, 4], [5, 6]]
    assert features.shape == (3, 48)
    assert roles.tolist() == [1, 0, -1]


def test_normalized_patch_is_float16_zero_mean_unit_norm() -> None:
    frame = np.arange(16 * 32 * 32, dtype=np.float32).reshape(
        16, 32, 32
    )
    patch = prepare._normalized_patch(
        frame, np.asarray([8.0, 16.0, 16.0], dtype=np.float32)
    )
    assert patch.shape == (16, 32, 32)
    assert patch.dtype == np.float16
    assert abs(float(patch.astype(np.float32).mean())) < 1.0e-4
    assert np.isclose(
        np.linalg.norm(patch.astype(np.float32)), 1.0, atol=1.0e-3
    )
