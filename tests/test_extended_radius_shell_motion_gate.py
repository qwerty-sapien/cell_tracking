import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_extended_radius_shell_motion_gate",
    ROOT / "scripts/audit_extended_radius_shell_motion_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


NAMES = (
    "source_history_available",
    "source_acceleration_residual_um",
    "target_future_available",
    "target_continuation_residual_um",
    "pair_ncc",
    "conflict_ncc_available",
    "pair_minus_maximum_conflict_ncc",
)


def test_motion_residual_uses_worse_available_direction():
    x = np.asarray(
        [
            [1, 2.0, 0, 9.0, 0.4, 0, -1],
            [0, 9.0, 1, 3.0, 0.4, 0, -1],
            [1, 2.0, 1, 4.0, 0.4, 0, -1],
            [0, 9.0, 0, 9.0, 0.4, 0, -1],
        ],
        dtype=np.float32,
    )
    residual, available = MODULE.motion_residual(x, NAMES)
    np.testing.assert_allclose(residual[:3], [2.0, 3.0, 4.0])
    assert np.isinf(residual[3])
    assert available.tolist() == [True, True, True, False]


def test_ncc_confidence_uses_pair_or_conflict_margin():
    x = np.asarray(
        [
            [1, 1, 0, 1, 0.4, 0, -0.7],
            [1, 1, 0, 1, 0.8, 1, -0.1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        MODULE.ncc_confidence(x, NAMES), [0.4, -0.1]
    )


def _video(identity, labels, residual, ncc, selected_edges=1000):
    return {
        "identity": identity,
        "selected_edges": selected_edges,
        "labels": np.asarray(labels, dtype=np.int8),
        "residual": np.asarray(residual, dtype=np.float64),
        "available": np.ones(len(labels), dtype=bool),
        "ncc": np.asarray(ncc, dtype=np.float64),
    }


def test_evaluate_reconciles_recall_precision_reduction_and_cap():
    videos = [
        _video(
            "train:44b6_a",
            [1, 1, 0, -1],
            [1, 3, 1, 1],
            [0.5, 0.5, 0.5, -1.0],
        ),
        _video(
            "train:6bba_b",
            [1, 0, -1],
            [1, 4, 1],
            [0.5, 0.5, -1.0],
        ),
    ]
    result = MODULE.evaluate(videos, tolerance_um=2, ncc_threshold=0)
    assert result["pre_cap"]["positive_rows"] == 3
    assert result["pre_cap"]["eligible_positive_rows"] == 2
    assert result["pre_cap"]["shell_positive_recall"] == pytest.approx(2 / 3)
    assert result["pre_cap"]["eligible_rows"] == 3
    assert result["pre_cap"]["candidate_reduction_fraction"] == pytest.approx(
        4 / 7
    )
    assert result["capped"]["correct_rescues"] == 2
    assert result["capped"]["harmful_swaps"] == 1
    assert result["capped"]["conservative_utility"] == 0


def test_validate_inputs_rejects_unpinned_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    with pytest.raises(MODULE.MotionShellError, match="exact fit-only"):
        MODULE.validate_inputs(
            {},
            {},
            input_path=source,
            contract_path=contract,
        )


def test_select_tolerance_is_train_only_and_deterministic():
    base = {
        "families": {
            "44b6": {"accepted_labeled": 20, "precision": 1.0},
            "6bba": {"accepted_labeled": 20, "precision": 1.0},
        },
        "pre_cap": {},
    }
    rows = [
        {
            **base,
            "tolerance_um": 1.0,
            "capped": {
                "correct_rescues": 30,
                "harmful_swaps": 0,
                "background_accepted": 0,
                "precision": 1.0,
                "conservative_utility": 30,
            },
        },
        {
            **base,
            "tolerance_um": 2.0,
            "capped": {
                "correct_rescues": 31,
                "harmful_swaps": 1,
                "background_accepted": 0,
                "precision": 31 / 32,
                "conservative_utility": 29,
            },
        },
    ]
    selected = MODULE.select_tolerance(rows)
    assert selected["admissible"] is True
    assert selected["selected"]["tolerance_um"] == 1.0


def test_select_tolerance_uses_supported_utility_for_park_diagnostic():
    base = {
        "families": {
            "44b6": {"accepted_labeled": 1, "precision": 0.5},
            "6bba": {"accepted_labeled": 1, "precision": 0.5},
        },
        "pre_cap": {},
    }
    rows = [
        {
            **base,
            "tolerance_um": 0.5,
            "capped": {
                "correct_rescues": 0,
                "harmful_swaps": 0,
                "background_accepted": 0,
                "precision": 0.0,
                "conservative_utility": 0,
            },
        },
        {
            **base,
            "tolerance_um": 8.0,
            "capped": {
                "correct_rescues": 14,
                "harmful_swaps": 2,
                "background_accepted": 854,
                "precision": 0.875,
                "conservative_utility": 10,
            },
        },
    ]
    selected = MODULE.select_tolerance(rows)
    assert selected["admissible"] is False
    assert selected["selected"]["tolerance_um"] == 8.0
