from __future__ import annotations

import json
from pathlib import Path

import pytest

import audit_point_context_oof_threshold_headroom as audit


ROOT = Path(__file__).resolve().parents[1]


def test_locked_contract_loads() -> None:
    contract = audit.load_contract(
        audit.DEFAULT_CONTRACT, audit.DEFAULT_CONTRACT_SHA256
    )
    assert contract["scope"]["family"] == "6bba"
    assert contract["decision"]["promotion_authorized"] is False
    assert contract["decision"]["graph_replay_authorized"] is False
    assert contract["decision"]["cv57_authorized"] is False


def test_contract_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(audit.DEFAULT_CONTRACT.read_text(encoding="ascii"))
    payload["diagnostic"]["minimum_calibration_accepted"] = 24
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(audit.ThresholdHeadroomError, match="SHA-256"):
        audit.load_contract(path, audit.DEFAULT_CONTRACT_SHA256)


def _metrics(
    *,
    accepted: int,
    correct: int,
    utility: int,
    precision: float,
) -> dict[str, int | float]:
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": accepted - correct,
        "conservative_utility": utility,
        "background_accepted": 0,
        "total_selected": accepted,
        "precision": precision,
    }


def test_plausibility_gate_requires_every_fold() -> None:
    gate = {
        "minimum_aggregate_held_precision": 0.95,
        "minimum_each_fold_held_precision": 0.93,
        "minimum_aggregate_held_accepted": 1000,
        "minimum_conservative_utility": 1500,
    }
    aggregate = _metrics(
        accepted=1800, correct=1720, utility=1560, precision=0.9556
    )
    folds = [
        _metrics(
            accepted=360, correct=345, utility=330, precision=0.9583
        )
        for _ in range(5)
    ]
    assert audit.calibration_floor_passes(aggregate, folds, gate)
    folds[-1] = _metrics(
        accepted=360, correct=320, utility=280, precision=0.8889
    )
    assert not audit.calibration_floor_passes(aggregate, folds, gate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted_labeled", 999),
        ("conservative_utility", 1499),
        ("precision", 0.949999),
    ],
)
def test_plausibility_gate_enforces_aggregate_bounds(
    field: str, value: int | float
) -> None:
    gate = {
        "minimum_aggregate_held_precision": 0.95,
        "minimum_each_fold_held_precision": 0.93,
        "minimum_aggregate_held_accepted": 1000,
        "minimum_conservative_utility": 1500,
    }
    aggregate = _metrics(
        accepted=1800, correct=1720, utility=1560, precision=0.9556
    )
    aggregate[field] = value
    folds = [
        _metrics(
            accepted=360, correct=345, utility=330, precision=0.9583
        )
        for _ in range(5)
    ]
    assert not audit.calibration_floor_passes(aggregate, folds, gate)
