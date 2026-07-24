from __future__ import annotations

import numpy as np

from scripts.evaluate_trackastra_linear_oof_graph_gate import (
    DEFAULT_CONTRACT,
    DEFAULT_CONTRACT_SHA256,
    _atomic_fold,
    _restore_fold,
    load_contract,
)
from celltrack_alpha.trackastra_linear import LinearHead


def test_locked_contract_loads() -> None:
    contract = load_contract(DEFAULT_CONTRACT, DEFAULT_CONTRACT_SHA256)
    assert contract["scope"]["eligible_video_count"] == 69
    assert contract["candidate"]["linear_head"]["feature_count"] == 49


def test_fold_checkpoint_round_trip(tmp_path) -> None:
    head = LinearHead(
        mean=np.zeros(49),
        scale=np.ones(49),
        weights=np.arange(49, dtype=np.float64),
        intercept=0.25,
        ridge_lambda=0.01,
    )
    code = {"fixture.py": "a" * 64}
    record = _atomic_fold(
        root=tmp_path,
        held_fold=0,
        head=head,
        threshold=0.4,
        fit_ids=("train:6bba_fit",),
        calibration_ids=("train:6bba_cal",),
        held_ids=("train:6bba_held",),
        calibration_metrics={"precision": 0.96},
        held_metrics={"precision": 0.95},
        contract_sha256=DEFAULT_CONTRACT_SHA256,
        code_identity=code,
    )
    restored = _restore_fold(
        root=tmp_path,
        held_fold=0,
        fit_ids=("train:6bba_fit",),
        calibration_ids=("train:6bba_cal",),
        held_ids=("train:6bba_held",),
        contract_sha256=DEFAULT_CONTRACT_SHA256,
        code_identity=code,
    )
    assert restored is not None
    restored_head, threshold, restored_record = restored
    assert threshold == 0.4
    assert np.array_equal(restored_head.weights, head.weights)
    assert restored_record == record
