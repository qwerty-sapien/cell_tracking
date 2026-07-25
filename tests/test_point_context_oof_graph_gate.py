from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import evaluate_point_context_oof_graph_gate as gate


def test_nested_partitions_are_fit_calibration_held_disjoint(monkeypatch) -> None:
    folds = {
        "video-a": 0,
        "video-b": 1,
        "video-c": 2,
        "video-d": 3,
        "video-e": 4,
        "video-f": 0,
    }
    monkeypatch.setattr(gate, "_fold", lambda identity, count: folds[identity])
    partitions = gate.nested_partitions(tuple(folds), 5)
    held_once = []
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(
        partitions
    ):
        fit = set(fit_ids)
        calibration = set(calibration_ids)
        held = set(held_ids)
        assert not fit & calibration
        assert not fit & held
        assert not calibration & held
        assert fit | calibration | held == set(folds)
        assert {folds[value] for value in calibration} == {
            (held_fold + 1) % 5
        }
        assert {folds[value] for value in held} == {held_fold}
        held_once.extend(held_ids)
    assert sorted(held_once) == sorted(folds)


def test_aggregate_metrics_does_not_mix_fold_thresholds() -> None:
    result = gate.aggregate_metrics(
        [
            {
                "accepted_labeled": 10,
                "correct_rescues": 9,
                "harmful_swaps": 1,
                "background_accepted": 2,
                "conservative_utility": 7,
                "total_selected": 12,
            },
            {
                "accepted_labeled": 20,
                "correct_rescues": 20,
                "harmful_swaps": 0,
                "background_accepted": 3,
                "conservative_utility": 20,
                "total_selected": 23,
            },
        ]
    )
    assert result == {
        "accepted_labeled": 30,
        "correct_rescues": 29,
        "harmful_swaps": 1,
        "background_accepted": 5,
        "conservative_utility": 27,
        "total_selected": 35,
        "precision": 29 / 30,
    }


def test_contract_is_hash_pinned_and_fail_closed() -> None:
    value = gate.load_contract(
        gate.DEFAULT_CONTRACT, gate.DEFAULT_CONTRACT_SHA256
    )
    assert value["scope"]["validation12_access"] is False
    assert value["scope"]["cv57_access"] is False
    assert value["candidate"]["deployment_pool"][
        "sampled_background_used_for_inference"
    ] is False


def _fold_metrics(correct: int, harmful: int) -> dict[str, object]:
    accepted = correct + harmful
    return {
        "accepted_labeled": accepted,
        "correct_rescues": correct,
        "harmful_swaps": harmful,
        "background_accepted": 0,
        "conservative_utility": correct - 2 * harmful,
        "precision": correct / accepted,
        "total_selected": accepted,
    }


def test_fold_checkpoint_roundtrip_is_identity_bound(tmp_path) -> None:
    architecture = {
        "feature_count": 48,
        "width": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.1,
    }
    model = gate.PointContextTransformer(**architecture)
    code_identity = {"implementation.py": "a" * 64}
    calibration = _fold_metrics(20, 1)
    held = _fold_metrics(18, 1)
    record = gate.write_fold_checkpoint(
        model_dir=tmp_path,
        held_fold=4,
        model=model,
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        threshold=0.75,
        seed=99,
        fit_ids=("fit-a", "fit-b"),
        calibration_ids=("cal-a",),
        held_ids=("held-a",),
        architecture=architecture,
        fit_record={"steps": 5, "final_loss": 0.1},
        calibration_metrics=calibration,
        held_metrics=held,
        contract_sha256="b" * 64,
        code_identity=code_identity,
        fold_count=5,
    )
    restored = gate.load_fold_checkpoint(
        model_dir=tmp_path,
        held_fold=4,
        fit_ids=("fit-a", "fit-b"),
        calibration_ids=("cal-a",),
        held_ids=("held-a",),
        architecture=architecture,
        seed=99,
        contract_sha256="b" * 64,
        code_identity=code_identity,
        fold_count=5,
    )
    assert restored is not None
    restored_model, mean, scale, threshold, restored_record = restored
    assert record == restored_record
    assert restored_record["calibration_fold"] == 0
    assert threshold == 0.75
    np.testing.assert_array_equal(mean, np.zeros(48, dtype=np.float32))
    np.testing.assert_array_equal(scale, np.ones(48, dtype=np.float32))
    for name, value in model.state_dict().items():
        assert np.array_equal(
            value.detach().numpy(),
            restored_model.state_dict()[name].detach().numpy(),
        )


def test_fold_checkpoint_rejects_code_drift_and_partial_state(tmp_path) -> None:
    architecture = {
        "feature_count": 48,
        "width": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.1,
    }
    model = gate.PointContextTransformer(**architecture)
    arguments = {
        "model_dir": tmp_path,
        "held_fold": 0,
        "fit_ids": ("fit-a",),
        "calibration_ids": ("cal-a",),
        "held_ids": ("held-a",),
        "architecture": architecture,
        "seed": 7,
        "contract_sha256": "b" * 64,
        "code_identity": {"implementation.py": "a" * 64},
        "fold_count": 5,
    }
    gate.write_fold_checkpoint(
        **arguments,
        model=model,
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        threshold=0.5,
        fit_record={"steps": 5},
        calibration_metrics=_fold_metrics(10, 1),
        held_metrics=_fold_metrics(9, 1),
    )
    with pytest.raises(
        gate.PointContextOOFGraphGateError,
        match="checkpoint identity drifted",
    ):
        gate.load_fold_checkpoint(
            **{
                **arguments,
                "code_identity": {"implementation.py": "c" * 64},
            }
        )

    _artifact, checkpoint = gate._fold_paths(tmp_path, 0)
    checkpoint.unlink()
    with pytest.raises(
        gate.PointContextOOFGraphGateError, match="checkpoint is partial"
    ):
        gate.load_fold_checkpoint(**arguments)


def test_fold_checkpoint_rejects_evidence_tampering(tmp_path) -> None:
    architecture = {
        "feature_count": 48,
        "width": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.1,
    }
    arguments = {
        "model_dir": tmp_path,
        "held_fold": 1,
        "fit_ids": ("fit-a",),
        "calibration_ids": ("cal-a",),
        "held_ids": ("held-a",),
        "architecture": architecture,
        "seed": 8,
        "contract_sha256": "b" * 64,
        "code_identity": {"implementation.py": "a" * 64},
        "fold_count": 5,
    }
    gate.write_fold_checkpoint(
        **arguments,
        model=gate.PointContextTransformer(**architecture),
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        threshold=0.5,
        fit_record={"steps": 5},
        calibration_metrics=_fold_metrics(10, 1),
        held_metrics=_fold_metrics(9, 1),
    )
    _artifact, checkpoint = gate._fold_paths(tmp_path, 1)
    payload = json.loads(checkpoint.read_text(encoding="ascii"))
    payload["fold"]["held"]["correct_rescues"] = 0
    checkpoint.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="ascii"
    )
    with pytest.raises(
        gate.PointContextOOFGraphGateError, match="artifact identity drifted"
    ):
        gate.load_fold_checkpoint(**arguments)
