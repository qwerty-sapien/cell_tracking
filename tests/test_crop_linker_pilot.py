from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from scripts import fit_crop_linker_pilot as pilot


def _inputs() -> tuple[object, dict, dict]:
    args = pilot.parser().parse_args([])
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    dataset = json.loads(args.dataset_report.read_text(encoding="ascii"))
    return args, contract, dataset


def test_frozen_pilot_contract_accepts_pinned_inputs() -> None:
    args, contract, dataset = _inputs()
    pilot.validate_pilot_contract(args, contract, dataset)


def test_frozen_pilot_contract_rejects_runtime_drift() -> None:
    args, contract, dataset = _inputs()
    args.max_steps -= 1
    with pytest.raises(pilot.CropPilotError, match="runtime knobs"):
        pilot.validate_pilot_contract(args, contract, dataset)


def test_frozen_pilot_contract_rejects_split_overlap() -> None:
    args, contract, dataset = _inputs()
    dataset["scope"]["partitions"]["validation"][0] = dataset["scope"][
        "partitions"
    ]["fit_train"][0]
    with pytest.raises(pilot.CropPilotError, match="contract"):
        pilot.validate_pilot_contract(args, contract, dataset)


def test_validation_examples_are_loaded_only_after_training_and_calibration() -> None:
    source = inspect.getsource(pilot.run)
    assert source.index('partition="validation"') > source.index(
        "fit_temperature("
    )
    assert source.index('partition="calibration"') > source.index(
        "for step in range("
    )


def test_score_records_and_family_metrics_preserve_video_boundaries() -> None:
    examples = pilot.Examples(
        source=np.zeros((4, 16, 32, 32), dtype=np.float16),
        target=np.zeros((4, 16, 32, 32), dtype=np.float16),
        context=np.zeros((4, 48), dtype=np.float32),
        labels=np.asarray([1, 0, 1, 0], dtype=np.int8),
        video_index=np.asarray([0, 0, 1, 1], dtype=np.int32),
        identities=("train:44b6_alpha", "train:6bba_beta"),
        families=("44b6", "6bba"),
    )
    scored = pilot.score_records(
        examples,
        np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64),
        selected_edges={"train:44b6_alpha": 100, "train:6bba_beta": 100},
    )
    assert [row["identity"] for row in scored] == [
        "train:44b6_alpha",
        "train:6bba_beta",
    ]
    assert [len(row["labels"]) for row in scored] == [2, 2]
    by_family = pilot.family_metrics(scored, 0.5)
    assert by_family["44b6"]["correct_rescues"] == 1
    assert by_family["6bba"]["correct_rescues"] == 1
