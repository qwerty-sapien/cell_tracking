from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import fit_crop_linker_6bba_oof as oof
from scripts import fit_crop_linker_pilot as pilot


def _inputs() -> tuple[object, dict, dict, tuple[str, ...]]:
    args = oof.parser().parse_args([])
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    dataset = json.loads(args.dataset_report.read_text(encoding="ascii"))
    eligible = oof.eligible_identities(dataset)
    return args, contract, dataset, eligible


def test_eligible_scope_excludes_44b6_and_old_validation() -> None:
    _, _, dataset, eligible = _inputs()
    validation = set(dataset["scope"]["partitions"]["validation"])
    assert len(eligible) == 69
    assert not set(eligible) & validation
    assert all(identity.startswith("train:6bba_") for identity in eligible)


def test_nested_roles_cover_every_video_once_as_held() -> None:
    _, _, _, eligible = _inputs()
    roles = [oof.fold_roles(eligible, outer) for outer in range(5)]
    assert [len(role.held_ids) for role in roles] == [11, 17, 17, 11, 13]
    held = [identity for role in roles for identity in role.held_ids]
    assert sorted(held) == sorted(eligible)
    assert len(held) == len(set(held))
    for role in roles:
        assert role.calibration == (role.outer + 1) % 5


def test_frozen_oof_contract_accepts_pinned_inputs() -> None:
    args, contract, dataset, eligible = _inputs()
    oof.validate_contract(args, contract, dataset, eligible)


def test_frozen_oof_contract_rejects_runtime_drift() -> None:
    args, contract, dataset, eligible = _inputs()
    args.steps_per_fold -= 1
    with pytest.raises(oof.CropOofError, match="runtime"):
        oof.validate_contract(args, contract, dataset, eligible)


def test_subset_examples_remaps_video_indices_without_scope_drift() -> None:
    examples = pilot.Examples(
        source=np.zeros((5, 16, 32, 32), dtype=np.float16),
        target=np.zeros((5, 16, 32, 32), dtype=np.float16),
        context=np.zeros((5, 48), dtype=np.float32),
        labels=np.asarray([1, 0, 1, 0, 1], dtype=np.int8),
        video_index=np.asarray([0, 0, 1, 2, 2], dtype=np.int32),
        identities=(
            "train:6bba_alpha",
            "train:6bba_beta",
            "train:6bba_gamma",
        ),
        families=("6bba", "6bba", "6bba"),
    )
    subset = oof.subset_examples(
        examples, ("train:6bba_gamma", "train:6bba_alpha")
    )
    assert subset.identities == (
        "train:6bba_gamma",
        "train:6bba_alpha",
    )
    assert subset.video_index.tolist() == [1, 1, 0, 0]
    assert subset.labels.tolist() == [1, 0, 0, 1]


def test_sum_metrics_uses_conservative_utility() -> None:
    value = oof.sum_metrics(
        [
            {"correct_rescues": 10, "harmful_swaps": 1},
            {"correct_rescues": 5, "harmful_swaps": 2},
        ]
    )
    assert value["accepted_labeled"] == 18
    assert value["precision"] == 15 / 18
    assert value["conservative_utility"] == 9
