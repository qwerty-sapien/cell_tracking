from __future__ import annotations

import numpy as np
import torch

from celltrack_alpha.point_context import (
    PointContextTransformer,
    ProposalWindow,
    collate_windows,
    parental_probabilities,
    parental_softmax_loss,
)
from scripts import fit_compact_point_context_pilot as pilot
from scripts.fit_compact_point_context_pilot import proposal_features


def _window() -> ProposalWindow:
    return ProposalWindow(
        identity="train:44b6_unit",
        family=0,
        target_t=1,
        features=np.asarray(
            [
                np.linspace(0.0, 1.0, 48),
                np.linspace(1.0, 0.0, 48),
                np.full(48, 0.5),
            ],
            dtype=np.float32,
        ),
        labels=np.asarray([1, 0, -1], dtype=np.int8),
        target_ids=np.asarray([10, 10, 11], dtype=np.int64),
    )


def test_parental_transformer_loss_and_probabilities_are_well_formed() -> None:
    torch.manual_seed(7)
    model = PointContextTransformer(width=16, heads=4, layers=1, dropout=0.0)
    batch = collate_windows(
        [_window()],
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        device=torch.device("cpu"),
    )
    edge_logits, hidden = model(
        batch["features"], batch["families"], batch["padding_mask"]
    )
    loss, groups = parental_softmax_loss(
        model,
        edge_logits,
        hidden,
        batch["labels"],
        batch["target_ids"],
        batch["padding_mask"],
        background_weight=0.05,
    )
    assert torch.isfinite(loss)
    assert groups == {
        "positive_groups": 1,
        "labeled_negative_groups": 0,
        "background_groups": 1,
    }
    probabilities = parental_probabilities(
        model,
        edge_logits,
        hidden,
        batch["target_ids"],
        batch["padding_mask"],
    )
    assert len(probabilities) == 1
    assert probabilities[0].shape == (3,)
    assert np.all((probabilities[0] >= 0.0) & (probabilities[0] <= 1.0))


def test_collation_pads_variable_windows() -> None:
    first = _window()
    second = ProposalWindow(
        identity="train:6bba_unit",
        family=1,
        target_t=2,
        features=np.zeros((1, 48), dtype=np.float32),
        labels=np.asarray([0], dtype=np.int8),
        target_ids=np.asarray([12], dtype=np.int64),
    )
    batch = collate_windows(
        [first, second],
        mean=np.zeros(48, dtype=np.float32),
        scale=np.ones(48, dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert batch["features"].shape == (2, 3, 48)
    assert batch["padding_mask"].tolist() == [
        [False, False, False],
        [False, True, True],
    ]


def test_proposal_features_select_worst_best_option_row() -> None:
    rows = np.zeros((4, 48), dtype=np.float32)
    rows[:, 44] = [0.2, 0.8, 0.5, 0.7]
    rows[:, 0] = [2.0, 8.0, 5.0, 7.0]
    features = proposal_features(rows, [[[0, 1], [2, 3]]])
    # Best per conflict is 0.8 and 0.7; the latter is the limiting conflict.
    assert features.shape == (1, 48)
    assert features[0, 44] == np.float32(0.7)
    assert features[0, 0] == np.float32(7.0)


def test_point_context_runtime_contract_is_fail_closed() -> None:
    args = pilot.parser().parse_args([])
    pilot.validate_runtime_contract(args)
    args.max_steps -= 1
    try:
        pilot.validate_runtime_contract(args)
    except pilot.PointContextPilotError as exc:
        assert "runtime knobs" in str(exc)
    else:
        raise AssertionError("runtime drift was accepted")


def test_validation_payloads_are_loaded_only_after_training() -> None:
    import inspect

    source = inspect.getsource(pilot.run)
    validation_load = source.index(
        "allowed_identities=validation_ids"
    )
    assert validation_load > source.index(
        "while step < args.max_steps"
    )
    assert validation_load > source.index("train_scored = score_windows")
