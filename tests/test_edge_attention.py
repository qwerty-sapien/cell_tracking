from __future__ import annotations

import numpy as np
import pytest
import torch

from celltrack_alpha.edge_attention import (
    EdgeFrame,
    EdgeResidualModel,
    attention_neighbours,
    supervised_source_loss,
)


def test_attention_neighbours_enforces_shared_nodes_and_midpoint_radius() -> None:
    source = np.asarray([0, 0, 1, 2], dtype=np.int64)
    target = np.asarray([0, 1, 2, 3], dtype=np.int64)
    geometry = np.asarray(
        [
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 2, 0, 0, 4],
            [100, 0, 0, 100, 0, 2],
        ],
        dtype=np.float32,
    )
    indices, mask = attention_neighbours(source, target, geometry, radius_um=3.2)
    rows = [set(indices[row, mask[row]].tolist()) for row in range(len(source))]
    assert rows[0] == {0, 1, 2}
    assert rows[1] == {0, 1, 2}
    assert rows[2] == {0, 1, 2}
    assert rows[3] == {3}


def test_attention_model_is_identity_at_initialization_and_backpropagates() -> None:
    torch.manual_seed(7)
    features = torch.randn(4, 16)
    geometry = torch.randn(4, 6)
    indices = torch.arange(4).repeat(4, 1)
    mask = torch.ones(4, 4, dtype=torch.bool)
    for enabled in (False, True):
        model = EdgeResidualModel(attention=enabled, hidden=16, heads=4)
        residual = model(features, geometry, indices, mask)
        assert torch.equal(residual, torch.zeros_like(residual))
        residual.sum().backward()
        assert model.output.weight.grad is not None


def test_supervised_source_loss_rewards_true_candidate() -> None:
    source = torch.tensor([0, 0, 1], dtype=torch.long)
    supervised = torch.tensor([0], dtype=torch.long)
    positive = torch.tensor([1], dtype=torch.long)
    poor = supervised_source_loss(
        torch.tensor([2.0, -2.0, 0.0]), source, supervised, positive, dummy_logit=0.0
    )
    good = supervised_source_loss(
        torch.tensor([-2.0, 2.0, 0.0]), source, supervised, positive, dummy_logit=0.0
    )
    assert good < poor


def test_edge_frame_rejects_positive_from_another_source() -> None:
    source = np.asarray([0, 1], dtype=np.int64)
    target = np.asarray([0, 1], dtype=np.int64)
    geometry = np.zeros((2, 6), dtype=np.float32)
    neighbours, mask = attention_neighbours(source, target, geometry)
    with pytest.raises(ValueError, match="does not belong"):
        EdgeFrame(
            source_frame=0,
            target_frame=1,
            source_ids=np.asarray([10, 20]),
            target_ids=np.asarray([30, 40]),
            source_index=source,
            target_index=target,
            features=np.zeros((2, 16), dtype=np.float32),
            base_logits=np.zeros(2, dtype=np.float32),
            geometry_um=geometry,
            neighbour_index=neighbours,
            neighbour_mask=mask,
            supervised_sources=np.asarray([0]),
            positive_rows=np.asarray([1]),
        )
