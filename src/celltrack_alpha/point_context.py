"""Compact proposal-set transformer with an explicit parental no-link option."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PointContextError(ValueError):
    """The proposal-window or parental-softmax contract was violated."""


@dataclass(frozen=True)
class ProposalWindow:
    identity: str
    family: int
    target_t: int
    features: np.ndarray
    labels: np.ndarray
    target_ids: np.ndarray

    def validate(self, feature_count: int = 48) -> None:
        count = len(self.labels)
        if (
            self.family not in (0, 1)
            or self.features.shape != (count, feature_count)
            or self.target_ids.shape != (count,)
            or count == 0
            or not np.isfinite(self.features).all()
            or np.any(~np.isin(self.labels, (-1, 0, 1)))
        ):
            raise PointContextError("proposal window arrays are malformed")
        for target in np.unique(self.target_ids):
            labels = self.labels[self.target_ids == target]
            if int(np.sum(labels == 1)) > 1:
                raise PointContextError(
                    "parental group contains multiple positive parents"
                )


class PointContextTransformer(nn.Module):
    """Set-equivariant transition context with per-target no-parent logits."""

    def __init__(
        self,
        *,
        feature_count: int = 48,
        width: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width % heads:
            raise PointContextError("width must be divisible by heads")
        self.feature_count = int(feature_count)
        self.width = int(width)
        self.input = nn.Linear(feature_count, width)
        self.family = nn.Embedding(2, width)
        self.input_norm = nn.LayerNorm(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.edge_head = nn.Linear(width, 1)
        self.no_parent_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def encode(
        self,
        features: torch.Tensor,
        families: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[-1] != self.feature_count
            or families.shape != (features.shape[0],)
            or padding_mask.shape != features.shape[:2]
        ):
            raise PointContextError("padded transformer inputs disagree")
        hidden = self.input(features)
        hidden = hidden + self.family(families)[:, None, :]
        hidden = self.input_norm(hidden)
        return self.encoder(hidden, src_key_padding_mask=padding_mask)

    def forward(
        self,
        features: torch.Tensor,
        families: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(features, families, padding_mask)
        return self.edge_head(hidden).squeeze(-1), hidden


def collate_windows(
    windows: Sequence[ProposalWindow],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not windows:
        raise PointContextError("cannot collate an empty window batch")
    for window in windows:
        window.validate(len(mean))
    maximum = max(len(window.labels) for window in windows)
    batch = len(windows)
    x = np.zeros((batch, maximum, len(mean)), dtype=np.float32)
    labels = np.full((batch, maximum), -2, dtype=np.int8)
    targets = np.full((batch, maximum), -1, dtype=np.int64)
    mask = np.ones((batch, maximum), dtype=bool)
    families = np.empty(batch, dtype=np.int64)
    for index, window in enumerate(windows):
        count = len(window.labels)
        x[index, :count] = (window.features - mean) / scale
        labels[index, :count] = window.labels
        targets[index, :count] = window.target_ids
        mask[index, :count] = False
        families[index] = window.family
    return {
        "features": torch.from_numpy(x).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "target_ids": torch.from_numpy(targets).to(device),
        "padding_mask": torch.from_numpy(mask).to(device),
        "families": torch.from_numpy(families).to(device),
    }


def parental_softmax_loss(
    model: PointContextTransformer,
    edge_logits: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    target_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    background_weight: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    if not 0.0 < background_weight <= 1.0:
        raise PointContextError("background weight must be in (0, 1]")
    losses = []
    weights = []
    positive_groups = labeled_negative_groups = background_groups = 0
    for batch_index in range(edge_logits.shape[0]):
        valid = ~padding_mask[batch_index]
        window_targets = target_ids[batch_index, valid]
        window_labels = labels[batch_index, valid]
        window_logits = edge_logits[batch_index, valid]
        window_hidden = hidden[batch_index, valid]
        for target in torch.unique(window_targets, sorted=True):
            group = window_targets == target
            group_labels = window_labels[group]
            positive = torch.nonzero(
                group_labels == 1, as_tuple=False
            ).flatten()
            if len(positive) > 1:
                raise PointContextError(
                    "parental group contains multiple positive parents"
                )
            no_parent = model.no_parent_head(
                window_hidden[group].mean(dim=0)
            ).reshape(1)
            logits = torch.cat((window_logits[group], no_parent), dim=0)
            if len(positive) == 1:
                target_index = int(positive.item())
                weight = 1.0
                positive_groups += 1
            else:
                target_index = len(logits) - 1
                if bool(torch.any(group_labels == 0)):
                    weight = 1.0
                    labeled_negative_groups += 1
                else:
                    weight = background_weight
                    background_groups += 1
            losses.append(
                F.cross_entropy(
                    logits.reshape(1, -1),
                    torch.tensor(
                        [target_index], device=logits.device, dtype=torch.long
                    ),
                )
            )
            weights.append(weight)
    if not losses:
        raise PointContextError("batch contains no parental groups")
    loss_values = torch.stack(losses)
    weight_values = torch.tensor(
        weights, device=loss_values.device, dtype=loss_values.dtype
    )
    loss = torch.sum(loss_values * weight_values) / torch.sum(weight_values)
    return loss, {
        "positive_groups": positive_groups,
        "labeled_negative_groups": labeled_negative_groups,
        "background_groups": background_groups,
    }


@torch.no_grad()
def parental_probabilities(
    model: PointContextTransformer,
    edge_logits: torch.Tensor,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    padding_mask: torch.Tensor,
) -> list[np.ndarray]:
    result = []
    for batch_index in range(edge_logits.shape[0]):
        valid = ~padding_mask[batch_index]
        window_targets = target_ids[batch_index, valid]
        window_logits = edge_logits[batch_index, valid]
        window_hidden = hidden[batch_index, valid]
        probabilities = torch.empty_like(window_logits)
        for target in torch.unique(window_targets, sorted=True):
            group = window_targets == target
            no_parent = model.no_parent_head(
                window_hidden[group].mean(dim=0)
            ).reshape(1)
            logits = torch.cat((window_logits[group], no_parent), dim=0)
            probabilities[group] = torch.softmax(logits, dim=0)[:-1]
        result.append(probabilities.detach().cpu().numpy())
    return result


__all__ = [
    "PointContextError",
    "PointContextTransformer",
    "ProposalWindow",
    "collate_windows",
    "parental_probabilities",
    "parental_softmax_loss",
]
