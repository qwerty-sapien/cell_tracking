"""One-layer edge-self-attention over the frozen sparse pairwise scorer.

The module deliberately operates on candidate edges, not detector nodes.  Its
attention graph is restricted to candidate edges that share an endpoint or
whose physical midpoints are at most 15 um apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


FEATURE_COUNT = 16


class EdgeAttentionError(ValueError):
    """Raised when an edge-attention frame or model contract is invalid."""


@dataclass(frozen=True, slots=True)
class EdgeFrame:
    """Frozen candidate rows and label-safe source supervision for one pair."""

    source_frame: int
    target_frame: int
    source_ids: np.ndarray
    target_ids: np.ndarray
    source_index: np.ndarray
    target_index: np.ndarray
    features: np.ndarray
    base_logits: np.ndarray
    geometry_um: np.ndarray
    neighbour_index: np.ndarray
    neighbour_mask: np.ndarray
    supervised_sources: np.ndarray
    positive_rows: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "source_ids": (self.source_ids, np.int64),
            "target_ids": (self.target_ids, np.int64),
            "source_index": (self.source_index, np.int64),
            "target_index": (self.target_index, np.int64),
            "features": (self.features, np.float32),
            "base_logits": (self.base_logits, np.float32),
            "geometry_um": (self.geometry_um, np.float32),
            "neighbour_index": (self.neighbour_index, np.int64),
            "neighbour_mask": (self.neighbour_mask, bool),
            "supervised_sources": (self.supervised_sources, np.int64),
            "positive_rows": (self.positive_rows, np.int64),
        }
        normalized = {
            name: np.ascontiguousarray(np.asarray(value, dtype=dtype))
            for name, (value, dtype) in arrays.items()
        }
        edge_count = len(normalized["source_ids"])
        if self.target_frame != self.source_frame + 1 or self.source_frame < 0:
            raise EdgeAttentionError("edge frame must describe adjacent forward frames")
        if any(normalized[name].shape != (edge_count,) for name in (
            "target_ids", "source_index", "target_index", "base_logits"
        )):
            raise EdgeAttentionError("candidate edge vectors have inconsistent lengths")
        if normalized["features"].shape != (edge_count, FEATURE_COUNT):
            raise EdgeAttentionError("candidate features must have shape (E, 16)")
        if normalized["geometry_um"].shape != (edge_count, 6):
            raise EdgeAttentionError("edge geometry must contain source and target xyz rows")
        neighbour_index = normalized["neighbour_index"]
        neighbour_mask = normalized["neighbour_mask"]
        if neighbour_index.ndim != 2 or neighbour_index.shape != neighbour_mask.shape:
            raise EdgeAttentionError("attention neighbour tensors must have equal 2-D shapes")
        if neighbour_index.shape[0] != edge_count:
            raise EdgeAttentionError("attention rows must cover every candidate edge")
        if edge_count and (
            np.any(neighbour_index[neighbour_mask] < 0)
            or np.any(neighbour_index[neighbour_mask] >= edge_count)
            or np.any(neighbour_mask.sum(axis=1) < 1)
        ):
            raise EdgeAttentionError("attention neighbourhood contains invalid candidate rows")
        supervised = normalized["supervised_sources"]
        positives = normalized["positive_rows"]
        if supervised.ndim != 1 or positives.shape != supervised.shape:
            raise EdgeAttentionError("supervised source and positive row vectors disagree")
        if len(supervised) and (
            np.any(positives < 0)
            or np.any(positives >= edge_count)
            or np.any(normalized["source_index"][positives] != supervised)
        ):
            raise EdgeAttentionError("a positive edge does not belong to its supervised source")
        if len(set(int(value) for value in supervised)) != len(supervised):
            raise EdgeAttentionError("a source may be supervised at most once per frame pair")
        if not np.isfinite(normalized["features"]).all() or not np.isfinite(
            normalized["base_logits"]
        ).all() or not np.isfinite(normalized["geometry_um"]).all():
            raise EdgeAttentionError("edge frame contains nonfinite values")
        for name, value in normalized.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def edge_count(self) -> int:
        return len(self.source_ids)

    @property
    def attention_cell_count(self) -> int:
        return int(self.neighbour_mask.sum())


def attention_neighbours(
    source_index: np.ndarray,
    target_index: np.ndarray,
    geometry_um: np.ndarray,
    *,
    radius_um: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded edge neighbourhoods without exceeding an E-squared scan."""

    source_index = np.asarray(source_index, dtype=np.int64)
    target_index = np.asarray(target_index, dtype=np.int64)
    geometry = np.asarray(geometry_um, dtype=np.float64)
    count = len(source_index)
    if (
        target_index.shape != (count,)
        or geometry.shape != (count, 6)
        or not math.isfinite(radius_um)
        or radius_um <= 0.0
        or not np.isfinite(geometry).all()
    ):
        raise EdgeAttentionError("invalid edge geometry for attention neighbourhood")
    if count == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=bool)
    midpoint = 0.5 * (geometry[:, :3] + geometry[:, 3:])
    close = np.sum(np.square(midpoint[:, None, :] - midpoint[None, :, :]), axis=2) <= radius_um**2
    permitted = (
        close
        | (source_index[:, None] == source_index[None, :])
        | (target_index[:, None] == target_index[None, :])
    )
    widths = permitted.sum(axis=1)
    width = int(widths.max())
    indices = np.zeros((count, width), dtype=np.int64)
    mask = np.zeros((count, width), dtype=bool)
    for row in range(count):
        local = np.flatnonzero(permitted[row])
        indices[row, : len(local)] = local
        mask[row, : len(local)] = True
    return indices, mask


class EdgeResidualModel(nn.Module):
    """Pairwise residual with an optional single edge-self-attention layer."""

    def __init__(self, *, attention: bool, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        if type(attention) is not bool or hidden <= 0 or heads <= 0 or hidden % heads:
            raise EdgeAttentionError("invalid edge residual architecture")
        self.attention_enabled = attention
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.input_norm = nn.LayerNorm(FEATURE_COUNT)
        self.input = nn.Linear(FEATURE_COUNT, hidden)
        if attention:
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
            self.value = nn.Linear(hidden, hidden, bias=False)
            self.geometry_bias = nn.Sequential(
                nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, heads)
            )
            self.attention_output = nn.Linear(hidden, hidden, bias=False)
            self.attention_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        geometry_um: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
            raise EdgeAttentionError("model features must have shape (E, 16)")
        edge_count = int(features.shape[0])
        if geometry_um.shape != (edge_count, 6):
            raise EdgeAttentionError("model geometry must have shape (E, 6)")
        hidden = torch.nn.functional.gelu(self.input(self.input_norm(features)))
        if self.attention_enabled and edge_count:
            if (
                neighbour_index.ndim != 2
                or neighbour_index.shape != neighbour_mask.shape
                or neighbour_index.shape[0] != edge_count
            ):
                raise EdgeAttentionError("model attention neighbourhood shape is invalid")
            width = int(neighbour_index.shape[1])
            head_width = self.hidden // self.heads
            query = self.query(hidden).view(edge_count, self.heads, head_width)
            key = self.key(hidden).view(edge_count, self.heads, head_width)
            value = self.value(hidden).view(edge_count, self.heads, head_width)
            local_key = key[neighbour_index]
            local_value = value[neighbour_index]
            relative = (geometry_um[:, None, :] - geometry_um[neighbour_index]) / 15.0
            scores = (query[:, None] * local_key).sum(dim=-1) / math.sqrt(head_width)
            scores = scores + self.geometry_bias(relative)
            scores = scores.masked_fill(~neighbour_mask[..., None], -torch.inf)
            weights = torch.softmax(scores, dim=1)
            attended = (weights[..., None] * local_value).sum(dim=1).reshape(edge_count, self.hidden)
            hidden = self.attention_norm(hidden + self.attention_output(attended))
        residual = self.output(hidden).squeeze(-1)
        if residual.shape != (edge_count,) or not bool(torch.isfinite(residual).all().item()):
            raise EdgeAttentionError("edge residual logits are invalid")
        return residual

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def supervised_source_loss(
    logits: torch.Tensor,
    source_index: torch.Tensor,
    supervised_sources: torch.Tensor,
    positive_rows: torch.Tensor,
    *,
    dummy_logit: float,
) -> torch.Tensor:
    """Source-local cross entropy with the frozen explicit no-link class."""

    losses: list[torch.Tensor] = []
    dummy = logits.new_tensor(float(dummy_logit))
    for source, positive in zip(supervised_sources, positive_rows, strict=True):
        rows = torch.nonzero(source_index == source, as_tuple=False).flatten()
        local = torch.cat((logits[rows], dummy[None]))
        target = torch.nonzero(rows == positive, as_tuple=False).flatten()
        if len(target) != 1:
            raise EdgeAttentionError("supervised positive is absent from its source candidates")
        losses.append(torch.nn.functional.cross_entropy(local[None], target[:1]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


__all__ = [
    "EdgeAttentionError",
    "EdgeFrame",
    "EdgeResidualModel",
    "FEATURE_COUNT",
    "attention_neighbours",
    "supervised_source_loss",
]
