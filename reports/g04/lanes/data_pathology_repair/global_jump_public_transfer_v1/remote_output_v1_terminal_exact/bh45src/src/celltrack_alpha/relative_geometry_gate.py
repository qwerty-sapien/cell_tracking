"""Competitor-relative, abstaining swap gate for sparse association candidates.

The gate is deliberately local.  It compares each challenger with the current
source-local winner and may nominate at most one replacement.  It does not
rescore every edge, change candidate topology, or create a link when the frozen
no-link class is preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


EDGE_FEATURE_COUNT = 16
GEOMETRY_COUNT = 6
RELATIVE_FEATURE_COUNT = 44


class RelativeGeometryGateError(ValueError):
    """The relative-geometry gate contract was violated."""


def relative_features(
    challenger_features: np.ndarray,
    winner_features: np.ndarray,
    challenger_geometry_um: np.ndarray,
    winner_geometry_um: np.ndarray,
    *,
    challenger_logit: np.ndarray,
    winner_logit: np.ndarray,
    dummy_logit: float,
    candidate_count: np.ndarray,
    winner_is_dummy: np.ndarray | None = None,
) -> np.ndarray:
    """Build explicit challenger-minus-winner geometry and confidence features."""

    challenger = np.asarray(challenger_features, dtype=np.float32)
    winner = np.asarray(winner_features, dtype=np.float32)
    challenger_geometry = np.asarray(challenger_geometry_um, dtype=np.float32)
    winner_geometry = np.asarray(winner_geometry_um, dtype=np.float32)
    challenger_score = np.asarray(challenger_logit, dtype=np.float32)
    winner_score = np.asarray(winner_logit, dtype=np.float32)
    counts = np.asarray(candidate_count, dtype=np.float32)
    count = len(challenger)
    dummy_winner = (
        np.zeros(count, dtype=np.float32)
        if winner_is_dummy is None
        else np.asarray(winner_is_dummy, dtype=np.float32)
    )
    if (
        challenger.shape != (count, EDGE_FEATURE_COUNT)
        or winner.shape != challenger.shape
        or challenger_geometry.shape != (count, GEOMETRY_COUNT)
        or winner_geometry.shape != challenger_geometry.shape
        or challenger_score.shape != (count,)
        or winner_score.shape != (count,)
        or counts.shape != (count,)
        or dummy_winner.shape != (count,)
        or np.any(~np.isin(dummy_winner, (0.0, 1.0)))
        or not math.isfinite(dummy_logit)
        or np.any(counts < 2)
    ):
        raise RelativeGeometryGateError("relative gate inputs have invalid shapes or values")
    arrays = (
        challenger,
        winner,
        challenger_geometry,
        winner_geometry,
        challenger_score,
        winner_score,
        counts,
        dummy_winner,
    )
    if any(not np.isfinite(value).all() for value in arrays):
        raise RelativeGeometryGateError("relative gate inputs contain nonfinite values")

    challenger_displacement = challenger_geometry[:, 3:] - challenger_geometry[:, :3]
    winner_displacement = winner_geometry[:, 3:] - winner_geometry[:, :3]
    challenger_distance = np.linalg.norm(challenger_displacement, axis=1)
    winner_distance = np.linalg.norm(winner_displacement, axis=1)
    dot = np.sum(challenger_displacement * winner_displacement, axis=1)
    cosine = dot / np.maximum(challenger_distance * winner_distance, 1.0e-6)
    delta = challenger - winner
    values = np.column_stack(
        (
            delta,
            np.abs(delta),
            challenger_displacement - winner_displacement,
            challenger_distance,
            winner_distance,
            challenger_distance - winner_distance,
            cosine,
            challenger_score - winner_score,
            challenger_score - float(dummy_logit),
            winner_score - float(dummy_logit),
            np.log(counts),
            dummy_winner,
        )
    ).astype(np.float32, copy=False)
    if values.shape != (count, RELATIVE_FEATURE_COUNT) or not np.isfinite(values).all():
        raise RelativeGeometryGateError("relative feature construction failed")
    return values


@dataclass(frozen=True, slots=True)
class SwapDecision:
    """One source-local abstaining decision."""

    challenger_row: int | None
    probability: float


def choose_swap(
    candidate_rows: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> SwapDecision:
    """Choose at most one challenger, abstaining unless it clears threshold."""

    rows = np.asarray(candidate_rows, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        rows.ndim != 1
        or scores.shape != rows.shape
        or len(set(int(value) for value in rows)) != len(rows)
        or not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or not math.isfinite(threshold)
        or threshold < 0.0
        or threshold > 1.0
    ):
        raise RelativeGeometryGateError("swap decision inputs are invalid")
    if not len(rows):
        return SwapDecision(None, 0.0)
    best = int(np.argmax(scores))
    probability = float(scores[best])
    return SwapDecision(int(rows[best]) if probability >= threshold else None, probability)


class RelativeGeometryGate(nn.Module):
    """Small pairwise gate with an explicit probability output."""

    def __init__(self, *, hidden: int = 16) -> None:
        super().__init__()
        if type(hidden) is not int or hidden <= 0:
            raise RelativeGeometryGateError("hidden width must be a positive integer")
        self.hidden = hidden
        self.norm = nn.LayerNorm(RELATIVE_FEATURE_COUNT)
        self.input = nn.Linear(RELATIVE_FEATURE_COUNT, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != RELATIVE_FEATURE_COUNT:
            raise RelativeGeometryGateError("gate features have the wrong shape")
        logits = self.output(torch.nn.functional.gelu(self.input(self.norm(features)))).squeeze(-1)
        if logits.shape != (features.shape[0],) or not bool(torch.isfinite(logits).all().item()):
            raise RelativeGeometryGateError("gate logits are invalid")
        return logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "EDGE_FEATURE_COUNT",
    "GEOMETRY_COUNT",
    "RELATIVE_FEATURE_COUNT",
    "RelativeGeometryGate",
    "RelativeGeometryGateError",
    "SwapDecision",
    "choose_swap",
    "relative_features",
]
