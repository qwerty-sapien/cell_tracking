"""Compact calibrated Siamese 3-D crop linker for fit-only association tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CropLinkerError(ValueError):
    """The crop, model, or calibration contract was violated."""


def normalize_crops(crops: torch.Tensor) -> torch.Tensor:
    """Normalize every crop independently to zero mean and unit L2 norm."""
    if crops.ndim != 5 or crops.shape[1] != 1:
        raise CropLinkerError("crops must have shape [batch, 1, z, y, x]")
    flat = crops.flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    safe = torch.where(norm > 1.0e-6, norm, torch.ones_like(norm))
    normalized = centered / safe
    normalized = torch.where(
        (norm > 1.0e-6).expand_as(normalized),
        normalized,
        torch.zeros_like(normalized),
    )
    return normalized.reshape_as(crops)


class CropEncoder(nn.Module):
    def __init__(self, embedding_width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.project = nn.Linear(32, embedding_width)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        hidden = self.body(normalize_crops(crops)).flatten(1)
        return F.normalize(self.project(hidden), dim=1)


class SiameseCropLinker(nn.Module):
    """Shared crop encoder plus directional context-aware link/no-link head."""

    def __init__(
        self,
        *,
        context_features: int = 48,
        embedding_width: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.context_features = int(context_features)
        self.encoder = CropEncoder(embedding_width)
        pair_width = 4 * embedding_width + context_features
        self.head = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source.shape != target.shape
            or source.ndim != 5
            or context.shape
            != (source.shape[0], self.context_features)
        ):
            raise CropLinkerError("crop-linker inputs disagree")
        source_embedding = self.encoder(source)
        target_embedding = self.encoder(target)
        pair = torch.cat(
            (
                source_embedding,
                target_embedding,
                torch.abs(source_embedding - target_embedding),
                source_embedding * target_embedding,
                context,
            ),
            dim=1,
        )
        return self.head(pair).squeeze(1)


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, float]:
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or torch.any((labels != 0) & (labels != 1))
        or bins <= 0
    ):
        raise CropLinkerError("calibration arrays are malformed")
    probabilities = torch.sigmoid(logits)
    labels_float = labels.to(dtype=logits.dtype)
    nll = F.binary_cross_entropy_with_logits(logits, labels_float)
    brier = torch.mean((probabilities - labels_float) ** 2)
    ece = torch.zeros((), dtype=logits.dtype, device=logits.device)
    boundaries = torch.linspace(
        0.0, 1.0, bins + 1, device=logits.device, dtype=logits.dtype
    )
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (
            (probabilities >= lower)
            & (
                probabilities <= upper
                if index == bins - 1
                else probabilities < upper
            )
        )
        if torch.any(mask):
            confidence = probabilities[mask].mean()
            accuracy = labels_float[mask].mean()
            ece = ece + mask.to(logits.dtype).mean() * torch.abs(
                confidence - accuracy
            )
    return {
        "negative_log_likelihood": float(nll.item()),
        "brier_score": float(brier.item()),
        "expected_calibration_error": float(ece.item()),
    }


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    before: dict[str, float]
    after: dict[str, float]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            raise CropLinkerError("temperature must be positive")
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 200,
    learning_rate: float = 0.05,
) -> TemperatureCalibration:
    """Fit one positive scalar on a disjoint calibration partition."""
    if (
        logits.ndim != 1
        or labels.shape != logits.shape
        or len(logits) < 2
        or not torch.any(labels == 0)
        or not torch.any(labels == 1)
    ):
        raise CropLinkerError(
            "temperature calibration needs both classes"
        )
    detached_logits = logits.detach()
    detached_labels = labels.detach().to(dtype=logits.dtype)
    log_temperature = torch.zeros(
        (), dtype=logits.dtype, device=logits.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            detached_logits / temperature, detached_labels
        )
        loss.backward()
        optimizer.step()
    temperature = float(
        torch.exp(log_temperature).clamp(0.05, 20.0).detach().item()
    )
    before = calibration_metrics(detached_logits, labels)
    after = calibration_metrics(detached_logits / temperature, labels)
    return TemperatureCalibration(
        temperature=temperature, before=before, after=after
    )


__all__ = [
    "CropEncoder",
    "CropLinkerError",
    "SiameseCropLinker",
    "TemperatureCalibration",
    "calibration_metrics",
    "fit_temperature",
    "normalize_crops",
]
