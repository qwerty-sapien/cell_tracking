"""Deterministic C1 conflict-set attention head.

This module deliberately operates on sets of competing proposal edges.  The
component construction is used only to batch mutually competing proposals; the
model receives no graph adjacency and performs no message passing between
components.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from celltrack_alpha.trackastra_linear import proposal_features


INPUT_DIM = 49
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
SEED = 20260727
TRAINABLE_PARAMETER_LIMIT = 100_000


class C1ConflictSetError(RuntimeError):
    """The C1 data or architecture contract was violated."""


def seed_everything(seed: int = SEED) -> None:
    """Set the deterministic CPU state used by the compact head."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fold_index(identity: str, fold_count: int = 5) -> int:
    """Return the locked stable video-group fold."""

    if fold_count < 3 or not identity.isascii():
        raise C1ConflictSetError("fold inputs are invalid")
    return int.from_bytes(
        hashlib.sha256(identity.encode("ascii")).digest()[:8], "big"
    ) % fold_count


def nested_partitions(
    identities: Sequence[str], fold_count: int = 5
) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    """Return fit/calibration/held video groups for each outer fold."""

    ordered = tuple(sorted(str(value) for value in identities))
    if not ordered or len(ordered) != len(set(ordered)):
        raise C1ConflictSetError("video identities are empty or duplicated")
    result = []
    held_once: list[str] = []
    for held_fold in range(fold_count):
        calibration_fold = (held_fold + 1) % fold_count
        fit_ids = tuple(
            value
            for value in ordered
            if fold_index(value, fold_count)
            not in (held_fold, calibration_fold)
        )
        calibration_ids = tuple(
            value
            for value in ordered
            if fold_index(value, fold_count) == calibration_fold
        )
        held_ids = tuple(
            value
            for value in ordered
            if fold_index(value, fold_count) == held_fold
        )
        groups = tuple(map(set, (fit_ids, calibration_ids, held_ids)))
        if (
            not all(groups)
            or groups[0] & groups[1]
            or groups[0] & groups[2]
            or groups[1] & groups[2]
            or set().union(*groups) != set(ordered)
        ):
            raise C1ConflictSetError("nested partitions are not disjoint")
        held_once.extend(held_ids)
        result.append((fit_ids, calibration_ids, held_ids))
    if sorted(held_once) != list(ordered):
        raise C1ConflictSetError("each video must be held exactly once")
    return result


def _union_find_components(endpoint_sets: Sequence[set[int]]) -> list[np.ndarray]:
    parent = list(range(len(endpoint_sets)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            if root_left > root_right:
                root_left, root_right = root_right, root_left
            parent[root_right] = root_left

    owner: dict[int, int] = {}
    for index, endpoints in enumerate(endpoint_sets):
        for endpoint in sorted(endpoints):
            previous = owner.setdefault(endpoint, index)
            union(index, previous)
    groups: dict[int, list[int]] = {}
    for index in range(len(endpoint_sets)):
        groups.setdefault(find(index), []).append(index)
    return [
        np.asarray(values, dtype=np.int64)
        for _root, values in sorted(
            groups.items(), key=lambda item: min(item[1])
        )
    ]


def conflict_components(payload: Mapping[str, Any]) -> list[np.ndarray]:
    """Build deterministic per-transition proposal sets from shared endpoints."""

    challengers = np.asarray(payload.get("challengers"), dtype=np.int64)
    conflicts = np.asarray(payload.get("conflicts"), dtype=np.int64)
    proposal_offsets = np.asarray(
        payload.get("proposal_offsets"), dtype=np.int64
    )
    transition_offsets = np.asarray(
        payload.get("transition_offsets"), dtype=np.int64
    )
    if (
        challengers.ndim != 2
        or challengers.shape[1:] != (2,)
        or conflicts.ndim != 2
        or conflicts.shape[1:] != (2,)
        or proposal_offsets.shape != (len(challengers) + 1,)
        or transition_offsets.ndim != 1
        or len(transition_offsets) < 2
        or transition_offsets[0] != 0
        or transition_offsets[-1] != len(challengers)
        or proposal_offsets[0] != 0
        or proposal_offsets[-1] != len(conflicts)
        or np.any(np.diff(proposal_offsets) <= 0)
        or np.any(np.diff(transition_offsets) < 0)
    ):
        raise C1ConflictSetError("cached conflict arrays are malformed")

    components: list[np.ndarray] = []
    for start, end in zip(
        transition_offsets[:-1], transition_offsets[1:], strict=True
    ):
        start_i, end_i = int(start), int(end)
        endpoint_sets = []
        for proposal in range(start_i, end_i):
            row_start = int(proposal_offsets[proposal])
            row_end = int(proposal_offsets[proposal + 1])
            endpoints = set(map(int, challengers[proposal]))
            endpoints.update(map(int, conflicts[row_start:row_end].reshape(-1)))
            endpoint_sets.append(endpoints)
        for local in _union_find_components(endpoint_sets):
            components.append(local + start_i)
    if sum(map(len, components)) != len(challengers):
        raise C1ConflictSetError("component coverage is not one-to-one")
    return components


@dataclass(frozen=True)
class ComponentExample:
    """One proposal conflict set and its optional binary targets."""

    identity: str
    ordinal: int
    features: np.ndarray
    labels: np.ndarray | None
    proposal_indices: np.ndarray


def component_examples(
    identity: str,
    payload: Mapping[str, Any],
    labels: np.ndarray | None = None,
) -> list[ComponentExample]:
    """Convert one authenticated video payload into component examples."""

    features = proposal_features(payload).astype(np.float32, copy=False)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape != (len(features),) or np.any(
            ~np.isin(labels, (-1, 0, 1))
        ):
            raise C1ConflictSetError("proposal labels are malformed")
    result = []
    for ordinal, indices in enumerate(conflict_components(payload)):
        result.append(
            ComponentExample(
                identity=identity,
                ordinal=ordinal,
                features=features[indices],
                labels=None if labels is None else labels[indices],
                proposal_indices=indices,
            )
        )
    return result


class ConflictSetAttention(nn.Module):
    """One permutation-equivariant attention block plus explicit no-link token."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.attention = nn.MultiheadAttention(
            HIDDEN_DIM, ATTENTION_HEADS, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.no_link = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return proposal logits and the component no-link logit.

        ``features`` has shape ``(batch, proposals, 49)``.  ``valid`` is true for
        real proposal tokens and false for padding.
        """

        if features.ndim != 3 or features.shape[-1] != INPUT_DIM:
            raise C1ConflictSetError("attention input shape is invalid")
        batch, count, _width = features.shape
        if valid is None:
            valid = torch.ones(
                (batch, count), dtype=torch.bool, device=features.device
            )
        if valid.shape != (batch, count) or valid.dtype != torch.bool:
            raise C1ConflictSetError("attention validity mask is invalid")
        projected = self.projection(features)
        no_link = self.no_link.expand(batch, -1, -1)
        tokens = torch.cat((no_link, projected), dim=1)
        token_valid = torch.cat(
            (
                torch.ones(
                    (batch, 1), dtype=torch.bool, device=features.device
                ),
                valid,
            ),
            dim=1,
        )
        attended, _weights = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~token_valid,
            need_weights=False,
        )
        encoded = self.norm(tokens + attended)
        logits = self.output(encoded).squeeze(-1)
        proposal_logits = logits[:, 1:].masked_fill(~valid, 0.0)
        return proposal_logits, logits[:, 0]


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters and enforce the C1 ceiling."""

    value = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if value > TRAINABLE_PARAMETER_LIMIT:
        raise C1ConflictSetError("C1 trainable parameter ceiling exceeded")
    return value


def component_loss(
    proposal_logits: torch.Tensor,
    no_link_logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: float,
    known: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary proposal loss plus equally weighted component no-link loss."""

    if known is None:
        known = valid
    if (
        proposal_logits.shape != labels.shape
        or proposal_logits.shape != valid.shape
        or known.shape != valid.shape
        or no_link_logits.shape != proposal_logits.shape[:1]
    ):
        raise C1ConflictSetError("loss tensors are incompatible")
    supervised = valid & known
    if not torch.any(supervised):
        raise C1ConflictSetError("batch has no supervised proposal")
    edge_loss = F.binary_cross_entropy_with_logits(
        proposal_logits[supervised],
        labels[supervised],
        pos_weight=torch.as_tensor(
            positive_weight,
            dtype=proposal_logits.dtype,
            device=proposal_logits.device,
        ),
    )
    complete = torch.all(known | ~valid, dim=1)
    if torch.any(complete):
        no_link_targets = (
            (labels.clamp_min(0) * valid).sum(dim=1) == 0
        ).to(dtype=proposal_logits.dtype)
        no_link_loss = F.binary_cross_entropy_with_logits(
            no_link_logits[complete], no_link_targets[complete]
        )
        return edge_loss + no_link_loss
    return edge_loss


def _padded_batch(
    examples: Sequence[ComponentExample],
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise C1ConflictSetError("component batch is empty")
    width = max(len(example.features) for example in examples)
    features = np.zeros((len(examples), width, INPUT_DIM), dtype=np.float32)
    labels = np.zeros((len(examples), width), dtype=np.float32)
    valid = np.zeros((len(examples), width), dtype=bool)
    known = np.zeros((len(examples), width), dtype=bool)
    for row, example in enumerate(examples):
        count = len(example.features)
        features[row, :count] = (
            np.asarray(example.features, dtype=np.float32) - mean
        ) / scale
        valid[row, :count] = True
        if example.labels is not None:
            raw = np.asarray(example.labels, dtype=np.float32)
            known[row, :count] = raw >= 0
            labels[row, :count] = np.maximum(raw, 0)
    return (
        torch.from_numpy(features),
        torch.from_numpy(labels),
        torch.from_numpy(valid),
        torch.from_numpy(known),
    )


@dataclass
class FittedConflictHead:
    """A fitted C1 fold plus fit-only standardization."""

    model: ConflictSetAttention
    mean: np.ndarray
    scale: np.ndarray
    positive_weight: float


def fit_conflict_head(
    examples: Sequence[ComponentExample],
    *,
    seed: int = SEED,
    epochs: int = 12,
    component_batch_size: int = 64,
) -> FittedConflictHead:
    """Fit the single preregistered head on labeled fit-video components."""

    if epochs <= 0 or component_batch_size <= 0:
        raise C1ConflictSetError("training schedule is invalid")
    all_examples = tuple(examples)
    train = tuple(
        example
        for example in all_examples
        if example.labels is not None
        and np.any(np.asarray(example.labels) >= 0)
    )
    if not train:
        raise C1ConflictSetError("fit partition has no exact labels")
    all_tokens = np.concatenate(
        [np.asarray(example.features, dtype=np.float64) for example in all_examples]
    )
    mean = all_tokens.mean(axis=0).astype(np.float32)
    scale = np.maximum(all_tokens.std(axis=0), 1.0e-6).astype(np.float32)
    joined = np.concatenate(
        [
            np.asarray(example.labels)[np.asarray(example.labels) >= 0]
            for example in train
        ]
    )
    positives = int(np.sum(joined == 1))
    negatives = int(np.sum(joined == 0))
    if positives == 0:
        raise C1ConflictSetError("fit partition has no positive proposal")
    positive_weight = float(np.clip(negatives / positives, 1.0, 20.0))

    seed_everything(seed)
    model = ConflictSetAttention()
    parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-4, weight_decay=1.0e-4
    )
    generator = np.random.default_rng(seed)
    model.train()
    for _epoch in range(epochs):
        order = generator.permutation(len(train))
        for start in range(0, len(train), component_batch_size):
            batch = [train[int(index)] for index in order[start : start + component_batch_size]]
            features, labels, valid, known = _padded_batch(batch, mean, scale)
            optimizer.zero_grad(set_to_none=True)
            logits, no_link = model(features, valid)
            loss = component_loss(
                logits,
                no_link,
                labels,
                valid,
                positive_weight,
                known,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return FittedConflictHead(
        model=model,
        mean=mean,
        scale=scale,
        positive_weight=positive_weight,
    )


def predict_components(
    fitted: FittedConflictHead,
    examples: Sequence[ComponentExample],
) -> list[np.ndarray]:
    """Score components without communicating across them."""

    result = []
    fitted.model.eval()
    with torch.no_grad():
        for example in examples:
            features, _labels, valid, _known = _padded_batch(
                [example], fitted.mean, fitted.scale
            )
            logits, no_link = fitted.model(features, valid)
            result.append(
                abstention_scores(logits, no_link)[0, : len(example.features)]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return result


def abstention_scores(
    proposal_logits: torch.Tensor, no_link_logits: torch.Tensor
) -> torch.Tensor:
    """Return proposal evidence relative to the learned no-link alternative."""

    return proposal_logits - no_link_logits[:, None]


def mechanism_ablated_scores(
    model: ConflictSetAttention,
    features: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenwise no-attention control sharing projection, norm, and output."""

    projected = model.projection(features)
    batch = features.shape[0]
    no_link = model.no_link.expand(batch, -1, -1)
    tokens = torch.cat((no_link, projected), dim=1)
    logits = model.output(model.norm(tokens)).squeeze(-1)
    return logits[:, 1:].masked_fill(~valid, 0.0), logits[:, 0]


__all__ = [
    "ATTENTION_HEADS",
    "C1ConflictSetError",
    "ComponentExample",
    "ConflictSetAttention",
    "HIDDEN_DIM",
    "INPUT_DIM",
    "SEED",
    "TRAINABLE_PARAMETER_LIMIT",
    "abstention_scores",
    "component_examples",
    "component_loss",
    "conflict_components",
    "fold_index",
    "fit_conflict_head",
    "mechanism_ablated_scores",
    "nested_partitions",
    "parameter_count",
    "predict_components",
    "seed_everything",
]
