#!/usr/bin/env python3
"""Outcome-blind action featurizer for staged DINOv2 XY endpoint embeddings."""

from __future__ import annotations

import hashlib

import numpy as np


SUPPLIER_ID = "F0-DINOV2-VITS14-XY-V1"
EMBEDDING_DIMENSION = 768
FEATURE_NAMES = (
    "cosine",
    "euclidean_over_sqrt768",
    "mean_absolute_difference",
    "mean_product",
    "source_norm_over_sqrt768",
    "target_norm_over_sqrt768",
    "norm_difference_over_sqrt768",
    "centered_correlation",
)


class FeatureContractError(ValueError):
    """Fail-closed feature contract violation."""


def action_features(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != EMBEDDING_DIMENSION
    ):
        raise FeatureContractError(
            f"expected equal (N,{EMBEDDING_DIMENSION}) arrays, got "
            f"{source.shape} and {target.shape}"
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise FeatureContractError("nonfinite endpoint embedding")
    difference = source - target
    source_norm = np.linalg.norm(source, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    norm_scale = np.sqrt(float(EMBEDDING_DIMENSION))
    cosine_denominator = np.maximum(source_norm * target_norm, 1e-8)
    centered_source = source - source.mean(axis=1, keepdims=True)
    centered_target = target - target.mean(axis=1, keepdims=True)
    correlation_denominator = np.maximum(
        np.linalg.norm(centered_source, axis=1)
        * np.linalg.norm(centered_target, axis=1),
        1e-8,
    )
    result = np.stack(
        (
            np.sum(source * target, axis=1) / cosine_denominator,
            np.linalg.norm(difference, axis=1) / norm_scale,
            np.mean(np.abs(difference), axis=1),
            np.mean(source * target, axis=1),
            source_norm / norm_scale,
            target_norm / norm_scale,
            (source_norm - target_norm) / norm_scale,
            np.sum(centered_source * centered_target, axis=1)
            / correlation_denominator,
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape != (len(source), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise FeatureContractError("invalid action-feature result")
    return result


def within_frame_permutation(
    values: np.ndarray,
    node_ids: np.ndarray,
    *,
    identity: str,
    frame: int,
    supplier_contract_sha256: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    node_ids = np.asarray(node_ids)
    if values.ndim != 2 or values.shape[1] != EMBEDDING_DIMENSION:
        raise FeatureContractError("embedding shape drift")
    if node_ids.ndim != 1 or len(node_ids) != len(values):
        raise FeatureContractError("node-id alignment drift")
    if not np.isfinite(values).all():
        raise FeatureContractError("nonfinite endpoint embedding")
    if len(supplier_contract_sha256) != 64:
        raise FeatureContractError("supplier contract SHA-256 must be 64 hex chars")
    try:
        bytes.fromhex(supplier_contract_sha256)
    except ValueError as exc:
        raise FeatureContractError("invalid supplier contract SHA-256") from exc

    unique_ids, first = np.unique(node_ids, return_index=True)
    vectors = values[first]
    seed = (
        "R-SEPARATION-FALSIFIER-44B6"
        f"{supplier_contract_sha256}{identity}{int(frame)}"
    )
    order = sorted(
        range(len(unique_ids)),
        key=lambda index: (
            hashlib.sha256(
                f"{seed}{int(unique_ids[index])}".encode("ascii")
            ).digest(),
            int(unique_ids[index]),
        ),
    )
    replacement: dict[int, np.ndarray] = {}
    for position, index in enumerate(order):
        replacement[int(unique_ids[index])] = vectors[order[(position + 1) % len(order)]]
    if not order:
        return np.empty_like(values)
    return np.stack([replacement[int(node)] for node in node_ids]).astype(np.float32)
