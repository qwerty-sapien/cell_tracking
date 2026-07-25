"""Fail-closed full-context temporal ranking for selected-graph swaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import appearance_swap
from .edge_data import SOURCE_CONFLICT, TARGET_CONFLICT


GATE_SCHEMA = "celltrack-alpha.selected-graph-temporal-family-rank-gate.v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class TemporalRankSwapError(RuntimeError):
    """A temporal-rank runtime identity or topology contract failed."""


class TemporalRankGate(torch.nn.Module):
    """Exact runtime architecture for the fitted family gate."""

    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.network = (
            torch.nn.Linear(feature_count, 1)
            if hidden == 0
            else torch.nn.Sequential(
                torch.nn.Linear(feature_count, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def load_gate(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_family: str = "6bba",
) -> tuple[TemporalRankGate, np.ndarray, np.ndarray, float]:
    """Load a hash-pinned fitted family gate."""

    source = Path(path)
    if (
        len(expected_sha256) != 64
        or appearance_swap.sha256_file(source) != expected_sha256
    ):
        raise TemporalRankSwapError("temporal gate SHA-256 mismatch")
    value = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != GATE_SCHEMA
        or value.get("family") != expected_family
        or int(value.get("feature_count", 0)) != 48
        or int(value.get("hidden", -1)) not in (0, 16)
    ):
        raise TemporalRankSwapError("temporal gate identity is invalid")
    mean = np.asarray(value.get("mean"), dtype=np.float32)
    scale = np.asarray(value.get("scale"), dtype=np.float32)
    threshold = float(value.get("threshold", math.nan))
    if (
        mean.shape != (48,)
        or scale.shape != (48,)
        or np.any(scale <= 0.0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not math.isfinite(threshold)
    ):
        raise TemporalRankSwapError("temporal gate normalization is invalid")
    gate = TemporalRankGate(48, int(value["hidden"]))
    gate.load_state_dict(value["state_dict"], strict=True)
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate, mean, scale, threshold


def _kind(challenger: tuple[int, int], winner: tuple[int, int]) -> int:
    if challenger[0] == winner[0] and challenger[1] != winner[1]:
        return SOURCE_CONFLICT
    if challenger[1] == winner[1] and challenger[0] != winner[0]:
        return TARGET_CONFLICT
    raise TemporalRankSwapError("winner is not a one-endpoint conflict")


def context_features(
    *,
    challenger: tuple[int, int],
    winner: tuple[int, int],
    selected_in: Mapping[int, int],
    selected_out: Mapping[int, int],
    detections: Mapping[int, tuple[int, np.ndarray]],
    vectors: Mapping[int, np.ndarray],
) -> np.ndarray:
    source, target = challenger
    winner_source, winner_target = winner
    source_t, source_point = detections[source]
    target_t, target_point = detections[target]
    if target_t != source_t + 1:
        raise TemporalRankSwapError("challenger is not adjacent")
    pair_margin = appearance_swap.ncc(vectors[source], vectors[target]) - (
        appearance_swap.ncc(vectors[winner_source], vectors[winner_target])
    )
    displacement = (target_point - source_point) * SCALE_ZYX
    context_ncc = motion_margin = 0.0
    available = False
    kind = _kind(challenger, winner)
    if kind == SOURCE_CONFLICT:
        predecessor = selected_in.get(source)
        if predecessor is not None and detections[predecessor][0] == source_t - 1:
            previous = (source_point - detections[predecessor][1]) * SCALE_ZYX
            incumbent = (
                detections[winner_target][1] - source_point
            ) * SCALE_ZYX
            context_ncc = appearance_swap.ncc(
                vectors[predecessor], vectors[target]
            ) - appearance_swap.ncc(
                vectors[predecessor], vectors[winner_target]
            )
            motion_margin = (
                np.linalg.norm(incumbent - previous)
                - np.linalg.norm(displacement - previous)
            ) / 10.0
            available = True
    else:
        successor = selected_out.get(target)
        if successor is not None and detections[successor][0] == source_t + 2:
            following = (
                detections[successor][1] - target_point
            ) * SCALE_ZYX
            incumbent = (
                target_point - detections[winner_source][1]
            ) * SCALE_ZYX
            context_ncc = appearance_swap.ncc(
                vectors[source], vectors[successor]
            ) - appearance_swap.ncc(
                vectors[winner_source], vectors[successor]
            )
            motion_margin = (
                np.linalg.norm(incumbent - following)
                - np.linalg.norm(displacement - following)
            ) / 10.0
            available = True
    result = np.asarray(
        (pair_margin, context_ncc, motion_margin, float(available)),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise TemporalRankSwapError("temporal context is non-finite")
    return result


def propose_swaps(
    cache: Any,
    scorer: Any,
    selected: set[tuple[int, int]],
    geometry_gate: torch.nn.Module,
    temporal_gate: TemporalRankGate,
    mean: np.ndarray,
    scale: np.ndarray,
    image: Any,
    *,
    geometry_threshold: float,
    temporal_threshold: float,
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Rank geometry-prefiltered conflicts by the fitted temporal family gate."""

    groups, counts = appearance_swap.proposal_feature_groups(
        cache, scorer, selected
    )
    selected_in = {target: source for source, target in selected}
    selected_out = {source: target for source, target in selected}
    detections = {
        int(node_id): (
            int(frame.frame_index),
            np.asarray(coordinate, dtype=np.float64),
        )
        for frame in cache.detections
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        )
    }
    filtered_by_frame: dict[int, list[dict[str, Any]]] = {}
    geometry_count = 0
    for group in groups:
        relative = np.asarray(group["relative_features"], dtype=np.float32)
        with torch.no_grad():
            geometry_probabilities = torch.sigmoid(
                geometry_gate(torch.from_numpy(relative))
            ).numpy()
        geometry_confidence = float(np.min(geometry_probabilities))
        if geometry_confidence < geometry_threshold:
            continue
        geometry_count += 1
        challenger = tuple(int(v) for v in group["challenger"])
        source_t = detections[challenger[0]][0]
        filtered_by_frame.setdefault(source_t, []).append(
            {
                "challenger": challenger,
                "conflicts": [
                    tuple(int(v) for v in pair) for pair in group["conflicts"]
                ],
                "relative": relative,
                "geometry_confidence": geometry_confidence,
                "geometry_probabilities": geometry_probabilities,
            }
        )
    raw = []
    extracted_nodes = 0
    for source_t in sorted(filtered_by_frame):
        frame_groups = filtered_by_frame[source_t]
        needed: dict[int, set[int]] = {
            source_t - 1: set(),
            source_t: set(),
            source_t + 1: set(),
            source_t + 2: set(),
        }
        for group in frame_groups:
            challenger = group["challenger"]
            for pair in (challenger, *group["conflicts"]):
                needed[detections[pair[0]][0]].add(pair[0])
                needed[detections[pair[1]][0]].add(pair[1])
            predecessor = selected_in.get(challenger[0])
            successor = selected_out.get(challenger[1])
            if predecessor is not None and detections[predecessor][0] == source_t - 1:
                needed[source_t - 1].add(predecessor)
            if successor is not None and detections[successor][0] == source_t + 2:
                needed[source_t + 2].add(successor)
        vectors = {}
        for time_index, node_ids in needed.items():
            if not node_ids:
                continue
            frame = np.asarray(image[time_index])
            for node_id in sorted(node_ids):
                vectors[node_id] = appearance_swap.unit_vector(
                    appearance_swap.extract_patch(
                        frame, detections[node_id][1]
                    )
                )
        extracted_nodes += len(vectors)
        for group in frame_groups:
            context = np.stack(
                [
                    context_features(
                        challenger=group["challenger"],
                        winner=winner,
                        selected_in=selected_in,
                        selected_out=selected_out,
                        detections=detections,
                        vectors=vectors,
                    )
                    for winner in group["conflicts"]
                ]
            )
            features = np.concatenate((group["relative"], context), axis=1)
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    temporal_gate(
                        torch.from_numpy((features - mean) / scale)
                    )
                ).numpy()
            confidence = float(np.min(probabilities))
            if confidence < temporal_threshold:
                continue
            raw.append(
                {
                    "challenger": list(group["challenger"]),
                    "conflicts": [list(pair) for pair in group["conflicts"]],
                    "confidence": confidence,
                    "appearance_margins": context[:, 0].tolist(),
                    "geometry_confidence": group["geometry_confidence"],
                    "row_probabilities": probabilities.tolist(),
                }
            )
    accepted_before_cap = appearance_swap._greedy_nonconflicting(raw)
    accepted = accepted_before_cap[:maximum]
    return accepted, {
        **counts,
        "geometry_threshold_proposals": geometry_count,
        "temporal_threshold_proposals": len(raw),
        "temporal_extracted_node_visits": extracted_nodes,
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": True,
        "cap_removed": len(accepted_before_cap) - len(accepted),
    }


__all__ = [
    "TemporalRankGate",
    "TemporalRankSwapError",
    "context_features",
    "load_gate",
    "propose_swaps",
]
