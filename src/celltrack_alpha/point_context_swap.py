"""Deployment-complete point-context ranking for selected-graph swaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from . import appearance_swap, temporal_rank_swap
from .point_context import (
    PointContextTransformer,
    ProposalWindow,
    collate_windows,
    parental_probabilities,
)


class PointContextSwapError(RuntimeError):
    """A full-pool point-context proposal contract failed."""


@dataclass(frozen=True)
class RuntimeProposalWindow:
    """One transition's model inputs with lossless graph identities."""

    identity: str
    family: int
    target_t: int
    features: np.ndarray
    target_ids: np.ndarray
    challengers: tuple[tuple[int, int], ...]
    conflicts: tuple[tuple[tuple[int, int], ...], ...]
    row_features: tuple[np.ndarray, ...]
    geometry_confidences: np.ndarray
    geometry_probabilities: tuple[np.ndarray, ...]

    def validate(self) -> None:
        count = len(self.challengers)
        if (
            self.family not in (0, 1)
            or not self.identity
            or count == 0
            or self.features.shape != (count, 48)
            or self.target_ids.shape != (count,)
            or self.geometry_confidences.shape != (count,)
            or len(self.conflicts) != count
            or len(self.row_features) != count
            or len(self.geometry_probabilities) != count
            or not np.isfinite(self.features).all()
            or not np.isfinite(self.geometry_confidences).all()
        ):
            raise PointContextSwapError("runtime proposal window is malformed")
        for index, (challenger, conflicts, rows, probabilities) in enumerate(
            zip(
                self.challengers,
                self.conflicts,
                self.row_features,
                self.geometry_probabilities,
                strict=True,
            )
        ):
            if (
                len(challenger) != 2
                or int(self.target_ids[index]) != challenger[1]
                or not conflicts
                or rows.shape != (len(conflicts), 48)
                or probabilities.shape != (len(conflicts),)
                or not np.isfinite(rows).all()
                or not np.isfinite(probabilities).all()
            ):
                raise PointContextSwapError(
                    "runtime proposal identities and feature rows disagree"
                )
            limiting = min(
                range(len(conflicts)),
                key=lambda row: (float(rows[row, 44]), row),
            )
            if not np.array_equal(self.features[index], rows[limiting]):
                raise PointContextSwapError(
                    "runtime token is not the limiting conflict row"
                )

    def model_window(self) -> ProposalWindow:
        self.validate()
        window = ProposalWindow(
            identity=self.identity,
            family=self.family,
            target_t=self.target_t,
            features=self.features,
            labels=np.full(len(self.challengers), -1, dtype=np.int8),
            target_ids=self.target_ids,
        )
        window.validate()
        return window


def _detections(cache: Any) -> dict[int, tuple[int, np.ndarray]]:
    result: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            key = int(node_id)
            if key in result:
                raise PointContextSwapError("detection node identity repeats")
            result[key] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    return result


def build_full_pool_windows(
    cache: Any,
    scorer: Any,
    selected: set[tuple[int, int]],
    geometry_gate: torch.nn.Module,
    image: Any,
    *,
    identity: str,
    family: int,
    geometry_threshold: float,
) -> tuple[list[RuntimeProposalWindow], dict[str, int]]:
    """Build all geometry-admissible deployment proposals without sampling."""

    if (
        not identity
        or family not in (0, 1)
        or not 0.0 <= geometry_threshold <= 1.0
    ):
        raise PointContextSwapError("runtime identity or geometry threshold is invalid")
    groups, counts = appearance_swap.proposal_feature_groups(
        cache, scorer, selected
    )
    detections = _detections(cache)
    selected_in = {target: source for source, target in selected}
    selected_out = {source: target for source, target in selected}
    by_source_t: dict[int, list[dict[str, Any]]] = {}
    geometry_count = 0
    for group in groups:
        relative = np.asarray(group["relative_features"], dtype=np.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(
                geometry_gate(torch.from_numpy(relative))
            ).numpy()
        confidence = float(np.min(probabilities))
        if confidence < geometry_threshold:
            continue
        challenger = tuple(int(value) for value in group["challenger"])
        source_t = detections[challenger[0]][0]
        if detections[challenger[1]][0] != source_t + 1:
            raise PointContextSwapError("challenger is not adjacent")
        geometry_count += 1
        by_source_t.setdefault(source_t, []).append(
            {
                "challenger": challenger,
                "conflicts": tuple(
                    tuple(int(value) for value in pair)
                    for pair in group["conflicts"]
                ),
                "relative": relative,
                "geometry_confidence": confidence,
                "geometry_probabilities": np.asarray(
                    probabilities, dtype=np.float32
                ),
            }
        )

    runtime_windows: list[RuntimeProposalWindow] = []
    extracted_nodes = 0
    for source_t in sorted(by_source_t):
        frame_groups = by_source_t[source_t]
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
            if (
                predecessor is not None
                and detections[predecessor][0] == source_t - 1
            ):
                needed[source_t - 1].add(predecessor)
            if (
                successor is not None
                and detections[successor][0] == source_t + 2
            ):
                needed[source_t + 2].add(successor)
        vectors: dict[int, np.ndarray] = {}
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

        tokens = []
        for group in frame_groups:
            rows = np.concatenate(
                (
                    group["relative"],
                    np.stack(
                        [
                            temporal_rank_swap.context_features(
                                challenger=group["challenger"],
                                winner=winner,
                                selected_in=selected_in,
                                selected_out=selected_out,
                                detections=detections,
                                vectors=vectors,
                            )
                            for winner in group["conflicts"]
                        ]
                    ),
                ),
                axis=1,
            ).astype(np.float32, copy=False)
            limiting = min(
                range(len(rows)),
                key=lambda row: (float(rows[row, 44]), row),
            )
            tokens.append(
                {
                    **group,
                    "rows": rows,
                    "feature": rows[limiting],
                }
            )
        tokens.sort(
            key=lambda row: (
                row["challenger"][1],
                row["challenger"][0],
                row["conflicts"],
            )
        )
        window = RuntimeProposalWindow(
            identity=identity,
            family=family,
            target_t=source_t + 1,
            features=np.stack([row["feature"] for row in tokens]),
            target_ids=np.asarray(
                [row["challenger"][1] for row in tokens], dtype=np.int64
            ),
            challengers=tuple(row["challenger"] for row in tokens),
            conflicts=tuple(row["conflicts"] for row in tokens),
            row_features=tuple(row["rows"] for row in tokens),
            geometry_confidences=np.asarray(
                [row["geometry_confidence"] for row in tokens],
                dtype=np.float32,
            ),
            geometry_probabilities=tuple(
                row["geometry_probabilities"] for row in tokens
            ),
        )
        window.validate()
        runtime_windows.append(window)
    return runtime_windows, {
        **counts,
        "geometry_threshold_proposals": geometry_count,
        "full_pool_proposals": sum(
            len(window.challengers) for window in runtime_windows
        ),
        "runtime_windows": len(runtime_windows),
        "runtime_extracted_node_visits": extracted_nodes,
        "sampled_background_proposals": 0,
    }


def propose_swaps(
    model: PointContextTransformer,
    windows: Sequence[RuntimeProposalWindow],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    maximum: int,
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Score all full-pool tokens, then apply threshold, greediness, and cap."""

    if (
        mean.shape != (48,)
        or scale.shape != (48,)
        or np.any(scale <= 0.0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not math.isfinite(threshold)
        or type(maximum) is not int
        or maximum < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise PointContextSwapError("point-context scoring contract is invalid")
    if not windows:
        return [], {
            "full_pool_proposals": 0,
            "threshold_proposals": 0,
            "accepted_before_cap": 0,
            "accepted_proposals": 0,
            "proposal_cap": maximum,
            "cap_active": True,
            "cap_removed": 0,
        }
    for window in windows:
        window.validate()
    model.eval()
    raw: list[dict[str, Any]] = []
    scored = 0
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start : start + batch_size]
            batch = collate_windows(
                [window.model_window() for window in batch_windows],
                mean=mean,
                scale=scale,
                device=torch.device("cpu"),
            )
            edge_logits, hidden = model(
                batch["features"],
                batch["families"],
                batch["padding_mask"],
            )
            probabilities = parental_probabilities(
                model,
                edge_logits,
                hidden,
                batch["target_ids"],
                batch["padding_mask"],
            )
            for window, scores in zip(
                batch_windows, probabilities, strict=True
            ):
                if len(scores) != len(window.challengers):
                    raise PointContextSwapError(
                        "model score count does not match runtime identities"
                    )
                scored += len(scores)
                for index, confidence in enumerate(scores):
                    if float(confidence) < threshold:
                        continue
                    raw.append(
                        {
                            "challenger": list(window.challengers[index]),
                            "conflicts": [
                                list(pair) for pair in window.conflicts[index]
                            ],
                            "confidence": float(confidence),
                            "appearance_margins": window.row_features[
                                index
                            ][:, 44].tolist(),
                            "geometry_confidence": float(
                                window.geometry_confidences[index]
                            ),
                            "row_probabilities": [
                                float(confidence)
                            ]
                            * len(window.conflicts[index]),
                        }
                    )
    accepted_before_cap = appearance_swap._greedy_nonconflicting(raw)
    accepted = accepted_before_cap[:maximum]
    return accepted, {
        "full_pool_proposals": scored,
        "threshold_proposals": len(raw),
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": True,
        "cap_removed": len(accepted_before_cap) - len(accepted),
    }


__all__ = [
    "PointContextSwapError",
    "RuntimeProposalWindow",
    "build_full_pool_windows",
    "propose_swaps",
]
