"""Fail-closed full-3D appearance swaps for the frozen V45 association graph.

The module contains only the runtime mechanism.  Training, threshold selection,
and ground-truth evaluation remain outside the submission path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import bh_graph
from .relative_geometry_gate import RelativeGeometryGate, relative_features


CONTRACT_SCHEMA = "celltrack-alpha.appearance-direct-swap-contract.v1"
GATE_SCHEMAS = {
    "celltrack-alpha.selected-graph-conflict-gate.v1",
    "celltrack-alpha.selected-graph-conflict-gate.v2",
}
PATCH_SHAPE = (16, 32, 32)


class AppearanceSwapError(RuntimeError):
    """The frozen appearance-swap runtime contract was violated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_patch(
    frame: np.ndarray,
    center_zyx: Sequence[float],
    shape: Sequence[int] = PATCH_SHAPE,
) -> np.ndarray:
    """Extract a fixed 3-D patch with deterministic reflected padding."""

    frame = np.asarray(frame)
    if frame.ndim != 3 or tuple(shape) != PATCH_SHAPE:
        raise AppearanceSwapError("appearance patches must be 16x32x32 in 3-D")
    center = np.rint(np.asarray(center_zyx, dtype=np.float64)).astype(int)
    shape_array = np.asarray(shape, dtype=int)
    starts = center - shape_array // 2
    ends = starts + shape_array
    slices = tuple(
        slice(max(0, int(start)), min(int(limit), int(end)))
        for start, end, limit in zip(starts, ends, frame.shape, strict=True)
    )
    patch = np.asarray(frame[slices], dtype=np.float32)
    padding = tuple(
        (max(0, -int(start)), max(0, int(end) - int(limit)))
        for start, end, limit in zip(starts, ends, frame.shape, strict=True)
    )
    if any(left or right for left, right in padding):
        mode = "reflect" if all(size > 1 for size in patch.shape) else "edge"
        patch = np.pad(patch, padding, mode=mode)
    if patch.shape != PATCH_SHAPE:
        raise AppearanceSwapError(
            f"patch shape mismatch: {patch.shape} != {PATCH_SHAPE}"
        )
    return patch


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.mean(values))
    norm = float(np.linalg.norm(centered.ravel()))
    if norm <= 1.0e-12:
        return np.zeros(centered.size, dtype=np.float32)
    return (centered / norm).ravel().astype(np.float32, copy=False)


def ncc(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source)
    target = np.asarray(target)
    if source.shape != target.shape or source.ndim != 1:
        raise AppearanceSwapError(
            "NCC vectors must be aligned one-dimensional arrays"
        )
    return float(np.dot(source, target))


def load_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    root: str | Path | None = None,
) -> tuple[dict[str, Any], RelativeGeometryGate]:
    """Load the immutable mechanism contract and its pinned geometry gate."""

    source = Path(path)
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not source.is_file()
        or sha256_file(source) != expected_sha256
    ):
        raise AppearanceSwapError("appearance contract SHA-256 mismatch")
    try:
        value = json.loads(source.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AppearanceSwapError("appearance contract is not valid ASCII JSON") from exc
    required = {
        "schema",
        "mechanism",
        "geometry_gate",
        "appearance",
        "proposal_cap",
        "local_confirmation",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise AppearanceSwapError(
            "appearance contract has missing or unexpected fields"
        )
    gate_spec = value.get("geometry_gate")
    appearance = value.get("appearance")
    cap = value.get("proposal_cap")
    local = value.get("local_confirmation")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("mechanism")
        != "post-gap-pre-prune-node-disjoint-full-3d-ncc-direct-swap"
        or not isinstance(gate_spec, Mapping)
        or set(gate_spec) != {"path", "sha256", "threshold"}
        or not isinstance(appearance, Mapping)
        or set(appearance) != {"metric", "patch_shape", "threshold"}
        or appearance.get("metric") != "zero-mean-unit-norm-ncc"
        or appearance.get("patch_shape") != list(PATCH_SHAPE)
        or not isinstance(cap, Mapping)
        or set(cap) != {"minimum_per_video", "selected_edge_fraction"}
        or not isinstance(local, Mapping)
        or set(local)
        != {
            "baseline_score",
            "candidate_score",
            "edge_fn",
            "edge_fp",
            "edge_tp",
            "report_sha256",
        }
    ):
        raise AppearanceSwapError("appearance contract identity is invalid")
    try:
        geometry_threshold = float(gate_spec["threshold"])
        appearance_threshold = float(appearance["threshold"])
        cap_fraction = float(cap["selected_edge_fraction"])
        minimum = int(cap["minimum_per_video"])
        baseline_score = float(local["baseline_score"])
        candidate_score = float(local["candidate_score"])
        counts = tuple(int(local[key]) for key in ("edge_tp", "edge_fp", "edge_fn"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AppearanceSwapError("appearance contract values are invalid") from exc
    report_sha = local.get("report_sha256")
    gate_sha = gate_spec.get("sha256")
    gate_relative = gate_spec.get("path")
    if (
        not 0.0 <= geometry_threshold <= 1.0
        or not math.isfinite(appearance_threshold)
        or not 0.0 < cap_fraction <= 1.0
        or minimum < 1
        or not math.isclose(baseline_score, 0.8731495935289003, abs_tol=1e-15)
        or not math.isclose(candidate_score, 0.8815220699597315, abs_tol=1e-15)
        or counts != (34320, 2459, 1948)
        or not isinstance(report_sha, str)
        or report_sha
        != "8da96a3da9e66869192f446fc6d59fedd6016bdefe4fb272b136df68258a0df4"
        or not isinstance(gate_sha, str)
        or len(gate_sha) != 64
        or any(character not in "0123456789abcdef" for character in gate_sha)
        or not isinstance(gate_relative, str)
        or Path(gate_relative).is_absolute()
        or ".." in Path(gate_relative).parts
    ):
        raise AppearanceSwapError("appearance contract frozen values differ")
    base = Path(root) if root is not None else source.parent
    gate_path = base / gate_relative
    if not gate_path.is_file() or sha256_file(gate_path) != gate_sha:
        raise AppearanceSwapError("appearance geometry-gate SHA-256 mismatch")
    try:
        payload = torch.load(gate_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AppearanceSwapError("appearance geometry gate cannot be loaded") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") not in GATE_SCHEMAS
        or not math.isclose(
            float(payload.get("threshold", math.nan)),
            geometry_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise AppearanceSwapError("appearance geometry-gate payload is invalid")
    gate = RelativeGeometryGate(hidden=int(payload["hidden"]))
    gate.load_state_dict(payload["state_dict"], strict=True)
    gate.eval()
    if any(parameter.requires_grad for parameter in gate.parameters()):
        for parameter in gate.parameters():
            parameter.requires_grad_(False)
    normalized = json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    normalized["contract_sha256"] = expected_sha256
    return normalized, gate


def selected_raw_pairs(
    graph_result: bh_graph.GraphResult,
    raw_node_ids: set[int],
) -> set[tuple[int, int]]:
    """Return raw-node edges selected after gap closure and postprocessing."""

    if not raw_node_ids:
        raise AppearanceSwapError("raw node identity set is empty")
    return {
        (int(edge.source_id), int(edge.target_id))
        for edge in graph_result.edges
        if int(edge.source_id) in raw_node_ids
        and int(edge.target_id) in raw_node_ids
    }


def proposal_feature_groups(
    cache: Any,
    scorer: Any,
    selected: set[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build selected-edge conflict groups from the frozen sparse scorer."""

    selected_out = {source: target for source, target in selected}
    selected_in = {target: source for source, target in selected}
    if len(selected_out) != len(selected) or len(selected_in) != len(selected):
        raise AppearanceSwapError("selected graph is not one-to-one")
    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            node_id_int = int(node_id)
            if node_id_int in detections:
                raise AppearanceSwapError("detection node identity repeats")
            detections[node_id_int] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    scale = np.asarray(scorer.config.scale_zyx, dtype=np.float64)
    feature_by_pair: dict[tuple[int, int], np.ndarray] = {}
    logit_by_pair: dict[tuple[int, int], float] = {}
    geometry_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for context, source, target in zip(
        cache.contexts,
        cache.detections[:-1],
        cache.detections[1:],
        strict=True,
    ):
        previous_velocity: dict[int, np.ndarray | None] = {}
        for node_id, coordinate in zip(
            source.node_ids, source.coords_zyx, strict=True
        ):
            source_id = int(node_id)
            parent_id = selected_in.get(source_id)
            parent = detections.get(parent_id) if parent_id is not None else None
            previous_velocity[source_id] = (
                None
                if parent is None or parent[0] != int(source.frame_index) - 1
                else (np.asarray(coordinate, dtype=np.float64) - parent[1]) * scale
            )
        candidates, checkpoint, features = scorer.candidate_features(
            context,
            source,
            target,
            previous_velocity_um=previous_velocity,
        )
        with torch.no_grad():
            logits = checkpoint + scorer.calibrator(features)
        feature_rows = features.detach().cpu().numpy()
        logit_rows = logits.detach().cpu().numpy()
        for row, (source_id, target_id) in enumerate(
            zip(candidates.source_ids, candidates.target_ids, strict=True)
        ):
            pair = (int(source_id), int(target_id))
            if pair in feature_by_pair:
                raise AppearanceSwapError("candidate pair repeats across contexts")
            feature_by_pair[pair] = np.asarray(
                feature_rows[row], dtype=np.float32
            )
            logit_by_pair[pair] = float(logit_rows[row])
            source_record = detections[pair[0]]
            target_record = detections[pair[1]]
            geometry_by_pair[pair] = np.concatenate(
                (source_record[1] * scale, target_record[1] * scale)
            ).astype(np.float32, copy=False)
    groups: list[dict[str, Any]] = []
    compared_rows = 0
    for challenger in sorted(set(feature_by_pair) - selected):
        conflicts: list[tuple[int, int]] = []
        source_target = selected_out.get(challenger[0])
        if source_target is not None and source_target != challenger[1]:
            conflict = (challenger[0], source_target)
            if conflict in feature_by_pair:
                conflicts.append(conflict)
        target_source = selected_in.get(challenger[1])
        if target_source is not None and target_source != challenger[0]:
            conflict = (target_source, challenger[1])
            if conflict in feature_by_pair and conflict not in conflicts:
                conflicts.append(conflict)
        if not conflicts:
            continue
        count = len(conflicts)
        relative = relative_features(
            np.repeat(feature_by_pair[challenger][None], count, axis=0),
            np.stack([feature_by_pair[pair] for pair in conflicts]),
            np.repeat(geometry_by_pair[challenger][None], count, axis=0),
            np.stack([geometry_by_pair[pair] for pair in conflicts]),
            challenger_logit=np.full(
                count, logit_by_pair[challenger], dtype=np.float32
            ),
            winner_logit=np.asarray(
                [logit_by_pair[pair] for pair in conflicts], dtype=np.float32
            ),
            dummy_logit=0.0,
            candidate_count=np.full(count, 2, dtype=np.float32),
        )
        compared_rows += count
        groups.append(
            {
                "challenger": list(challenger),
                "conflicts": [list(pair) for pair in conflicts],
                "relative_features": relative,
            }
        )
    return groups, {
        "candidate_pairs": len(feature_by_pair),
        "selected_raw_edges": len(selected),
        "compared_rows": compared_rows,
        "raw_proposals": len(groups),
    }


def _greedy_nonconflicting(
    proposals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    occupied_nodes: set[int] = set()
    for raw in sorted(
        proposals,
        key=lambda row: (
            -float(row["confidence"]),
            tuple(int(value) for value in row["challenger"]),
        ),
    ):
        challenger = tuple(int(value) for value in raw["challenger"])
        conflicts = [
            tuple(int(value) for value in pair) for pair in raw["conflicts"]
        ]
        affected = set(challenger)
        for pair in conflicts:
            affected.update(pair)
        if occupied_nodes.intersection(affected):
            continue
        occupied_nodes.update(affected)
        accepted.append(
            {
                "challenger": list(challenger),
                "conflicts": [list(pair) for pair in conflicts],
                "confidence": float(raw["confidence"]),
                "appearance_margins": [
                    float(value) for value in raw["appearance_margins"]
                ],
                "geometry_confidence": float(raw["geometry_confidence"]),
                "row_probabilities": [
                    float(value) for value in raw["row_probabilities"]
                ],
            }
        )
    return accepted


def propose_swaps(
    cache: Any,
    scorer: Any,
    selected: set[tuple[int, int]],
    gate: RelativeGeometryGate,
    image: Any,
    *,
    geometry_threshold: float,
    appearance_threshold: float,
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Apply the frozen geometry prefilter, 3-D NCC rule, greediness, and cap."""

    if (
        not 0.0 <= geometry_threshold <= 1.0
        or not math.isfinite(appearance_threshold)
        or type(maximum) is not int
        or maximum < 1
    ):
        raise AppearanceSwapError("appearance proposal thresholds/cap are invalid")
    groups, counts = proposal_feature_groups(cache, scorer, selected)
    detections: dict[int, tuple[int, np.ndarray]] = {}
    for frame in cache.detections:
        for node_id, coordinate in zip(
            frame.node_ids, frame.coords_zyx, strict=True
        ):
            detections[int(node_id)] = (
                int(frame.frame_index),
                np.asarray(coordinate, dtype=np.float64),
            )
    filtered_by_frame: dict[int, list[dict[str, Any]]] = {}
    geometry_count = 0
    for group in groups:
        relative = np.asarray(group["relative_features"], dtype=np.float32)
        with torch.no_grad():
            probabilities = torch.sigmoid(
                gate(torch.from_numpy(relative))
            ).numpy()
        geometry_confidence = float(np.min(probabilities))
        if geometry_confidence < geometry_threshold:
            continue
        geometry_count += 1
        challenger = tuple(int(value) for value in group["challenger"])
        source_t = detections[challenger[0]][0]
        if detections[challenger[1]][0] != source_t + 1:
            raise AppearanceSwapError("appearance challenger is not adjacent")
        filtered_by_frame.setdefault(source_t, []).append(
            {
                "challenger": challenger,
                "conflicts": [
                    tuple(int(value) for value in pair)
                    for pair in group["conflicts"]
                ],
                "geometry_confidence": geometry_confidence,
                "row_probabilities": probabilities,
            }
        )
    raw: list[dict[str, Any]] = []
    extracted_nodes = 0
    for source_t in sorted(filtered_by_frame):
        frame_groups = filtered_by_frame[source_t]
        needed_by_time: dict[int, set[int]] = {
            source_t: set(),
            source_t + 1: set(),
        }
        for group in frame_groups:
            for pair in (group["challenger"], *group["conflicts"]):
                for node_id in pair:
                    time_index = detections[node_id][0]
                    if time_index not in needed_by_time:
                        raise AppearanceSwapError(
                            "appearance conflict escaped its frame pair"
                        )
                    needed_by_time[time_index].add(node_id)
        vectors: dict[int, np.ndarray] = {}
        for time_index in (source_t, source_t + 1):
            frame = np.asarray(image[time_index])
            for node_id in sorted(needed_by_time[time_index]):
                vectors[node_id] = unit_vector(
                    extract_patch(frame, detections[node_id][1])
                )
        extracted_nodes += len(vectors)
        pair_ncc: dict[tuple[int, int], float] = {}

        def score(pair: tuple[int, int]) -> float:
            if pair not in pair_ncc:
                pair_ncc[pair] = ncc(vectors[pair[0]], vectors[pair[1]])
            return pair_ncc[pair]

        for group in frame_groups:
            challenger_score = score(group["challenger"])
            margins = [
                challenger_score - score(pair) for pair in group["conflicts"]
            ]
            confidence = float(min(margins))
            if confidence < appearance_threshold:
                continue
            raw.append(
                {
                    "challenger": list(group["challenger"]),
                    "conflicts": [list(pair) for pair in group["conflicts"]],
                    "confidence": confidence,
                    "appearance_margins": margins,
                    "geometry_confidence": group["geometry_confidence"],
                    "row_probabilities": group["row_probabilities"].tolist(),
                }
            )
    accepted_before_cap = _greedy_nonconflicting(raw)
    accepted = accepted_before_cap[:maximum]
    return accepted, {
        **counts,
        "geometry_threshold_proposals": geometry_count,
        "appearance_threshold_proposals": len(raw),
        "appearance_extracted_node_visits": extracted_nodes,
        "accepted_before_cap": len(accepted_before_cap),
        "accepted_proposals": len(accepted),
        "proposal_cap": maximum,
        "cap_active": True,
        "cap_removed": len(accepted_before_cap) - len(accepted),
        "accepted_source_only": sum(
            len(row["conflicts"]) == 1 for row in accepted
        ),
        "accepted_two_endpoint": sum(
            len(row["conflicts"]) == 2 for row in accepted
        ),
    }


def swap_post_gap_edges(
    nodes: Sequence[bh_graph.GraphNode],
    edges: Sequence[bh_graph.GraphEdge],
    proposals: Sequence[Mapping[str, Any]],
    probabilities: Mapping[tuple[int, int], float],
) -> tuple[tuple[bh_graph.GraphEdge, ...], dict[str, int]]:
    """Apply each frozen node-disjoint proposal exactly once before pruning."""

    node_by_id = {int(node.node_id): node for node in nodes}
    by_pair = {(int(edge.source_id), int(edge.target_id)): edge for edge in edges}
    if len(by_pair) != len(edges):
        raise AppearanceSwapError("post-gap graph contains duplicate edges")
    removed: set[tuple[int, int]] = set()
    declared_removed: set[tuple[int, int]] = set()
    added: dict[tuple[int, int], bh_graph.GraphEdge] = {}
    affected_nodes: set[int] = set()
    for proposal in proposals:
        challenger = tuple(int(value) for value in proposal["challenger"])
        conflicts = [
            tuple(int(value) for value in pair) for pair in proposal["conflicts"]
        ]
        if (
            challenger in by_pair
            or challenger in added
            or challenger not in probabilities
            or not conflicts
            or any(pair not in by_pair for pair in conflicts)
        ):
            raise AppearanceSwapError("proposal does not describe a valid local swap")
        affected = set(challenger)
        for pair in conflicts:
            affected.update(pair)
        if affected_nodes.intersection(affected):
            raise AppearanceSwapError("frozen proposals are not node-disjoint")
        affected_nodes.update(affected)
        declared_removed.update(conflicts)
        removed.update(
            pair
            for pair in by_pair
            if pair[0] == challenger[0] or pair[1] == challenger[1]
        )
        source = node_by_id[challenger[0]]
        target = node_by_id[challenger[1]]
        if target.t != source.t + 1:
            raise AppearanceSwapError("challenger is not adjacent in time")
        probability = float(probabilities[challenger])
        if not 0.0 <= probability <= 1.0:
            raise AppearanceSwapError("challenger probability is invalid")
        distance = bh_graph.physical_distance(source, target)
        cost = (
            bh_graph.motion_only_edge_cost(distance)
            - bh_graph.learned_edge_score_bonus(probability)
        )
        added[challenger] = bh_graph.GraphEdge(
            source_id=challenger[0],
            target_id=challenger[1],
            distance_um=distance,
            cost=cost,
            kind="appearance_direct_swap",
            learned_probability=probability,
        )
    swapped = tuple(
        sorted(
            (
                *(edge for pair, edge in by_pair.items() if pair not in removed),
                *added.values(),
            ),
            key=lambda edge: (
                node_by_id[int(edge.source_id)].t,
                int(edge.source_id),
                int(edge.target_id),
            ),
        )
    )
    bh_graph.validate_graph(nodes, swapped)
    swapped_pairs = {
        (int(edge.source_id), int(edge.target_id)) for edge in swapped
    }
    if not set(added).issubset(swapped_pairs) or removed.intersection(swapped_pairs):
        raise AppearanceSwapError("direct swap did not realize its exact topology")
    return swapped, {
        "proposals": len(proposals),
        "removed_conflicts": len(removed),
        "declared_conflicts": len(declared_removed),
        "extra_preprune_incident_removals": len(removed - declared_removed),
        "added_challengers": len(added),
        "edge_count_before": len(edges),
        "edge_count_after": len(swapped),
    }


__all__ = [
    "AppearanceSwapError",
    "CONTRACT_SCHEMA",
    "PATCH_SHAPE",
    "extract_patch",
    "load_contract",
    "ncc",
    "proposal_feature_groups",
    "propose_swaps",
    "selected_raw_pairs",
    "sha256_file",
    "swap_post_gap_edges",
    "unit_vector",
]
