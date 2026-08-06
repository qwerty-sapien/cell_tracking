"""End-to-end scenario-routed low-compute tracking pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from celltrack_baseline.rule_based import BaselineConfig, dog_response, normalize_volume, parse_scale

from .pipeline_inputs import (
    validate_array_window as _validate_array_window,
    validate_output_offset_override,
)
from .proposals import (
    _best_score_coord_in_radius,
    _robust_unit,
    _sigma_voxels,
    annotate_local_density,
    generate_candidates,
)
from .routing import default_route_configs, profile_scenario, route_for_profile
from .scoring import ScoringCalibration, correct_nodes_by_t, proposal_policy_for_dataset, prune_nodes_by_t, score_edge_candidates, scoring_for_dataset
from .solver import build_edge_candidates, graph_diagnostics, select_edges
from .types import CandidateEdge, CandidateNode, TrackingRunResult

def _resolve_output_offset(dataset: str, route_config: Any) -> tuple[int, int, int]:
    for prefix, dz, dy, dx in route_config.prefix_output_offsets_zyx:
        if dataset.startswith(str(prefix)):
            return int(dz), int(dy), int(dx)
    return tuple(int(v) for v in route_config.output_offset_zyx)


def _shift_node_dicts(
    nodes: list[dict[str, Any]],
    offset_zyx: tuple[int, int, int],
    spatial_shape_zyx: tuple[int, int, int],
) -> list[dict[str, Any]]:
    dz, dy, dx = offset_zyx
    if (dz, dy, dx) == (0, 0, 0):
        return nodes
    z_size, y_size, x_size = spatial_shape_zyx
    shifted: list[dict[str, Any]] = []
    for node in nodes:
        row = dict(node)
        row["z"] = min(max(int(row["z"]) + dz, 0), z_size - 1)
        row["y"] = min(max(int(row["y"]) + dy, 0), y_size - 1)
        row["x"] = min(max(int(row["x"]) + dx, 0), x_size - 1)
        shifted.append(row)
    return shifted


def _refresh_node_intensities(nodes_by_t: list[list[Any]], array: Any, start: int) -> None:
    for rel_t, nodes in enumerate(nodes_by_t):
        if not nodes:
            continue
        volume = np.asarray(array[start + rel_t])
        for node in nodes:
            node.intensity = float(volume[int(node.z), int(node.y), int(node.x)])



def _localmax_refinement_maps(
    volume: np.ndarray,
    attrs: dict[str, Any] | None,
    scale_zyx: tuple[float, float, float],
    sources: tuple[str, ...],
) -> dict[str, np.ndarray]:
    requested = tuple(dict.fromkeys(str(source) for source in sources))
    supported = {"raw", "smooth", "local_contrast", "dog_abs", "dog_bright", "dog_dark"}
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"unsupported localmax refinement source(s): {unknown}")

    baseline = BaselineConfig(polarity="both")
    norm = normalize_volume(volume, attrs, baseline)
    maps: dict[str, np.ndarray] = {}
    if "raw" in requested:
        maps["raw"] = _robust_unit(norm)
    if "smooth" in requested:
        maps["smooth"] = _robust_unit(
            gaussian_filter(norm, _sigma_voxels(0.7, scale_zyx), mode="nearest")
        )
    if "local_contrast" in requested:
        local_sigma = _sigma_voxels(2.5, scale_zyx)
        maps["local_contrast"] = _robust_unit(
            norm - gaussian_filter(norm, local_sigma, mode="nearest")
        )
    if "dog_abs" in requested:
        maps["dog_abs"] = _robust_unit(
            dog_response(norm, scale_zyx, replace(baseline, polarity="both"))
        )
    if "dog_bright" in requested:
        maps["dog_bright"] = _robust_unit(
            dog_response(norm, scale_zyx, replace(baseline, polarity="bright"))
        )
    if "dog_dark" in requested:
        maps["dog_dark"] = _robust_unit(
            dog_response(norm, scale_zyx, replace(baseline, polarity="dark"))
        )
    return maps


def _refine_selected_nodes_by_localmax(
    nodes_by_t: list[list[Any]],
    selected_edges: list[Any],
    array: Any,
    attrs: dict[str, Any] | None,
    start: int,
    scale_zyx: tuple[float, float, float],
    route_config: Any,
    score_maps_by_rel_t: dict[int, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any] | None:
    if not getattr(route_config, "enable_localmax_refinement", False):
        return None
    radius_um = float(route_config.localmax_refine_radius_um)
    if not np.isfinite(radius_um) or radius_um <= 0.0:
        raise ValueError(f"localmax_refine_radius_um must be positive and finite, got {radius_um!r}")
    sources = tuple(str(source) for source in route_config.localmax_refine_sources)
    if not sources:
        raise ValueError("localmax_refine_sources must contain at least one score map source")
    raw_center_offsets = getattr(route_config, "localmax_refine_center_offsets_zyx", ())
    if raw_center_offsets:
        try:
            center_offsets_zyx = tuple(
                tuple(int(value) for value in offset)
                for offset in raw_center_offsets
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "localmax_refine_center_offsets_zyx must contain Z/Y/X integer triples"
            ) from exc
    else:
        try:
            center_offsets_zyx = (
                tuple(
                    int(value)
                    for value in getattr(route_config, "localmax_refine_center_offset_zyx", (0, 0, 0))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("localmax_refine_center_offset_zyx must contain exactly three integers") from exc
    if not center_offsets_zyx or any(len(offset) != 3 for offset in center_offsets_zyx):
        raise ValueError("localmax refinement center offsets must contain Z/Y/X integer triples")

    endpoint_ids = {
        int(node_id)
        for edge in selected_edges
        for node_id in (edge.source_id, edge.target_id)
    }
    stats: dict[str, Any] = {
        "enabled": True,
        "sources": list(sources),
        "radius_um": radius_um,
        "center_offset_zyx": list(center_offsets_zyx[-1]),
        "center_offsets_zyx": [list(offset) for offset in center_offsets_zyx],
        "iterations": len(center_offsets_zyx),
        "endpoint_nodes": 0,
        "refined_nodes": 0,
        "mean_abs_offset_zyx": [0.0, 0.0, 0.0],
    }
    if not endpoint_ids:
        return stats

    source_ranks = {source: rank for rank, source in enumerate(sources)}
    offset_abs_sum = np.zeros(3, dtype=np.float64)
    moved_node_ids: set[int] = set()
    stats["endpoint_nodes"] = len(endpoint_ids)
    for rel_t, nodes in enumerate(nodes_by_t):
        endpoint_nodes = [node for node in nodes if int(node.node_id) in endpoint_ids]
        if not endpoint_nodes:
            continue
        volume = np.asarray(array[start + rel_t])
        spatial_max = np.asarray(volume.shape, dtype=np.int64) - 1
        if score_maps_by_rel_t is not None and rel_t in score_maps_by_rel_t:
            score_maps = score_maps_by_rel_t[rel_t]
        else:
            score_maps = _localmax_refinement_maps(volume, attrs, scale_zyx, sources)
        for center_offset_zyx in center_offsets_zyx:
            center_offset = np.asarray(center_offset_zyx, dtype=np.int64)
            for node in endpoint_nodes:
                old_coord = (int(node.z), int(node.y), int(node.x))
                refine_center = tuple(
                    int(value)
                    for value in np.clip(
                        np.asarray(old_coord, dtype=np.int64) - center_offset,
                        0,
                        spatial_max,
                    )
                )
                best_coord: tuple[int, int, int] | None = None
                best_source: str | None = None
                best_value = -np.inf
                for source in sources:
                    candidate = _best_score_coord_in_radius(
                        score_maps[source], refine_center, scale_zyx, radius_um
                    )
                    if candidate is None:
                        continue
                    candidate_coord, value = candidate
                    if float(value) > float(best_value):
                        best_coord = candidate_coord
                        best_source = source
                        best_value = float(value)
                if best_coord is None or best_source is None:
                    continue
                output_coord = tuple(
                    int(value)
                    for value in np.clip(
                        np.asarray(best_coord, dtype=np.int64) + center_offset,
                        0,
                        spatial_max,
                    )
                )
                if output_coord != old_coord:
                    node.z, node.y, node.x = output_coord
                    moved_node_ids.add(int(node.node_id))
                    offset_abs_sum += np.abs(
                        np.asarray(output_coord, dtype=np.float64)
                        - np.asarray(old_coord, dtype=np.float64)
                    )
                node.intensity = float(volume[int(node.z), int(node.y), int(node.x)])
                node.features["localmax_refine_dz"] = float(int(node.z) - old_coord[0])
                node.features["localmax_refine_dy"] = float(int(node.y) - old_coord[1])
                node.features["localmax_refine_dx"] = float(int(node.x) - old_coord[2])
                node.features["localmax_refine_score"] = float(best_value)
                node.features["localmax_refine_source_rank"] = float(source_ranks[best_source])
    stats["refined_nodes"] = len(moved_node_ids)
    if moved_node_ids:
        stats["mean_abs_offset_zyx"] = (
            offset_abs_sum / float(len(moved_node_ids))
        ).round(6).tolist()
    return stats



def _clone_selected_nodes_by_localmax_layers(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    array: Any,
    attrs: dict[str, Any] | None,
    start: int,
    scale_zyx: tuple[float, float, float],
    route_config: Any,
    score_maps_by_rel_t: dict[int, dict[str, np.ndarray]] | None = None,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_localmax_clone_layers", False):
        return emitted_nodes, selected_edges, None
    radius_um = float(route_config.localmax_clone_radius_um)
    if not np.isfinite(radius_um) or radius_um <= 0.0:
        raise ValueError(f"localmax_clone_radius_um must be positive and finite, got {radius_um!r}")
    sources = tuple(str(source) for source in route_config.localmax_clone_sources)
    if not sources:
        raise ValueError("localmax_clone_sources must contain at least one score map source")
    try:
        center_offsets_zyx = tuple(
            tuple(int(value) for value in offset)
            for offset in route_config.localmax_clone_center_offsets_zyx
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("localmax_clone_center_offsets_zyx must contain Z/Y/X integer triples") from exc
    if not center_offsets_zyx or any(len(offset) != 3 for offset in center_offsets_zyx):
        raise ValueError("localmax_clone_center_offsets_zyx must contain at least one Z/Y/X integer triple")

    base_nodes = list(emitted_nodes)
    node_ids = {int(node.node_id) for node in base_nodes}
    source_ranks = {source: rank for rank, source in enumerate(sources)}
    next_node_id = max(node_ids or {0}) + 1
    all_nodes = list(base_nodes)
    all_edges = list(selected_edges)
    cloned_edges = 0
    cloned_nodes = 0
    offset_abs_sum = np.zeros(3, dtype=np.float64)

    nodes_by_t: dict[int, list[CandidateNode]] = {}
    for node in base_nodes:
        nodes_by_t.setdefault(int(node.t), []).append(node)

    for layer_idx, center_offset_zyx in enumerate(center_offsets_zyx):
        center_offset = np.asarray(center_offset_zyx, dtype=np.int64)
        remap: dict[int, int] = {}
        for rel_t, nodes in sorted(nodes_by_t.items()):
            volume = np.asarray(array[start + int(rel_t)])
            spatial_max = np.asarray(volume.shape, dtype=np.int64) - 1
            if score_maps_by_rel_t is not None and rel_t in score_maps_by_rel_t:
                score_maps = score_maps_by_rel_t[rel_t]
            else:
                score_maps = _localmax_refinement_maps(volume, attrs, scale_zyx, sources)
            for node in nodes:
                old_coord = (int(node.z), int(node.y), int(node.x))
                refine_center = tuple(
                    int(value)
                    for value in np.clip(
                        np.asarray(old_coord, dtype=np.int64) - center_offset,
                        0,
                        spatial_max,
                    )
                )
                best_coord: tuple[int, int, int] | None = None
                best_source: str | None = None
                best_value = -np.inf
                for source in sources:
                    candidate = _best_score_coord_in_radius(
                        score_maps[source], refine_center, scale_zyx, radius_um
                    )
                    if candidate is None:
                        continue
                    candidate_coord, value = candidate
                    if float(value) > float(best_value):
                        best_coord = candidate_coord
                        best_source = source
                        best_value = float(value)
                if best_coord is None or best_source is None:
                    continue
                output_coord = tuple(
                    int(value)
                    for value in np.clip(
                        np.asarray(best_coord, dtype=np.int64) + center_offset,
                        0,
                        spatial_max,
                    )
                )
                clone = CandidateNode(
                    dataset=node.dataset,
                    node_id=next_node_id,
                    t=int(node.t),
                    z=int(output_coord[0]),
                    y=int(output_coord[1]),
                    x=int(output_coord[2]),
                    score=float(node.score),
                    sources=tuple((*node.sources, f"localmax_clone_{layer_idx}")),
                    radius_um=float(node.radius_um),
                    uncertainty=float(node.uncertainty),
                    local_density=float(node.local_density),
                    intensity=float(volume[int(output_coord[0]), int(output_coord[1]), int(output_coord[2])]),
                    features={
                        **dict(node.features),
                        "localmax_clone_layer": float(layer_idx),
                        "localmax_clone_source_rank": float(source_ranks[best_source]),
                        "localmax_clone_score": float(best_value),
                    },
                )
                remap[int(node.node_id)] = int(clone.node_id)
                next_node_id += 1
                cloned_nodes += 1
                offset_abs_sum += np.abs(
                    np.asarray(output_coord, dtype=np.float64)
                    - np.asarray(old_coord, dtype=np.float64)
                )
                all_nodes.append(clone)
        existing_edges = {(int(edge.source_id), int(edge.target_id)) for edge in all_edges}
        for edge in selected_edges:
            source_id = remap.get(int(edge.source_id))
            target_id = remap.get(int(edge.target_id))
            if source_id is None or target_id is None or source_id == target_id:
                continue
            pair = (source_id, target_id)
            if pair in existing_edges:
                continue
            existing_edges.add(pair)
            all_edges.append(
                CandidateEdge(
                    source_id=source_id,
                    target_id=target_id,
                    dt=int(edge.dt),
                    distance_um=float(edge.distance_um),
                    cost=float(edge.cost),
                    score=float(edge.score),
                    kind=f"localmax_clone:{edge.kind}",
                    features={**dict(edge.features), "localmax_clone_layer": float(layer_idx)},
                )
            )
            cloned_edges += 1
    stats: dict[str, Any] = {
        "enabled": True,
        "sources": list(sources),
        "radius_um": radius_um,
        "center_offsets_zyx": [list(offset) for offset in center_offsets_zyx],
        "layers": len(center_offsets_zyx),
        "base_nodes": len(base_nodes),
        "base_edges": len(selected_edges),
        "cloned_nodes": cloned_nodes,
        "cloned_edges": cloned_edges,
        "mean_abs_offset_zyx": [0.0, 0.0, 0.0],
    }
    if cloned_nodes:
        stats["mean_abs_offset_zyx"] = (
            offset_abs_sum / float(cloned_nodes)
        ).round(6).tolist()
    return all_nodes, all_edges, stats



def _rewire_localmax_clone_targets(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_localmax_clone_target_rewire", False):
        return emitted_nodes, selected_edges, None
    target_radius_um = float(getattr(route_config, "localmax_clone_rewire_target_radius_um", 4.0))
    source_radius_um = float(getattr(route_config, "localmax_clone_rewire_source_radius_um", 4.0))
    if not np.isfinite(target_radius_um) or target_radius_um <= 0.0:
        raise ValueError(
            f"localmax_clone_rewire_target_radius_um must be positive and finite, got {target_radius_um!r}"
        )
    if not np.isfinite(source_radius_um) or source_radius_um <= 0.0:
        raise ValueError(
            f"localmax_clone_rewire_source_radius_um must be positive and finite, got {source_radius_um!r}"
        )
    if not emitted_nodes or not selected_edges:
        return emitted_nodes, selected_edges, {
            "enabled": True,
            "target_radius_um": target_radius_um,
            "source_radius_um": source_radius_um,
            "before_nodes": len(emitted_nodes),
            "before_edges": len(selected_edges),
            "replacements": 0,
            "after_nodes": len(emitted_nodes),
            "after_edges": len(selected_edges),
            "rewired": [],
        }

    scale = np.asarray(scale_zyx, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale_zyx must contain exactly three positive finite values")

    def _has_source(node: CandidateNode, prefix: str) -> bool:
        return any(str(source).startswith(prefix) for source in node.sources)

    def _is_base_node(node: CandidateNode) -> bool:
        return not (
            _has_source(node, "localmax_clone_")
            or _has_source(node, "offset_clone_")
            or _has_source(node, "endpoint_forward:")
            or _has_source(node, "endpoint_backward:")
        )

    def _node_distance_um(a: CandidateNode, b: CandidateNode) -> float:
        delta = (
            np.asarray(a.coord_zyx, dtype=np.float64)
            - np.asarray(b.coord_zyx, dtype=np.float64)
        ) * scale
        return float(np.sqrt(np.dot(delta, delta)))

    nodes_by_id = {int(node.node_id): node for node in emitted_nodes}
    indeg: Counter[int] = Counter(int(edge.target_id) for edge in selected_edges)
    outdeg: Counter[int] = Counter(int(edge.source_id) for edge in selected_edges)
    by_t: dict[int, list[CandidateNode]] = {}
    for node in emitted_nodes:
        by_t.setdefault(int(node.t), []).append(node)

    existing_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
    used_rewire_targets: set[int] = set()
    rewired_edges: list[CandidateEdge] = []
    replacement_records: list[dict[str, Any]] = []
    for edge in selected_edges:
        source = nodes_by_id.get(int(edge.source_id))
        target = nodes_by_id.get(int(edge.target_id))
        if source is None or target is None:
            rewired_edges.append(edge)
            continue
        if not (_has_source(source, "localmax_clone_") and _has_source(target, "localmax_clone_")):
            rewired_edges.append(edge)
            continue
        if outdeg[int(target.node_id)] != 0:
            rewired_edges.append(edge)
            continue

        candidates: list[tuple[float, float, int, CandidateNode]] = []
        for candidate in by_t.get(int(target.t), []):
            candidate_id = int(candidate.node_id)
            if candidate_id == int(target.node_id) or candidate_id in used_rewire_targets:
                continue
            if not _is_base_node(candidate):
                continue
            if indeg[candidate_id] != 0 or outdeg[candidate_id] == 0:
                continue
            if (int(source.node_id), candidate_id) in existing_pairs:
                continue
            target_distance_um = _node_distance_um(target, candidate)
            source_distance_um = _node_distance_um(source, candidate)
            if target_distance_um <= target_radius_um and source_distance_um <= source_radius_um:
                candidates.append((target_distance_um, source_distance_um, candidate_id, candidate))
        if not candidates:
            rewired_edges.append(edge)
            continue

        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        target_distance_um, source_distance_um, candidate_id, candidate = candidates[0]
        features = dict(edge.features)
        features.update(
            {
                "localmax_clone_target_rewire": 1.0,
                "localmax_clone_rewire_old_target_id": float(int(target.node_id)),
                "localmax_clone_rewire_target_distance_um": float(target_distance_um),
                "localmax_clone_rewire_source_distance_um": float(source_distance_um),
            }
        )
        rewired_edges.append(
            replace(
                edge,
                target_id=candidate_id,
                distance_um=float(source_distance_um),
                kind=f"localmax_rewire:{edge.kind}",
                features=features,
            )
        )
        used_rewire_targets.add(candidate_id)
        replacement_records.append(
            {
                "source_id": int(source.node_id),
                "old_target_id": int(target.node_id),
                "new_target_id": candidate_id,
                "old_target_zyx": list(target.coord_zyx),
                "new_target_zyx": list(candidate.coord_zyx),
                "target_distance_um": float(target_distance_um),
                "source_distance_um": float(source_distance_um),
            }
        )

    if not replacement_records:
        return emitted_nodes, selected_edges, {
            "enabled": True,
            "target_radius_um": target_radius_um,
            "source_radius_um": source_radius_um,
            "before_nodes": len(emitted_nodes),
            "before_edges": len(selected_edges),
            "replacements": 0,
            "after_nodes": len(emitted_nodes),
            "after_edges": len(selected_edges),
            "rewired": [],
        }

    deduped_edges: list[CandidateEdge] = []
    seen_pairs: set[tuple[int, int]] = set()
    for edge in rewired_edges:
        pair = (int(edge.source_id), int(edge.target_id))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped_edges.append(edge)
    used_node_ids = {
        int(node_id)
        for edge in deduped_edges
        for node_id in (edge.source_id, edge.target_id)
    }
    endpoint_nodes = [node for node in emitted_nodes if int(node.node_id) in used_node_ids]
    return endpoint_nodes, deduped_edges, {
        "enabled": True,
        "target_radius_um": target_radius_um,
        "source_radius_um": source_radius_um,
        "before_nodes": len(emitted_nodes),
        "before_edges": len(selected_edges),
        "replacements": len(replacement_records),
        "after_nodes": len(endpoint_nodes),
        "after_edges": len(deduped_edges),
        "rewired": replacement_records,
    }


def _rewire_base_chain_targets(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_base_chain_target_rewire", False):
        return emitted_nodes, selected_edges, None
    target_radius_um = float(getattr(route_config, "base_chain_rewire_target_radius_um", 5.5))
    source_radius_um = float(getattr(route_config, "base_chain_rewire_source_radius_um", 4.0))
    if not np.isfinite(target_radius_um) or target_radius_um <= 0.0:
        raise ValueError(
            f"base_chain_rewire_target_radius_um must be positive and finite, got {target_radius_um!r}"
        )
    if not np.isfinite(source_radius_um) or source_radius_um <= 0.0:
        raise ValueError(
            f"base_chain_rewire_source_radius_um must be positive and finite, got {source_radius_um!r}"
        )
    if not emitted_nodes or not selected_edges:
        return emitted_nodes, selected_edges, {
            "enabled": True,
            "target_radius_um": target_radius_um,
            "source_radius_um": source_radius_um,
            "before_nodes": len(emitted_nodes),
            "before_edges": len(selected_edges),
            "replacements": 0,
            "after_nodes": len(emitted_nodes),
            "after_edges": len(selected_edges),
            "rewired": [],
        }

    scale = np.asarray(scale_zyx, dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale_zyx must contain exactly three positive finite values")

    def _has_source(node: CandidateNode, prefix: str) -> bool:
        return any(str(source).startswith(prefix) for source in node.sources)

    def _is_base_node(node: CandidateNode) -> bool:
        return not (
            _has_source(node, "localmax_clone_")
            or _has_source(node, "offset_clone_")
            or _has_source(node, "endpoint_forward:")
            or _has_source(node, "endpoint_backward:")
        )

    def _node_distance_um(a: CandidateNode, b: CandidateNode) -> float:
        delta = (
            np.asarray(a.coord_zyx, dtype=np.float64)
            - np.asarray(b.coord_zyx, dtype=np.float64)
        ) * scale
        return float(np.sqrt(np.dot(delta, delta)))

    nodes_by_id = {int(node.node_id): node for node in emitted_nodes}
    indeg: Counter[int] = Counter(int(edge.target_id) for edge in selected_edges)
    outdeg: Counter[int] = Counter(int(edge.source_id) for edge in selected_edges)
    by_t: dict[int, list[CandidateNode]] = {}
    for node in emitted_nodes:
        by_t.setdefault(int(node.t), []).append(node)

    existing_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
    used_rewire_targets: set[int] = set()
    rewired_edges: list[CandidateEdge] = []
    replacement_records: list[dict[str, Any]] = []
    for edge in selected_edges:
        source = nodes_by_id.get(int(edge.source_id))
        target = nodes_by_id.get(int(edge.target_id))
        if source is None or target is None:
            rewired_edges.append(edge)
            continue
        if not (_is_base_node(source) and _is_base_node(target)):
            rewired_edges.append(edge)
            continue

        candidates: list[tuple[float, float, int, CandidateNode]] = []
        for candidate in by_t.get(int(target.t), []):
            candidate_id = int(candidate.node_id)
            if candidate_id == int(target.node_id) or candidate_id in used_rewire_targets:
                continue
            if not _is_base_node(candidate):
                continue
            if indeg[candidate_id] != 0 or outdeg[candidate_id] == 0:
                continue
            if (int(source.node_id), candidate_id) in existing_pairs:
                continue
            target_distance_um = _node_distance_um(target, candidate)
            source_distance_um = _node_distance_um(source, candidate)
            if target_distance_um <= target_radius_um and source_distance_um <= source_radius_um:
                candidates.append((target_distance_um, source_distance_um, candidate_id, candidate))
        if not candidates:
            rewired_edges.append(edge)
            continue

        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        target_distance_um, source_distance_um, candidate_id, candidate = candidates[0]
        features = dict(edge.features)
        features.update(
            {
                "base_chain_target_rewire": 1.0,
                "base_chain_rewire_old_target_id": float(int(target.node_id)),
                "base_chain_rewire_target_distance_um": float(target_distance_um),
                "base_chain_rewire_source_distance_um": float(source_distance_um),
            }
        )
        rewired_edges.append(
            replace(
                edge,
                target_id=candidate_id,
                distance_um=float(source_distance_um),
                kind=f"base_chain_rewire:{edge.kind}",
                features=features,
            )
        )
        used_rewire_targets.add(candidate_id)
        replacement_records.append(
            {
                "source_id": int(source.node_id),
                "old_target_id": int(target.node_id),
                "new_target_id": candidate_id,
                "old_target_zyx": list(target.coord_zyx),
                "new_target_zyx": list(candidate.coord_zyx),
                "target_distance_um": float(target_distance_um),
                "source_distance_um": float(source_distance_um),
            }
        )

    if not replacement_records:
        return emitted_nodes, selected_edges, {
            "enabled": True,
            "target_radius_um": target_radius_um,
            "source_radius_um": source_radius_um,
            "before_nodes": len(emitted_nodes),
            "before_edges": len(selected_edges),
            "replacements": 0,
            "after_nodes": len(emitted_nodes),
            "after_edges": len(selected_edges),
            "rewired": [],
        }

    deduped_edges: list[CandidateEdge] = []
    seen_pairs: set[tuple[int, int]] = set()
    for edge in rewired_edges:
        pair = (int(edge.source_id), int(edge.target_id))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped_edges.append(edge)
    used_node_ids = {
        int(node_id)
        for edge in deduped_edges
        for node_id in (edge.source_id, edge.target_id)
    }
    endpoint_nodes = [node for node in emitted_nodes if int(node.node_id) in used_node_ids]
    return endpoint_nodes, deduped_edges, {
        "enabled": True,
        "target_radius_um": target_radius_um,
        "source_radius_um": source_radius_um,
        "before_nodes": len(emitted_nodes),
        "before_edges": len(selected_edges),
        "replacements": len(replacement_records),
        "after_nodes": len(endpoint_nodes),
        "after_edges": len(deduped_edges),
        "rewired": replacement_records,
    }


def _prune_unlinked_submission_nodes(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
) -> tuple[list[CandidateNode], dict[str, Any]]:
    """Drop final nodes that cannot contribute to the submitted edge graph."""
    if not selected_edges:
        return emitted_nodes, {
            "enabled": True,
            "before_nodes": len(emitted_nodes),
            "after_nodes": len(emitted_nodes),
            "removed_nodes": 0,
            "edges": 0,
        }
    used_ids = {
        int(node_id)
        for edge in selected_edges
        for node_id in (edge.source_id, edge.target_id)
    }
    pruned_nodes = [node for node in emitted_nodes if int(node.node_id) in used_ids]
    return pruned_nodes, {
        "enabled": True,
        "before_nodes": len(emitted_nodes),
        "after_nodes": len(pruned_nodes),
        "removed_nodes": len(emitted_nodes) - len(pruned_nodes),
        "edges": len(selected_edges),
    }


def _prune_submission_components(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    route_config: Any,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    """Drop whole submitted components under route-gated node-penalty rules."""
    if not getattr(route_config, "enable_component_pruning", False):
        return emitted_nodes, selected_edges, None
    min_edges = int(getattr(route_config, "component_prune_min_edges", 0) or 0)
    min_x = getattr(route_config, "component_prune_min_x", None)
    max_x = getattr(route_config, "component_prune_max_x", None)
    x_offset_only = bool(getattr(route_config, "component_prune_x_offset_only", False))
    drop_one_edge_offset = bool(getattr(route_config, "component_prune_drop_one_edge_offset", False))
    if min_edges < 0:
        raise ValueError("component_prune_min_edges must be non-negative")
    if min_x is not None:
        min_x = int(min_x)
    if max_x is not None:
        max_x = int(max_x)
    if min_x is not None and max_x is not None and min_x > max_x:
        raise ValueError("component_prune_min_x must not exceed component_prune_max_x")

    nodes_by_id = {int(node.node_id): node for node in emitted_nodes}
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in nodes_by_id}
    for edge in selected_edges:
        source_id = int(edge.source_id)
        target_id = int(edge.target_id)
        if source_id not in adjacency or target_id not in adjacency:
            continue
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)

    keep_ids: set[int] = set()
    component_count = 0
    removed_components = 0
    removed_nodes = 0
    removed_edges = 0
    removal_reasons: Counter[str] = Counter()
    unseen = set(adjacency)
    while unseen:
        first = min(unseen)
        stack = [first]
        component_ids: set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component_ids:
                continue
            component_ids.add(node_id)
            unseen.discard(node_id)
            stack.extend(sorted(adjacency.get(node_id, set()) - component_ids, reverse=True))
        component_edges = [
            edge
            for edge in selected_edges
            if int(edge.source_id) in component_ids and int(edge.target_id) in component_ids
        ]
        component_nodes = [nodes_by_id[node_id] for node_id in component_ids]
        xs = [int(node.x) for node in component_nodes]
        offset_nodes = sum(
            1
            for node in component_nodes
            if any(str(source).startswith("offset_clone_") for source in node.sources)
        )
        component_count += 1
        reasons: list[str] = []
        if min_edges and len(component_edges) < min_edges:
            reasons.append("min_edges")
        apply_x_prune = not x_offset_only or offset_nodes > 0
        if apply_x_prune and min_x is not None and xs and min(xs) < min_x:
            reasons.append("min_x")
        if apply_x_prune and max_x is not None and xs and max(xs) > max_x:
            reasons.append("max_x")
        if drop_one_edge_offset and len(component_edges) <= 1 and offset_nodes > 0:
            reasons.append("one_edge_offset")

        if reasons:
            removed_components += 1
            removed_nodes += len(component_nodes)
            removed_edges += len(component_edges)
            for reason in reasons:
                removal_reasons[reason] += 1
            continue
        keep_ids.update(component_ids)

    filtered_edges = [
        edge
        for edge in selected_edges
        if int(edge.source_id) in keep_ids and int(edge.target_id) in keep_ids
    ]
    used_ids = {
        int(node_id)
        for edge in filtered_edges
        for node_id in (edge.source_id, edge.target_id)
    }
    filtered_nodes = [node for node in emitted_nodes if int(node.node_id) in used_ids]
    return filtered_nodes, filtered_edges, {
        "enabled": True,
        "min_edges": min_edges,
        "min_x": min_x,
        "max_x": max_x,
        "x_offset_only": x_offset_only,
        "drop_one_edge_offset": drop_one_edge_offset,
        "before_nodes": len(emitted_nodes),
        "before_edges": len(selected_edges),
        "components": component_count,
        "removed_components": removed_components,
        "removed_nodes": removed_nodes,
        "removed_edges": removed_edges,
        "removal_reasons": dict(removal_reasons),
        "after_nodes": len(filtered_nodes),
        "after_edges": len(filtered_edges),
    }



def _clone_selected_nodes_by_offset_layers(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    array: Any,
    start: int,
    route_config: Any,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_offset_clone_layers", False):
        return emitted_nodes, selected_edges, None
    try:
        offsets_zyx = tuple(
            tuple(int(value) for value in offset)
            for offset in route_config.offset_clone_offsets_zyx
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("offset_clone_offsets_zyx must contain Z/Y/X integer triples") from exc
    if not offsets_zyx or any(len(offset) != 3 for offset in offsets_zyx):
        raise ValueError("offset_clone_offsets_zyx must contain at least one Z/Y/X integer triple")
    skip_clipped = bool(getattr(route_config, "offset_clone_skip_clipped", False))
    filter_source_mode = str(getattr(route_config, "offset_clone_filter_source_mode", "any") or "any").strip().lower()
    filter_min_timepoints = int(getattr(route_config, "offset_clone_filter_min_component_timepoints", 0) or 0)
    filter_component_max_x = getattr(route_config, "offset_clone_filter_component_max_x", None)
    if filter_source_mode not in {"any", "contrast_only", "raw_only", "raw_contrast", "contrast_any", "raw_any"}:
        raise ValueError(f"unknown offset_clone_filter_source_mode: {filter_source_mode!r}")
    if filter_min_timepoints < 0:
        raise ValueError("offset_clone_filter_min_component_timepoints must be non-negative")
    if filter_component_max_x is not None:
        filter_component_max_x = int(filter_component_max_x)
        if filter_component_max_x < 0:
            raise ValueError("offset_clone_filter_component_max_x must be non-negative")
    filter_enabled = (
        filter_source_mode != "any"
        or filter_min_timepoints > 0
        or filter_component_max_x is not None
    )

    base_nodes = list(emitted_nodes)
    all_nodes = list(base_nodes)
    all_edges = list(selected_edges)
    node_ids = {int(node.node_id) for node in base_nodes}
    next_node_id = max(node_ids or {0}) + 1
    spatial_max = np.asarray(array.shape[1:4], dtype=np.int64) - 1
    offset_abs_sum = np.zeros(3, dtype=np.float64)
    clipped_nodes = 0
    skipped_clipped_nodes = 0
    cloned_nodes = 0
    cloned_edges = 0

    def _volume_at_t(t: int) -> np.ndarray:
        absolute_t = int(t)
        if 0 <= absolute_t < int(array.shape[0]):
            return np.asarray(array[absolute_t])
        return np.asarray(array[int(start) + absolute_t])

    def _offset_source_mode(node: CandidateNode) -> str:
        has_raw = any(str(source).startswith("tile_raw@") for source in node.sources)
        has_contrast = any(str(source).startswith("tile_contrast@") for source in node.sources)
        if has_raw and has_contrast:
            return "raw_contrast"
        if has_contrast:
            return "contrast_only"
        if has_raw:
            return "raw_only"
        return "other"

    def _source_mode_ok(node: CandidateNode) -> bool:
        mode = _offset_source_mode(node)
        if filter_source_mode == "any":
            return True
        if filter_source_mode == "contrast_any":
            return mode in {"contrast_only", "raw_contrast"}
        if filter_source_mode == "raw_any":
            return mode in {"raw_only", "raw_contrast"}
        return mode == filter_source_mode

    def _filter_layer(
        layer_nodes: list[CandidateNode],
        layer_edges: list[CandidateEdge],
    ) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any]]:
        if not filter_enabled:
            return layer_nodes, layer_edges, {
                "enabled": False,
                "kept_nodes": len(layer_nodes),
                "kept_edges": len(layer_edges),
            }
        nodes_by_id = {int(node.node_id): node for node in layer_nodes}
        adjacency: dict[int, set[int]] = {int(node.node_id): set() for node in layer_nodes}
        for edge in layer_edges:
            source_id = int(edge.source_id)
            target_id = int(edge.target_id)
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        component_info: dict[int, dict[str, Any]] = {}
        unseen = set(adjacency)
        while unseen:
            first = min(unseen)
            stack = [first]
            component: set[int] = set()
            while stack:
                node_id = stack.pop()
                if node_id in component:
                    continue
                component.add(node_id)
                unseen.discard(node_id)
                stack.extend(sorted(adjacency.get(node_id, set()) - component, reverse=True))
            times = {int(nodes_by_id[node_id].t) for node_id in component}
            max_x = max(int(nodes_by_id[node_id].x) for node_id in component)
            info = {"timepoints": len(times), "max_x": max_x}
            for node_id in component:
                component_info[node_id] = info

        candidate_keep_ids: set[int] = set()
        dropped_by_source = 0
        dropped_by_timepoints = 0
        dropped_by_x = 0
        for node in layer_nodes:
            node_id = int(node.node_id)
            info = component_info.get(node_id, {"timepoints": 1, "max_x": int(node.x)})
            if not _source_mode_ok(node):
                dropped_by_source += 1
                continue
            if int(info["timepoints"]) < filter_min_timepoints:
                dropped_by_timepoints += 1
                continue
            if filter_component_max_x is not None and int(info["max_x"]) > int(filter_component_max_x):
                dropped_by_x += 1
                continue
            candidate_keep_ids.add(node_id)

        filtered_edges = [
            edge
            for edge in layer_edges
            if int(edge.source_id) in candidate_keep_ids and int(edge.target_id) in candidate_keep_ids
        ]
        used_ids = {
            int(node_id)
            for edge in filtered_edges
            for node_id in (edge.source_id, edge.target_id)
        }
        filtered_nodes = [node for node in layer_nodes if int(node.node_id) in used_ids]
        return filtered_nodes, filtered_edges, {
            "enabled": True,
            "source_mode": filter_source_mode,
            "min_component_timepoints": filter_min_timepoints,
            "component_max_x": filter_component_max_x,
            "before_nodes": len(layer_nodes),
            "before_edges": len(layer_edges),
            "candidate_kept_nodes": len(candidate_keep_ids),
            "kept_nodes": len(filtered_nodes),
            "kept_edges": len(filtered_edges),
            "dropped_by_source": dropped_by_source,
            "dropped_by_timepoints": dropped_by_timepoints,
            "dropped_by_x": dropped_by_x,
        }

    layer_filter_stats: list[dict[str, Any]] = []
    for layer_idx, offset_zyx in enumerate(offsets_zyx):
        offset = np.asarray(offset_zyx, dtype=np.int64)
        remap: dict[int, int] = {}
        layer_nodes: list[CandidateNode] = []
        for node in base_nodes:
            old_coord = np.asarray(node.coord_zyx, dtype=np.int64)
            raw_coord = old_coord + offset
            output_coord = np.clip(raw_coord, 0, spatial_max).astype(np.int64)
            clipped = bool(np.any(output_coord != raw_coord))
            if clipped:
                clipped_nodes += 1
                if skip_clipped:
                    skipped_clipped_nodes += 1
                    continue
            volume = _volume_at_t(int(node.t))
            clone = CandidateNode(
                dataset=node.dataset,
                node_id=next_node_id,
                t=int(node.t),
                z=int(output_coord[0]),
                y=int(output_coord[1]),
                x=int(output_coord[2]),
                score=float(node.score),
                sources=tuple((*node.sources, f"offset_clone_{layer_idx}")),
                radius_um=float(node.radius_um),
                uncertainty=float(node.uncertainty),
                local_density=float(node.local_density),
                intensity=float(volume[int(output_coord[0]), int(output_coord[1]), int(output_coord[2])]),
                features={
                    **dict(node.features),
                    "offset_clone_layer": float(layer_idx),
                    "offset_clone_dz": float(int(offset[0])),
                    "offset_clone_dy": float(int(offset[1])),
                    "offset_clone_dx": float(int(offset[2])),
                    "offset_clone_clipped": 1.0 if clipped else 0.0,
                    "offset_clone_skip_clipped": 1.0 if skip_clipped else 0.0,
                },
            )
            remap[int(node.node_id)] = int(clone.node_id)
            next_node_id += 1
            cloned_nodes += 1
            offset_abs_sum += np.abs(output_coord.astype(np.float64) - old_coord.astype(np.float64))
            layer_nodes.append(clone)

        existing_edges = {(int(edge.source_id), int(edge.target_id)) for edge in all_edges}
        layer_edges: list[CandidateEdge] = []
        for edge in selected_edges:
            source_id = remap.get(int(edge.source_id))
            target_id = remap.get(int(edge.target_id))
            if source_id is None or target_id is None or source_id == target_id:
                continue
            pair = (source_id, target_id)
            if pair in existing_edges:
                continue
            existing_edges.add(pair)
            layer_edges.append(
                CandidateEdge(
                    source_id=source_id,
                    target_id=target_id,
                    dt=int(edge.dt),
                    distance_um=float(edge.distance_um),
                    cost=float(edge.cost),
                    score=float(edge.score),
                    kind=f"offset_clone:{edge.kind}",
                    features={
                        **dict(edge.features),
                        "offset_clone_layer": float(layer_idx),
                        "offset_clone_dz": float(int(offset[0])),
                        "offset_clone_dy": float(int(offset[1])),
                        "offset_clone_dx": float(int(offset[2])),
                        "offset_clone_skip_clipped": 1.0 if skip_clipped else 0.0,
                    },
                )
            )
            cloned_edges += 1
        kept_layer_nodes, kept_layer_edges, layer_stats = _filter_layer(layer_nodes, layer_edges)
        layer_stats["layer"] = int(layer_idx)
        layer_filter_stats.append(layer_stats)
        all_nodes.extend(kept_layer_nodes)
        all_edges.extend(kept_layer_edges)

    stats: dict[str, Any] = {
        "enabled": True,
        "skip_clipped": skip_clipped,
        "offsets_zyx": [list(offset) for offset in offsets_zyx],
        "layers": len(offsets_zyx),
        "base_nodes": len(base_nodes),
        "base_edges": len(selected_edges),
        "cloned_nodes": cloned_nodes,
        "cloned_edges": cloned_edges,
        "clipped_nodes": clipped_nodes,
        "skipped_clipped_nodes": skipped_clipped_nodes,
        "filter": {
            "enabled": filter_enabled,
            "source_mode": filter_source_mode,
            "min_component_timepoints": filter_min_timepoints,
            "component_max_x": filter_component_max_x,
            "layers": layer_filter_stats,
        },
        "mean_abs_offset_zyx": [0.0, 0.0, 0.0],
    }
    if cloned_nodes:
        stats["mean_abs_offset_zyx"] = (
            offset_abs_sum / float(cloned_nodes)
        ).round(6).tolist()
    return all_nodes, all_edges, stats


def _add_endpoint_forward_clones(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    array: Any,
    attrs: dict[str, Any] | None,
    start: int,
    scale_zyx: tuple[float, float, float],
    route_config: Any,
    score_maps_by_rel_t: dict[int, dict[str, np.ndarray]] | None = None,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_endpoint_forward_clones", False):
        return emitted_nodes, selected_edges, None
    radius_um = float(route_config.endpoint_forward_radius_um)
    if not np.isfinite(radius_um) or radius_um <= 0.0:
        raise ValueError(f"endpoint_forward_radius_um must be positive and finite, got {radius_um!r}")
    sources = tuple(str(source) for source in route_config.endpoint_forward_sources)
    if not sources:
        raise ValueError("endpoint_forward_sources must contain at least one score map source")
    max_per_t = int(route_config.endpoint_forward_max_per_t)
    if max_per_t < 1:
        raise ValueError("endpoint_forward_max_per_t must be at least one")

    base_nodes = list(emitted_nodes)
    all_nodes = list(base_nodes)
    all_edges = list(selected_edges)
    nodes_by_id = {int(node.node_id): node for node in all_nodes}
    next_node_id = max(nodes_by_id or {0}) + 1
    indeg = {int(edge.target_id): int(edge.source_id) for edge in all_edges}
    outdeg = {int(edge.source_id): int(edge.target_id) for edge in all_edges}
    existing_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in all_edges}
    source_ranks = {source: rank for rank, source in enumerate(sources)}

    nodes_by_t: dict[int, list[CandidateNode]] = {}
    for node in base_nodes:
        nodes_by_t.setdefault(int(node.t), []).append(node)
    max_rel_t = max(nodes_by_t.keys(), default=0)
    new_per_t: dict[int, int] = {}
    cloned_nodes = 0
    cloned_edges = 0
    offset_abs_sum = np.zeros(3, dtype=np.float64)

    def _prediction_center(node: CandidateNode) -> tuple[int, int, int]:
        current = np.asarray(node.coord_zyx, dtype=np.float64)
        previous_id = indeg.get(int(node.node_id))
        if previous_id is None or previous_id not in nodes_by_id:
            return tuple(int(value) for value in current)
        previous = np.asarray(nodes_by_id[previous_id].coord_zyx, dtype=np.float64)
        predicted = current + (current - previous)
        return tuple(int(round(value)) for value in predicted)

    for rel_t in range(max_rel_t):
        target_rel_t = rel_t + 1
        if new_per_t.get(target_rel_t, 0) >= max_per_t:
            continue
        volume = np.asarray(array[start + target_rel_t])
        spatial_max = np.asarray(volume.shape, dtype=np.int64) - 1
        if score_maps_by_rel_t is not None and target_rel_t in score_maps_by_rel_t:
            score_maps = score_maps_by_rel_t[target_rel_t]
        else:
            score_maps = _localmax_refinement_maps(volume, attrs, scale_zyx, sources)
        for node in sorted(nodes_by_t.get(rel_t, []), key=lambda value: int(value.node_id)):
            node_id = int(node.node_id)
            if node_id in outdeg:
                continue
            if bool(route_config.endpoint_forward_requires_history) and node_id not in indeg:
                continue
            if new_per_t.get(target_rel_t, 0) >= max_per_t:
                break
            center = tuple(
                int(value)
                for value in np.clip(
                    np.asarray(_prediction_center(node), dtype=np.int64),
                    0,
                    spatial_max,
                )
            )
            best_coord: tuple[int, int, int] | None = None
            best_source: str | None = None
            best_value = -np.inf
            for source in sources:
                candidate = _best_score_coord_in_radius(
                    score_maps[source], center, scale_zyx, radius_um
                )
                if candidate is None:
                    continue
                candidate_coord, value = candidate
                if float(value) > float(best_value):
                    best_coord = candidate_coord
                    best_source = source
                    best_value = float(value)
            if best_coord is None or best_source is None:
                continue
            distance_um = float(
                np.linalg.norm(
                    (
                        np.asarray(node.coord_zyx, dtype=np.float64)
                        - np.asarray(best_coord, dtype=np.float64)
                    )
                    * np.asarray(scale_zyx, dtype=np.float64)
                )
            )
            if distance_um > max(radius_um * 1.35, radius_um + 1.0):
                continue
            clone = CandidateNode(
                dataset=node.dataset,
                node_id=next_node_id,
                t=int(target_rel_t),
                z=int(best_coord[0]),
                y=int(best_coord[1]),
                x=int(best_coord[2]),
                score=float(node.score),
                sources=tuple((*node.sources, f"endpoint_forward:{best_source}")),
                radius_um=float(node.radius_um),
                uncertainty=float(node.uncertainty),
                local_density=float(node.local_density),
                intensity=float(volume[int(best_coord[0]), int(best_coord[1]), int(best_coord[2])]),
                features={
                    **dict(node.features),
                    "endpoint_forward_score": float(best_value),
                    "endpoint_forward_source_rank": float(source_ranks[best_source]),
                },
            )
            next_node_id += 1
            pair = (node_id, int(clone.node_id))
            if pair in existing_pairs:
                continue
            all_nodes.append(clone)
            nodes_by_id[int(clone.node_id)] = clone
            all_edges.append(
                CandidateEdge(
                    source_id=pair[0],
                    target_id=pair[1],
                    dt=1,
                    distance_um=distance_um,
                    cost=0.25,
                    score=0.75,
                    kind="endpoint_forward",
                    features={"endpoint_forward": 1.0},
                )
            )
            existing_pairs.add(pair)
            outdeg[pair[0]] = pair[1]
            indeg[pair[1]] = pair[0]
            new_per_t[target_rel_t] = new_per_t.get(target_rel_t, 0) + 1
            cloned_nodes += 1
            cloned_edges += 1
            offset_abs_sum += np.abs(
                np.asarray(best_coord, dtype=np.float64)
                - np.asarray(node.coord_zyx, dtype=np.float64)
            )

    stats: dict[str, Any] = {
        "enabled": True,
        "sources": list(sources),
        "radius_um": radius_um,
        "max_per_t": max_per_t,
        "requires_history": bool(route_config.endpoint_forward_requires_history),
        "base_nodes": len(base_nodes),
        "base_edges": len(selected_edges),
        "cloned_nodes": cloned_nodes,
        "cloned_edges": cloned_edges,
        "new_per_t": {str(key): int(value) for key, value in sorted(new_per_t.items())},
        "mean_abs_offset_zyx": [0.0, 0.0, 0.0],
    }
    if cloned_nodes:
        stats["mean_abs_offset_zyx"] = (
            offset_abs_sum / float(cloned_nodes)
        ).round(6).tolist()
    return all_nodes, all_edges, stats



def _add_endpoint_backward_clones(
    emitted_nodes: list[CandidateNode],
    selected_edges: list[CandidateEdge],
    array: Any,
    start: int,
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[CandidateNode], list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_endpoint_backward_clones", False):
        return emitted_nodes, selected_edges, None
    max_edge_um = float(route_config.endpoint_backward_max_edge_um)
    min_edge_um = float(getattr(route_config, "endpoint_backward_min_edge_um", 0.0))
    duplicate_um = float(route_config.endpoint_backward_duplicate_um)
    max_per_t = int(route_config.endpoint_backward_max_per_t)
    axis_mode = str(route_config.endpoint_backward_axis_mode or "x_preferred").strip().lower()
    if not np.isfinite(max_edge_um) or max_edge_um <= 0.0:
        raise ValueError("endpoint_backward_max_edge_um must be positive and finite")
    if not np.isfinite(min_edge_um) or min_edge_um < 0.0:
        raise ValueError("endpoint_backward_min_edge_um must be finite and non-negative")
    if min_edge_um > max_edge_um:
        raise ValueError("endpoint_backward_min_edge_um must not exceed endpoint_backward_max_edge_um")
    if not np.isfinite(duplicate_um) or duplicate_um < 0.0:
        raise ValueError("endpoint_backward_duplicate_um must be finite and non-negative")
    if max_per_t < 1:
        raise ValueError("endpoint_backward_max_per_t must be at least one")
    if axis_mode not in {"any", "x", "x_preferred", "interior_dup"}:
        raise ValueError(f"unknown endpoint_backward_axis_mode: {axis_mode!r}")

    all_nodes = list(emitted_nodes)
    all_edges = list(selected_edges)
    nodes_by_id = {int(node.node_id): node for node in all_nodes}
    next_node_id = max(nodes_by_id or {0}) + 1
    incoming = {int(edge.target_id): int(edge.source_id) for edge in all_edges}
    outgoing: dict[int, list[int]] = {}
    for edge in all_edges:
        outgoing.setdefault(int(edge.source_id), []).append(int(edge.target_id))
    spatial_max = np.asarray(array.shape[1:4], dtype=np.int64) - 1
    scale = np.asarray(scale_zyx, dtype=np.float64)
    interior_mode = axis_mode == "interior_dup"

    def _distance_um(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)) * scale))

    def _is_x_boundary(low: np.ndarray, high: np.ndarray, clipped: np.ndarray) -> bool:
        return bool(low[2] or high[2] or clipped[2] <= 0 or clipped[2] >= spatial_max[2])

    def _duplicate_distance_um(coord: np.ndarray, t: int, nodes: list[CandidateNode]) -> float:
        distances = [
            _distance_um(np.asarray(node.coord_zyx, dtype=np.float64), coord.astype(np.float64))
            for node in nodes
            if int(node.t) == int(t)
        ]
        return float(min(distances)) if distances else 999.0

    def _volume_at_t(t: int) -> np.ndarray:
        absolute_t = int(t)
        if 0 <= absolute_t < int(array.shape[0]):
            return np.asarray(array[absolute_t])
        return np.asarray(array[int(start) + absolute_t])

    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for source_id, target_ids in sorted(outgoing.items()):
        if source_id in incoming or len(target_ids) != 1:
            continue
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(int(target_ids[0]))
        if source is None or target is None:
            continue
        source_t = int(source.t)
        target_t = int(target.t)
        if source_t <= 0 or target_t != source_t + 1:
            continue
        source_coord = np.asarray(source.coord_zyx, dtype=np.float64)
        target_coord = np.asarray(target.coord_zyx, dtype=np.float64)
        edge_um = _distance_um(source_coord, target_coord)
        if edge_um < min_edge_um or edge_um > max_edge_um:
            continue
        predicted = source_coord - (target_coord - source_coord)
        clipped = np.clip(np.rint(predicted).astype(np.int64), 0, spatial_max)
        low = predicted < 0
        high = predicted > spatial_max
        boundary = bool(np.any(low | high) or np.any(clipped <= 0) or np.any(clipped >= spatial_max))
        if interior_mode:
            if boundary:
                continue
        elif not boundary:
            continue
        x_boundary = _is_x_boundary(low, high, clipped)
        if axis_mode == "x" and not x_boundary:
            continue
        back_um = _distance_um(clipped.astype(np.float64), source_coord)
        if back_um > max_edge_um:
            continue
        prev_t = source_t - 1
        duplicate_initial_um = _duplicate_distance_um(clipped, prev_t, all_nodes)
        if interior_mode and duplicate_initial_um <= duplicate_um:
            continue
        oob_vox = np.maximum(-predicted, 0.0) + np.maximum(predicted - spatial_max, 0.0)
        oob_um = float(np.linalg.norm(oob_vox * scale))
        if interior_mode:
            rank_key = (-duplicate_initial_um, -edge_um, float(source_id))
        else:
            x_rank = -1.0 if axis_mode == "x_preferred" and x_boundary else 0.0
            rank_key = (x_rank, -oob_um, -edge_um, float(source_id))
        ranked.append(
            (
                rank_key,
                {
                    "source": source,
                    "target": target,
                    "prev_t": prev_t,
                    "coord": tuple(int(v) for v in clipped),
                    "edge_um": edge_um,
                    "back_um": back_um,
                    "duplicate_um": duplicate_initial_um,
                    "oob_um": oob_um,
                    "x_boundary": x_boundary,
                    "predicted": [float(v) for v in predicted],
                },
            )
        )

    ranked.sort(key=lambda row: row[0])
    new_per_t: dict[int, int] = {}
    cloned_nodes = 0
    cloned_edges = 0
    details: list[dict[str, Any]] = []
    for _rank_key, item in ranked:
        prev_t = int(item["prev_t"])
        if new_per_t.get(prev_t, 0) >= max_per_t:
            continue
        coord = np.asarray(item["coord"], dtype=np.int64)
        duplicate_distance_um = _duplicate_distance_um(coord, prev_t, all_nodes)
        if duplicate_distance_um <= duplicate_um:
            continue
        source = item["source"]
        target = item["target"]
        volume = _volume_at_t(prev_t)
        clone = CandidateNode(
            dataset=source.dataset,
            node_id=next_node_id,
            t=prev_t,
            z=int(coord[0]),
            y=int(coord[1]),
            x=int(coord[2]),
            score=float(source.score),
            sources=tuple((*source.sources, "endpoint_backward:pred")),
            radius_um=float(source.radius_um),
            uncertainty=float(source.uncertainty),
            local_density=float(source.local_density),
            intensity=float(volume[int(coord[0]), int(coord[1]), int(coord[2])]),
            features={
                **dict(source.features),
                "endpoint_backward": 1.0,
                "endpoint_backward_duplicate_um": float(duplicate_distance_um),
                "endpoint_backward_oob_um": float(item["oob_um"]),
                "endpoint_backward_x_boundary": 1.0 if bool(item["x_boundary"]) else 0.0,
            },
        )
        next_node_id += 1
        pair = (int(clone.node_id), int(source.node_id))
        all_nodes.append(clone)
        nodes_by_id[int(clone.node_id)] = clone
        all_edges.append(
            CandidateEdge(
                source_id=pair[0],
                target_id=pair[1],
                dt=1,
                distance_um=float(item["back_um"]),
                cost=0.25,
                score=0.75,
                kind="endpoint_backward",
                features={
                    "endpoint_backward": 1.0,
                    "endpoint_backward_duplicate_um": float(duplicate_distance_um),
                    "endpoint_backward_oob_um": float(item["oob_um"]),
                    "endpoint_backward_x_boundary": 1.0 if bool(item["x_boundary"]) else 0.0,
                },
            )
        )
        incoming[pair[1]] = pair[0]
        outgoing.setdefault(pair[0], []).append(pair[1])
        new_per_t[prev_t] = new_per_t.get(prev_t, 0) + 1
        cloned_nodes += 1
        cloned_edges += 1
        details.append(
            {
                "source_id": int(clone.node_id),
                "target_id": int(source.node_id),
                "future_id": int(target.node_id),
                "t": prev_t,
                "z": int(coord[0]),
                "y": int(coord[1]),
                "x": int(coord[2]),
                "edge_um": float(item["edge_um"]),
                "back_um": float(item["back_um"]),
                "duplicate_um": float(duplicate_distance_um),
                "oob_um": float(item["oob_um"]),
                "x_boundary": bool(item["x_boundary"]),
            }
        )

    return all_nodes, all_edges, {
        "enabled": True,
        "max_edge_um": max_edge_um,
        "min_edge_um": min_edge_um,
        "duplicate_um": duplicate_um,
        "max_per_t": max_per_t,
        "axis_mode": axis_mode,
        "base_nodes": len(emitted_nodes),
        "base_edges": len(selected_edges),
        "candidate_count": len(ranked),
        "cloned_nodes": cloned_nodes,
        "cloned_edges": cloned_edges,
        "new_per_t": {str(key): int(value) for key, value in sorted(new_per_t.items())},
        "clones": details,
    }



def _candidate_source_families(node: CandidateNode) -> tuple[str, ...]:
    families: list[str] = []
    for source in node.sources:
        family = str(source).split("@", 1)[0]
        if family not in families:
            families.append(family)
    return tuple(families)


def _is_refined_local_candidate(node: CandidateNode) -> bool:
    return any(family.startswith("refine_local_score") for family in _candidate_source_families(node))


def _relabel_candidate_frames(
    nodes_by_t: list[list[CandidateNode]],
    *,
    next_node_id: int,
) -> tuple[list[list[CandidateNode]], int]:
    relabeled: list[list[CandidateNode]] = []
    current_id = int(next_node_id)
    for frame in nodes_by_t:
        new_frame: list[CandidateNode] = []
        for node in frame:
            new_frame.append(replace(node, node_id=current_id))
            current_id += 1
        relabeled.append(new_frame)
    return relabeled, current_id


def _min_distance_to_nodes_um(
    node: CandidateNode,
    nodes: list[CandidateNode],
    scale_zyx: tuple[float, float, float],
) -> float:
    if not nodes:
        return float("inf")
    scale = np.asarray(scale_zyx, dtype=np.float32)
    coords = np.asarray([candidate.coord_zyx for candidate in nodes], dtype=np.float32) * scale
    query = np.asarray(node.coord_zyx, dtype=np.float32) * scale
    distances = np.sqrt(np.sum((coords - query[None, :]) ** 2, axis=1))
    return float(np.min(distances))


def _temporal_support_by_node(
    nodes_by_t: list[list[CandidateNode]],
    scale_zyx: tuple[float, float, float],
    radius_um: float,
) -> dict[int, dict[str, float]]:
    scale = np.asarray(scale_zyx, dtype=np.float32)
    coords_by_t: list[np.ndarray] = []
    trees: list[cKDTree | None] = []
    for frame in nodes_by_t:
        if frame:
            coords = np.asarray([node.coord_zyx for node in frame], dtype=np.float32) * scale
            coords_by_t.append(coords)
            trees.append(cKDTree(coords))
        else:
            coords_by_t.append(np.empty((0, 3), dtype=np.float32))
            trees.append(None)

    support: dict[int, dict[str, float]] = {}
    for rel_t, frame in enumerate(nodes_by_t):
        for index, node in enumerate(frame):
            query = coords_by_t[rel_t][index]
            counts: list[int] = []
            min_distances: list[float] = []
            for neighbor_t in (rel_t - 1, rel_t + 1):
                if neighbor_t < 0 or neighbor_t >= len(nodes_by_t) or trees[neighbor_t] is None:
                    counts.append(0)
                    min_distances.append(float("inf"))
                    continue
                hits = trees[neighbor_t].query_ball_point(query, r=float(radius_um))
                counts.append(len(hits))
                if hits:
                    deltas = coords_by_t[neighbor_t][hits] - query[None, :]
                    distances = np.sqrt(np.sum(deltas * deltas, axis=1))
                    min_distances.append(float(np.min(distances)))
                else:
                    min_distances.append(float("inf"))
            finite_distances = [value for value in min_distances if np.isfinite(value)]
            support[int(node.node_id)] = {
                "temporal_count": float(sum(counts)),
                "temporal_min_um": float(min(finite_distances) if finite_distances else 999.0),
                "temporal_two_sided": float(counts[0] > 0 and counts[1] > 0),
            }
    return support


def _admit_refined_candidate_nodes(
    dataset: str,
    array: Any,
    attrs: dict[str, Any] | None,
    start: int,
    end: int,
    nodes_by_t: list[list[CandidateNode]],
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[list[CandidateNode]], dict[str, Any] | None]:
    if not getattr(route_config, "enable_refined_candidate_admission", False):
        return nodes_by_t, None
    max_per_t = int(route_config.refined_candidate_admission_max_per_t)
    if max_per_t < 1:
        raise ValueError("refined_candidate_admission_max_per_t must be at least one")
    min_dist_um = float(route_config.refined_candidate_admission_min_dist_um)
    if not np.isfinite(min_dist_um) or min_dist_um < 0.0:
        raise ValueError("refined_candidate_admission_min_dist_um must be finite and non-negative")
    score_scale = float(route_config.refined_candidate_admission_score_scale)
    if not np.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("refined_candidate_admission_score_scale must be positive and finite")
    route_label = str(route_config.refined_candidate_admission_route)
    configs = default_route_configs()
    if route_label not in configs:
        raise ValueError(f"unknown refined candidate admission route: {route_label!r}")

    base_by_t = [list(frame) for frame in nodes_by_t]
    base_node_count = sum(len(frame) for frame in base_by_t)
    next_node_id = max((int(node.node_id) for frame in base_by_t for node in frame), default=0) + 1
    refined_by_t, _ = generate_candidates(
        dataset=dataset,
        array=array,
        attrs=attrs,
        config=configs[route_label],
        start=start,
        end=end,
        progress=False,
        proposal_policy=None,
    )
    refined_by_t, _ = _relabel_candidate_frames(refined_by_t, next_node_id=next_node_id)
    refined_only_by_t = [
        [node for node in frame if _is_refined_local_candidate(node)]
        for frame in refined_by_t
    ]
    support_pool = [
        [*base_frame, *refined_frame]
        for base_frame, refined_frame in zip(base_by_t, refined_only_by_t, strict=True)
    ]
    support = _temporal_support_by_node(
        support_pool,
        scale_zyx,
        float(route_config.max_link_um),
    )

    admitted_by_t: list[int] = []
    admitted_total = 0
    combined_by_t: list[list[CandidateNode]] = []
    for base_frame, refined_frame in zip(base_by_t, refined_only_by_t, strict=True):
        ranked: list[tuple[tuple[float, float, float, float, int], CandidateNode, float, dict[str, float]]] = []
        for node in refined_frame:
            dist_to_base = _min_distance_to_nodes_um(node, base_frame, scale_zyx)
            if dist_to_base < min_dist_um:
                continue
            node_support = support.get(
                int(node.node_id),
                {"temporal_count": 0.0, "temporal_min_um": 999.0, "temporal_two_sided": 0.0},
            )
            key = (
                float(node_support["temporal_two_sided"]),
                float(node_support["temporal_count"]),
                -float(node_support["temporal_min_um"]),
                float(node.intensity),
                -int(node.node_id),
            )
            ranked.append((key, node, dist_to_base, node_support))
        ranked.sort(key=lambda row: row[0], reverse=True)
        admitted: list[CandidateNode] = []
        for _, node, dist_to_base, node_support in ranked[:max_per_t]:
            scaled_score = float(np.clip(float(node.score) * score_scale, 0.05, 1.0))
            features = dict(node.features)
            features.update(
                {
                    "refined_admission": 1.0,
                    "refined_admission_dist_to_base_um": float(dist_to_base),
                    "refined_admission_temporal_count": float(node_support["temporal_count"]),
                    "refined_admission_temporal_min_um": float(node_support["temporal_min_um"]),
                    "refined_admission_temporal_two_sided": float(node_support["temporal_two_sided"]),
                }
            )
            admitted.append(
                replace(
                    node,
                    score=scaled_score,
                    uncertainty=float(1.0 - scaled_score),
                    features=features,
                )
            )
        admitted_by_t.append(len(admitted))
        admitted_total += len(admitted)
        combined_by_t.append([*base_frame, *admitted])

    annotate_local_density(combined_by_t, scale_zyx)
    stats = {
        "enabled": True,
        "source_route": route_label,
        "max_per_t": max_per_t,
        "min_dist_um": min_dist_um,
        "score_scale": score_scale,
        "base_nodes": base_node_count,
        "refined_nodes": sum(len(frame) for frame in refined_by_t),
        "refined_local_nodes": sum(len(frame) for frame in refined_only_by_t),
        "admitted_nodes": admitted_total,
        "base_by_t": [len(frame) for frame in base_by_t],
        "refined_local_by_t": [len(frame) for frame in refined_only_by_t],
        "admitted_by_t": admitted_by_t,
    }
    return combined_by_t, stats

def _rescue_refined_chain_conflicts(
    selected_edges: list[CandidateEdge],
    edge_candidates: list[CandidateEdge],
    candidate_nodes: list[CandidateNode],
    route_config: Any,
) -> tuple[list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_refined_chain_conflict_rescue", False):
        return selected_edges, None
    max_distance_um = float(route_config.refined_chain_conflict_max_distance_um)
    max_cost = float(route_config.refined_chain_conflict_max_cost)
    max_target_rank = int(route_config.refined_chain_conflict_max_target_rank)
    min_temporal_count = float(route_config.refined_chain_conflict_min_temporal_count)
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("refined_chain_conflict_max_distance_um must be positive and finite")
    if not np.isfinite(max_cost):
        raise ValueError("refined_chain_conflict_max_cost must be finite")
    if max_target_rank < 1:
        raise ValueError("refined_chain_conflict_max_target_rank must be at least one")
    if not np.isfinite(min_temporal_count) or min_temporal_count < 0.0:
        raise ValueError("refined_chain_conflict_min_temporal_count must be finite and non-negative")

    nodes_by_id = {int(node.node_id): node for node in candidate_nodes}
    selected_by_target = {int(edge.target_id): edge for edge in selected_edges}
    selected_by_source = {int(edge.source_id): edge for edge in selected_edges}
    selected_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
    candidate_pairs = {(int(edge.source_id), int(edge.target_id)): edge for edge in edge_candidates}

    replacements: list[tuple[CandidateEdge, CandidateEdge]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()
    for edge in sorted(
        edge_candidates,
        key=lambda value: (
            int(value.source_id),
            float(value.cost),
            float(value.distance_um),
            int(value.target_id),
        ),
    ):
        source_id = int(edge.source_id)
        target_id = int(edge.target_id)
        if (source_id, target_id) in selected_pairs:
            continue
        current = selected_by_target.get(target_id)
        if current is None or int(current.source_id) == source_id:
            continue
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        current_source = nodes_by_id.get(int(current.source_id))
        if source is None or target is None or current_source is None:
            continue
        if source_id not in selected_by_target or source_id in selected_by_source:
            continue
        if target_id in used_targets or source_id in used_sources:
            continue
        if float(edge.distance_um) > max_distance_um or float(edge.cost) > max_cost:
            continue
        if float(edge.features.get("source_out_rank", 999.0)) != 1.0:
            continue
        if float(edge.features.get("source_out_candidate_count", 999.0)) != 1.0:
            continue
        if float(edge.features.get("target_in_rank", 999.0)) > float(max_target_rank):
            continue
        if float(source.features.get("refined_admission", 0.0)) < 1.0:
            continue
        if float(source.features.get("refined_admission_temporal_two_sided", 0.0)) < 1.0:
            continue
        if float(source.features.get("refined_admission_temporal_count", 0.0)) < min_temporal_count:
            continue
        if float(source.features.get("refined_admission_temporal_min_um", np.inf)) > max_distance_um:
            continue
        if (int(current.source_id), int(current.target_id)) not in candidate_pairs:
            continue
        replacements.append((current, edge))
        used_targets.add(target_id)
        used_sources.add(source_id)

    if not replacements:
        return selected_edges, {
            "enabled": True,
            "before_edges": len(selected_edges),
            "after_edges": len(selected_edges),
            "replacements": 0,
            "max_distance_um": max_distance_um,
            "max_cost": max_cost,
            "max_target_rank": max_target_rank,
            "min_temporal_count": min_temporal_count,
        }

    remove_pairs = {
        (int(current.source_id), int(current.target_id))
        for current, _replacement in replacements
    }
    rescued = [
        edge
        for edge in selected_edges
        if (int(edge.source_id), int(edge.target_id)) not in remove_pairs
    ]
    rescued.extend(replacement for _current, replacement in replacements)
    rescued.sort(key=lambda edge: (int(edge.source_id), int(edge.target_id)))
    return rescued, {
        "enabled": True,
        "before_edges": len(selected_edges),
        "after_edges": len(rescued),
        "replacements": len(replacements),
        "max_distance_um": max_distance_um,
        "max_cost": max_cost,
        "max_target_rank": max_target_rank,
        "min_temporal_count": min_temporal_count,
        "replaced_edges": [
            {
                "removed_source_id": int(current.source_id),
                "removed_target_id": int(current.target_id),
                "added_source_id": int(replacement.source_id),
                "added_target_id": int(replacement.target_id),
                "added_distance_um": float(replacement.distance_um),
                "added_cost": float(replacement.cost),
                "added_target_rank": float(replacement.features.get("target_in_rank", np.nan)),
            }
            for current, replacement in replacements
        ],
    }

def _prune_short_selected_edges(
    selected_edges: list[CandidateEdge],
    route_config: Any,
) -> tuple[list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_short_edge_pruning", False):
        return selected_edges, None
    min_distance_um = float(route_config.short_edge_prune_min_distance_um)
    if not np.isfinite(min_distance_um) or min_distance_um < 0.0:
        raise ValueError("short_edge_prune_min_distance_um must be finite and non-negative")
    if min_distance_um <= 0.0:
        return selected_edges, {
            "enabled": True,
            "min_distance_um": min_distance_um,
            "before_edges": len(selected_edges),
            "after_edges": len(selected_edges),
            "dropped_edges": 0,
            "dropped_by_kind": {},
        }
    kept: list[CandidateEdge] = []
    dropped_by_kind: dict[str, int] = {}
    dropped_distances: list[float] = []
    for edge in selected_edges:
        if float(edge.distance_um) < min_distance_um:
            kind = str(edge.kind)
            dropped_by_kind[kind] = dropped_by_kind.get(kind, 0) + 1
            dropped_distances.append(float(edge.distance_um))
            continue
        kept.append(edge)
    return kept, {
        "enabled": True,
        "min_distance_um": min_distance_um,
        "before_edges": len(selected_edges),
        "after_edges": len(kept),
        "dropped_edges": len(selected_edges) - len(kept),
        "dropped_by_kind": dict(sorted(dropped_by_kind.items())),
        "min_dropped_distance_um": float(min(dropped_distances)) if dropped_distances else None,
        "max_dropped_distance_um": float(max(dropped_distances)) if dropped_distances else None,
    }

def _admit_free_candidate_edges(
    selected_edges: list[CandidateEdge],
    edge_candidates: list[CandidateEdge],
    candidate_nodes: list[CandidateNode],
    route_config: Any,
) -> tuple[list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_free_edge_admission", False):
        return selected_edges, None
    max_cost = float(route_config.free_edge_admission_max_cost)
    min_distance_um = float(route_config.free_edge_admission_min_distance_um)
    max_distance_um = float(route_config.free_edge_admission_max_distance_um)
    max_source_candidates = int(route_config.free_edge_admission_max_source_candidates)
    max_target_candidates = int(route_config.free_edge_admission_max_target_candidates)
    max_edges = int(route_config.free_edge_admission_max_edges)
    require_mutual = bool(route_config.free_edge_admission_require_mutual_nearest)
    if not np.isfinite(max_cost):
        raise ValueError("free_edge_admission_max_cost must be finite")
    if not np.isfinite(min_distance_um) or min_distance_um < 0.0:
        raise ValueError("free_edge_admission_min_distance_um must be finite and non-negative")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("free_edge_admission_max_distance_um must be positive and finite")
    if min_distance_um > max_distance_um:
        raise ValueError("free_edge_admission_min_distance_um must not exceed max distance")
    if max_source_candidates < 1 or max_target_candidates < 1:
        raise ValueError("free edge candidate-count limits must be at least one")
    if max_edges < 1:
        raise ValueError("free_edge_admission_max_edges must be at least one")

    nodes_by_id = {int(node.node_id): node for node in candidate_nodes}
    selected_out = {int(edge.source_id) for edge in selected_edges}
    selected_in = {int(edge.target_id) for edge in selected_edges}
    selected_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
    additions: list[CandidateEdge] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for edge in sorted(
        edge_candidates,
        key=lambda value: (
            float(value.cost),
            float(value.distance_um),
            int(value.source_id),
            int(value.target_id),
        ),
    ):
        source_id = int(edge.source_id)
        target_id = int(edge.target_id)
        if (source_id, target_id) in selected_pairs:
            continue
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        if source_id in selected_out or target_id in selected_in:
            continue
        if source_id in used_sources or target_id in used_targets:
            continue
        if int(edge.dt) != 1:
            continue
        if float(edge.cost) > max_cost:
            continue
        if not (min_distance_um <= float(edge.distance_um) <= max_distance_um):
            continue
        if require_mutual and float(edge.features.get("mutual_nearest", 0.0)) < 1.0:
            continue
        if float(edge.features.get("source_out_rank", 999.0)) > 1.0:
            continue
        if float(edge.features.get("target_in_rank", 999.0)) > 1.0:
            continue
        if float(edge.features.get("source_out_candidate_count", 999.0)) > float(max_source_candidates):
            continue
        if float(edge.features.get("target_in_candidate_count", 999.0)) > float(max_target_candidates):
            continue
        additions.append(edge)
        used_sources.add(source_id)
        used_targets.add(target_id)
        if len(additions) >= max_edges:
            break

    if not additions:
        return selected_edges, {
            "enabled": True,
            "before_edges": len(selected_edges),
            "after_edges": len(selected_edges),
            "admitted_edges": 0,
            "max_cost": max_cost,
            "min_distance_um": min_distance_um,
            "max_distance_um": max_distance_um,
            "require_mutual_nearest": require_mutual,
            "max_source_candidates": max_source_candidates,
            "max_target_candidates": max_target_candidates,
            "max_edges": max_edges,
        }

    admitted = [*selected_edges, *additions]
    admitted.sort(
        key=lambda edge: (
            int(nodes_by_id[int(edge.source_id)].t),
            int(nodes_by_id[int(edge.target_id)].t),
            int(edge.source_id),
            int(edge.target_id),
        )
    )
    return admitted, {
        "enabled": True,
        "before_edges": len(selected_edges),
        "after_edges": len(admitted),
        "admitted_edges": len(additions),
        "max_cost": max_cost,
        "min_distance_um": min_distance_um,
        "max_distance_um": max_distance_um,
        "require_mutual_nearest": require_mutual,
        "max_source_candidates": max_source_candidates,
        "max_target_candidates": max_target_candidates,
        "max_edges": max_edges,
        "admitted": [
            {
                "source_id": int(edge.source_id),
                "target_id": int(edge.target_id),
                "distance_um": float(edge.distance_um),
                "cost": float(edge.cost),
                "source_out_candidate_count": float(edge.features.get("source_out_candidate_count", np.nan)),
                "target_in_candidate_count": float(edge.features.get("target_in_candidate_count", np.nan)),
            }
            for edge in additions
        ],
    }


def _rescue_duplicate_endpoint_conflicts(
    selected_edges: list[CandidateEdge],
    edge_candidates: list[CandidateEdge],
    candidate_nodes: list[CandidateNode],
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_duplicate_endpoint_rescue", False):
        return selected_edges, None
    max_um = float(route_config.duplicate_endpoint_rescue_max_um)
    min_cost = float(route_config.duplicate_endpoint_rescue_min_cost)
    max_cost = float(route_config.duplicate_endpoint_rescue_max_cost)
    max_distance_um = float(route_config.duplicate_endpoint_rescue_max_distance_um)
    max_swaps = int(route_config.duplicate_endpoint_rescue_max_swaps)
    if not np.isfinite(max_um) or max_um < 0.0:
        raise ValueError("duplicate_endpoint_rescue_max_um must be finite and non-negative")
    if not np.isfinite(min_cost):
        raise ValueError("duplicate_endpoint_rescue_min_cost must be finite")
    if not np.isfinite(max_cost):
        raise ValueError("duplicate_endpoint_rescue_max_cost must be finite")
    if min_cost > max_cost:
        raise ValueError("duplicate_endpoint_rescue_min_cost must not exceed max cost")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("duplicate_endpoint_rescue_max_distance_um must be positive and finite")
    if max_swaps < 1:
        raise ValueError("duplicate_endpoint_rescue_max_swaps must be at least one")

    nodes_by_id = {int(node.node_id): node for node in candidate_nodes}
    candidate_by_pair = {(int(edge.source_id), int(edge.target_id)): edge for edge in edge_candidates}
    selected_by_pair = {(int(edge.source_id), int(edge.target_id)): edge for edge in selected_edges}
    selected_out = {int(edge.source_id): int(edge.target_id) for edge in selected_edges}
    selected_in = {int(edge.target_id): int(edge.source_id) for edge in selected_edges}
    scale = np.asarray(scale_zyx, dtype=np.float64)

    def _distance(a: CandidateNode, b: CandidateNode) -> float:
        return float(
            np.linalg.norm(
                (np.asarray(a.coord_zyx, dtype=np.float64) - np.asarray(b.coord_zyx, dtype=np.float64))
                * scale
            )
        )

    def _candidate_ok(edge: CandidateEdge | None) -> bool:
        return (
            edge is not None
            and int(edge.dt) == 1
            and min_cost <= float(edge.cost) <= max_cost
            and float(edge.distance_um) <= max_distance_um
        )

    proposals: list[tuple[float, float, int, str, tuple[int, int], CandidateEdge, int]] = []
    for selected in selected_edges:
        source_id = int(selected.source_id)
        target_id = int(selected.target_id)
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue

        if target_id not in selected_out:
            for duplicate in candidate_nodes:
                duplicate_id = int(duplicate.node_id)
                if duplicate_id == target_id or int(duplicate.t) != int(target.t):
                    continue
                if duplicate_id in selected_in or duplicate_id not in selected_out:
                    continue
                duplicate_um = _distance(target, duplicate)
                if duplicate_um > max_um:
                    continue
                replacement = candidate_by_pair.get((source_id, duplicate_id))
                if not _candidate_ok(replacement):
                    continue
                proposals.append(
                    (
                        float(replacement.cost),
                        duplicate_um,
                        source_id,
                        "target",
                        (source_id, target_id),
                        replacement,
                        duplicate_id,
                    )
                )

        if source_id not in selected_in:
            for duplicate in candidate_nodes:
                duplicate_id = int(duplicate.node_id)
                if duplicate_id == source_id or int(duplicate.t) != int(source.t):
                    continue
                if duplicate_id not in selected_in or duplicate_id in selected_out:
                    continue
                duplicate_um = _distance(source, duplicate)
                if duplicate_um > max_um:
                    continue
                replacement = candidate_by_pair.get((duplicate_id, target_id))
                if not _candidate_ok(replacement):
                    continue
                proposals.append(
                    (
                        float(replacement.cost),
                        duplicate_um,
                        source_id,
                        "source",
                        (source_id, target_id),
                        replacement,
                        duplicate_id,
                    )
                )

    if not proposals:
        return selected_edges, {
            "enabled": True,
            "max_um": max_um,
            "min_cost": min_cost,
            "max_cost": max_cost,
            "max_distance_um": max_distance_um,
            "max_swaps": max_swaps,
            "candidate_swaps": 0,
            "applied_swaps": 0,
        }

    proposals.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    current = dict(selected_by_pair)
    selected_out = {int(edge.source_id): int(edge.target_id) for edge in current.values()}
    selected_in = {int(edge.target_id): int(edge.source_id) for edge in current.values()}
    applied: list[dict[str, Any]] = []
    for cost, duplicate_um, _rank_source, mode, old_pair, replacement, duplicate_id in proposals:
        if len(applied) >= max_swaps:
            break
        if old_pair not in current:
            continue
        old_source, old_target = old_pair
        new_pair = (int(replacement.source_id), int(replacement.target_id))
        if new_pair in current:
            continue
        if mode == "target":
            if selected_out.get(old_source) != old_target:
                continue
            if new_pair[1] in selected_in:
                continue
        elif mode == "source":
            if selected_in.get(old_target) != old_source:
                continue
            if new_pair[0] in selected_out:
                continue
        else:
            continue
        old_edge = current.pop(old_pair)
        selected_out.pop(old_pair[0], None)
        selected_in.pop(old_pair[1], None)
        current[new_pair] = replacement
        selected_out[new_pair[0]] = new_pair[1]
        selected_in[new_pair[1]] = new_pair[0]
        applied.append(
            {
                "mode": mode,
                "old_source_id": int(old_pair[0]),
                "old_target_id": int(old_pair[1]),
                "new_source_id": int(new_pair[0]),
                "new_target_id": int(new_pair[1]),
                "duplicate_id": int(duplicate_id),
                "duplicate_um": float(duplicate_um),
                "old_cost": float(old_edge.cost),
                "new_cost": float(replacement.cost),
                "new_distance_um": float(replacement.distance_um),
            }
        )

    if not applied:
        return selected_edges, {
            "enabled": True,
            "max_um": max_um,
            "min_cost": min_cost,
            "max_cost": max_cost,
            "max_distance_um": max_distance_um,
            "max_swaps": max_swaps,
            "candidate_swaps": len(proposals),
            "applied_swaps": 0,
        }

    rescued = sorted(
        current.values(),
        key=lambda edge: (
            int(nodes_by_id[int(edge.source_id)].t) if int(edge.source_id) in nodes_by_id else 0,
            int(nodes_by_id[int(edge.target_id)].t) if int(edge.target_id) in nodes_by_id else 0,
            int(edge.source_id),
            int(edge.target_id),
        ),
    )
    return rescued, {
        "enabled": True,
        "max_um": max_um,
        "max_cost": max_cost,
        "max_distance_um": max_distance_um,
        "max_swaps": max_swaps,
        "candidate_swaps": len(proposals),
        "applied_swaps": len(applied),
        "swaps": applied,
    }


def _motion_endpoint_source_families(node: CandidateNode) -> set[str]:
    families: set[str] = set()
    for source in node.sources:
        token = str(source).split("@", 1)[0]
        if ":" in token:
            token = token.rsplit(":", 1)[-1]
        if token:
            families.add(token)
    return families


def _candidate_matches_motion_endpoint_mode(node: CandidateNode, mode: str) -> bool:
    normalized = str(mode or "any").strip().lower()
    if normalized in {"", "any", "all", "same_source", "source_consistent"}:
        return True
    sources = "|".join(str(source) for source in node.sources)
    if normalized == "zshift":
        return "zshift_" in sources
    if normalized == "not_tile":
        return "tile_" not in sources
    raise ValueError(f"unknown motion_endpoint_target_source_mode: {mode!r}")


def _admit_motion_endpoint_candidates(
    selected_edges: list[CandidateEdge],
    candidate_nodes: list[CandidateNode],
    scale_zyx: tuple[float, float, float],
    route_config: Any,
) -> tuple[list[CandidateEdge], dict[str, Any] | None]:
    if not getattr(route_config, "enable_motion_endpoint_admission", False):
        return selected_edges, None
    pred_max_um = float(route_config.motion_endpoint_pred_max_um)
    source_max_um = float(route_config.motion_endpoint_source_max_um)
    max_edges_per_t = int(route_config.motion_endpoint_max_edges_per_t)
    target_mode = str(route_config.motion_endpoint_target_source_mode)
    normalized_target_mode = target_mode.strip().lower()
    source_consistent = normalized_target_mode in {"same_source", "source_consistent"}
    rank_mode = str(getattr(route_config, "motion_endpoint_rank_mode", "source_order") or "source_order").strip().lower()
    requires_history = bool(route_config.motion_endpoint_requires_history)
    if not np.isfinite(pred_max_um) or pred_max_um <= 0.0:
        raise ValueError("motion_endpoint_pred_max_um must be positive and finite")
    if not np.isfinite(source_max_um) or source_max_um <= 0.0:
        raise ValueError("motion_endpoint_source_max_um must be positive and finite")
    if max_edges_per_t < 1:
        raise ValueError("motion_endpoint_max_edges_per_t must be at least one")
    if rank_mode not in {"source_order", "pred", "source_desc", "id"}:
        raise ValueError(f"unknown motion_endpoint_rank_mode: {rank_mode!r}")

    nodes_by_id = {int(node.node_id): node for node in candidate_nodes}
    nodes_by_t: dict[int, list[CandidateNode]] = {}
    for node in candidate_nodes:
        nodes_by_t.setdefault(int(node.t), []).append(node)
    selected_out = {int(edge.source_id) for edge in selected_edges}
    selected_in = {int(edge.target_id) for edge in selected_edges}
    selected_pairs = {(int(edge.source_id), int(edge.target_id)) for edge in selected_edges}
    incoming_source = {int(edge.target_id): int(edge.source_id) for edge in selected_edges}
    scale = np.asarray(scale_zyx, dtype=np.float64)

    additions: list[CandidateEdge] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    per_t: dict[int, int] = {}
    details: list[dict[str, Any]] = []

    def _source_and_prediction(source: CandidateNode) -> tuple[np.ndarray, np.ndarray] | None:
        source_id = int(source.node_id)
        if source_id in selected_out or source_id in used_sources:
            return None
        previous_id = incoming_source.get(source_id)
        if requires_history and previous_id is None:
            return None
        previous = nodes_by_id.get(previous_id) if previous_id is not None else None
        current_center = np.asarray(source.coord_zyx, dtype=np.float64)
        if previous is None:
            predicted_center = current_center
        else:
            previous_center = np.asarray(previous.coord_zyx, dtype=np.float64)
            predicted_center = current_center + (current_center - previous_center)
        return current_center, predicted_center

    def _target_pool(target_t: int) -> list[CandidateNode]:
        return [
            node
            for node in nodes_by_t.get(target_t, [])
            if int(node.node_id) not in selected_in
            and _candidate_matches_motion_endpoint_mode(node, target_mode)
        ]

    def _target_is_allowed(source: CandidateNode, target: CandidateNode) -> bool:
        if not source_consistent:
            return True
        return bool(_motion_endpoint_source_families(source) & _motion_endpoint_source_families(target))

    def _add_edge(source: CandidateNode, target: CandidateNode, pred_distance: float, source_distance: float) -> None:
        source_id = int(source.node_id)
        target_id = int(target.node_id)
        additions.append(
            CandidateEdge(
                source_id=source_id,
                target_id=target_id,
                dt=1,
                distance_um=source_distance,
                cost=0.50,
                score=0.50,
                kind="motion_endpoint_candidate",
                features={
                    "motion_endpoint_candidate": 1.0,
                    "motion_pred_distance_um": float(pred_distance),
                },
            )
        )
        used_sources.add(source_id)
        used_targets.add(target_id)
        per_t[int(target.t)] = per_t.get(int(target.t), 0) + 1
        details.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "t": int(target.t),
                "pred_distance_um": float(pred_distance),
                "source_distance_um": float(source_distance),
                "target_sources": list(target.sources),
            }
        )

    for rel_t in sorted(nodes_by_t):
        target_t = rel_t + 1
        target_nodes = _target_pool(target_t)
        if not target_nodes:
            continue
        if rank_mode == "source_order":
            for source in sorted(nodes_by_t[rel_t], key=lambda node: int(node.node_id)):
                source_id = int(source.node_id)
                centers = _source_and_prediction(source)
                if centers is None:
                    continue
                if per_t.get(target_t, 0) >= max_edges_per_t:
                    continue
                current_center, predicted_center = centers
                source_scaled = current_center * scale
                predicted_scaled = predicted_center * scale
                ranked: list[tuple[float, float, float, int, CandidateNode]] = []
                for target in target_nodes:
                    target_id = int(target.node_id)
                    if target_id in used_targets or (source_id, target_id) in selected_pairs:
                        continue
                    if not _target_is_allowed(source, target):
                        continue
                    target_scaled = np.asarray(target.coord_zyx, dtype=np.float64) * scale
                    pred_distance = float(np.linalg.norm(predicted_scaled - target_scaled))
                    source_distance = float(np.linalg.norm(source_scaled - target_scaled))
                    if pred_distance > pred_max_um or source_distance > source_max_um:
                        continue
                    ranked.append((pred_distance, source_distance, -float(target.score), target_id, target))
                if not ranked:
                    continue
                ranked.sort()
                pred_distance, source_distance, _score, target_id, target = ranked[0]
                _add_edge(source, target, pred_distance, source_distance)
            continue

        proposals: list[tuple[tuple[float, ...], float, float, int, int, CandidateNode, CandidateNode]] = []
        for source in sorted(nodes_by_t[rel_t], key=lambda node: int(node.node_id)):
            source_id = int(source.node_id)
            centers = _source_and_prediction(source)
            if centers is None:
                continue
            current_center, predicted_center = centers
            source_scaled = current_center * scale
            predicted_scaled = predicted_center * scale
            best: tuple[float, float, float, int, CandidateNode] | None = None
            for target in target_nodes:
                target_id = int(target.node_id)
                if target_id in used_targets or (source_id, target_id) in selected_pairs:
                    continue
                if not _target_is_allowed(source, target):
                    continue
                target_scaled = np.asarray(target.coord_zyx, dtype=np.float64) * scale
                pred_distance = float(np.linalg.norm(predicted_scaled - target_scaled))
                source_distance = float(np.linalg.norm(source_scaled - target_scaled))
                if pred_distance > pred_max_um or source_distance > source_max_um:
                    continue
                row = (pred_distance, source_distance, -float(target.score), target_id, target)
                if best is None or row < best:
                    best = row
            if best is None:
                continue
            pred_distance, source_distance, _score, target_id, target = best
            if rank_mode == "source_desc":
                rank_key = (-source_distance, pred_distance, float(source_id), float(target_id))
            elif rank_mode == "id":
                rank_key = (float(source_id), float(target_id))
            else:
                rank_key = (pred_distance, source_distance, float(source_id), float(target_id))
            proposals.append((rank_key, pred_distance, source_distance, source_id, target_id, source, target))

        proposals.sort(key=lambda row: row[0])
        for _rank_key, pred_distance, source_distance, source_id, target_id, source, target in proposals:
            if per_t.get(target_t, 0) >= max_edges_per_t:
                break
            if source_id in used_sources or target_id in used_targets:
                continue
            _add_edge(source, target, pred_distance, source_distance)

    if not additions:
        return selected_edges, {
            "enabled": True,
            "before_edges": len(selected_edges),
            "after_edges": len(selected_edges),
            "admitted_edges": 0,
            "pred_max_um": pred_max_um,
            "source_max_um": source_max_um,
            "max_edges_per_t": max_edges_per_t,
            "target_source_mode": target_mode,
            "rank_mode": rank_mode,
            "requires_history": requires_history,
        }

    admitted = [*selected_edges, *additions]
    admitted.sort(
        key=lambda edge: (
            int(nodes_by_id[int(edge.source_id)].t),
            int(nodes_by_id[int(edge.target_id)].t),
            int(edge.source_id),
            int(edge.target_id),
        )
    )
    return admitted, {
        "enabled": True,
        "before_edges": len(selected_edges),
        "after_edges": len(admitted),
        "admitted_edges": len(additions),
        "pred_max_um": pred_max_um,
        "source_max_um": source_max_um,
        "max_edges_per_t": max_edges_per_t,
        "target_source_mode": target_mode,
        "rank_mode": rank_mode,
        "requires_history": requires_history,
        "per_t": {str(key): int(value) for key, value in sorted(per_t.items())},
        "admitted": details,
    }

def _nodes_used_by_edges(nodes: list[Any], edges: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    if not edges:
        return nodes, {
            "policy": "all_nodes_no_edges",
            "before_nodes": len(nodes),
            "after_nodes": len(nodes),
            "edges": 0,
        }
    endpoint_ids = {
        int(node_id)
        for edge in edges
        for node_id in (edge.source_id, edge.target_id)
    }
    filtered = [node for node in nodes if int(node.node_id) in endpoint_ids]
    return filtered, {
        "policy": "edge_endpoint_nodes",
        "before_nodes": len(nodes),
        "after_nodes": len(filtered),
        "edges": len(edges),
    }


def predict_dataset_window(
    dataset: str,
    array: Any,
    attrs: dict[str, Any] | None,
    start: int = 0,
    end: int | None = None,
    forced_route: str | None = None,
    allow_division_route: bool = False,
    output_offset_override: tuple[int, int, int] | None = None,
    estimated_total_nodes: float | None = None,
    scoring_calibration: ScoringCalibration | None = None,
    progress: bool = False,
) -> TrackingRunResult:
    """Predict a graph for one dataset/window."""

    start, end = _validate_array_window(dataset, array, start, end)
    output_offset_override = validate_output_offset_override(output_offset_override)

    profile = profile_scenario(
        dataset=dataset,
        array=array,
        attrs=attrs,
        start=start,
        end=end,
        estimated_total_nodes=estimated_total_nodes,
    )
    route = route_for_profile(
        profile,
        default_route_configs(),
        forced_route,
        allow_division_route=allow_division_route,
    )
    if progress:
        print(
            f"{dataset}: route={route.label} confidence={route.confidence:.2f} "
            f"reason={route.reason}",
            flush=True,
        )

    scoring_payload = scoring_for_dataset(dataset, scoring_calibration)
    proposal_policy = proposal_policy_for_dataset(
        scoring_payload, dataset=dataset, route_label=route.label
    )
    nodes_by_t, candidate_nodes = generate_candidates(
        dataset=dataset,
        array=array,
        attrs=attrs,
        config=route.config,
        start=start,
        end=end,
        progress=progress,
        proposal_policy=proposal_policy,
    )
    scale_zyx = parse_scale(attrs)
    nodes_by_t, correction_stats = correct_nodes_by_t(
        nodes_by_t,
        dataset=dataset,
        route_label=route.label,
        calibration_payload=scoring_payload,
        spatial_shape_zyx=tuple(int(v) for v in array.shape[1:4]),
    )
    if correction_stats is not None:
        _refresh_node_intensities(nodes_by_t, array, start)
        annotate_local_density(nodes_by_t, scale_zyx)
        candidate_nodes = [node for nodes in nodes_by_t for node in nodes]
    nodes_by_t, pruning_stats = prune_nodes_by_t(
        nodes_by_t,
        dataset=dataset,
        route_label=route.label,
        calibration_payload=scoring_payload,
    )
    if pruning_stats is not None:
        annotate_local_density(nodes_by_t, scale_zyx)
        candidate_nodes = [node for nodes in nodes_by_t for node in nodes]
    nodes_by_t, refined_admission_stats = _admit_refined_candidate_nodes(
        dataset, array, attrs, start, end, nodes_by_t, scale_zyx, route.config
    )
    if refined_admission_stats is not None:
        candidate_nodes = [node for nodes in nodes_by_t for node in nodes]
    edge_candidates = build_edge_candidates(nodes_by_t, scale_zyx, route.config)
    edge_candidates, edge_scoring_stats = score_edge_candidates(
        edge_candidates,
        nodes_by_t,
        dataset=dataset,
        route_label=route.label,
        route_config=route.config,
        calibration_payload=scoring_payload,
    )
    selected_edges, forks = select_edges(nodes_by_t, edge_candidates, scale_zyx, route.config)
    selected_edges, refined_chain_conflict_stats = _rescue_refined_chain_conflicts(
        selected_edges, edge_candidates, candidate_nodes, route.config
    )
    selected_edges, short_edge_pruning_stats = _prune_short_selected_edges(
        selected_edges, route.config
    )
    localmax_refinement_stats = _refine_selected_nodes_by_localmax(
        nodes_by_t, selected_edges, array, attrs, start, scale_zyx, route.config
    )
    selected_edges, free_edge_admission_stats = _admit_free_candidate_edges(
        selected_edges, edge_candidates, candidate_nodes, route.config
    )
    selected_edges, duplicate_endpoint_rescue_stats = _rescue_duplicate_endpoint_conflicts(
        selected_edges, edge_candidates, candidate_nodes, scale_zyx, route.config
    )
    selected_edges, motion_endpoint_admission_stats = _admit_motion_endpoint_candidates(
        selected_edges, candidate_nodes, scale_zyx, route.config
    )

    emitted_candidate_nodes, node_emission_stats = _nodes_used_by_edges(
        candidate_nodes, selected_edges
    )
    emitted_candidate_nodes, selected_edges, localmax_clone_stats = (
        _clone_selected_nodes_by_localmax_layers(
            emitted_candidate_nodes, selected_edges, array, attrs, start, scale_zyx, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, offset_clone_stats = (
        _clone_selected_nodes_by_offset_layers(
            emitted_candidate_nodes, selected_edges, array, start, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, endpoint_backward_stats = (
        _add_endpoint_backward_clones(
            emitted_candidate_nodes, selected_edges, array, start, scale_zyx, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, endpoint_forward_stats = (
        _add_endpoint_forward_clones(
            emitted_candidate_nodes, selected_edges, array, attrs, start, scale_zyx, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, localmax_clone_rewire_stats = (
        _rewire_localmax_clone_targets(
            emitted_candidate_nodes, selected_edges, scale_zyx, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, base_chain_rewire_stats = (
        _rewire_base_chain_targets(
            emitted_candidate_nodes, selected_edges, scale_zyx, route.config
        )
    )
    emitted_candidate_nodes, selected_edges, component_pruning_stats = (
        _prune_submission_components(
            emitted_candidate_nodes, selected_edges, route.config
        )
    )
    emitted_candidate_nodes, final_node_pruning_stats = _prune_unlinked_submission_nodes(
        emitted_candidate_nodes, selected_edges
    )
    nodes = [node.to_submission_node() for node in emitted_candidate_nodes]
    output_offset = (
        tuple(int(v) for v in output_offset_override)
        if output_offset_override is not None
        else _resolve_output_offset(dataset, route.config)
    )
    nodes = _shift_node_dicts(nodes, output_offset, tuple(int(v) for v in array.shape[1:4]))
    edges = [edge.to_submission_edge() for edge in selected_edges]
    diagnostics = graph_diagnostics(emitted_candidate_nodes, selected_edges)
    diagnostics.update(
        {
            "route": route.label,
            "route_confidence": route.confidence,
            "route_reason": route.reason,
            "node_emission": node_emission_stats,
            "num_edge_candidates": len(edge_candidates),
            "num_selected_forks": len(forks),
            "coordinate_correction": correction_stats.as_dict() if correction_stats is not None else None,
            "node_pruning": pruning_stats.as_dict() if pruning_stats is not None else None,
            "refined_candidate_admission": refined_admission_stats,
            "edge_scoring": edge_scoring_stats.as_dict() if edge_scoring_stats is not None else None,
            "refined_chain_conflict_rescue": refined_chain_conflict_stats,
            "short_edge_pruning": short_edge_pruning_stats,
            "localmax_refinement": localmax_refinement_stats,
            "free_edge_admission": free_edge_admission_stats,
            "duplicate_endpoint_rescue": duplicate_endpoint_rescue_stats,
            "motion_endpoint_admission": motion_endpoint_admission_stats,
            "localmax_clone_layers": localmax_clone_stats,
            "offset_clone_layers": offset_clone_stats,
            "endpoint_backward_clones": endpoint_backward_stats,
            "endpoint_forward_clones": endpoint_forward_stats,
            "localmax_clone_target_rewire": localmax_clone_rewire_stats,
            "base_chain_target_rewire": base_chain_rewire_stats,
            "component_pruning": component_pruning_stats,
            "final_node_pruning": final_node_pruning_stats,
            "output_offset_zyx": output_offset,
            "profile": asdict(profile),
            "route_config": asdict(route.config),
        }
    )
    return TrackingRunResult(
        dataset=dataset,
        nodes=nodes,
        edges=edges,
        profile=profile,
        route=route,
        diagnostics=diagnostics,
    )
