"""End-to-end scenario-routed low-compute tracking pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from numbers import Integral
from typing import Any

import numpy as np

from celltrack_baseline.rule_based import parse_scale

from .proposals import generate_candidates
from .routing import default_route_configs, profile_scenario, route_for_profile
from .solver import build_edge_candidates, graph_diagnostics, select_edges
from .types import TrackingRunResult


def _validate_array_window(
    dataset: str,
    array: Any,
    start: int,
    end: int | None,
) -> tuple[int, int]:
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    try:
        shape = tuple(int(value) for value in array.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("array must expose a four-dimensional T/Z/Y/X shape") from exc
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(
            f"array must have a non-empty four-dimensional T/Z/Y/X shape, got {shape!r}"
        )
    dtype = getattr(array, "dtype", None)
    if dtype is not None:
        try:
            resolved_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"array dtype is not supported: {dtype!r}") from exc
        if not np.issubdtype(resolved_dtype, np.number) or np.issubdtype(
            resolved_dtype, np.complexfloating
        ):
            raise ValueError(f"array dtype must be real numeric, got {resolved_dtype}")
    if isinstance(start, bool) or not isinstance(start, Integral) or int(start) < 0:
        raise ValueError(f"start must be a non-negative integer, got {start!r}")
    resolved_start = int(start)
    if end is None:
        resolved_end = shape[0]
    elif isinstance(end, bool) or not isinstance(end, Integral):
        raise ValueError(f"end must be an integer or None, got {end!r}")
    else:
        resolved_end = int(end)
    if resolved_end <= resolved_start or resolved_end > shape[0]:
        raise ValueError(
            f"window must satisfy 0 <= start < end <= {shape[0]}, got "
            f"start={start!r}, end={end!r}"
        )
    return resolved_start, resolved_end


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
    progress: bool = False,
) -> TrackingRunResult:
    """Predict a graph for one dataset/window."""

    start, end = _validate_array_window(dataset, array, start, end)
    if output_offset_override is not None:
        if isinstance(output_offset_override, (str, bytes)) or not isinstance(
            output_offset_override, Iterable
        ):
            raise ValueError("output_offset_override must contain exactly three integers")
        offset_values = tuple(output_offset_override)
        if len(offset_values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in offset_values
        ):
            raise ValueError("output_offset_override must contain exactly three integers")
        output_offset_override = tuple(int(value) for value in offset_values)

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

    nodes_by_t, candidate_nodes = generate_candidates(
        dataset=dataset,
        array=array,
        attrs=attrs,
        config=route.config,
        start=start,
        end=end,
        progress=progress,
    )
    scale_zyx = parse_scale(attrs)
    edge_candidates = build_edge_candidates(nodes_by_t, scale_zyx, route.config)
    selected_edges, forks = select_edges(nodes_by_t, edge_candidates, scale_zyx, route.config)

    nodes = [node.to_submission_node() for node in candidate_nodes]
    output_offset = (
        tuple(int(v) for v in output_offset_override)
        if output_offset_override is not None
        else _resolve_output_offset(dataset, route.config)
    )
    nodes = _shift_node_dicts(nodes, output_offset, tuple(int(v) for v in array.shape[1:4]))
    edges = [edge.to_submission_edge() for edge in selected_edges]
    diagnostics = graph_diagnostics(candidate_nodes, selected_edges)
    diagnostics.update(
        {
            "route": route.label,
            "route_confidence": route.confidence,
            "route_reason": route.reason,
            "num_edge_candidates": len(edge_candidates),
            "num_selected_forks": len(forks),
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
