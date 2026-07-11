"""Scenario profiling and routing for low-compute cell tracking."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from celltrack_baseline.rule_based import BaselineConfig, dog_response, normalize_volume, parse_scale

from .types import RouteConfig, RouteDecision, ScenarioProfile


def default_route_configs() -> dict[str, RouteConfig]:
    """Return deterministic route configs for the first alpha."""

    return {
        "clean_point": RouteConfig(
            label="clean_point",
            detector_sources=("dog_bright", "dog_dark", "local_contrast"),
            threshold_quantiles=(0.9994, 0.99965),
            min_distance_um=3.0,
            fusion_radius_um=2.6,
            max_points_per_t=120,
            max_link_um=8.0,
            edge_cost_threshold=0.84,
            prefix_output_offsets_zyx=(("44b6", 0, -16, 8),),
            notes="Moderate candidate count; one-step links; divisions off.",
        ),
        "dense_fusion": RouteConfig(
            label="dense_fusion",
            detector_sources=("dog_bright", "dog_dark", "dog_abs", "local_contrast", "hmax", "components"),
            threshold_quantiles=(0.9986, 0.9991, 0.99955),
            min_distance_um=2.2,
            fusion_radius_um=2.0,
            max_points_per_t=260,
            max_link_um=7.5,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.08,
            notes="High-recall proposal fusion for crowded windows.",
        ),
        "low_signal": RouteConfig(
            label="low_signal",
            detector_sources=("dog_dark", "dog_abs", "local_contrast", "hmax", "components"),
            threshold_quantiles=(0.9978, 0.9988, 0.99935),
            min_distance_um=2.6,
            fusion_radius_um=2.2,
            max_points_per_t=220,
            max_link_um=8.5,
            max_neighbors=8,
            edge_cost_threshold=0.91,
            node_score_weight=0.18,
            notes="Lower thresholds and contrast-normalized proposals.",
        ),
        "high_motion_gap": RouteConfig(
            label="high_motion_gap",
            detector_sources=("dog_bright", "dog_dark", "dog_abs", "local_contrast", "hmax"),
            threshold_quantiles=(0.9988, 0.9993, 0.99965),
            min_distance_um=2.8,
            fusion_radius_um=2.4,
            max_points_per_t=200,
            max_link_um=10.0,
            max_gap=2,
            gap_link_um=14.0,
            max_neighbors=10,
            edge_cost_threshold=0.95,
            notes="Allows conservative 2-frame gap links for dropout/high motion.",
        ),
        "division_likely": RouteConfig(
            label="division_likely",
            detector_sources=("dog_bright", "dog_dark", "dog_abs", "local_contrast", "hmax", "components"),
            threshold_quantiles=(0.9988, 0.99925, 0.99955),
            min_distance_um=2.4,
            fusion_radius_um=2.0,
            max_points_per_t=240,
            max_link_um=8.5,
            max_neighbors=10,
            edge_cost_threshold=0.90,
            enable_divisions=True,
            division_cost_threshold=1.08,
            division_requires_unlinked_parent=False,
            division_requires_child_future=True,
            division_child_max_um=7.5,
            division_child_min_sep_um=3.0,
            division_child_max_sep_um=8.0,
            notes="Precision-first fork selection on top of high-recall candidates.",
        ),
    }


def _sample_timepoints(start: int, end: int, max_samples: int) -> tuple[int, ...]:
    if end <= start:
        return (start,)
    n = min(max_samples, end - start)
    return tuple(int(v) for v in np.linspace(start, end - 1, n).round())


def _robust_unit(x: np.ndarray, low_q: float = 0.001, high_q: float = 0.999) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.quantile(finite, [low_q, high_q]).astype(np.float32)
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return np.clip((x.astype(np.float32, copy=False) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def _candidate_coords(score: np.ndarray, quantile: float, limit: int = 400) -> np.ndarray:
    threshold = float(np.quantile(score, quantile))
    local = maximum_filter(score, size=(3, 9, 9), mode="nearest")
    coords = np.argwhere((score >= threshold) & (score == local))
    if len(coords) <= limit:
        return coords.astype(np.float32, copy=False)
    values = score[coords[:, 0], coords[:, 1], coords[:, 2]]
    keep = np.argsort(values)[-limit:]
    return coords[keep].astype(np.float32, copy=False)


def _median_nn_um(coords: np.ndarray, scale_zyx: tuple[float, float, float]) -> float:
    if len(coords) < 3:
        return float("inf")
    physical = coords * np.asarray(scale_zyx, dtype=np.float32)
    tree = cKDTree(physical)
    dists, _ = tree.query(physical, k=2)
    return float(np.median(dists[:, 1]))


def _division_motif_rate(
    prev_coords: np.ndarray,
    curr_coords: np.ndarray,
    scale_zyx: tuple[float, float, float],
) -> float:
    if len(prev_coords) == 0 or len(curr_coords) < 2:
        return 0.0
    curr_phys = curr_coords * np.asarray(scale_zyx, dtype=np.float32)
    prev_phys = prev_coords * np.asarray(scale_zyx, dtype=np.float32)
    tree = cKDTree(curr_phys)
    count = 0
    for parent in prev_phys[:400]:
        nearby = tree.query_ball_point(parent, r=9.0)
        if len(nearby) < 2:
            continue
        pts = curr_phys[nearby]
        if len(pts) > 20:
            pts = pts[:20]
        delta = pts[:, None, :] - pts[None, :, :]
        sep = np.sqrt(np.sum(delta * delta, axis=2))
        if np.any((sep >= 2.0) & (sep <= 11.0)):
            count += 1
    return float(count / max(1, len(prev_coords)))


def profile_scenario(
    dataset: str,
    array: Any,
    attrs: dict[str, Any] | None,
    start: int = 0,
    end: int | None = None,
    max_samples: int = 5,
    estimated_total_nodes: float | None = None,
) -> ScenarioProfile:
    """Compute cheap image/proposal features for route selection."""

    total_t = int(array.shape[0])
    end = total_t if end is None else min(int(end), total_t)
    start = max(0, min(int(start), max(0, end - 1)))
    samples = _sample_timepoints(start, end, max_samples)
    scale_zyx = parse_scale(attrs)
    baseline = BaselineConfig(polarity="both")

    foreground_fracs: list[float] = []
    bright_counts: list[int] = []
    dark_counts: list[int] = []
    both_counts: list[int] = []
    nn_values: list[float] = []
    p999_values: list[float] = []
    sample_coords: list[np.ndarray] = []
    sampled_norms: list[np.ndarray] = []

    for t in samples:
        volume = np.asarray(array[t])
        norm = normalize_volume(volume, attrs, baseline)
        sampled_norms.append(norm[::4, ::8, ::8])
        p999_values.append(float(np.quantile(volume[::4, ::8, ::8], 0.999)))
        foreground_fracs.append(float(np.mean(norm > 0.35)))

        bright = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="bright")))
        dark = _robust_unit(dog_response(norm, scale_zyx, replace(baseline, polarity="dark")))
        both = np.maximum(bright, dark)
        bcoords = _candidate_coords(bright, 0.9992)
        dcoords = _candidate_coords(dark, 0.9992)
        coords = _candidate_coords(both, 0.9988)
        bright_counts.append(len(bcoords))
        dark_counts.append(len(dcoords))
        both_counts.append(len(coords))
        sample_coords.append(coords)
        nn_values.append(_median_nn_um(coords, scale_zyx))

    temporal_delta = 0.0
    if len(sampled_norms) >= 2:
        deltas = [float(np.mean(np.abs(b - a))) for a, b in zip(sampled_norms[:-1], sampled_norms[1:])]
        temporal_delta = float(np.mean(deltas))

    motif_rates = [
        _division_motif_rate(a, b, scale_zyx)
        for a, b in zip(sample_coords[:-1], sample_coords[1:])
    ]

    return ScenarioProfile(
        dataset=dataset,
        t_start=start,
        t_end=end,
        sampled_timepoints=samples,
        foreground_fraction=float(np.mean(foreground_fracs)) if foreground_fracs else 0.0,
        bright_candidate_density=float(np.mean(bright_counts)) if bright_counts else 0.0,
        dark_candidate_density=float(np.mean(dark_counts)) if dark_counts else 0.0,
        both_candidate_density=float(np.mean(both_counts)) if both_counts else 0.0,
        median_nn_um=float(np.nanmedian(nn_values)) if nn_values else float("inf"),
        temporal_delta=temporal_delta,
        division_motif_rate=float(np.mean(motif_rates)) if motif_rates else 0.0,
        intensity_p999=float(np.mean(p999_values)) if p999_values else 0.0,
        estimated_total_nodes=estimated_total_nodes,
    )


def route_for_profile(
    profile: ScenarioProfile,
    configs: dict[str, RouteConfig] | None = None,
    forced_label: str | None = None,
    allow_division_route: bool = False,
) -> RouteDecision:
    """Choose a route. The heuristic labels become training targets later."""

    configs = configs or default_route_configs()
    if forced_label and forced_label != "auto":
        if forced_label not in configs:
            raise ValueError(f"Unknown forced route {forced_label!r}; expected one of {sorted(configs)}")
        return RouteDecision(
            label=forced_label,
            confidence=1.0,
            reason="forced by CLI",
            alternatives=tuple(k for k in configs if k != forced_label),
            config=configs[forced_label],
        )

    scores = {label: 0.0 for label in configs}
    reasons: list[str] = []
    density = profile.both_candidate_density
    dark_ratio = profile.dark_candidate_density / max(1.0, profile.bright_candidate_density)

    clean_signal = (
        profile.foreground_fraction < 0.06
        and profile.temporal_delta < 0.04
        and profile.intensity_p999 > 256
    )
    if clean_signal:
        scores["clean_point"] += 2.4
        reasons.append("low foreground/low motion/high signal")

    if (
        profile.foreground_fraction > 0.16
        or density >= 180 and profile.foreground_fraction > 0.08
        or profile.median_nn_um < 3.5 and profile.foreground_fraction > 0.08
    ):
        scores["dense_fusion"] += 2.2
        reasons.append("crowded/high foreground")
    elif not clean_signal:
        scores["clean_point"] += 1.0

    if dark_ratio > 1.25 or profile.intensity_p999 < 64:
        scores["low_signal"] += 1.8
        reasons.append("dark/low-signal proposals dominate")

    if profile.temporal_delta > 0.075:
        scores["high_motion_gap"] += 1.4
        reasons.append("large sampled temporal change")

    if profile.division_motif_rate > 0.035 and not clean_signal:
        scores["division_likely"] += 1.7
        reasons.append("division-like close-pair motifs")

    if profile.estimated_total_nodes and profile.estimated_total_nodes > 30000:
        scores["dense_fusion"] += 0.5
        reasons.append("high estimated total node count")

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not allow_division_route and ordered and ordered[0][0] == "division_likely":
        ordered = [row for row in ordered if row[0] != "division_likely"] + [
            row for row in ordered if row[0] == "division_likely"
        ]
        reasons.append("division route held for explicit CV gate")
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = 0.50 + min(0.49, max(0.0, best_score - second_score) / 4.0)
    if best_score <= 0:
        best_label = "clean_point"
        confidence = 0.55
        reasons.append("default clean route")

    return RouteDecision(
        label=best_label,
        confidence=float(confidence),
        reason="; ".join(reasons) or "default route",
        alternatives=tuple(label for label, _ in ordered[1:]),
        config=configs[best_label],
    )
