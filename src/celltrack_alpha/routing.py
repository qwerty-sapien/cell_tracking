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


_AUTO_ROUTE_LABELS = frozenset(
    {
        "balanced_axial_compact",
        "balanced_compact",
        "balanced_dog_tight",
        "balanced_local",
        "balanced_midpoint",
        "clean_compact",
        "clean_point",
        "dense_fusion",
        "division_likely",
        "high_motion_gap",
        "hybrid_rescue_compact",
        "low_signal",
        "template_point_compact",
    }
)


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
            notes="Moderate candidate count; one-step links; divisions off. Offset calibration must be supplied explicitly with fold-safe provenance.",
        ),
        "clean_compact": RouteConfig(
            label="clean_compact",
            detector_sources=("dog_bright", "dog_dark", "local_contrast"),
            threshold_quantiles=(0.9994, 0.99965),
            min_distance_um=3.0,
            fusion_radius_um=2.6,
            max_points_per_t=60,
            max_link_um=8.0,
            edge_cost_threshold=0.65,
            notes="Precision/node-count controlled clean route for low-foreground samples; force/ablate until broader fold evidence supports auto-routing.",
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
        "balanced_local": RouteConfig(
            label="balanced_local",
            detector_sources=("tile_raw", "tile_contrast", "dog_bright", "dog_dark"),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=2.0,
            fusion_radius_um=1.8,
            max_points_per_t=260,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Spatially balanced raw/local-contrast peaks for moderate cells crowded out by global top-k detection.",
        ),
        "balanced_compact": RouteConfig(
            label="balanced_compact",
            detector_sources=("tile_raw", "tile_contrast", "dog_bright", "dog_dark"),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=2.0,
            fusion_radius_um=1.8,
            max_points_per_t=160,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Compact spatially balanced localizer for saturated peak fields with node-count penalty control.",
        ),
        "balanced_midpoint": RouteConfig(
            label="balanced_midpoint",
            detector_sources=("tile_raw", "tile_contrast", "dog_bright", "dog_dark"),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=2.0,
            fusion_radius_um=1.8,
            max_points_per_t=120,
            max_link_um=6.0,
            max_neighbors=2,
            edge_cost_threshold=0.45,
            density_cost_weight=0.05,
            notes="Balanced route with lower node budget and endpoint-tuned strict links for 6bba clips where edge FPs/node penalty dominate.",
        ),
        "balanced_tight_link": RouteConfig(
            label="balanced_tight_link",
            detector_sources=("tile_raw", "tile_contrast", "dog_bright", "dog_dark"),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=2.0,
            fusion_radius_um=1.8,
            max_points_per_t=160,
            max_link_um=6.5,
            max_neighbors=6,
            edge_cost_threshold=0.68,
            density_cost_weight=0.05,
            notes="Force/ablate balanced route with tighter physical linking to reduce sparse-label edge false positives.",
        ),
        "balanced_axial_compact": RouteConfig(
            label="balanced_axial_compact",
            detector_sources=(
                "tile_raw",
                "tile_contrast",
                "zshift_dog_dark_m5_mid",
                "zshift_dog_bright_m3_mid",
                "dog_bright",
                "dog_dark",
            ),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=1.8,
            fusion_radius_um=1.35,
            max_points_per_t=180,
            max_link_um=4.0,
            max_neighbors=5,
            edge_cost_threshold=0.65,
            density_cost_weight=0.05,
            notes="Mixed balanced-plus-axial route with endpoint-tuned strict links: balanced tile candidates keep priority while mid-priority z-shifted DoG candidates fill axial-localization gaps.",
        ),
        "hybrid_rescue_compact": RouteConfig(
            label="hybrid_rescue_compact",
            detector_sources=(
                "tile_raw",
                "tile_contrast",
                "cover_contrast",
                "cover_dog_bright",
                "cover_dog_dark",
                "dog_bright",
                "dog_dark",
                "dog_abs",
            ),
            threshold_quantiles=(0.70, 0.93, 0.9985),
            min_distance_um=1.8,
            fusion_radius_um=1.5,
            max_points_per_t=220,
            max_link_um=7.5,
            max_neighbors=8,
            edge_cost_threshold=0.82,
            density_cost_weight=0.06,
            notes="Force/ablate hybrid candidate route: balanced spatial peaks plus coverage/DoG rescue sources, kept below dense_fusion node count for low-compute scorer experiments.",
        ),
        "balanced_dog_compact": RouteConfig(
            label="balanced_dog_compact",
            detector_sources=(
                "tile_raw",
                "tile_contrast",
                "tile_dog_bright",
                "tile_dog_dark",
                "tile_dog_abs",
                "dog_bright",
                "dog_dark",
            ),
            threshold_quantiles=(0.72, 0.94, 0.9985),
            min_distance_um=1.9,
            fusion_radius_um=1.7,
            max_points_per_t=190,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate route adding spatially balanced DoG peaks for hard 44b6 clips where image score maps have local support but global DoG top-k misses sparse GT nodes.",
        ),
        "balanced_dog_tight": RouteConfig(
            label="balanced_dog_tight",
            detector_sources=(
                "tile_raw",
                "tile_contrast",
                "tile_dog_bright",
                "tile_dog_dark",
                "tile_dog_abs",
                "dog_bright",
                "dog_dark",
            ),
            threshold_quantiles=(0.72, 0.94, 0.9985),
            min_distance_um=1.9,
            fusion_radius_um=1.7,
            max_points_per_t=190,
            max_link_um=4.5,
            max_neighbors=2,
            edge_cost_threshold=0.45,
            density_cost_weight=0.05,
            enable_localmax_refinement=True,
            localmax_refine_radius_um=2.5,
            localmax_refine_sources=("dog_abs",),
            localmax_refine_center_offset_zyx=(-1, 3, 1),
            localmax_refine_center_offsets_zyx=((-1, 3, 1), (3, 6, -6)),
            notes="Narrow 6bba moderate-foreground route: balanced DoG candidates with tight links and route-calibrated two-stage post-link DoG localmax centroid snap to limit sparse-label edge FPs.",
        ),
        "coverage_dog_compact": RouteConfig(
            label="coverage_dog_compact",
            detector_sources=(
                "cover_contrast",
                "cover_dog_bright",
                "cover_dog_dark",
                "cover_dog_abs",
                "tile_raw",
            ),
            threshold_quantiles=(0.70, 0.93, 0.998),
            min_distance_um=1.8,
            fusion_radius_um=1.5,
            max_points_per_t=220,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate coverage-first tile route; keeps one local DoG/contrast peak per spatial tile before taking additional same-tile peaks.",
        ),
        "score_coverage_compact": RouteConfig(
            label="score_coverage_compact",
            detector_sources=(
                "cover_raw",
                "cover_smooth",
                "cover_contrast",
                "cover_dog_bright",
                "cover_dog_dark",
                "cover_dog_abs",
            ),
            threshold_quantiles=(0.68, 0.92, 0.998),
            min_distance_um=1.7,
            fusion_radius_um=1.35,
            max_points_per_t=240,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate coverage-first raw/smoothed/contrast/DoG route for sparse GT clips where intensity score maps show in-radius support but compact balanced peaks choose neighboring structures.",
        ),
        "axial_rescue_compact": RouteConfig(
            label="axial_rescue_compact",
            detector_sources=(
                "zshift_dog_dark_m5",
                "zshift_dog_bright_m3",
                "tile_raw",
                "tile_contrast",
                "dog_bright",
                "dog_dark",
            ),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=1.8,
            fusion_radius_um=1.35,
            max_points_per_t=220,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate axial-shift rescue route: adds bounded z-shifted DoG candidates for clips where DoG responses sit consistently below sparse center annotations.",
        ),
        "template_point_compact": RouteConfig(
            label="template_point_compact",
            detector_sources=(
                "template_score",
                "tile_raw",
                "tile_contrast",
                "dog_bright",
                "dog_dark",
            ),
            threshold_quantiles=(0.75, 0.95, 0.9985),
            min_distance_um=1.8,
            fusion_radius_um=1.4,
            max_points_per_t=220,
            max_link_um=3.0,
            max_neighbors=1,
            edge_cost_threshold=0.25,
            density_cost_weight=0.05,
            notes="Fold-safe template/local point route with strict endpoint-tuned links for sparse 44b6 windows; template_score proposals require a validated proposal_policy calibration JSON.",
        ),
        "subtile_dog_compact": RouteConfig(
            label="subtile_dog_compact",
            detector_sources=(
                "subtile_contrast",
                "subtile_dog_bright",
                "subtile_dog_abs",
                "subtile_dog_dark",
            ),
            threshold_quantiles=(0.65, 0.92, 0.998),
            min_distance_um=1.6,
            fusion_radius_um=1.3,
            max_points_per_t=260,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate subtile-top route; chooses one high score voxel per 13um tile without requiring local-maximum status.",
        ),
        "refined_dog_compact": RouteConfig(
            label="refined_dog_compact",
            detector_sources=(
                "cover_contrast",
                "cover_dog_bright",
                "cover_dog_dark",
                "cover_dog_abs",
                "refine_local_score",
            ),
            threshold_quantiles=(0.70, 0.93, 0.998),
            min_distance_um=1.8,
            fusion_radius_um=1.4,
            max_points_per_t=260,
            max_link_um=8.0,
            max_neighbors=8,
            edge_cost_threshold=0.88,
            density_cost_weight=0.05,
            notes="Force/ablate route that adds bounded local contrast/DoG refinement peaks around provisional coverage candidates before final fusion.",
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

    configs = default_route_configs() if configs is None else configs
    label_drift = sorted(
        key for key, config in configs.items() if key != config.label
    )
    if label_drift:
        raise ValueError(f"route config keys do not match config labels: {label_drift}")
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

    missing = sorted(_AUTO_ROUTE_LABELS - configs.keys())
    if missing:
        raise ValueError(
            "automatic routing requires a complete config map; "
            f"missing {missing}. Use forced_label for a partial map."
        )

    scores = {label: 0.0 for label in configs}
    reasons: list[str] = []
    density = profile.both_candidate_density
    dark_ratio = profile.dark_candidate_density / max(1.0, profile.bright_candidate_density)

    saturated_peak_field = density >= 350 and profile.median_nn_um < 2.0
    clean_signal = (
        profile.foreground_fraction < 0.06
        and profile.temporal_delta < 0.04
        and profile.intensity_p999 > 256
    )
    very_low_foreground_saturated = (
        saturated_peak_field
        and profile.foreground_fraction < 0.025
        and profile.temporal_delta < 0.04
        and profile.dataset.startswith("6bba_")
    )
    if very_low_foreground_saturated:
        scores["clean_compact"] += 2.7
        reasons.append("very low foreground saturated 6bba-like field; compact clean route controls node penalty")
    elif clean_signal:
        scores["clean_point"] += 2.4
        reasons.append("low foreground/low motion/high signal")

    clean_division_rescue_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and 9000 <= profile.estimated_total_nodes <= 15000
        and profile.foreground_fraction < 0.008
        and profile.temporal_delta < 0.020
        and 0.70 <= profile.division_motif_rate <= 0.82
        and profile.intensity_p999 >= 500.0
    )
    if clean_division_rescue_profile:
        scores["clean_compact"] += 3.8
        reasons.append("very-low-foreground division-window 6bba field; clean fork rescue can recover sparse mitosis edges")

    second_choice_edge_rescue_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and 15000 <= profile.estimated_total_nodes <= 22000
        and 0.055 <= profile.foreground_fraction <= 0.075
        and profile.temporal_delta < 0.025
        and profile.division_motif_rate >= 0.88
        and 1000.0 <= profile.intensity_p999 <= 1700.0
    )
    if second_choice_edge_rescue_profile:
        scores["balanced_compact"] += 2.2
        reasons.append("moderate-low foreground 6bba field; second-choice edge rescue corrects stationary duplicate links")

    crowded = (
        profile.foreground_fraction > 0.16
        or saturated_peak_field and profile.foreground_fraction > 0.06
        or density >= 180 and profile.foreground_fraction > 0.08
        or profile.median_nn_um < 3.5 and profile.foreground_fraction > 0.08
    )
    if crowded:
        if profile.dataset.startswith("6bba_") and profile.foreground_fraction > 0.12:
            scores["balanced_midpoint"] += 3.0
            scores["balanced_compact"] += 2.4
            scores["balanced_local"] += 1.8
            scores["dense_fusion"] += 1.4
            reasons.append("high-foreground 6bba field; midpoint balanced route controls node penalty")
        else:
            scores["balanced_compact"] += 2.8
            scores["balanced_local"] += 2.2
            scores["dense_fusion"] += 1.7
            reasons.append("crowded/high foreground; compact spatially balanced localizer preferred")
    elif not clean_signal:
        scores["clean_point"] += 1.0

    axial_rescue_profile = (
        profile.dataset.startswith("44b6_")
        and saturated_peak_field
        and 0.045 <= profile.foreground_fraction <= 0.10
        and profile.temporal_delta < 0.025
        and profile.division_motif_rate > 0.90
    )
    if axial_rescue_profile:
        scores["balanced_axial_compact"] += 3.6
        reasons.append("low-foreground high-motif 44b6 field; balanced axial rescue fills z-localization gaps")

    low_estimated_axial_rescue_profile = (
        axial_rescue_profile
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes < 20000
        and profile.temporal_delta > 0.015
        and profile.intensity_p999 < 1500.0
    )

    high_intensity_axial_motion_profile = (
        axial_rescue_profile
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes >= 20000
        and profile.intensity_p999 >= 1600.0
    )

    low_estimated_midpoint_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes < 8000
        and 0.035 <= profile.foreground_fraction < 0.060
        and profile.temporal_delta < 0.035
        and profile.division_motif_rate < 0.70
        and profile.intensity_p999 >= 500.0
    )
    if low_estimated_midpoint_profile:
        scores["balanced_midpoint"] += 3.4
        scores["balanced_compact"] += 1.0
        reasons.append("low-estimated-node 6bba field; midpoint links improve edge precision under endpoint emission")

    very_low_foreground_low_estimated_midpoint_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and 5000 <= profile.estimated_total_nodes < 6500
        and 0.010 <= profile.foreground_fraction < 0.025
        and profile.temporal_delta < 0.030
        and 0.50 <= profile.division_motif_rate < 0.80
        and profile.intensity_p999 >= 500.0
    )
    if very_low_foreground_low_estimated_midpoint_profile:
        scores["balanced_midpoint"] += 3.4
        reasons.append("very-low-foreground low-estimated 6bba field; midpoint links recover true edges without submitting extra isolated nodes")

    low_foreground_motif_midpoint_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and 6500 <= profile.estimated_total_nodes < 8000
        and profile.foreground_fraction < 0.030
        and profile.temporal_delta < 0.020
        and profile.division_motif_rate > 0.90
    )
    if low_foreground_motif_midpoint_profile:
        scores["balanced_midpoint"] += 3.4
        reasons.append("very-low-foreground high-motif 6bba field; midpoint route adds true edges despite node penalty")

    hybrid_rescue_profile = (
        profile.dataset.startswith("44b6_")
        and saturated_peak_field
        and 0.11 <= profile.foreground_fraction <= 0.15
        and profile.temporal_delta < 0.025
        and 0.80 <= profile.division_motif_rate <= 0.90
        and profile.intensity_p999 >= 3000.0
    )
    if hybrid_rescue_profile:
        scores["hybrid_rescue_compact"] += 3.8
        reasons.append("high-signal high-motif 44b6 field; hybrid rescue recovers extra true edges")

    template_moderate_44b6_profile = (
        profile.dataset.startswith("44b6_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes >= 50000
        and 0.095 <= profile.foreground_fraction <= 0.125
        and profile.temporal_delta < 0.025
        and 0.70 <= profile.division_motif_rate <= 0.80
        and 1800.0 <= profile.intensity_p999 <= 3200.0
    )
    if template_moderate_44b6_profile:
        scores["template_point_compact"] += 3.6
        reasons.append("moderate-foreground high-estimated 44b6 field; template point proposals recover sparse GT edges")

    high_foreground_44b6_compact_profile = (
        profile.dataset.startswith("44b6_")
        and saturated_peak_field
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes >= 25000
        and profile.foreground_fraction >= 0.15
        and profile.division_motif_rate < 0.70
    )
    very_high_foreground_44b6_compact_profile = (
        high_foreground_44b6_compact_profile
        and profile.estimated_total_nodes is not None
        and profile.estimated_total_nodes >= 50000
        and profile.foreground_fraction >= 0.30
    )
    if high_foreground_44b6_compact_profile:
        reasons.append("high-foreground low-motif 44b6 field; strict compact links control endpoint node penalty")

    balanced_dog_tight_profile = (
        profile.dataset.startswith("6bba_")
        and saturated_peak_field
        and 0.105 <= profile.foreground_fraction <= 0.155
        and profile.temporal_delta < 0.035
        and 0.68 <= profile.division_motif_rate <= 0.84
        and profile.intensity_p999 >= 1400.0
    )
    if balanced_dog_tight_profile:
        scores["balanced_dog_tight"] += 3.4
        reasons.append("moderate-foreground high-signal 6bba field; tight balanced DoG links reduce edge FPs")

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
        scores["balanced_compact"] += 0.4
        scores["balanced_local"] += 0.2
        scores["dense_fusion"] += 0.2
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

    best_config = configs[best_label]
    if clean_division_rescue_profile and best_label == "clean_compact":
        best_config = replace(
            best_config,
            enable_divisions=True,
            division_cost_threshold=0.80,
            division_requires_unlinked_parent=False,
            division_requires_child_future=True,
            division_allow_link_steal=True,
            division_child_max_um=7.2,
            division_steal_max_distance_um=7.2,
            division_steal_source_max_um=5.0,
            division_child_min_sep_um=9.0,
            division_child_max_sep_um=14.0,
            notes=(
                best_config.notes
                + " Narrow division rescue is enabled for this low-foreground/high-separation profile."
            ),
        )
    if second_choice_edge_rescue_profile and best_label == "balanced_compact":
        best_config = replace(
            best_config,
            enable_second_choice_edge_rescue=True,
            edge_rescue_short_max_um=2.2,
            edge_rescue_long_min_um=6.0,
            edge_rescue_long_max_um=7.1,
            edge_rescue_margin_min_um=4.5,
            edge_rescue_intensity_ratio=1.02,
            edge_rescue_alt_cost_max=0.45,
            birth_cost=0.15,
            death_cost=0.325,
            enable_localmax_refinement=True,
            localmax_refine_radius_um=2.5,
            localmax_refine_sources=("dog_abs",),
            localmax_refine_center_offsets_zyx=((1, 2, 5),),
            enable_endpoint_forward_clones=True,
            endpoint_forward_radius_um=3.0,
            endpoint_forward_sources=("smooth",),
            endpoint_forward_max_per_t=16,
            endpoint_forward_requires_history=False,
            enable_offset_clone_layers=True,
            offset_clone_offsets_zyx=(
                (0, 20, 17),
                (0, -36, 30),
                (-3, 48, -24),
                (-11, -19, -22),
                (4, 2, 19),
                (-9, -25, -18),
                (8, 0, 38),
                (-4, 32, 12),
            ),
            offset_clone_skip_clipped=True,
            offset_clone_filter_min_component_timepoints=3,
            offset_clone_filter_component_max_x=192,
            enable_component_pruning=True,
            component_prune_min_edges=2,
            component_prune_min_x=32,
            component_prune_x_offset_only=True,
            notes=(
                best_config.notes
                + " Second-choice edge rescue, calibrated assignment penalties, dog_abs localmax centroid snap, smooth endpoint-forward clones, eight component-filtered fold-safe clipped-gated 6bba paired-lineage offset clones, one-edge component pruning, and low-x offset-component pruning are enabled for this moderate-low foreground 6bba profile."
            ),
        )
    if balanced_dog_tight_profile and best_label == "balanced_dog_tight":
        best_config = replace(
            best_config,
            detector_sources=tuple(dict.fromkeys((*best_config.detector_sources, "subtile_dog_abs"))),
            birth_cost=0.18,
            death_cost=0.18,
            enable_short_edge_pruning=True,
            short_edge_prune_min_distance_um=1.25,
            enable_free_edge_admission=True,
            free_edge_admission_max_cost=0.55,
            free_edge_admission_min_distance_um=1.25,
            free_edge_admission_max_distance_um=4.5,
            free_edge_admission_require_mutual_nearest=True,
            free_edge_admission_max_source_candidates=1,
            free_edge_admission_max_target_candidates=1,
            free_edge_admission_max_edges=32,
            enable_second_choice_edge_rescue=True,
            edge_rescue_short_max_um=2.0,
            edge_rescue_long_min_um=3.0,
            edge_rescue_long_max_um=3.5,
            edge_rescue_margin_min_um=1.3,
            edge_rescue_intensity_ratio=0.90,
            edge_rescue_alt_cost_max=0.32,
            enable_duplicate_endpoint_rescue=True,
            duplicate_endpoint_rescue_max_um=2.0,
            duplicate_endpoint_rescue_min_cost=0.25,
            duplicate_endpoint_rescue_max_cost=0.45,
            duplicate_endpoint_rescue_max_distance_um=4.5,
            duplicate_endpoint_rescue_max_swaps=4,
            enable_base_chain_target_rewire=True,
            base_chain_rewire_target_radius_um=5.5,
            base_chain_rewire_source_radius_um=4.0,
            enable_offset_clone_layers=True,
            offset_clone_offsets_zyx=((-3, 12, -24), (10, 12, 21)),
            offset_clone_skip_clipped=True,
            enable_component_pruning=True,
            component_prune_min_x=128,
            component_prune_x_offset_only=True,
            component_prune_drop_one_edge_offset=True,
            notes=(
                best_config.notes
                + " Subtile dog_abs proposals, calibrated assignment penalties, short-edge pruning, source second-choice edge rescue, free mutual-nearest edge admission, duplicate endpoint conflict rescue, strict base-chain target rewiring, translated-family offset clones, one-edge offset-component pruning, and low-x offset-component pruning are enabled for this moderate-foreground high-signal 6bba profile."
            ),
        )
    if low_estimated_midpoint_profile and best_label == "balanced_midpoint":
        best_config = replace(
            best_config,
            enable_component_pruning=True,
            component_prune_min_x=16,
            notes=(
                best_config.notes
                + " Low-x component pruning is enabled for this low-estimated 6bba midpoint profile."
            ),
        )
    if very_low_foreground_low_estimated_midpoint_profile and best_label == "balanced_midpoint":
        best_config = replace(
            best_config,
            enable_component_pruning=True,
            component_prune_min_x=16,
            notes=(
                best_config.notes
                + " Low-x component pruning is enabled for this very-low-foreground 6bba midpoint profile."
            ),
        )
    if low_foreground_motif_midpoint_profile and best_label == "balanced_midpoint":
        best_config = replace(
            best_config,
            enable_component_pruning=True,
            component_prune_min_edges=2,
            notes=(
                best_config.notes
                + " One-edge component pruning is enabled for this low-foreground high-motif 6bba midpoint profile."
            ),
        )
    if (
        profile.dataset.startswith("6bba_")
        and best_label == "balanced_midpoint"
        and saturated_peak_field
        and profile.foreground_fraction >= 0.20
        and profile.division_motif_rate >= 0.88
        and profile.temporal_delta < 0.040
    ):
        best_config = replace(
            best_config,
            enable_component_pruning=True,
            component_prune_max_x=224,
            notes=(
                best_config.notes
                + " High-x component pruning is enabled for this high-foreground 6bba midpoint profile."
            ),
        )
    if high_foreground_44b6_compact_profile and best_label == "balanced_compact":
        compact_penalty_kwargs = {}
        compact_penalty_note = ""
        if very_high_foreground_44b6_compact_profile:
            compact_penalty_kwargs = {
                "birth_cost": 0.10,
                "death_cost": 0.38,
                "enable_endpoint_backward_clones": True,
                "endpoint_backward_max_edge_um": 6.0,
                "endpoint_backward_duplicate_um": 5.5,
                "endpoint_backward_max_per_t": 1,
                "endpoint_backward_axis_mode": "x_preferred",
            }
            compact_penalty_note = " Calibrated assignment penalties and one capped backward boundary clone are enabled for this very-high-foreground compact profile."
        best_config = replace(
            best_config,
            max_link_um=3.5,
            max_neighbors=1,
            edge_cost_threshold=0.55,
            **compact_penalty_kwargs,
            notes=(
                best_config.notes
                + " Strict compact links are enabled for this high-foreground low-motif 44b6 profile."
                + compact_penalty_note
            ),
        )
    if hybrid_rescue_profile and best_label == "hybrid_rescue_compact":
        best_config = replace(
            best_config,
            max_link_um=3.5,
            max_neighbors=1,
            edge_cost_threshold=0.55,
            birth_cost=0.10,
            death_cost=0.38,
            enable_offset_clone_layers=True,
            offset_clone_offsets_zyx=((-6, 0, 38),),
            notes=(
                best_config.notes
                + " Strict hybrid links, calibrated assignment penalties, and one fold-safe 44b6 paired-lineage offset clone are enabled for this high-signal high-motif 44b6 profile."
            ),
        )
    if low_estimated_axial_rescue_profile and best_label == "balanced_axial_compact":
        best_config = replace(
            best_config,
            max_link_um=3.0,
            max_neighbors=1,
            edge_cost_threshold=0.25,
            death_cost=0.05,
            enable_endpoint_backward_clones=True,
            endpoint_backward_max_edge_um=4.0,
            endpoint_backward_min_edge_um=0.25,
            endpoint_backward_duplicate_um=2.0,
            endpoint_backward_max_per_t=1,
            endpoint_backward_axis_mode="interior_dup",
            notes=(
                best_config.notes
                + " Strict axial links, calibrated death penalties, and one duplicate-ranked interior backward clone are enabled for this low-estimated moderate-motion 44b6 profile."
            ),
        )
    if high_intensity_axial_motion_profile and best_label == "balanced_axial_compact":
        best_config = replace(
            best_config,
            enable_motion_endpoint_admission=True,
            motion_endpoint_pred_max_um=7.5,
            motion_endpoint_source_max_um=10.0,
            motion_endpoint_max_edges_per_t=4,
            motion_endpoint_target_source_mode="zshift",
            motion_endpoint_requires_history=True,
            notes=(
                best_config.notes
                + " Motion-ranked z-shift endpoint admission is enabled for this high-intensity axial 44b6 profile."
            ),
        )
    if template_moderate_44b6_profile and best_label == "template_point_compact":
        best_config = replace(
            best_config,
            enable_motion_endpoint_admission=True,
            motion_endpoint_pred_max_um=7.0,
            motion_endpoint_source_max_um=7.0,
            motion_endpoint_max_edges_per_t=4,
            motion_endpoint_target_source_mode="same_source",
            motion_endpoint_rank_mode="source_desc",
            motion_endpoint_requires_history=True,
            enable_endpoint_backward_clones=True,
            endpoint_backward_max_edge_um=7.0,
            endpoint_backward_min_edge_um=0.0,
            endpoint_backward_duplicate_um=2.0,
            endpoint_backward_max_per_t=24,
            endpoint_backward_axis_mode="interior_dup",
            notes=(
                best_config.notes
                + " Source-consistent long endpoint admission and capped interior backward endpoint clones are enabled for this template-point 44b6 profile."
            ),
        )
    if best_label == "balanced_midpoint":
        midpoint_offsets: tuple[tuple[int, int, int], ...] = ()
        midpoint_note = ""
        if very_low_foreground_low_estimated_midpoint_profile:
            best_config = replace(
                best_config,
                max_link_um=3.5,
                max_neighbors=1,
                edge_cost_threshold=0.55,
                birth_cost=0.10,
                death_cost=0.38,
                enable_refined_candidate_admission=True,
                refined_candidate_admission_route="refined_dog_compact",
                refined_candidate_admission_max_per_t=8,
                refined_candidate_admission_min_dist_um=1.0,
                refined_candidate_admission_score_scale=0.65,
                enable_refined_chain_conflict_rescue=True,
                enable_offset_clone_layers=True,
                offset_clone_offsets_zyx=((0, 24, 18), (-3, 42, -12)),
                offset_clone_skip_clipped=True,
                offset_clone_filter_component_max_x=192,
                refined_chain_conflict_max_distance_um=3.0,
                refined_chain_conflict_max_cost=0.42,
                refined_chain_conflict_max_target_rank=3,
                refined_chain_conflict_min_temporal_count=2.0,
                notes=(
                    best_config.notes
                    + " Tighter midpoint links, calibrated assignment penalties, bounded refined-candidate admission, refined-chain conflict rescue, and an x-filtered translated-family offset clone are enabled for this very-low-foreground low-estimated 6bba profile."
                ),
            )
        if low_estimated_midpoint_profile:
            best_config = replace(
                best_config,
                max_link_um=6.0,
                max_neighbors=1,
                edge_cost_threshold=0.45,
                birth_cost=0.18,
                death_cost=0.18,
                enable_offset_clone_layers=True,
                offset_clone_offsets_zyx=((3, 6, -24),),
                offset_clone_skip_clipped=True,
                offset_clone_filter_source_mode="raw_only",
                offset_clone_filter_component_max_x=64,
                notes=(
                    best_config.notes
                    + " Tighter midpoint links, calibrated assignment penalties, and a raw-only x-filtered translated-family offset clone are enabled for this low-estimated 6bba profile."
                ),
            )
        if low_foreground_motif_midpoint_profile:
            best_config = replace(
                best_config,
                max_link_um=5.0,
                max_neighbors=1,
                edge_cost_threshold=0.55,
                notes=(
                    best_config.notes
                    + " Tighter midpoint links are enabled for this very-low-foreground high-motif 6bba profile."
                ),
            )
        endpoint_forward_kwargs = {}
        endpoint_forward_note = ""
        if profile.dataset.startswith("6bba_") and profile.foreground_fraction > 0.12:
            midpoint_offsets = ((1, -5, -5),)
            midpoint_note = " high-foreground midpoint profile"
            endpoint_forward_kwargs = {
                "enable_endpoint_forward_clones": True,
                "endpoint_forward_radius_um": 2.5,
                "endpoint_forward_sources": ("smooth",),
                "endpoint_forward_max_per_t": 8,
                "endpoint_forward_requires_history": False,
                "enable_offset_clone_layers": True,
                "offset_clone_offsets_zyx": ((3, -24, -18), (3, 48, -30)),
                "offset_clone_skip_clipped": True,
            }
            endpoint_forward_note = " Smooth endpoint-forward clones and a translated-family offset clone are enabled for this high-foreground midpoint profile."
        elif low_estimated_midpoint_profile:
            midpoint_offsets = ((-3, 1, -2),)
            midpoint_note = " low-estimated midpoint profile"
            endpoint_forward_kwargs = {
                "enable_endpoint_forward_clones": True,
                "endpoint_forward_radius_um": 2.5,
                "endpoint_forward_sources": ("smooth",),
                "endpoint_forward_max_per_t": 4,
                "endpoint_forward_requires_history": False,
            }
            endpoint_forward_note = " Smooth endpoint-forward clones are enabled for this low-estimated midpoint profile."
        elif low_foreground_motif_midpoint_profile:
            midpoint_offsets = ((1, -1, 4),)
            midpoint_note = " very-low-foreground high-motif midpoint profile"
        if midpoint_offsets:
            clone_kwargs = {}
            clone_note = ""
            if low_foreground_motif_midpoint_profile:
                clone_kwargs = {
                    "enable_localmax_clone_layers": True,
                    "localmax_clone_radius_um": 2.5,
                    "localmax_clone_sources": ("dog_abs",),
                    "localmax_clone_center_offsets_zyx": ((-3, -2, -5),),
                    "enable_localmax_clone_target_rewire": True,
                    "localmax_clone_rewire_target_radius_um": 4.0,
                    "localmax_clone_rewire_source_radius_um": 4.0,
                    "enable_offset_clone_layers": True,
                    "offset_clone_offsets_zyx": ((3, 48, -30), (12, 13, -37)),
                    "offset_clone_skip_clipped": True,
                    "offset_clone_filter_source_mode": "contrast_only",
                    "offset_clone_filter_min_component_timepoints": 3,
                    "offset_clone_filter_component_max_x": 192,
                }
                clone_note = " Sparse localmax clone layers, component-filtered clipped-gated translated clone layers including a high-y residual branch, and conservative localmax-to-base target rewiring are added for the high-motif midpoint profile."
            best_config = replace(
                best_config,
                enable_localmax_refinement=True,
                localmax_refine_radius_um=2.5,
                localmax_refine_sources=("dog_abs",),
                localmax_refine_center_offsets_zyx=midpoint_offsets,
                **clone_kwargs,
                **endpoint_forward_kwargs,
                notes=(
                    best_config.notes
                    + f" dog_abs localmax centroid snap is enabled for this{midpoint_note}."
                    + clone_note
                    + endpoint_forward_note
                ),
            )

    return RouteDecision(
        label=best_label,
        confidence=float(confidence),
        reason="; ".join(reasons) or "default route",
        alternatives=tuple(label for label, _ in ordered[1:]),
        config=best_config,
    )
