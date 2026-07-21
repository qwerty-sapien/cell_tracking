"""Deterministic low-compute routing for the Biohub sparse hybrid tracker.

The router deliberately consumes measured video/linking summaries rather than
dataset names.  It uses a small, finite feature schema and a shallow NumPy-only
axis-aligned tree, so inference and artifact loading do not require sklearn or
pickle.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


FEATURE_SCHEMA_VERSION = "celltrack-alpha.bh-route-features.v1"
FEATURE_NAMES = (
    "is_volume",
    "log_depth_planes",
    "log_xy_area",
    "z_to_xy_spacing",
    "log_frame_count",
    "detections_q10",
    "detections_q50",
    "detections_q90",
    "empty_frame_fraction",
    "nn_um_q10",
    "nn_um_q50",
    "nn_um_q90",
    "count_cv",
    "count_jump_q90",
    "unmatched_6um_fraction",
    "unmatched_10um_fraction",
    "assignment_margin_q10",
    "assignment_margin_q50",
    "detection_confidence_q10",
    "detection_confidence_q50",
    "detection_confidence_q90",
    "border_fraction",
    "depth_edge_fraction",
    "gap_demand_fraction",
    "division_motif_rate",
    "intensity_q10",
    "intensity_q50",
    "intensity_q90",
)
MODEL_FORMAT = "celltrack-alpha.bh-route-model"
MODEL_FORMAT_VERSION = 1


class RouteModelError(RuntimeError):
    """Base error for router fitting, prediction, and artifacts."""


class FeatureValidationError(RouteModelError):
    """A detector/link summary cannot populate the locked feature schema."""


class ProvenanceError(RouteModelError):
    """A fit attempted to use an identity outside its declared train members."""


class ModelSerializationError(RouteModelError):
    """A router JSON artifact is malformed, incompatible, or tampered."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


_FEATURE_SCHEMA_CORE = {
    "names": list(FEATURE_NAMES),
    "version": FEATURE_SCHEMA_VERSION,
}
FEATURE_SCHEMA_SHA256 = _sha256(_FEATURE_SCHEMA_CORE)
FEATURE_SCHEMA = {
    **_FEATURE_SCHEMA_CORE,
    "sha256": FEATURE_SCHEMA_SHA256,
}


def _array(summary: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in summary:
            try:
                result = np.asarray(summary[name], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise FeatureValidationError(f"{name} must be numeric") from exc
            if not np.all(np.isfinite(result)):
                raise FeatureValidationError(f"{name} contains non-finite values")
            return result
    return np.empty(0, dtype=np.float64)


def _number(summary: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    values = _array(summary, *names)
    if not values.size:
        return float(default)
    if values.size != 1:
        raise FeatureValidationError(f"{names[0]} must be scalar")
    return float(values[0])


def _quantiles(values: np.ndarray, probabilities: Sequence[float]) -> tuple[float, ...]:
    if not values.size:
        return tuple(0.0 for _ in probabilities)
    return tuple(float(value) for value in np.quantile(values, probabilities, method="linear"))


def _distribution_quantiles(
    summary: Mapping[str, Any],
    distribution_names: Sequence[str],
    output_names: Sequence[str],
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    if all(name in summary for name in output_names):
        return tuple(_number(summary, name) for name in output_names)
    return _quantiles(_array(summary, *distribution_names), probabilities)


def _fraction(summary: Mapping[str, Any], scalar_name: str, *sample_names: str) -> float:
    if scalar_name in summary:
        value = _number(summary, scalar_name)
    else:
        samples = _array(summary, *sample_names)
        value = float(np.mean(samples)) if samples.size else 0.0
    if not 0.0 <= value <= 1.0:
        raise FeatureValidationError(f"{scalar_name} must be in [0, 1]")
    return value


def _validate_feature_dict(features: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name not in features:
            raise FeatureValidationError(f"missing feature {name!r}")
        try:
            value = float(features[name])
        except (TypeError, ValueError) as exc:
            raise FeatureValidationError(f"feature {name!r} must be numeric") from exc
        if not math.isfinite(value):
            raise FeatureValidationError(f"feature {name!r} must be finite")
        result[name] = value
    for name in (
        "is_volume",
        "empty_frame_fraction",
        "unmatched_6um_fraction",
        "unmatched_10um_fraction",
        "border_fraction",
        "depth_edge_fraction",
        "gap_demand_fraction",
        "division_motif_rate",
    ):
        if not 0.0 <= result[name] <= 1.0:
            raise FeatureValidationError(f"feature {name!r} must be in [0, 1]")
    return result


def extract_features(summary: Mapping[str, Any]) -> dict[str, float]:
    """Extract the locked schema from cheap detector and sparse-link summaries.

    Accepted values are plain mappings and numeric sequences.  Raw identity,
    dataset, or prefix strings are never read.  The modality/prefix signal is
    represented only by measurable geometry: dimensionality, shape, spacing,
    and frame count.
    """

    if not isinstance(summary, Mapping):
        raise FeatureValidationError("feature summary must be a mapping")
    if all(name in summary for name in FEATURE_NAMES):
        return _validate_feature_dict(summary)

    shape = _array(summary, "frame_shape_zyx", "frame_shape")
    if shape.size not in {0, 2, 3}:
        raise FeatureValidationError("frame_shape must contain two or three dimensions")
    if shape.size and (np.any(shape <= 0) or np.any(shape != np.floor(shape))):
        raise FeatureValidationError("frame_shape dimensions must be positive integers")
    if shape.size == 3:
        depth, height, width = (float(value) for value in shape)
    elif shape.size == 2:
        depth, height, width = 1.0, float(shape[0]), float(shape[1])
    else:
        depth, height, width = 1.0, 1.0, 1.0

    spacing = _array(summary, "voxel_size_um", "spacing_um")
    if spacing.size not in {0, 2, 3}:
        raise FeatureValidationError("voxel spacing must contain two or three values")
    if spacing.size and np.any(spacing <= 0):
        raise FeatureValidationError("voxel spacing must be positive")
    spatial_ndim = int(_number(summary, "spatial_ndim", default=3.0 if depth > 1.0 else 2.0))
    if spatial_ndim not in {2, 3}:
        raise FeatureValidationError("spatial_ndim must be 2 or 3")
    is_volume = float(spatial_ndim == 3 and depth > 1.0)
    if spacing.size == 3 and is_volume:
        z_to_xy = float(spacing[0] / np.median(spacing[1:]))
    else:
        z_to_xy = 1.0

    counts = _array(summary, "detections_per_frame", "detection_counts")
    if np.any(counts < 0):
        raise FeatureValidationError("detections per frame cannot be negative")
    frame_count = _number(summary, "frame_count", default=float(counts.size))
    if frame_count < 0:
        raise FeatureValidationError("frame_count cannot be negative")
    count_q10, count_q50, count_q90 = _distribution_quantiles(
        summary,
        ("detections_per_frame", "detection_counts"),
        ("detections_q10", "detections_q50", "detections_q90"),
        (0.1, 0.5, 0.9),
    )
    count_mean = float(np.mean(counts)) if counts.size else 0.0
    count_cv = _number(
        summary,
        "count_cv",
        default=float(np.std(counts) / count_mean) if count_mean > 0.0 else 0.0,
    )
    jumps = np.abs(np.diff(counts)) if counts.size > 1 else np.empty(0)
    count_jump_q90 = _number(
        summary,
        "count_jump_q90",
        default=_quantiles(jumps, (0.9,))[0],
    )

    nn_q10, nn_q50, nn_q90 = _distribution_quantiles(
        summary,
        ("nearest_neighbor_um", "nn_spacing_um"),
        ("nn_um_q10", "nn_um_q50", "nn_um_q90"),
        (0.1, 0.5, 0.9),
    )
    margin_q10, margin_q50 = _distribution_quantiles(
        summary,
        ("assignment_margins", "assignment_margin"),
        ("assignment_margin_q10", "assignment_margin_q50"),
        (0.1, 0.5),
    )
    confidence_q10, confidence_q50, confidence_q90 = _distribution_quantiles(
        summary,
        ("detection_confidences", "detection_confidence"),
        (
            "detection_confidence_q10",
            "detection_confidence_q50",
            "detection_confidence_q90",
        ),
        (0.1, 0.5, 0.9),
    )
    intensity_q10, intensity_q50, intensity_q90 = _distribution_quantiles(
        summary,
        ("detection_intensities", "intensities"),
        ("intensity_q10", "intensity_q50", "intensity_q90"),
        (0.1, 0.5, 0.9),
    )

    if "gap_demand_fraction" in summary:
        gap_fraction = _fraction(summary, "gap_demand_fraction")
    else:
        gap_lengths = _array(summary, "gap_lengths", "requested_gap_lengths")
        gap_fraction = float(np.mean(gap_lengths > 1.0)) if gap_lengths.size else 0.0
    if "division_motif_rate" in summary:
        division_rate = _fraction(summary, "division_motif_rate")
    else:
        motifs = _number(summary, "division_motif_count")
        opportunities = _number(
            summary,
            "division_motif_opportunities",
            default=float(np.sum(counts)),
        )
        division_rate = motifs / opportunities if opportunities > 0.0 else 0.0
        if not 0.0 <= division_rate <= 1.0:
            raise FeatureValidationError("derived division_motif_rate must be in [0, 1]")

    features = {
        "is_volume": is_volume,
        "log_depth_planes": math.log1p(depth),
        "log_xy_area": math.log1p(height * width),
        "z_to_xy_spacing": z_to_xy,
        "log_frame_count": math.log1p(frame_count),
        "detections_q10": count_q10,
        "detections_q50": count_q50,
        "detections_q90": count_q90,
        "empty_frame_fraction": _fraction(
            summary,
            "empty_frame_fraction",
            "empty_frame_flags",
        ) if "empty_frame_fraction" in summary or "empty_frame_flags" in summary else (
            float(np.mean(counts == 0.0)) if counts.size else 0.0
        ),
        "nn_um_q10": nn_q10,
        "nn_um_q50": nn_q50,
        "nn_um_q90": nn_q90,
        "count_cv": count_cv,
        "count_jump_q90": count_jump_q90,
        "unmatched_6um_fraction": _fraction(
            summary,
            "unmatched_6um_fraction",
            "unmatched_at_6um",
        ),
        "unmatched_10um_fraction": _fraction(
            summary,
            "unmatched_10um_fraction",
            "unmatched_at_10um",
        ),
        "assignment_margin_q10": margin_q10,
        "assignment_margin_q50": margin_q50,
        "detection_confidence_q10": confidence_q10,
        "detection_confidence_q50": confidence_q50,
        "detection_confidence_q90": confidence_q90,
        "border_fraction": _fraction(summary, "border_fraction", "border_flags"),
        "depth_edge_fraction": _fraction(
            summary,
            "depth_edge_fraction",
            "depth_edge_flags",
        ) if is_volume else 0.0,
        "gap_demand_fraction": gap_fraction,
        "division_motif_rate": division_rate,
        "intensity_q10": intensity_q10,
        "intensity_q50": intensity_q50,
        "intensity_q90": intensity_q90,
    }
    return _validate_feature_dict(features)


def feature_vector(values: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    """Return one finite vector in exact schema order."""

    if isinstance(values, Mapping):
        source: Mapping[str, Any]
        nested = values.get("features")
        if isinstance(nested, Mapping):
            source = nested
        else:
            source = values
        features = extract_features(source)
        return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError("features must be a mapping or numeric sequence") from exc
    if result.size != len(FEATURE_NAMES):
        raise FeatureValidationError(f"expected {len(FEATURE_NAMES)} features, got {result.size}")
    if not np.all(np.isfinite(result)):
        raise FeatureValidationError("features must be finite")
    return result


def construct_route_label(
    official_scores: Mapping[str, float | Mapping[str, Any]],
    *,
    runtimes: Mapping[str, float] | None = None,
    runtime_penalty: float = 0.0,
) -> str:
    """Choose the largest ``official_score - penalty * runtime`` utility.

    Candidate mappings may contain ``official_score`` and ``runtime_seconds``;
    alternatively scores and runtimes can be supplied as separate mappings.
    Exact utility ties are resolved by the ASCII/Unicode route-name ordering.
    """

    if not isinstance(official_scores, Mapping) or not official_scores:
        raise ValueError("official_scores must be a non-empty mapping")
    penalty = float(runtime_penalty)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("runtime_penalty must be finite and non-negative")
    if any(not isinstance(route, str) or not route for route in official_scores):
        raise ValueError("route names must be non-empty strings")
    ranked: list[tuple[float, str]] = []
    for route in sorted(official_scores):
        candidate = official_scores[route]
        if isinstance(candidate, Mapping):
            if "official_score" not in candidate:
                raise ValueError(f"candidate {route!r} lacks official_score")
            score = float(candidate["official_score"])
            runtime = float(candidate.get("runtime_seconds", 0.0))
        else:
            score = float(candidate)
            runtime = float(runtimes.get(route, 0.0)) if runtimes is not None else 0.0
        if not math.isfinite(score) or not math.isfinite(runtime) or runtime < 0.0:
            raise ValueError(f"candidate {route!r} has an invalid score or runtime")
        ranked.append((score - penalty * runtime, route))
    return min(ranked, key=lambda item: (-item[0], item[1]))[1]


@dataclass(frozen=True)
class RoutePrediction:
    route: str
    suggested_route: str
    confidence: float
    margin: float
    used_default: bool
    ood: bool
    probabilities: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class OOFFitResult:
    identities: tuple[str, ...]
    folds: tuple[int, ...]
    predictions: tuple[RoutePrediction, ...]
    fold_by_group: tuple[tuple[str, int], ...]


@dataclass
class _TreeNode:
    probabilities: tuple[float, ...]
    feature: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None


def _record_identity(record: Mapping[str, Any]) -> str:
    identity = record.get("identity")
    if not isinstance(identity, str) or not identity:
        raise ProvenanceError("every fit record requires a non-empty identity")
    return identity


def validate_fit_provenance(
    records: Sequence[Mapping[str, Any]],
    fit_members: Collection[str],
) -> tuple[str, ...]:
    """Validate the fail-closed ``train:`` identity boundary for fitting."""

    try:
        members = frozenset(fit_members)
    except TypeError as exc:
        raise ProvenanceError("fit_members must be a collection of identities") from exc
    if not members:
        raise ProvenanceError("fit_members cannot be empty")
    for identity in members:
        if not isinstance(identity, str) or not identity.startswith("train:") or len(identity) == 6:
            raise ProvenanceError(f"non-training fit member rejected: {identity!r}")
    if any(not isinstance(record, Mapping) for record in records):
        raise ProvenanceError("every fit record must be a mapping")
    identities = tuple(_record_identity(record) for record in records)
    if len(set(identities)) != len(identities):
        raise ProvenanceError("fit record identities must be unique")
    for identity in identities:
        if not identity.startswith("train:") or len(identity) == 6:
            raise ProvenanceError(f"non-training identity rejected: {identity!r}")
        if identity not in members:
            raise ProvenanceError(f"identity is not a declared fit member: {identity!r}")
    return identities


class ScenarioRouter:
    """A shallow deterministic CART classifier with conservative fallback."""

    def __init__(
        self,
        *,
        default_route: str,
        max_depth: int = 3,
        min_leaf: int = 2,
        min_confidence: float = 0.55,
        min_margin: float = 0.10,
        ood_tolerance: float = 0.20,
        max_thresholds: int = 32,
    ) -> None:
        if not isinstance(default_route, str) or not default_route:
            raise ValueError("default_route must be a non-empty string")
        if type(max_depth) is not int or not 1 <= max_depth <= 8:
            raise ValueError("max_depth must be an integer in [1, 8]")
        if type(min_leaf) is not int or min_leaf < 1:
            raise ValueError("min_leaf must be positive")
        if type(max_thresholds) is not int or max_thresholds < 1:
            raise ValueError("max_thresholds must be positive")
        for name, value in (
            ("min_confidence", min_confidence),
            ("min_margin", min_margin),
            ("ood_tolerance", ood_tolerance),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        self.default_route = default_route
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.ood_tolerance = float(ood_tolerance)
        self.max_thresholds = max_thresholds
        self._classes: tuple[str, ...] = ()
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._root: _TreeNode | None = None

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    def _new_unfit(self) -> ScenarioRouter:
        return type(self)(
            default_route=self.default_route,
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
            ood_tolerance=self.ood_tolerance,
            max_thresholds=self.max_thresholds,
        )

    @staticmethod
    def _record_label(record: Mapping[str, Any]) -> str:
        label = record.get("label", record.get("route"))
        if label is None and isinstance(record.get("candidate_routes"), Mapping):
            label = construct_route_label(
                record["candidate_routes"],
                runtime_penalty=float(record.get("runtime_penalty", 0.0)),
            )
        if not isinstance(label, str) or not label:
            raise ValueError("every fit record requires a route label or candidate_routes")
        return label

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        fit_members: Collection[str],
    ) -> ScenarioRouter:
        """Fit only declared ``train:`` members, independent of input order."""

        rows = list(records)
        if not rows:
            raise ValueError("at least one fit record is required")
        validate_fit_provenance(rows, fit_members)
        ordered = sorted(rows, key=_record_identity)
        labels = tuple(self._record_label(record) for record in ordered)
        classes = tuple(sorted(set(labels)))
        class_index = {label: index for index, label in enumerate(classes)}
        x = np.vstack([feature_vector(record) for record in ordered])
        y = np.asarray([class_index[label] for label in labels], dtype=np.int64)
        self._classes = classes
        self._lower = np.min(x, axis=0)
        self._upper = np.max(x, axis=0)
        self._root = self._build_tree(x, y, depth=0)
        return self

    def _probabilities(self, y: np.ndarray) -> tuple[float, ...]:
        counts = np.bincount(y, minlength=len(self._classes)).astype(np.float64)
        smoothed = counts + 1.0
        return tuple(float(value) for value in smoothed / np.sum(smoothed))

    @staticmethod
    def _gini(y: np.ndarray, class_count: int) -> float:
        counts = np.bincount(y, minlength=class_count).astype(np.float64)
        probabilities = counts / max(1, y.size)
        return float(1.0 - np.dot(probabilities, probabilities))

    def _thresholds(self, values: np.ndarray) -> np.ndarray:
        unique = np.unique(values)
        if unique.size < 2:
            return np.empty(0, dtype=np.float64)
        positions = np.arange(1, unique.size)
        if positions.size > self.max_thresholds:
            positions = np.unique(
                np.linspace(1, unique.size - 1, self.max_thresholds, dtype=np.int64)
            )
        lower = unique[positions - 1]
        upper = unique[positions]
        return lower + (upper - lower) * 0.5

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(probabilities=self._probabilities(y))
        if depth >= self.max_depth or y.size < 2 * self.min_leaf or np.unique(y).size == 1:
            return node
        base = self._gini(y, len(self._classes))
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        best_mask: np.ndarray | None = None
        for feature in range(x.shape[1]):
            for threshold in self._thresholds(x[:, feature]):
                mask = x[:, feature] <= threshold
                left_count = int(np.sum(mask))
                right_count = y.size - left_count
                if left_count < self.min_leaf or right_count < self.min_leaf:
                    continue
                weighted = (
                    left_count * self._gini(y[mask], len(self._classes))
                    + right_count * self._gini(y[~mask], len(self._classes))
                ) / y.size
                gain = base - weighted
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_mask = mask
        if best_feature is None or best_mask is None or best_gain <= 1e-12:
            return node
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(x[best_mask], y[best_mask], depth + 1)
        node.right = self._build_tree(x[~best_mask], y[~best_mask], depth + 1)
        return node

    def _require_fit(self) -> tuple[_TreeNode, np.ndarray, np.ndarray]:
        if self._root is None or self._lower is None or self._upper is None or not self._classes:
            raise RouteModelError("router is not fitted")
        return self._root, self._lower, self._upper

    def predict(self, values: Mapping[str, Any] | Sequence[float]) -> RoutePrediction:
        """Predict one route, falling back on OOD or weak leaf evidence."""

        root, lower, upper = self._require_fit()
        vector = feature_vector(values)
        span = np.maximum(upper - lower, np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1.0) * 1e-9)
        pad = self.ood_tolerance * span
        ood = bool(np.any(vector < lower - pad) or np.any(vector > upper + pad))
        node = root
        while node.feature is not None:
            if node.left is None or node.right is None or node.threshold is None:
                raise RouteModelError("fitted tree is internally inconsistent")
            node = node.left if vector[node.feature] <= node.threshold else node.right
        ranked = sorted(
            zip(self._classes, node.probabilities),
            key=lambda item: (-item[1], item[0]),
        )
        suggested, confidence = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = confidence - runner_up
        weak = confidence < self.min_confidence or margin < self.min_margin
        used_default = ood or weak
        return RoutePrediction(
            route=self.default_route if used_default else suggested,
            suggested_route=suggested,
            confidence=float(confidence),
            margin=float(margin),
            used_default=used_default,
            ood=ood,
            probabilities=tuple((route, float(probability)) for route, probability in ranked),
        )

    def fit_oof(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        fit_members: Collection[str],
        n_splits: int = 5,
    ) -> OOFFitResult:
        """Return group-disjoint OOF predictions, then fit this router on all rows."""

        rows = list(records)
        identities = validate_fit_provenance(rows, fit_members)
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be at least two")
        groups: list[str] = []
        for record in rows:
            group = record.get("group")
            if not isinstance(group, str) or not group:
                raise ValueError("group-aware OOF requires a non-empty group per record")
            groups.append(group)
        unique_groups = set(groups)
        if len(unique_groups) < 2:
            raise ValueError("group-aware OOF requires at least two distinct groups")
        split_count = min(n_splits, len(unique_groups))
        ordered_groups = sorted(
            unique_groups,
            key=lambda group: (hashlib.sha256(group.encode("utf-8")).hexdigest(), group),
        )
        fold_by_group = {group: index % split_count for index, group in enumerate(ordered_groups)}
        predictions: list[RoutePrediction | None] = [None] * len(rows)
        folds = tuple(fold_by_group[group] for group in groups)
        for fold in range(split_count):
            train_indices = [index for index, value in enumerate(folds) if value != fold]
            valid_indices = [index for index, value in enumerate(folds) if value == fold]
            fold_rows = [rows[index] for index in train_indices]
            fold_members = {_record_identity(record) for record in fold_rows}
            fold_router = self._new_unfit().fit(fold_rows, fit_members=fold_members)
            for index in valid_indices:
                predictions[index] = fold_router.predict(rows[index])
        if any(prediction is None for prediction in predictions):
            raise RouteModelError("internal OOF prediction coverage failure")
        self.fit(rows, fit_members=fit_members)
        return OOFFitResult(
            identities=identities,
            folds=folds,
            predictions=tuple(prediction for prediction in predictions if prediction is not None),
            fold_by_group=tuple(sorted(fold_by_group.items())),
        )

    @staticmethod
    def _node_to_dict(node: _TreeNode) -> dict[str, Any]:
        result: dict[str, Any] = {"probabilities": list(node.probabilities)}
        if node.feature is None:
            result.update({"feature": None, "left": None, "right": None, "threshold": None})
        else:
            if node.left is None or node.right is None or node.threshold is None:
                raise RouteModelError("fitted tree is internally inconsistent")
            result.update(
                {
                    "feature": node.feature,
                    "left": ScenarioRouter._node_to_dict(node.left),
                    "right": ScenarioRouter._node_to_dict(node.right),
                    "threshold": node.threshold,
                }
            )
        return result

    def _model_dict(self) -> dict[str, Any]:
        root, lower, upper = self._require_fit()
        return {
            "classes": list(self._classes),
            "config": {
                "default_route": self.default_route,
                "max_depth": self.max_depth,
                "max_thresholds": self.max_thresholds,
                "min_confidence": self.min_confidence,
                "min_leaf": self.min_leaf,
                "min_margin": self.min_margin,
                "ood_tolerance": self.ood_tolerance,
            },
            "feature_lower": lower.tolist(),
            "feature_upper": upper.tolist(),
            "tree": self._node_to_dict(root),
        }

    def to_json(self) -> str:
        """Serialize as deterministic self-digesting ASCII JSON (never pickle)."""

        core = {
            "feature_schema": dict(FEATURE_SCHEMA),
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION,
            "model": self._model_dict(),
        }
        payload = {**core, "sha256": _sha256(core)}
        return _canonical_json(payload).decode("ascii")

    @staticmethod
    def _node_from_dict(value: Any, class_count: int, depth: int = 0) -> _TreeNode:
        if not isinstance(value, dict) or set(value) != {
            "feature", "left", "probabilities", "right", "threshold"
        }:
            raise ModelSerializationError("invalid tree node shape")
        try:
            probabilities = tuple(float(item) for item in value["probabilities"])
        except (TypeError, ValueError) as exc:
            raise ModelSerializationError("invalid tree probabilities") from exc
        if (
            len(probabilities) != class_count
            or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ModelSerializationError("invalid tree probabilities")
        feature = value["feature"]
        if feature is None:
            if value["threshold"] is not None or value["left"] is not None or value["right"] is not None:
                raise ModelSerializationError("leaf contains split fields")
            return _TreeNode(probabilities=probabilities)
        if type(feature) is not int or not 0 <= feature < len(FEATURE_NAMES):
            raise ModelSerializationError("tree feature index is outside the schema")
        try:
            threshold = float(value["threshold"])
        except (TypeError, ValueError) as exc:
            raise ModelSerializationError("invalid tree threshold") from exc
        if not math.isfinite(threshold) or depth >= 8:
            raise ModelSerializationError("invalid tree threshold or depth")
        return _TreeNode(
            probabilities=probabilities,
            feature=feature,
            threshold=threshold,
            left=ScenarioRouter._node_from_dict(value["left"], class_count, depth + 1),
            right=ScenarioRouter._node_from_dict(value["right"], class_count, depth + 1),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> ScenarioRouter:
        """Load an exact-schema model after duplicate-key and digest checks."""

        try:
            if isinstance(text, bytes):
                text = text.decode("ascii")
            if not isinstance(text, str) or not text.isascii():
                raise ValueError("model JSON must be ASCII")
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelSerializationError(f"invalid model JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "feature_schema", "format", "format_version", "model", "sha256"
        }:
            raise ModelSerializationError("invalid model envelope")
        digest = payload.pop("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not hmac.compare_digest(digest, _sha256(payload))
        ):
            raise ModelSerializationError("model SHA-256 mismatch")
        if payload["format"] != MODEL_FORMAT or payload["format_version"] != MODEL_FORMAT_VERSION:
            raise ModelSerializationError("unsupported model format")
        schema = payload["feature_schema"]
        if schema != FEATURE_SCHEMA:
            raise ModelSerializationError("feature schema mismatch")
        if schema.get("sha256") != _sha256({"names": schema.get("names"), "version": schema.get("version")}):
            raise ModelSerializationError("feature schema SHA-256 mismatch")
        model = payload["model"]
        if not isinstance(model, dict) or set(model) != {
            "classes", "config", "feature_lower", "feature_upper", "tree"
        }:
            raise ModelSerializationError("invalid model body")
        classes = model["classes"]
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(item, str) or not item for item in classes)
            or classes != sorted(set(classes))
        ):
            raise ModelSerializationError("model classes must be sorted and unique")
        config = model["config"]
        if not isinstance(config, dict) or set(config) != {
            "default_route", "max_depth", "max_thresholds", "min_confidence",
            "min_leaf", "min_margin", "ood_tolerance"
        }:
            raise ModelSerializationError("invalid model config")
        try:
            router = cls(**config)
            lower = np.asarray(model["feature_lower"], dtype=np.float64)
            upper = np.asarray(model["feature_upper"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ModelSerializationError(f"invalid model values: {exc}") from exc
        if (
            lower.shape != (len(FEATURE_NAMES),)
            or upper.shape != (len(FEATURE_NAMES),)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ModelSerializationError("invalid feature bounds")
        root = cls._node_from_dict(model["tree"], len(classes))
        router._classes = tuple(classes)
        router._lower = lower
        router._upper = upper
        router._root = root
        return router

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save deterministic ASCII JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="ascii",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(self.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ScenarioRouter:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ModelSerializationError(f"cannot read model: {exc}") from exc
        return cls.from_json(raw)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_SHA256",
    "FEATURE_SCHEMA_VERSION",
    "FeatureValidationError",
    "ModelSerializationError",
    "OOFFitResult",
    "ProvenanceError",
    "RouteModelError",
    "RoutePrediction",
    "ScenarioRouter",
    "construct_route_label",
    "extract_features",
    "feature_vector",
    "validate_fit_provenance",
]
