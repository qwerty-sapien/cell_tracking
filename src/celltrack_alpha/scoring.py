"""Fold-safe candidate graph scoring and pruning utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .localization import sha256_file
from .types import CandidateEdge, CandidateNode, RouteConfig


class UnsafeScoringCalibrationError(ValueError):
    """Raised when candidate scoring calibration lacks fold-safe provenance."""


@dataclass(frozen=True)
class ScoringCalibration:
    """Validated candidate scoring calibration payload."""

    path: Path
    payload: dict[str, Any]
    fold_id: str
    calibration_train: frozenset[str]
    validation_datasets: frozenset[str]
    excluded_duplicate_public_ids: frozenset[str]
    split_manifest_sha256: str


@dataclass(frozen=True)
class NodePruningStats:
    """Summary of an applied node-pruning pass."""

    route: str
    dataset: str
    before_nodes: int
    after_nodes: int
    before_by_t: tuple[int, ...]
    after_by_t: tuple[int, ...]
    policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "dataset": self.dataset,
            "before_nodes": self.before_nodes,
            "after_nodes": self.after_nodes,
            "before_by_t": list(self.before_by_t),
            "after_by_t": list(self.after_by_t),
            "policy": self.policy,
        }


@dataclass(frozen=True)
class EdgeScoringStats:
    """Summary of an applied edge-scoring pass."""

    route: str
    dataset: str
    before_edges: int
    after_edges: int
    dropped_edges: int
    mean_base_cost: float
    mean_adjusted_cost: float
    mean_calibrated_probability: float
    policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "dataset": self.dataset,
            "before_edges": self.before_edges,
            "after_edges": self.after_edges,
            "dropped_edges": self.dropped_edges,
            "mean_base_cost": self.mean_base_cost,
            "mean_adjusted_cost": self.mean_adjusted_cost,
            "mean_calibrated_probability": self.mean_calibrated_probability,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class CoordinateCorrectionStats:
    """Summary of an applied candidate-coordinate correction pass."""

    route: str
    dataset: str
    before_nodes: int
    corrected_nodes: int
    mean_abs_offset_zyx: tuple[float, float, float]
    policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "dataset": self.dataset,
            "before_nodes": self.before_nodes,
            "corrected_nodes": self.corrected_nodes,
            "mean_abs_offset_zyx": list(self.mean_abs_offset_zyx),
            "policy": self.policy,
        }


def _as_dataset_set(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise UnsafeScoringCalibrationError(f"{name} must be a list of dataset strings")
    return frozenset(value)


def _load_split_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise UnsafeScoringCalibrationError(f"split manifest is required for scoring calibration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnsafeScoringCalibrationError("split manifest root must be an object")
    return payload, sha256_file(path)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise UnsafeScoringCalibrationError(f"{name} must be a finite number")
    return float(value)


def _validate_weights(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise UnsafeScoringCalibrationError(f"{name} must be an object mapping feature names to weights")
    for key, weight in value.items():
        _finite_number(weight, f"{name}.{key}")


def _validate_policy_row(row: Any, name: str) -> None:
    if not isinstance(row, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    _validate_weights(row.get("weights"), f"{name}.weights")
    if "bias" in row:
        _finite_number(row["bias"], f"{name}.bias")
    if "min_score" in row:
        _finite_number(row["min_score"], f"{name}.min_score")
    for key in ("keep_top_k_per_t", "min_keep_per_t"):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UnsafeScoringCalibrationError(f"{name}.{key} must be a non-negative integer or null")
    if "replace_node_score" in row and not isinstance(row["replace_node_score"], bool):
        raise UnsafeScoringCalibrationError(f"{name}.replace_node_score must be boolean when present")
    if "replace_edge_score" in row and not isinstance(row["replace_edge_score"], bool):
        raise UnsafeScoringCalibrationError(f"{name}.replace_edge_score must be boolean when present")
    if "drop_below_min_score" in row and not isinstance(row["drop_below_min_score"], bool):
        raise UnsafeScoringCalibrationError(f"{name}.drop_below_min_score must be boolean when present")
    for key in ("base_cost_weight", "probability_cost_weight", "cost_offset", "cost_min", "cost_max"):
        if key in row and row[key] is not None:
            _finite_number(row[key], f"{name}.{key}")


def _validate_policy_section(section: Any, name: str) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    if "default" in section:
        _validate_policy_row(section["default"], f"{name}.default")
    for subsection in ("by_prefix", "by_route", "by_route_prefix"):
        value = section.get(subsection, {})
        if not isinstance(value, dict):
            raise UnsafeScoringCalibrationError(f"{name}.{subsection} must be an object")
        for key, row in value.items():
            if not isinstance(key, str):
                raise UnsafeScoringCalibrationError(f"{name}.{subsection} keys must be strings")
            _validate_policy_row(row, f"{name}.{subsection}.{key}")


def _offset_triplet(value: Any, name: str, *, allow_float: bool = True) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise UnsafeScoringCalibrationError(f"{name} must contain exactly three numbers")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise UnsafeScoringCalibrationError(f"{name} must contain exactly three finite numbers")
        if not allow_float and int(item) != item:
            raise UnsafeScoringCalibrationError(f"{name} must contain integer values")
        out.append(float(item))
    return tuple(out)  # type: ignore[return-value]


def _validate_coordinate_policy_row(row: Any, name: str) -> None:
    if not isinstance(row, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    if "offset_zyx" in row:
        _offset_triplet(row["offset_zyx"], f"{name}.offset_zyx")
    if "max_abs_offset_zyx" in row:
        values = _offset_triplet(row["max_abs_offset_zyx"], f"{name}.max_abs_offset_zyx")
        if any(value < 0 for value in values):
            raise UnsafeScoringCalibrationError(f"{name}.max_abs_offset_zyx must be non-negative")
    if "source_offsets_zyx" in row:
        source_offsets = row["source_offsets_zyx"]
        if not isinstance(source_offsets, dict):
            raise UnsafeScoringCalibrationError(f"{name}.source_offsets_zyx must be an object")
        for source, offset in source_offsets.items():
            if not isinstance(source, str):
                raise UnsafeScoringCalibrationError(f"{name}.source_offsets_zyx keys must be strings")
            _offset_triplet(offset, f"{name}.source_offsets_zyx.{source}")


def _validate_coordinate_policy_section(section: Any, name: str) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    if "default" in section:
        _validate_coordinate_policy_row(section["default"], f"{name}.default")
    for subsection in ("by_prefix", "by_route", "by_route_prefix"):
        value = section.get(subsection, {})
        if not isinstance(value, dict):
            raise UnsafeScoringCalibrationError(f"{name}.{subsection} must be an object")
        for key, row in value.items():
            if not isinstance(key, str):
                raise UnsafeScoringCalibrationError(f"{name}.{subsection} keys must be strings")
            _validate_coordinate_policy_row(row, f"{name}.{subsection}.{key}")


def _validate_template_policy_row(row: Any, name: str) -> None:
    if not isinstance(row, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    templates = row.get("templates", [])
    if not isinstance(templates, list):
        raise UnsafeScoringCalibrationError(f"{name}.templates must be a list")
    for index, template in enumerate(templates):
        template_name = f"{name}.templates[{index}]"
        if not isinstance(template, dict):
            raise UnsafeScoringCalibrationError(f"{template_name} must be an object")
        if not isinstance(template.get("name"), str) or not template.get("name"):
            raise UnsafeScoringCalibrationError(f"{template_name}.name must be a non-empty string")
        kernel = template.get("kernel")
        try:
            arr = np.asarray(kernel, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise UnsafeScoringCalibrationError(f"{template_name}.kernel must be numeric") from exc
        if arr.ndim != 3 or arr.size == 0 or arr.size > 50000 or not np.all(np.isfinite(arr)):
            raise UnsafeScoringCalibrationError(
                f"{template_name}.kernel must be a finite non-empty 3D array with at most 50000 values"
            )
        if "score_quantiles" in template:
            quantiles = template["score_quantiles"]
            if not isinstance(quantiles, list) or not quantiles:
                raise UnsafeScoringCalibrationError(f"{template_name}.score_quantiles must be a non-empty list")
            for q_index, quantile in enumerate(quantiles):
                value = _finite_number(quantile, f"{template_name}.score_quantiles[{q_index}]")
                if value < 0.0 or value > 1.0:
                    raise UnsafeScoringCalibrationError(f"{template_name}.score_quantiles[{q_index}] must be in [0, 1]")
        if "max_points" in template:
            value = template["max_points"]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise UnsafeScoringCalibrationError(f"{template_name}.max_points must be a positive integer")
        if "min_distance_um" in template:
            value = _finite_number(template["min_distance_um"], f"{template_name}.min_distance_um")
            if value <= 0.0:
                raise UnsafeScoringCalibrationError(f"{template_name}.min_distance_um must be positive")
        if "source" in template and not isinstance(template["source"], str):
            raise UnsafeScoringCalibrationError(f"{template_name}.source must be a string when present")


def _validate_proposal_policy_section(section: Any, name: str) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise UnsafeScoringCalibrationError(f"{name} must be an object")
    if "default" in section:
        _validate_template_policy_row(section["default"], f"{name}.default")
    for subsection in ("by_prefix", "by_route", "by_route_prefix"):
        value = section.get(subsection, {})
        if not isinstance(value, dict):
            raise UnsafeScoringCalibrationError(f"{name}.{subsection} must be an object")
        for key, row in value.items():
            if not isinstance(key, str):
                raise UnsafeScoringCalibrationError(f"{name}.{subsection} keys must be strings")
            _validate_template_policy_row(row, f"{name}.{subsection}.{key}")


def validate_scoring_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    split_manifest_path: Path,
    expected_fold_id: str | None = None,
) -> ScoringCalibration:
    """Validate a scoring calibration before it can affect held-out scoring."""

    if payload.get("kind") not in (None, "candidate_graph_scoring"):
        raise UnsafeScoringCalibrationError("scoring calibration kind must be 'candidate_graph_scoring'")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise UnsafeScoringCalibrationError("scoring calibration must include a provenance object")
    if provenance.get("status") != "fold_safe":
        raise UnsafeScoringCalibrationError("scoring calibration provenance.status must be 'fold_safe'")
    fold_id = str(provenance.get("fold_id", ""))
    if not fold_id:
        raise UnsafeScoringCalibrationError("scoring calibration provenance.fold_id is required")
    if expected_fold_id is not None and fold_id != str(expected_fold_id):
        raise UnsafeScoringCalibrationError(
            f"scoring calibration fold_id={fold_id!r} does not match expected {expected_fold_id!r}"
        )

    split_manifest, split_hash = _load_split_manifest(split_manifest_path)
    fold = split_manifest.get("folds", {}).get(fold_id)
    if not isinstance(fold, dict):
        raise UnsafeScoringCalibrationError(f"fold {fold_id!r} is not present in split manifest")

    calibration_train = _as_dataset_set(provenance.get("calibration_train"), "provenance.calibration_train")
    validation_datasets = _as_dataset_set(provenance.get("validation_datasets"), "provenance.validation_datasets")
    expected_train = frozenset(str(v) for v in fold.get("calibration_train", []))
    expected_validation = frozenset(str(v) for v in fold.get("validation", []))
    if calibration_train - expected_train:
        raise UnsafeScoringCalibrationError("calibration_train contains datasets outside the locked fold training set")
    if validation_datasets - expected_validation:
        raise UnsafeScoringCalibrationError("validation_datasets contains datasets outside the locked fold validation set")
    if calibration_train & validation_datasets:
        raise UnsafeScoringCalibrationError("calibration_train overlaps validation_datasets")

    excluded = frozenset(
        str(v)
        for v in split_manifest.get("excluded_from_general_score", {}).get(
            "duplicate_public_ids", []
        )
    )
    source_datasets = _as_dataset_set(provenance.get("source_datasets"), "provenance.source_datasets")
    if source_datasets - calibration_train:
        raise UnsafeScoringCalibrationError("source_datasets must be a subset of calibration_train")
    if source_datasets & excluded:
        raise UnsafeScoringCalibrationError("source_datasets include duplicate public IDs excluded from general score")

    recorded_split_hash = provenance.get("split_manifest_sha256")
    if recorded_split_hash is not None and str(recorded_split_hash) != split_hash:
        raise UnsafeScoringCalibrationError("calibration split_manifest_sha256 does not match current split manifest")

    _validate_policy_section(payload.get("node_policy", {}), "node_policy")
    _validate_policy_section(payload.get("edge_policy", {}), "edge_policy")
    _validate_coordinate_policy_section(payload.get("coordinate_policy", {}), "coordinate_policy")
    _validate_proposal_policy_section(payload.get("proposal_policy", {}), "proposal_policy")

    return ScoringCalibration(
        path=path,
        payload=payload,
        fold_id=fold_id,
        calibration_train=calibration_train,
        validation_datasets=validation_datasets,
        excluded_duplicate_public_ids=excluded,
        split_manifest_sha256=split_hash,
    )


def load_scoring_calibration(
    path: Path | None,
    *,
    split_manifest_path: Path = Path("reports/general_model_split_manifest.json"),
    expected_fold_id: str | None = None,
) -> ScoringCalibration | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UnsafeScoringCalibrationError("scoring calibration root must be an object")
    return validate_scoring_payload(
        payload,
        path=path,
        split_manifest_path=split_manifest_path,
        expected_fold_id=expected_fold_id,
    )


def scoring_for_dataset(dataset: str, calibration: ScoringCalibration | None) -> dict[str, Any] | None:
    if calibration is None:
        return None
    if dataset in calibration.calibration_train:
        raise UnsafeScoringCalibrationError(
            f"refusing to apply held-out scoring calibration to calibration dataset {dataset!r}"
        )
    if dataset not in calibration.validation_datasets:
        raise UnsafeScoringCalibrationError(
            f"dataset {dataset!r} is not in the scoring validation set for fold {calibration.fold_id}"
        )
    return calibration.payload


def _source_families(sources: tuple[str, ...]) -> tuple[str, ...]:
    families: list[str] = []
    for source in sources:
        family = str(source).split("@", 1)[0]
        if family not in families:
            families.append(family)
    return tuple(families)


def node_feature_dict(node: CandidateNode) -> dict[str, float]:
    """Return deterministic engineered features for a candidate node."""

    families = _source_families(tuple(str(s) for s in node.sources))
    family_set = set(families)
    features: dict[str, float] = {}
    for key, value in node.features.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[str(key)] = float(value)
    source_count = float(len(node.sources))
    family_count = float(len(families))
    intensity = float(node.intensity) if math.isfinite(float(node.intensity)) else 0.0
    density = float(node.local_density) if math.isfinite(float(node.local_density)) else 0.0
    features.update(
        {
            "score": float(node.score),
            "uncertainty": float(node.uncertainty),
            "local_density": density,
            "log1p_local_density": math.log1p(max(0.0, density)),
            "intensity": intensity,
            "log1p_intensity": math.log1p(max(0.0, intensity)),
            "source_count": source_count,
            "source_family_count": family_count,
            "has_tile_raw": float("tile_raw" in families),
            "has_tile_contrast": float("tile_contrast" in families),
            "has_dog_bright": float("dog_bright" in families),
            "has_dog_dark": float("dog_dark" in families),
            "has_dog_abs": float("dog_abs" in families),
            "has_local_contrast": float("local_contrast" in families),
            "has_tile_dog_bright": float("tile_dog_bright" in families),
            "has_tile_dog_dark": float("tile_dog_dark" in families),
            "has_tile_dog_abs": float("tile_dog_abs" in families),
            "has_cover_raw": float("cover_raw" in families),
            "has_cover_smooth": float("cover_smooth" in families),
            "has_cover_contrast": float("cover_contrast" in families),
            "has_cover_dog_bright": float("cover_dog_bright" in families),
            "has_cover_dog_dark": float("cover_dog_dark" in families),
            "has_cover_dog_abs": float("cover_dog_abs" in families),
            "has_subtile_contrast": float("subtile_contrast" in families),
            "has_subtile_dog_bright": float("subtile_dog_bright" in families),
            "has_subtile_dog_dark": float("subtile_dog_dark" in families),
            "has_subtile_dog_abs": float("subtile_dog_abs" in families),
            "has_template_score": float(any(family.startswith("template_score") for family in families)),
            "has_refine_local_score": float(any(family.startswith("refine_local_score") for family in families)),
            "has_zshift_dog_dark_m5": float(any(family.startswith("zshift_dog_dark_m5") for family in families)),
            "has_zshift_dog_bright_m3": float(any(family.startswith("zshift_dog_bright_m3") for family in families)),
            "has_hmax": float("hmax" in families),
            "has_components": float("components" in families),
        }
    )
    tile_families = {"tile_raw", "tile_contrast", "tile_dog_bright", "tile_dog_dark", "tile_dog_abs"}
    cover_families = {
        "cover_raw",
        "cover_smooth",
        "cover_contrast",
        "cover_dog_bright",
        "cover_dog_dark",
        "cover_dog_abs",
    }
    dog_families = {
        "dog_bright",
        "dog_dark",
        "dog_abs",
        "tile_dog_bright",
        "tile_dog_dark",
        "tile_dog_abs",
        "cover_dog_bright",
        "cover_dog_dark",
        "cover_dog_abs",
        "subtile_dog_bright",
        "subtile_dog_dark",
        "subtile_dog_abs",
    }
    subtile_families = {"subtile_contrast", "subtile_dog_bright", "subtile_dog_dark", "subtile_dog_abs"}
    zshift_families = {family for family in family_set if family.startswith("zshift_")}
    features["tile_only"] = float(bool(family_set) and family_set <= tile_families)
    features["dog_only"] = float(bool(family_set) and family_set <= dog_families)
    features["has_tile"] = float(bool(family_set & tile_families))
    features["has_cover"] = float(bool(family_set & cover_families))
    features["has_zshift"] = float(bool(zshift_families))
    features["zshift_only"] = float(bool(family_set) and family_set <= zshift_families)
    features["has_dog"] = float(bool(family_set & dog_families) or bool(zshift_families))
    features["has_subtile"] = float(bool(family_set & subtile_families))
    features["cover_only"] = float(bool(family_set) and family_set <= cover_families)
    features["rescue_source_count"] = float(
        sum(
            1
            for family in family_set
            if family in cover_families
            or family in subtile_families
            or family.startswith("template_score")
            or family.startswith("zshift_")
        )
    )
    return features


def edge_feature_dict(
    edge: CandidateEdge,
    source: CandidateNode,
    target: CandidateNode,
    *,
    max_link_um: float,
) -> dict[str, float]:
    """Return deterministic engineered features for a candidate edge."""

    source_features = node_feature_dict(source)
    target_features = node_feature_dict(target)
    denominator = abs(float(source.intensity)) + abs(float(target.intensity)) + 1e-6
    intensity_delta = min(1.0, abs(float(source.intensity) - float(target.intensity)) / denominator)
    features: dict[str, float] = {
        "edge_cost": float(edge.cost),
        "edge_score": float(edge.score),
        "dt": float(edge.dt),
        "distance_um": float(edge.distance_um),
        "distance_norm": float(edge.distance_um) / max(float(max_link_um), 1e-6),
        "source_score": float(source.score),
        "target_score": float(target.score),
        "source_density": float(source.local_density),
        "target_density": float(target.local_density),
        "mean_density": (float(source.local_density) + float(target.local_density)) / 2.0,
        "intensity_delta": float(intensity_delta),
        "source_log1p_intensity": source_features["log1p_intensity"],
        "target_log1p_intensity": target_features["log1p_intensity"],
        "source_has_tile": source_features["has_tile"],
        "target_has_tile": target_features["has_tile"],
        "source_has_cover": source_features["has_cover"],
        "target_has_cover": target_features["has_cover"],
        "source_has_dog": source_features["has_dog"],
        "target_has_dog": target_features["has_dog"],
        "source_has_subtile": source_features["has_subtile"],
        "target_has_subtile": target_features["has_subtile"],
        "source_has_zshift": source_features["has_zshift"],
        "target_has_zshift": target_features["has_zshift"],
        "source_cover_only": source_features["cover_only"],
        "target_cover_only": target_features["cover_only"],
        "source_rescue_source_count": source_features["rescue_source_count"],
        "target_rescue_source_count": target_features["rescue_source_count"],
    }
    for key, value in edge.features.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[str(key)] = float(value)
    return features


def linear_score(features: dict[str, float], policy: dict[str, Any]) -> float:
    score = float(policy.get("bias", 0.0))
    weights = policy.get("weights", {})
    if not isinstance(weights, dict):
        return score
    for key, weight in weights.items():
        value = features.get(str(key), 0.0)
        if math.isfinite(float(value)):
            score += float(weight) * float(value)
    return float(score)


def _merge_policy(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if key == "weights" and isinstance(value, dict):
            weights = dict(merged.get("weights", {}))
            weights.update(value)
            merged["weights"] = weights
        else:
            merged[key] = value
    return merged


def _policy_for_dataset(
    calibration_payload: dict[str, Any] | None,
    *,
    section_name: str,
    dataset: str,
    route_label: str,
) -> dict[str, Any]:
    """Resolve default/prefix/route policy overrides for one dataset route."""

    if calibration_payload is None:
        return {}
    section = calibration_payload.get(section_name, {})
    if not isinstance(section, dict):
        return {}
    prefix = dataset.split("_", 1)[0]
    policy: dict[str, Any] = {}
    policy = _merge_policy(policy, section.get("default"))
    by_prefix = section.get("by_prefix", {})
    if isinstance(by_prefix, dict):
        policy = _merge_policy(policy, by_prefix.get(prefix))
    by_route = section.get("by_route", {})
    if isinstance(by_route, dict):
        policy = _merge_policy(policy, by_route.get(route_label))
    by_route_prefix = section.get("by_route_prefix", {})
    if isinstance(by_route_prefix, dict):
        policy = _merge_policy(policy, by_route_prefix.get(f"{route_label}:{prefix}"))
    return policy


def node_policy_for_dataset(
    calibration_payload: dict[str, Any] | None,
    *,
    dataset: str,
    route_label: str,
) -> dict[str, Any]:
    return _policy_for_dataset(
        calibration_payload,
        section_name="node_policy",
        dataset=dataset,
        route_label=route_label,
    )


def edge_policy_for_dataset(
    calibration_payload: dict[str, Any] | None,
    *,
    dataset: str,
    route_label: str,
) -> dict[str, Any]:
    return _policy_for_dataset(
        calibration_payload,
        section_name="edge_policy",
        dataset=dataset,
        route_label=route_label,
    )


def coordinate_policy_for_dataset(
    calibration_payload: dict[str, Any] | None,
    *,
    dataset: str,
    route_label: str,
) -> dict[str, Any]:
    return _policy_for_dataset(
        calibration_payload,
        section_name="coordinate_policy",
        dataset=dataset,
        route_label=route_label,
    )


def proposal_policy_for_dataset(
    calibration_payload: dict[str, Any] | None,
    *,
    dataset: str,
    route_label: str,
) -> dict[str, Any]:
    return _policy_for_dataset(
        calibration_payload,
        section_name="proposal_policy",
        dataset=dataset,
        route_label=route_label,
    )


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value)))))


def _clipped_offset(policy: dict[str, Any], raw_offset: np.ndarray) -> np.ndarray:
    max_abs = policy.get("max_abs_offset_zyx")
    if max_abs is None:
        return raw_offset
    limits = np.asarray(_offset_triplet(max_abs, "max_abs_offset_zyx"), dtype=np.float32)
    return np.clip(raw_offset, -limits, limits)


def _offset_for_node(node: CandidateNode, policy: dict[str, Any]) -> np.ndarray | None:
    offsets: list[np.ndarray] = []
    source_offsets = policy.get("source_offsets_zyx", {})
    if isinstance(source_offsets, dict):
        for family in _source_families(tuple(str(source) for source in node.sources)):
            if family in source_offsets:
                offsets.append(np.asarray(_offset_triplet(source_offsets[family], f"source_offsets_zyx.{family}"), dtype=np.float32))
    if offsets:
        raw = np.median(np.vstack(offsets), axis=0)
    elif "offset_zyx" in policy:
        raw = np.asarray(_offset_triplet(policy["offset_zyx"], "offset_zyx"), dtype=np.float32)
    else:
        return None
    return np.rint(_clipped_offset(policy, raw)).astype(np.int64)


def correct_nodes_by_t(
    nodes_by_t: list[list[CandidateNode]],
    *,
    dataset: str,
    route_label: str,
    calibration_payload: dict[str, Any] | None,
    spatial_shape_zyx: tuple[int, int, int],
) -> tuple[list[list[CandidateNode]], CoordinateCorrectionStats | None]:
    """Apply optional deterministic per-candidate coordinate corrections."""

    policy = coordinate_policy_for_dataset(calibration_payload, dataset=dataset, route_label=route_label)
    if not policy:
        return nodes_by_t, None
    shape = np.asarray(spatial_shape_zyx, dtype=np.int64)
    corrected: list[list[CandidateNode]] = []
    offsets: list[np.ndarray] = []
    corrected_count = 0
    before_nodes = sum(len(nodes) for nodes in nodes_by_t)
    for nodes in nodes_by_t:
        frame: list[CandidateNode] = []
        for node in nodes:
            offset = _offset_for_node(node, policy)
            if offset is None or not np.any(offset):
                frame.append(node)
                continue
            current = np.asarray(node.coord_zyx, dtype=np.int64)
            new_coord = np.clip(current + offset, 0, shape - 1)
            if not np.array_equal(new_coord, current):
                features = dict(node.features)
                features.update(
                    {
                        "coordinate_correction_dz": float(new_coord[0] - current[0]),
                        "coordinate_correction_dy": float(new_coord[1] - current[1]),
                        "coordinate_correction_dx": float(new_coord[2] - current[2]),
                    }
                )
                node = replace(
                    node,
                    z=int(new_coord[0]),
                    y=int(new_coord[1]),
                    x=int(new_coord[2]),
                    features=features,
                )
                corrected_count += 1
                offsets.append(np.abs(new_coord - current).astype(np.float32))
            frame.append(node)
        corrected.append(frame)
    mean_abs = tuple(float(v) for v in (np.mean(np.vstack(offsets), axis=0) if offsets else np.zeros(3, dtype=np.float32)))
    stats = CoordinateCorrectionStats(
        route=route_label,
        dataset=dataset,
        before_nodes=before_nodes,
        corrected_nodes=corrected_count,
        mean_abs_offset_zyx=mean_abs,
        policy=policy,
    )
    return corrected, stats


def prune_nodes_by_t(
    nodes_by_t: list[list[CandidateNode]],
    *,
    dataset: str,
    route_label: str,
    calibration_payload: dict[str, Any] | None,
) -> tuple[list[list[CandidateNode]], NodePruningStats | None]:
    """Apply an optional deterministic node-pruning policy."""

    policy = node_policy_for_dataset(calibration_payload, dataset=dataset, route_label=route_label)
    if not policy:
        return nodes_by_t, None

    min_score = float(policy.get("min_score", -math.inf))
    keep_top_k = policy.get("keep_top_k_per_t")
    min_keep = int(policy.get("min_keep_per_t", 0) or 0)
    replace_node_score = bool(policy.get("replace_node_score", False))
    before_by_t = tuple(len(nodes) for nodes in nodes_by_t)
    pruned: list[list[CandidateNode]] = []
    for nodes in nodes_by_t:
        scored: list[tuple[float, CandidateNode]] = []
        for node in nodes:
            features = node_feature_dict(node)
            calibrated = linear_score(features, policy)
            node.features["calibrated_node_score"] = calibrated
            node.features["calibrated_node_probability"] = _sigmoid(calibrated)
            if replace_node_score:
                node.score = float(np.clip(_sigmoid(calibrated), 0.0, 1.0))
                node.uncertainty = float(1.0 - node.score)
            scored.append((calibrated, node))
        ranked = sorted(scored, key=lambda row: (-row[0], row[1].node_id))
        kept = [(score, node) for score, node in ranked if score >= min_score]
        if keep_top_k is not None:
            kept = kept[: int(keep_top_k)]
        if min_keep and len(kept) < min(min_keep, len(ranked)):
            selected_ids = {node.node_id for _, node in kept}
            for score, node in ranked:
                if node.node_id in selected_ids:
                    continue
                kept.append((score, node))
                selected_ids.add(node.node_id)
                if len(kept) >= min_keep:
                    break
        keep_ids = {node.node_id for _, node in kept}
        pruned.append([node for node in nodes if node.node_id in keep_ids])

    after_by_t = tuple(len(nodes) for nodes in pruned)
    stats = NodePruningStats(
        route=route_label,
        dataset=dataset,
        before_nodes=sum(before_by_t),
        after_nodes=sum(after_by_t),
        before_by_t=before_by_t,
        after_by_t=after_by_t,
        policy=policy,
    )
    return pruned, stats

def score_edge_candidates(
    edge_candidates: list[CandidateEdge],
    nodes_by_t: list[list[CandidateNode]],
    *,
    dataset: str,
    route_label: str,
    route_config: RouteConfig,
    calibration_payload: dict[str, Any] | None,
) -> tuple[list[CandidateEdge], EdgeScoringStats | None]:
    """Apply an optional deterministic edge-cost calibration before solving."""

    policy = edge_policy_for_dataset(calibration_payload, dataset=dataset, route_label=route_label)
    if not policy:
        return edge_candidates, None
    nodes_by_id = {node.node_id: node for nodes in nodes_by_t for node in nodes}
    min_score = float(policy.get("min_score", -math.inf))
    drop_below = bool(policy.get("drop_below_min_score", False))
    base_cost_weight = float(policy.get("base_cost_weight", 1.0))
    probability_cost_weight = float(policy.get("probability_cost_weight", 0.0))
    cost_offset = float(policy.get("cost_offset", 0.0))
    cost_min = policy.get("cost_min")
    cost_max = policy.get("cost_max")
    replace_edge_score = bool(policy.get("replace_edge_score", False))

    adjusted: list[CandidateEdge] = []
    base_costs: list[float] = []
    adjusted_costs: list[float] = []
    probabilities: list[float] = []
    dropped = 0
    for edge in edge_candidates:
        source = nodes_by_id.get(int(edge.source_id))
        target = nodes_by_id.get(int(edge.target_id))
        if source is None or target is None:
            dropped += 1
            continue
        features = edge_feature_dict(edge, source, target, max_link_um=route_config.max_link_um)
        calibrated = linear_score(features, policy)
        probability = _sigmoid(calibrated)
        if drop_below and calibrated < min_score:
            dropped += 1
            continue
        cost = (base_cost_weight * float(edge.cost)) + (probability_cost_weight * (1.0 - probability)) + cost_offset
        if cost_min is not None:
            cost = max(float(cost_min), cost)
        if cost_max is not None:
            cost = min(float(cost_max), cost)
        features.update(
            {
                "calibrated_edge_score": float(calibrated),
                "calibrated_edge_probability": float(probability),
                "base_edge_cost": float(edge.cost),
                "adjusted_edge_cost": float(cost),
            }
        )
        score = float(probability) if replace_edge_score else float(edge.score)
        new_edge = replace(edge, cost=float(cost), score=score, features=features)
        adjusted.append(new_edge)
        base_costs.append(float(edge.cost))
        adjusted_costs.append(float(cost))
        probabilities.append(float(probability))

    stats = EdgeScoringStats(
        route=route_label,
        dataset=dataset,
        before_edges=len(edge_candidates),
        after_edges=len(adjusted),
        dropped_edges=dropped,
        mean_base_cost=float(np.mean(base_costs)) if base_costs else float("nan"),
        mean_adjusted_cost=float(np.mean(adjusted_costs)) if adjusted_costs else float("nan"),
        mean_calibrated_probability=float(np.mean(probabilities)) if probabilities else float("nan"),
        policy=policy,
    )
    return adjusted, stats

