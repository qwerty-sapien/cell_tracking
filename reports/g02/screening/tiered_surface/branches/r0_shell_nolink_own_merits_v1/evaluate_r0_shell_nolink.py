#!/usr/bin/env python3
"""Evaluate the preregistered narrow R0 free/free shell no-link screen.

Selection is truth-blind.  The selected-actions artifact is created before this
program opens any GT arrays.  The graph branch is permanently closed by the
preregistered marginal-value gate; this program only measures precision at B=4.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
import zarr
from zarr.storage import ZipStore


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE
while WORKTREE_ROOT.parent != WORKTREE_ROOT and not (
    WORKTREE_ROOT / "src/celltrack_alpha"
).is_dir():
    WORKTREE_ROOT = WORKTREE_ROOT.parent
INPUT_ROOT = Path("/Users/agent/Desktop/cell_tracking")
for location in (
    WORKTREE_ROOT / "src",
    WORKTREE_ROOT / "scripts",
    INPUT_ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from audit_selected_graph_appearance import extract_patch, unit_vector  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import _cache_identity, _load_base_scorer  # noqa: E402
from prepare_extended_radius_shell import (  # noqa: E402
    SCALE_ZYX,
    _baseline,
    label_pair,
    shell_candidate_indices,
)


SCHEMA = "g02.r0-shell-freefree-oof-appearance-nolink.result.v1"
MECHANISM = "R0-SHELL-FREEFREE-OOF-APPEARANCE-NOLINK-B4-V1"
FEATURE_NAMES = (
    "pair_ncc",
    "source_no_link_probability",
    "distance_um",
)
BUDGET = 4
RIDGE_LAMBDA = 0.01
STD_FLOOR = 1.0e-6
APNCC_COUNTS = {"edge_tp": 10327, "edge_fp": 958, "edge_fn": 687}
GRAPH_FLOOR = 0.005
SCIENTIFIC_CONTRACT_SHA256 = (
    "3822ceea710d22a4326eb61775e85243e1a54faeb9c656c157b8ad8893416a2d"
)
EXPECTED_INPUT_SHA256 = {
    "prepare": "cb882835d9b33bcad016684f988ea81e2fff3713432cbde0dadc9b6879cd6386",
    "appearance": "8183f347bd89f3405dfdb7e14e564f2b285ef624ab6bd6f5e12d7460b43f7aa8",
}


class R0Error(RuntimeError):
    """The frozen R0 selection/evaluation contract was violated."""


@dataclass(frozen=True)
class RidgeModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficient: tuple[float, ...]
    intercept: float

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        coefficient = np.asarray(self.coefficient, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != mean.shape:
            raise R0Error("ridge score feature shape drifted")
        return ((values - mean) / scale) @ coefficient + self.intercept


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R0Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise R0Error(f"JSON must contain an object: {path}")
    return value


def outer_fold(identity: str) -> int:
    return int(hashlib.sha256(identity.encode("ascii")).hexdigest()[:8], 16) % 5


def fit_balanced_ridge(
    features: np.ndarray,
    positive: np.ndarray,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    std_floor: float = STD_FLOOR,
) -> RidgeModel:
    """Fit the exact class-balanced weighted ridge contract."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(positive, dtype=np.float64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isin(y, (0.0, 1.0)).all()
        or not math.isfinite(ridge_lambda)
        or ridge_lambda < 0.0
        or not math.isfinite(std_floor)
        or std_floor <= 0.0
    ):
        raise R0Error("balanced-ridge inputs are malformed")
    n_positive = int(np.sum(y == 1.0))
    n_background = int(np.sum(y == 0.0))
    if n_positive < 1 or n_background < 1:
        raise R0Error("balanced ridge requires both classes")
    weights = np.where(
        y == 1.0,
        0.5 / n_positive,
        0.5 / n_background,
    )
    mean = np.sum(weights[:, None] * x, axis=0)
    variance = np.sum(weights[:, None] * np.square(x - mean), axis=0)
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), std_floor)
    standardized = (x - mean) / scale
    design = np.column_stack((standardized, np.ones(len(x), dtype=np.float64)))
    penalty = np.diag([ridge_lambda] * x.shape[1] + [0.0])
    lhs = design.T @ (weights[:, None] * design) + penalty
    rhs = design.T @ (weights * y)
    solution = np.linalg.solve(lhs, rhs)
    return RidgeModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficient=tuple(float(value) for value in solution[:-1]),
        intercept=float(solution[-1]),
    )


def empirical_upper_tail(
    calibration_scores: np.ndarray, held_scores: np.ndarray
) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    held = np.asarray(held_scores, dtype=np.float64)
    if (
        calibration.ndim != 1
        or held.ndim != 1
        or not len(calibration)
        or not np.isfinite(calibration).all()
        or not np.isfinite(held).all()
    ):
        raise R0Error("empirical-tail inputs are malformed")
    left = np.searchsorted(calibration, held, side="left")
    count_greater_equal = len(calibration) - left
    return (1.0 + count_greater_equal) / (1.0 + len(calibration))


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["tail_probability"]),
        -float(row["raw_score"]),
        str(row["identity"]),
        int(row["source_id"]),
        int(row["target_id"]),
    )


def greedy_node_disjoint(
    rows: Sequence[Mapping[str, Any]], *, budget: int = BUDGET
) -> list[dict[str, Any]]:
    if type(budget) is not int or budget < 1:
        raise R0Error("selection budget must be a positive integer")
    occupied: set[tuple[str, int]] = set()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=rank_key):
        identity = str(row["identity"])
        source = int(row["source_id"])
        target = int(row["target_id"])
        source_key = (identity, source)
        target_key = (identity, target)
        if source == target or source_key in occupied or target_key in occupied:
            continue
        occupied.update((source_key, target_key))
        selected.append(dict(row))
        if len(selected) == budget:
            break
    return selected


def marginal_value_bound(
    *, budget: int = BUDGET, counts: Mapping[str, int] = APNCC_COUNTS
) -> dict[str, Any]:
    denominator = sum(int(counts[key]) for key in ("edge_tp", "edge_fp", "edge_fn"))
    if denominator <= 0 or budget < 0:
        raise R0Error("marginal-value inputs are invalid")
    raw = budget / denominator
    adjusted = budget * 1.1 / denominator
    return {
        "edge_denominator": denominator,
        "perfect_edit_count": budget,
        "maximum_raw_edge_jaccard_delta": raw,
        "maximum_official_node_adjustment_factor": 1.1,
        "maximum_adjusted_edge_jaccard_delta": adjusted,
        "maximum_node_recall_delta": 0.0,
        "graph_floor": GRAPH_FLOOR,
        "graph_physically_capable_of_floor": adjusted >= GRAPH_FLOOR,
    }


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
    except FileExistsError as exc:
        raise R0Error(f"refusing to overwrite frozen artifact: {path}") from exc


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def model_fingerprint(model: RidgeModel, calibration: np.ndarray) -> str:
    calibration_array = np.asarray(calibration, dtype="<f8")
    if calibration_array.ndim != 1 or not np.isfinite(calibration_array).all():
        raise R0Error("cannot fingerprint malformed calibration")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            asdict(model), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    digest.update(calibration_array.tobytes(order="C"))
    return digest.hexdigest()


def _contains_truth_or_terminal_field(value: Any) -> bool:
    forbidden = ("label", "correct", "precision", "selected", "graph")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(token in str(key).lower() for token in forbidden):
                return True
            if _contains_truth_or_terminal_field(child):
                return True
    elif isinstance(value, list):
        return any(_contains_truth_or_terminal_field(child) for child in value)
    return False


def _validate_candidate_row(row: Mapping[str, Any], *, identity: str) -> None:
    expected = {
        "identity",
        "held_fold",
        "source_id",
        "target_id",
        "source_frame",
        "features",
        "raw_score",
        "tail_probability",
        "topology",
        "action",
    }
    if set(row) != expected or row.get("identity") != identity:
        raise R0Error(f"truth-blind shard candidate schema drifted: {identity}")
    features = row.get("features")
    try:
        numeric = [
            float(row["raw_score"]),
            float(row["tail_probability"]),
            *(float(features[name]) for name in FEATURE_NAMES),
        ]
        source = int(row["source_id"])
        target = int(row["target_id"])
        held_fold = int(row["held_fold"])
        source_frame = int(row["source_frame"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise R0Error(f"truth-blind shard candidate is malformed: {identity}") from exc
    if (
        not isinstance(features, Mapping)
        or set(features) != set(FEATURE_NAMES)
        or not all(math.isfinite(value) for value in numeric)
        or not 0.0 < numeric[1] <= 1.0
        or source == target
        or held_fold != outer_fold(identity)
        or source_frame < 0
        or row.get("topology") != 0
        or row.get("action") != "add_edge_remove_none"
    ):
        raise R0Error(f"truth-blind shard candidate values drifted: {identity}")


def _load_truth_blind_shard(
    path: Path,
    *,
    identity: str,
    expected_rows: int,
    expected_full_shell_rows: int,
    expected_model_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = load_json(path)
    if _contains_truth_or_terminal_field(value):
        raise R0Error(f"truth or terminal field leaked into shard: {path}")
    candidates = value.get("candidates")
    video_report = value.get("video_report")
    if (
        value.get("schema") != "g02.r0-shell-freefree.truth-blind-video-shard.v1"
        or value.get("mechanism") != MECHANISM
        or value.get("scientific_contract_sha256") != SCIENTIFIC_CONTRACT_SHA256
        or value.get("identity") != identity
        or value.get("model_fingerprint") != expected_model_fingerprint
        or not isinstance(candidates, list)
        or len(candidates) != expected_rows
        or not isinstance(video_report, Mapping)
        or video_report.get("identity") != identity
        or int(video_report.get("eligible_topology_0_rows", -1)) != expected_rows
        or int(video_report.get("full_shell_rows", -1)) != expected_full_shell_rows
    ):
        raise R0Error(f"truth-blind shard authentication failed: {path}")
    observed_pairs: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise R0Error(f"truth-blind shard candidate is not an object: {path}")
        _validate_candidate_row(candidate, identity=identity)
        pair = (int(candidate["source_id"]), int(candidate["target_id"]))
        if pair in observed_pairs:
            raise R0Error(f"truth-blind shard candidate pair repeats: {path}")
        observed_pairs.add(pair)
        normalized.append(dict(candidate))
    return normalized, dict(video_report)


def _write_shard_manifest(
    shard_root: Path,
    expected_identities: Sequence[str],
    authenticated: Sequence[Mapping[str, Any]],
) -> None:
    value = {
        "schema": "g02.r0-shell-freefree.truth-blind-shard-manifest.v1",
        "mechanism": MECHANISM,
        "scientific_contract_sha256": SCIENTIFIC_CONTRACT_SHA256,
        "status": (
            "COMPLETE_TRUTH_BLIND_POPULATION"
            if len(authenticated) == len(expected_identities)
            else "PARTIAL_TRUTH_BLIND_POPULATION"
        ),
        "expected_video_count": len(expected_identities),
        "completed_video_count": len(authenticated),
        "expected_identities": list(expected_identities),
        "shards": list(authenticated),
        "selection_emitted": False,
        "ground_truth_opened": False,
        "precision_emitted": False,
        "graph_outcome_emitted": False,
    }
    _atomic_replace_json(shard_root / "MANIFEST.json", value)


def _load_training(
    appearance_report_path: Path,
    input_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if sha256_file(appearance_report_path) != EXPECTED_INPUT_SHA256["appearance"]:
        raise R0Error("appearance aggregate SHA-256 drifted")
    report = load_json(appearance_report_path)
    names = tuple(str(value) for value in report.get("feature_names", ()))
    try:
        indices = [names.index(name) for name in FEATURE_NAMES]
    except ValueError as exc:
        raise R0Error("frozen training feature is absent") from exc
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    topology: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    videos = report.get("videos")
    if not isinstance(videos, list):
        raise R0Error("appearance report videos are malformed")
    for row in videos:
        identity = str(row.get("identity", ""))
        if row.get("partition") != "train" or not identity.startswith("train:44b6_"):
            continue
        source = input_root / str(row["path"])
        observed_sha = sha256_file(source)
        if observed_sha != str(row["sha256"]):
            raise R0Error(f"appearance example SHA-256 drifted: {identity}")
        value = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping) or tuple(value.get("feature_names", ())) != names:
            raise R0Error(f"appearance example contract drifted: {identity}")
        full = np.asarray(value["features"], dtype=np.float64)
        current_labels = np.asarray(value["labels"], dtype=np.int8)
        current_topology = np.asarray(value["topology"], dtype=np.int8)
        if (
            full.ndim != 2
            or current_labels.shape != (len(full),)
            or current_topology.shape != (len(full),)
        ):
            raise R0Error(f"appearance arrays disagree: {identity}")
        features.append(full[:, indices])
        labels.append(current_labels)
        topology.append(current_topology)
        identities.append(np.full(len(full), identity, dtype=object))
        manifest.append(
            {"identity": identity, "rows": len(full), "sha256": observed_sha}
        )
    if len(manifest) != 36:
        raise R0Error(f"expected 36 44b6 training videos, found {len(manifest)}")
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(topology),
        np.concatenate(identities),
        manifest,
    )


def fit_nested_models(
    features: np.ndarray,
    labels: np.ndarray,
    topology: np.ndarray,
    identities: np.ndarray,
) -> tuple[dict[int, RidgeModel], dict[int, np.ndarray], list[dict[str, Any]]]:
    folds = np.asarray([outer_fold(str(value)) for value in identities], dtype=np.int8)
    models: dict[int, RidgeModel] = {}
    calibrations: dict[int, np.ndarray] = {}
    reports: list[dict[str, Any]] = []
    for held_fold in range(5):
        calibration_fold = (held_fold + 1) % 5
        fit_mask = (folds != held_fold) & (folds != calibration_fold)
        calibration_mask = (folds == calibration_fold) & (topology == 0)
        model = fit_balanced_ridge(features[fit_mask], labels[fit_mask] == 1)
        calibration = model.score(features[calibration_mask])
        if not len(calibration):
            raise R0Error(f"outer fold {held_fold} has no topology-0 calibration rows")
        models[held_fold] = model
        calibrations[held_fold] = np.sort(calibration)
        reports.append(
            {
                "held_fold": held_fold,
                "calibration_fold": calibration_fold,
                "fit_rows": int(np.sum(fit_mask)),
                "fit_positive_rows": int(np.sum(fit_mask & (labels == 1))),
                "fit_background_rows": int(np.sum(fit_mask & (labels != 1))),
                "calibration_topology_0_rows": len(calibration),
                "model": asdict(model),
            }
        )
    return models, calibrations, reports


def _free_free_rows(
    source_frame: Any,
    target_frame: Any,
    selected_out: Mapping[int, int],
    selected_in: Mapping[int, int],
) -> list[tuple[int, int, int, int, float]]:
    source_ids = np.asarray(source_frame.node_ids, dtype=np.int64)
    target_ids = np.asarray(target_frame.node_ids, dtype=np.int64)
    source_points = np.asarray(source_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
    target_points = np.asarray(target_frame.coords_zyx, dtype=np.float64) * SCALE_ZYX
    if not len(source_points) or not len(target_points):
        return []
    tree = cKDTree(target_points)
    result: list[tuple[int, int, int, int, float]] = []
    for source_index, target_index, _rank, distance, _displacement in shell_candidate_indices(
        source_ids,
        source_points,
        target_ids,
        target_points,
        target_tree=tree,
    ):
        source = int(source_ids[source_index])
        target = int(target_ids[target_index])
        if source not in selected_out and target not in selected_in:
            result.append((source_index, target_index, source, target, float(distance)))
    return result


def score_eligible_population(
    *,
    prepare_report_path: Path,
    raw_zip_path: Path,
    cache_root: Path,
    scorer: Any,
    models: Mapping[int, RidgeModel],
    calibrations: Mapping[int, np.ndarray],
    shard_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256_file(prepare_report_path) != EXPECTED_INPUT_SHA256["prepare"]:
        raise R0Error("shell-prepare aggregate SHA-256 drifted")
    prepare = load_json(prepare_report_path)
    videos = [
        row
        for row in prepare.get("videos", [])
        if row.get("partition") == "train"
        and str(row.get("identity", "")).startswith("train:44b6_")
    ]
    if len(videos) != 36:
        raise R0Error(f"expected 36 prepared 44b6 videos, found {len(videos)}")
    expected_identities = [str(row["identity"]) for row in videos]
    expected_shard_names = {
        f"{identity.replace(':', '__', 1)}.json" for identity in expected_identities
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in shard_root.glob("*.json")
        if path.name != "MANIFEST.json" and path.name not in expected_shard_names
    )
    if unexpected:
        raise R0Error(f"unexpected truth-blind shard files: {unexpected}")
    candidates: list[dict[str, Any]] = []
    video_reports: list[dict[str, Any]] = []
    authenticated_shards: list[dict[str, Any]] = []
    _write_shard_manifest(shard_root, expected_identities, authenticated_shards)
    with ZipStore(raw_zip_path, mode="r") as store:
        for video_index, row in enumerate(videos, start=1):
            identity = str(row["identity"])
            slug = identity.replace(":", "__", 1)
            expected = int(row.get("counts", {}).get("topology_0", -1))
            expected_full_shell = int(row.get("counts", {}).get("shell_pairs", -1))
            held_fold = outer_fold(identity)
            fingerprint = model_fingerprint(models[held_fold], calibrations[held_fold])
            shard_path = shard_root / f"{slug}.json"
            if shard_path.exists():
                shard_candidates, shard_video_report = _load_truth_blind_shard(
                    shard_path,
                    identity=identity,
                    expected_rows=expected,
                    expected_full_shell_rows=expected_full_shell,
                    expected_model_fingerprint=fingerprint,
                )
                candidates.extend(shard_candidates)
                video_reports.append(shard_video_report)
                authenticated_shards.append(
                    {
                        "identity": identity,
                        "path": str(shard_path),
                        "sha256": sha256_file(shard_path),
                        "rows": len(shard_candidates),
                        "model_fingerprint": fingerprint,
                    }
                )
                _write_shard_manifest(
                    shard_root, expected_identities, authenticated_shards
                )
                print(
                    f"R0 eligible {video_index:02d}/36 {identity} "
                    f"rows={len(shard_candidates)} resumed=1",
                    flush=True,
                )
                continue
            cache = bh_model.load_sparse_cache(
                cache_root / f"{slug}.bhc",
                expected_identity=_cache_identity(cache_root / f"{slug}.json"),
            )
            selected, _probabilities, no_link, graph = _baseline(cache, scorer)
            if int(graph["raw_selected_edges"]) != int(row["graph"]["raw_selected_edges"]):
                raise R0Error(f"selected graph count drifted: {identity}")
            selected_out = {int(source): int(target) for source, target in selected}
            selected_in = {int(target): int(source) for source, target in selected}
            dataset = identity.split(":", 1)[1]
            image = zarr.open_array(
                store=store, path=f"train/{dataset}.zarr/0", mode="r"
            )
            model = models[held_fold]
            calibration = calibrations[held_fold]
            identity_rows = 0
            for source_frame, target_frame in zip(
                cache.detections[:-1], cache.detections[1:], strict=True
            ):
                free_rows = _free_free_rows(
                    source_frame, target_frame, selected_out, selected_in
                )
                if not free_rows:
                    continue
                source_time = int(source_frame.frame_index)
                if int(target_frame.frame_index) != source_time + 1:
                    raise R0Error(f"nonadjacent cache frames: {identity}")
                source_needed = sorted({item[0] for item in free_rows})
                target_needed = sorted({item[1] for item in free_rows})
                source_image = np.asarray(image[source_time])
                target_image = np.asarray(image[source_time + 1])
                source_vectors = {
                    index: unit_vector(
                        extract_patch(
                            source_image,
                            np.asarray(source_frame.coords_zyx[index], dtype=np.float64),
                        )
                    )
                    for index in source_needed
                }
                target_vectors = {
                    index: unit_vector(
                        extract_patch(
                            target_image,
                            np.asarray(target_frame.coords_zyx[index], dtype=np.float64),
                        )
                    )
                    for index in target_needed
                }
                feature_rows = np.asarray(
                    [
                        (
                            float(np.dot(source_vectors[s_index], target_vectors[t_index])),
                            float(no_link.get(source, 1.0)),
                            distance,
                        )
                        for s_index, t_index, source, _target, distance in free_rows
                    ],
                    dtype=np.float64,
                )
                raw_scores = model.score(feature_rows)
                tails = empirical_upper_tail(calibration, raw_scores)
                for item, features, raw_score, tail in zip(
                    free_rows, feature_rows, raw_scores, tails, strict=True
                ):
                    _s_index, _t_index, source, target, _distance = item
                    candidates.append(
                        {
                            "identity": identity,
                            "held_fold": outer_fold(identity),
                            "source_id": source,
                            "target_id": target,
                            "source_frame": source_time,
                            "features": {
                                name: float(value)
                                for name, value in zip(FEATURE_NAMES, features, strict=True)
                            },
                            "raw_score": float(raw_score),
                            "tail_probability": float(tail),
                            "topology": 0,
                            "action": "add_edge_remove_none",
                        }
                    )
                identity_rows += len(free_rows)
                del source_image, target_image, source_vectors, target_vectors
            if identity_rows != expected:
                raise R0Error(
                    f"topology-0 population drifted for {identity}: {identity_rows} != {expected}"
                )
            video_report = {
                "identity": identity,
                "held_fold": held_fold,
                "eligible_topology_0_rows": identity_rows,
                "full_shell_rows": expected_full_shell,
            }
            identity_candidates = candidates[-identity_rows:] if identity_rows else []
            shard_value = {
                "schema": "g02.r0-shell-freefree.truth-blind-video-shard.v1",
                "mechanism": MECHANISM,
                "scientific_contract_sha256": SCIENTIFIC_CONTRACT_SHA256,
                "identity": identity,
                "model_fingerprint": fingerprint,
                "candidates": identity_candidates,
                "video_report": video_report,
            }
            if _contains_truth_or_terminal_field(shard_value):
                raise R0Error(f"truth or terminal field leaked before shard: {identity}")
            _atomic_create_json(shard_path, shard_value)
            authenticated_candidates, authenticated_video_report = (
                _load_truth_blind_shard(
                    shard_path,
                    identity=identity,
                    expected_rows=expected,
                    expected_full_shell_rows=expected_full_shell,
                    expected_model_fingerprint=fingerprint,
                )
            )
            if authenticated_candidates != identity_candidates:
                raise R0Error(f"truth-blind shard replay drifted: {identity}")
            video_reports.append(authenticated_video_report)
            authenticated_shards.append(
                {
                    "identity": identity,
                    "path": str(shard_path),
                    "sha256": sha256_file(shard_path),
                    "rows": len(authenticated_candidates),
                    "model_fingerprint": fingerprint,
                }
            )
            _write_shard_manifest(shard_root, expected_identities, authenticated_shards)
            print(
                f"R0 eligible {video_index:02d}/36 {identity} rows={identity_rows}",
                flush=True,
            )
    return candidates, video_reports


def label_sealed_actions(
    *,
    actions: Sequence[Mapping[str, Any]],
    raw_zip_path: Path,
    cache_root: Path,
) -> list[dict[str, Any]]:
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in actions:
        by_identity[str(row["identity"])].append(row)
    labeled: list[dict[str, Any]] = []
    with ZipStore(raw_zip_path, mode="r") as store:
        for identity in sorted(by_identity):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                cache_root / f"{slug}.bhc",
                expected_identity=_cache_identity(cache_root / f"{slug}.json"),
            )
            dataset = identity.split(":", 1)[1]
            gt = read_geff_arrays(store, "train", dataset)
            detection_to_gt, gt_edges = match_detections_to_gt(cache.detections, gt)
            outdegree: dict[int, int] = defaultdict(int)
            indegree: dict[int, int] = defaultdict(int)
            for source, target in gt_edges:
                outdegree[int(source)] += 1
                indegree[int(target)] += 1
            for action in by_identity[identity]:
                pair = (int(action["source_id"]), int(action["target_id"]))
                label = label_pair(
                    pair, detection_to_gt, gt_edges, outdegree, indegree
                )
                labeled.append(
                    {
                        **dict(action),
                        "official_fit_label": int(label),
                        "correct_nondivision_continuation": label == 1,
                    }
                )
    return sorted(labeled, key=rank_key)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists():
        raise R0Error(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    bound = marginal_value_bound()
    if bound["graph_physically_capable_of_floor"]:
        raise R0Error("marginal-value gate unexpectedly opened")
    training, labels, topology, identities, training_manifest = _load_training(
        args.appearance, args.input_root
    )
    models, calibrations, fold_reports = fit_nested_models(
        training, labels, topology, identities
    )
    scorer, _model, scorer_record = _load_base_scorer(
        args.edge_artifact,
        args.edge_auxiliary,
        args.support_root,
        "cpu",
    )
    candidates, video_reports = score_eligible_population(
        prepare_report_path=args.prepare,
        raw_zip_path=args.zip,
        cache_root=args.cache,
        scorer=scorer,
        models=models,
        calibrations=calibrations,
        shard_root=args.shard_root,
    )
    selected = greedy_node_disjoint(candidates)
    repeat = greedy_node_disjoint(list(reversed(candidates)))
    if selected != repeat:
        raise R0Error("selection depends on population traversal order")
    if len(selected) != BUDGET:
        raise R0Error(f"fewer than {BUDGET} node-disjoint eligible actions")
    if any(row.get("topology") != 0 for row in selected):
        raise R0Error("selected action escaped topology-0 eligibility")
    sealed = {
        "schema": "g02.r0-shell-freefree.selected-actions.v1",
        "mechanism": MECHANISM,
        "status": "sealed_before_ground_truth_open",
        "budget": BUDGET,
        "actions": selected,
    }
    selected_path = args.output / "SELECTED_ACTIONS_TRUTH_BLIND.json"
    _atomic_create_json(selected_path, sealed)
    selected_sha = sha256_file(selected_path)
    if any("label" in key.lower() or "correct" in key.lower() for row in selected for key in row):
        raise R0Error("truth field leaked into sealed action")
    labeled = label_sealed_actions(
        actions=selected,
        raw_zip_path=args.zip,
        cache_root=args.cache,
    )
    correct = sum(bool(row["correct_nondivision_continuation"]) for row in labeled)
    precision = correct / BUDGET
    precision_status = (
        "SUPPORTS_NARROW_EVIDENCE" if correct == BUDGET else "KILL_EARLY"
    )
    result = {
        "schema": SCHEMA,
        "status": "complete_graph_closed_by_marginal_value_gate",
        "mechanism": MECHANISM,
        "scope": {
            "family": "44b6",
            "fit_only": True,
            "video_count": len(video_reports),
            "full_shell_rows": sum(row["full_shell_rows"] for row in video_reports),
            "eligible_topology_0_rows": len(candidates),
            "occupied_shell_rows_learned_rescored": 0,
            "validation12_used": False,
            "six_bba_used": False,
            "cv57_used": False,
        },
        "model": {
            "features": list(FEATURE_NAMES),
            "ridge_lambda": RIDGE_LAMBDA,
            "folds": fold_reports,
            "training_examples": training_manifest,
        },
        "selection": {
            "budget": BUDGET,
            "sealed_edit_set_size": len(selected),
            "applied_edit_set_size": 0,
            "selected_actions_path": str(selected_path),
            "selected_actions_sha256": selected_sha,
            "node_disjoint": True,
            "traversal_order_replay_exact": True,
            "actions_with_truth": labeled,
        },
        "precision_gate": {
            "correct": correct,
            "denominator": BUDGET,
            "precision": precision,
            "required": 1.0,
            "status": precision_status,
        },
        "marginal_value_gate": bound,
        "graph": {
            "opened": False,
            "measured": False,
            "reason": (
                "preregistered perfect-B4 adjusted ceiling is below +0.005; "
                "candidate abstains regardless of precision"
            ),
            "exact_executed_candidate": "zero-edit no-op",
            "delta_adjusted_raw_node": [0.0, 0.0, 0.0],
            "potential_perfect_b4_upper_bound_adjusted_raw_node": [
                bound["maximum_adjusted_edge_jaccard_delta"],
                bound["maximum_raw_edge_jaccard_delta"],
                0.0,
            ],
        },
        "properties": {
            "appearance_grounded": True,
            "narrow": True,
            "node_disjoint": True,
            "recall_preserving": True,
            "details": {
                "appearance_grounded": "raw 16x32x32 3-D pair NCC is mandatory",
                "narrow": "only incumbent-free source and target endpoints; B=4",
                "node_disjoint": "both endpoints disjoint across sealed actions",
                "recall_preserving": "proposed actions are add-only; executed candidate abstains",
            },
        },
        "videos": video_reports,
        "provenance": {
            "prepare_sha256": sha256_file(args.prepare),
            "appearance_sha256": sha256_file(args.appearance),
            "frozen_pairwise": scorer_record,
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
    }
    _atomic_create_json(args.output / "RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    parser_value = argparse.ArgumentParser(description=__doc__)
    parser_value.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser_value.add_argument(
        "--prepare",
        type=Path,
        default=INPUT_ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json",
    )
    parser_value.add_argument(
        "--appearance",
        type=Path,
        default=INPUT_ROOT / "reports/g02/extended_radius_shell_appearance_v1/report.json",
    )
    parser_value.add_argument(
        "--zip",
        type=Path,
        default=INPUT_ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser_value.add_argument(
        "--cache", type=Path, default=INPUT_ROOT / "reports/g02/fitcache_117/bh9out/cache"
    )
    parser_value.add_argument(
        "--edge-artifact",
        type=Path,
        default=INPUT_ROOT
        / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json",
    )
    parser_value.add_argument(
        "--edge-auxiliary",
        type=Path,
        default=INPUT_ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt",
    )
    parser_value.add_argument(
        "--support-root", type=Path, default=INPUT_ROOT / "external/Tracking_Support"
    )
    parser_value.add_argument("--output", type=Path, default=HERE / "run_v1")
    parser_value.add_argument(
        "--shard-root", type=Path, default=HERE / "truth_blind_shards_v1"
    )
    return parser_value


if __name__ == "__main__":
    final = evaluate(parser().parse_args())
    print(
        json.dumps(
            {
                "status": final["status"],
                "precision": final["precision_gate"]["precision"],
                "eligible": final["scope"]["eligible_topology_0_rows"],
                "graph_delta": final["graph"]["delta_adjusted_raw_node"],
            },
            sort_keys=True,
        )
    )
