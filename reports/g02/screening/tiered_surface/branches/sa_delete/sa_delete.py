#!/usr/bin/env python3
"""DELETE-O0 / AE-P1 conservative true-edge retention graph gate.

This branch-local runner consumes only hash-authenticated, fit-only artifacts
transitively enumerated by the frozen P0 manifest.  It never opens audit, test,
CV57, public, or sibling-agent artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
BRANCH_ROOT = Path(__file__).resolve().parent
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import analyze_action_class_oracle as p0_engine  # noqa: E402
import scoreall  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.sa-delete-retention.v1"
FAMILIES = ("44b6", "6bba")
INPUT_PINS = {
    "n1_interface": (
        ROOT
        / "reports/g02/screening/tiered_surface/n1_feasibility_compiler/interface_v1.json",
        "2a72950c65b5fd338f4d8935e086afcb9bba884365224e014e86e7900509f865",
    ),
    "n1_implementation": (
        ROOT / "src/celltrack_alpha/action_compiler.py",
        "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
    ),
    "folds": (
        ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
        "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769",
    ),
    "p0": (
        ROOT
        / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json",
        "8ad548c134c419e2e76b8661848e90493d674731ceb34aab231c5c18c8889c06",
    ),
    "fit_provenance": (
        ROOT / "reports/g02/fit_provenance_v1.json",
        "6a9139bf281b98d2c3869c8f12f80fb9655f85ba1c0f7f91f0a51acd7c37f14f",
    ),
}
FORBIDDEN_RUNTIME_TOKENS = (
    "cv57",
    "holdout",
    "audit",
    "/test/",
    "submission",
    "top50_parallel_dags",
)
FEATURE_NAMES = (
    "distance_um",
    "delta_t",
    "source_detection_score",
    "target_detection_score",
    "minimum_detection_score",
    "raw_endpoint_fraction",
    "synthetic_endpoint_fraction",
    "source_indegree",
    "source_outdegree",
    "target_indegree",
    "target_outdegree",
    "midpoint_time_fraction",
)
INNER_FOLDS = 5
SEED = 20260728
NEGATIVE_TO_POSITIVE_RATIO = 8
MIN_NEGATIVES_PER_VIDEO = 512
L2_STRENGTH = 1.0e-2
MAX_OPTIMIZER_ITERATIONS = 50
TRUE_EDGE_LOWER_QUANTILE = 0.001
MIN_CALIBRATION_TRUE_EDGES = 100
MIN_ACTION_COUNT = 16
MIN_ACTION_FRACTION = 0.005
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


class DeleteGateError(RuntimeError):
    """The frozen retention/no-link contract could not be preserved."""


@dataclass(frozen=True)
class VideoData:
    identity: str
    family: str
    edges: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    baseline: Mapping[str, Any]
    fragment_sha256: str
    cache_sha256: str
    meta_sha256: str


@dataclass(frozen=True)
class LinearRetention:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    objective: float
    converged: bool
    iterations: int

    @property
    def parameter_count(self) -> int:
        return int(len(self.coefficients) + 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeleteGateError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DeleteGateError(f"JSON root is not an object: {path}")
    return value


def json_safe(value: Any) -> Any:
    """Map undefined numeric diagnostics to JSON null without changing results."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise DeleteGateError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def safe_runtime_path(path: Path) -> None:
    if path.resolve() == INPUT_PINS["folds"][0].resolve():
        return
    lowered = str(path).lower()
    token = next((token for token in FORBIDDEN_RUNTIME_TOKENS if token in lowered), None)
    if token is not None:
        raise DeleteGateError(f"forbidden runtime path token {token!r}: {path}")


def inner_fold(identity: str) -> int:
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % INNER_FOLDS


def nested_partitions(
    identities: Sequence[str],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    values = tuple(sorted(identities))
    if len(values) != len(set(values)):
        raise DeleteGateError("nested-fold identities are not unique")
    result = []
    for held_fold in range(INNER_FOLDS):
        calibration_fold = (held_fold + 1) % INNER_FOLDS
        held = tuple(value for value in values if inner_fold(value) == held_fold)
        calibration = tuple(
            value for value in values if inner_fold(value) == calibration_fold
        )
        fit = tuple(
            value
            for value in values
            if inner_fold(value) not in (held_fold, calibration_fold)
        )
        if (
            not fit
            or not calibration
            or not held
            or set(fit) & set(calibration)
            or set(fit) & set(held)
            or set(calibration) & set(held)
            or set((*fit, *calibration, *held)) != set(values)
        ):
            raise DeleteGateError("invalid nested fit/calibration/held partition")
        result.append((fit, calibration, held))
    return tuple(result)


def _edge_set(graph: Any) -> np.ndarray:
    values = np.asarray(
        sorted((int(edge.source_id), int(edge.target_id)) for edge in graph.edges),
        dtype=np.int64,
    )
    if values.ndim != 2 or values.shape[1:] != (2,):
        raise DeleteGateError("baseline graph edge array is malformed")
    return values


def edge_features(graph: Any, cache: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical edges and label-free pretrained/structural features."""

    edges = _edge_set(graph)
    scores: dict[int, float] = {}
    for frame in cache.detections:
        for node_id, score in zip(frame.node_ids, frame.scores, strict=True):
            node = int(node_id)
            if node in scores:
                raise DeleteGateError("duplicate raw detection node")
            value = float(score)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise DeleteGateError("invalid pretrained detection score")
            scores[node] = value

    indegree: dict[int, int] = defaultdict(int)
    outdegree: dict[int, int] = defaultdict(int)
    for source, target in edges:
        outdegree[int(source)] += 1
        indegree[int(target)] += 1
    maximum_t = max((int(node.t) for node in graph.nodes.values()), default=0)
    denominator_t = max(1, maximum_t)
    features = np.empty((len(edges), len(FEATURE_NAMES)), dtype=np.float32)
    for index, (raw_source, raw_target) in enumerate(edges):
        source = int(raw_source)
        target = int(raw_target)
        source_node = graph.nodes[source]
        target_node = graph.nodes[target]
        delta_t = int(target_node.t) - int(source_node.t)
        displacement = (
            np.asarray(target_node.coord_zyx, dtype=np.float64)
            - np.asarray(source_node.coord_zyx, dtype=np.float64)
        ) * SCALE_ZYX_UM
        distance = float(np.linalg.norm(displacement))
        source_raw = source in scores
        target_raw = target in scores
        source_score = scores.get(source, 0.0)
        target_score = scores.get(target, 0.0)
        features[index] = (
            distance,
            delta_t,
            source_score,
            target_score,
            min(source_score, target_score),
            0.5 * (int(source_raw) + int(target_raw)),
            0.5 * (int(not source_raw) + int(not target_raw)),
            indegree[source],
            outdegree[source],
            indegree[target],
            outdegree[target],
            0.5 * (int(source_node.t) + int(target_node.t)) / denominator_t,
        )
    if not np.isfinite(features).all():
        raise DeleteGateError("non-finite retention features")
    return edges, features


def true_edge_labels(
    graph: Any,
    gt: Mapping[str, Any],
    scale: tuple[float, float, float],
    edges: np.ndarray,
) -> np.ndarray:
    pred_to_gt, _gt_to_pred = p0_engine.official_node_matching(graph, gt, scale)
    truth, _successors, _predecessors, _nodes = p0_engine._truth_topology(gt)
    return np.asarray(
        [
            int(
                p0_engine._true_prediction_edge(
                    (int(source), int(target)), pred_to_gt, truth
                )
            )
            for source, target in edges
        ],
        dtype=np.int8,
    )


def deterministic_training_indices(
    identity: str, edges: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    wanted = min(
        len(negative),
        max(MIN_NEGATIVES_PER_VIDEO, NEGATIVE_TO_POSITIVE_RATIO * len(positive)),
    )
    if wanted < len(negative):
        hashes = np.empty(len(negative), dtype=np.uint64)
        for offset, index in enumerate(negative):
            source, target = edges[index]
            digest = hashlib.sha256(
                f"{identity}|{int(source)}|{int(target)}|{SEED}".encode("ascii")
            ).digest()
            hashes[offset] = int.from_bytes(digest[:8], "big")
        chosen = negative[np.argsort(hashes, kind="stable")[:wanted]]
    else:
        chosen = negative
    return np.sort(np.concatenate((positive, chosen))).astype(np.int64, copy=False)


def fit_linear_retention(
    feature_chunks: Sequence[np.ndarray],
    label_chunks: Sequence[np.ndarray],
) -> LinearRetention:
    if not feature_chunks or len(feature_chunks) != len(label_chunks):
        raise DeleteGateError("empty or misaligned fit chunks")
    x = np.concatenate(feature_chunks).astype(np.float64, copy=False)
    y = np.concatenate(label_chunks).astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y):
        raise DeleteGateError("fit matrix shape drifted")
    if not np.any(y == 0) or not np.any(y == 1):
        raise DeleteGateError("fit data requires both retention classes")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    z = (x - mean) / scale
    positive_weight = 0.5 / float(np.mean(y == 1))
    negative_weight = 0.5 / float(np.mean(y == 0))
    sample_weight = np.where(y == 1, positive_weight, negative_weight)

    def sigmoid(logits: np.ndarray) -> np.ndarray:
        values = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        values[~positive] = exponent / (1.0 + exponent)
        return values

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-1]
        intercept = float(parameters[-1])
        logits = z @ coefficients + intercept
        losses = np.logaddexp(0.0, logits) - y * logits
        value = float(
            np.mean(sample_weight * losses)
            + 0.5 * L2_STRENGTH * np.dot(coefficients, coefficients)
        )
        residual = sample_weight * (sigmoid(logits) - y) / len(y)
        gradient = np.concatenate(
            (
                z.T @ residual + L2_STRENGTH * coefficients,
                np.asarray([residual.sum()], dtype=np.float64),
            )
        )
        return value, gradient

    parameters = np.zeros(len(FEATURE_NAMES) + 1, dtype=np.float64)
    converged = False
    iterations = 0
    current_value, gradient = objective(parameters)
    for iteration in range(1, MAX_OPTIMIZER_ITERATIONS + 1):
        logits = z @ parameters[:-1] + parameters[-1]
        probabilities = sigmoid(logits)
        curvature = (
            sample_weight * probabilities * (1.0 - probabilities) / len(y)
        )
        weighted_z = z * curvature[:, None]
        hessian = np.empty(
            (len(FEATURE_NAMES) + 1, len(FEATURE_NAMES) + 1),
            dtype=np.float64,
        )
        hessian[:-1, :-1] = z.T @ weighted_z
        hessian[:-1, :-1] += L2_STRENGTH * np.eye(len(FEATURE_NAMES))
        cross = z.T @ curvature
        hessian[:-1, -1] = cross
        hessian[-1, :-1] = cross
        hessian[-1, -1] = curvature.sum() + 1.0e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise DeleteGateError("retention Newton system is singular") from exc
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            candidate = parameters - step_scale * step
            candidate_value, candidate_gradient = objective(candidate)
            if candidate_value <= current_value:
                parameters = candidate
                current_value = candidate_value
                gradient = candidate_gradient
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        iterations = iteration
        if np.max(np.abs(gradient)) <= 1.0e-8:
            converged = True
            break
    if not np.isfinite(parameters).all() or not math.isfinite(current_value):
        raise DeleteGateError("retention fit produced non-finite parameters")
    return LinearRetention(
        mean=mean,
        scale=scale,
        coefficients=np.asarray(parameters[:-1], dtype=np.float64),
        intercept=float(parameters[-1]),
        objective=current_value,
        converged=converged,
        iterations=iterations,
    )


def retention_scores(model: LinearRetention, features: np.ndarray) -> np.ndarray:
    standardized = (
        features.astype(np.float64, copy=False) - model.mean
    ) / model.scale
    logits = standardized @ model.coefficients + model.intercept
    values = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    values[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    values[~positive] = exponent / (1.0 + exponent)
    if not np.isfinite(values).all():
        raise DeleteGateError("retention inference produced non-finite scores")
    return np.asarray(values, dtype=np.float64)


def conservative_threshold(
    score_chunks: Sequence[np.ndarray], label_chunks: Sequence[np.ndarray]
) -> tuple[float, dict[str, Any]]:
    scores = np.concatenate(score_chunks)
    labels = np.concatenate(label_chunks)
    true_scores = scores[labels == 1]
    if len(true_scores) < MIN_CALIBRATION_TRUE_EDGES:
        raise DeleteGateError("insufficient calibration true-edge support")
    threshold = float(
        np.quantile(true_scores, TRUE_EDGE_LOWER_QUANTILE, method="lower")
    )
    retained = scores >= threshold
    diagnostics = {
        "eligible_edges": int(len(scores)),
        "true_edges": int((labels == 1).sum()),
        "false_edges": int((labels == 0).sum()),
        "retained_true_edges": int(((labels == 1) & retained).sum()),
        "deleted_true_edges": int(((labels == 1) & ~retained).sum()),
        "deleted_false_edges": int(((labels == 0) & ~retained).sum()),
        "true_edge_retention": float(np.mean(retained[labels == 1])),
        "threshold": threshold,
        "quantile": TRUE_EDGE_LOWER_QUANTILE,
    }
    return threshold, diagnostics


def _family(identity: str) -> str:
    value = identity.split(":", 1)[1].split("_", 1)[0]
    if value not in FAMILIES:
        raise DeleteGateError(f"unexpected family: {identity}")
    return value


def authenticate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed: dict[str, Any] = {}
    for name, (path, expected) in INPUT_PINS.items():
        safe_runtime_path(path)
        actual = sha256_file(path)
        if actual != expected:
            raise DeleteGateError(f"frozen input hash mismatch: {name}")
        observed[name] = {"path": str(path), "sha256": actual}
    p0 = load_json(INPUT_PINS["p0"][0])
    folds = load_json(INPUT_PINS["folds"][0])
    provenance = load_json(INPUT_PINS["fit_provenance"][0])
    interface = load_json(INPUT_PINS["n1_interface"][0])
    if (
        p0.get("status") != "COMPLETE"
        or p0.get("scope", {}).get("fit_only") is not True
        or any(
            p0.get("scope", {}).get(key) is not False
            for key in ("audit_used", "test_used", "cv57_used", "held_out_used")
        )
    ):
        raise DeleteGateError("P0 scope is not complete fit-only evidence")
    identities = {str(row["identity"]) for row in p0["inputs"]["fit117_files"]}
    fold_zero = folds.get("folds", {}).get("0", {})
    if (
        len(identities) != 117
        or identities != set(fold_zero.get("fit_video_ids", ()))
        or identities & set(folds.get("audit", {}).get("video_ids", ()))
        or set(provenance.get("audit_group_ids", ()))
        != set(folds.get("audit", {}).get("group_ids", ()))
        or interface.get("schema_version") != 1
        or "delete_edges" not in interface.get("action_kinds", ())
    ):
        raise DeleteGateError("fit117/fold/audit/N1 interface contract drifted")

    archive = ROOT / p0["inputs"]["archive"]["path"]
    safe_runtime_path(archive)
    archive_sha = sha256_file(archive)
    if archive_sha != p0["inputs"]["archive"]["sha256"]:
        raise DeleteGateError("P0 raw archive hash mismatch")
    observed["p0_archive"] = {"path": str(archive), "sha256": archive_sha}
    for key in (
        "official_eval_sha256",
        "official_metrics_sha256",
        "official_division_metrics_sha256",
    ):
        if key == "official_eval_sha256":
            path = ROOT / "src/celltrack_alpha/official_eval.py"
        elif key == "official_metrics_sha256":
            path = (
                ROOT
                / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py"
            )
        else:
            path = (
                ROOT
                / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
            )
        actual = sha256_file(path)
        if actual != p0["inputs"][key]:
            raise DeleteGateError(f"P0 transitive scorer hash mismatch: {key}")
        observed[key] = {"path": str(path), "sha256": actual}
    return p0, folds, provenance, observed


def load_population(
    p0: Mapping[str, Any],
    *,
    archive: Path,
) -> tuple[dict[str, VideoData], list[dict[str, Any]]]:
    provenance_rows = {
        str(row["identity"]): row for row in p0["inputs"]["fit117_files"]
    }
    sample_rows = {str(row["identity"]): row for row in p0["samples"]}
    fragment_root = ROOT / str(p0["control"]["fragment_root"])
    cache_root = ROOT / "reports/g02/fitcache_117/bh9out/cache"
    store = ZipStore(archive, mode="r")
    outputs: dict[str, VideoData] = {}
    file_records = []
    try:
        for ordinal, identity in enumerate(sorted(provenance_rows), start=1):
            record = provenance_rows[identity]
            stem = identity.replace(":", "__", 1)
            fragment = fragment_root / f"{stem}.csv"
            cache_path = cache_root / f"{stem}.bhc"
            meta_path = cache_root / f"{stem}.json"
            for path in (fragment, cache_path, meta_path):
                safe_runtime_path(path)
            actual = {
                "fragment_sha256": sha256_file(fragment),
                "cache_sha256": sha256_file(cache_path),
                "meta_sha256": sha256_file(meta_path),
            }
            if any(actual[key] != str(record[key]) for key in actual):
                raise DeleteGateError(f"P0 fit117 payload hash mismatch: {identity}")
            graphs = read_submission(fragment)
            dataset = identity.split(":", 1)[1]
            graph = graphs.get(dataset)
            if graph is None or len(graphs) != 1:
                raise DeleteGateError(f"fragment identity mismatch: {identity}")
            cache = bh_model.load_sparse_cache(
                cache_path,
                expected_identity=p0_engine._cache_identity(meta_path),
            )
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            edges, features = edge_features(graph, cache)
            labels = true_edge_labels(graph, gt, scale, edges)
            sample = sample_rows[identity]
            expected_true = int(
                sample["actions"]["delete"]["counts"]["raw_background_actions"]
            )
            expected_false = int(
                sample["actions"]["delete"]["counts"]["positive_actions"]
            )
            if (
                int(labels.sum()) != expected_true
                or int((labels == 0).sum()) != expected_false
                or sample["baseline_fragment_sha256"] != actual["fragment_sha256"]
            ):
                raise DeleteGateError(f"P0 deletion population mismatch: {identity}")
            outputs[identity] = VideoData(
                identity=identity,
                family=_family(identity),
                edges=edges,
                features=features,
                labels=labels,
                baseline=dict(sample["baseline"]),
                fragment_sha256=actual["fragment_sha256"],
                cache_sha256=actual["cache_sha256"],
                meta_sha256=actual["meta_sha256"],
            )
            file_records.append({"identity": identity, **actual})
            print(f"[prepare {ordinal:03d}/117] {identity}", flush=True)
            del graph, cache, gt, group
    finally:
        store.close()
    return outputs, file_records


def fit_nested_models(
    population: Mapping[str, VideoData],
) -> tuple[dict[int, tuple[LinearRetention, float]], list[dict[str, Any]]]:
    partitions = nested_partitions(tuple(population))
    fitted: dict[int, tuple[LinearRetention, float]] = {}
    records = []
    for held_fold, (fit_ids, calibration_ids, held_ids) in enumerate(partitions):
        fit_features = []
        fit_labels = []
        sampled_count = 0
        for identity in fit_ids:
            video = population[identity]
            indices = deterministic_training_indices(
                identity, video.edges, video.labels
            )
            fit_features.append(video.features[indices])
            fit_labels.append(video.labels[indices])
            sampled_count += len(indices)
        model = fit_linear_retention(fit_features, fit_labels)
        calibration_scores = [
            retention_scores(model, population[identity].features)
            for identity in calibration_ids
        ]
        calibration_labels = [
            population[identity].labels for identity in calibration_ids
        ]
        threshold, diagnostics = conservative_threshold(
            calibration_scores, calibration_labels
        )
        fitted[held_fold] = (model, threshold)
        records.append(
            {
                "held_fold": held_fold,
                "calibration_fold": (held_fold + 1) % INNER_FOLDS,
                "fit_ids": list(fit_ids),
                "calibration_ids": list(calibration_ids),
                "held_ids": list(held_ids),
                "sampled_fit_edges": sampled_count,
                "parameter_count": model.parameter_count,
                "objective": model.objective,
                "optimizer_converged": model.converged,
                "optimizer_iterations": model.iterations,
                "mean": model.mean.tolist(),
                "scale": model.scale.tolist(),
                "coefficients": model.coefficients.tolist(),
                "intercept": model.intercept,
                "threshold": threshold,
                "calibration": diagnostics,
            }
        )
    return fitted, records


def delta(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float]:
    return {
        "adjusted_edge_jaccard": float(candidate["adj_edge_jaccard"])
        - float(control["adj_edge_jaccard"]),
        "raw_edge_jaccard": float(candidate["edge_jaccard"])
        - float(control["edge_jaccard"]),
        "node_recall": float(candidate["node_recall"])
        - float(control["node_recall"]),
        "score": float(candidate["score"]) - float(control["score"]),
    }


def evaluate_family(
    family: str,
    p0: Mapping[str, Any],
    population: Mapping[str, VideoData],
    fitted: Mapping[int, tuple[LinearRetention, float]],
    *,
    archive: Path,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise DeleteGateError("unsupported family")
    fragment_root = ROOT / str(p0["control"]["fragment_root"])
    store = ZipStore(archive, mode="r")
    baseline_rows = []
    candidate_rows = []
    videos = []
    invalid_action_count = 0
    try:
        identities = sorted(
            identity for identity, video in population.items() if video.family == family
        )
        for ordinal, identity in enumerate(identities, start=1):
            video = population[identity]
            held_fold = inner_fold(identity)
            model, threshold = fitted[held_fold]
            scores = retention_scores(model, video.features)
            deletion_mask = scores < threshold
            proposed_count = int(deletion_mask.sum())
            minimum_count = max(
                MIN_ACTION_COUNT,
                int(math.ceil(MIN_ACTION_FRACTION * len(video.edges))),
            )
            use_noop = proposed_count < minimum_count
            stem = identity.replace(":", "__", 1)
            fragment = fragment_root / f"{stem}.csv"
            graphs = read_submission(fragment)
            dataset = identity.split(":", 1)[1]
            graph = graphs[dataset]
            if graph_sha256(graph) != graph_sha256(graph):
                raise DeleteGateError("baseline graph hash is not deterministic")
            if use_noop:
                action = GraphAction(
                    action_id=f"delete-o0:{dataset}",
                    kind=ActionKind.NOOP,
                )
                selected_edges: tuple[tuple[int, int], ...] = ()
            else:
                selected_edges = tuple(
                    (int(source), int(target))
                    for source, target in video.edges[deletion_mask]
                )
                action = GraphAction(
                    action_id=f"delete-o0:{dataset}",
                    kind=ActionKind.DELETE_EDGES,
                    delete_edges=selected_edges,
                )
            try:
                individual = compile_action(graph, action)
                final_batch = compile_batch(graph, [action])
            except Exception as exc:
                invalid_action_count += 1
                raise DeleteGateError(f"N1 rejected {identity}") from exc
            if (
                individual.after_graph_sha256 != final_batch.after_graph_sha256
                or graph_sha256(individual.graph) != graph_sha256(final_batch.graph)
            ):
                invalid_action_count += 1
                raise DeleteGateError(f"N1 individual/final-batch drift: {identity}")
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            candidate_row = p0_engine._metric(final_batch.graph, gt, scale)
            baseline_rows.append(dict(video.baseline))
            candidate_rows.append(candidate_row)
            deleted_true = int(
                video.labels[deletion_mask].sum() if not use_noop else 0
            )
            deleted_false = int(
                ((video.labels[deletion_mask] == 0).sum()) if not use_noop else 0
            )
            videos.append(
                {
                    "identity": identity,
                    "held_fold": held_fold,
                    "threshold": threshold,
                    "baseline_edge_count": int(len(video.edges)),
                    "proposed_deletions": proposed_count,
                    "minimum_action_count": minimum_count,
                    "no_op_selected": use_noop,
                    "compiled_deletions": len(selected_edges),
                    "deleted_true_edges_diagnostic": deleted_true,
                    "deleted_false_edges_diagnostic": deleted_false,
                    "action_sha256": individual.action_sha256,
                    "before_graph_sha256": individual.before_graph_sha256,
                    "after_graph_sha256": individual.after_graph_sha256,
                    "batch_sha256": final_batch.batch_sha256,
                    "individual_final_batch_equal": True,
                    "baseline_metrics": dict(video.baseline),
                    "candidate_metrics": candidate_row,
                }
            )
            print(
                f"[score {family} {ordinal:03d}/{len(identities):03d}] {identity}",
                flush=True,
            )
            del graph, gt, group
    finally:
        store.close()
    control = p0_engine._summarise(baseline_rows)
    candidate = p0_engine._summarise(candidate_rows)
    change = delta(candidate, control)
    gates = {
        "adjusted_ge_0_005": change["adjusted_edge_jaccard"] >= 0.005,
        "raw_ge_0": change["raw_edge_jaccard"] >= 0.0,
        "node_ge_0": change["node_recall"] >= 0.0,
        "invalid_action_count_zero": invalid_action_count == 0,
    }
    return {
        "family": family,
        "video_count": len(videos),
        "control": control,
        "candidate": candidate,
        "delta": change,
        "gates": gates,
        "go": all(gates.values()),
        "invalid_action_count": invalid_action_count,
        "videos": videos,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    p0, folds, provenance, input_records = authenticate_inputs()
    archive = ROOT / p0["inputs"]["archive"]["path"]
    population, payload_records = load_population(p0, archive=archive)
    fitted, fold_records = fit_nested_models(population)

    result_44b6 = evaluate_family(
        "44b6", p0, population, fitted, archive=archive
    )
    family_results: dict[str, Any] = {"44b6": result_44b6}
    if result_44b6["go"]:
        family_results["6bba"] = evaluate_family(
            "6bba", p0, population, fitted, archive=archive
        )
        terminal = "GO" if family_results["6bba"]["go"] else "PARK_C_6BBA_KILL"
    else:
        terminal = "PARK_C_44B6_KILL"

    return {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "node": "DELETE-O0 / AE-P1",
        "terminal_verdict": terminal,
        "scope": {
            "population": "P0 fit117 fold-0 fit videos only",
            "family_counts": {
                family: sum(video.family == family for video in population.values())
                for family in FAMILIES
            },
            "audit_access": False,
            "test_access": False,
            "cv57_access": False,
            "public_access": False,
            "gpu_action": False,
            "kaggle_action": False,
            "runpod_action": False,
            "submission_action": False,
        },
        "contract": {
            "feature_names": list(FEATURE_NAMES),
            "parameter_count": len(FEATURE_NAMES) + 1,
            "fold_count": INNER_FOLDS,
            "fold_rule": "uint64_be(sha256(identity)[0:8]) modulo 5",
            "calibration_fold": "(held_fold + 1) modulo 5",
            "seed": SEED,
            "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
            "minimum_negatives_per_video": MIN_NEGATIVES_PER_VIDEO,
            "l2_strength": L2_STRENGTH,
            "maximum_optimizer_iterations": MAX_OPTIMIZER_ITERATIONS,
            "true_edge_lower_quantile": TRUE_EDGE_LOWER_QUANTILE,
            "minimum_calibration_true_edges": MIN_CALIBRATION_TRUE_EDGES,
            "minimum_action_count": MIN_ACTION_COUNT,
            "minimum_action_fraction": MIN_ACTION_FRACTION,
            "action": "one explicit N1 delete_edges bundle per video; N1 noop when admission fails",
            "graph_gate": "44b6 adj>=+0.005 and raw,node>=0; only then identical 6bba gate",
        },
        "input_authentication": input_records,
        "transitive_fit117_files": payload_records,
        "fold_manifest_assertions": {
            "artifact": folds.get("artifact"),
            "fit117_equals_fold0_fit": True,
            "audit_video_count": len(folds.get("audit", {}).get("video_ids", ())),
            "audit_overlap": 0,
            "provenance_artifact": provenance.get("artifact"),
        },
        "fit_folds": fold_records,
        "family_results": family_results,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "external_action": False,
            "gpu_action": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--output",
        type=Path,
        default=BRANCH_ROOT / "graph_result.json",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.output.resolve().is_relative_to(BRANCH_ROOT):
        raise SystemExit("output must remain inside SA-DELETE subtree")
    report = run(args)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "terminal_verdict": report["terminal_verdict"],
                "families": {
                    family: payload["delta"]
                    for family, payload in report["family_results"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
