#!/usr/bin/env python3
"""Compute the fit-only CG0 candidate-generation oracle ceiling ladder.

The diagnostic deliberately reports two different quantities:

* the legacy target-limited projection used by ACH-HEADROOM / GA-MGSHELL, so
  the published 0.895678 and 0.897746 controls remain reproducible; and
* a literal perfect-selection oracle over each deterministic proposal policy,
  evaluated with the frozen official edge/node scorer on the exact fit117
  detection caches, with divisions disabled.

No model is fitted.  Every path is rejected for holdout/CV57 tokens before it
is checked or opened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
from scipy.spatial import cKDTree
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from tracking_cellmot.metrics import EvaluationResult, per_sample_metrics  # noqa: E402

from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import official_summary  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.candgen-ceiling-diagnostic.v1"
FAMILIES = ("44b6", "6bba")
FORBIDDEN_REFERENCE_TOKENS = ("cv57", "holdout")
SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
GOAL = 0.92

DEFAULT_HEADROOM = ROOT / "reports/g02/screening/achievable_headroom_v1.json"
DEFAULT_REACHABILITY = ROOT / "reports/g02/extended_radius_reachability_v1.json"
DEFAULT_SHELL = ROOT / "reports/g02/extended_radius_shell_prepare_v1/prepare.json"
DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_CACHE = ROOT / "reports/g02/fitcache_117/bh9out/cache"
DEFAULT_FRAGMENTS = ROOT / "reports/g02/fitcache_117/bh9out/fragments"
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "reports/g02/screening/candgen/cg0_ceiling_diagnostic_v1.md"
)

POLICIES: tuple[tuple[str, float, int, bool], ...] = (
    ("current_r10_k08", 10.0, 8, False),
    ("current_r10_k16", 10.0, 16, False),
    ("shell_r12_k08", 12.0, 8, False),
    ("current_r10_k08_plus_gap", 10.0, 8, True),
    ("shell_r12_k08_plus_gap", 12.0, 8, True),
    ("wide_r20_k32_oracle_filter", 20.0, 32, False),
    ("wide_r20_k32_oracle_filter_plus_gap", 20.0, 32, True),
)

# Published aggregate constants used only for legacy-control reproduction.
BANKED_SCORE = 0.8815220699597315
BANKED_EDGE_TP = 34320
BANKED_EDGE_FP = 2459
BANKED_EDGE_FN = 1948
BANKED_NODE_ADJUSTMENT = 0.9947175175795606
TARGET_IDEAL_SWAPS = 378


class CandgenCeilingError(RuntimeError):
    """A fit-only identity, scorer invariant, or accounting check failed."""


def assert_safe_fit_only_path(path: Path) -> None:
    """Reject a forbidden artifact identity before existence checks or reads."""

    lowered = str(path).lower()
    token = next(
        (candidate for candidate in FORBIDDEN_REFERENCE_TOKENS if candidate in lowered),
        None,
    )
    if token is not None:
        raise CandgenCeilingError(
            f"forbidden {token!r} artifact reference: {path}"
        )


def _guard_embedded_references(value: Any, *, source: Path) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        token = next(
            (
                candidate
                for candidate in FORBIDDEN_REFERENCE_TOKENS
                if candidate in lowered
            ),
            None,
        )
        if token is not None:
            raise CandgenCeilingError(
                f"{source} embeds forbidden {token!r} reference: {value}"
            )
    elif isinstance(value, Mapping):
        for child in value.values():
            _guard_embedded_references(child, source=source)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        for child in value:
            _guard_embedded_references(child, source=source)


def _load_json(path: Path) -> Mapping[str, Any]:
    assert_safe_fit_only_path(path)
    if not path.is_file():
        raise CandgenCeilingError(f"missing required input: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandgenCeilingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CandgenCeilingError(f"expected JSON object: {path}")
    _guard_embedded_references(value, source=path)
    return value


def _sha256(path: Path) -> str:
    assert_safe_fit_only_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family(identity: str) -> str:
    family = identity.split(":", 1)[1].split("_", 1)[0]
    if family not in FAMILIES:
        raise CandgenCeilingError(f"unexpected family: {identity}")
    return family


def _assert_scope(
    report: Mapping[str, Any], *, source: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scope = report.get("scope")
    if not isinstance(scope, Mapping):
        raise CandgenCeilingError(f"missing scope: {source}")
    train_ids = tuple(str(value) for value in scope.get("train_video_ids", ()))
    validation_ids = tuple(
        str(value) for value in scope.get("validation_video_ids", ())
    )
    if (
        scope.get("fit_only") is not True
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
        or len(train_ids) != 105
        or len(validation_ids) != 12
        or len(set((*train_ids, *validation_ids))) != 117
        or set(train_ids) & set(validation_ids)
        or any(
            not identity.startswith("train:")
            for identity in (*train_ids, *validation_ids)
        )
    ):
        raise CandgenCeilingError(
            f"artifact is not the exact fit-only 105/12 scope: {source}"
        )
    return train_ids, validation_ids


def legacy_score_after_perfect_swaps(swaps: int) -> float:
    if not 0 <= swaps <= min(BANKED_EDGE_FP, BANKED_EDGE_FN):
        raise CandgenCeilingError("legacy perfect-swap count is invalid")
    return BANKED_NODE_ADJUSTMENT * (
        (BANKED_EDGE_TP + swaps)
        / (
            BANKED_EDGE_TP
            + swaps
            + BANKED_EDGE_FP
            - swaps
            + BANKED_EDGE_FN
            - swaps
        )
    )


def background_budget(
    *,
    edge_tp: int,
    edge_fn: int,
    node_adjustment: float,
    raw_background: int,
    goal: float = GOAL,
) -> dict[str, int | float | bool]:
    """Maximum extra FP edges that preserve the requested adjusted score."""

    if (
        edge_tp < 0
        or edge_fn < 0
        or raw_background < 0
        or not math.isfinite(node_adjustment)
        or not 0.0 < node_adjustment
        or not 0.0 < goal < node_adjustment
    ):
        raise CandgenCeilingError("invalid background-budget inputs")
    required_edge_jaccard = goal / node_adjustment
    maximum_fp = math.floor(
        edge_tp / required_edge_jaccard - edge_tp - edge_fn + 1.0e-12
    )
    maximum_fp = max(0, maximum_fp)
    acceptance = maximum_fp / raw_background if raw_background else 0.0
    return {
        "goal": goal,
        "maximum_background_false_positive_edges": maximum_fp,
        "raw_background_proposals": raw_background,
        "maximum_background_acceptance_fraction": acceptance,
        "minimum_background_rejection_fraction": 1.0 - acceptance,
        "minimum_selected_edge_precision_at_budget": (
            edge_tp / (edge_tp + maximum_fp)
            if edge_tp + maximum_fp
            else 0.0
        ),
        "positive_budget": maximum_fp > 0,
    }


def _aggregate_artifact_by_family(
    report: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    result = {family: Counter() for family in FAMILIES}
    rows = report.get("videos")
    if not isinstance(rows, Sequence):
        raise CandgenCeilingError("reachability video rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise CandgenCeilingError("malformed reachability video row")
        family = _family(str(row["identity"]))
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise CandgenCeilingError("reachability metrics are missing")
        result[family].update(
            {str(k): int(v) for k, v in metrics["counts"].items()}
        )
        result[family].update(
            {
                f"reachable_{k}": int(v)
                for k, v in metrics["policy_reachable"].items()
            }
        )
        result[family].update(
            {
                f"pool_{k}": int(v)
                for k, v in metrics["candidate_pool"].items()
            }
        )
    return {family: dict(values) for family, values in result.items()}


def _shell_by_family(report: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result = {family: Counter() for family in FAMILIES}
    rows = report.get("videos")
    if not isinstance(rows, Sequence):
        raise CandgenCeilingError("shell video rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise CandgenCeilingError("malformed shell video row")
        family = _family(str(row["identity"]))
        result[family].update(
            {str(k): int(v) for k, v in row["counts"].items()}
        )
    return {family: dict(values) for family, values in result.items()}


def _candidate_rank(
    source: np.ndarray,
    target_id: int,
    target_ids: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, int]:
    locations = np.flatnonzero(target_ids == target_id)
    if len(locations) != 1:
        raise CandgenCeilingError("matched target is absent or duplicated")
    distances = np.linalg.norm(target_points - source[None, :], axis=1)
    order = sorted(
        range(len(target_ids)),
        key=lambda index: (
            float(distances[index]),
            int(target_ids[index]),
            index,
        ),
    )
    location = int(locations[0])
    return float(distances[location]), order.index(location) + 1


def _metric_row(
    *,
    identity: str,
    edge_tp: int,
    gt_edges: int,
    matched_nodes: int,
    gt_nodes: int,
    predicted_nodes: int,
    estimated_nodes: float,
) -> dict[str, Any]:
    if not 0 <= edge_tp <= gt_edges or not 0 <= matched_nodes <= gt_nodes:
        raise CandgenCeilingError(f"invalid oracle counts for {identity}")
    metrics = per_sample_metrics(
        EvaluationResult(
            edge_tp=edge_tp,
            edge_fp=0,
            edge_fn=gt_edges - edge_tp,
            division_tp=0,
            division_fp=0,
            division_fn=0,
            num_pred_nodes=predicted_nodes,
        ),
        estimated_nodes,
        matched_nodes / gt_nodes if gt_nodes else float("nan"),
    )
    return {
        **metrics,
        "division_jaccard": float("nan"),
        "estimated_number_of_nodes": estimated_nodes,
        "identity": identity,
        "family": _family(identity),
    }


def _scan_fit117(
    *,
    ids: Sequence[str],
    zip_path: Path,
    cache_dir: Path,
    fragment_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    for path in (zip_path, cache_dir, fragment_dir):
        assert_safe_fit_only_path(path)
    if not zip_path.is_file() or not cache_dir.is_dir() or not fragment_dir.is_dir():
        raise CandgenCeilingError("fit117 ZIP/cache/fragment input is missing")

    metric_rows = {name: [] for name, *_ in POLICIES}
    gap_counts = {family: Counter() for family in (*FAMILIES, "overall")}
    selected_counts = {family: Counter() for family in (*FAMILIES, "overall")}
    store = ZipStore(zip_path, mode="r")
    try:
        for identity in ids:
            family = _family(identity)
            dataset = identity.split(":", 1)[1]
            slug = identity.replace(":", "__", 1)
            cache_path = cache_dir / f"{slug}.bhc"
            meta_path = cache_dir / f"{slug}.json"
            fragment_path = fragment_dir / f"{slug}.csv"
            for path in (cache_path, meta_path, fragment_path):
                assert_safe_fit_only_path(path)
                if not path.is_file():
                    raise CandgenCeilingError(f"missing fit117 input: {path}")

            cache = bh_model.load_sparse_cache(
                cache_path,
                expected_identity=_cache_identity(meta_path),
            )
            gt = read_geff_arrays(store, "train", dataset)
            detection_to_gt, gt_edges_raw = match_detections_to_gt(
                cache.detections, gt
            )
            gt_to_detection = {
                int(gt_id): int(detection_id)
                for detection_id, gt_id in detection_to_gt.items()
                if gt_id is not None
            }
            gt_edges = tuple(
                (int(source), int(target))
                for source, target in gt_edges_raw
            )
            truth = set(gt_edges)
            successors: dict[int, list[int]] = defaultdict(list)
            predecessors: dict[int, list[int]] = defaultdict(list)
            for source, target in gt_edges:
                successors[source].append(target)
                predecessors[target].append(source)

            detections: dict[int, tuple[int, np.ndarray]] = {}
            frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            predicted_nodes = 0
            for frame in cache.detections:
                node_ids = np.asarray(frame.node_ids, dtype=np.int64)
                points = (
                    np.asarray(frame.coords_zyx, dtype=np.float64)
                    * SCALE_ZYX_UM
                )
                frames[int(frame.frame_index)] = (node_ids, points)
                predicted_nodes += len(node_ids)
                for node_id, point in zip(node_ids, points, strict=True):
                    detections[int(node_id)] = (
                        int(frame.frame_index),
                        point,
                    )

            adjacent_tp = {name: 0 for name, *_ in POLICIES}
            for gt_source, gt_target in gt_edges:
                if (
                    len(successors[gt_source]) != 1
                    or len(predecessors[gt_target]) != 1
                ):
                    continue
                source_id = gt_to_detection.get(gt_source)
                target_id = gt_to_detection.get(gt_target)
                if source_id is None or target_id is None:
                    continue
                source_t, source_point = detections[source_id]
                target_t, _ = detections[target_id]
                if target_t != source_t + 1:
                    continue
                target_ids, target_points = frames[target_t]
                distance, rank = _candidate_rank(
                    source_point, target_id, target_ids, target_points
                )
                for name, radius, cap, _use_gap in POLICIES:
                    adjacent_tp[name] += int(
                        distance <= radius + 1.0e-9 and rank <= cap
                    )

            gap_pool = 0
            for frame_index in sorted(frames):
                if frame_index + 2 not in frames:
                    continue
                _, source_points = frames[frame_index]
                target_ids, target_points = frames[frame_index + 2]
                if not len(source_points) or not len(target_points):
                    continue
                tree = cKDTree(target_points)
                for source_point in source_points:
                    indices = tree.query_ball_point(source_point, 20.0)
                    indices = sorted(
                        indices,
                        key=lambda index: (
                            float(
                                np.linalg.norm(
                                    target_points[index] - source_point
                                )
                            ),
                            int(target_ids[index]),
                            int(index),
                        ),
                    )[:8]
                    gap_pool += len(indices)

            gt_node_ids = np.asarray(gt["nodes"]["node_id"], dtype=np.int64)
            valid_gap_events = 0
            one_missed_frame_paths = 0
            for gt_source, middle_values in successors.items():
                if len(middle_values) != 1:
                    continue
                middle = middle_values[0]
                if (
                    len(predecessors[middle]) != 1
                    or len(successors.get(middle, ())) != 1
                ):
                    continue
                gt_target = successors[middle][0]
                if len(predecessors[gt_target]) != 1:
                    continue
                source_id = gt_to_detection.get(gt_source)
                middle_id = gt_to_detection.get(middle)
                target_id = gt_to_detection.get(gt_target)
                if (
                    source_id is None
                    or target_id is None
                    or middle_id is not None
                ):
                    continue
                source_t, source_point = detections[source_id]
                target_t, target_point = detections[target_id]
                if target_t != source_t + 2:
                    continue
                one_missed_frame_paths += 1
                target_ids, target_points = frames[target_t]
                distance, rank = _candidate_rank(
                    source_point, target_id, target_ids, target_points
                )
                if distance > 20.0 + 1.0e-9 or rank > 8:
                    continue
                middle_locations = np.flatnonzero(gt_node_ids == middle)
                if len(middle_locations) != 1:
                    raise CandgenCeilingError("GT middle node is not unique")
                index = int(middle_locations[0])
                gt_middle = (
                    np.asarray(
                        (
                            gt["nodes"]["z"][index],
                            gt["nodes"]["y"][index],
                            gt["nodes"]["x"][index],
                        ),
                        dtype=np.float64,
                    )
                    * SCALE_ZYX_UM
                )
                midpoint = (source_point + target_point) / 2.0
                if np.linalg.norm(midpoint - gt_middle) <= 7.0 + 1.0e-9:
                    valid_gap_events += 1

            graph = read_submission(fragment_path).get(dataset)
            if graph is None:
                raise CandgenCeilingError(
                    f"fragment dataset identity mismatch: {fragment_path}"
                )
            selected_pairs = {
                (int(edge.source_id), int(edge.target_id))
                for edge in graph.edges
            }
            selected_direct_tp = sum(
                1
                for source, target in selected_pairs
                if detection_to_gt.get(source) is not None
                and detection_to_gt.get(target) is not None
                and (
                    int(detection_to_gt[source]),
                    int(detection_to_gt[target]),
                )
                in truth
            )

            gap_row = {
                "gap_pool_pairs": gap_pool,
                "one_missed_frame_paths": one_missed_frame_paths,
                "valid_gap_events": valid_gap_events,
                "recoverable_correct_edges": 2 * valid_gap_events,
            }
            selected_row = {
                "selected_direct_correct_edges": selected_direct_tp,
            }
            for bucket in (family, "overall"):
                gap_counts[bucket].update(gap_row)
                selected_counts[bucket].update(selected_row)

            gt_node_count = len(gt_node_ids)
            matched_nodes = len(gt_to_detection)
            estimated_nodes = float(gt.get("estimated_number_of_nodes"))
            for name, _radius, _cap, use_gap in POLICIES:
                gap_events = valid_gap_events if use_gap else 0
                metric_rows[name].append(
                    _metric_row(
                        identity=identity,
                        edge_tp=adjacent_tp[name] + 2 * gap_events,
                        gt_edges=len(gt_edges),
                        matched_nodes=matched_nodes + gap_events,
                        gt_nodes=gt_node_count,
                        predicted_nodes=predicted_nodes + gap_events,
                        estimated_nodes=estimated_nodes,
                    )
                )
    finally:
        store.close()
    return (
        metric_rows,
        {key: dict(value) for key, value in gap_counts.items()},
        {key: dict(value) for key, value in selected_counts.items()},
    )


def _policy_key(radius: float, cap: int) -> str:
    return f"r{int(radius):02d}_k{cap:02d}"


def _policy_report(
    *,
    metric_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reachability: Mapping[str, Mapping[str, int]],
    gap_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    result = []
    for name, radius, cap, use_gap in POLICIES:
        key = _policy_key(radius, cap)
        family_rows: dict[str, Any] = {}
        for family in ("overall", *FAMILIES):
            rows = (
                list(metric_rows[name])
                if family == "overall"
                else [
                    row
                    for row in metric_rows[name]
                    if row["family"] == family
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                summary = official_summary(list(rows))
            counts = {
                "edge_tp": sum(int(row["edge_tp"]) for row in rows),
                "edge_fp": sum(int(row["edge_fp"]) for row in rows),
                "edge_fn": sum(int(row["edge_fn"]) for row in rows),
            }
            if family == "overall":
                adjacent_pool = sum(
                    int(reachability[value][f"pool_{key}"])
                    for value in FAMILIES
                )
                adjacent_correct = sum(
                    int(reachability[value][f"reachable_{key}"])
                    for value in FAMILIES
                )
            else:
                adjacent_pool = int(reachability[family][f"pool_{key}"])
                adjacent_correct = int(
                    reachability[family][f"reachable_{key}"]
                )
            gap_pool = (
                int(gap_counts[family]["gap_pool_pairs"]) if use_gap else 0
            )
            gap_correct = (
                int(gap_counts[family]["valid_gap_events"]) if use_gap else 0
            )
            pool_size = adjacent_pool + gap_pool
            correct_proposals = adjacent_correct + gap_correct
            raw_background = pool_size - correct_proposals
            node_adjustment = float(summary["adj_edge_jaccard"]) / float(
                summary["edge_jaccard"]
            )
            family_rows[family] = {
                "frozen_scorer": {
                    "score": float(summary["score"]),
                    "adjusted_edge_jaccard": float(
                        summary["adj_edge_jaccard"]
                    ),
                    "edge_jaccard": float(summary["edge_jaccard"]),
                    "node_recall": float(summary["node_recall"]),
                    "node_adjustment": node_adjustment,
                    **counts,
                },
                "burden": {
                    "pool_size": pool_size,
                    "correct_proposals": correct_proposals,
                    "raw_background_proposals": raw_background,
                    "correct_to_background_ratio": (
                        correct_proposals / raw_background
                        if raw_background
                        else None
                    ),
                    "raw_precision": (
                        correct_proposals / pool_size if pool_size else None
                    ),
                },
                "background_budget_at_goal": background_budget(
                    edge_tp=counts["edge_tp"],
                    edge_fn=counts["edge_fn"],
                    node_adjustment=node_adjustment,
                    raw_background=raw_background,
                ),
            }
        result.append(
            {
                "policy": name,
                "radius_um": radius,
                "neighbor_cap": cap,
                "bridge_one_missed_frame": use_gap,
                "oracle_background_filter": "oracle_filter" in name,
                "families": family_rows,
                "go": float(family_rows["overall"]["frozen_scorer"]["score"])
                >= GOAL,
            }
        )
    return result


def _bucket_report(
    *,
    reachability: Mapping[str, Mapping[str, int]],
    shell: Mapping[str, Mapping[str, int]],
    gap_counts: Mapping[str, Mapping[str, int]],
    selected_counts: Mapping[str, Mapping[str, int]],
    policies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    score_by_policy = {
        str(row["policy"]): row["families"] for row in policies
    }
    buckets = []
    for family in ("overall", *FAMILIES):
        reach = (
            {
                key: sum(int(reachability[value][key]) for value in FAMILIES)
                for key in reachability[FAMILIES[0]]
            }
            if family == "overall"
            else reachability[family]
        )
        shell_positive = (
            sum(int(shell[value]["positive_pairs"]) for value in FAMILIES)
            if family == "overall"
            else int(shell[family]["positive_pairs"])
        )
        selected_direct = int(
            selected_counts[family]["selected_direct_correct_edges"]
        )
        current_reachable = int(reach["reachable_r10_k08"])
        default_excluded = int(reach["default_excluded_edges"])
        subcap = int(reach["excluded_rank_above_8_within_10um"])
        beyond12 = default_excluded - shell_positive - subcap
        if min(
            current_reachable - selected_direct,
            shell_positive,
            beyond12,
            subcap,
        ) < 0:
            raise CandgenCeilingError("exclusion buckets do not reconcile")
        rows = (
            (
                "present-but-unrankable",
                current_reachable - selected_direct,
                "correct adjacent edges in r10/k8 but absent from the banked selected graph",
                "current_r10_k08",
            ),
            (
                "out-of-radius-10-to-12",
                shell_positive,
                "correct adjacent edges admitted by r12/k8 but not r10/k8",
                "shell_r12_k08",
            ),
            (
                "out-of-radius-beyond-12",
                beyond12,
                "default-excluded correct adjacent edges not recovered by the r10--12 shell",
                "wide_r20_k32_oracle_filter",
            ),
            (
                "sub-cap",
                subcap,
                "correct adjacent edges within 10 um with deterministic rank above 8",
                "current_r10_k16",
            ),
            (
                "gap-closing-multi-frame",
                int(gap_counts[family]["valid_gap_events"]),
                "one-missed-frame bridge events; each event recovers two correct edges",
                "current_r10_k08_plus_gap",
            ),
        )
        for name, count, definition, policy in rows:
            bucket = {
                "family": family,
                "bucket": name,
                "count": count,
                "definition": definition,
                "oracle_policy_if_perfectly_admitted": policy,
                "recoverable_oracle_ceiling": float(
                    score_by_policy[policy][family]["frozen_scorer"]["score"]
                ),
            }
            if name == "gap-closing-multi-frame":
                bucket["recoverable_correct_edges"] = int(
                    gap_counts[family]["recoverable_correct_edges"]
                )
            buckets.append(bucket)
    return buckets


def _legacy_controls(
    headroom: Mapping[str, Any],
    reachability: Mapping[str, Any],
) -> dict[str, Any]:
    recall = headroom.get("recall_side_ceiling")
    if not isinstance(recall, Mapping):
        raise CandgenCeilingError("headroom recall ceiling is missing")
    current_swaps = int(recall["target_swaps_in_current_pool"])
    excluded_swaps = int(recall["target_swaps_excluded_from_current_pool"])
    if current_swaps != 290 or excluded_swaps != 88:
        raise CandgenCeilingError("legacy target allocation drifted")
    all_counts = reachability["all_fit"]["counts"]
    excluded_edges = int(all_counts["default_excluded_edges"])
    if excluded_edges != 719:
        raise CandgenCeilingError("legacy excluded-edge count drifted")
    motion_retained = 344
    projected_motion = int(
        math.floor(excluded_swaps * motion_retained / excluded_edges + 0.5)
    )
    full_shell = int(
        next(
            row["additional_reachable_vs_r10_k08"]
            for row in reachability["all_fit"]["policies"]
            if row["policy"] == "r12_k08"
        )
    )
    projected_full_shell = int(
        math.floor(excluded_swaps * full_shell / excluded_edges + 0.5)
    )
    current_score = legacy_score_after_perfect_swaps(current_swaps)
    motion_score = legacy_score_after_perfect_swaps(
        current_swaps + projected_motion
    )
    if (
        abs(current_score - 0.8956779478999035) > 1.0e-12
        or abs(motion_score - 0.8977458371966905) > 1.0e-12
    ):
        raise CandgenCeilingError("legacy control score did not reproduce")
    return {
        "role": (
            "reproduction controls only; these allocate the 378 swaps needed "
            "for the old 0.90 target and are not literal full-pool ceilings"
        ),
        "current_pool": {
            "perfect_swaps": current_swaps,
            "score": current_score,
            "expected_rounded": 0.895678,
            "reproduced": True,
        },
        "published_0p897746_control": {
            "actual_policy": "motion-gated subset of the r10--12 shell",
            "motion_retained_correct_edges": motion_retained,
            "projected_recovered_swaps": projected_motion,
            "perfect_swaps": current_swaps + projected_motion,
            "score": motion_score,
            "expected_rounded": 0.897746,
            "reproduced": True,
        },
        "full_r12_k08_projection_for_disambiguation": {
            "shell_correct_edges": full_shell,
            "projected_recovered_swaps": projected_full_shell,
            "perfect_swaps": current_swaps + projected_full_shell,
            "score": legacy_score_after_perfect_swaps(
                current_swaps + projected_full_shell
            ),
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    policy_rows = report["policy_ladder"]
    buckets = report["exclusion_buckets"]
    maxima = report["maxima"]
    legacy = report["legacy_projection_controls"]
    lines = [
        "# CG0 candidate-generation ceiling diagnostic",
        "",
        f"**{report['verdict']}**",
        "",
        "## Root-gate result",
        "",
        (
            "The literal fit117 perfect-selection oracle already exceeds 0.92 "
            "for the current r10/cap-8 proposal policy. The prior 0.895678 value "
            "is reproduced, but it is a 378-swap target allocation rather than "
            "a full-pool oracle ceiling. Candidate generation is therefore not "
            "the binding mathematical ceiling; selection/abstention remains the "
            "realization problem."
        ),
        "",
        "## Legacy controls",
        "",
        "| control | score | interpretation |",
        "|---|---:|---|",
        (
            f"| current pool | {legacy['current_pool']['score']:.9f} | "
            "reproduces 0.895678; target-limited projection |"
        ),
        (
            f"| published shell control | "
            f"{legacy['published_0p897746_control']['score']:.9f} | "
            "reproduces 0.897746; actually the 344/456 motion-retained subset |"
        ),
        (
            f"| full r12/k8 under the same legacy projection | "
            f"{legacy['full_r12_k08_projection_for_disambiguation']['score']:.9f} | "
            "456/719 shell recall |"
        ),
        "",
        "## Exclusion buckets",
        "",
        "| family | cause | count | correct edges recovered | oracle ceiling |",
        "|---|---|---:|---:|---:|",
    ]
    for row in buckets:
        recovered = row.get("recoverable_correct_edges", row["count"])
        lines.append(
            f"| {row['family']} | {row['bucket']} | {row['count']:,} | "
            f"{int(recovered):,} | {row['recoverable_oracle_ceiling']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Counts are mutually exclusive by primary cause for adjacent edges: "
            "r10/cap-8 present-but-unselected; 10–12 µm; beyond 12 µm; within-10 "
            "cap exclusion. Gap rows are separate one-missed-frame bridge events "
            "and recover two correct edges per valid event.",
            "",
            "## Deterministic policy ladder",
            "",
            (
                "| policy | exact ceiling | pool | correct | background | raw "
                "correct/background | max background FP at 0.92 | min rejection |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policy_rows:
        overall = row["families"]["overall"]
        burden = overall["burden"]
        budget = overall["background_budget_at_goal"]
        lines.append(
            f"| {row['policy']} | "
            f"{overall['frozen_scorer']['score']:.6f} | "
            f"{burden['pool_size']:,} | "
            f"{burden['correct_proposals']:,} | "
            f"{burden['raw_background_proposals']:,} | "
            f"{burden['correct_to_background_ratio']:.8f} | "
            f"{budget['maximum_background_false_positive_edges']:,} | "
            f"{budget['minimum_background_rejection_fraction']:.6%} |"
        )
    lines.extend(
        [
            "",
            "The wider-radius `oracle_filter` rows are upper bounds: their raw "
            "background is reported before the perfect abstention oracle, so a "
            "real proposer must meet the stated false-positive budget.",
            "",
            "## Per-family maxima",
            "",
            "| family | maximum ceiling | policy | GO at 0.92 |",
            "|---|---:|---|---|",
        ]
    )
    for family in ("overall", *FAMILIES):
        row = maxima[family]
        lines.append(
            f"| {family} | {row['score']:.6f} | {row['policy']} | "
            f"{'yes' if row['score'] >= GOAL else 'no'} |"
        )
    recommended = report["gate"]["minimum_burden_go_policy"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{report['verdict']}**",
            "",
            (
                f"The least-burden policy already over 0.92 is "
                f"`{recommended['policy']}` at {recommended['score']:.6f}. "
                f"It may admit at most "
                f"{recommended['maximum_background_false_positive_edges']:,} "
                "background FP edges on fit117 while retaining 0.92, equivalent "
                f"to rejecting at least "
                f"{recommended['minimum_background_rejection_fraction']:.6%} "
                "of its raw background proposals. This is plausible as an "
                "absolute budget because it is the existing pool and is comparable "
                "to the banked graph's error scale; no wider pool is required to "
                "clear the mathematical gate."
            ),
            "",
            "CG1 should target the present-but-unrankable/abstention realization "
            "inside the current pool first. Wider-radius and gap unions have higher "
            "maxima but substantially harsher background-rejection requirements.",
            "",
        ]
    )
    return "\n".join(lines)


def _immutable_write(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise CandgenCeilingError(
                f"refusing to overwrite different report: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.headroom,
        args.reachability,
        args.shell,
        args.zip,
        args.cache,
        args.fragments,
        args.output,
        args.markdown,
    ):
        assert_safe_fit_only_path(path)
    started = time.perf_counter()
    headroom = _load_json(args.headroom)
    reachability_report = _load_json(args.reachability)
    shell_report = _load_json(args.shell)
    train_ids, validation_ids = _assert_scope(
        reachability_report, source=args.reachability
    )
    shell_scope = _assert_scope(shell_report, source=args.shell)
    if (
        reachability_report.get("schema")
        != "celltrack-alpha.extended-radius-reachability-audit.v1"
        or shell_report.get("schema")
        != "celltrack-alpha.extended-radius-shell-prepare.v1"
        or shell_scope != (train_ids, validation_ids)
    ):
        raise CandgenCeilingError("fit-only input identities do not reconcile")

    reachability = _aggregate_artifact_by_family(reachability_report)
    shell = _shell_by_family(shell_report)
    metric_rows, gap_counts, selected_counts = _scan_fit117(
        ids=(*train_ids, *validation_ids),
        zip_path=args.zip,
        cache_dir=args.cache,
        fragment_dir=args.fragments,
    )
    policy_ladder = _policy_report(
        metric_rows=metric_rows,
        reachability=reachability,
        gap_counts=gap_counts,
    )
    buckets = _bucket_report(
        reachability=reachability,
        shell=shell,
        gap_counts=gap_counts,
        selected_counts=selected_counts,
        policies=policy_ladder,
    )
    legacy = _legacy_controls(headroom, reachability_report)
    maxima = {}
    for family in ("overall", *FAMILIES):
        selected = max(
            policy_ladder,
            key=lambda row: float(
                row["families"][family]["frozen_scorer"]["score"]
            ),
        )
        maxima[family] = {
            "policy": selected["policy"],
            "score": float(
                selected["families"][family]["frozen_scorer"]["score"]
            ),
            "background_budget_at_goal": selected["families"][family][
                "background_budget_at_goal"
            ],
        }
    go_policies = [
        row
        for row in policy_ladder
        if float(row["families"]["overall"]["frozen_scorer"]["score"]) >= GOAL
        and bool(
            row["families"]["overall"]["background_budget_at_goal"][
                "positive_budget"
            ]
        )
    ]
    minimum_burden = min(
        go_policies,
        key=lambda row: int(
            row["families"]["overall"]["burden"]["raw_background_proposals"]
        ),
    )
    minimum_overall = minimum_burden["families"]["overall"]
    verdict = "GO" if go_policies else "ESCALATE"
    report = {
        "schema": SCHEMA,
        "status": "completed_fit_only_cpu_diagnostic",
        "verdict": verdict,
        "objective": {
            "equation": (
                "ceiling(P) = frozen official fit117 adjusted edge Jaccard "
                "after perfect selection of GT-supported proposals from P"
            ),
            "goal": GOAL,
            "division_enabled": False,
            "node_front_end": "exact fit117 sparse-inference detections",
            "gap_contract": (
                "one-to-one two-edge GT path; middle detection absent; endpoints "
                "detected two frames apart; endpoint r20/k8; midpoint within 7 um"
            ),
        },
        "scope": {
            "fit_only": True,
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "legacy_projection_controls": legacy,
        "exclusion_buckets": buckets,
        "gap_audit": gap_counts,
        "policy_ladder": policy_ladder,
        "maxima": maxima,
        "gate": {
            "goal": GOAL,
            "policies_at_or_above_goal": [
                row["policy"] for row in go_policies
            ],
            "minimum_burden_go_policy": {
                "policy": minimum_burden["policy"],
                "score": minimum_overall["frozen_scorer"]["score"],
                **minimum_overall["background_budget_at_goal"],
            },
            "plausibility_basis": (
                "The minimum-burden passing policy is the already-materialized "
                "r10/k8 pool; its positive FP budget is comparable to the banked "
                "graph error scale and requires no wider raw candidate universe."
            ),
            "verdict": verdict,
        },
        "interpretation": {
            "root_finding": (
                "The old 0.895678/0.897746 numbers are target-limited swap "
                "projections, not literal proposal-pool ceilings. The exact "
                "current-pool oracle is already above 0.92."
            ),
            "cg1_target": (
                "present-but-unrankable current-pool associations and explicit "
                "background abstention before wider-radius generation"
            ),
        },
        "provenance": {
            "headroom": str(args.headroom),
            "headroom_sha256": _sha256(args.headroom),
            "reachability": str(args.reachability),
            "reachability_sha256": _sha256(args.reachability),
            "shell": str(args.shell),
            "shell_sha256": _sha256(args.shell),
            "zip": str(args.zip),
            "cache": str(args.cache),
            "fragments": str(args.fragments),
            "official_metrics_sha256": _sha256(
                ROOT
                / "external/kaggle-cell-tracking-competition/src/"
                "tracking_cellmot/metrics.py"
            ),
            "referenced_artifacts_fit_only_guarded": True,
        },
        "execution": {
            "device": "cpu",
            "deterministic": True,
            "model_training": False,
            "gpu_action": False,
            "kaggle_action": False,
            "submission_action": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )
    markdown = _markdown(report).encode("utf-8")
    _immutable_write(args.output, payload)
    _immutable_write(args.markdown, markdown)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--headroom", type=Path, default=DEFAULT_HEADROOM)
    value.add_argument("--reachability", type=Path, default=DEFAULT_REACHABILITY)
    value.add_argument("--shell", type=Path, default=DEFAULT_SHELL)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return value


if __name__ == "__main__":
    result = diagnose(parser().parse_args())
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "current_exact_ceiling": result["policy_ladder"][0]["families"][
                    "overall"
                ]["frozen_scorer"]["score"],
                "maximum": result["maxima"]["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )
