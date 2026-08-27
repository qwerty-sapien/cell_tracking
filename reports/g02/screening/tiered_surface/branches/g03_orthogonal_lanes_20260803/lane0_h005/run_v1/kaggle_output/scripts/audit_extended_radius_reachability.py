#!/usr/bin/env python3
"""Audit fit-only association reachability beyond the frozen 10-um/cap-8 tensor.

This diagnostic uses official 7-um detection-to-GT matching on the existing
105/12 fit split.  For every one-to-one adjacent GT edge whose endpoints are
both detected, it measures whether the exact predicted endpoint pair is
reachable under deterministic radius/cap policies and accounts for the
candidate-pool multiplier.  It does not score or emit predictions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_bh  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.edge_data import match_detections_to_gt  # noqa: E402
from celltrack_baseline.rule_based import read_geff_arrays  # noqa: E402
from p4_edge_attention import DEFAULT_CACHE, DEFAULT_ZIP, _cache_identity  # noqa: E402


SCHEMA = "celltrack-alpha.extended-radius-reachability-audit.v1"
SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
DEFAULT_PREPARE = (
    ROOT / "reports/g02/selected_graph_conflict_prepare_v1/prepare.json"
)
DEFAULT_OUTPUT = (
    ROOT / "reports/g02/extended_radius_reachability_v1.json"
)
POLICIES = (
    (10.0, 8),
    (10.0, 16),
    (12.0, 8),
    (12.0, 16),
    (15.0, 8),
    (15.0, 16),
    (20.0, 8),
    (20.0, 16),
    (20.0, 32),
)


class ReachabilityAuditError(RuntimeError):
    """A fit-only reachability identity or counting invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReachabilityAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReachabilityAuditError(f"expected JSON object: {path}")
    return value


def policy_name(radius_um: float, cap: int) -> str:
    return f"r{int(radius_um):02d}_k{cap:02d}"


def deterministic_rank(
    source_physical: np.ndarray,
    target_physical: np.ndarray,
    target_ids: np.ndarray,
    target_index: int,
) -> tuple[float, int]:
    """Return exact distance and one-based distance/node-ID deterministic rank."""

    source = np.asarray(source_physical, dtype=np.float64)
    targets = np.asarray(target_physical, dtype=np.float64)
    ids = np.asarray(target_ids, dtype=np.int64)
    if (
        source.shape != (3,)
        or targets.ndim != 2
        or targets.shape[1:] != (3,)
        or ids.shape != (len(targets),)
        or not 0 <= target_index < len(targets)
    ):
        raise ReachabilityAuditError("rank inputs are malformed")
    distances = np.linalg.norm(targets - source[None, :], axis=1)
    order = sorted(
        range(len(targets)),
        key=lambda index: (
            float(distances[index]),
            int(ids[index]),
            index,
        ),
    )
    return float(distances[target_index]), order.index(target_index) + 1


def reachable(distance_um: float, rank: int, radius_um: float, cap: int) -> bool:
    if distance_um < 0.0 or rank < 1 or radius_um <= 0.0 or cap < 1:
        raise ReachabilityAuditError("policy reachability inputs are invalid")
    return distance_um <= radius_um + 1.0e-9 and rank <= cap


def _candidate_pool_counts(
    cache: bh_model.SparseInferenceCache,
    policies: Sequence[tuple[float, int]],
) -> dict[str, int]:
    result = {policy_name(radius, cap): 0 for radius, cap in policies}
    for source, target in zip(
        cache.detections[:-1], cache.detections[1:], strict=True
    ):
        source_points = np.asarray(source.coords_zyx, dtype=np.float64) * SCALE_ZYX
        target_points = np.asarray(target.coords_zyx, dtype=np.float64) * SCALE_ZYX
        if not len(source_points) or not len(target_points):
            continue
        tree = cKDTree(target_points)
        for radius, cap in policies:
            counts = tree.query_ball_point(
                source_points, radius, return_length=True
            )
            result[policy_name(radius, cap)] += int(
                np.minimum(np.asarray(counts, dtype=np.int64), cap).sum()
            )
    return result


def _video(
    cache: bh_model.SparseInferenceCache,
    gt: Mapping[str, Any],
    *,
    policies: Sequence[tuple[float, int]],
) -> dict[str, Any]:
    detection_to_gt, gt_edges = match_detections_to_gt(cache.detections, gt)
    gt_to_detection = {
        int(gt_id): int(detection_id)
        for detection_id, gt_id in detection_to_gt.items()
        if gt_id is not None
    }
    if len(gt_to_detection) != sum(
        gt_id is not None for gt_id in detection_to_gt.values()
    ):
        raise ReachabilityAuditError("official matching is not one-to-one")
    detections: dict[int, tuple[int, np.ndarray]] = {}
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in cache.detections:
        ids = np.asarray(frame.node_ids, dtype=np.int64)
        coords = np.asarray(frame.coords_zyx, dtype=np.float64)
        by_frame[int(frame.frame_index)] = (ids, coords * SCALE_ZYX)
        for node_id, coordinate in zip(ids, coords, strict=True):
            detections[int(node_id)] = (
                int(frame.frame_index),
                coordinate * SCALE_ZYX,
            )
    outdegree: dict[int, int] = defaultdict(int)
    indegree: dict[int, int] = defaultdict(int)
    for source, target in gt_edges:
        outdegree[int(source)] += 1
        indegree[int(target)] += 1

    counts: dict[str, int] = defaultdict(int)
    policy_reachable = {
        policy_name(radius, cap): 0 for radius, cap in policies
    }
    distances: list[float] = []
    ranks: list[int] = []
    for gt_source, gt_target in sorted(gt_edges):
        counts["gt_edges"] += 1
        if outdegree[gt_source] != 1 or indegree[gt_target] != 1:
            counts["division_or_merge_edges"] += 1
            continue
        counts["one_to_one_gt_edges"] += 1
        source_id = gt_to_detection.get(gt_source)
        target_id = gt_to_detection.get(gt_target)
        if source_id is None or target_id is None:
            counts["unmatched_endpoint_edges"] += 1
            continue
        counts["matched_endpoint_edges"] += 1
        source_t, source_point = detections[source_id]
        target_t, _target_point = detections[target_id]
        if target_t != source_t + 1:
            counts["nonadjacent_detection_edges"] += 1
            continue
        counts["matched_adjacent_edges"] += 1
        target_ids, target_points = by_frame[target_t]
        locations = np.flatnonzero(target_ids == target_id)
        if len(locations) != 1:
            raise ReachabilityAuditError("matched target is absent or repeated")
        distance, rank = deterministic_rank(
            source_point,
            target_points,
            target_ids,
            int(locations[0]),
        )
        distances.append(distance)
        ranks.append(rank)
        default = reachable(distance, rank, 10.0, 8)
        counts["default_reachable_edges"] += int(default)
        counts["default_excluded_edges"] += int(not default)
        counts["excluded_distance_above_10um"] += int(distance > 10.0 + 1.0e-9)
        counts["excluded_rank_above_8_within_10um"] += int(
            distance <= 10.0 + 1.0e-9 and rank > 8
        )
        for radius, cap in policies:
            policy_reachable[policy_name(radius, cap)] += int(
                reachable(distance, rank, radius, cap)
            )
    if (
        counts["matched_endpoint_edges"]
        != counts["matched_adjacent_edges"]
        + counts["nonadjacent_detection_edges"]
        or counts["matched_adjacent_edges"]
        != counts["default_reachable_edges"] + counts["default_excluded_edges"]
    ):
        raise ReachabilityAuditError("reachability categories do not partition")
    candidate_pool = _candidate_pool_counts(cache, policies)
    return {
        "counts": dict(counts),
        "policy_reachable": policy_reachable,
        "candidate_pool": candidate_pool,
        "distance_quantiles_um": {
            str(q): float(np.quantile(distances, q)) if distances else None
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "rank_quantiles": {
            str(q): float(np.quantile(ranks, q)) if ranks else None
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count_names = sorted(
        {
            name
            for row in rows
            for name in row["metrics"]["counts"]
        }
    )
    counts = {
        name: sum(int(row["metrics"]["counts"].get(name, 0)) for row in rows)
        for name in count_names
    }
    policies = sorted(rows[0]["metrics"]["policy_reachable"]) if rows else []
    reachable_totals = {
        policy: sum(
            int(row["metrics"]["policy_reachable"][policy]) for row in rows
        )
        for policy in policies
    }
    pool_totals = {
        policy: sum(int(row["metrics"]["candidate_pool"][policy]) for row in rows)
        for policy in policies
    }
    baseline_pool = pool_totals.get("r10_k08", 0)
    baseline_reachable = reachable_totals.get("r10_k08", 0)
    policy_rows = []
    for policy in policies:
        policy_rows.append(
            {
                "policy": policy,
                "reachable_one_to_one_matched_adjacent_edges": (
                    reachable_totals[policy]
                ),
                "additional_reachable_vs_r10_k08": (
                    reachable_totals[policy] - baseline_reachable
                ),
                "candidate_pairs": pool_totals[policy],
                "candidate_pool_multiplier_vs_r10_k08": (
                    pool_totals[policy] / baseline_pool
                    if baseline_pool
                    else None
                ),
            }
        )
    return {
        "video_count": len(rows),
        "counts": counts,
        "policies": policy_rows,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ReachabilityAuditError(f"refusing to overwrite output: {args.output}")
    prepare = _load_json(args.prepare)
    scope = prepare.get("scope")
    if (
        prepare.get("schema")
        != "celltrack-alpha.selected-graph-conflict-prepare.v1"
        or not isinstance(scope, Mapping)
        or len(scope.get("train_video_ids", [])) != 105
        or len(scope.get("validation_video_ids", [])) != 12
        or any(
            scope.get(key) is not False
            for key in ("held_out_used", "audit_used", "test_used")
        )
    ):
        raise ReachabilityAuditError("prepare report is not exact fit-only 105/12")
    train_ids = tuple(str(v) for v in scope["train_video_ids"])
    validation_ids = tuple(str(v) for v in scope["validation_video_ids"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ZipStore(args.zip, mode="r") as store:
        for index, identity in enumerate((*train_ids, *validation_ids), start=1):
            slug = identity.replace(":", "__", 1)
            cache = bh_model.load_sparse_cache(
                args.cache / f"{slug}.bhc",
                expected_identity=_cache_identity(args.cache / f"{slug}.json"),
            )
            dataset = identity.split(":", 1)[1]
            metrics = _video(
                cache,
                read_geff_arrays(store, "train", dataset),
                policies=POLICIES,
            )
            rows.append(
                {
                    "identity": identity,
                    "partition": "train" if identity in train_ids else "validation",
                    "metrics": metrics,
                }
            )
            print(
                f"extended reachability {index:03d}/117 {identity} "
                f"excluded={metrics['counts'].get('default_excluded_edges', 0)}",
                flush=True,
            )
    train_rows = [row for row in rows if row["partition"] == "train"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    result = {
        "schema": SCHEMA,
        "status": "completed_read_only_diagnostic",
        "scope": {
            "train_video_ids": list(train_ids),
            "validation_video_ids": list(validation_ids),
            "fit_only": True,
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "matching": {
            "method": "official physical one-to-one detection matching",
            "radius_um": 7.0,
            "scale_zyx_um": SCALE_ZYX.tolist(),
        },
        "baseline_candidate_contract": {
            "radius_um": 10.0,
            "neighbor_cap": 8,
            "ordering": "distance, target node ID, target input index",
        },
        "train": _aggregate(train_rows),
        "validation": _aggregate(validation_rows),
        "all_fit": _aggregate(rows),
        "videos": rows,
        "decision_contract": {
            "role": "reachability/headroom only; no prediction rule selected",
            "next_gate": (
                "fit-only correctness and deployment-background abstention "
                "before any graph replay"
            ),
            "gnn": False,
        },
        "provenance": {
            "prepare": str(args.prepare),
            "prepare_sha256": _sha256(args.prepare),
            "zip": str(args.zip),
            "cache": str(args.cache),
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_action": False,
            "kaggle_action": False,
        },
    }
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--prepare", type=Path, default=DEFAULT_PREPARE)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "train": report["train"]["counts"],
                "validation": report["validation"]["counts"],
                "validation_policies": report["validation"]["policies"],
            },
            sort_keys=True,
        )
    )
