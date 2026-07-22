#!/usr/bin/env python3
"""Tune the pinned GPL CELLECT embedding head on fit117 point links.

The exact upstream 3D U-Net and published weights are used. Detection and
segmentation outputs are ignored: supervision is point centers, temporal GT
links, and guaranteed same-frame negatives. The script refuses to train unless
E4 has passed and never reads CV57 labels or pixels.

This integration is GPL-2.0-only. See third_party_licenses/CELLECT_NOTICE.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch
from torch import nn
import tracksdata as td
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.cellect_adapter import (  # noqa: E402
    CellectAdapterConfig,
    CellectContractError,
    CellectEmbeddingAdapter,
    UPSTREAM_COMMIT,
    batch_hard_temporal_triplet,
    load_published_backbone,
    rank_accuracy,
    same_frame_separation,
    sample_embeddings_zyx,
    sha256_file,
    squared_embedding_distance,
    topk_candidate_mask,
    validate_upstream_tree,
)
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.kaggle_calibration import SubmissionGraph, read_submission  # noqa: E402
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import _evaluate  # noqa: E402


SCHEMA = "celltrack-alpha.cellect-fit117-training.v1"
CHECKPOINT_SCHEMA = "celltrack-alpha.cellect-edge-checkpoint.v1"
EXPECTED_INDEX_SCHEMA = "celltrack-alpha.cellect-fit117-index.v1"
EXPECTED_GATE_SCHEMA = "celltrack-alpha.phase1-e4-e5.v1"
EXPECTED_RAW_COHORT = {
    "fn_eligible_raw_endpoint": 1_549,
    "fn_raw_in_gate": 1_232,
    "fn_raw_above_gate": 317,
    "fn_official_unmatched_endpoint": 464,
    "fn_synthetic_or_pruned_raw_endpoint": 81,
}
EXPECTED_METRIC_SHA256 = "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444"
EXPECTED_DIVISION_METRIC_SHA256 = "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9"
EXPECTED_FIT_MANIFEST_SHA256 = "20ef8056bea8afb60f76cf99357c7b62215b2db6c408d78a1c3ba21c41431a33"


@dataclass(frozen=True)
class VideoLabels:
    identity: str
    coords_by_t: Mapping[int, np.ndarray]
    ids_by_t: Mapping[int, np.ndarray]
    edges_by_t: Mapping[int, tuple[tuple[int, int], ...]]
    counts: Mapping[str, int]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CropExample:
    pair_source_yxz: np.ndarray
    pair_target_yxz: np.ndarray
    source_points_zyx: np.ndarray
    target_points_zyx: np.ndarray
    positive_mask: np.ndarray
    candidate_mask: np.ndarray
    physical_distance_sq: np.ndarray


def valid_triplet_rows(example: CropExample) -> np.ndarray:
    """Rows with at least one GT positive and one candidate negative."""

    positive = np.asarray(example.positive_mask, dtype=bool)
    candidates = np.asarray(example.candidate_mask, dtype=bool) | positive
    if positive.shape != candidates.shape:
        raise CellectContractError("CELLECT crop masks have incompatible shapes")
    return positive.any(axis=1) & (candidates & ~positive).any(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "reports/g02/cellect_fit117_index_v1.json",
    )
    parser.add_argument(
        "--appearance-gate",
        type=Path,
        default=ROOT / "reports/g02/phase1_e4_e5_holdout57_v1.json",
    )
    parser.add_argument(
        "--fitcache-root",
        type=Path,
        default=ROOT / "reports/g02/fitcache_117",
    )
    parser.add_argument(
        "--scorer-parity",
        type=Path,
        default=ROOT / "reports/g02/scorer_parity_v1.json",
    )
    parser.add_argument("--upstream", type=Path, default=ROOT / "external/CELLECT")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/cellect/cellect_fit117_best_v1.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/g02/cellect_fit117_training_v1.json",
    )
    parser.add_argument(
        "--supervision-audit-only",
        action="store_true",
        help="Validate and count raw FrameDet supervision without loading or training the model.",
    )
    parser.add_argument(
        "--supervision-audit-report",
        type=Path,
        default=ROOT / "reports/g02/cellect_fit117_raw_supervision_v1.json",
    )
    parser.add_argument("--epochs", type=int, default=149)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--validation-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41_116_016)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Allow CPU only when epochs=steps-per-epoch=validation-steps=1.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CellectContractError(f"JSON root is not an object: {path}")
    return value


def validate_gate(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != EXPECTED_GATE_SCHEMA or payload.get("status") != "complete":
        raise CellectContractError("E4/E5 report is absent, incomplete, or has the wrong schema")
    e4 = payload.get("e4")
    if not isinstance(e4, dict) or e4.get("decision") != "PASS":
        raise CellectContractError("E4 did not pass; CELLECT training is forbidden")
    cohort = payload.get("audit", {}).get("cohort_counts", {})
    observed = {key: int(cohort.get(key, -1)) for key in EXPECTED_RAW_COHORT}
    if observed != EXPECTED_RAW_COHORT:
        raise CellectContractError(f"E4 raw FrameDet cohort changed: {observed}")
    return payload


def validate_index(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != EXPECTED_INDEX_SCHEMA or payload.get("status") != "validated":
        raise CellectContractError("fit117 CELLECT index is not validated")
    boundary = payload.get("data_boundary", {})
    if boundary != {
        "train_fit_only": True,
        "held_out_used_for_exclusion_only": True,
        "held_out_labels_or_pixels_read": False,
        "test_used": False,
        "kaggle_action_performed": False,
    }:
        raise CellectContractError(f"fit117 CELLECT index boundary changed: {boundary}")
    split = payload.get("internal_split")
    if not isinstance(split, dict):
        raise CellectContractError("fit117 CELLECT index has no internal split")
    train = [str(value) for value in split.get("train_video_ids", ())]
    validation = [str(value) for value in split.get("validation_video_ids", ())]
    if len(train) != 94 or len(validation) != 23 or set(train) & set(validation):
        raise CellectContractError("fit117 internal 94/23 whole-video split changed")
    return payload


def validate_scorer_parity(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    fixture = payload.get("fixture_report", {})
    reference = payload.get("scorer_reference", {})
    if payload.get("artifact") != "scorer_parity_v1" or payload.get("schema_version") != 1:
        raise CellectContractError("official scorer parity ledger has the wrong identity")
    if fixture.get("overall_pass") is not True or len(fixture.get("cases", ())) != 7:
        raise CellectContractError("patched official scorer parity fixtures are not green")
    expected = {
        "metric_source_sha256": EXPECTED_METRIC_SHA256,
        "division_metric_source_sha256": EXPECTED_DIVISION_METRIC_SHA256,
    }
    observed = {key: reference.get(key) for key in expected}
    if observed != expected:
        raise CellectContractError(f"patched official scorer hashes changed: {observed}")
    for relative, expected_hash in (
        (reference.get("metric_source"), EXPECTED_METRIC_SHA256),
        (reference.get("division_metric_source"), EXPECTED_DIVISION_METRIC_SHA256),
    ):
        source = ROOT / str(relative)
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise CellectContractError(f"patched official scorer source changed: {source}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "official_checkout": fixture.get("official_checkout"),
        **expected,
        "fixture_count": 7,
        "overall_pass": True,
    }


def validate_fitcache(index: Mapping[str, Any], root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CellectContractError(f"fit117 manifest is absent: {manifest_path}")
    manifest_hash = sha256_file(manifest_path)
    indexed = index.get("inputs", {}).get("fit_manifest", {})
    if manifest_hash != EXPECTED_FIT_MANIFEST_SHA256 or indexed.get("sha256") != manifest_hash:
        raise CellectContractError(
            f"fit117 manifest hash changed: observed={manifest_hash} indexed={indexed.get('sha256')}"
        )
    manifest = load_json(manifest_path)
    boundary = manifest.get("data_boundary")
    expected_boundary = {
        "audit_used": False,
        "competition_submission_performed": False,
        "held_out_used": False,
        "test_used": False,
        "train_fit_only": True,
    }
    if manifest.get("status") != "validated" or boundary != expected_boundary:
        raise CellectContractError("fit117 cache manifest is not a validated fit-only corpus")
    indexed_ids = set(index["internal_split"]["train_video_ids"]) | set(
        index["internal_split"]["validation_video_ids"]
    )
    manifest_ids = {str(value) for value in manifest.get("video_ids", ())}
    if len(manifest_ids) != 117 or manifest_ids != indexed_ids:
        raise CellectContractError("fit117 cache identities differ from the frozen 94/23 split")
    return manifest


def prediction_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    nodes = [
        {
            "node_id": int(node.node_id),
            "t": int(node.t),
            "z": float(node.z),
            "y": float(node.y),
            "x": float(node.x),
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {"source_id": int(edge.source_id), "target_id": int(edge.target_id)}
        for edge in graph.edges
    ]
    return nodes, edges


def official_gt_to_pred(
    graph: SubmissionGraph,
    gt: Mapping[str, Any],
    scale_zyx: Sequence[float],
) -> dict[int, int]:
    """Match GT IDs to final prediction IDs with the patched official matcher."""

    nodes, edges = prediction_rows(graph)
    prediction = prediction_to_graph(nodes, edges)
    truth = geff_arrays_to_graph(dict(gt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", tuple(float(v) for v in scale_zyx), 7.0)
    ordered_pred_ids = [
        int(row["node_id"])
        for row in sorted(nodes, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    gt_nodes = gt["nodes"]
    ordered_gt_ids = [
        int(row[0])
        for row in sorted(
            zip(gt_nodes["node_id"], gt_nodes["t"], strict=True),
            key=lambda row: (int(row[1]), int(row[0])),
        )
    ]
    gt_to_pred: dict[int, int] = {}
    matched = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    for pred_internal, gt_internal in matched.iter_rows():
        if gt_internal is None or int(gt_internal) == -1:
            continue
        pred_index, gt_index = int(pred_internal), int(gt_internal)
        if not 0 <= pred_index < len(ordered_pred_ids) or not 0 <= gt_index < len(ordered_gt_ids):
            raise CellectContractError("official matcher returned an out-of-range node ID")
        gt_id, pred_id = ordered_gt_ids[gt_index], ordered_pred_ids[pred_index]
        if gt_id in gt_to_pred:
            raise CellectContractError("official GT matching is not one-to-one")
        gt_to_pred[gt_id] = pred_id
    return gt_to_pred


def cache_identity(meta: Mapping[str, Any]) -> bh_model.CacheIdentity:
    try:
        model = meta["contract"]["model"]
        return bh_model.CacheIdentity(
            config_sha256=str(model["config_sha256"]),
            source_sha256=str(model["source_sha256"]),
            weight_sha256=str(model["weight_sha256"]),
            pair_batch_size=int(model["pair_batch_size"]),
            view_batch_size=int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CellectContractError("fit117 cache metadata has no valid model identity") from exc


def raw_labels_from_match(
    gt: Mapping[str, Any],
    detections: Sequence[bh_model.FrameDet],
    gt_to_pred: Mapping[int, int],
    *,
    identity: str,
    total_frames: int,
    provenance: Mapping[str, Any] | None = None,
) -> VideoLabels:
    """Build point-link supervision on raw FrameDet IDs and coordinates."""

    ids_by_t: dict[int, np.ndarray] = {}
    coords_by_t: dict[int, np.ndarray] = {}
    raw_time_by_id: dict[int, int] = {}
    same_frame_negative_pairs = 0
    for detection in detections:
        t = int(detection.frame_index)
        ids = np.asarray(detection.node_ids, dtype=np.int64)
        coords = np.asarray(detection.coords_zyx, dtype=np.float64)
        if t in ids_by_t or coords.shape != (len(ids), 3) or len(set(ids.tolist())) != len(ids):
            raise CellectContractError(f"invalid raw FrameDet frame in {identity}: t={t}")
        ids_by_t[t], coords_by_t[t] = ids, coords
        same_frame_negative_pairs += len(ids) * (len(ids) - 1) // 2
        for row, node_id in enumerate(ids):
            node_id_int = int(node_id)
            if node_id_int in raw_time_by_id:
                raise CellectContractError(f"duplicate raw FrameDet node ID in {identity}: {node_id_int}")
            raw_time_by_id[node_id_int] = t

    nodes = gt["nodes"]
    gt_time_by_id = {
        int(node_id): int(t)
        for node_id, t in zip(nodes["node_id"], nodes["t"], strict=True)
    }
    edges_by_t_list: dict[int, list[tuple[int, int]]] = defaultdict(list)
    counts = defaultdict(int)
    positive_outdegree: dict[int, int] = defaultdict(int)
    for source, target in np.asarray(gt["edges"], dtype=np.int64):
        source_id, target_id = int(source), int(target)
        if source_id not in gt_time_by_id or target_id not in gt_time_by_id:
            raise CellectContractError(f"GT edge references absent node: {identity}")
        source_t, target_t = gt_time_by_id[source_id], gt_time_by_id[target_id]
        if target_t != source_t + 1:
            counts["gt_nonadjacent_edges"] += 1
            continue
        counts["gt_adjacent_edges"] += 1
        pred_source, pred_target = gt_to_pred.get(source_id), gt_to_pred.get(target_id)
        if pred_source is None or pred_target is None:
            counts["official_unmatched_endpoint_edges"] += 1
            continue
        counts["official_matched_endpoint_edges"] += 1
        if pred_source not in raw_time_by_id or pred_target not in raw_time_by_id:
            counts["synthetic_or_pruned_raw_endpoint_edges"] += 1
            continue
        if raw_time_by_id[pred_source] != source_t or raw_time_by_id[pred_target] != target_t:
            counts["raw_or_gt_nonadjacent_edges"] += 1
            continue
        edges_by_t_list[source_t].append((pred_source, pred_target))
        positive_outdegree[pred_source] += 1
        counts["raw_positive_edges"] += 1
        if source_t + 2 < int(total_frames):
            counts["training_positive_edges"] += 1
    for frame, edges in edges_by_t_list.items():
        target_count = len(ids_by_t.get(frame + 1, ()))
        by_source = defaultdict(int)
        for source, _ in edges:
            by_source[source] += 1
        counts["temporal_guaranteed_negative_pairs"] += sum(
            max(0, target_count - positive_count) for positive_count in by_source.values()
        )
    counts["raw_detection_nodes"] = len(raw_time_by_id)
    counts["raw_detection_frames"] = len(ids_by_t)
    counts["same_frame_guaranteed_negative_pairs"] = same_frame_negative_pairs
    counts["raw_positive_sources"] = len(positive_outdegree)
    counts["raw_division_sources"] = sum(value > 1 for value in positive_outdegree.values())
    counts["training_sample_frames"] = sum(
        bool(edges) and frame + 2 < int(total_frames) for frame, edges in edges_by_t_list.items()
    )
    return VideoLabels(
        identity=identity,
        coords_by_t=coords_by_t,
        ids_by_t=ids_by_t,
        edges_by_t={t: tuple(sorted(edges)) for t, edges in edges_by_t_list.items()},
        counts=dict(sorted(counts.items())),
        provenance=dict(provenance or {}),
    )


def labels_for_video(
    store: ZipStore,
    identity: str,
    fitcache_root: Path,
    manifest_record: Mapping[str, Any],
) -> VideoLabels:
    split_name, dataset = identity.split(":", 1)
    if split_name != "train":
        raise CellectContractError(f"non-fit identity in CELLECT labels: {identity}")
    paths = {
        "fragment": fitcache_root / "bh9out/fragments" / f"train__{dataset}.csv",
        "cache": fitcache_root / "bh9out/cache" / f"train__{dataset}.bhc",
        "cache_meta": fitcache_root / "bh9out/cache" / f"train__{dataset}.json",
    }
    expected_files = manifest_record.get("files", {})
    for label, manifest_key in (
        ("fragment", "fragments"),
        ("cache", "cache_bhc"),
        ("cache_meta", "cache_meta"),
    ):
        path = paths[label]
        expected_hash = expected_files.get(manifest_key, {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise CellectContractError(f"fit117 {label} hash mismatch: {identity}")
    meta = load_json(paths["cache_meta"])
    if meta.get("schema") != "celltrack-alpha.bh-cache-meta.v1":
        raise CellectContractError(f"unexpected fit117 cache schema: {identity}")
    if meta.get("cache_sha256") != expected_files["cache_bhc"]["sha256"]:
        raise CellectContractError(f"fit117 cache metadata does not bind its BHC: {identity}")
    detections = bh_model.load_sparse_cache(
        paths["cache"], expected_identity=cache_identity(meta)
    ).detections
    graphs = read_submission(paths["fragment"])
    if set(graphs) != {dataset}:
        raise CellectContractError(f"fit117 fragment identity mismatch: {identity}")
    gt = read_geff_arrays(store, split_name, dataset)
    image = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
    scale = tuple(float(value) for value in parse_scale(gt["attrs"]))
    gt_to_pred = official_gt_to_pred(graphs[dataset], gt, scale)
    return raw_labels_from_match(
        gt,
        detections,
        gt_to_pred,
        identity=identity,
        total_frames=int(image.shape[0]),
        provenance={
            "fragment": {"path": str(paths["fragment"]), "sha256": expected_files["fragments"]["sha256"]},
            "raw_cache": {"path": str(paths["cache"]), "sha256": expected_files["cache_bhc"]["sha256"]},
            "raw_cache_metadata": {"path": str(paths["cache_meta"]), "sha256": expected_files["cache_meta"]["sha256"]},
            "official_matched_gt_nodes": len(gt_to_pred),
            "coordinate_stage": "raw FrameDet before graph smoothing/projection",
        },
    )


def crop_with_pad(
    array: Any,
    frame_start: int,
    start_zyx: Sequence[int],
    shape_zyx: Sequence[int],
) -> np.ndarray:
    """Read three frames and return a zero-padded ``(3,Z,Y,X)`` crop."""

    starts = np.asarray(tuple(start_zyx), dtype=np.int64)
    shape = np.asarray(tuple(shape_zyx), dtype=np.int64)
    spatial = np.asarray(array.shape[1:], dtype=np.int64)
    if starts.shape != (3,) or shape.shape != (3,) or np.any(shape <= 0):
        raise CellectContractError("invalid CELLECT crop geometry")
    read_start = np.maximum(starts, 0)
    read_end = np.minimum(starts + shape, spatial)
    if np.any(read_start >= read_end):
        raise CellectContractError("CELLECT crop does not intersect the image")
    source = np.asarray(
        array[
            frame_start : frame_start + 3,
            int(read_start[0]) : int(read_end[0]),
            int(read_start[1]) : int(read_end[1]),
            int(read_start[2]) : int(read_end[2]),
        ],
        dtype=np.float32,
    )
    if source.shape[0] != 3:
        raise CellectContractError("CELLECT training needs three consecutive frames")
    before = np.maximum(-starts, 0)
    after = np.maximum(starts + shape - spatial, 0)
    result = np.pad(
        source,
        ((0, 0), (int(before[0]), int(after[0])), (int(before[1]), int(after[1])), (int(before[2]), int(after[2]))),
        mode="constant",
    )
    expected = (3, *(int(value) for value in shape))
    if result.shape != expected or not np.isfinite(result).all():
        raise CellectContractError(f"invalid padded CELLECT crop: {result.shape} != {expected}")
    return result


def build_crop_example(
    array: Any,
    labels: VideoLabels,
    frame: int,
    anchor_id: int,
    scale_zyx: Sequence[float],
    config: CellectAdapterConfig,
) -> CropExample:
    source_ids_all = labels.ids_by_t.get(frame, np.empty(0, dtype=np.int64))
    source_coords_all = labels.coords_by_t.get(frame, np.empty((0, 3), dtype=np.int64))
    target_ids_all = labels.ids_by_t.get(frame + 1, np.empty(0, dtype=np.int64))
    target_coords_all = labels.coords_by_t.get(frame + 1, np.empty((0, 3), dtype=np.int64))
    if anchor_id not in set(int(value) for value in source_ids_all):
        raise CellectContractError("selected CELLECT anchor is absent from its source frame")
    anchor_row = int(np.flatnonzero(source_ids_all == int(anchor_id))[0])
    anchor = source_coords_all[anchor_row]
    shape = np.asarray(config.crop_zyx, dtype=np.int64)
    # The feature map is voxel-indexed. Preserve raw subvoxel coordinates for
    # top-k/physical distance, but center and sample at the nearest image voxel.
    start = np.rint(anchor).astype(np.int64) - shape // 2
    raw = crop_with_pad(array, frame, start, shape)

    def local_points(
        ids: np.ndarray, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_raw = np.asarray(coords, dtype=np.float64) - start[None, :]
        keep = np.all((local_raw >= 0) & (local_raw < shape[None, :]), axis=1)
        local_raw = local_raw[keep]
        local_sample = np.rint(local_raw).astype(np.int64)
        local_sample = np.clip(local_sample, 0, shape[None, :] - 1)
        return ids[keep], local_raw, local_sample

    source_ids, source_coords, source_points = local_points(source_ids_all, source_coords_all)
    target_ids, target_coords, target_points = local_points(target_ids_all, target_coords_all)
    if not len(source_ids) or not len(target_ids):
        raise CellectContractError("CELLECT crop has an empty source or target frame")
    source_rows = {int(value): index for index, value in enumerate(source_ids)}
    target_rows = {int(value): index for index, value in enumerate(target_ids)}
    positive = np.zeros((len(source_ids), len(target_ids)), dtype=bool)
    for source_id, target_id in labels.edges_by_t.get(frame, ()):
        if source_id in source_rows and target_id in target_rows:
            positive[source_rows[source_id], target_rows[target_id]] = True
    if anchor_id not in source_rows or not positive[source_rows[anchor_id]].any():
        raise CellectContractError("CELLECT crop lost the selected anchor's positive edge")
    candidates = topk_candidate_mask(
        source_coords,
        target_coords,
        scale_zyx,
        k=config.candidate_top_k,
    ) | positive
    scale = np.asarray(tuple(scale_zyx), dtype=np.float64)
    physical_sq = np.sum(
        np.square((source_coords[:, None, :] - target_coords[None, :, :]) * scale),
        axis=2,
    )
    # CELLECT expects (B,C,Y,X,Z); batch is added in the training step.
    yxz = raw.transpose(0, 2, 3, 1)
    return CropExample(
        pair_source_yxz=np.ascontiguousarray(yxz[0:2]),
        pair_target_yxz=np.ascontiguousarray(yxz[1:3]),
        source_points_zyx=np.ascontiguousarray(source_points),
        target_points_zyx=np.ascontiguousarray(target_points),
        positive_mask=positive,
        candidate_mask=candidates,
        physical_distance_sq=physical_sq,
    )


def example_tensors(example: CropExample, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pair_source": torch.from_numpy(example.pair_source_yxz).unsqueeze(0).to(device),
        "pair_target": torch.from_numpy(example.pair_target_yxz).unsqueeze(0).to(device),
        "source_points": torch.from_numpy(example.source_points_zyx).to(device),
        "target_points": torch.from_numpy(example.target_points_zyx).to(device),
        "positive": torch.from_numpy(example.positive_mask).to(device),
        "candidate": torch.from_numpy(example.candidate_mask).to(device),
    }


def score_example(
    model: CellectEmbeddingAdapter,
    example: CropExample,
    device: torch.device,
    config: CellectAdapterConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    valid_rows = valid_triplet_rows(example)
    if not np.any(valid_rows):
        raise CellectContractError("CELLECT crop contains no confusable triplet anchor")
    tensors = example_tensors(example, device)
    source_map = model(tensors["pair_source"])
    target_map = model(tensors["pair_target"])
    source = sample_embeddings_zyx(source_map, tensors["source_points"])
    target = sample_embeddings_zyx(target_map, tensors["target_points"])
    temporal, counts = batch_hard_temporal_triplet(
        source,
        target,
        tensors["positive"],
        candidate_mask=tensors["candidate"],
        margin=config.triplet_margin,
    )
    separation = same_frame_separation(source, margin=config.triplet_margin)
    separation = separation + same_frame_separation(target, margin=config.triplet_margin)
    total = temporal + config.same_frame_weight * separation
    embedding_distance = squared_embedding_distance(source, target).detach().float().cpu().numpy()
    embedding_rank = rank_accuracy(
        embedding_distance[valid_rows],
        example.positive_mask[valid_rows],
        example.candidate_mask[valid_rows],
    )
    physical_rank = rank_accuracy(
        example.physical_distance_sq[valid_rows],
        example.positive_mask[valid_rows],
        example.candidate_mask[valid_rows],
    )
    return total, {
        "loss": float(total.detach().item()),
        "temporal_loss": float(temporal.detach().item()),
        "separation_loss": float(separation.detach().item()),
        "triplet": counts,
        "embedding_rank": embedding_rank,
        "physical_rank": physical_rank,
    }


def sample_specifications(
    index: Mapping[str, Any],
    labels: Mapping[str, VideoLabels],
    split_name: str,
) -> list[tuple[str, int]]:
    internal = index["internal_split"]
    identities = internal[f"{split_name}_video_ids"]
    videos = index["videos"]
    result: list[tuple[str, int]] = []
    for identity in identities:
        indexed_frames = {int(value) for value in videos[identity]["sample_frames"]}
        for frame, edges in sorted(labels[identity].edges_by_t.items()):
            if edges and frame + 2 < int(videos[identity]["image_shape_tzyx"][0]):
                if frame not in indexed_frames:
                    raise CellectContractError(
                        f"raw-mapped training frame was absent from the GT index: {identity} t={frame}"
                    )
                result.append((identity, int(frame)))
    if not result:
        raise CellectContractError(f"empty CELLECT {split_name} sample list")
    return result


def choose_anchor(labels: VideoLabels, frame: int, rng: random.Random) -> int:
    sources = sorted({source for source, _ in labels.edges_by_t.get(frame, ())})
    if not sources:
        raise CellectContractError("indexed CELLECT frame has no adjacent GT source")
    return sources[rng.randrange(len(sources))]


def choose_confusable_example(
    specifications: Sequence[tuple[str, int]],
    arrays: Mapping[str, Any],
    labels: Mapping[str, VideoLabels],
    index: Mapping[str, Any],
    config: CellectAdapterConfig,
    rng: random.Random,
    *,
    attempts: int = 128,
) -> CropExample:
    """Bounded sampling that cannot silently turn an optimizer step into a no-op."""

    for _ in range(attempts):
        identity, frame = specifications[rng.randrange(len(specifications))]
        anchor = choose_anchor(labels[identity], frame, rng)
        example = build_crop_example(
            arrays[identity],
            labels[identity],
            frame,
            anchor,
            index["videos"][identity]["scale_zyx_um"],
            config,
        )
        if np.any(valid_triplet_rows(example)):
            return example
    raise CellectContractError(
        f"no confusable CELLECT triplet found in {attempts} bounded sample attempts"
    )


def validate_model(
    model: CellectEmbeddingAdapter,
    store: ZipStore,
    arrays: Mapping[str, Any],
    labels: Mapping[str, VideoLabels],
    index: Mapping[str, Any],
    specifications: Sequence[tuple[str, int]],
    *,
    steps: int,
    device: torch.device,
    config: CellectAdapterConfig,
) -> dict[str, float | int]:
    model.eval()
    embedding_correct = physical_correct = rows = valid_examples = 0
    loss_sum = 0.0
    with torch.no_grad():
        for identity, frame in specifications:
            if valid_examples >= steps:
                break
            source_ids = sorted({source for source, _ in labels[identity].edges_by_t.get(frame, ())})
            if not source_ids:
                continue
            example = None
            for anchor_id in source_ids:
                candidate = build_crop_example(
                    arrays[identity],
                    labels[identity],
                    frame,
                    anchor_id,
                    index["videos"][identity]["scale_zyx_um"],
                    config,
                )
                if np.any(valid_triplet_rows(candidate)):
                    example = candidate
                    break
            if example is None:
                continue
            loss, metrics = score_example(model, example, device, config)
            embedding = metrics["embedding_rank"]
            physical = metrics["physical_rank"]
            if int(embedding["rows"]) != int(physical["rows"]):
                raise CellectContractError("embedding and distance rank denominators differ")
            rows += int(embedding["rows"])
            embedding_correct += int(embedding["correct"])
            physical_correct += int(physical["correct"])
            loss_sum += float(loss.item())
            valid_examples += 1
    if not rows or not valid_examples:
        raise CellectContractError("CELLECT validation produced no ranked edges")
    embedding_accuracy = embedding_correct / rows
    physical_accuracy = physical_correct / rows
    return {
        "examples": valid_examples,
        "rows": rows,
        "loss": loss_sum / valid_examples,
        "embedding_correct": embedding_correct,
        "embedding_top1_accuracy": embedding_accuracy,
        "physical_correct": physical_correct,
        "physical_top1_accuracy": physical_accuracy,
        "delta_vs_physical": embedding_accuracy - physical_accuracy,
    }


def checkpoint_payload(
    model: CellectEmbeddingAdapter,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "license": "GPL-2.0-only",
        "upstream_commit": UPSTREAM_COMMIT,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation": dict(validation),
        "provenance": dict(provenance),
    }


def load_fit117_labels(
    args: argparse.Namespace,
    index: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, VideoLabels]:
    identities = sorted(
        set(index["internal_split"]["train_video_ids"])
        | set(index["internal_split"]["validation_video_ids"])
    )
    records = manifest.get("videos", {})
    if set(records) != set(identities):
        raise CellectContractError("fit117 manifest records do not exactly cover the 94/23 split")
    labels: dict[str, VideoLabels] = {}
    with ZipStore(args.archive, mode="r") as store:
        for identity in identities:
            labels[identity] = labels_for_video(
                store,
                identity,
                args.fitcache_root,
                records[identity],
            )
    return labels


def supervision_audit(
    labels: Mapping[str, VideoLabels],
    index: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    scorer: Mapping[str, Any],
    upstream: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_by_identity = {
        identity: split_name
        for split_name in ("train", "validation")
        for identity in index["internal_split"][f"{split_name}_video_ids"]
    }
    aggregate: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "all"):
        selected = [
            value
            for identity, value in labels.items()
            if split_name == "all" or split_by_identity[identity] == split_name
        ]
        keys = sorted({key for value in selected for key in value.counts})
        aggregate[split_name] = {
            "videos": len(selected),
            **{key: sum(int(value.counts.get(key, 0)) for value in selected) for key in keys},
        }
    totals = aggregate["all"]
    if aggregate["train"]["videos"] != 94 or aggregate["validation"]["videos"] != 23:
        raise CellectContractError("raw supervision changed the frozen 94/23 video split")
    if totals.get("gt_adjacent_edges") != int(index["totals"]["adjacent_edge_count"]):
        raise CellectContractError(
            "raw supervision GT adjacent-edge denominator differs from the fit117 index"
        )
    partitioned = (
        totals.get("raw_positive_edges", 0)
        + totals.get("official_unmatched_endpoint_edges", 0)
        + totals.get("synthetic_or_pruned_raw_endpoint_edges", 0)
        + totals.get("raw_or_gt_nonadjacent_edges", 0)
    )
    if partitioned != totals.get("gt_adjacent_edges"):
        raise CellectContractError(
            f"raw supervision edge partition is incomplete: {partitioned} != {totals.get('gt_adjacent_edges')}"
        )
    for required in (
        "raw_positive_edges",
        "training_positive_edges",
        "raw_positive_sources",
        "training_sample_frames",
        "same_frame_guaranteed_negative_pairs",
        "temporal_guaranteed_negative_pairs",
    ):
        if totals.get(required, 0) <= 0:
            raise CellectContractError(f"raw supervision has an empty required count: {required}")
    train_specs = sample_specifications(index, labels, "train")
    validation_specs = sample_specifications(index, labels, "validation")
    if len(train_specs) != aggregate["train"]["training_sample_frames"]:
        raise CellectContractError("train sample-frame count differs from raw labels")
    if len(validation_specs) != aggregate["validation"]["training_sample_frames"]:
        raise CellectContractError("validation sample-frame count differs from raw labels")
    return {
        "schema": "celltrack-alpha.cellect-fit117-raw-supervision.v1",
        "status": "validated_no_training",
        "decision": "READY_FOR_EXPLICITLY_APPROVED_TRAINING",
        "supervision_contract": {
            "positive_identity": (
                "Adjacent GT edge whose two endpoints are mapped by patched official node "
                "matching to final prediction IDs and both IDs exist in raw FrameDet."
            ),
            "positive_coordinates": "raw fit117 FrameDet z,y,x; nearest voxel only at feature-map sampling",
            "same_frame_negative_universe": (
                "every unordered pair of distinct raw detections in each frame"
            ),
            "same_frame_negatives_optimized": (
                "every distinct raw-detection pair co-occurring in a sampled crop is passed "
                "to same_frame_separation"
            ),
            "temporal_negatives": (
                "for each raw-mapped positive source, every distinct raw next-frame detection "
                "other than its GT-mapped positive target(s); optimizer retains top-k=10"
            ),
            "division_labels_in_training": "multiple positive targets are retained",
            "division_output_enabled": False,
            "candidate_top_k": CellectAdapterConfig().candidate_top_k,
        },
        "counts": aggregate,
        "coverage": {
            "positive_edge_fraction_of_gt_adjacent": (
                totals["raw_positive_edges"] / totals["gt_adjacent_edges"]
            ),
            "training_edge_fraction_of_gt_training": (
                totals["training_positive_edges"] / int(index["totals"]["training_edge_count"])
            ),
            "train_sample_specifications": len(train_specs),
            "validation_sample_specifications": len(validation_specs),
        },
        "per_video": {
            identity: {
                "split": split_by_identity[identity],
                "counts": dict(labels[identity].counts),
                "provenance": dict(labels[identity].provenance),
            }
            for identity in sorted(labels)
        },
        "provenance": {
            "archive": {"path": str(args.archive), "bytes": args.archive.stat().st_size},
            "index": {"path": str(args.index), "sha256": sha256_file(args.index)},
            "fitcache_manifest": {
                "path": str(args.fitcache_root / "manifest.json"),
                "sha256": sha256_file(args.fitcache_root / "manifest.json"),
            },
            "appearance_gate": {
                "path": str(args.appearance_gate),
                "sha256": sha256_file(args.appearance_gate),
                "decision": gate["e4"]["decision"],
            },
            "patched_official_scorer": dict(scorer),
            "upstream": dict(upstream),
            "internal_split": {
                "train_videos": 94,
                "validation_videos": 23,
                "unchanged_from_index": True,
            },
            "cv57_labels_or_pixels_read": False,
            "test_read": False,
            "kaggle_action_performed": False,
            "training_performed": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.archive,
        args.index,
        args.appearance_gate,
        args.fitcache_root,
        args.scorer_parity,
        args.upstream,
    ):
        if not path.exists():
            raise CellectContractError(f"missing CELLECT training input: {path}")
    outputs = (
        (args.supervision_audit_report,)
        if args.supervision_audit_only
        else (args.output, args.report)
    )
    for output in outputs:
        if output.exists():
            raise CellectContractError(f"refusing to overwrite CELLECT artifact: {output}")
    if min(args.epochs, args.steps_per_epoch, args.validation_steps, args.gradient_accumulation) <= 0:
        raise CellectContractError("training loop counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise CellectContractError("learning rate must be finite and positive")

    gate = validate_gate(args.appearance_gate)
    index = validate_index(args.index)
    manifest = validate_fitcache(index, args.fitcache_root)
    scorer = validate_scorer_parity(args.scorer_parity)
    upstream = validate_upstream_tree(args.upstream)
    labels = load_fit117_labels(args, index, manifest)
    audit = supervision_audit(
        labels,
        index,
        gate=gate,
        scorer=scorer,
        upstream=upstream,
        args=args,
    )
    if args.supervision_audit_only:
        args.supervision_audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.supervision_audit_report.write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return audit

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise CellectContractError("full CELLECT tuning requires CUDA")
        device = torch.device("cuda")
    else:
        if not (
            args.allow_cpu_smoke
            and args.epochs == args.steps_per_epoch == args.validation_steps == 1
        ):
            raise CellectContractError(
                "CPU is restricted to --allow-cpu-smoke with one epoch/train/validation step"
            )
        device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    config = CellectAdapterConfig()
    train_specs = sample_specifications(index, labels, "train")
    validation_specs = sample_specifications(index, labels, "validation")
    validation_specs = sorted(
        validation_specs,
        key=lambda value: hashlib.sha256(f"validation:{value[0]}:{value[1]}".encode("ascii")).hexdigest(),
    )

    model = CellectEmbeddingAdapter(load_published_backbone(args.upstream, device=device), config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    amp_enabled = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    provenance = {
        "archive": {"path": str(args.archive), "bytes": args.archive.stat().st_size},
        "index": {"path": str(args.index), "sha256": sha256_file(args.index)},
        "appearance_gate": {
            "path": str(args.appearance_gate),
            "sha256": sha256_file(args.appearance_gate),
            "decision": gate["e4"]["decision"],
        },
        "fitcache_manifest": {
            "path": str(args.fitcache_root / "manifest.json"),
            "sha256": sha256_file(args.fitcache_root / "manifest.json"),
        },
        "patched_official_scorer": scorer,
        "upstream": upstream,
        "config": config.as_dict(),
        "seed": int(args.seed),
        "device": str(device),
        "amp_mandatory_on_cuda": True,
        "cv57_read": False,
        "test_read": False,
        "kaggle_action_performed": False,
        "division_output_enabled": False,
        "supervision": {
            "coordinate_stage": "raw fit117 FrameDet before graph smoothing/projection",
            "positive_edges": audit["counts"]["all"]["raw_positive_edges"],
            "training_positive_edges": audit["counts"]["all"]["training_positive_edges"],
            "same_frame_guaranteed_negative_pairs": audit["counts"]["all"]["same_frame_guaranteed_negative_pairs"],
            "candidate_top_k": config.candidate_top_k,
            "split_train_videos": audit["counts"]["train"]["videos"],
            "split_validation_videos": audit["counts"]["validation"]["videos"],
        },
    }

    history: list[dict[str, Any]] = []
    best_delta = -math.inf
    best_epoch = -1
    started = time.perf_counter()
    all_ids = sorted(labels)
    with ZipStore(args.archive, mode="r") as store:
        arrays = {
            identity: zarr.open_array(
                store=store,
                path=f"train/{identity.split(':', 1)[1]}.zarr/0",
                mode="r",
            )
            for identity in all_ids
        }
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            trained_steps = 0
            for step in range(args.steps_per_epoch):
                example = choose_confusable_example(
                    train_specs,
                    arrays,
                    labels,
                    index,
                    config,
                    rng,
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    loss, step_metrics = score_example(model, example, device, config)
                    scaled_loss = loss / args.gradient_accumulation
                if not bool(torch.isfinite(loss).item()):
                    raise CellectContractError("CELLECT training loss became non-finite")
                if int(step_metrics["triplet"]["valid_anchors"]) <= 0:
                    raise CellectContractError("CELLECT optimizer step has no valid triplet anchor")
                scaler.scale(scaled_loss).backward()
                if (step + 1) % args.gradient_accumulation == 0 or step + 1 == args.steps_per_epoch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_loss += float(loss.detach().item())
                trained_steps += 1

            validation = validate_model(
                model,
                store,
                arrays,
                labels,
                index,
                validation_specs,
                steps=min(args.validation_steps, len(validation_specs)),
                device=device,
                config=config,
            )
            epoch_row = {
                "epoch": epoch,
                "train_steps": trained_steps,
                "train_loss": train_loss / max(1, trained_steps),
                "validation": validation,
            }
            history.append(epoch_row)
            delta = float(validation["delta_vs_physical"])
            if delta > best_delta:
                best_delta, best_epoch = delta, epoch
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        epoch=epoch,
                        validation=validation,
                        provenance=provenance,
                    ),
                    args.output,
                )
            print(json.dumps(epoch_row, sort_keys=True), flush=True)

    report = {
        "schema": SCHEMA,
        "status": "qualified_fit_validation" if best_delta > 0.0 else "failed_fit_validation",
        "decision_rule": (
            "Qualify for one untouched CV57 evaluation only if the best fit117 whole-video "
            "validation embedding top-1 accuracy exceeds physical-distance top-1 accuracy."
        ),
        "best_epoch": best_epoch,
        "best_delta_vs_physical": best_delta,
        "checkpoint": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "license": "GPL-2.0-only",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "provenance": provenance,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run(parse_args())
    if report["status"] == "validated_no_training":
        print(json.dumps({
            "status": report["status"],
            "decision": report["decision"],
            "counts": report["counts"]["all"],
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps({
        "status": report["status"],
        "best_epoch": report["best_epoch"],
        "best_delta_vs_physical": report["best_delta_vs_physical"],
        "checkpoint": report["checkpoint"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
