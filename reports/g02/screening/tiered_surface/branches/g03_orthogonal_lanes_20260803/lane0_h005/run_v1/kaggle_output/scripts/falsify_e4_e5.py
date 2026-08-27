#!/usr/bin/env python3
"""Run the CPU-only E4 appearance and E5 division-morphology falsifiers.

This script is deliberately diagnostic.  It reads the exact 57-video fold-0
division-off prediction fragments, reconstructs official node matches, checks
the frozen 34,174/2,736/2,094 baseline contract, and only then reads images.
It never changes tracking geometry or emits a submission/model artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import tracksdata as td
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (
    ROOT / "src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.kaggle_calibration import SubmissionGraph, read_submission  # noqa: E402
from celltrack_alpha import bh_model  # noqa: E402
from celltrack_alpha.official_eval import geff_arrays_to_graph, prediction_to_graph  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph  # noqa: E402


SCHEMA = "celltrack-alpha.phase1-e4-e5.v1"
EXCLUDED_IDENTITY = "train:6bba_f4ae811c"
EXPECTED = {
    "videos": 57,
    "edge_tp": 34_174,
    "edge_fp": 2_736,
    "edge_fn": 2_094,
    "gt_divisions": 42,
    "adjusted_edge_jaccard": 0.8711885823490648,
}
EXPECTED_ZIP_BYTES = 87_393_127_165
EXPECTED_SPLIT_SHA256 = "ddb313cbecb11e64a05f836f503b8cb30605143ef3b3e64ab9358ea9d8af8769"
GATE_UM = 10.0
MATCH_UM = 7.0
PATCH_SHAPE = (16, 32, 32)
SPHERE_RADIUS_UM = 4.5
SHELL_RADIUS_UM = 6.0
PCA_COMPONENTS = 16
PCA_TRAIN_CAP = 20_000
CV_FOLDS = 5
BOOTSTRAPS = 2_000
RNG_SEED = 45_045


class FalsificationError(RuntimeError):
    """A frozen input, count, or analysis invariant changed unexpectedly."""


@dataclass(frozen=True)
class MatchedVideo:
    identity: str
    graph: SubmissionGraph
    gt: dict[str, Any]
    gt_nodes: dict[int, tuple[int, float, float, float]]
    gt_edges: tuple[tuple[int, int], ...]
    pred_to_gt: dict[int, int]
    gt_to_pred: dict[int, int]
    pred_edges: frozenset[tuple[int, int]]
    scale: tuple[float, float, float]
    official_edge_counts: tuple[int, int, int]
    detections: tuple[bh_model.FrameDet, ...]


@dataclass
class AppearanceEvent:
    identity: str
    kind: str
    candidate_count: int
    true_index: int
    distance_scores: np.ndarray
    ncc_3d_scores: np.ndarray
    ncc_lateral_scores: np.ndarray
    pca_3d_source: np.ndarray
    pca_3d_candidates: np.ndarray
    pca_lateral_source: np.ndarray
    pca_lateral_candidates: np.ndarray


@dataclass(frozen=True)
class MorphologyRecord:
    identity: str
    label: int
    node_id: int
    t: int
    features: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json",
    )
    parser.add_argument(
        "--fragments",
        type=Path,
        default=ROOT / "kaggle_kernel/bhcv57out_v3/fragments",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "kaggle_kernel/bhcv57out_v3/cache",
    )
    parser.add_argument(
        "--baseline-ledger",
        type=Path,
        default=ROOT / "reports/g02/graph_cost_sweeps_holdout57_v1.json",
    )
    parser.add_argument(
        "--division-provenance",
        type=Path,
        default=ROOT / "reports/g02/division_failure_holdout57_postpatch_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/g02/phase1_e4_e5_holdout57_v1.json",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FalsificationError(f"JSON root is not an object: {path}")
    return value


def prediction_rows(graph: SubmissionGraph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"node_id": n.node_id, "t": n.t, "z": n.z, "y": n.y, "x": n.x}
        for n in graph.nodes.values()
    ]
    edges = [
        {"source_id": e.source_id, "target_id": e.target_id}
        for e in graph.edges
    ]
    return nodes, edges


def load_scope(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    for path in (args.archive, args.split, args.fragments, args.cache, args.baseline_ledger, args.division_provenance):
        if not path.exists():
            raise FalsificationError(f"missing required active input: {path}")
    if args.archive.stat().st_size != EXPECTED_ZIP_BYTES:
        raise FalsificationError(
            f"raw ZIP size changed: {args.archive.stat().st_size} != {EXPECTED_ZIP_BYTES}"
        )
    split_hash = sha256_file(args.split)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise FalsificationError(f"fold split hash changed: {split_hash}")
    split = load_json(args.split)
    fold = split.get("folds", {}).get("0")
    if not isinstance(fold, dict):
        raise FalsificationError("fold 0 is absent from the split")
    selected_58 = [str(value) for value in fold.get("held_out_video_ids", ())]
    if len(selected_58) != 58 or EXCLUDED_IDENTITY not in selected_58:
        raise FalsificationError("fold-0 held-out selection is not the frozen 58-video set")
    identities = [value for value in selected_58 if value != EXCLUDED_IDENTITY]
    if len(identities) != EXPECTED["videos"]:
        raise FalsificationError("exact baseline scope is not 57 videos")

    observed_files = {path.name for path in args.fragments.glob("train__*.csv")}
    expected_files = {f"train__{value.split(':', 1)[1]}.csv" for value in identities}
    if observed_files != expected_files:
        raise FalsificationError(
            f"fragment selection mismatch: missing={sorted(expected_files-observed_files)} "
            f"extra={sorted(observed_files-expected_files)}"
        )
    expected_cache = {name.removesuffix(".csv") + ".bhc" for name in expected_files}
    observed_cache = {path.name for path in args.cache.glob("train__*.bhc")}
    if observed_cache != expected_cache:
        raise FalsificationError(
            f"raw detection cache selection mismatch: missing={sorted(expected_cache-observed_cache)} "
            f"extra={sorted(observed_cache-expected_cache)}"
        )

    ledger = load_json(args.baseline_ledger)
    baseline = ledger.get("baseline", {})
    frozen = {
        "adjusted_edge_jaccard": float(baseline.get("adj_edge_jaccard", float("nan"))),
        "edge_tp": int(baseline.get("edge_tp", -1)),
        "edge_fp": int(baseline.get("edge_fp", -1)),
        "edge_fn": int(baseline.get("edge_fn", -1)),
    }
    expected_frozen = {
        "adjusted_edge_jaccard": EXPECTED["adjusted_edge_jaccard"],
        "edge_tp": EXPECTED["edge_tp"],
        "edge_fp": EXPECTED["edge_fp"],
        "edge_fn": EXPECTED["edge_fn"],
    }
    if frozen != expected_frozen:
        raise FalsificationError(f"baseline ledger changed: {frozen}")

    division = load_json(args.division_provenance)
    inputs = division.get("inputs", {})
    summary = division.get("summary", {})
    if int(summary.get("gt_divisions", -1)) != EXPECTED["gt_divisions"]:
        raise FalsificationError("division provenance no longer contains 42 GT events")
    if int(inputs.get("zip", {}).get("bytes", -1)) != EXPECTED_ZIP_BYTES:
        raise FalsificationError("division provenance points to a different raw ZIP")
    if Path(str(inputs.get("fragments", ""))).resolve() != args.fragments.resolve():
        raise FalsificationError("division provenance points to different fragments")

    fragment_hashes = {
        identity: sha256_file(args.fragments / f"train__{identity.split(':', 1)[1]}.csv")
        for identity in identities
    }
    cache_records: dict[str, dict[str, Any]] = {}
    for identity in identities:
        stem = identity.replace(":", "__", 1)
        meta_path = args.cache / f"{stem}.json"
        meta = load_json(meta_path)
        cache_records[identity] = {
            "path": str(args.cache / f"{stem}.bhc"),
            "sha256_from_metadata_verified_during_load": meta.get("cache_sha256"),
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
        }
    provenance = {
        "git_pre_edit_checkpoint": "d6c5ace",
        "scope": "fold-0 held-out 58 minus frozen excluded identity",
        "excluded_identity": EXCLUDED_IDENTITY,
        "selected_video_ids": identities,
        "split": {"path": str(args.split), "sha256": split_hash},
        "archive": {
            "path": str(args.archive),
            "bytes": args.archive.stat().st_size,
            "sha256_from_verified_active_provenance": inputs["zip"]["sha256"],
            "full_sha256_recomputed": False,
        },
        "fragments": {"root": str(args.fragments), "sha256_by_identity": fragment_hashes},
        "raw_detection_cache": {
            "root": str(args.cache),
            "geometry_role": "pre-smoothing FrameDet coordinates used by the association gate",
            "records": cache_records,
        },
        "baseline_ledger": {
            "path": str(args.baseline_ledger),
            "sha256": sha256_file(args.baseline_ledger),
            **frozen,
        },
        "division_provenance": {
            "path": str(args.division_provenance),
            "sha256": sha256_file(args.division_provenance),
        },
    }
    return identities, provenance


def load_detections(args: argparse.Namespace, identity: str) -> tuple[bh_model.FrameDet, ...]:
    stem = identity.replace(":", "__", 1)
    cache_path = args.cache / f"{stem}.bhc"
    meta_path = args.cache / f"{stem}.json"
    meta = load_json(meta_path)
    if meta.get("schema") != "celltrack-alpha.bh-cache-meta.v1":
        raise FalsificationError(f"unexpected raw cache metadata schema: {meta_path}")
    observed_sha = sha256_file(cache_path)
    if observed_sha != meta.get("cache_sha256"):
        raise FalsificationError(f"raw cache hash mismatch: {cache_path}")
    try:
        model = meta["contract"]["model"]
        expected_identity = bh_model.CacheIdentity(
            config_sha256=str(model["config_sha256"]),
            source_sha256=str(model["source_sha256"]),
            weight_sha256=str(model["weight_sha256"]),
            pair_batch_size=int(model["pair_batch_size"]),
            view_batch_size=int(model["view_batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FalsificationError(f"invalid raw cache identity: {meta_path}") from exc
    return bh_model.load_sparse_cache(cache_path, expected_identity=expected_identity).detections


def load_matched_video(store: ZipStore, args: argparse.Namespace, identity: str) -> MatchedVideo:
    dataset = identity.split(":", 1)[1]
    fragment = args.fragments / f"train__{dataset}.csv"
    graphs = read_submission(fragment)
    if set(graphs) != {dataset}:
        raise FalsificationError(f"fragment identity mismatch: {fragment}")
    graph = graphs[dataset]
    gt = read_geff_arrays(store, "train", dataset)
    group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
    scale = tuple(float(v) for v in parse_scale(dict(group.attrs)))
    node_rows, edge_rows = prediction_rows(graph)
    prediction = prediction_to_graph(node_rows, edge_rows)
    truth = geff_arrays_to_graph(gt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _evaluate(prediction, truth, "jaccard", scale, MATCH_UM)
    edge_attrs = _evaluate_matched_graph(prediction, truth)
    official_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    official_valid_pred = int(edge_attrs["pred_valid"].sum())
    official_counts = (
        official_tp,
        official_valid_pred - official_tp,
        truth.num_edges() - official_tp,
    )

    ordered_original_ids = [
        int(row["node_id"])
        for row in sorted(node_rows, key=lambda row: (int(row["t"]), int(row["node_id"])))
    ]
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    matched = prediction.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    for pred_internal, gt_internal in matched.iter_rows():
        if gt_internal is None or int(gt_internal) == -1:
            continue
        pred_original = ordered_original_ids[int(pred_internal)]
        pred_to_gt[pred_original] = int(gt_internal)
        if int(gt_internal) in gt_to_pred:
            raise FalsificationError(f"non-unique official match in {identity}")
        gt_to_pred[int(gt_internal)] = pred_original

    truth_nodes = truth.node_attrs().select("node_id", "t", "z", "y", "x")
    gt_nodes = {
        int(node_id): (int(t), float(z), float(y), float(x))
        for node_id, t, z, y, x in truth_nodes.iter_rows()
    }
    gt_edges = tuple(
        sorted(
            (int(source), int(target))
            for source, target in truth.edge_attrs().select("source_id", "target_id").iter_rows()
        )
    )
    pred_edges = frozenset((int(edge.source_id), int(edge.target_id)) for edge in graph.edges)
    detections = load_detections(args, identity)
    return MatchedVideo(
        identity=identity,
        graph=graph,
        gt=gt,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        pred_to_gt=pred_to_gt,
        gt_to_pred=gt_to_pred,
        pred_edges=pred_edges,
        scale=scale,
        official_edge_counts=official_counts,
        detections=detections,
    )


def build_edge_cohorts(video: MatchedVideo) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_coords: dict[int, np.ndarray] = {}
    raw_times: dict[int, int] = {}
    by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in video.detections:
        ids = np.asarray(frame.node_ids, dtype=np.int64)
        coords = np.asarray(frame.coords_zyx, dtype=np.float64)
        if len(ids) != len(coords) or coords.shape != (len(ids), 3):
            raise FalsificationError(f"invalid FrameDet arrays in {video.identity}")
        by_frame[int(frame.frame_index)] = (ids, coords)
        for row, node_id in enumerate(ids):
            if int(node_id) in raw_coords:
                raise FalsificationError(f"duplicate raw node ID in {video.identity}")
            raw_coords[int(node_id)] = coords[row]
            raw_times[int(node_id)] = int(frame.frame_index)
    scale = np.asarray(video.scale, dtype=np.float64)
    trees = {
        t: (cKDTree(coords * scale[None, :]), ids, coords)
        for t, (ids, coords) in by_frame.items()
    }

    rows: list[dict[str, Any]] = []
    audit = Counter()
    for gt_source, gt_target in video.gt_edges:
        pred_source = video.gt_to_pred.get(gt_source)
        pred_target = video.gt_to_pred.get(gt_target)
        selected = (
            pred_source is not None
            and pred_target is not None
            and (pred_source, pred_target) in video.pred_edges
        )
        kind = "tp" if selected else "fn"
        audit[kind] += 1
        if pred_source is None or pred_target is None:
            audit[f"{kind}_official_unmatched_endpoint"] += 1
            continue
        if pred_source not in raw_coords or pred_target not in raw_coords:
            audit[f"{kind}_synthetic_or_pruned_raw_endpoint"] += 1
            continue
        source_t = raw_times[pred_source]
        target_t = raw_times[pred_target]
        gt_source_t = video.gt_nodes[gt_source][0]
        gt_target_t = video.gt_nodes[gt_target][0]
        if target_t != source_t + 1 or gt_target_t != gt_source_t + 1:
            audit[f"{kind}_nonadjacent"] += 1
            continue
        tree, ids, coords = trees[target_t]
        source_physical = raw_coords[pred_source] * scale
        indices = tree.query_ball_point(source_physical, r=GATE_UM + 1.0e-12)
        candidates = []
        for index in indices:
            node_id = int(ids[int(index)])
            coord = coords[int(index)]
            candidates.append((float(np.linalg.norm(coord * scale - source_physical)), node_id, coord))
        candidates.sort(key=lambda item: (item[0], item[1]))
        candidate_ids = [node_id for _, node_id, _ in candidates]
        raw_distance = float(np.linalg.norm((raw_coords[pred_target] - raw_coords[pred_source]) * scale))
        final_source = np.asarray(video.graph.nodes[pred_source].coord_zyx, dtype=np.float64)
        final_target = np.asarray(video.graph.nodes[pred_target].coord_zyx, dtype=np.float64)
        final_distance = float(np.linalg.norm((final_target - final_source) * scale))
        audit[f"{kind}_eligible_raw_endpoint"] += 1
        audit[f"{kind}_raw_{'in_gate' if raw_distance <= GATE_UM + 1e-12 else 'above_gate'}"] += 1
        audit[f"{kind}_final_{'in_gate' if final_distance <= GATE_UM + 1e-12 else 'above_gate'}_among_raw_eligible"] += 1
        transition = (
            ("in" if raw_distance <= GATE_UM + 1e-12 else "above")
            + "_raw_to_"
            + ("in" if final_distance <= GATE_UM + 1e-12 else "above")
            + "_final"
        )
        audit[f"{kind}_{transition}"] += 1
        if pred_target not in candidate_ids:
            audit[f"{kind}_above_gate"] += 1
            if raw_distance <= GATE_UM + 1.0e-9:
                raise FalsificationError(f"KD-tree gate inconsistency in {video.identity}")
            continue
        audit[f"{kind}_in_gate"] += 1
        audit[f"{kind}_in_gate_candidates_{len(candidates)}"] += 1
        rows.append(
            {
                "identity": video.identity,
                "kind": kind,
                "source_frame": source_t,
                "pred_source": pred_source,
                "pred_target": pred_target,
                "source_coord_zyx": raw_coords[pred_source].tolist(),
                "candidate_ids": candidate_ids,
                "candidate_coords_zyx": [coord.tolist() for _, _, coord in candidates],
                "candidate_distances": [distance for distance, _, _ in candidates],
                "true_index": candidate_ids.index(pred_target),
            }
        )
    return rows, dict(audit)


def extract_patch(frame: np.ndarray, center_zyx: Sequence[float], shape: Sequence[int] = PATCH_SHAPE) -> np.ndarray:
    """Extract a fixed patch with reflected boundary padding."""

    if frame.ndim != 3 or len(shape) != 3:
        raise ValueError("frame and patch shape must be 3-D")
    center = np.rint(np.asarray(center_zyx, dtype=np.float64)).astype(int)
    shape_array = np.asarray(shape, dtype=int)
    starts = center - shape_array // 2
    ends = starts + shape_array
    slices = tuple(slice(max(0, int(start)), min(int(limit), int(end))) for start, end, limit in zip(starts, ends, frame.shape, strict=True))
    patch = np.asarray(frame[slices], dtype=np.float32)
    padding = tuple(
        (max(0, -int(start)), max(0, int(end) - int(limit)))
        for start, end, limit in zip(starts, ends, frame.shape, strict=True)
    )
    if any(left or right for left, right in padding):
        mode = "reflect" if all(size > 1 for size in patch.shape) else "edge"
        patch = np.pad(patch, padding, mode=mode)
    if patch.shape != tuple(shape_array):
        raise FalsificationError(f"patch shape mismatch: {patch.shape} != {tuple(shape_array)}")
    return patch


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - float(np.mean(values))
    norm = float(np.linalg.norm(centered.ravel()))
    if norm <= 1.0e-12:
        return np.zeros(centered.size, dtype=np.float32)
    return (centered / norm).ravel().astype(np.float32, copy=False)


def block_mean_3d(patch: np.ndarray) -> np.ndarray:
    if patch.shape != PATCH_SHAPE:
        raise ValueError(f"unexpected patch shape: {patch.shape}")
    reduced = patch.reshape(4, 4, 8, 4, 8, 4).mean(axis=(1, 3, 5))
    return unit_vector(reduced)


def lateral_projection(patch: np.ndarray) -> np.ndarray:
    return np.max(patch, axis=0)


def block_mean_lateral(projection: np.ndarray) -> np.ndarray:
    if projection.shape != PATCH_SHAPE[1:]:
        raise ValueError(f"unexpected lateral shape: {projection.shape}")
    reduced = projection.reshape(16, 2, 16, 2).mean(axis=(1, 3))
    return unit_vector(reduced)


def appearance_patch_features(frame: np.ndarray, center: Sequence[float]) -> dict[str, np.ndarray]:
    patch = extract_patch(frame, center)
    lateral = lateral_projection(patch)
    return {
        "ncc_3d": unit_vector(patch),
        "ncc_lateral": unit_vector(lateral),
        "pca_3d": block_mean_3d(patch),
        "pca_lateral": block_mean_lateral(lateral),
    }


def extract_appearance_events(
    array: zarr.Array,
    video: MatchedVideo,
    rows: Sequence[Mapping[str, Any]],
) -> list[AppearanceEvent]:
    eligible = [
        row
        for row in rows
        if 2 <= len(row["candidate_ids"]) <= 4
        or (row["kind"] == "fn" and len(row["candidate_ids"]) >= 1)
    ]
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[int(row["source_frame"])].append(row)

    events: list[AppearanceEvent] = []
    for source_t in sorted(grouped):
        source_frame = np.asarray(array[source_t])
        target_frame = np.asarray(array[source_t + 1])
        source_ids = sorted({int(row["pred_source"]) for row in grouped[source_t]})
        candidate_ids = sorted(
            {int(node_id) for row in grouped[source_t] for node_id in row["candidate_ids"]}
        )
        source_features = {
            node_id: appearance_patch_features(
                source_frame,
                next(row["source_coord_zyx"] for row in grouped[source_t] if int(row["pred_source"]) == node_id),
            )
            for node_id in source_ids
        }
        candidate_coord_by_id = {
            int(node_id): coord
            for row in grouped[source_t]
            for node_id, coord in zip(row["candidate_ids"], row["candidate_coords_zyx"], strict=True)
        }
        candidate_features = {
            node_id: appearance_patch_features(target_frame, candidate_coord_by_id[node_id])
            for node_id in candidate_ids
        }
        for row in grouped[source_t]:
            source = source_features[int(row["pred_source"])]
            candidates = [candidate_features[int(node_id)] for node_id in row["candidate_ids"]]
            events.append(
                AppearanceEvent(
                    identity=video.identity,
                    kind=str(row["kind"]),
                    candidate_count=len(candidates),
                    true_index=int(row["true_index"]),
                    distance_scores=-np.asarray(row["candidate_distances"], dtype=np.float32),
                    ncc_3d_scores=np.asarray(
                        [float(np.dot(source["ncc_3d"], item["ncc_3d"])) for item in candidates],
                        dtype=np.float32,
                    ),
                    ncc_lateral_scores=np.asarray(
                        [float(np.dot(source["ncc_lateral"], item["ncc_lateral"])) for item in candidates],
                        dtype=np.float32,
                    ),
                    pca_3d_source=source["pca_3d"],
                    pca_3d_candidates=np.stack([item["pca_3d"] for item in candidates]),
                    pca_lateral_source=source["pca_lateral"],
                    pca_lateral_candidates=np.stack([item["pca_lateral"] for item in candidates]),
                )
            )
    return events


def make_identity_folds(identities: Iterable[str], folds: int = CV_FOLDS) -> dict[str, int]:
    ordered = sorted(set(str(value) for value in identities), key=lambda value: (value.split(":", 1)[1][:4], value))
    return {identity: index % folds for index, identity in enumerate(ordered)}


def fit_pca(observations: Sequence[np.ndarray], components: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    if not observations:
        raise FalsificationError("PCA received no training observations")
    matrix = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1, 256) for value in observations], axis=0)
    original_n = int(matrix.shape[0])
    if matrix.shape[0] > PCA_TRAIN_CAP:
        chosen = np.sort(rng.choice(matrix.shape[0], size=PCA_TRAIN_CAP, replace=False))
        matrix = matrix[chosen]
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = matrix - mean[None, :]
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    keep = np.argsort(eigenvalues)[::-1][: min(components, matrix.shape[1], matrix.shape[0] - 1)]
    return mean, eigenvectors[:, keep].astype(np.float32), original_n


def pca_similarity(source: np.ndarray, candidates: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    source_z = (source - mean) @ basis
    candidate_z = (candidates - mean[None, :]) @ basis
    source_norm = max(float(np.linalg.norm(source_z)), 1.0e-12)
    candidate_norm = np.maximum(np.linalg.norm(candidate_z, axis=1), 1.0e-12)
    return (candidate_z @ source_z / (candidate_norm * source_norm)).astype(np.float32)


def score_pca_cross_validated(events: Sequence[AppearanceEvent], mode: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if mode not in {"3d", "lateral"}:
        raise ValueError(mode)
    folds = make_identity_folds(event.identity for event in events)
    scores = np.full(len(events), -1, dtype=np.int8)
    details: list[dict[str, Any]] = []
    for fold in range(CV_FOLDS):
        train = [event for event in events if folds[event.identity] != fold]
        test_indices = [index for index, event in enumerate(events) if folds[event.identity] == fold]
        observations: list[np.ndarray] = []
        for event in train:
            observations.append(getattr(event, f"pca_{mode}_source"))
            observations.append(getattr(event, f"pca_{mode}_candidates"))
        mean, basis, original_n = fit_pca(
            observations,
            PCA_COMPONENTS,
            np.random.default_rng(RNG_SEED + fold + (100 if mode == "lateral" else 0)),
        )
        for index in test_indices:
            event = events[index]
            similarity = pca_similarity(
                getattr(event, f"pca_{mode}_source"),
                getattr(event, f"pca_{mode}_candidates"),
                mean,
                basis,
            )
            scores[index] = int(np.argmax(similarity) == event.true_index)
        details.append(
            {
                "fold": fold,
                "held_out_identities": sorted(identity for identity, value in folds.items() if value == fold),
                "train_events": len(train),
                "train_patch_observations_before_cap": original_n,
                "train_patch_observations_used": min(original_n, PCA_TRAIN_CAP),
                "test_events": len(test_indices),
                "components": int(basis.shape[1]),
            }
        )
    if np.any(scores < 0):
        raise FalsificationError(f"PCA {mode} left events unscored")
    return scores.astype(bool), details


def percentile_ci(values: Sequence[float]) -> list[float]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def clustered_accuracy_summary(
    identities: Sequence[str],
    correctness: Mapping[str, np.ndarray],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    identity_array = np.asarray(identities, dtype=object)
    groups = np.asarray(sorted(set(identities)), dtype=object)
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {name: [] for name in correctness}
    delta_boot: dict[str, list[float]] = {name: [] for name in correctness if name != "distance"}
    for _ in range(bootstrap):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(identity_array == group) for group in chosen])
        distance = float(np.mean(correctness["distance"][indices]))
        for name, values in correctness.items():
            accuracy = float(np.mean(values[indices]))
            boot[name].append(accuracy)
            if name != "distance":
                delta_boot[name].append(accuracy - distance)
    result: dict[str, Any] = {}
    distance_point = float(np.mean(correctness["distance"]))
    for name, values in correctness.items():
        point = float(np.mean(values))
        row: dict[str, Any] = {
            "correct": int(np.sum(values)),
            "n": int(len(values)),
            "top1_accuracy": point,
            "cluster_bootstrap_95ci": percentile_ci(boot[name]),
        }
        if name != "distance":
            row["delta_vs_distance"] = point - distance_point
            row["delta_cluster_bootstrap_95ci"] = percentile_ci(delta_boot[name])
        result[name] = row
    return result


def summarize_e4(events: Sequence[AppearanceEvent], bootstrap: int) -> dict[str, Any]:
    pca_3d, pca_3d_folds = score_pca_cross_validated(events, "3d")
    pca_lateral, pca_lateral_folds = score_pca_cross_validated(events, "lateral")
    correctness = {
        "distance": np.asarray([np.argmax(event.distance_scores) == event.true_index for event in events]),
        "ncc_3d": np.asarray([np.argmax(event.ncc_3d_scores) == event.true_index for event in events]),
        "ncc_lateral": np.asarray([np.argmax(event.ncc_lateral_scores) == event.true_index for event in events]),
        "pca_3d": pca_3d,
        "pca_lateral": pca_lateral,
    }
    subsets = {
        "confusable_2_to_4_all_edges": np.asarray([2 <= event.candidate_count <= 4 for event in events]),
        "confusable_2_to_4_fn": np.asarray([event.kind == "fn" and 2 <= event.candidate_count <= 4 for event in events]),
        "exact_in_gate_fn_all_candidate_counts": np.asarray([event.kind == "fn" for event in events]),
        "exact_in_gate_fn_multicandidate": np.asarray([event.kind == "fn" and event.candidate_count >= 2 for event in events]),
    }
    summaries: dict[str, Any] = {}
    for subset_index, (name, mask) in enumerate(subsets.items()):
        if not np.any(mask):
            summaries[name] = {"n": 0}
            continue
        summaries[name] = clustered_accuracy_summary(
            [event.identity for event, keep in zip(events, mask, strict=True) if keep],
            {method: values[mask] for method, values in correctness.items()},
            bootstrap,
            RNG_SEED + subset_index,
        )

    primary = summaries["confusable_2_to_4_all_edges"]
    appearance_methods = ("ncc_3d", "ncc_lateral", "pca_3d", "pca_lateral")
    passing = [
        method
        for method in appearance_methods
        if primary[method]["delta_vs_distance"] > 0.0
        and primary[method]["delta_cluster_bootstrap_95ci"][0] > 0.0
    ]
    decision = "PASS" if passing else "FAIL"
    return {
        "experiment": "E4 appearance separability",
        "decision": decision,
        "decision_rule": (
            "PASS only if at least one predeclared appearance ranker has higher top-1 accuracy "
            "than physical distance on all 2-4-candidate confusable edges and the paired "
            "video-cluster bootstrap 95% CI for the improvement excludes zero. Otherwise FAIL."
        ),
        "passing_methods": passing,
        "patch_shape_zyx": list(PATCH_SHAPE),
        "gate_um": GATE_UM,
        "pca_contract": {
            "descriptor": "per-patch normalized 4x8x8 block means (3D) or 16x16 lateral-max block means",
            "components": PCA_COMPONENTS,
            "training_cap_per_fold": PCA_TRAIN_CAP,
            "leakage_control": "five-fold video-grouped CV; PCA mean/basis fit on training-video patches only",
            "folds_3d": pca_3d_folds,
            "folds_lateral": pca_lateral_folds,
        },
        "ncc_contract": {
            "3d": "zero-mean unit-norm NCC over the full 16x32x32 centroid-aligned patch",
            "lateral": "zero-mean unit-norm NCC over the axial-max 32x32 projection",
        },
        "subsets": summaries,
        "candidate_count_distribution": dict(sorted(Counter(event.candidate_count for event in events).items())),
    }


def sphere_moments(
    frame: np.ndarray,
    center_zyx: Sequence[float],
    scale_zyx: Sequence[float],
    radius_um: float = SPHERE_RADIUS_UM,
    shell_radius_um: float = SHELL_RADIUS_UM,
) -> dict[str, float]:
    center = np.asarray(center_zyx, dtype=np.float64)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    half = np.ceil(shell_radius_um / scale).astype(int)
    rounded = np.rint(center).astype(int)
    starts = np.maximum(0, rounded - half)
    ends = np.minimum(np.asarray(frame.shape), rounded + half + 1)
    slices = tuple(slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True))
    local = np.asarray(frame[slices], dtype=np.float64)
    axes = [np.arange(start, end, dtype=np.float64) for start, end in zip(starts, ends, strict=True)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    delta = np.stack((zz - center[0], yy - center[1], xx - center[2]), axis=-1) * scale
    distance = np.linalg.norm(delta, axis=-1)
    sphere = distance <= radius_um + 1.0e-12
    shell = (distance > radius_um) & (distance <= shell_radius_um + 1.0e-12)
    values = local[sphere]
    if values.size == 0:
        raise FalsificationError("empty physical sphere")
    background = float(np.median(local[shell])) if np.any(shell) else float(np.median(values))
    weights = np.clip(values - background, 0.0, None)
    points = delta[sphere]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0e-12:
        covariance = np.zeros((3, 3), dtype=np.float64)
    else:
        centroid = np.sum(points * weights[:, None], axis=0) / weight_sum
        centered = points - centroid[None, :]
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigen = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    largest = max(float(eigen[-1]), 1.0e-12)
    mean = float(np.mean(values))
    variance = float(np.var(values))
    return {
        "raw_sum": float(np.sum(values)),
        "excess_sum": weight_sum,
        "mean": mean,
        "variance": variance,
        "relative_variance": variance / max(mean * mean, 1.0e-12),
        "eig_min_over_max": float(eigen[0]) / largest,
        "eig_mid_over_max": float(eigen[1]) / largest,
        "anisotropy": float(eigen[-1] - eigen[0]) / max(float(np.sum(eigen)), 1.0e-12),
        "background": background,
        "voxel_count": int(values.size),
    }


MORPH_FEATURE_NAMES = (
    "log_raw_sum_t",
    "log_excess_sum_t",
    "log_variance_t",
    "log_relative_variance_t",
    "eig_min_over_max_t",
    "eig_mid_over_max_t",
    "anisotropy_t",
    "log_raw_sum_ratio_t1_t",
    "log_excess_sum_ratio_t1_t",
    "log_variance_ratio_t1_t",
    "log_relative_variance_ratio_t1_t",
    "delta_eig_min_over_max",
    "delta_eig_mid_over_max",
    "delta_anisotropy",
)


def safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log((max(float(numerator), 0.0) + 1.0) / (max(float(denominator), 0.0) + 1.0)))


def morphology_feature_vector(parent: Mapping[str, float], future: Mapping[str, float]) -> tuple[float, ...]:
    return (
        float(np.log1p(max(parent["raw_sum"], 0.0))),
        float(np.log1p(max(parent["excess_sum"], 0.0))),
        float(np.log1p(max(parent["variance"], 0.0))),
        float(np.log1p(max(parent["relative_variance"], 0.0))),
        float(parent["eig_min_over_max"]),
        float(parent["eig_mid_over_max"]),
        float(parent["anisotropy"]),
        safe_log_ratio(future["raw_sum"], parent["raw_sum"]),
        safe_log_ratio(future["excess_sum"], parent["excess_sum"]),
        safe_log_ratio(future["variance"], parent["variance"]),
        safe_log_ratio(future["relative_variance"], parent["relative_variance"]),
        float(future["eig_min_over_max"] - parent["eig_min_over_max"]),
        float(future["eig_mid_over_max"] - parent["eig_mid_over_max"]),
        float(future["anisotropy"] - parent["anisotropy"]),
    )


def select_morphology_nodes(video: MatchedVideo, controls_per_positive: int = 4) -> list[dict[str, Any]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in video.gt_edges:
        outgoing[source].append(target)
    positives: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for source, targets in sorted(outgoing.items()):
        source_meta = video.gt_nodes[source]
        adjacent = [target for target in sorted(targets) if video.gt_nodes[target][0] == source_meta[0] + 1]
        if len(targets) == 2 and len(adjacent) == 2:
            positives.append({"label": 1, "source": source, "targets": adjacent, "t": source_meta[0]})
        elif len(targets) == 1 and len(adjacent) == 1:
            controls.append({"label": 0, "source": source, "targets": adjacent, "t": source_meta[0]})
    selected = list(positives)
    unused = set(range(len(controls)))
    for positive in sorted(positives, key=lambda row: (row["t"], row["source"])):
        ranked = sorted(
            unused,
            key=lambda index: (
                abs(int(controls[index]["t"]) - int(positive["t"])),
                int(controls[index]["t"]),
                int(controls[index]["source"]),
            ),
        )
        chosen = ranked[:controls_per_positive]
        for index in chosen:
            unused.remove(index)
            selected.append(controls[index])
    return selected


def extract_morphology_records(
    array: zarr.Array,
    video: MatchedVideo,
    selections: Sequence[Mapping[str, Any]],
) -> list[MorphologyRecord]:
    by_time: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        by_time[int(row["t"])].append(row)
    records: list[MorphologyRecord] = []
    for t in sorted(by_time):
        frame_t = np.asarray(array[t])
        frame_t1 = np.asarray(array[t + 1])
        for row in by_time[t]:
            source = video.gt_nodes[int(row["source"])]
            targets = [video.gt_nodes[int(target)] for target in row["targets"]]
            parent_center = source[1:]
            future_center = np.mean(np.asarray([target[1:] for target in targets], dtype=np.float64), axis=0)
            parent = sphere_moments(frame_t, parent_center, video.scale)
            future = sphere_moments(frame_t1, future_center, video.scale)
            records.append(
                MorphologyRecord(
                    identity=video.identity,
                    label=int(row["label"]),
                    node_id=int(row["source"]),
                    t=t,
                    features=morphology_feature_vector(parent, future),
                )
            )
    return records


def balanced_group_folds(records: Sequence[MorphologyRecord], folds: int = CV_FOLDS) -> dict[str, int]:
    counts = Counter(record.identity for record in records if record.label == 1)
    if len(counts) < folds:
        raise FalsificationError(f"only {len(counts)} positive videos for {folds}-fold grouped CV")
    fold_positive = [0] * folds
    fold_groups = [0] * folds
    assignment: dict[str, int] = {}
    for identity, positive_count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(folds), key=lambda index: (fold_positive[index], fold_groups[index], index))
        assignment[identity] = fold
        fold_positive[fold] += positive_count
        fold_groups[fold] += 1
    return assignment


def fit_logistic_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    z = (x - mean[None, :]) / scale[None, :]
    design = np.column_stack((np.ones(len(z)), z))
    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    sample_weight = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss = np.sum(sample_weight * (np.logaddexp(0.0, logits) - y * logits)) / len(y)
        penalty = 0.5 * ridge * float(np.dot(beta[1:], beta[1:])) / len(y)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gradient = design.T @ (sample_weight * (probabilities - y)) / len(y)
        gradient[1:] += ridge * beta[1:] / len(y)
        return float(loss + penalty), gradient

    result = minimize(
        lambda beta: objective(beta),
        np.zeros(design.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1_000, "ftol": 1.0e-12},
    )
    if not result.success:
        raise FalsificationError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x), mean, scale


def logistic_predict(x: np.ndarray, beta: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), (x - mean[None, :]) / scale[None, :]))
    logits = design @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = y == 1
    negative = y == 0
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    ranks = rankdata(score, method="average")
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    prediction = score >= 0.5
    sensitivity = float(np.mean(prediction[y == 1])) if np.any(y == 1) else float("nan")
    specificity = float(np.mean(~prediction[y == 0])) if np.any(y == 0) else float("nan")
    return float((sensitivity + specificity) / 2.0)


def summarize_e5(records: Sequence[MorphologyRecord], bootstrap: int) -> dict[str, Any]:
    x = np.asarray([record.features for record in records], dtype=np.float64)
    y = np.asarray([record.label for record in records], dtype=np.int8)
    identities = np.asarray([record.identity for record in records], dtype=object)
    folds = balanced_group_folds(records)
    oof = np.full(len(records), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(CV_FOLDS):
        test = np.asarray([folds[identity] == fold for identity in identities])
        train = ~test
        beta, mean, scale = fit_logistic_ridge(x[train], y[train], ridge=1.0)
        oof[test] = logistic_predict(x[test], beta, mean, scale)
        fold_rows.append(
            {
                "fold": fold,
                "held_out_identities": sorted(identity for identity, value in folds.items() if value == fold),
                "train_n": int(np.sum(train)),
                "test_n": int(np.sum(test)),
                "test_positive": int(np.sum(y[test] == 1)),
                "test_negative": int(np.sum(y[test] == 0)),
                "roc_auc": roc_auc(y[test], oof[test]),
                "balanced_accuracy": balanced_accuracy(y[test], oof[test]),
            }
        )
    if np.any(~np.isfinite(oof)):
        raise FalsificationError("grouped CV left morphology records unscored")

    point_auc = roc_auc(y, oof)
    point_balanced = balanced_accuracy(y, oof)
    groups = np.asarray(sorted(set(identities)), dtype=object)
    rng = np.random.default_rng(RNG_SEED + 500)
    boot_auc: list[float] = []
    boot_balanced: list[float] = []
    for _ in range(bootstrap):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(identities == identity) for identity in chosen])
        boot_auc.append(roc_auc(y[indices], oof[indices]))
        boot_balanced.append(balanced_accuracy(y[indices], oof[indices]))
    auc_ci = percentile_ci(boot_auc)
    balanced_ci = percentile_ci(boot_balanced)

    univariate = []
    for index, name in enumerate(MORPH_FEATURE_NAMES):
        raw_auc = roc_auc(y, x[:, index])
        univariate.append(
            {
                "feature": name,
                "roc_auc": raw_auc,
                "orientation_free_roc_auc": max(raw_auc, 1.0 - raw_auc),
                "positive_mean": float(np.mean(x[y == 1, index])),
                "negative_mean": float(np.mean(x[y == 0, index])),
            }
        )
    univariate.sort(key=lambda row: (-row["orientation_free_roc_auc"], row["feature"]))

    decision = (
        "PASS"
        if point_auc >= 0.70 and auc_ci[0] > 0.60 and point_balanced >= 0.60 and balanced_ci[0] > 0.50
        else "FAIL"
    )
    return {
        "experiment": "E5 division morphology separability",
        "decision": decision,
        "decision_rule": (
            "PASS only if fixed-feature, fixed-ridge video-grouped out-of-fold ROC-AUC is at least "
            "0.70 with video-cluster bootstrap lower 95% bound above 0.60, and balanced accuracy "
            "is at least 0.60 with lower bound above 0.50."
        ),
        "class_counts": {
            "division": int(np.sum(y == 1)),
            "matched_nondivision_controls": int(np.sum(y == 0)),
            "positive_videos": len(set(identities[y == 1])),
            "all_videos": len(set(identities)),
        },
        "feature_names": list(MORPH_FEATURE_NAMES),
        "feature_contract": {
            "sphere_radius_um": SPHERE_RADIUS_UM,
            "background_shell_outer_radius_um": SHELL_RADIUS_UM,
            "future_center": "mean GT successor centroid (two daughters for division; one successor for control)",
            "controls": "up to four without-replacement one-child adjacent nodes per division, same video, nearest frame",
            "masks_or_segmentation": False,
            "classifier": "L2-ridge logistic regression, ridge=1, class-balanced training weights",
            "leakage_control": "five-fold grouped CV by video; standardization and model fit within training videos",
        },
        "grouped_cv": {
            "folds": fold_rows,
            "oof_roc_auc": point_auc,
            "oof_roc_auc_video_cluster_bootstrap_95ci": auc_ci,
            "oof_balanced_accuracy": point_balanced,
            "oof_balanced_accuracy_video_cluster_bootstrap_95ci": balanced_ci,
            "bootstrap_replicates": bootstrap,
        },
        "univariate_features": univariate,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    identities, provenance = load_scope(args)
    edge_rows_by_identity: dict[str, list[dict[str, Any]]] = {}
    edge_audit_by_identity: dict[str, dict[str, int]] = {}
    official_totals = Counter()
    raw_selection_totals = Counter()
    matched_videos: dict[str, MatchedVideo] = {}

    with ZipStore(args.archive, mode="r") as store:
        for identity in identities:
            video = load_matched_video(store, args, identity)
            matched_videos[identity] = video
            tp, fp, fn = video.official_edge_counts
            official_totals.update(edge_tp=tp, edge_fp=fp, edge_fn=fn)
            rows, audit = build_edge_cohorts(video)
            edge_rows_by_identity[identity] = rows
            edge_audit_by_identity[identity] = audit
            raw_selection_totals.update(tp=audit.get("tp", 0), fn=audit.get("fn", 0))

        expected_edge = {key: EXPECTED[key] for key in ("edge_tp", "edge_fp", "edge_fn")}
        if {key: official_totals[key] for key in expected_edge} != expected_edge:
            raise FalsificationError(
                f"exact official baseline accounting changed: {dict(official_totals)} != {expected_edge}"
            )
        if raw_selection_totals["tp"] != EXPECTED["edge_tp"] or raw_selection_totals["fn"] != EXPECTED["edge_fn"]:
            raise FalsificationError(
                f"raw selected-edge accounting differs from official: {dict(raw_selection_totals)}"
            )

        aggregate_audit = Counter()
        for audit in edge_audit_by_identity.values():
            aggregate_audit.update(audit)
        expected_raw_cohort = {
            "fn_eligible_raw_endpoint": 1_549,
            "fn_raw_in_gate": 1_232,
            "fn_raw_above_gate": 317,
            "fn_official_unmatched_endpoint": 464,
            "fn_synthetic_or_pruned_raw_endpoint": 81,
        }
        observed_raw_cohort = {key: aggregate_audit[key] for key in expected_raw_cohort}
        if observed_raw_cohort != expected_raw_cohort:
            raise FalsificationError(
                f"raw FrameDet E1/E4 cohort changed: {observed_raw_cohort} != {expected_raw_cohort}"
            )
        audit_report = {
            "official_edge_totals": dict(official_totals),
            "reconstructed_selected_edge_totals": dict(raw_selection_totals),
            "cohort_counts": dict(sorted(aggregate_audit.items())),
            "coordinate_stage_reconciliation": {
                "cause": (
                    "E1/E4 gate geometry is raw FrameDet. Final CSV postprocessing moved 221 "
                    "common FN pairs across the 10um boundary, and 81 additional officially "
                    "matched FN pairs have a synthetic/pruned endpoint absent from FrameDet."
                ),
                "raw_in_to_final_in": aggregate_audit["fn_in_raw_to_in_final"],
                "raw_in_to_final_above": aggregate_audit["fn_in_raw_to_above_final"],
                "raw_above_to_final_in": aggregate_audit["fn_above_raw_to_in_final"],
                "raw_above_to_final_above": aggregate_audit["fn_above_raw_to_above_final"],
                "fn_raw_endpoint_absent": aggregate_audit["fn_synthetic_or_pruned_raw_endpoint"],
            },
            "per_video": edge_audit_by_identity,
        }
        if args.audit_only:
            return {"schema": SCHEMA, "status": "audit_only", "provenance": provenance, "audit": audit_report}

        appearance_events: list[AppearanceEvent] = []
        morphology_records: list[MorphologyRecord] = []
        morphology_selection_counts: dict[str, dict[str, int]] = {}
        for identity in identities:
            dataset = identity.split(":", 1)[1]
            array = zarr.open_array(store=store, path=f"train/{dataset}.zarr/0", mode="r")
            video = matched_videos[identity]
            appearance_events.extend(extract_appearance_events(array, video, edge_rows_by_identity[identity]))
            selections = select_morphology_nodes(video)
            positive = sum(int(row["label"]) == 1 for row in selections)
            negative = sum(int(row["label"]) == 0 for row in selections)
            if positive or negative:
                morphology_selection_counts[identity] = {"division": positive, "control": negative}
                morphology_records.extend(extract_morphology_records(array, video, selections))

    division_count = sum(record.label for record in morphology_records)
    if division_count != EXPECTED["gt_divisions"]:
        raise FalsificationError(f"E5 extracted {division_count} divisions, expected 42")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "constraints": {
            "cpu_only": True,
            "gpu_used": False,
            "kaggle_action": False,
            "production_training": False,
            "segmenter_or_masks": False,
            "geometry_changed": False,
            "divisions_enabled": False,
        },
        "provenance": provenance,
        "audit": audit_report,
        "e4": summarize_e4(appearance_events, args.bootstrap),
        "e5": {
            **summarize_e5(morphology_records, args.bootstrap),
            "selection_counts_by_identity": morphology_selection_counts,
        },
    }
    return report


def main() -> int:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    if args.output.exists():
        raise FalsificationError(f"refusing to overwrite versioned report: {args.output}")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "e4": report.get("e4", {}).get("decision"), "e5": report.get("e5", {}).get("decision")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
