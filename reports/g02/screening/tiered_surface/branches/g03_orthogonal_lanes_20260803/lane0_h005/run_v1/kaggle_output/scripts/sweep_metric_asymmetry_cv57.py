#!/usr/bin/env python3
"""Measure and exploit edge-metric admission asymmetry on cached CV57 data.

This is an offline diagnostic.  It never runs the detector: detector nodes and
pair contexts come from the frozen ``bhcv57out_v3`` caches.  The submitted
division-off fragments remain the baseline, and experimental edges/nodes are
assembled only in memory for scorer-equivalent counting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import tracksdata as td
import zarr
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
import scoreall  # noqa: E402
import score_bh_cv57  # noqa: E402
import sweep_bh_graph_weights as sweep_support  # noqa: E402
from celltrack_alpha import bh_edge, bh_model  # noqa: E402
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    geff_arrays_to_graph,
    prediction_to_graph,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.metric-asymmetry-cv57.v1"
BASELINE_ADJUSTED = 0.8711885823490648
EXPECTED_BASELINE = {
    "edge_tp": 34_174,
    "edge_fp": 2_736,
    "edge_fn": 2_094,
    "edge_jaccard": 0.8761665470208184,
    "adj_edge_jaccard": BASELINE_ADJUSTED,
}
DEFAULT_THRESHOLDS = (
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
)
DEFAULT_GATES = (6.0, 10.0)
DEFAULT_CAPS = (2, 4, 8)
DEFAULT_NODE_TARGETS = {
    "6bba": (None, 0.10, 0.125, 0.15, 0.175),
    "44b6": (None, -0.10, -0.075, -0.05, -0.025),
}
JOINT_EDGE_FINALISTS = 5

DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_SPLIT = ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
DEFAULT_CACHE = ROOT / "kaggle_kernel/bhcv57out_v3/cache"
DEFAULT_FRAGMENTS = ROOT / "kaggle_kernel/bhcv57out_v3/fragments"
DEFAULT_EDGE = (
    ROOT / "_ARCHIVE/conflict_residual/reports/g02/conflict_edge_f0_fit117_v1/edge.json"
)
DEFAULT_AUXILIARY = ROOT / "_ARCHIVE/compact_head/reports/g02/v45f4.pt"
DEFAULT_SUPPORT = ROOT / "external/Tracking_Support"
DEFAULT_PARITY = ROOT / "reports/g02/scorer_parity_v1.json"
DEFAULT_BASELINE_SCORES = (
    ROOT
    / "_ARCHIVE/conflict_residual/reports/g02/"
    "run_bh_d4_conflict_edge_f0_holdout57_v1/scores_postpatch.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/metric_asymmetry_sweep_cv57_v1.json"
DEFAULT_COARSE_REPORT = ROOT / "reports/g02/metric_asymmetry_sweep_cv57_v1_coarse.json"


class AsymmetrySweepError(RuntimeError):
    """The cached CV57 asymmetry experiment violated its contract."""


@dataclass(frozen=True)
class MatchState:
    nodes: tuple[dict[str, int | float], ...]
    node_time: dict[int, int]
    matched_gt: dict[int, int | None]
    gt_edges: frozenset[tuple[int, int]]
    gt_out_valid: frozenset[int]
    gt_in_valid: frozenset[int]
    gt_edge_count: int
    gt_node_count: int
    estimated_nodes: float
    node_recall: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AsymmetrySweepError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AsymmetrySweepError(f"{label} is not a JSON object: {path}")
    return value


def _family(identity: str) -> str:
    value = identity.split(":", 1)[1].split("_", 1)[0]
    if value not in DEFAULT_NODE_TARGETS:
        raise AsymmetrySweepError(f"unknown family for {identity}")
    return value


def _setting_key(threshold: float, gate_um: float, neighbor_cap: int) -> str:
    return f"p{threshold:g}_r{gate_um:g}_k{neighbor_cap}"


def _node_target_key(value: float | None) -> str:
    return "baseline" if value is None else f"ratio_{value:+.3f}"


def _fragment_graph(path: Path, dataset: str) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    graphs = read_submission(path)
    if set(graphs) != {dataset}:
        raise AsymmetrySweepError(f"fragment identity mismatch: {path}")
    graph = graphs[dataset]
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
    edges = [(int(edge.source_id), int(edge.target_id)) for edge in graph.edges]
    return nodes, edges


def _raw_nodes(cache: Any) -> dict[int, dict[str, int | float]]:
    result: dict[int, dict[str, int | float]] = {}
    for frame in cache.detections:
        for node_id, coord, score in zip(
            frame.node_ids, frame.coords_zyx, frame.scores, strict=True
        ):
            key = int(node_id)
            if key in result:
                raise AsymmetrySweepError("cached detector node IDs are not unique")
            result[key] = {
                "node_id": key,
                "t": int(frame.frame_index),
                "z": float(coord[0]),
                "y": float(coord[1]),
                "x": float(coord[2]),
                "score": float(score),
            }
    return result


def _match_state(
    nodes: Sequence[Mapping[str, Any]],
    gt: dict[str, Any],
    scale_zyx: tuple[float, float, float],
) -> MatchState:
    canonical = tuple(
        {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in sorted(nodes, key=lambda row: (int(row["t"]), int(row["node_id"])))
    )
    pred = prediction_to_graph(list(canonical), [])
    gt_graph = geff_arrays_to_graph(gt)
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        pred.match(gt_graph, matching=DistanceMatching(max_distance=7.0, scale=scale_zyx))
    finally:
        set_options(show_progress=previous)

    pred_attrs = pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    ).sort(td.DEFAULT_ATTR_KEYS.NODE_ID)
    graph_ids = [int(value) for value in pred_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID]]
    if graph_ids != list(range(len(canonical))):
        raise AsymmetrySweepError("prediction graph IDs are not canonical insertion order")
    matched_values = pred_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()
    matched_gt: dict[int, int | None] = {}
    for row, value in zip(canonical, matched_values, strict=True):
        match = None if value is None or int(value) == -1 else int(value)
        matched_gt[int(row["node_id"])] = match

    gt_edge_attrs = gt_graph.edge_attrs()
    sources = gt_edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    targets = gt_edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()
    gt_edges = frozenset((int(source), int(target)) for source, target in zip(sources, targets, strict=True))
    out_valid = frozenset(source for source, _ in gt_edges)
    in_valid = frozenset(target for _, target in gt_edges)
    matched_unique = {value for value in matched_gt.values() if value is not None}
    estimated = float(gt["estimated_number_of_nodes"])
    return MatchState(
        nodes=canonical,
        node_time={int(row["node_id"]): int(row["t"]) for row in canonical},
        matched_gt=matched_gt,
        gt_edges=gt_edges,
        gt_out_valid=out_valid,
        gt_in_valid=in_valid,
        gt_edge_count=len(gt_edges),
        gt_node_count=gt_graph.num_nodes(),
        estimated_nodes=estimated,
        node_recall=len(matched_unique) / gt_graph.num_nodes(),
    )


def classify_edges(
    state: MatchState,
    edges: Sequence[tuple[int, int]],
) -> dict[str, int | float]:
    """Replicate patched edge filtering and expose its ignored-edge classes."""

    seen_pairs: set[tuple[int, int]] = set()
    consecutive: list[tuple[int, int]] = []
    filtered_missing = filtered_duplicate = filtered_nonconsecutive = 0
    for source, target in edges:
        pair = (int(source), int(target))
        if pair[0] not in state.node_time or pair[1] not in state.node_time or pair[0] == pair[1]:
            filtered_missing += 1
            continue
        if pair in seen_pairs:
            filtered_duplicate += 1
            continue
        seen_pairs.add(pair)
        if state.node_time[pair[1]] - state.node_time[pair[0]] != 1:
            filtered_nonconsecutive += 1
            continue
        consecutive.append(pair)

    merge_seen: set[tuple[int, int]] = set()
    collapsed: list[tuple[int, int]] = []
    filtered_merge_duplicate = 0
    for source, target in consecutive:
        matched_source = state.matched_gt[source]
        matched_target = state.matched_gt[target]
        if matched_source is not None and matched_target is not None:
            mapped = (matched_source, matched_target)
            if mapped in merge_seen:
                filtered_merge_duplicate += 1
                continue
            merge_seen.add(mapped)
        collapsed.append((source, target))

    out_counts: Counter[int] = Counter()
    retained: list[tuple[int, int]] = []
    filtered_outdegree = 0
    for source, target in collapsed:
        out_counts[source] += 1
        if out_counts[source] > 2:
            filtered_outdegree += 1
            continue
        retained.append((source, target))

    counts: Counter[str] = Counter()
    for source, target in retained:
        matched_source = state.matched_gt[source]
        matched_target = state.matched_gt[target]
        is_tp = (
            matched_source is not None
            and matched_target is not None
            and (matched_source, matched_target) in state.gt_edges
        )
        if is_tp:
            counts["tp"] += 1
            continue
        case_a = matched_target is not None and matched_target in state.gt_in_valid
        case_b = matched_source is not None and matched_source in state.gt_out_valid
        if case_a and case_b:
            counts["fp_both"] += 1
        elif case_a:
            counts["fp_case_a_only"] += 1
        elif case_b:
            counts["fp_case_b_only"] += 1
        else:
            counts["ignored"] += 1

    fp_a = counts["fp_case_a_only"] + counts["fp_both"]
    fp_b = counts["fp_case_b_only"] + counts["fp_both"]
    edge_fp = counts["fp_case_a_only"] + counts["fp_case_b_only"] + counts["fp_both"]
    denominator = len(retained)
    return {
        "input_edges": len(edges),
        "retained_edges": denominator,
        "tp": counts["tp"],
        "fp_case_a": fp_a,
        "fp_case_b": fp_b,
        "fp_case_a_only": counts["fp_case_a_only"],
        "fp_case_b_only": counts["fp_case_b_only"],
        "fp_both": counts["fp_both"],
        "edge_fp_union": edge_fp,
        "ignored": counts["ignored"],
        "ignored_fraction": counts["ignored"] / denominator if denominator else float("nan"),
        "filtered_missing_or_self": filtered_missing,
        "filtered_duplicate": filtered_duplicate,
        "filtered_nonconsecutive": filtered_nonconsecutive,
        "filtered_merge_duplicate": filtered_merge_duplicate,
        "filtered_outdegree": filtered_outdegree,
    }


def _metric_row(state: MatchState, classification: Mapping[str, Any]) -> dict[str, float]:
    tp = int(classification["tp"])
    fp = int(classification["edge_fp_union"])
    fn = state.gt_edge_count - tp
    denominator = tp + fp + fn
    edge_jaccard = tp / denominator if denominator else float("nan")
    ratio = (len(state.nodes) - state.estimated_nodes) / state.estimated_nodes
    adjusted = max(0.0, edge_jaccard * (1.0 - 0.1 * ratio))
    return {
        "edge_tp": float(tp),
        "edge_fp": float(fp),
        "edge_fn": float(fn),
        "num_pred_nodes": float(len(state.nodes)),
        "estimated_number_of_nodes": state.estimated_nodes,
        "node_recall": state.node_recall,
        "total_node_ratio": ratio,
        "edge_jaccard": edge_jaccard,
        "adj_edge_jaccard": adjusted,
        "ignored_edges": float(classification["ignored"]),
        "retained_edges": float(classification["retained_edges"]),
    }


def _summary(rows: Sequence[Mapping[str, float]]) -> dict[str, float | int]:
    tp = sum(row["edge_tp"] for row in rows)
    fp = sum(row["edge_fp"] for row in rows)
    fn = sum(row["edge_fn"] for row in rows)
    weights = [row["edge_tp"] + row["edge_fp"] + row["edge_fn"] for row in rows]
    total_weight = sum(weights)
    estimated = sum(row["estimated_number_of_nodes"] for row in rows)
    predicted = sum(row["num_pred_nodes"] for row in rows)
    retained = sum(row["retained_edges"] for row in rows)
    ignored = sum(row["ignored_edges"] for row in rows)
    return {
        "n": len(rows),
        "edge_tp": int(tp),
        "edge_fp": int(fp),
        "edge_fn": int(fn),
        "total_node_ratio": (
            sum(weight * row["total_node_ratio"] for weight, row in zip(weights, rows, strict=True))
            / total_weight
        ),
        "aggregate_node_ratio": (predicted - estimated) / estimated,
        "predicted_nodes": int(predicted),
        "estimated_nodes": int(estimated),
        "node_recall": sum(row["node_recall"] for row in rows) / len(rows),
        "edge_jaccard": tp / (tp + fp + fn),
        "adj_edge_jaccard": (
            sum(weight * row["adj_edge_jaccard"] for weight, row in zip(weights, rows, strict=True))
            / total_weight
        ),
        "ignored_edges": int(ignored),
        "retained_edges": int(retained),
        "ignored_fraction": ignored / retained if retained else float("nan"),
    }


def _candidate_records(
    probabilities: Mapping[tuple[int, int], float],
    raw_nodes: Mapping[int, Mapping[str, Any]],
    scale_zyx: tuple[float, float, float],
) -> list[tuple[int, int, float, float, int]]:
    by_source: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    scale = np.asarray(scale_zyx, dtype=np.float64)
    for (source, target), probability in probabilities.items():
        left = raw_nodes[int(source)]
        right = raw_nodes[int(target)]
        delta = np.asarray(
            [right["z"] - left["z"], right["y"] - left["y"], right["x"] - left["x"]],
            dtype=np.float64,
        ) * scale
        distance = float(np.linalg.norm(delta))
        by_source[int(source)].append((int(target), float(probability), distance))
    records: list[tuple[int, int, float, float, int]] = []
    for source, values in by_source.items():
        for rank, (target, probability, distance) in enumerate(
            sorted(values, key=lambda row: (row[2], row[0])), start=1
        ):
            records.append((source, target, probability, distance, rank))
    return records


def _admitted_edges(
    baseline_edges: Sequence[tuple[int, int]],
    candidates: Sequence[tuple[int, int, float, float, int]],
    selected_ids: frozenset[int],
    *,
    threshold: float,
    gate_um: float,
    neighbor_cap: int,
) -> list[tuple[int, int]]:
    baseline = [pair for pair in baseline_edges if pair[0] in selected_ids and pair[1] in selected_ids]
    baseline_set = set(baseline)
    additions = [
        row
        for row in candidates
        if row[0] in selected_ids
        and row[1] in selected_ids
        and row[2] >= threshold
        and row[3] <= gate_um + 1.0e-12
        and row[4] <= neighbor_cap
        and (row[0], row[1]) not in baseline_set
    ]
    additions.sort(key=lambda row: (row[0], -row[2], row[3], row[1]))
    return [*baseline, *((source, target) for source, target, _p, _d, _r in additions)]


def _select_nodes(
    baseline_nodes: Sequence[Mapping[str, Any]],
    raw_nodes: Mapping[int, Mapping[str, Any]],
    estimated_nodes: float,
    target_ratio: float | None,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in baseline_nodes]
    if target_ratio is None:
        return selected
    desired = max(len(selected), int(round(estimated_nodes * (1.0 + target_ratio))))
    existing = {int(row["node_id"]) for row in selected}
    additions = sorted(
        (dict(row) for node_id, row in raw_nodes.items() if node_id not in existing),
        key=lambda row: (-float(row["score"]), int(row["t"]), int(row["node_id"])),
    )
    selected.extend(additions[: max(0, desired - len(selected))])
    return selected


def _sum_classifications(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    integer_keys = [
        key for key in rows[0] if key not in {"identity", "ignored_fraction"}
    ]
    result: dict[str, int | float] = {
        key: sum(int(row[key]) for row in rows) for key in integer_keys
    }
    retained = int(result["retained_edges"])
    result["ignored_fraction"] = int(result["ignored"]) / retained if retained else float("nan")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists():
        raise AsymmetrySweepError(f"refusing to overwrite output: {args.output}")
    thresholds = tuple(float(value) for value in args.edge_threshold)
    gates = tuple(float(value) for value in args.gate_um)
    caps = tuple(int(value) for value in args.neighbor_cap)
    if thresholds != DEFAULT_THRESHOLDS or gates != DEFAULT_GATES or caps != DEFAULT_CAPS:
        raise AsymmetrySweepError("the v1 sweep grid must remain frozen")

    split = _load(args.split, "frozen split")
    baseline_scores = _load(args.baseline_scores, "baseline score")
    score_bh_cv57.validate_revalidated_scorer(args.scorer_parity)
    archive_ids = scoreall.archive_train_video_ids(args.zip)
    selected = score_bh_cv57.exact_selection(
        split, split_path=args.split, archive_video_ids=archive_ids
    )
    manifest = scoreall.validate_fragment_set(
        scoreall.resolve_fragment_paths(str(args.baseline_fragments)), selected
    )
    fragment_paths = dict(manifest.ordered)
    cache_paths = {path.stem.replace("__", ":", 1): path for path in args.cache.glob("*.bhc")}
    meta_paths = {
        path.stem.replace("__", ":", 1): path
        for path in args.cache.glob("*.json")
        if not path.name.endswith(".intent.json")
    }
    if set(cache_paths) != set(selected) or set(meta_paths) != set(selected):
        raise AsymmetrySweepError("cache/meta coverage is not exact CV57")

    model, bundle = bh_model.load_v45_model(args.support_root, device="cpu")
    edge_sha256 = _sha256(args.edge_artifact)
    edge_scorer, edge_record = run_bh.load_sparse_edge_scorer(
        args.edge_artifact,
        edge_sha256,
        args.edge_auxiliary,
        model,
        bundle,
        edge_module=bh_edge,
    )
    if edge_scorer is None or edge_record is None or edge_record.get("division_enabled") is not False:
        raise AsymmetrySweepError("released divisions-off edge scorer was not restored")

    settings = [
        (threshold, gate, cap, _setting_key(threshold, gate, cap))
        for threshold in thresholds
        for gate in gates
        for cap in caps
    ]
    baseline_classifications: list[dict[str, Any]] = []
    metric_rows: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    candidate_counts = 0
    part_dir = args.output.with_suffix(args.output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    with ZipStore(args.zip, mode="r") as store:
        # Zarr 3 opens ZipStore lazily.  A checkpoint-only replay performs no
        # reads, so initialize it explicitly to keep __exit__ well-defined.
        if not store._is_open:
            store._sync_open()
        for index, identity in enumerate(selected, start=1):
            part_path = part_dir / f"{identity.replace(':', '__', 1)}.json"
            if part_path.is_file():
                part = _load(part_path, f"checkpoint for {identity}")
                if part.get("identity") != identity or not isinstance(part.get("metrics"), list):
                    raise AsymmetrySweepError(f"invalid replay checkpoint: {part_path}")
                baseline_classifications.append(dict(part["baseline_classification"]))
                candidate_counts += int(part["candidate_count"])
                for record in part["metrics"]:
                    metric_rows[
                        (str(record["setting"]), str(record["target"]), identity)
                    ].append(dict(record["row"]))
                print(f"asymmetry replay {index:02d}/{len(selected)} {identity} [checkpoint]", flush=True)
                continue
            dataset = identity.split(":", 1)[1]
            family = _family(identity)
            baseline_nodes, baseline_edges = _fragment_graph(fragment_paths[identity], dataset)
            meta = _load(meta_paths[identity], f"cache metadata for {identity}")
            cache = bh_model.load_sparse_cache(
                cache_paths[identity], expected_identity=sweep_support._cache_identity(meta)
            )
            raw_nodes = _raw_nodes(cache)
            probabilities, _no_link, association = run_bh.score_sparse_edge_cache(
                cache, edge_scorer, edge_module=bh_edge
            )
            candidate_counts += int(association["candidate_count"])

            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            gt = read_geff_arrays(store, "train", dataset)
            candidates = _candidate_records(probabilities, raw_nodes, scale)
            if len(candidates) != int(association["candidate_count"]):
                raise AsymmetrySweepError("candidate record accounting mismatch")

            baseline_state = _match_state(baseline_nodes, gt, scale)
            baseline_classification = classify_edges(baseline_state, baseline_edges)
            baseline_classifications.append({"identity": identity, **baseline_classification})
            baseline_row = _metric_row(baseline_state, baseline_classification)
            metric_rows[("baseline", "baseline", identity)].append(baseline_row)

            node_rows_by_target = {
                target_ratio: _select_nodes(
                    baseline_nodes, raw_nodes, baseline_state.estimated_nodes, target_ratio
                )
                for target_ratio in DEFAULT_NODE_TARGETS[family]
            }
            states: dict[float | None, MatchState] = {None: baseline_state}
            numeric_targets = [
                target for target in DEFAULT_NODE_TARGETS[family] if target is not None
            ]
            with ThreadPoolExecutor(max_workers=min(3, len(numeric_targets))) as executor:
                futures = {
                    target: executor.submit(_match_state, node_rows_by_target[target], gt, scale)
                    for target in numeric_targets
                }
                for target in numeric_targets:
                    states[target] = futures[target].result()

            for target_ratio in DEFAULT_NODE_TARGETS[family]:
                target_key = _node_target_key(target_ratio)
                state = states[target_ratio]
                selected_ids = frozenset(state.node_time)
                for threshold, gate, cap, setting_key in settings:
                    edges = _admitted_edges(
                        baseline_edges,
                        candidates,
                        selected_ids,
                        threshold=threshold,
                        gate_um=gate,
                        neighbor_cap=cap,
                    )
                    classification = classify_edges(state, edges)
                    metric_rows[(setting_key, target_key, identity)].append(
                        _metric_row(state, classification)
                    )
            checkpoint = {
                "schema": "celltrack-alpha.metric-asymmetry-cv57-part.v1",
                "identity": identity,
                "candidate_count": int(association["candidate_count"]),
                "baseline_classification": {
                    "identity": identity,
                    **baseline_classification,
                },
                "metrics": [
                    {"setting": setting, "target": target, "row": values[0]}
                    for (setting, target, row_identity), values in metric_rows.items()
                    if row_identity == identity
                ],
            }
            run_bh.atomic_create_json(part_path, checkpoint)
            print(f"asymmetry replay {index:02d}/{len(selected)} {identity}", flush=True)

    baseline_rows = [metric_rows[("baseline", "baseline", identity)][0] for identity in selected]
    baseline_summary = _summary(baseline_rows)
    observed = {key: baseline_summary[key] for key in EXPECTED_BASELINE}
    if any(
        not math.isclose(float(observed[key]), float(value), rel_tol=0.0, abs_tol=1.0e-12)
        for key, value in EXPECTED_BASELINE.items()
    ):
        raise AsymmetrySweepError(f"manual scorer did not reproduce baseline: {observed}")

    asymmetry = _sum_classifications(baseline_classifications)
    edge_sweep: list[dict[str, Any]] = []
    for threshold, gate, cap, setting_key in settings:
        rows = [metric_rows[(setting_key, "baseline", identity)][0] for identity in selected]
        edge_sweep.append(
            {
                "setting": setting_key,
                "edge_threshold": threshold,
                "gate_um": gate,
                "gate_label": "motion_tight" if gate == 6.0 else "motion_relaxed",
                "neighbor_cap": cap,
                **_summary(rows),
            }
        )
    edge_sweep.sort(
        key=lambda row: (-float(row["adj_edge_jaccard"]), -float(row["edge_jaccard"]), row["setting"])
    )
    coarse_provenance = None
    if args.coarse_report.is_file():
        coarse = _load(args.coarse_report, "coarse asymmetry sweep")
        coarse_rows = coarse.get("edge_sweep")
        if coarse.get("schema") != SCHEMA or not isinstance(coarse_rows, list):
            raise AsymmetrySweepError("coarse report has incompatible schema")
        by_setting = {str(row["setting"]): row for row in edge_sweep}
        for row in coarse_rows:
            by_setting.setdefault(str(row["setting"]), dict(row))
        edge_sweep = sorted(
            by_setting.values(),
            key=lambda row: (
                -float(row["adj_edge_jaccard"]),
                -float(row["edge_jaccard"]),
                row["setting"],
            ),
        )
        coarse_provenance = {
            "path": str(args.coarse_report),
            "sha256": _sha256(args.coarse_report),
        }
    best_edge = edge_sweep[0]
    finalists = edge_sweep[:JOINT_EDGE_FINALISTS]

    node_sweep: dict[str, list[dict[str, Any]]] = {}
    for family in DEFAULT_NODE_TARGETS:
        family_ids = [identity for identity in selected if _family(identity) == family]
        values: list[dict[str, Any]] = []
        for target in DEFAULT_NODE_TARGETS[family]:
            target_key = _node_target_key(target)
            rows = [metric_rows[(str(best_edge["setting"]), target_key, identity)][0] for identity in family_ids]
            values.append(
                {
                    "family": family,
                    "target_node_ratio": target,
                    "target_key": target_key,
                    "fixed_edge_setting": best_edge["setting"],
                    **_summary(rows),
                }
            )
        node_sweep[family] = values

    joint: list[dict[str, Any]] = []
    for edge in finalists:
        setting_key = str(edge["setting"])
        for target_6bba in DEFAULT_NODE_TARGETS["6bba"]:
            key_6bba = _node_target_key(target_6bba)
            for target_44b6 in DEFAULT_NODE_TARGETS["44b6"]:
                key_44b6 = _node_target_key(target_44b6)
                rows = [
                    metric_rows[
                        (
                            setting_key,
                            key_6bba if _family(identity) == "6bba" else key_44b6,
                            identity,
                        )
                    ][0]
                    for identity in selected
                ]
                joint.append(
                    {
                        "edge_setting": setting_key,
                        "edge_threshold": edge["edge_threshold"],
                        "gate_um": edge["gate_um"],
                        "neighbor_cap": edge["neighbor_cap"],
                        "node_target_6bba": target_6bba,
                        "node_target_44b6": target_44b6,
                        **_summary(rows),
                    }
                )
    joint.sort(
        key=lambda row: (-float(row["adj_edge_jaccard"]), -float(row["edge_jaccard"]))
    )
    optimum = dict(joint[0])
    optimum["delta_vs_baseline"] = float(optimum["adj_edge_jaccard"]) - BASELINE_ADJUSTED
    best_adjusted = float(optimum["adj_edge_jaccard"])
    ignored_small = float(asymmetry["ignored_fraction"]) < 0.10
    if ignored_small and best_adjusted <= 0.880:
        verdict = "HYPOTHESIS_WRONG_STOP"
    elif best_adjusted > 0.89:
        verdict = "PROMOTE_IMMEDIATELY"
    else:
        verdict = "GAIN_BELOW_PROMOTION_BAR"

    score_samples = baseline_scores.get("samples")
    if not isinstance(score_samples, list) or len(score_samples) != 57:
        raise AsymmetrySweepError("baseline score provenance is not exact CV57")
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "selection": {"video_count": len(selected), "video_ids": list(selected)},
        "execution": {
            "device": "cpu",
            "detector_inference": False,
            "detector_forward_calls": 0,
            "training": False,
            "gpu_used": False,
            "division_enabled": False,
            "kaggle_action": False,
            "push": False,
            "cached_candidate_count": candidate_counts,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "method": {
            "edge_thresholds": sorted(
                {float(row["edge_threshold"]) for row in edge_sweep}, reverse=True
            ),
            "gates_um": list(gates),
            "neighbor_caps": list(caps),
            "node_targets": {
                family: list(values) for family, values in DEFAULT_NODE_TARGETS.items()
            },
            "node_target_semantics": (
                "per-video desired node count = round(estimated nodes * (1 + family target)); "
                "never remove baseline nodes; restore highest-confidence cached detector nodes first"
            ),
            "edge_admission_semantics": (
                "append cached radius10/cap8 candidates passing probability threshold, raw-distance "
                "gate and distance-rank cap; exact baseline edges retain lowest scorer edge IDs"
            ),
            "joint_edge_finalists": JOINT_EDGE_FINALISTS,
        },
        "baseline": baseline_summary,
        "asymmetry": {
            **asymmetry,
            "fp_case_note": (
                "case-a and case-b totals overlap by fp_both; edge_fp_union counts the overlap once"
            ),
            "per_video": baseline_classifications,
        },
        "edge_sweep": edge_sweep,
        "node_sweep": node_sweep,
        "joint_sweep": joint,
        "optimum": optimum,
        "decision": {
            "verdict": verdict,
            "ignored_fraction_small_threshold": 0.10,
            "wrong_hypothesis_adjusted_ceiling": 0.880,
            "promotion_threshold": 0.89,
            "ignored_fraction_is_small": ignored_small,
            "best_adjusted_edge_jaccard": best_adjusted,
        },
        "provenance": {
            "git_pre_edit_checkpoint": "9012886",
            "baseline_fragments": str(args.baseline_fragments),
            "baseline_scores": {"path": str(args.baseline_scores), "sha256": _sha256(args.baseline_scores)},
            "edge_artifact": {"path": str(args.edge_artifact), "sha256": edge_sha256},
            "edge_auxiliary": {"path": str(args.edge_auxiliary), "sha256": _sha256(args.edge_auxiliary)},
            "scorer_parity": {"path": str(args.scorer_parity), "sha256": _sha256(args.scorer_parity)},
            "script": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
            "coarse_report": coarse_provenance,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_bh.atomic_create_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    value.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    value.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    value.add_argument("--baseline-fragments", type=Path, default=DEFAULT_FRAGMENTS)
    value.add_argument("--baseline-scores", type=Path, default=DEFAULT_BASELINE_SCORES)
    value.add_argument("--edge-artifact", type=Path, default=DEFAULT_EDGE)
    value.add_argument("--edge-auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    value.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT)
    value.add_argument("--scorer-parity", type=Path, default=DEFAULT_PARITY)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--coarse-report", type=Path, default=DEFAULT_COARSE_REPORT)
    value.add_argument("--edge-threshold", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    value.add_argument("--gate-um", type=float, nargs="+", default=list(DEFAULT_GATES))
    value.add_argument("--neighbor-cap", type=int, nargs="+", default=list(DEFAULT_CAPS))
    return value


def main() -> None:
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "decision": result["decision"], "optimum": result["optimum"]}))


if __name__ == "__main__":
    main()
