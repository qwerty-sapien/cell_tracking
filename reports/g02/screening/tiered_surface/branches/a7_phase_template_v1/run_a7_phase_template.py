#!/usr/bin/env python3
"""Materialize and screen the frozen A7 phase-template occupied-swap realizer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore


HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = Path(__file__).resolve().parents[6]
CURRENT_ROOT = Path("/Users/agent/Desktop/cell_tracking")
PREREG = HERE / "PREREGISTRATION.json"
PHASE_MODULE = HERE / "phase_template.py"
for location in (
    HERE,
    CURRENT_ROOT / "src",
    CURRENT_ROOT / "scripts",
    CURRENT_ROOT / "external/Tracking_Support/repo/src",
    CURRENT_ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import phase_template as phase  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_action,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import (  # noqa: E402
    read_submission,
    write_submission,
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402
import scoreall  # noqa: E402


SCHEMA = "celltrack-alpha.a7-phase-tracklet-swap-result.v1"


class A7Error(RuntimeError):
    """The preregistered A7 realization failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A7Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise A7Error(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise A7Error(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise A7Error(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_or_verify_submission(graphs: Mapping[str, Any], path: Path) -> None:
    """Write once, or prove a resumed evaluation reproduced the sealed fragment."""

    if not path.exists():
        write_submission(graphs, path)
        return
    temporary = path.with_name(path.name + ".resume-verify")
    if temporary.exists():
        raise A7Error(f"stale resume-verification file: {temporary}")
    write_submission(graphs, temporary)
    try:
        if sha256(temporary) != sha256(path):
            raise A7Error(f"resumed candidate fragment drift: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise A7Error(f"{label} pin drift: expected {expected}, got {actual}")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise A7Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticate() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    prereg = load_json(PREREG)
    if prereg.get("status") != "FROZEN_BEFORE_PHASE_FEATURE_OR_GRAPH_OUTCOME":
        raise A7Error("preregistration is not frozen")
    pins = prereg["frozen_inputs"]
    for key in (
        "ae_p7_contract",
        "apncc_44b6_report",
        "occupied_swap_population",
        "n1_interface",
        "n1_implementation",
        "grouped_folds",
    ):
        pin = pins[key]
        verify(CURRENT_ROOT / str(pin["path"]), str(pin["sha256"]), key)
    archive = CURRENT_ROOT / str(pins["raw_archive"]["path"])
    if archive.stat().st_size != int(pins["raw_archive"]["bytes"]):
        raise A7Error("raw archive byte-size drift")

    manifest = load_json(CURRENT_ROOT / str(pins["occupied_swap_population"]["path"]))
    rows = manifest.get("classes", {}).get("occupied_swap", {}).get("videos", [])
    if len(rows) != 36 or sum(int(row["count"]) for row in rows) != 45_961:
        raise A7Error("occupied-swap population cardinality drift")
    for row in rows:
        verify(
            CURRENT_ROOT / str(row["population_path"]),
            str(row["population_sha256"]),
            f"population {row['identity']}",
        )

    report_pin = pins["apncc_44b6_report"]
    report = load_json(CURRENT_ROOT / str(report_pin["path"]))
    files = report.get("graph", {}).get("control", {}).get("input_provenance", {}).get("files", [])
    if len(files) != 36:
        raise A7Error("APNCC fragment coverage drift")
    records: dict[str, dict[str, Any]] = {}
    logical = []
    for row in files:
        identity = str(row["identity"])
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = CURRENT_ROOT / path
        verify(path, str(row["sha256"]), f"APNCC fragment {identity}")
        if identity in records or path.stat().st_size != int(row["bytes"]):
            raise A7Error(f"APNCC fragment record drift: {identity}")
        records[identity] = {**dict(row), "path": str(path)}
        logical.append((identity, str(row["sha256"])))
    aggregate = scoreall.combined_fragment_sha256(logical)
    if aggregate != report_pin["fragment_aggregate_sha256"]:
        raise A7Error("APNCC fragment aggregate drift")
    return prereg, manifest, records


def _node_center(graph: Any, node_id: int) -> tuple[int, int, int]:
    node = graph.nodes[int(node_id)]
    return int(node.z), int(node.y), int(node.x)


def _frame(
    array: Any,
    frame_index: int,
    cache: dict[int, np.ndarray],
    low: float,
    high: float,
) -> np.ndarray:
    if frame_index not in cache:
        cache[frame_index] = phase.normalize_frame(
            np.asarray(array[frame_index]), low, high
        )
    return cache[frame_index]


def materialize_video_features(
    *,
    video_index: int,
    offset: int,
    row: Mapping[str, Any],
    fragment: Mapping[str, Any],
    store: Any,
    output_path: Path,
) -> dict[str, Any]:
    identity = str(row["identity"])
    dataset = identity.split(":", 1)[1]
    graph = read_submission(Path(str(fragment["path"])))[dataset]
    incoming, outgoing = phase.graph_incidence(graph)
    edges = {
        (int(edge.source_id), int(edge.target_id)) for edge in graph.edges
    }
    with np.load(CURRENT_ROOT / str(row["population_path"]), allow_pickle=False) as data:
        source = np.asarray(data["source_id"], dtype=np.int64)
        target = np.asarray(data["target_id"], dtype=np.int64)
        source_t = np.asarray(data["source_t"], dtype=np.int16)
        target_t = np.asarray(data["target_t"], dtype=np.int16)
        source_zyx = np.asarray(data["source_zyx"], dtype=np.int16)
        target_zyx = np.asarray(data["target_zyx"], dtype=np.int16)
    count = len(source)
    if count != int(row["count"]) or np.any(target_t != source_t + 1):
        raise A7Error(f"population geometry drift: {identity}")
    for index in range(count):
        if (
            _node_center(graph, int(source[index])) != tuple(int(value) for value in source_zyx[index])
            or _node_center(graph, int(target[index])) != tuple(int(value) for value in target_zyx[index])
        ):
            raise A7Error(f"population/APNCC node identity drift: {identity}:{index}")

    group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
    array = group["0"]
    scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
    quantiles = dict(group.attrs["image_statistics"]["quantiles"])
    low = float(quantiles["0.01"])
    high = float(quantiles["0.99"])
    if tuple(array.shape) != (100, 64, 256, 256):
        raise A7Error(f"raw array geometry drift: {identity}")

    candidate_peak = np.empty(count, dtype=np.float32)
    candidate_ratio = np.empty(count, dtype=np.float32)
    candidate_shift = np.empty(count, dtype=np.float32)
    candidate_quality = np.empty(count, dtype=np.float32)
    candidate_wrapped = np.empty(count, dtype=np.uint8)
    source_frames = np.empty(count, dtype=np.uint8)
    destination_frames = np.empty(count, dtype=np.uint8)
    conflict_max_quality = np.empty(count, dtype=np.float32)
    action_margin = np.empty(count, dtype=np.float32)
    conflict_count = np.empty(count, dtype=np.uint8)
    frame_cache: dict[int, np.ndarray] = {}

    for transition in sorted({int(value) for value in source_t}):
        local_indices = np.flatnonzero(source_t == transition)
        edge_pairs: set[tuple[int, int]] = set()
        conflicts_by_index: dict[int, tuple[tuple[int, int], ...]] = {}
        for raw_index in local_indices:
            index = int(raw_index)
            pair = (int(source[index]), int(target[index]))
            conflicts = tuple(
                sorted(edge for edge in edges if edge[0] == pair[0] or edge[1] == pair[1])
            )
            if len(conflicts) not in (1, 2) or pair in edges:
                raise A7Error(f"occupied conflict boundary drift: {identity}:{index}")
            conflicts_by_index[index] = conflicts
            edge_pairs.add(pair)
            edge_pairs.update(conflicts)

        source_ids = sorted({pair[0] for pair in edge_pairs})
        destination_ids = sorted({pair[1] for pair in edge_pairs})
        source_paths = {
            node_id: phase.source_tracklet(graph, incoming, node_id)
            for node_id in source_ids
        }
        destination_paths = {
            node_id: phase.destination_tracklet(graph, outgoing, node_id)
            for node_id in destination_ids
        }
        requested_by_frame: dict[int, set[int]] = defaultdict(set)
        for path in (*source_paths.values(), *destination_paths.values()):
            for node_id in path:
                requested_by_frame[int(graph.nodes[node_id].t)].add(int(node_id))
        keep_frames = set(requested_by_frame)
        for frame_index in tuple(frame_cache):
            if frame_index not in keep_frames:
                del frame_cache[frame_index]
        crops: dict[int, np.ndarray] = {}
        for frame_index, node_ids_raw in sorted(requested_by_frame.items()):
            node_ids = sorted(node_ids_raw)
            centers = np.asarray([_node_center(graph, node_id) for node_id in node_ids])
            values = phase.sample_crops(
                _frame(array, frame_index, frame_cache, low, high), centers, scale
            )
            for node_id, crop in zip(node_ids, values, strict=True):
                crops[node_id] = crop

        source_fft = {}
        for node_id, path in source_paths.items():
            template = np.median(
                np.stack([crops[value] for value in path], axis=0), axis=0
            ).astype(np.float32)
            source_fft[node_id] = phase.template_fft(template)
        destination_fft = {}
        for node_id, path in destination_paths.items():
            template = np.median(
                np.stack([crops[value] for value in path], axis=0), axis=0
            ).astype(np.float32)
            destination_fft[node_id] = phase.template_fft(template)
        evidence = {
            pair: phase.phase_evidence_from_ffts(
                source_fft[pair[0]], destination_fft[pair[1]]
            )
            for pair in sorted(edge_pairs)
        }
        for raw_index in local_indices:
            index = int(raw_index)
            pair = (int(source[index]), int(target[index]))
            candidate = evidence[pair]
            conflicts = conflicts_by_index[index]
            incumbent_quality = max(evidence[value].quality for value in conflicts)
            candidate_peak[index] = candidate.peak
            candidate_ratio[index] = candidate.peak_ratio
            candidate_shift[index] = candidate.shift_norm_um
            candidate_quality[index] = candidate.quality
            candidate_wrapped[index] = int(candidate.wrapped_peak_boundary)
            source_frames[index] = len(source_paths[pair[0]])
            destination_frames[index] = len(destination_paths[pair[1]])
            conflict_max_quality[index] = incumbent_quality
            action_margin[index] = candidate.quality - incumbent_quality
            conflict_count[index] = len(conflicts)
        print(
            f"FEATURE {identity} t={transition} actions={len(local_indices)}",
            flush=True,
        )

    finite = np.column_stack(
        (
            candidate_peak,
            candidate_ratio,
            candidate_shift,
            candidate_quality,
            conflict_max_quality,
            action_margin,
        )
    )
    if not np.isfinite(finite).all():
        raise A7Error(f"non-finite phase feature: {identity}")
    write_npz(
        output_path,
        global_index=np.arange(offset, offset + count, dtype=np.int64),
        candidate_peak=candidate_peak,
        candidate_peak_ratio=candidate_ratio,
        candidate_shift_norm_um=candidate_shift,
        candidate_quality=candidate_quality,
        candidate_wrapped_peak_boundary=candidate_wrapped,
        source_template_frames=source_frames,
        destination_template_frames=destination_frames,
        conflict_max_quality=conflict_max_quality,
        action_margin=action_margin,
        conflict_count=conflict_count,
    )
    return {
        "video_index": video_index,
        "identity": identity,
        "actions": count,
        "path": str(output_path.relative_to(WORKTREE_ROOT)),
        "sha256": sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def materialize_features(output: Path) -> Path:
    started = time.monotonic()
    prereg, manifest, records = authenticate()
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest["classes"]["occupied_swap"]["videos"]
    offset = 0
    receipts = []
    archive = CURRENT_ROOT / str(prereg["frozen_inputs"]["raw_archive"]["path"])
    with ZipStore(archive, mode="r") as store:
        for video_index, row in enumerate(rows):
            identity = str(row["identity"])
            path = feature_dir / f"{identity.replace(':', '__', 1)}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    observed = np.asarray(data["global_index"], dtype=np.int64)
                    expected = np.arange(offset, offset + int(row["count"]), dtype=np.int64)
                    if not np.array_equal(observed, expected):
                        raise A7Error(f"partial feature shard drift: {identity}")
                receipt = {
                    "video_index": video_index,
                    "identity": identity,
                    "actions": int(row["count"]),
                    "path": str(path.relative_to(WORKTREE_ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "resumed": True,
                }
            else:
                receipt = materialize_video_features(
                    video_index=video_index,
                    offset=offset,
                    row=row,
                    fragment=records[identity],
                    store=store,
                    output_path=path,
                )
                receipt["resumed"] = False
            receipts.append(receipt)
            offset += int(row["count"])
            if time.monotonic() - started > 7_200:
                raise A7Error("feature materialization exceeded 7,200-second hard limit")
    logical = scoreall.combined_fragment_sha256(
        [(str(row["identity"]), str(row["sha256"])) for row in receipts]
    )
    feature_manifest = output / "FEATURE_MANIFEST.json"
    write_json(
        feature_manifest,
        {
            "schema": "celltrack-alpha.a7-phase-tracklet-feature-manifest.v1",
            "status": "COMPLETE_FEATURES_BEFORE_SELECTION_OUTCOMES",
            "preregistration_sha256": sha256(PREREG),
            "implementation_sha256": sha256(Path(__file__)),
            "primitive_sha256": sha256(PHASE_MODULE),
            "actions": offset,
            "videos": len(receipts),
            "aggregate_sha256": logical,
            "elapsed_seconds": time.monotonic() - started,
            "shards": receipts,
            "scope": {
                "cpu_only": True,
                "held_labels_read": False,
                "raw_shell_read": False,
                "whole_action_pool_read": False,
                "occupied_conflicts_only": True,
            },
        },
    )
    return feature_manifest


def load_features(output: Path, expected_count: int) -> dict[str, np.ndarray]:
    manifest_path = output / "FEATURE_MANIFEST.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "COMPLETE_FEATURES_BEFORE_SELECTION_OUTCOMES"
        or int(manifest.get("actions", -1)) != expected_count
        or manifest.get("preregistration_sha256") != sha256(PREREG)
    ):
        raise A7Error("feature manifest is not sealed")
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in manifest["shards"]:
        path = WORKTREE_ROOT / str(row["path"])
        verify(path, str(row["sha256"]), f"phase feature {row['identity']}")
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                chunks[key].append(np.asarray(data[key]))
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    if not np.array_equal(arrays["global_index"], np.arange(expected_count)):
        raise A7Error("feature/action alignment drift")
    return arrays


def frozen_population() -> tuple[Any, Any, dict[str, Any]]:
    selector_dir = CURRENT_ROOT / "reports/g02/screening/selector_requirement_44b6"
    if str(selector_dir) not in sys.path:
        sys.path.insert(0, str(selector_dir))
    os.environ["CELLTRACK_ROOT"] = str(CURRENT_ROOT)
    module = load_module("a7_frozen_selector_class", selector_dir / "selector_requirement_class.py")
    population, baseline = module.load_swap_population()
    return module, population, baseline


def partition_delta(
    module: Any,
    population: Any,
    selected: np.ndarray,
    video_indices: Sequence[int],
) -> dict[str, float]:
    base = module.base
    delta_tp = np.bincount(
        population.video[selected],
        weights=population.d_tp[selected],
        minlength=len(population.identities),
    ).astype(np.int64)
    delta_fp = np.bincount(
        population.video[selected],
        weights=population.d_fp[selected],
        minlength=len(population.identities),
    ).astype(np.int64)
    delta_fn = np.bincount(
        population.video[selected],
        weights=population.d_fn[selected],
        minlength=len(population.identities),
    ).astype(np.int64)
    rows = base.edge_candidate_rows(population, delta_tp, delta_fp, delta_fn)
    chosen = [int(value) for value in video_indices]
    before = base.edge_summary([population.baseline_rows[index] for index in chosen])
    after = base.edge_summary([rows[index] for index in chosen])
    return base.metric_delta(before, after)


def scaled_budget(global_budget: int, video_count: int) -> int:
    return max(1, int(math.floor(global_budget * video_count / 36.0 + 0.5)))


def configuration_mask(features: Mapping[str, np.ndarray], config: Mapping[str, float]) -> np.ndarray:
    return (
        (features["source_template_frames"] >= 2)
        & (features["destination_template_frames"] >= 2)
        & (features["candidate_peak"] >= float(config["candidate_peak_min"]))
        & (features["candidate_peak_ratio"] >= float(config["candidate_peak_ratio_min"]))
        & (features["candidate_shift_norm_um"] <= float(config["candidate_shift_norm_um_max"]))
        & (features["action_margin"] >= float(config["action_margin_min"]))
    )


def ordered_eligible(
    features: Mapping[str, np.ndarray],
    eligible: np.ndarray,
    partition: np.ndarray,
    population: Any,
) -> np.ndarray:
    indices = np.flatnonzero(eligible & partition)
    if not len(indices):
        return indices.astype(np.int64)
    order = np.lexsort(
        (
            population.target[indices],
            population.source[indices],
            population.video[indices],
            -features["candidate_quality"][indices],
            -features["action_margin"][indices],
        )
    )
    return indices[order].astype(np.int64)


def nested_select(
    prereg: Mapping[str, Any],
    module: Any,
    population: Any,
    features: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    selection = prereg["sealed_nested_selection"]
    grid = selection["threshold_grid"]
    video_folds = []
    for video_index in range(len(population.identities)):
        local = population.fold[population.video == video_index]
        if not len(local) or np.any(local != local[0]):
            raise A7Error("video fold identity drift")
        video_folds.append(int(local[0]))
    video_folds_array = np.asarray(video_folds, dtype=np.int8)
    held_selections = []
    fold_receipts = []
    proposed_total = 0
    for held_fold in range(5):
        train_videos = np.flatnonzero(video_folds_array != held_fold)
        held_videos = np.flatnonzero(video_folds_array == held_fold)
        train_partition = np.isin(population.video, train_videos)
        held_partition = np.isin(population.video, held_videos)
        best: tuple[tuple[Any, ...], dict[str, Any], np.ndarray] | None = None
        for peak in grid["candidate_peak_min"]:
            for ratio in grid["candidate_peak_ratio_min"]:
                for shift in grid["candidate_shift_norm_um_max"]:
                    for margin in grid["action_margin_min"]:
                        config = {
                            "candidate_peak_min": float(peak),
                            "candidate_peak_ratio_min": float(ratio),
                            "candidate_shift_norm_um_max": float(shift),
                            "action_margin_min": float(margin),
                        }
                        eligible = configuration_mask(features, config)
                        ordered = ordered_eligible(
                            features, eligible, train_partition, population
                        )
                        for global_budget in selection["global_final_edit_budget_grid"]:
                            train_budget = scaled_budget(int(global_budget), len(train_videos))
                            chosen = phase.strict_select(
                                ordered,
                                video=population.video,
                                source=population.source,
                                target=population.target,
                                conflicts=population.conflicts,
                                budget=train_budget,
                                per_video_cap=int(selection["per_video_edit_cap"]),
                            )
                            if len(chosen) < int(selection["training_eligibility"]["minimum_applied_edits"]):
                                continue
                            precision = float(population.label[chosen].mean())
                            if precision < float(selection["training_eligibility"]["minimum_oracle_precision_report_metric"]):
                                continue
                            delta = partition_delta(
                                module, population, chosen, train_videos
                            )
                            if not (
                                delta["adjusted_edge_jaccard"] > 0.0
                                and delta["raw_edge_jaccard"] >= 0.0
                                and delta["node_recall"] >= 0.0
                            ):
                                continue
                            key = (
                                delta["adjusted_edge_jaccard"],
                                -len(chosen),
                                -int(global_budget),
                                float(margin),
                                float(ratio),
                                float(peak),
                                -float(shift),
                            )
                            record = {
                                **config,
                                "global_final_edit_budget": int(global_budget),
                                "training_scaled_budget": train_budget,
                                "training_threshold_pass": int(np.sum(eligible & train_partition)),
                                "training_applied_edits": int(len(chosen)),
                                "training_precision": precision,
                                "training_correct": int(population.label[chosen].sum()),
                                "training_harmful": int(len(chosen) - population.label[chosen].sum()),
                                "training_graph_delta": delta,
                            }
                            if best is None or key > best[0]:
                                best = (key, record, eligible)
        if best is None:
            fold_receipts.append(
                {
                    "held_fold": held_fold,
                    "train_videos": train_videos.tolist(),
                    "held_videos": held_videos.tolist(),
                    "status": "ABSTAIN_NO_ELIGIBLE_TRAINING_CONFIGURATION",
                    "held_threshold_pass": 0,
                    "held_applied_edits": 0,
                }
            )
            continue
        _key, record, eligible = best
        held_order = ordered_eligible(features, eligible, held_partition, population)
        held_budget = scaled_budget(
            int(record["global_final_edit_budget"]), len(held_videos)
        )
        chosen = phase.strict_select(
            held_order,
            video=population.video,
            source=population.source,
            target=population.target,
            conflicts=population.conflicts,
            budget=held_budget,
            per_video_cap=int(selection["per_video_edit_cap"]),
        )
        threshold_pass = int(np.sum(eligible & held_partition))
        proposed_total += threshold_pass
        held_selections.extend(int(value) for value in chosen)
        fold_receipts.append(
            {
                "held_fold": held_fold,
                "train_videos": train_videos.tolist(),
                "held_videos": held_videos.tolist(),
                "status": "SEALED_ON_TRAINING_BEFORE_HELD_APPLICATION",
                "configuration": record,
                "held_scaled_budget": held_budget,
                "held_threshold_pass": threshold_pass,
                "held_applied_edits": int(len(chosen)),
                "held_selected_indices_sha256": hashlib.sha256(
                    np.asarray(chosen, dtype="<i8").tobytes()
                ).hexdigest(),
            }
        )
    selected = np.asarray(sorted(held_selections), dtype=np.int64)
    if len(selected) != len(set(int(value) for value in selected)):
        raise A7Error("held selections overlap")
    if len(selected) > max(selection["global_final_edit_budget_grid"]) + 4:
        raise A7Error("scaled OOF edit support exceeds conservative bound")
    return selected, fold_receipts, proposed_total


def compile_selection(
    module: Any,
    population: Any,
    selected: np.ndarray,
    fragment_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    fragment_dir.mkdir(parents=True, exist_ok=False)
    by_video: dict[int, list[int]] = defaultdict(list)
    for raw_index in selected:
        by_video[int(population.video[int(raw_index)])].append(int(raw_index))
    graphs = []
    receipts = []
    for video_index, (identity, graph) in enumerate(zip(population.identities, population.graphs, strict=True)):
        indices = by_video.get(video_index, [])
        if indices:
            add_edges = tuple(
                sorted(
                    (int(population.source[index]), int(population.target[index]))
                    for index in indices
                )
            )
            conflicts = tuple(
                sorted(
                    {
                        pair
                        for index in indices
                        for pair in population.conflicts[index]
                    }
                )
            )
            action = GraphAction(
                action_id=f"a7.phase.bulk:{video_index}",
                kind=ActionKind.SWAP,
                add_edges=add_edges,
                replace_outgoing_from=tuple(source for source, _target in add_edges),
                replace_incoming_to=tuple(target for _source, target in add_edges),
                expected_conflict_edges=conflicts,
            )
            compiled = compile_action(graph, action)
            candidate = compiled.graph
            removed_edges = len(compiled.removed_edges)
            batch_sha = compiled.action_sha256
        else:
            candidate = graph
            removed_edges = 0
            batch_sha = None
        if set(candidate.nodes) != set(graph.nodes):
            raise A7Error(f"node identity changed: {identity}")
        dataset = identity.split(":", 1)[1]
        path = fragment_dir / f"{identity.replace(':', '__', 1)}.csv"
        write_or_verify_submission({dataset: candidate}, path)
        graphs.append(candidate)
        receipts.append(
            {
                "video_index": video_index,
                "identity": identity,
                "edits": len(indices),
                "removed_edges": removed_edges,
                "before_graph_sha256": graph_sha256(graph),
                "after_graph_sha256": graph_sha256(candidate),
                "compiler_batch_sha256": batch_sha,
                "fragment": str(path.relative_to(WORKTREE_ROOT)),
                "fragment_sha256": sha256(path),
            }
        )
    return graphs, receipts


def exact_score(graphs: Sequence[Any], identities: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    archive = CURRENT_ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
    with ZipStore(archive, mode="r") as store:
        for identity, graph in zip(identities, graphs, strict=True):
            dataset = identity.split(":", 1)[1]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            rows.append({"identity": identity, **official_metric_row(nodes, edges, gt, scale)})
    return official_summary(rows), rows


def evaluate(output: Path) -> Path:
    started = time.monotonic()
    prereg, _manifest, _records = authenticate()
    module, population, baseline = frozen_population()
    if len(population.label) != 45_961:
        raise A7Error("frozen evaluator population drift")
    features = load_features(output, len(population.label))
    selected, fold_receipts, proposed_total = nested_select(
        prereg, module, population, features
    )
    graphs, compiler_receipts = compile_selection(
        module, population, selected, output / "candidate_fragments"
    )
    candidate_summary, per_video = exact_score(graphs, population.identities)
    base_summary = official_summary(list(population.baseline_rows))
    delta = module.base.metric_delta(base_summary, candidate_summary)
    algebra_delta = partition_delta(
        module, population, selected, range(len(population.identities))
    )
    for key in delta:
        if not math.isclose(float(delta[key]), float(algebra_delta[key]), abs_tol=1.0e-12):
            raise A7Error(f"exact/algebra graph parity failed: {key}")
    if not math.isclose(
        float(base_summary["adj_edge_jaccard"]),
        float(baseline["adj_edge_jaccard"]),
        abs_tol=1.0e-12,
    ):
        raise A7Error("APNCC baseline parity failed")

    correct = int(population.label[selected].sum()) if len(selected) else 0
    harmful = int(len(selected) - correct)
    beats_apncc = bool(
        len(selected)
        and delta["adjusted_edge_jaccard"] > 0.0
        and delta["raw_edge_jaccard"] >= 0.0
        and delta["node_recall"] >= 0.0
    )
    campaign_pass = bool(
        delta["adjusted_edge_jaccard"] >= 0.005
        and delta["raw_edge_jaccard"] >= 0.0
        and delta["node_recall"] >= 0.0
    )
    if not beats_apncc:
        status = "KILL_44B6_CANNOT_BEAT_APNCC"
    elif not campaign_pass:
        status = "BEATS_APNCC_BELOW_CAMPAIGN_ACTIONABILITY"
    else:
        status = "PASS_44B6_ACTIONABLE_STOP_BEFORE_6BBA"
    properties = {
        "appearance_grounded": True,
        "narrow": bool(len(selected) <= 64 and proposed_total < len(population.label)),
        "node_disjoint": True,
        "recall_preserving": bool(
            delta["raw_edge_jaccard"] >= 0.0
            and delta["node_recall"] >= 0.0
            and all(set(left.nodes) == set(right.nodes) for left, right in zip(graphs, population.graphs, strict=True))
        ),
    }
    base_edge_counts = {
        key: int(sum(float(row[key]) for row in population.baseline_rows))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    candidate_edge_counts = {
        key: int(sum(float(row[key]) for row in per_video))
        for key in ("edge_tp", "edge_fp", "edge_fn")
    }
    result = {
        "schema": SCHEMA,
        "status": status,
        "node": prereg["node"],
        "preregistration_sha256": sha256(PREREG),
        "implementation_sha256": sha256(Path(__file__)),
        "primitive_sha256": sha256(PHASE_MODULE),
        "feature_manifest_sha256": sha256(output / "FEATURE_MANIFEST.json"),
        "population": {
            "actions": len(population.label),
            "videos": len(population.identities),
            "scope": "occupied-endpoint conflicts only",
        },
        "edit_set": {
            "threshold_pass_before_disjoint_budget_sum": proposed_total,
            "applied_edits": int(len(selected)),
            "affected_videos": int(sum(bool(row["edits"]) for row in compiler_receipts)),
            "postselection_correct": correct,
            "postselection_harmful": harmful,
            "postselection_precision": correct / len(selected) if len(selected) else None,
            "selected_indices_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "exact_44b6": {
            "apncc": {
                "adjusted_edge_jaccard": float(base_summary["adj_edge_jaccard"]),
                "raw_edge_jaccard": float(base_summary["edge_jaccard"]),
                "node_recall": float(base_summary["node_recall"]),
            },
            "candidate": {
                "adjusted_edge_jaccard": float(candidate_summary["adj_edge_jaccard"]),
                "raw_edge_jaccard": float(candidate_summary["edge_jaccard"]),
                "node_recall": float(candidate_summary["node_recall"]),
            },
            "delta": delta,
            "algebra_parity_delta": algebra_delta,
            "edge_counts": {
                "apncc": {
                    "tp": base_edge_counts["edge_tp"],
                    "fp": base_edge_counts["edge_fp"],
                    "fn": base_edge_counts["edge_fn"],
                },
                "candidate": {
                    "tp": candidate_edge_counts["edge_tp"],
                    "fp": candidate_edge_counts["edge_fp"],
                    "fn": candidate_edge_counts["edge_fn"],
                },
            },
            "beats_apncc": beats_apncc,
            "campaign_adjusted_plus_0p005": campaign_pass,
            "historical_apncc_cv57_gain_context": 0.008372476430831166,
        },
        "nested_selection": fold_receipts,
        "compiler_receipts": compiler_receipts,
        "per_video_metrics": per_video,
        "properties": properties,
        "downstream": {
            "family_6bba_run": False,
            "cv57_run": False,
            "kaggle_action": False,
            "gpu_action": False,
            "submission_action": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("features", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("features", "all") and not (output / "FEATURE_MANIFEST.json").exists():
        materialize_features(output)
    if args.stage in ("evaluate", "all"):
        result = evaluate(output)
        value = load_json(result)
        print(
            json.dumps(
                {
                    "status": value["status"],
                    "edits": value["edit_set"]["applied_edits"],
                    "delta": value["exact_44b6"]["delta"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
