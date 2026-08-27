#!/usr/bin/env python3
"""Exact 44b6 selector dose-response and controlled-AUC requirement diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import gzip
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
from zarr.storage import LocalStore, ZipStore


ORIGINAL_ROOT = Path("/Users/agent/Desktop/cell_tracking")
ROOT = Path(os.environ.get("CELLTRACK_ROOT", str(ORIGINAL_ROOT))).resolve()
HERE = Path(__file__).resolve().parent
for location in (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "external/Tracking_Support/repo/src",
    ROOT / "external/kaggle-cell-tracking-competition/src",
):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_alpha.action_compiler import (  # noqa: E402
    ActionKind,
    GraphAction,
    compile_action,
    compile_batch,
    graph_sha256,
)
from celltrack_alpha.kaggle_calibration import read_submission  # noqa: E402
from celltrack_alpha.official_eval import (  # noqa: E402
    official_metric_row,
    official_summary,
)
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


SCHEMA = "celltrack-alpha.selector-requirement-44b6.v1"
PREREG = HERE / "PREREGISTRATION.json"
POP_ROOT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_repr/"
    "phase2_separation_gate/run_v1"
)
POP_MANIFEST = POP_ROOT / "population_manifest.json"
DELETE_RESULT = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_delete/graph_result.json"
)
SWAP_DIR = ROOT / "reports/g02/screening/tiered_surface/branches/sa_swap"
SWAP_IMPL = SWAP_DIR / "run_sa_swap.py"
SWAP_PREREG = SWAP_DIR / "PREREGISTRATION.json"
SWAP_RESULT = SWAP_DIR / "result_44b6_postrepair_v1/report.json"
RESTORE_DIR = ROOT / "reports/g02/screening/tiered_surface/branches/sa_nodetopo"
RESTORE_IMPL = RESTORE_DIR / "evaluate_nodetopo_44b6_graph_gate.py"
RESTORE_ACTIONS = (
    ROOT
    / "reports/g02/screening/tiered_surface/branches/sa_nodetopo_gen/"
    "generated/action_population.jsonl.gz"
)
RESTORE_RESULT = RESTORE_DIR / "graph_gate_44b6/report.json"
P0 = ROOT / "reports/g02/screening/action_class_oracle/p0_action_class_oracle_v1.json"
RAW_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
RAW_ROOT = (
    Path(os.environ["CELLTRACK_RAW_ROOT"]).resolve()
    if os.environ.get("CELLTRACK_RAW_ROOT")
    else None
)
ACTION_COMPILER = ROOT / "src/celltrack_alpha/action_compiler.py"

PINS = {
    PREREG: "4f8e15095003f14ad2f3e839a251cf40ce71c2744d53e4a82ca8cd9febf0d001",
    POP_MANIFEST: "4ab7bb237b20dcf01c91af0771ef4022a6a3e5bacc900ce956e6b12777076f0c",
    DELETE_RESULT: "28517a7c9170cedbf550d55395f38ac330888949ead7467086c3e306c7f7485a",
    SWAP_RESULT: "62953d270ae383c9bacdea9ab33a3f6980358a31a329b219447460abd5604864",
    RESTORE_ACTIONS: "6be1c77b4869c0a83eb3e540f13b16d71ba097b30bd808647d688bea0a5a5324",
    RESTORE_RESULT: "73c02c27fa6a2fc46a8e7453c40ae630e946ddd2a0b29a679f0dd40ea5ec8a14",
    ACTION_COMPILER: "457f6607a493d51d972913fa155d687e9ffc4045475605bc109a4ca9e90124f5",
}
RAW_ARCHIVE_BYTES = 87_393_127_165
TARGET_AUCS = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 0.99)
SEED_COUNT = 20
AUC_TOLERANCE = 5.0e-7
GRAPH_TARGET = 0.005
CG0_RATE = 848.0 / 4_022_839.0


class DiagnosticError(RuntimeError):
    """The frozen diagnostic contract could not be preserved."""


def resolve_repo_path(value: str | Path) -> Path:
    """Remap frozen absolute repository paths into a transported repository."""
    path = Path(value)
    if path.is_absolute():
        try:
            return ROOT / path.relative_to(ORIGINAL_ROOT)
        except ValueError:
            return path
    return ROOT / path


def verify_raw_transport(identities: Sequence[str]) -> None:
    if RAW_ROOT is None:
        if RAW_ARCHIVE.stat().st_size != RAW_ARCHIVE_BYTES:
            raise DiagnosticError("raw 44b6 archive byte size drifted")
        return
    if not RAW_ROOT.is_dir():
        raise DiagnosticError(f"extracted competition root missing: {RAW_ROOT}")
    missing = [
        identity
        for identity in identities
        if not (RAW_ROOT / "train" / f"{identity.split(':', 1)[1]}.zarr").is_dir()
    ]
    if missing:
        raise DiagnosticError(
            f"extracted competition transport lacks {len(missing)} frozen videos"
        )


@contextmanager
def open_raw_store():
    if RAW_ROOT is not None:
        verify_raw_transport(())
        store = LocalStore(RAW_ROOT, read_only=True)
    else:
        store = ZipStore(RAW_ARCHIVE, mode="r")
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


@dataclass
class EdgePopulation:
    action_class: str
    identities: tuple[str, ...]
    graphs: tuple[Any, ...]
    baseline_rows: tuple[dict[str, Any], ...]
    video: np.ndarray
    source: np.ndarray
    target: np.ndarray
    delta_s: np.ndarray
    label: np.ndarray
    fold: np.ndarray
    budgets: np.ndarray
    d_tp: np.ndarray
    d_fp: np.ndarray
    d_fn: np.ndarray
    conflicts: tuple[tuple[tuple[int, int], ...], ...] | None = None


@dataclass
class RestorePopulation:
    identities: tuple[str, ...]
    graphs: tuple[Any, ...]
    baseline_rows: tuple[dict[str, Any], ...]
    video: np.ndarray
    actions: tuple[GraphAction, ...]
    delta_s: np.ndarray
    label: np.ndarray
    fold: np.ndarray
    budgets: np.ndarray
    local_bits: np.ndarray
    subset_rows: tuple[dict[int, dict[str, Any]], ...]


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
        raise DiagnosticError(f"cannot read JSON {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON root is not an object: {path}")
    return value


def canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise DiagnosticError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fold_for(identity: str) -> int:
    digest = hashlib.sha256(identity.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % 5


def support_grid(size: int) -> list[int]:
    if size < 1:
        return [0]
    values = [0]
    current = 1
    while current <= size:
        values.append(current)
        current *= 2
    if values[-1] != size:
        values.append(size)
    return values


def edge_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if float(row["edge_tp"]) == float(row["edge_tp"])]
    if not valid:
        raise DiagnosticError("cannot summarize an empty exact edge population")
    tp = sum(int(float(row["edge_tp"])) for row in valid)
    fp = sum(int(float(row["edge_fp"])) for row in valid)
    fn = sum(int(float(row["edge_fn"])) for row in valid)
    denominator = tp + fp + fn
    raw = tp / denominator if denominator else float("nan")
    weights = [
        int(float(row["edge_tp"]))
        + int(float(row["edge_fp"]))
        + int(float(row["edge_fn"]))
        for row in valid
    ]
    total_weight = sum(weights)
    adjusted = (
        sum(weight * float(row["adj_edge_jaccard"]) for weight, row in zip(weights, valid))
        / total_weight
        if total_weight
        else float("nan")
    )
    return {
        "n": len(valid),
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_jaccard": raw,
        "adj_edge_jaccard": adjusted,
        "node_recall": sum(float(row["node_recall"]) for row in valid) / len(valid),
        "n_adj": len(valid),
    }


def edge_candidate_rows(
    population: EdgePopulation,
    delta_tp: np.ndarray,
    delta_fp: np.ndarray,
    delta_fn: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, baseline in enumerate(population.baseline_rows):
        row = dict(baseline)
        tp = int(float(row["edge_tp"])) + int(delta_tp[index])
        fp = int(float(row["edge_fp"])) + int(delta_fp[index])
        fn = int(float(row["edge_fn"])) + int(delta_fn[index])
        if min(tp, fp, fn) < 0:
            raise DiagnosticError("action effects produced negative official counts")
        denominator = tp + fp + fn
        raw = tp / denominator if denominator else float("nan")
        baseline_raw = float(baseline["edge_jaccard"])
        if baseline_raw <= 0.0:
            raise DiagnosticError("baseline adjustment factor is undefined")
        factor = float(baseline["adj_edge_jaccard"]) / baseline_raw
        row["edge_tp"] = tp
        row["edge_fp"] = fp
        row["edge_fn"] = fn
        row["edge_jaccard"] = raw
        row["adj_edge_jaccard"] = max(0.0, raw * factor)
        rows.append(row)
    return rows


def metric_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "adjusted_edge_jaccard": float(candidate["adj_edge_jaccard"])
        - float(baseline["adj_edge_jaccard"]),
        "raw_edge_jaccard": float(candidate["edge_jaccard"])
        - float(baseline["edge_jaccard"]),
        "node_recall": float(candidate["node_recall"])
        - float(baseline["node_recall"]),
    }


def empirical_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = np.sort(scores[labels == 0])
    if not len(positive) or not len(negative):
        raise DiagnosticError("AUC is undefined for a class-degenerate population")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float(np.mean((lower + upper) * 0.5) / len(negative))


def controlled_scores(
    labels: np.ndarray,
    action_class: str,
    seed_index: int,
    target_auc: float,
) -> tuple[np.ndarray, float, float]:
    if not 0.5 < target_auc < 1.0:
        raise DiagnosticError("target AUC must be strictly between 0.5 and 1")
    namespace = (
        f"SELECTOR-REQUIREMENT-44B6|{action_class}|{seed_index}".encode("ascii")
    )
    seed = int.from_bytes(hashlib.sha256(namespace).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    base = rng.standard_normal(len(labels), dtype=np.float64)
    positive_base = base[labels == 1]
    negative_sorted = np.sort(base[labels == 0])
    if not len(positive_base) or not len(negative_sorted):
        raise DiagnosticError("controlled scoring requires both action classes")

    def auc_at(shift: float) -> float:
        shifted = positive_base + shift
        lower = np.searchsorted(negative_sorted, shifted, side="left")
        upper = np.searchsorted(negative_sorted, shifted, side="right")
        return float(np.mean((lower + upper) * 0.5) / len(negative_sorted))

    center_low, center_high = -8.0, 8.0
    if not auc_at(center_low) <= 0.5 <= auc_at(center_high):
        raise DiagnosticError("cannot center the deterministic corruption at AUC 0.5")
    for _ in range(64):
        middle = (center_low + center_high) * 0.5
        if auc_at(middle) < 0.5:
            center_low = middle
        else:
            center_high = middle
    center = (center_low + center_high) * 0.5
    low, high = center, 8.0
    if not auc_at(low) <= target_auc <= auc_at(high):
        raise DiagnosticError("target AUC is outside the deterministic corruption range")
    for _ in range(64):
        middle = (low + high) * 0.5
        if auc_at(middle) < target_auc:
            low = middle
        else:
            high = middle
    candidates = (low, high, (low + high) * 0.5)
    best_shift = min(candidates, key=lambda value: abs(auc_at(value) - target_auc))
    scores = base.copy()
    scores[labels == 1] += best_shift
    observed = empirical_auc(labels, scores)
    lattice_tolerance = 0.5 / (len(positive_base) * len(negative_sorted))
    tolerance = max(AUC_TOLERANCE, lattice_tolerance + 1.0e-15)
    if abs(observed - target_auc) > tolerance:
        raise DiagnosticError(
            f"empirical AUC miss {observed} for target {target_auc}"
        )
    return scores, observed, best_shift - center


def retention_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    video: np.ndarray,
    fold: np.ndarray,
    budgets: np.ndarray,
) -> dict[str, Any]:
    selected_positive = 0
    selected_negative = 0
    total_positive = int(labels.sum())
    videos = []
    fold_counts = {
        value: {"selected_positive": 0, "selected_negative": 0, "total_positive": 0}
        for value in range(5)
    }
    for video_index in range(len(budgets)):
        indices = np.flatnonzero(video == video_index)
        local_labels = labels[indices]
        local_scores = scores[indices]
        harmful_scores = local_scores[local_labels == 0]
        budget = int(budgets[video_index])
        if budget < 0 or budget > len(harmful_scores):
            raise DiagnosticError("CG0 harmful-action budget is invalid")
        if budget == len(harmful_scores):
            chosen = np.ones(len(indices), dtype=bool)
        else:
            kth = len(harmful_scores) - budget - 1
            cutoff = float(np.partition(harmful_scores, kth)[kth])
            chosen = local_scores > cutoff
        beneficial = int(local_labels[chosen].sum())
        harmful = int((local_labels[chosen] == 0).sum())
        if harmful > budget:
            raise DiagnosticError("retention selector exceeded a per-video CG0 budget")
        local_fold = int(fold[indices[0]])
        if np.any(fold[indices] != local_fold):
            raise DiagnosticError("one video spans multiple folds")
        local_total = int(local_labels.sum())
        selected_positive += beneficial
        selected_negative += harmful
        fold_counts[local_fold]["selected_positive"] += beneficial
        fold_counts[local_fold]["selected_negative"] += harmful
        fold_counts[local_fold]["total_positive"] += local_total
        videos.append(
            {
                "video_index": video_index,
                "budget": budget,
                "selected_beneficial": beneficial,
                "selected_harmful": harmful,
                "total_beneficial": local_total,
            }
        )
    folds = []
    for fold_index, counts in fold_counts.items():
        denominator = counts["total_positive"]
        folds.append(
            {
                "fold": fold_index,
                **counts,
                "retention": (
                    counts["selected_positive"] / denominator
                    if denominator
                    else None
                ),
            }
        )
    return {
        "retention": selected_positive / total_positive if total_positive else None,
        "selected_beneficial": selected_positive,
        "selected_harmful": selected_negative,
        "total_beneficial": total_positive,
        "folds": folds,
        "videos": videos,
    }


def fold_aucs(
    labels: np.ndarray, scores: np.ndarray, fold: np.ndarray
) -> list[dict[str, Any]]:
    result = []
    for fold_index in range(5):
        held = fold == fold_index
        local = labels[held]
        if not held.any() or not np.any(local == 1) or not np.any(local == 0):
            result.append({"fold": fold_index, "auc": None})
        else:
            result.append(
                {"fold": fold_index, "auc": empirical_auc(local, scores[held])}
            )
    return result


def spread(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise DiagnosticError("cannot summarize empty or nonfinite values")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "maximum": float(array.max()),
    }


def _verify_pins() -> dict[str, str]:
    observed = {}
    for path, expected in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise DiagnosticError(
                f"frozen input drifted: {path}; expected {expected}, got {actual}"
            )
        observed[str(path.relative_to(ROOT))] = actual
    manifest = load_json(POP_MANIFEST)
    identities = [
        str(row["identity"])
        for action_class in ("delete", "occupied_swap")
        for row in manifest.get("classes", {}).get(action_class, {}).get("videos", ())
    ]
    verify_raw_transport(identities)
    for action_class in ("delete", "occupied_swap"):
        rows = manifest.get("classes", {}).get(action_class, {}).get("videos", ())
        for row in rows:
            for path_key, hash_key in (
                ("population_path", "population_sha256"),
                ("request_path", "request_sha256"),
            ):
                path = ROOT / str(row[path_key])
                actual = sha256_file(path)
                if actual != row[hash_key]:
                    raise DiagnosticError(
                        f"{action_class} population member drifted: {path}"
                    )
    return observed


def _load_npz_class(
    action_class: str, manifest: Mapping[str, Any]
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows = manifest["classes"][action_class]["videos"]
    identities = tuple(str(row["identity"]) for row in rows)
    source_chunks = []
    target_chunks = []
    delta_chunks = []
    label_chunks = []
    fold_chunks = []
    video_chunks = []
    budgets = []
    for video_index, row in enumerate(rows):
        path = ROOT / str(row["population_path"])
        with np.load(path, allow_pickle=False) as data:
            count = len(data["oracle_label"])
            source_chunks.append(np.asarray(data["source_id"], dtype=np.int64))
            target_chunks.append(np.asarray(data["target_id"], dtype=np.int64))
            delta_chunks.append(np.asarray(data["oracle_delta_S"], dtype=np.float64))
            label_chunks.append(np.asarray(data["oracle_label"], dtype=np.uint8))
            fold_chunks.append(np.asarray(data["fold"], dtype=np.int8))
            video_chunks.append(np.full(count, video_index, dtype=np.int16))
            budget = int(data["fp_budget"][0])
        if (
            count != int(row["count"])
            or int(label_chunks[-1].sum()) != int(row["beneficial"])
            or count - int(label_chunks[-1].sum()) != int(row["harmful"])
            or budget != int(row["fp_budget"])
            or np.any(label_chunks[-1] != (delta_chunks[-1] > 0.0))
            or np.any(fold_chunks[-1] != fold_for(str(row["identity"])))
        ):
            raise DiagnosticError(f"{action_class} frozen population content drifted")
        budgets.append(budget)
    return (
        identities,
        np.concatenate(video_chunks),
        np.concatenate(source_chunks),
        np.concatenate(target_chunks),
        np.concatenate(delta_chunks),
        np.concatenate(label_chunks),
        np.concatenate(fold_chunks),
        np.asarray(budgets, dtype=np.int64),
    )


def load_edge_populations(
    swap_module: Any,
) -> tuple[EdgePopulation, EdgePopulation, dict[str, Any]]:
    manifest = load_json(POP_MANIFEST)
    p0 = load_json(P0)
    delete_result = load_json(DELETE_RESULT)["family_results"]["44b6"]
    delete_video_results = {
        str(row["identity"]): row for row in delete_result["videos"]
    }
    delete_arrays = _load_npz_class("delete", manifest)
    swap_arrays = _load_npz_class("occupied_swap", manifest)
    delete_ids = delete_arrays[0]
    swap_ids = swap_arrays[0]
    fragment_root = ROOT / str(p0["control"]["fragment_root"])

    swap_prereg = swap_module.authenticate(SWAP_PREREG, swap_module.PREREG_SHA256)
    swap_records, swap_control = swap_module._control_records(swap_prereg)
    if tuple(sorted(swap_records)) != tuple(sorted(swap_ids)):
        raise DiagnosticError("occupied-swap control identity coverage drifted")

    delete_graphs = []
    delete_rows = []
    for identity in delete_ids:
        dataset = identity.split(":", 1)[1]
        path = fragment_root / f"{identity.replace(':', '__', 1)}.csv"
        graph = read_submission(path)[dataset]
        record = delete_video_results[identity]
        if graph_sha256(graph) != record["before_graph_sha256"]:
            raise DiagnosticError(f"delete baseline graph drifted: {identity}")
        delete_graphs.append(graph)
        delete_rows.append(dict(record["baseline_metrics"]))
    if metric_delta(
        edge_summary(delete_rows), delete_result["control"]
    ) != {"adjusted_edge_jaccard": 0.0, "raw_edge_jaccard": 0.0, "node_recall": 0.0}:
        raise DiagnosticError("delete baseline summary does not reproduce")

    swap_graphs = []
    swap_rows = []
    delete_dtp = np.empty(len(delete_arrays[1]), dtype=np.int8)
    delete_dfp = np.empty(len(delete_arrays[1]), dtype=np.int8)
    delete_dfn = np.empty(len(delete_arrays[1]), dtype=np.int8)
    swap_dtp = np.empty(len(swap_arrays[1]), dtype=np.int8)
    swap_dfp = np.empty(len(swap_arrays[1]), dtype=np.int8)
    swap_dfn = np.empty(len(swap_arrays[1]), dtype=np.int8)
    swap_conflicts: list[tuple[tuple[int, int], ...]] = [
        () for _ in range(len(swap_arrays[1]))
    ]

    delete_offsets = np.cumsum(
        [0]
        + [
            int(row["count"])
            for row in manifest["classes"]["delete"]["videos"]
        ]
    )
    swap_offsets = np.cumsum(
        [0]
        + [
            int(row["count"])
            for row in manifest["classes"]["occupied_swap"]["videos"]
        ]
    )
    with open_raw_store() as store:
        for video_index, identity in enumerate(delete_ids):
            dataset = identity.split(":", 1)[1]
            graph = delete_graphs[video_index]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            pred_to_gt, _ = swap_module.official_node_matching(graph, gt, scale)
            truth, gt_out, gt_in, _ = swap_module._truth_topology(gt)
            start, end = delete_offsets[video_index : video_index + 2]
            tp = fp = 0
            for global_index in range(int(start), int(end)):
                pair = (
                    int(delete_arrays[2][global_index]),
                    int(delete_arrays[3][global_index]),
                )
                is_true, is_valid = swap_module._edge_class(
                    pair, pred_to_gt, truth, gt_out, gt_in
                )
                delete_dtp[global_index] = -int(is_true)
                delete_dfp[global_index] = -int(is_valid and not is_true)
                delete_dfn[global_index] = int(is_true)
                tp += int(is_true)
                fp += int(is_valid and not is_true)
            baseline = delete_rows[video_index]
            if tp != int(baseline["edge_tp"]) or fp != int(baseline["edge_fp"]):
                raise DiagnosticError(f"delete edge truth replay drifted: {identity}")

        for video_index, identity in enumerate(swap_ids):
            dataset = identity.split(":", 1)[1]
            graph = swap_module._graph(
                dataset, Path(swap_records[identity]["resolved_path"])
            )
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            baseline = official_metric_row(nodes, edges, gt, scale)
            pred_to_gt, _ = swap_module.official_node_matching(graph, gt, scale)
            truth, gt_out, gt_in, _ = swap_module._truth_topology(gt)
            baseline_edges, _outgoing, _incoming = swap_module._edge_sets(graph)
            truth_by_edge = {
                pair: swap_module._edge_class(
                    pair, pred_to_gt, truth, gt_out, gt_in
                )
                for pair in baseline_edges
            }
            start, end = swap_offsets[video_index : video_index + 2]
            for global_index in range(int(start), int(end)):
                source = int(swap_arrays[2][global_index])
                target = int(swap_arrays[3][global_index])
                pair = (source, target)
                conflicts = tuple(
                    sorted(
                        edge
                        for edge in baseline_edges
                        if edge[0] == source or edge[1] == target
                    )
                )
                if len(conflicts) not in (1, 2):
                    raise DiagnosticError("occupied-swap conflict boundary drifted")
                added_true, added_valid = swap_module._edge_class(
                    pair, pred_to_gt, truth, gt_out, gt_in
                )
                removed_true = sum(int(truth_by_edge[edge][0]) for edge in conflicts)
                removed_fp = sum(
                    int(truth_by_edge[edge][1] and not truth_by_edge[edge][0])
                    for edge in conflicts
                )
                swap_dtp[global_index] = int(added_true) - removed_true
                swap_dfp[global_index] = (
                    int(added_valid and not added_true) - removed_fp
                )
                swap_dfn[global_index] = -swap_dtp[global_index]
                swap_conflicts[global_index] = conflicts
            swap_graphs.append(graph)
            swap_rows.append(baseline)

    swap_baseline = edge_summary(swap_rows)
    frozen_swap = swap_control["graph"]["control"]["summary"]
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall"):
        if not math.isclose(
            float(swap_baseline[key]), float(frozen_swap[key]), abs_tol=1e-12
        ):
            raise DiagnosticError(f"occupied-swap baseline summary drifted: {key}")
    delete = EdgePopulation(
        "delete",
        delete_ids,
        tuple(delete_graphs),
        tuple(delete_rows),
        *delete_arrays[1:],
        delete_dtp,
        delete_dfp,
        delete_dfn,
        None,
    )
    swap = EdgePopulation(
        "occupied_swap",
        swap_ids,
        tuple(swap_graphs),
        tuple(swap_rows),
        *swap_arrays[1:],
        swap_dtp,
        swap_dfp,
        swap_dfn,
        tuple(swap_conflicts),
    )
    return delete, swap, {
        "delete": edge_summary(delete_rows),
        "occupied_swap": swap_baseline,
    }


def load_restore_population(restore_module: Any) -> RestorePopulation:
    restore_module._verify_pins()
    action_rows = restore_module._load_actions()
    raw_manifest = load_json(restore_module.RAW_MANIFEST)
    baseline_manifest = load_json(restore_module.BASELINE_MANIFEST)
    identities = tuple(str(row["identity"]) for row in raw_manifest["videos"])
    by_identity: dict[str, list[GraphAction]] = defaultdict(list)
    for identity, action in action_rows:
        by_identity[identity].append(action)
    graphs = []
    baseline_rows = []
    subset_rows: list[dict[int, dict[str, Any]]] = []
    action_video = []
    action_objects = []
    local_bits = []
    with open_raw_store() as store:
        for video_index, identity in enumerate(identities):
            dataset = identity.split(":", 1)[1]
            path = restore_module.BASELINE_FRAGMENTS / (
                f"{identity.replace(':', '__', 1)}.csv"
            )
            expected = baseline_manifest["videos"][identity]["files"]["fragments"][
                "sha256"
            ]
            if sha256_file(path) != expected:
                raise DiagnosticError(f"restore baseline fragment drifted: {identity}")
            graph = read_submission(path)[dataset]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, graph)
            baseline = official_metric_row(nodes, edges, gt, scale)
            local_actions = by_identity.get(identity, [])
            table: dict[int, dict[str, Any]] = {0: baseline}
            for local_index, action in enumerate(local_actions):
                action_video.append(video_index)
                action_objects.append(action)
                local_bits.append(1 << local_index)
            for mask in range(1, 1 << len(local_actions)):
                chosen = [
                    action
                    for local_index, action in enumerate(local_actions)
                    if mask & (1 << local_index)
                ]
                compiled = compile_batch(graph, chosen)
                candidate_nodes, candidate_edges = scoreall._prediction_rows(
                    dataset, compiled.graph
                )
                table[mask] = official_metric_row(
                    candidate_nodes, candidate_edges, gt, scale
                )
            graphs.append(graph)
            baseline_rows.append(baseline)
            subset_rows.append(table)
    baseline_summary = official_summary(baseline_rows)
    frozen = load_json(RESTORE_RESULT)["graph"]["baseline"]["summary"]
    for key in ("adj_edge_jaccard", "edge_jaccard", "node_recall", "score"):
        if not math.isclose(
            float(baseline_summary[key]), float(frozen[key]), abs_tol=1e-12
        ):
            raise DiagnosticError(f"restore baseline summary drifted: {key}")
    action_video_array = np.asarray(action_video, dtype=np.int16)
    local_bits_array = np.asarray(local_bits, dtype=np.int16)
    delta = np.empty(len(action_objects), dtype=np.float64)
    for index, video_index in enumerate(action_video_array):
        rows = list(baseline_rows)
        rows[int(video_index)] = subset_rows[int(video_index)][
            int(local_bits_array[index])
        ]
        delta[index] = (
            float(official_summary(rows)["score"]) - float(baseline_summary["score"])
        )
    labels = (delta > 0.0).astype(np.uint8)
    folds = np.asarray(
        [fold_for(identities[int(index)]) for index in action_video_array],
        dtype=np.int8,
    )
    budgets = np.asarray(
        [
            math.floor(CG0_RATE * int(np.sum((action_video_array == index) & (labels == 0))))
            for index in range(len(identities))
        ],
        dtype=np.int64,
    )
    return RestorePopulation(
        identities,
        tuple(graphs),
        tuple(baseline_rows),
        action_video_array,
        tuple(action_objects),
        delta,
        labels,
        folds,
        budgets,
        local_bits_array,
        tuple(subset_rows),
    )


def oracle_order(delta_s: np.ndarray) -> np.ndarray:
    canonical = np.arange(len(delta_s), dtype=np.int64)
    return np.lexsort((canonical, -delta_s))


def score_edge_order(
    population: EdgePopulation, order: np.ndarray, supports: Sequence[int]
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    baseline = edge_summary(population.baseline_rows)
    curve = []
    selections = []
    if population.action_class == "delete":
        for prefix in supports:
            selected = np.asarray(order[:prefix], dtype=np.int64)
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
            summary = edge_summary(
                edge_candidate_rows(population, delta_tp, delta_fp, delta_fn)
            )
            curve.append(
                {
                    "candidate_prefix_N": int(prefix),
                    "applied_support": int(len(selected)),
                    "graph_delta": metric_delta(baseline, summary),
                }
            )
            selections.append(selected)
        return curve, selections

    if population.conflicts is None:
        raise DiagnosticError("occupied-swap population lacks conflict ownership")
    support_set = set(int(value) for value in supports)
    delta_tp = np.zeros(len(population.identities), dtype=np.int64)
    delta_fp = np.zeros(len(population.identities), dtype=np.int64)
    delta_fn = np.zeros(len(population.identities), dtype=np.int64)
    owned_sources = [set() for _ in population.identities]
    owned_targets = [set() for _ in population.identities]
    owned_edges = [set() for _ in population.identities]
    accepted: list[int] = []
    snapshot: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    if 0 in support_set:
        snapshot[0] = (
            {
                "candidate_prefix_N": 0,
                "applied_support": 0,
                "graph_delta": metric_delta(baseline, baseline),
            },
            np.asarray([], dtype=np.int64),
        )
    for position, action_index in enumerate(order, start=1):
        index = int(action_index)
        video_index = int(population.video[index])
        source = int(population.source[index])
        target = int(population.target[index])
        conflicts = set(population.conflicts[index])
        if (
            source not in owned_sources[video_index]
            and target not in owned_targets[video_index]
            and not conflicts.intersection(owned_edges[video_index])
        ):
            owned_sources[video_index].add(source)
            owned_targets[video_index].add(target)
            owned_edges[video_index].update(conflicts)
            accepted.append(index)
            delta_tp[video_index] += int(population.d_tp[index])
            delta_fp[video_index] += int(population.d_fp[index])
            delta_fn[video_index] += int(population.d_fn[index])
        if position in support_set:
            summary = edge_summary(
                edge_candidate_rows(population, delta_tp, delta_fp, delta_fn)
            )
            snapshot[position] = (
                {
                    "candidate_prefix_N": position,
                    "applied_support": len(accepted),
                    "batch_conflict_excluded": position - len(accepted),
                    "graph_delta": metric_delta(baseline, summary),
                },
                np.asarray(accepted, dtype=np.int64),
            )
    for support in supports:
        row, selected = snapshot[int(support)]
        curve.append(row)
        selections.append(selected)
    return curve, selections


def score_restore_order(
    population: RestorePopulation, order: np.ndarray, supports: Sequence[int]
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    baseline = official_summary(list(population.baseline_rows))
    curve = []
    selections = []
    for prefix in supports:
        selected = np.asarray(order[:prefix], dtype=np.int64)
        masks = np.zeros(len(population.identities), dtype=np.int16)
        for action_index in selected:
            video_index = int(population.video[action_index])
            masks[video_index] |= int(population.local_bits[action_index])
        rows = [
            population.subset_rows[index][int(mask)]
            for index, mask in enumerate(masks)
        ]
        summary = official_summary(rows)
        curve.append(
            {
                "candidate_prefix_N": int(prefix),
                "applied_support": int(len(selected)),
                "graph_delta": metric_delta(baseline, summary),
                "combined_score_delta": float(summary["score"])
                - float(baseline["score"]),
            }
        )
        selections.append(selected)
    return curve, selections


def _swap_action(population: EdgePopulation, index: int) -> GraphAction:
    if population.conflicts is None:
        raise DiagnosticError("swap action requested from non-swap population")
    source = int(population.source[index])
    target = int(population.target[index])
    return GraphAction(
        action_id=f"diagnostic.swap:{source}:{target}",
        kind=ActionKind.SWAP,
        add_edges=((source, target),),
        replace_outgoing_from=(source,),
        replace_incoming_to=(target,),
        expected_conflict_edges=population.conflicts[index],
    )


def replay_compiler(
    population: EdgePopulation,
    selected: np.ndarray,
) -> dict[str, Any]:
    by_video: dict[int, list[int]] = defaultdict(list)
    for index in selected:
        by_video[int(population.video[index])].append(int(index))
    receipts = []
    for video_index, indices in sorted(by_video.items()):
        graph = population.graphs[video_index]
        if population.action_class == "delete":
            action = GraphAction(
                action_id=f"diagnostic.delete:{video_index}",
                kind=ActionKind.DELETE_EDGES,
                delete_edges=tuple(
                    (int(population.source[index]), int(population.target[index]))
                    for index in indices
                ),
            )
            compiled = compile_action(graph, action)
        else:
            actions = [_swap_action(population, index) for index in indices]
            compiled = compile_batch(graph, actions)
        receipts.append(
            {
                "video_index": video_index,
                "action_count": len(indices),
                "before_graph_sha256": compiled.before_graph_sha256,
                "after_graph_sha256": compiled.after_graph_sha256,
            }
        )
    return {
        "affected_videos": len(receipts),
        "compiled_actions": int(len(selected)),
        "receipts": receipts,
    }


def replay_restore_compiler(
    population: RestorePopulation, selected: np.ndarray
) -> dict[str, Any]:
    by_video: dict[int, list[GraphAction]] = defaultdict(list)
    for index in selected:
        by_video[int(population.video[index])].append(population.actions[int(index)])
    receipts = []
    for video_index, actions in sorted(by_video.items()):
        compiled = compile_batch(population.graphs[video_index], actions)
        receipts.append(
            {
                "video_index": video_index,
                "action_count": len(actions),
                "before_graph_sha256": compiled.before_graph_sha256,
                "after_graph_sha256": compiled.after_graph_sha256,
            }
        )
    return {
        "affected_videos": len(receipts),
        "compiled_actions": int(len(selected)),
        "receipts": receipts,
    }


def curve_result(
    action_class: str,
    population: EdgePopulation | RestorePopulation,
    order: np.ndarray,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    supports = support_grid(len(order))
    if isinstance(population, EdgePopulation):
        return score_edge_order(population, order, supports)
    return score_restore_order(population, order, supports)


def aggregate_noise(
    action_class: str, seed_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_runs:
        grouped[float(row["target_auc"])].append(row)
    results = []
    for target_auc in TARGET_AUCS:
        runs = grouped[float(target_auc)]
        if len(runs) != SEED_COUNT:
            raise DiagnosticError(f"{action_class} AUC {target_auc} lacks 20 seeds")
        supports = [int(row["candidate_prefix_N"]) for row in runs[0]["curve"]]
        curves = []
        for position, support in enumerate(supports):
            adjusted = [
                float(run["curve"][position]["graph_delta"]["adjusted_edge_jaccard"])
                for run in runs
            ]
            raw = [
                float(run["curve"][position]["graph_delta"]["raw_edge_jaccard"])
                for run in runs
            ]
            node = [
                float(run["curve"][position]["graph_delta"]["node_recall"])
                for run in runs
            ]
            applied = [
                float(run["curve"][position]["applied_support"]) for run in runs
            ]
            curves.append(
                {
                    "candidate_prefix_N": support,
                    "applied_support": spread(applied),
                    "adjusted_edge_jaccard_delta": spread(adjusted),
                    "raw_edge_jaccard_delta": spread(raw),
                    "node_recall_delta": spread(node),
                }
            )
        results.append(
            {
                "target_auc": target_auc,
                "empirical_auc": spread([run["empirical_auc"] for run in runs]),
                "retention": spread([run["retention"]["retention"] for run in runs]),
                "fold_auc": [
                    {
                        "fold": fold_index,
                        "auc": spread(
                            [
                                run["fold_auc"][fold_index]["auc"]
                                for run in runs
                                if run["fold_auc"][fold_index]["auc"] is not None
                            ]
                        ),
                    }
                    for fold_index in range(5)
                ],
                "fold_retention": [
                    {
                        "fold": fold_index,
                        "retention": spread(
                            [
                                run["retention"]["folds"][fold_index]["retention"]
                                for run in runs
                                if run["retention"]["folds"][fold_index]["retention"]
                                is not None
                            ]
                        ),
                    }
                    for fold_index in range(5)
                ],
                "curve": curves,
            }
        )
    return {"action_class": action_class, "auc_results": results}


def requirement_for(
    oracle_curve: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any] | None,
    threshold: float,
) -> dict[str, Any]:
    oracle_best = max(
        oracle_curve,
        key=lambda row: float(row["graph_delta"]["adjusted_edge_jaccard"]),
    )
    oracle_ceiling = float(
        oracle_best["graph_delta"]["adjusted_edge_jaccard"]
    )
    if oracle_ceiling < threshold:
        return {
            "status": "INFEASIBLE_ON_FROZEN_POPULATION",
            "threshold": threshold,
            "oracle_ceiling": oracle_ceiling,
            "oracle_best_support": int(oracle_best["candidate_prefix_N"]),
            "mean_requirement": None,
            "robust_p05_requirement": None,
        }
    if aggregate is None:
        return {
            "status": "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE",
            "threshold": threshold,
            "oracle_ceiling": oracle_ceiling,
            "oracle_best_support": int(oracle_best["candidate_prefix_N"]),
            "mean_requirement": None,
            "robust_p05_requirement": None,
        }
    mean_requirement = None
    robust_requirement = None
    trace = []
    for auc_result in aggregate["auc_results"]:
        best_mean = max(
            auc_result["curve"],
            key=lambda row: float(row["adjusted_edge_jaccard_delta"]["mean"]),
        )
        best_p05 = max(
            auc_result["curve"],
            key=lambda row: float(row["adjusted_edge_jaccard_delta"]["p05"]),
        )
        row = {
            "target_auc": float(auc_result["target_auc"]),
            "empirical_auc": auc_result["empirical_auc"],
            "retention": auc_result["retention"],
            "best_mean": best_mean,
            "best_p05": best_p05,
        }
        trace.append(row)
        if (
            mean_requirement is None
            and float(best_mean["adjusted_edge_jaccard_delta"]["mean"]) >= threshold
        ):
            mean_requirement = {
                "target_auc": row["target_auc"],
                "empirical_auc": row["empirical_auc"],
                "retention": row["retention"],
                "candidate_prefix_N": int(best_mean["candidate_prefix_N"]),
                "applied_support": best_mean["applied_support"],
                "adjusted_delta": best_mean["adjusted_edge_jaccard_delta"],
            }
        if (
            robust_requirement is None
            and float(best_p05["adjusted_edge_jaccard_delta"]["p05"]) >= threshold
        ):
            robust_requirement = {
                "target_auc": row["target_auc"],
                "empirical_auc": row["empirical_auc"],
                "retention": row["retention"],
                "candidate_prefix_N": int(best_p05["candidate_prefix_N"]),
                "applied_support": best_p05["applied_support"],
                "adjusted_delta": best_p05["adjusted_edge_jaccard_delta"],
            }
    return {
        "status": "MEASURED_ON_TESTED_GRID",
        "threshold": threshold,
        "oracle_ceiling": oracle_ceiling,
        "oracle_best_support": int(oracle_best["candidate_prefix_N"]),
        "mean_requirement": mean_requirement,
        "robust_p05_requirement": robust_requirement,
        "trace": trace,
    }


def threshold_verdict(target_requirement: Mapping[str, Any]) -> dict[str, Any]:
    if target_requirement["status"] != "MEASURED_ON_TESTED_GRID":
        return {
            "verdict": "NOT_APPLICABLE_POPULATION_LIMIT",
            "reason": target_requirement["status"],
        }
    requirement = target_requirement.get("mean_requirement")
    if requirement is None:
        return {
            "verdict": "TOO_LENIENT",
            "reason": "No tested AUC through 0.99 reached +0.005 in mean.",
        }
    auc_needed = float(requirement["target_auc"])
    retention_needed = float(requirement["retention"]["mean"])
    sufficient = auc_needed <= 0.65 and retention_needed <= 0.15
    return {
        "verdict": "SUFFICIENT" if sufficient else "TOO_LENIENT",
        "required_tested_auc": auc_needed,
        "required_mean_retention": retention_needed,
        "frozen_auc_floor": 0.65,
        "frozen_retention_floor": 0.15,
        "reason": (
            "Both measured mean requirements are at or below the frozen floors."
            if sufficient
            else "At least one measured mean requirement exceeds the frozen floor."
        ),
    }


def markdown_report(
    population_summary: Mapping[str, Mapping[str, int]],
    oracle_results: Mapping[str, Any],
    requirements: Mapping[str, Any],
    barrier: Mapping[str, Any],
) -> str:
    lines = [
        "# Selector requirement on frozen Phase-1 44b6 actions",
        "",
        f"**Barrier:** `{barrier['status']}`. This is diagnostic-only: no tier, "
        "threshold, plan, ledger, DEAD, GPU, Kaggle, RunPod, submission, 6bba, "
        "or CV57 action occurred.",
        "",
        "## Result",
        "",
        "| Action class | Actions | Beneficial | Oracle ceiling Δ adjusted | "
        "Mean +0.005 requirement | Frozen gate verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for action_class in ("delete", "occupied_swap", "restore_topology"):
        population = population_summary[action_class]
        req = requirements[action_class]["target_plus_0_005"]
        measured = req.get("mean_requirement")
        requirement_text = (
            f"AUC {measured['target_auc']:.2f}, retention "
            f"{measured['retention']['mean']:.4f}"
            if measured is not None
            else req["status"]
        )
        verdict = requirements[action_class]["frozen_gate_verdict"]["verdict"]
        lines.append(
            f"| `{action_class}` | {population['actions']:,} | "
            f"{population['beneficial']:,} | "
            f"{req['oracle_ceiling']:+.9f} | {requirement_text} | **{verdict}** |"
        )
    lines.extend(
        [
            "",
            "Requirements are discrete over the preregistered AUC and power-of-two "
            "support grids; no interpolation was used. The detailed JSON contains "
            "all oracle points, all 20-seed curves, mean/spread summaries, per-fold "
            "AUC/retention, breakeven requirements, compiler receipts, and exact "
            "support counts.",
            "",
            "## Governance consequence",
            "",
        ]
    )
    if barrier["status"] == "HUMAN_REVIEW_REQUIRED":
        lines.append(
            "At least one feasible action class needs a stricter selector than the "
            "frozen AUC 0.65 / retention 0.15 gate. An amendment is proposed, not "
            "applied. Execution halts here for human review."
        )
    else:
        lines.append(
            "The diagnostic did not create a threshold amendment condition. Frozen "
            "thresholds and all campaign tiers remain unchanged."
        )
    lines.append("")
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise DiagnosticError("the binding local zero-swap guard is required")
    if output.exists():
        raise DiagnosticError(f"refusing to overwrite output {output}")
    started = time.perf_counter()
    pins = _verify_pins()
    swap_module = load_module("selector_requirement_sa_swap", SWAP_IMPL)
    restore_module = load_module("selector_requirement_restore", RESTORE_IMPL)
    output.mkdir(parents=True)

    baselines: dict[str, Any] = {}
    population_summary: dict[str, dict[str, int]] = {}
    oracle_results = {}
    oracle_compiler = {}
    raw_noise = {}
    aggregate_noise_results = {}
    terminal_compiler = {}

    def process_population(
        action_class: str,
        population: EdgePopulation | RestorePopulation,
        baseline: Mapping[str, Any],
    ) -> None:
        baselines[action_class] = dict(baseline)
        population_summary[action_class] = {
            "actions": len(population.label),
            "beneficial": int(population.label.sum()),
            "harmful": int((population.label == 0).sum()),
            "videos": len(population.identities),
        }
        order = oracle_order(population.delta_s)
        curve, selections = curve_result(action_class, population, order)
        oracle_results[action_class] = {
            "action_class": action_class,
            "population": len(population.label),
            "beneficial": int(population.label.sum()),
            "harmful": int((population.label == 0).sum()),
            "support_grid": support_grid(len(population.label)),
            "curve": curve,
        }
        receipts = []
        for row, selected in zip(curve, selections):
            if isinstance(population, EdgePopulation):
                replay = replay_compiler(population, selected)
            else:
                replay = replay_restore_compiler(population, selected)
            receipts.append(
                {
                    "candidate_prefix_N": row["candidate_prefix_N"],
                    **replay,
                }
            )
        oracle_compiler[action_class] = receipts
        print(
            f"ORACLE {action_class} actions={len(population.label)} "
            f"beneficial={int(population.label.sum())}",
            flush=True,
        )

        labels = population.label
        if not np.any(labels == 1) or not np.any(labels == 0):
            raw_noise[action_class] = {
                "status": "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE",
                "beneficial": int(labels.sum()),
                "harmful": int((labels == 0).sum()),
                "seed_runs": [],
            }
            aggregate_noise_results[action_class] = None
            terminal_compiler[action_class] = []
            print(f"NOISE {action_class} class-degenerate; skipped", flush=True)
            return
        seed_runs = []
        compiler_runs = []
        for target_auc in TARGET_AUCS:
            for seed_index in range(SEED_COUNT):
                scores, observed_auc, shift = controlled_scores(
                    labels, action_class, seed_index, target_auc
                )
                order = np.lexsort((np.arange(len(scores), dtype=np.int64), -scores))
                curve, selections = curve_result(action_class, population, order)
                retention = retention_metrics(
                    labels,
                    scores,
                    population.video,
                    population.fold,
                    population.budgets,
                )
                run_row = {
                    "target_auc": target_auc,
                    "seed_index": seed_index,
                    "empirical_auc": observed_auc,
                    "positive_shift": shift,
                    "fold_auc": fold_aucs(labels, scores, population.fold),
                    "retention": retention,
                    "curve": curve,
                }
                seed_runs.append(run_row)
                terminal = selections[-1]
                if isinstance(population, EdgePopulation):
                    if action_class == "delete":
                        replay = {
                            "proof": "all delete prefixes are subsets of the frozen exact N1-valid edge population; the full-population delete replay is rank-invariant",
                            "compiled_actions": len(terminal),
                        }
                    else:
                        replay = replay_compiler(population, terminal)
                else:
                    replay = replay_restore_compiler(population, terminal)
                compiler_runs.append(
                    {
                        "target_auc": target_auc,
                        "seed_index": seed_index,
                        **replay,
                    }
                )
                print(
                    f"NOISE {action_class} auc={target_auc:.2f} "
                    f"seed={seed_index:02d} empirical={observed_auc:.8f}",
                    flush=True,
                )
        raw_noise[action_class] = {
            "status": "COMPLETE",
            "beneficial": int(labels.sum()),
            "harmful": int((labels == 0).sum()),
            "seed_runs": seed_runs,
        }
        aggregate_noise_results[action_class] = aggregate_noise(
            action_class, seed_runs
        )
        terminal_compiler[action_class] = compiler_runs

    # Keep only one scientific class live during outcome computation. The two
    # edge banks must be authenticated together by their frozen loader, but the
    # restore graph/subset bank is not opened until both edge classes are done.
    delete, swap, edge_baselines = load_edge_populations(swap_module)
    process_population("delete", delete, edge_baselines["delete"])
    del delete
    gc.collect()
    process_population("occupied_swap", swap, edge_baselines["occupied_swap"])
    del swap, edge_baselines
    gc.collect()
    restore = load_restore_population(restore_module)
    process_population(
        "restore_topology",
        restore,
        official_summary(list(restore.baseline_rows)),
    )
    del restore
    gc.collect()

    canonical_json(
        output / "dose_response.json",
        {
            "schema": f"{SCHEMA}.dose-response",
            "status": "COMPLETE",
            "primary_metric": "adjusted_edge_jaccard_delta",
            "baselines": baselines,
            "classes": oracle_results,
            "compiler_replay": oracle_compiler,
        },
    )

    canonical_json(
        output / "noise_surface.json",
        {
            "schema": f"{SCHEMA}.noise-surface",
            "status": "COMPLETE",
            "target_aucs": list(TARGET_AUCS),
            "seed_count": SEED_COUNT,
            "classes": raw_noise,
            "aggregates": aggregate_noise_results,
            "terminal_compiler_replay": terminal_compiler,
        },
    )

    requirements = {}
    for action_class in ("delete", "occupied_swap", "restore_topology"):
        target_req = requirement_for(
            oracle_results[action_class]["curve"],
            aggregate_noise_results[action_class],
            GRAPH_TARGET,
        )
        breakeven_req = requirement_for(
            oracle_results[action_class]["curve"],
            aggregate_noise_results[action_class],
            0.0,
        )
        requirements[action_class] = {
            "target_plus_0_005": target_req,
            "breakeven_0": breakeven_req,
            "frozen_gate_verdict": threshold_verdict(target_req),
        }
    too_lenient = [
        action_class
        for action_class, value in requirements.items()
        if value["frozen_gate_verdict"]["verdict"] == "TOO_LENIENT"
    ]
    barrier_status = (
        "HUMAN_REVIEW_REQUIRED" if too_lenient else "DIAGNOSTIC_COMPLETE"
    )
    barrier = {
        "schema": f"{SCHEMA}.barrier",
        "status": barrier_status,
        "amendment_proposed_for": too_lenient,
        "thresholds_changed": False,
        "tier_moves": 0,
        "promotions": 0,
        "dead_actions": 0,
        "ledger_written": False,
        "plan_changed": False,
        "gpu_action": False,
        "kaggle_action": False,
        "runpod_action": False,
        "submission_action": False,
        "family_6bba_used": False,
        "cv57_used": False,
        "external_action": False,
    }
    canonical_json(
        output / "requirement_surface.json",
        {
            "schema": f"{SCHEMA}.requirement-surface",
            "status": "COMPLETE",
            "grid_only": True,
            "interpolation": False,
            "requirements": requirements,
        },
    )
    canonical_json(output / "BARRIER.json", barrier)
    state_fragment = {
        "schema": f"{SCHEMA}.state-fragment",
        "status": barrier_status,
        "node": "SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC",
        "diagnostic_only": True,
        "requirements": requirements,
        "tier_moves": [],
        "threshold_changes": [],
        "proposed_amendment": bool(too_lenient),
        "human_review_required": bool(too_lenient),
        "hard_stops_preserved": barrier,
    }
    canonical_json(output / "state_fragment.json", state_fragment)

    ledger_lines = [
        "# Proposed ledger row — root review only",
        "",
        "| ID | Status | Evidence | Finding | Consequence |",
        "|---|---|---|---|---|",
        "| `SELECTOR-REQUIREMENT-44B6-DIAGNOSTIC` | "
        f"`{barrier_status}` (diagnostic; no tier move) | "
        "`dose_response.json`, `noise_surface.json`, `requirement_surface.json`, "
        "`BARRIER.json` | "
        + "; ".join(
            f"{name}: {value['frozen_gate_verdict']['verdict']}"
            for name, value in requirements.items()
        )
        + " | Frozen thresholds unchanged; "
        + (
            "amendment proposal awaits human review."
            if too_lenient
            else "no threshold amendment triggered."
        )
        + " |",
        "",
    ]
    (output / "PROPOSED_LEDGER_ROW.md").write_text(
        "\n".join(ledger_lines), encoding="utf-8"
    )
    if too_lenient:
        proposal = [
            "# Proposed separation-gate threshold amendment — human decision required",
            "",
            "This proposal is evidence only. It does not modify the frozen gate.",
            "",
            "The following feasible action classes exceeded at least one current "
            "floor (AUC 0.65, retention 0.15) at the minimum tested selector setting "
            "whose mean exact 44b6 adjusted-edge-Jaccard delta reached +0.005:",
            "",
        ]
        for action_class in too_lenient:
            req = requirements[action_class]["target_plus_0_005"]["mean_requirement"]
            if req is None:
                proposal.append(
                    f"- `{action_class}`: no tested AUC through 0.99 reached +0.005."
                )
            else:
                proposal.append(
                    f"- `{action_class}`: tested AUC {req['target_auc']:.2f}; "
                    f"mean CG0-budget retention {req['retention']['mean']:.6f}; "
                    f"support prefix {req['candidate_prefix_N']}."
                )
        proposal.extend(
            [
                "",
                "Proposed human decision: either retain the current permissive "
                "separation screen as an unlock-only diagnostic with an explicit "
                "warning that it is not graph-sufficient, or raise class-specific "
                "floors no lower than the measured grid requirements after reviewing "
                "the complete seed distributions. Do not infer a threshold for any "
                "class whose frozen population is ceiling-infeasible or class-degenerate.",
                "",
                "**HALT: HUMAN_REVIEW_REQUIRED.**",
                "",
            ]
        )
        (output / "AMENDMENT_PROPOSAL.md").write_text(
            "\n".join(proposal), encoding="utf-8"
        )

    report_text = markdown_report(
        population_summary, oracle_results, requirements, barrier
    )
    (output / "TERMINAL_REPORT.md").write_text(report_text, encoding="utf-8")
    result = {
        "schema": SCHEMA,
        "status": barrier_status,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_input_sha256": pins,
        "population": population_summary,
        "requirements": requirements,
        "barrier": barrier,
    }
    canonical_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output.resolve())
    except Exception as exc:
        print(f"selector requirement diagnostic failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
