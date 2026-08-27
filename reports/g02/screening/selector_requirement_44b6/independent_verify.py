#!/usr/bin/env python3
"""Separate-pass verification of selector curves, exact scoring, and governance."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path("/Users/agent/Desktop/cell_tracking")
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
)
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


class ReviewError(RuntimeError):
    pass


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReviewError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"JSON root is not an object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ReviewError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    temporary.replace(path)


def independent_greedy_swap(population: Any, order: np.ndarray, prefix: int) -> np.ndarray:
    owned_source: dict[int, set[int]] = defaultdict(set)
    owned_target: dict[int, set[int]] = defaultdict(set)
    owned_edge: dict[int, set[tuple[int, int]]] = defaultdict(set)
    selected = []
    for raw_index in order[:prefix]:
        index = int(raw_index)
        video = int(population.video[index])
        source = int(population.source[index])
        target = int(population.target[index])
        conflicts = set(population.conflicts[index])
        if (
            source in owned_source[video]
            or target in owned_target[video]
            or conflicts.intersection(owned_edge[video])
        ):
            continue
        owned_source[video].add(source)
        owned_target[video].add(target)
        owned_edge[video].update(conflicts)
        selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def direct_edge_score(module: Any, population: Any, selected: np.ndarray) -> dict[str, Any]:
    by_video: dict[int, list[int]] = defaultdict(list)
    for raw_index in selected:
        by_video[int(population.video[raw_index])].append(int(raw_index))
    rows = []
    with ZipStore(module.RAW_ARCHIVE, mode="r") as store:
        for video_index, identity in enumerate(population.identities):
            graph = population.graphs[video_index]
            indices = by_video.get(video_index, [])
            if population.action_class == "delete":
                if indices:
                    action = GraphAction(
                        action_id=f"review.delete:{video_index}",
                        kind=ActionKind.DELETE_EDGES,
                        delete_edges=tuple(
                            (
                                int(population.source[index]),
                                int(population.target[index]),
                            )
                            for index in indices
                        ),
                    )
                    candidate = compile_action(graph, action).graph
                else:
                    candidate = graph
            else:
                actions = [module._swap_action(population, index) for index in indices]
                candidate = compile_batch(graph, actions).graph if actions else graph
            dataset = identity.split(":", 1)[1]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, candidate)
            rows.append(official_metric_row(nodes, edges, gt, scale))
    return official_summary(rows)


def direct_restore_score(
    module: Any, population: Any, selected: np.ndarray
) -> dict[str, Any]:
    by_video: dict[int, list[Any]] = defaultdict(list)
    for raw_index in selected:
        index = int(raw_index)
        by_video[int(population.video[index])].append(population.actions[index])
    rows = []
    with ZipStore(module.RAW_ARCHIVE, mode="r") as store:
        for video_index, identity in enumerate(population.identities):
            graph = population.graphs[video_index]
            actions = by_video.get(video_index, [])
            candidate = compile_batch(graph, actions).graph if actions else graph
            dataset = identity.split(":", 1)[1]
            gt = read_geff_arrays(store, "train", dataset)
            group = zarr.open_group(
                store=store, path=f"train/{dataset}.zarr", mode="r"
            )
            scale = tuple(float(value) for value in parse_scale(dict(group.attrs)))
            nodes, edges = scoreall._prediction_rows(dataset, candidate)
            rows.append(official_metric_row(nodes, edges, gt, scale))
    return official_summary(rows)


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), abs_tol=1e-12):
        raise ReviewError(f"{label}: observed {observed}, expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CELLTRACK_GUARD_ACTIVE") != "1":
        raise SystemExit("independent verification requires the binding local guard")
    run_dir = args.run_dir.resolve()
    validation_path = run_dir / "VALIDATION.json"
    review_path = run_dir / "INDEPENDENT_REVIEW.json"
    if validation_path.exists() or review_path.exists():
        raise SystemExit("refusing to overwrite review outputs")

    module = load_module("selector_requirement_review_target", HERE / "selector_requirement.py")
    module._verify_pins()
    dose = load(run_dir / "dose_response.json")
    noise = load(run_dir / "noise_surface.json")
    requirements = load(run_dir / "requirement_surface.json")
    barrier = load(run_dir / "BARRIER.json")
    result = load(run_dir / "RESULT.json")
    swap_module = module.load_module("selector_requirement_review_swap", module.SWAP_IMPL)
    restore_module = module.load_module(
        "selector_requirement_review_restore", module.RESTORE_IMPL
    )
    delete, swap, _ = module.load_edge_populations(swap_module)
    restore = module.load_restore_population(restore_module)
    populations = {
        "delete": delete,
        "occupied_swap": swap,
        "restore_topology": restore,
    }

    checks = []
    direct_points = {}
    for action_class, population in populations.items():
        class_dose = dose["classes"][action_class]
        curve = class_dose["curve"]
        best = max(
            curve,
            key=lambda row: float(
                row["graph_delta"]["adjusted_edge_jaccard"]
            ),
        )
        order = module.oracle_order(population.delta_s)
        prefix = int(best["candidate_prefix_N"])
        if action_class == "delete":
            selected = np.asarray(order[:prefix], dtype=np.int64)
            candidate = direct_edge_score(module, population, selected)
            baseline = official_summary(list(population.baseline_rows))
        elif action_class == "occupied_swap":
            selected = independent_greedy_swap(population, order, prefix)
            candidate = direct_edge_score(module, population, selected)
            baseline = official_summary(list(population.baseline_rows))
        else:
            selected = np.asarray(order[:prefix], dtype=np.int64)
            candidate = direct_restore_score(module, population, selected)
            baseline = official_summary(list(population.baseline_rows))
        observed = {
            "adjusted_edge_jaccard": float(candidate["adj_edge_jaccard"])
            - float(baseline["adj_edge_jaccard"]),
            "raw_edge_jaccard": float(candidate["edge_jaccard"])
            - float(baseline["edge_jaccard"]),
            "node_recall": float(candidate["node_recall"])
            - float(baseline["node_recall"]),
        }
        for key, value in observed.items():
            assert_close(value, best["graph_delta"][key], f"{action_class} {key}")
        if len(selected) != int(best["applied_support"]):
            raise ReviewError(f"{action_class} applied support drifted")
        direct_points[action_class] = {
            "candidate_prefix_N": prefix,
            "applied_support": len(selected),
            "direct_official_delta": observed,
            "reported_delta": best["graph_delta"],
        }
        checks.append(f"{action_class}_oracle_best_direct_official_match")

    for action_class in ("delete", "occupied_swap", "restore_topology"):
        raw = noise["classes"][action_class]
        labels = populations[action_class].label
        if not np.any(labels == 1) or not np.any(labels == 0):
            if raw["status"] != "CLASS_DEGENERATE_NO_REQUIREMENT_SURFACE":
                raise ReviewError(f"{action_class} degenerate status drifted")
            checks.append(f"{action_class}_degenerate_handling")
            continue
        runs = raw["seed_runs"]
        if len(runs) != len(module.TARGET_AUCS) * module.SEED_COUNT:
            raise ReviewError(f"{action_class} seed coverage drifted")
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in runs:
            grouped[float(row["target_auc"])].append(row)
            target = float(row["target_auc"])
            lattice = 0.5 / (
                int(labels.sum()) * int((labels == 0).sum())
            )
            if abs(float(row["empirical_auc"]) - target) > max(
                module.AUC_TOLERANCE, lattice + 1e-15
            ):
                raise ReviewError(f"{action_class} empirical AUC tolerance drifted")
            for video_row in row["retention"]["videos"]:
                if (
                    int(video_row["selected_harmful"])
                    > int(video_row["budget"])
                ):
                    raise ReviewError(f"{action_class} CG0 budget violation")
        for target, rows in grouped.items():
            if len(rows) != module.SEED_COUNT:
                raise ReviewError(f"{action_class} AUC {target} seed count drifted")
            aggregate = next(
                value
                for value in noise["aggregates"][action_class]["auc_results"]
                if float(value["target_auc"]) == target
            )
            mean = float(np.mean([row["empirical_auc"] for row in rows]))
            assert_close(
                mean,
                aggregate["empirical_auc"]["mean"],
                f"{action_class} AUC {target} aggregate mean",
            )
        checks.append(f"{action_class}_noise_coverage_auc_budget")

    if any(
        barrier.get(key)
        for key in (
            "thresholds_changed",
            "ledger_written",
            "plan_changed",
            "gpu_action",
            "kaggle_action",
            "runpod_action",
            "submission_action",
            "family_6bba_used",
            "cv57_used",
            "external_action",
        )
    ):
        raise ReviewError("a hard-stop flag is true")
    if barrier.get("tier_moves") != 0 or barrier.get("dead_actions") != 0:
        raise ReviewError("tier or DEAD mutation was reported")
    expected_too_lenient = sorted(
        action_class
        for action_class, value in requirements["requirements"].items()
        if value["frozen_gate_verdict"]["verdict"] == "TOO_LENIENT"
    )
    if sorted(barrier["amendment_proposed_for"]) != expected_too_lenient:
        raise ReviewError("amendment barrier is inconsistent with verdicts")
    expected_status = (
        "HUMAN_REVIEW_REQUIRED"
        if expected_too_lenient
        else "DIAGNOSTIC_COMPLETE"
    )
    if barrier["status"] != expected_status or result["status"] != expected_status:
        raise ReviewError("terminal barrier status drifted")
    checks.append("governance_hard_stops_and_barrier")

    review = {
        "schema": "celltrack-alpha.selector-requirement-independent-review.v1",
        "status": "PASS",
        "source": "review",
        "method": "Fresh process, direct frozen official scoring of every class oracle-best point, independent occupied-swap greedy ownership replay, complete seed/AUC/budget audit, and governance barrier review.",
        "direct_points": direct_points,
        "findings": [],
        "hard_stops_preserved": True,
    }
    validation = {
        "schema": "celltrack-alpha.selector-requirement-validation.v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks": checks,
        "class_count": 3,
        "target_auc_count": len(module.TARGET_AUCS),
        "seed_count_per_auc": module.SEED_COUNT,
        "direct_official_points": direct_points,
        "human_review_required": expected_status == "HUMAN_REVIEW_REQUIRED",
        "tier_moves": 0,
        "threshold_changes": 0,
        "external_actions": 0,
    }
    write(review_path, review)
    write(validation_path, validation)
    print(json.dumps({"status": "PASS", "checks": len(checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
