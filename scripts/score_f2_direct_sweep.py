#!/usr/bin/env python3
"""Score a retrieved F2 direct D/E sweep against its same-run V32 control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.evaluation import submission_graph_diagnostics  # noqa: E402
from celltrack_alpha.official_eval import official_metric_row, official_summary  # noqa: E402
from celltrack_baseline.rule_based import list_zip_datasets, parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
METRICS = (
    "score",
    "edge_jaccard",
    "adj_edge_jaccard",
    "node_recall",
    "division_jaccard",
    "total_node_ratio",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            dataset = str(row["dataset"])
            if row["row_type"] == "node":
                nodes[dataset].append(row)
            elif row["row_type"] == "edge":
                edges[dataset].append(row)
            else:
                raise ValueError(f"unknown row type {row['row_type']!r}")
    return nodes, edges, rows


def finite_delta(candidate: Any, control: Any) -> float | None:
    try:
        left, right = float(candidate), float(control)
    except (TypeError, ValueError):
        return None
    return left - right if math.isfinite(left) and math.isfinite(right) else None


def score_candidate(
    *, candidate_root: Path, store: ZipStore, allowed_datasets: set[str]
) -> dict[str, Any]:
    config = json.loads((candidate_root / "CANDIDATE_CONFIG.json").read_text(encoding="utf-8"))
    graph_path = candidate_root / "V32_FINAL_GRAPH_ROWS.csv"
    nodes, edges, all_rows = read_predictions(graph_path)
    datasets = sorted(nodes)
    if set(datasets) - allowed_datasets:
        raise ValueError(f"candidate contains unknown train identities: {set(datasets) - allowed_datasets}")
    metric_rows = []
    for dataset in datasets:
        group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
        gt = read_geff_arrays(store, "train", dataset)
        row = official_metric_row(
            nodes.get(dataset, []), edges.get(dataset, []), gt, parse_scale(dict(group.attrs))
        )
        row["dataset"] = dataset
        metric_rows.append(row)
    return {
        "candidate": config,
        "final_graph_rows": {
            "path": str(graph_path),
            "sha256": sha256_file(graph_path),
            "bytes": graph_path.stat().st_size,
        },
        "graph_diagnostics": submission_graph_diagnostics(all_rows),
        "rows": metric_rows,
        "summary": official_summary(metric_rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    completion_path = args.sweep_root.parent / "F2_SWEEP_COMPLETION.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS" or completion.get("submission_absent") is not True:
        raise ValueError("sweep completion is not a no-submission PASS")
    expected_ids = [str(value) for value in completion["candidate_ids"]]
    observed_ids = sorted(path.name for path in args.sweep_root.iterdir() if path.is_dir())
    if sorted(expected_ids) != observed_ids:
        raise ValueError("sweep candidate directory set differs from completion receipt")
    allowed = set(list_zip_datasets(args.zip, "train"))
    scored = []
    store = ZipStore(args.zip, mode="r")
    try:
        for candidate_id in expected_ids:
            scored.append(
                score_candidate(
                    candidate_root=args.sweep_root / candidate_id,
                    store=store,
                    allowed_datasets=allowed,
                )
            )
    finally:
        store.close()

    control = next(
        row for row in scored if row["candidate"]["id"] == "CONTROL_V32"
    )
    decisions = []
    for row in scored:
        if row is control:
            continue
        delta = {
            metric: finite_delta(row["summary"].get(metric), control["summary"].get(metric))
            for metric in METRICS
        }
        graph = row["graph_diagnostics"]
        illegal = int(graph["out_gt2"]) > 0 or int(graph["merges"]) > 0
        adjusted = delta["adj_edge_jaccard"]
        node = delta["node_recall"]
        kill_reasons = []
        if adjusted is None or adjusted <= -0.002:
            kill_reasons.append("adjusted_delta_lte_minus_0.002")
        if node is None or node < -0.01:
            kill_reasons.append("node_recall_drop_gt_0.01")
        if illegal:
            kill_reasons.append("illegal_graph")
        raw_delta = delta["edge_jaccard"]
        adjustment_delta = (
            None
            if adjusted is None or raw_delta is None
            else adjusted - raw_delta
        )
        score_delta = delta["score"]
        f2_adjusted_edge_gate_passed = bool(
            adjusted is not None
            and adjusted >= 0.005
            and raw_delta is not None
            and raw_delta >= 0.0
            and node is not None
            and node >= 0.0
            and not illegal
        )
        competition_score_gate_passed = bool(
            score_delta is not None
            and score_delta >= 0.005
            and raw_delta is not None
            and raw_delta >= 0.0
            and node is not None
            and node >= 0.0
            and not illegal
        )
        decisions.append(
            {
                "candidate_id": row["candidate"]["id"],
                "delta": delta,
                "decision": "KILL" if kill_reasons else "SURVIVE_UNRANKED",
                "kill_reasons": kill_reasons,
                # Keep the frozen F2 family estimand distinct from the actual
                # competition objective.  A family-level adjusted-edge pass
                # is evidence, but cannot by itself be called a score pass if
                # the division term offsets it.
                "f2_adjusted_edge_gate_passed": f2_adjusted_edge_gate_passed,
                "competition_score_gate_passed": competition_score_gate_passed,
                "adjustment_component_delta": adjustment_delta,
                "multiplier_dominant": bool(
                    adjusted is not None
                    and adjusted > 0
                    and adjustment_delta is not None
                    and raw_delta is not None
                    and adjustment_delta > raw_delta
                ),
            }
        )
    payload = {
        "schema": "celltrack-alpha.f2-direct-sweep-score.v1",
        "status": "PASS",
        "sweep_completion": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
        },
        "gt_zip": {"path": str(args.zip), "bytes": args.zip.stat().st_size},
        "control": control,
        "candidates": scored,
        "decisions": decisions,
        "survivors_unranked": [
            row["candidate_id"]
            for row in decisions
            if row["decision"] == "SURVIVE_UNRANKED"
        ],
        "f2_adjusted_edge_gate_passers": [
            row["candidate_id"]
            for row in decisions
            if row["f2_adjusted_edge_gate_passed"]
        ],
        "competition_score_gate_passers": [
            row["candidate_id"]
            for row in decisions
            if row["competition_score_gate_passed"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "survivors_unranked": payload["survivors_unranked"],
        "f2_adjusted_edge_gate_passers": payload[
            "f2_adjusted_edge_gate_passers"
        ],
        "competition_score_gate_passers": payload[
            "competition_score_gate_passers"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
