"""Deterministic scorer-parity fixtures for the Kaggle official metric wrapper."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .official_eval import OFFICIAL_SRC, official_edge_metric_row, official_metric_row, official_summary

ROOT = Path(__file__).resolve().parents[2]
SCALE_ZYX = (1.625, 0.40625, 0.40625)

LINE_NODES = [
    (1, 0, 0, 0, 0),
    (2, 1, 0, 1, 0),
    (3, 2, 0, 2, 0),
]
LINE_EDGES = [(1, 2), (2, 3)]

DIVISION_NODES = [
    (1, 0, 0, 0, 0),
    (2, 1, 0, 1, 0),
    (3, 2, 0, 0, 0),
    (4, 2, 0, 2, 0),
    (5, 3, 0, 0, 1),
    (6, 3, 0, 2, 1),
]
DIVISION_EDGES = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 6)]

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "gt_self_with_adjusted_node_count_penalty",
        "gt_nodes": LINE_NODES,
        "gt_edges": LINE_EDGES,
        "estimated_nodes": 10.0,
        "pred_nodes": LINE_NODES,
        "pred_edges": LINE_EDGES,
        "expected": {
            "edge_tp": 2.0,
            "edge_fp": 0.0,
            "edge_fn": 0.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
            "num_pred_nodes": 3.0,
            "node_recall": 1.0,
            "total_node_ratio": -0.7,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 1.07,
            "division_jaccard": float("nan"),
            "estimated_number_of_nodes": 10.0,
        },
    },
    {
        "name": "nonconsecutive_edge_is_dropped_but_fork_counts_division_fp",
        "gt_nodes": LINE_NODES,
        "gt_edges": LINE_EDGES,
        "estimated_nodes": 10.0,
        "pred_nodes": LINE_NODES,
        "pred_edges": [(1, 2), (1, 3)],
        "expected": {
            "edge_tp": 1.0,
            "edge_fp": 0.0,
            "edge_fn": 1.0,
            "division_tp": 0.0,
            "division_fp": 1.0,
            "division_fn": 0.0,
            "num_pred_nodes": 3.0,
            "node_recall": 1.0,
            "total_node_ratio": -0.7,
            "edge_jaccard": 0.5,
            "adj_edge_jaccard": 0.535,
            "division_jaccard": 0.0,
            "estimated_number_of_nodes": 10.0,
        },
    },
    {
        "name": "distance_miss_keeps_valid_edge_fp_and_edge_fn",
        "gt_nodes": LINE_NODES,
        "gt_edges": LINE_EDGES,
        "estimated_nodes": 10.0,
        "pred_nodes": [(1, 0, 0, 0, 0), (2, 1, 0, 30, 0)],
        "pred_edges": [(1, 2)],
        "expected": {
            "edge_tp": 0.0,
            "edge_fp": 1.0,
            "edge_fn": 2.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
            "num_pred_nodes": 2.0,
            "node_recall": 1.0 / 3.0,
            "total_node_ratio": -0.8,
            "edge_jaccard": 0.0,
            "adj_edge_jaccard": 0.0,
            "division_jaccard": float("nan"),
            "estimated_number_of_nodes": 10.0,
        },
    },
    {
        "name": "duplicate_pred_edges_are_deduplicated",
        "gt_nodes": LINE_NODES,
        "gt_edges": LINE_EDGES,
        "estimated_nodes": 10.0,
        "pred_nodes": LINE_NODES,
        "pred_edges": [(1, 2), (1, 2), (2, 3)],
        "expected": {
            "edge_tp": 2.0,
            "edge_fp": 0.0,
            "edge_fn": 0.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
            "num_pred_nodes": 3.0,
            "node_recall": 1.0,
            "total_node_ratio": -0.7,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 1.07,
            "division_jaccard": float("nan"),
            "estimated_number_of_nodes": 10.0,
        },
    },
    {
        "name": "division_self_scores_division_jaccard_one",
        "gt_nodes": DIVISION_NODES,
        "gt_edges": DIVISION_EDGES,
        "estimated_nodes": 6.0,
        "pred_nodes": DIVISION_NODES,
        "pred_edges": DIVISION_EDGES,
        "expected": {
            "edge_tp": 5.0,
            "edge_fp": 0.0,
            "edge_fn": 0.0,
            "division_tp": 1.0,
            "division_fp": 0.0,
            "division_fn": 0.0,
            "num_pred_nodes": 6.0,
            "node_recall": 1.0,
            "total_node_ratio": 0.0,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 1.0,
            "division_jaccard": 1.0,
            "estimated_number_of_nodes": 6.0,
        },
    },
    {
        "name": "division_missing_fork_counts_fn",
        "gt_nodes": DIVISION_NODES,
        "gt_edges": DIVISION_EDGES,
        "estimated_nodes": 6.0,
        "pred_nodes": DIVISION_NODES,
        "pred_edges": [(1, 2), (2, 3), (3, 5), (4, 6)],
        "expected": {
            "edge_tp": 4.0,
            "edge_fp": 0.0,
            "edge_fn": 1.0,
            "division_tp": 0.0,
            "division_fp": 0.0,
            "division_fn": 1.0,
            "num_pred_nodes": 6.0,
            "node_recall": 1.0,
            "total_node_ratio": 0.0,
            "edge_jaccard": 0.8,
            "adj_edge_jaccard": 0.8,
            "division_jaccard": 0.0,
            "estimated_number_of_nodes": 6.0,
        },
    },
    {
        "name": "shared_child_makes_both_forks_malformed",
        "gt_nodes": DIVISION_NODES,
        "gt_edges": DIVISION_EDGES,
        "estimated_nodes": 6.0,
        "pred_nodes": DIVISION_NODES,
        "pred_edges": [*DIVISION_EDGES, (1, 3)],
        "expected": {
            "edge_tp": 5.0,
            "edge_fp": 0.0,
            "edge_fn": 0.0,
            "division_tp": 0.0,
            "division_fp": 2.0,
            "division_fn": 1.0,
            "num_pred_nodes": 6.0,
            "node_recall": 1.0,
            "total_node_ratio": 0.0,
            "edge_jaccard": 1.0,
            "adj_edge_jaccard": 1.0,
            "division_jaccard": 0.0,
            "estimated_number_of_nodes": 6.0,
        },
    },
)

SUMMARY_EXPECTED = {
    "n": 2,
    "edge_jaccard": 1.0,
    "division_jaccard": 1.0,
    "division_tp": 1.0,
    "division_fp": 0.0,
    "division_fn": 0.0,
    "node_recall": 1.0,
    "adj_edge_jaccard": 1.02,
    "n_adj": 2,
    "score": 1.12,
}

EDGE_ONLY_EXPECTED_KEYS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "num_pred_nodes",
    "node_recall",
    "total_node_ratio",
    "edge_jaccard",
    "adj_edge_jaccard",
    "estimated_number_of_nodes",
)


def _gt(nodes: list[tuple[int, int, int, int, int]], edges: list[tuple[int, int]], estimated: float) -> dict[str, Any]:
    keys = ("node_id", "t", "z", "y", "x")
    return {
        "attrs": {},
        "nodes": {
            key: np.asarray([row[index] for row in nodes], dtype=np.int64)
            for index, key in enumerate(keys)
        },
        "edges": np.asarray(edges, dtype=np.int64),
        "estimated_number_of_nodes": float(estimated),
    }


def _pred_nodes(nodes: list[tuple[int, int, int, int, int]]) -> list[dict[str, Any]]:
    return [
        {"dataset": "parity", "node_id": node_id, "t": t, "z": z, "y": y, "x": x}
        for node_id, t, z, y, x in nodes
    ]


def _pred_edges(edges: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"source_id": source, "target_id": target} for source, target in edges]


def _float_equal(actual: Any, expected: Any, tol: float = 1e-9) -> bool:
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except (TypeError, ValueError):
        return actual == expected
    if math.isnan(actual_f) and math.isnan(expected_f):
        return True
    return math.isclose(actual_f, expected_f, rel_tol=tol, abs_tol=tol)


def _compare(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    ok = True
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        passed = _float_equal(actual_value, expected_value)
        checks[key] = {
            "actual": actual_value,
            "expected": expected_value,
            "pass": passed,
        }
        ok = ok and passed
    return {"pass": ok, "checks": checks}


def _git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_scorer_parity() -> dict[str, Any]:
    """Run deterministic parity cases against the official wrapper."""

    rows: list[dict[str, Any]] = []
    edge_only_rows: list[dict[str, Any]] = []
    for case in CASES:
        pred_nodes = _pred_nodes(case["pred_nodes"])
        pred_edges = _pred_edges(case["pred_edges"])
        gt = _gt(case["gt_nodes"], case["gt_edges"], case["estimated_nodes"])
        row = official_metric_row(pred_nodes, pred_edges, gt, SCALE_ZYX)
        comparison = _compare(row, case["expected"])
        rows.append(
            {
                "name": case["name"],
                "pass": comparison["pass"],
                "actual": row,
                "expected": case["expected"],
                "checks": comparison["checks"],
            }
        )

        edge_only = official_edge_metric_row(pred_nodes, pred_edges, gt, SCALE_ZYX)
        edge_only_expected = {key: case["expected"][key] for key in EDGE_ONLY_EXPECTED_KEYS}
        edge_only_expected.update(
            {
                "division_tp": 0.0,
                "division_fp": 0.0,
                "division_fn": 0.0,
                "division_jaccard": float("nan"),
            }
        )
        edge_only_comparison = _compare(edge_only, edge_only_expected)
        edge_only_rows.append(
            {
                "name": case["name"],
                "pass": edge_only_comparison["pass"],
                "actual": edge_only,
                "expected": edge_only_expected,
                "checks": edge_only_comparison["checks"],
            }
        )

    summary_rows = [rows[0]["actual"], rows[4]["actual"]]
    summary = official_summary(summary_rows)
    summary_comparison = _compare(summary, SUMMARY_EXPECTED)
    official_metrics = ROOT / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/metrics.py"
    official_divisions = ROOT / "external/kaggle-cell-tracking-competition/src/tracking_cellmot/division_metrics.py"
    full_pass = all(row["pass"] for row in rows)
    edge_only_pass = all(row["pass"] for row in edge_only_rows)
    return {
        "schema_version": 2,
        "metric_source": "official_wrapper",
        "official_src": str(OFFICIAL_SRC),
        "official_checkout": _git_rev(ROOT / "external/kaggle-cell-tracking-competition"),
        "official_files": {
            "metrics.py": _file_sha256(official_metrics),
            "division_metrics.py": _file_sha256(official_divisions),
        },
        "scale_zyx": SCALE_ZYX,
        "cases": rows,
        "edge_only_cases": edge_only_rows,
        "edge_only_overall_pass": edge_only_pass,
        "summary_case": {
            "name": "summary_weighted_adjusted_edge_plus_division",
            "pass": summary_comparison["pass"],
            "actual": summary,
            "expected": SUMMARY_EXPECTED,
            "checks": summary_comparison["checks"],
        },
        "overall_pass": full_pass and edge_only_pass and summary_comparison["pass"],
    }


def write_scorer_parity_report(path: Path) -> dict[str, Any]:
    report = run_scorer_parity()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report
