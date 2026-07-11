#!/usr/bin/env python3
"""Tune global coordinate offsets for an existing alpha prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.official_eval import official_metric_row  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--z-offsets", default="-4,-3,-2,-1,0,1,2,3,4")
    parser.add_argument("--y-offsets", default="-28,-24,-20,-16,-12,-8,-4,0,4")
    parser.add_argument("--x-offsets", default="-16,-12,-8,-4,0,4,8,12,16")
    parser.add_argument("--output", type=Path, default=Path("reports/alpha_coordinate_offset_tuning.json"))
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_prediction_dataset(path: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] != dataset:
                continue
            if row["row_type"] == "node":
                nodes.append(row)
            elif row["row_type"] == "edge":
                edges.append(row)
    return nodes, edges


def shifted_prediction(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    offset: tuple[int, int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dz, dy, dx = offset
    pred_nodes: list[dict[str, Any]] = []
    id_map: dict[int, int] = {}
    for new_id, row in enumerate(nodes, start=1):
        old_id = int(row["node_id"])
        id_map[old_id] = new_id
        pred_nodes.append(
            {
                "dataset": row["dataset"],
                "node_id": new_id,
                "t": int(row["t"]),
                "z": min(max(int(row["z"]) + dz, 0), 63),
                "y": min(max(int(row["y"]) + dy, 0), 255),
                "x": min(max(int(row["x"]) + dx, 0), 255),
            }
        )
    pred_edges = [
        {"source_id": id_map[int(edge["source_id"])], "target_id": id_map[int(edge["target_id"])]}
        for edge in edges
        if int(edge["source_id"]) in id_map and int(edge["target_id"]) in id_map
    ]
    return pred_nodes, pred_edges


def main() -> int:
    args = parse_args()
    nodes, edges = read_prediction_dataset(args.prediction_csv, args.dataset)
    if not nodes:
        raise ValueError(f"No node rows for {args.dataset} in {args.prediction_csv}")
    store = ZipStore(args.zip, mode="r")
    try:
        attrs = dict(zarr.open_group(store=store, path=f"train/{args.dataset}.zarr", mode="r").attrs)
        gt = read_geff_arrays(store, "train", args.dataset)
    finally:
        store.close()
    scale = parse_scale(attrs)
    rows: list[dict[str, Any]] = []
    for dz in parse_ints(args.z_offsets):
        for dy in parse_ints(args.y_offsets):
            for dx in parse_ints(args.x_offsets):
                pred_nodes, pred_edges = shifted_prediction(nodes, edges, (dz, dy, dx))
                metric = official_metric_row(pred_nodes, pred_edges, gt, scale)
                rows.append(
                    {
                        "offset_zyx": [dz, dy, dx],
                        "score": metric["adj_edge_jaccard"],
                        "edge_jaccard": metric["edge_jaccard"],
                        "node_recall": metric["node_recall"],
                        "edge_tp": metric["edge_tp"],
                        "edge_fp": metric["edge_fp"],
                        "edge_fn": metric["edge_fn"],
                    }
                )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    payload = {
        "prediction_csv": str(args.prediction_csv),
        "dataset": args.dataset,
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"best": payload["best"], "top5": rows[:5]}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
