#!/usr/bin/env python3
"""Diagnose alpha candidate recall against sparse GEFF nodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celltrack_alpha.proposals import detect_candidates_one_timepoint  # noqa: E402
from celltrack_alpha.routing import default_route_configs  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--routes", default=",".join(sorted(default_route_configs())))
    parser.add_argument("--max-gt-timepoints", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/alpha_detection_diagnostic.json"))
    return parser.parse_args()


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def nearest_distances_um(
    candidates: list[Any],
    gt_xyz: np.ndarray,
    scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    if len(gt_xyz) == 0:
        return np.empty((0,), dtype=np.float32)
    if not candidates:
        return np.full((len(gt_xyz),), np.inf, dtype=np.float32)
    cand_xyz = np.asarray([node.coord_zyx for node in candidates], dtype=np.float32)
    scale = np.asarray(scale_zyx, dtype=np.float32)
    dist = np.sqrt(np.sum(((gt_xyz[:, None, :] - cand_xyz[None, :, :]) * scale) ** 2, axis=2))
    return np.min(dist, axis=1)


def main() -> int:
    args = parse_args()
    configs = default_route_configs()
    routes = parse_csv_strings(args.routes)
    unknown = [route for route in routes if route not in configs]
    if unknown:
        raise ValueError(f"Unknown routes: {unknown}")

    rows: list[dict[str, Any]] = []
    store = ZipStore(args.zip, mode="r")
    try:
        group = zarr.open_group(store=store, path=f"train/{args.dataset}.zarr", mode="r")
        attrs = dict(group.attrs)
        array = zarr.open_array(store=store, path=f"train/{args.dataset}.zarr/0", mode="r")
        gt = read_geff_arrays(store, "train", args.dataset)
        scale = parse_scale(attrs)
        gt_times = sorted(set(int(t) for t in gt["nodes"]["t"]))
        if args.max_gt_timepoints is not None:
            gt_times = gt_times[: int(args.max_gt_timepoints)]
        gt_by_t: dict[int, np.ndarray] = {}
        for t in gt_times:
            mask = gt["nodes"]["t"] == t
            gt_by_t[t] = np.asarray(
                list(zip(gt["nodes"]["z"][mask], gt["nodes"]["y"][mask], gt["nodes"]["x"][mask], strict=True)),
                dtype=np.float32,
            )
        for route in routes:
            config = configs[route]
            dists: list[float] = []
            node_counts: list[int] = []
            started = time.time()
            next_node_id = 1
            for t in gt_times:
                nodes, next_node_id = detect_candidates_one_timepoint(
                    volume=np.asarray(array[t]),
                    dataset=args.dataset,
                    t=t,
                    attrs=attrs,
                    scale_zyx=scale,
                    config=config,
                    next_node_id=next_node_id,
                )
                node_counts.append(len(nodes))
                dists.extend(float(v) for v in nearest_distances_um(nodes, gt_by_t[t], scale))
            finite = np.asarray([v for v in dists if np.isfinite(v)], dtype=np.float32)
            row = {
                "dataset": args.dataset,
                "route": route,
                "gt_timepoints": len(gt_times),
                "gt_nodes": len(dists),
                "mean_candidates_per_gt_t": float(np.mean(node_counts)) if node_counts else float("nan"),
                "recall_7um": float(np.mean(finite <= 7.0)) if len(finite) else 0.0,
                "recall_10um": float(np.mean(finite <= 10.0)) if len(finite) else 0.0,
                "median_nearest_um": float(np.median(finite)) if len(finite) else float("inf"),
                "p90_nearest_um": float(np.quantile(finite, 0.90)) if len(finite) else float("inf"),
                "runtime_sec": time.time() - started,
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    finally:
        store.close()

    payload = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
