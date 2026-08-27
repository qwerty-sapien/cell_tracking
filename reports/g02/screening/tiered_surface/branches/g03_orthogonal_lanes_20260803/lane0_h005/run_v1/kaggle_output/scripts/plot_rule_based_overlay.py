#!/usr/bin/env python3
"""Create a quick GT/prediction overlay from the local Kaggle ZIP."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip"))
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--z", type=int, default=None, help="Central z slice. Defaults to first GT z at t.")
    parser.add_argument("--z-window", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/plots/rule_based_overlay.png"))
    return parser.parse_args()


def normalize_u8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def read_prediction_nodes(path: Path, dataset: str, t: int) -> list[tuple[int, int, int]]:
    nodes = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["row_type"] == "node" and row["dataset"] == dataset and int(row["t"]) == t:
                nodes.append((int(row["z"]), int(row["y"]), int(row["x"])))
    return nodes


def main() -> int:
    args = parse_args()
    store = ZipStore(args.zip, mode="r")
    try:
        volume = zarr.open_array(store=store, path=f"train/{args.dataset}.zarr/0", mode="r")[args.t]
        base = f"train/{args.dataset}.geff"
        gt_t = zarr.open_array(store=store, path=f"{base}/nodes/props/t/values", mode="r")[:]
        gt_z = zarr.open_array(store=store, path=f"{base}/nodes/props/z/values", mode="r")[:]
        gt_y = zarr.open_array(store=store, path=f"{base}/nodes/props/y/values", mode="r")[:]
        gt_x = zarr.open_array(store=store, path=f"{base}/nodes/props/x/values", mode="r")[:]
    finally:
        store.close()

    gt_points = [
        (int(z), int(y), int(x))
        for z, y, x, t in zip(gt_z, gt_y, gt_x, gt_t)
        if int(t) == args.t
    ]
    pred_points = read_prediction_nodes(args.prediction_csv, args.dataset, args.t)
    z_center = args.z if args.z is not None else (gt_points[0][0] if gt_points else volume.shape[0] // 2)
    z0 = max(0, z_center - args.z_window)
    z1 = min(volume.shape[0], z_center + args.z_window + 1)
    mip = normalize_u8(np.max(volume[z0:z1], axis=0))
    rgb = Image.fromarray(np.stack([mip, mip, mip], axis=-1), mode="RGB")
    draw = ImageDraw.Draw(rgb)

    for z, y, x in pred_points:
        if z0 <= z < z1:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 255), width=2)
    for z, y, x in gt_points:
        if z0 <= z < z1:
            draw.line((x - 5, y, x + 5, y), fill=(255, 0, 0), width=2)
            draw.line((x, y - 5, x, y + 5), fill=(255, 0, 0), width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(args.output)
    print(f"Wrote {args.output} with {len(gt_points)} GT points and {len(pred_points)} predicted points at t={args.t}, z-window={z0}:{z1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
