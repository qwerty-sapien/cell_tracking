#!/usr/bin/env python3
"""Run Ultrack CV clips directly from the Kaggle ZIP without extracting raw zarrs."""

from __future__ import annotations

import argparse
import dataclasses
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

from celltrack_baseline.rule_based import (  # noqa: E402
    approximate_sparse_metrics,
    list_zip_datasets,
    parse_scale,
    read_geff_arrays,
    submission_rows,
    validate_submission_rows,
    write_submission,
)
from celltrack_ultrack import (  # noqa: E402
    UltrackConfig,
    build_ultrack_inputs,
    run_ultrack_clip,
    ultrack_tracks_to_submission,
)


DEFAULT_ZIP = Path("data/kaggle/raw/biohub-cell-tracking-during-development.zip")
DEFAULT_GUROBI_REPORT = Path("reports/gurobi_validation.json")


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets. Default: first 44b6 and first 6bba train sample.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--solver", choices=("auto", "GUROBI", "CBC"), default="auto")
    parser.add_argument("--gurobi-report", type=Path, default=DEFAULT_GUROBI_REPORT)
    parser.add_argument("--require-gurobi", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path("data/derived/ultrack"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/ultrack_cv_submission.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/ultrack_cv_metrics.json"))
    parser.add_argument("--cache-inputs", action="store_true", help="Save foreground/contour clips as compressed derived artifacts.")
    parser.add_argument("--plot-overlays", action="store_true", help="Save optional max-projection overlay plots when matplotlib is installed.")
    parser.add_argument("--io-only", action="store_true", help="Build/cache inputs but skip Ultrack execution.")
    parser.add_argument("--foreground-quantiles", default="0.9975", help="Comma-separated grid, e.g. 0.995,0.9975,0.999")
    parser.add_argument("--contour-presets", default="medium", help="Comma-separated grid: low,medium,high")
    parser.add_argument("--max-link-um", default="8.0", help="Comma-separated grid of physical link radii.")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--max-segments-per-time", type=int, default=100_000)
    parser.add_argument("--solution-gap", type=float, default=0.01)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--n-workers", type=int, default=1)
    return parser.parse_args()


def default_datasets(zip_path: Path, split: str) -> list[str]:
    datasets = list_zip_datasets(zip_path, split)
    selected: list[str] = []
    for prefix in ("44b6", "6bba"):
        match = next((name for name in datasets if name.startswith(prefix)), None)
        if match is not None:
            selected.append(match)
    return selected or datasets[:2]


def gurobi_validation_passed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(
        report.get("status") == "pass"
        and report.get("gurobi_milp", {}).get("success")
        and report.get("ultrack_check_gurobi", {}).get("success")
    )


def resolve_solver(args: argparse.Namespace) -> str:
    if args.solver != "auto":
        if args.solver == "GUROBI" and not gurobi_validation_passed(args.gurobi_report):
            raise RuntimeError(
                f"--solver GUROBI requires a passing validation report at {args.gurobi_report}. "
                "Run scripts/validate_gurobi.py first."
            )
        return args.solver
    if gurobi_validation_passed(args.gurobi_report):
        return "GUROBI"
    if args.require_gurobi:
        raise RuntimeError(
            f"Gurobi validation did not pass at {args.gurobi_report}; refusing CBC fallback because --require-gurobi was set."
        )
    return "CBC"


def crop_geff_arrays(gt: dict[str, Any], start: int, end: int, total_timepoints: int) -> dict[str, Any]:
    nodes = gt["nodes"]
    mask = (nodes["t"] >= start) & (nodes["t"] < end)
    kept_ids = set(int(v) for v in nodes["node_id"][mask])
    cropped_edges = np.asarray(
        [edge for edge in gt["edges"] if int(edge[0]) in kept_ids and int(edge[1]) in kept_ids],
        dtype=np.int64,
    )
    if cropped_edges.size == 0:
        cropped_edges = np.empty((0, 2), dtype=np.int64)
    estimated = gt.get("estimated_number_of_nodes")
    try:
        estimated = float(estimated) * max(0, end - start) / max(1, int(total_timepoints))
    except (TypeError, ValueError):
        estimated = float("nan")
    return {
        "attrs": gt.get("attrs", {}),
        "nodes": {key: np.asarray(value)[mask] for key, value in nodes.items()},
        "edges": cropped_edges,
        "estimated_number_of_nodes": estimated,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "edge_jaccard",
        "adj_edge_jaccard",
        "node_recall",
        "num_pred_nodes",
        "estimated_number_of_nodes",
        "total_node_ratio",
    ]
    summary = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row and float(row[key]) == float(row[key])]
        summary[key] = float(np.mean(vals)) if vals else float("nan")
    return summary


def config_label(foreground_quantile: float, contour_preset: str, max_link_um: float) -> str:
    fq = str(foreground_quantile).replace(".", "p")
    link = str(max_link_um).replace(".", "p")
    return f"fq{fq}_contour-{contour_preset}_link{link}"


def write_overlay(
    raw_clip: np.ndarray,
    foreground: np.ndarray,
    contours: np.ndarray,
    nodes: list[dict[str, Any]],
    gt: dict[str, Any] | None,
    dataset: str,
    t_abs: int,
    t_rel: int,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping overlay plots")
        return

    raw_proj = np.max(raw_clip[t_rel].astype(np.float32, copy=False), axis=0)
    lo, hi = np.quantile(raw_proj, [0.01, 0.999]).astype(np.float32)
    if hi <= lo:
        hi = lo + np.float32(1.0)
    raw_proj = np.clip((raw_proj - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
    fg_proj = np.max(foreground[t_rel], axis=0)
    contour_proj = np.max(contours[t_rel], axis=0)

    pred_xy = np.asarray([[n["x"], n["y"]] for n in nodes if int(n["t"]) == int(t_abs)], dtype=np.float32)
    gt_xy = np.empty((0, 2), dtype=np.float32)
    if gt is not None:
        gt_nodes = gt["nodes"]
        gt_mask = gt_nodes["t"] == t_abs
        gt_xy = np.asarray(
            [[x, y] for x, y in zip(gt_nodes["x"][gt_mask], gt_nodes["y"][gt_mask], strict=True)],
            dtype=np.float32,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(raw_proj, cmap="gray")
    ax.imshow(np.ma.masked_less(fg_proj, 0.5), cmap="Greens", alpha=0.25)
    ax.imshow(np.ma.masked_less(contour_proj, 0.4), cmap="magma", alpha=0.25)
    if len(pred_xy):
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=12, c="cyan", linewidths=0, label="pred")
    if len(gt_xy):
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=18, facecolors="none", edgecolors="yellow", linewidths=0.8, label="gt")
    ax.set_title(f"{dataset} t={t_abs}")
    ax.set_axis_off()
    if len(pred_xy) or len(gt_xy):
        ax.legend(loc="upper right")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    solver = resolve_solver(args)
    clip_length = int(args.clip_length) if args.clip_length is not None else (16 if solver == "GUROBI" else 12)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.cache_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    else:
        datasets = default_datasets(args.zip, args.split)
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]

    foreground_grid = parse_csv_floats(args.foreground_quantiles)
    contour_grid = parse_csv_strings(args.contour_presets)
    link_grid = parse_csv_floats(args.max_link_um)
    grid = [(fq, cp, link_um) for fq in foreground_grid for cp in contour_grid for link_um in link_grid]

    print(f"Ultrack ZIP CV run_id={run_id} solver={solver} clip_length={clip_length} datasets={datasets}")
    print(f"cache_root={run_dir}")
    (run_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "resolved_solver": solver, "clip_length": clip_length}, indent=2, default=str),
        encoding="utf-8",
    )

    all_config_payloads = []
    first_submission_written = False
    store = ZipStore(args.zip, mode="r")
    try:
        for foreground_quantile, contour_preset, max_link_um in grid:
            label = config_label(foreground_quantile, contour_preset, max_link_um)
            config_dir = run_dir / label
            config_dir.mkdir(parents=True, exist_ok=True)
            predictions = []
            metric_rows = []
            failures = []

            for dataset in datasets:
                print(f"{label} {dataset}: reading clip from ZIP", flush=True)
                group = zarr.open_group(store=store, path=f"{args.split}/{dataset}.zarr", mode="r")
                attrs = dict(group.attrs)
                array = zarr.open_array(store=store, path=f"{args.split}/{dataset}.zarr/0", mode="r")
                total_timepoints = int(array.shape[0])
                start = min(max(0, int(args.clip_start)), max(0, total_timepoints - 1))
                end = min(total_timepoints, start + clip_length)
                raw_clip = np.asarray(array[start:end])
                scale_zyx = parse_scale(attrs)
                dataset_dir = config_dir / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)

                cfg = UltrackConfig(
                    scale_zyx=scale_zyx,
                    foreground_quantile=foreground_quantile,
                    contour_preset=contour_preset,
                    min_area=args.min_area,
                    max_area=args.max_area,
                    max_link_um=max_link_um,
                    solver_name=solver,
                    window_size=args.window_size if args.window_size is not None else clip_length,
                    overlap_size=args.overlap,
                    solution_gap=args.solution_gap,
                    time_limit=args.time_limit,
                    n_workers=args.n_workers,
                    working_dir=dataset_dir,
                    database_file_name="ultrack.db",
                    max_segments_per_time=args.max_segments_per_time,
                )
                print(f"{label} {dataset}: building foreground/contours for t={start}:{end}", flush=True)
                foreground, contours = build_ultrack_inputs(raw_clip, attrs, cfg)
                if args.cache_inputs:
                    np.savez_compressed(dataset_dir / "ultrack_inputs.npz", foreground=foreground, contours=contours)

                nodes: list[dict[str, Any]] = []
                edges: list[dict[str, int]] = []
                if not args.io_only:
                    try:
                        print(f"{label} {dataset}: running Ultrack", flush=True)
                        tracks, graph = run_ultrack_clip(foreground, contours, cfg)
                        nodes, edges = ultrack_tracks_to_submission({"name": dataset, "t_offset": start}, tracks, graph)
                    except Exception as exc:
                        failures.append({"dataset": dataset, "failure_kind": type(exc).__name__})
                        print(f"{label} {dataset}: Ultrack failed with {type(exc).__name__}", flush=True)
                        continue

                predictions.append((dataset, nodes, edges))
                print(f"{label} {dataset}: nodes={len(nodes)} edges={len(edges)}", flush=True)

                gt_clip = None
                if args.split == "train":
                    gt = read_geff_arrays(store, args.split, dataset)
                    gt_clip = crop_geff_arrays(gt, start, end, total_timepoints)
                    if not args.io_only:
                        metrics = approximate_sparse_metrics(nodes, edges, gt_clip, scale_zyx)
                        metrics.update(
                            {
                                "dataset": dataset,
                                "clip_start": start,
                                "clip_end": end,
                                "foreground_quantile": foreground_quantile,
                                "contour_preset": contour_preset,
                                "max_link_um": max_link_um,
                                "solver": solver,
                            }
                        )
                        metric_rows.append(metrics)
                        print(
                            f"{label} {dataset}: adj={metrics['adj_edge_jaccard']:.4f} "
                            f"edge={metrics['edge_jaccard']:.4f} recall={metrics['node_recall']:.4f}",
                            flush=True,
                        )
                if args.plot_overlays:
                    write_overlay(
                        raw_clip=raw_clip,
                        foreground=foreground,
                        contours=contours,
                        nodes=nodes,
                        gt=gt_clip,
                        dataset=dataset,
                        t_abs=start,
                        t_rel=0,
                        output_path=dataset_dir / f"overlay_t{start:03d}.png",
                    )

            rows = submission_rows(predictions)
            validation = validate_submission_rows(rows)
            if not validation["valid"]:
                failures.append({"dataset": "*", "failure_kind": "submission_validation_failed", "errors": validation["errors"]})
            config_submission_path = config_dir / "submission.csv"
            write_submission(rows, config_submission_path)
            if not first_submission_written:
                write_submission(rows, args.output)
                first_submission_written = True

            payload = {
                "label": label,
                "config": dataclasses.asdict(
                    UltrackConfig(
                        foreground_quantile=foreground_quantile,
                        contour_preset=contour_preset,
                        max_link_um=max_link_um,
                        solver_name=solver,
                        window_size=args.window_size if args.window_size is not None else clip_length,
                        overlap_size=args.overlap,
                        solution_gap=args.solution_gap,
                        time_limit=args.time_limit,
                        n_workers=args.n_workers,
                        min_area=args.min_area,
                        max_area=args.max_area,
                        max_segments_per_time=args.max_segments_per_time,
                    )
                ),
                "submission": str(config_submission_path),
                "validation": validation,
                "rows": metric_rows,
                "mean": summarize_metrics(metric_rows) if metric_rows else {},
                "failures": failures,
            }
            (config_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            all_config_payloads.append(payload)
            print(json.dumps({"label": label, "validation": validation, "mean": payload["mean"], "failures": failures}, indent=2))
    finally:
        store.close()

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps({"run_id": run_id, "cache_dir": str(run_dir), "configs": all_config_payloads}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
