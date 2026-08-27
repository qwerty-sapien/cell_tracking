#!/usr/bin/env python3
"""Evaluate affine and coarse 3-D TV-L1 motion on the frozen 336-event cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.registration import optical_flow_tvl1
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import scoreall  # noqa: E402
from celltrack_baseline.rule_based import parse_scale  # noqa: E402


SCHEMA = "celltrack-alpha.phase1-e6-affine-tvl1-prefix52.v1"
CAUSE = "final_pair_geometry_above_gate_radius"
GATE_UM = 10.0
EXPECTED = {
    "video_count": 52,
    "edge_tp": 29676,
    "edge_fp": 2660,
    "edge_fn": 1810,
    "cohort_count": 336,
}
DEFAULT_ARCHIVE = ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
DEFAULT_FRAGMENTATION = (
    ROOT
    / "reports/g02/source_softmax_edge_f0_fit117_holdout_complete57_v2/fragmentation.json"
)
DEFAULT_SCORES = (
    ROOT / "reports/g02/source_softmax_edge_f0_fit117_holdout_prefix52_v1/scores.json"
)
DEFAULT_OUTPUT = ROOT / "reports/g02/phase1_e6_affine_tvl1_prefix52_v1.json"
DEFAULT_DOWNSAMPLE = (1, 4, 4)


class E6Error(RuntimeError):
    """Raised when the frozen E6 contract cannot be reproduced."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise E6Error(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "p25": None, "median": None,
                "p75": None, "p90": None, "max": None, "mean": None}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise E6Error("affine correspondences must be paired N x 3 arrays")
    if source.shape[0] < 4:
        raise E6Error("12-DOF affine requires at least four correspondences")
    center = np.mean(source, axis=0)
    design = np.column_stack((source - center, np.ones(source.shape[0])))
    coefficient, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) != 4:
        raise E6Error(f"rank-deficient affine design: {rank}")
    condition = float(singular[0] / singular[-1])
    if not math.isfinite(condition):
        raise E6Error("nonfinite affine condition number")
    return center, coefficient, int(rank), condition


def _predict_affine(point: np.ndarray, center: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.append(point - center, 1.0) @ coefficient


def _prepare_volume(volume: np.ndarray, downsample: tuple[int, int, int]) -> np.ndarray:
    value = np.asarray(volume, dtype=np.float32)
    sample = value[::2, ::4, ::4]
    low, high = np.quantile(sample, (0.01, 0.999))
    if not math.isfinite(float(high)) or high <= low:
        high = low + 1.0
    value = np.clip((value - float(low)) / float(high - low), 0.0, 1.0)
    sigma = tuple(0.4 if factor == 1 else 1.0 for factor in downsample)
    value = gaussian_filter(value, sigma=sigma, mode="nearest")
    return np.asarray(
        value[tuple(slice(None, None, factor) for factor in downsample)],
        dtype=np.float32,
    )


def _flow_at(flow: np.ndarray, point_zyx: np.ndarray, downsample: np.ndarray) -> np.ndarray:
    query = (point_zyx / downsample).reshape((3, 1))
    return np.asarray(
        [map_coordinates(axis, query, order=1, mode="nearest")[0] for axis in flow],
        dtype=np.float64,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.output.exists() and not args.force:
        raise E6Error(f"refusing to overwrite {args.output}; pass --force to reproduce")

    fragmentation = _load(args.fragmentation)
    scores = _load(args.scores)
    samples = scores.get("samples")
    if not isinstance(samples, list):
        raise E6Error("scores samples are missing")
    observed = {
        "video_count": len(samples),
        "edge_tp": int(sum(float(row["edge_tp"]) for row in samples)),
        "edge_fp": int(sum(float(row["edge_fp"]) for row in samples)),
        "edge_fn": int(sum(float(row["edge_fn"]) for row in samples)),
    }
    events = [
        dict(row)
        for row in fragmentation.get("breaks", [])
        if row.get("prefix_52") is True and row.get("cause") == CAUSE
    ]
    observed["cohort_count"] = len(events)
    if observed != EXPECTED:
        raise E6Error(f"frozen prefix-52 contract changed: {observed}")

    sample_by_identity = {str(row["identity"]): row for row in samples}
    event_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if float(event["distance_um"]) <= GATE_UM:
            raise E6Error("cohort contains an in-gate event")
        event_by_pair[(str(event["identity"]), int(event["source_frame"]))].append(event)

    affine_rows: list[dict[str, Any]] = []
    graph_by_identity: dict[str, Any] = {}
    scale_by_identity: dict[str, np.ndarray] = {}
    fit_by_pair: dict[tuple[str, int], list[tuple[int, int, np.ndarray, np.ndarray]]] = {}

    for identity in sorted({str(row["identity"]) for row in events}):
        sample = sample_by_identity[identity]
        fragment = ROOT / str(sample["source"]["path"])
        if _sha256(fragment) != str(sample["source"]["sha256"]):
            raise E6Error(f"fragment hash changed for {identity}")
        graph = scoreall._fragment_graph(fragment, identity)
        graph_by_identity[identity] = graph
        scale = np.asarray(sample["scale_zyx"], dtype=np.float64)
        scale_by_identity[identity] = scale
        pair_rows: dict[int, list[tuple[int, int, np.ndarray, np.ndarray]]] = defaultdict(list)
        for edge in graph.edges:
            source = graph.nodes[int(edge.source_id)]
            target = graph.nodes[int(edge.target_id)]
            if int(target.t) != int(source.t) + 1:
                continue
            source_xyz = np.asarray((source.z, source.y, source.x), dtype=np.float64) * scale
            target_xyz = np.asarray((target.z, target.y, target.x), dtype=np.float64) * scale
            pair_rows[int(source.t)].append(
                (int(edge.source_id), int(edge.target_id), source_xyz, target_xyz)
            )
        for frame, rows in pair_rows.items():
            fit_by_pair[(identity, frame)] = rows

    for event in events:
        identity = str(event["identity"])
        frame = int(event["source_frame"])
        source_id = int(event["pred_source_id"])
        target_id = int(event["pred_target_id"])
        graph = graph_by_identity[identity]
        scale = scale_by_identity[identity]
        source_node = graph.nodes[source_id]
        target_node = graph.nodes[target_id]
        if int(source_node.t) != frame or int(target_node.t) != frame + 1:
            raise E6Error(f"event time mismatch for {identity} t={frame}")
        source = np.asarray((source_node.z, source_node.y, source_node.x), dtype=np.float64) * scale
        target = np.asarray((target_node.z, target_node.y, target_node.x), dtype=np.float64) * scale
        direct = float(np.linalg.norm(target - source))
        if not math.isclose(direct, float(event["distance_um"]), abs_tol=1.0e-9):
            raise E6Error(f"event geometry changed for {identity} t={frame}")
        full = fit_by_pair[(identity, frame)]
        fit = [row for row in full if source_id not in row[:2] and target_id not in row[:2]]
        left = np.asarray([row[2] for row in fit], dtype=np.float64)
        right = np.asarray([row[3] for row in fit], dtype=np.float64)
        center, coefficient, rank, condition = _fit_affine(left, right)
        prediction = _predict_affine(source, center, coefficient)
        residual = float(np.linalg.norm(target - prediction))
        affine_rows.append(
            {
                "identity": identity,
                "source_frame": frame,
                "pred_source_id": source_id,
                "pred_target_id": target_id,
                "direct_distance_um": direct,
                "affine_loo_residual_um": residual,
                "below_10um_after_affine": residual <= GATE_UM,
                "fit_correspondences_full": len(full),
                "fit_correspondences_loo": len(fit),
                "fit_rank": rank,
                "fit_condition": condition,
            }
        )

    flow_rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    downsample = np.asarray(args.downsample, dtype=np.float64)
    if not args.skip_flow:
        with ZipStore(args.archive, mode="r") as store:
            for pair_index, ((identity, frame), pair_events) in enumerate(
                sorted(event_by_pair.items()), start=1
            ):
                _, dataset = identity.split(":", 1)
                group = zarr.open_group(
                    store=store, path=f"train/{dataset}.zarr", mode="r"
                )
                image = group["0"]
                scale = np.asarray(parse_scale(group.attrs.asdict()), dtype=np.float64)
                if not np.array_equal(scale, scale_by_identity[identity]):
                    raise E6Error(f"image/score scale mismatch for {identity}")
                reference = _prepare_volume(np.asarray(image[frame]), args.downsample)
                moving = _prepare_volume(np.asarray(image[frame + 1]), args.downsample)
                flow = optical_flow_tvl1(
                    reference,
                    moving,
                    attachment=args.attachment,
                    tightness=args.tightness,
                    num_warp=args.num_warp,
                    num_iter=args.num_iter,
                    prefilter=True,
                    dtype=np.float32,
                )
                graph = graph_by_identity[identity]
                for event in pair_events:
                    source_id = int(event["pred_source_id"])
                    target_id = int(event["pred_target_id"])
                    source_node = graph.nodes[source_id]
                    target_node = graph.nodes[target_id]
                    source_voxel = np.asarray(
                        (source_node.z, source_node.y, source_node.x), dtype=np.float64
                    )
                    target_voxel = np.asarray(
                        (target_node.z, target_node.y, target_node.x), dtype=np.float64
                    )
                    displacement_coarse = _flow_at(flow, source_voxel, downsample)
                    prediction_voxel = source_voxel + displacement_coarse * downsample
                    residual = float(np.linalg.norm((target_voxel - prediction_voxel) * scale))
                    flow_rows_by_key[(identity, source_id, target_id)] = {
                        "tvl1_residual_um": residual,
                        "below_10um_after_tvl1": residual <= GATE_UM,
                        "tvl1_displacement_coarse_zyx": displacement_coarse.tolist(),
                    }
                print(
                    f"flow {pair_index}/{len(event_by_pair)} {identity} t={frame}",
                    flush=True,
                )

    combined: list[dict[str, Any]] = []
    for row in affine_rows:
        key = (str(row["identity"]), int(row["pred_source_id"]), int(row["pred_target_id"]))
        combined.append({**row, **flow_rows_by_key.get(key, {})})

    affine_residuals = [float(row["affine_loo_residual_um"]) for row in combined]
    affine_recovered = sum(value <= GATE_UM for value in affine_residuals)
    affine_fraction = affine_recovered / len(combined)
    flow_residuals = [float(row["tvl1_residual_um"]) for row in combined if "tvl1_residual_um" in row]
    flow_recovered = sum(value <= GATE_UM for value in flow_residuals)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "summary": {
            **observed,
            "affine_below_10um_count": affine_recovered,
            "affine_below_10um_fraction": affine_fraction,
            "affine_implementation_gate": "PASS" if affine_fraction >= 0.5 else "FAIL",
            "motion_axis_bounded_rule": "PASS" if affine_fraction < 0.5 else "FAIL",
            "tvl1_below_10um_count": flow_recovered if flow_residuals else None,
            "tvl1_below_10um_fraction": (
                flow_recovered / len(flow_residuals) if flow_residuals else None
            ),
        },
        "decision": {
            "rule": "implement affine iff >=50% of the frozen 336-event cohort falls below 10um",
            "affine_status": "PASS" if affine_fraction >= 0.5 else "FAIL",
            "recommendation": (
                "implement affine before learned flow"
                if affine_fraction >= 0.5
                else "do not implement affine; coherent motion explains less than half of the tail"
            ),
        },
        "residuals": {
            "direct_um": _quantiles([float(row["direct_distance_um"]) for row in combined]),
            "affine_loo_um": _quantiles(affine_residuals),
            "tvl1_um": _quantiles(flow_residuals),
        },
        "fit": {
            "affine": "per-video/per-frame-pair 12-DOF least-squares affine over selected consecutive-frame edges; evaluated edge endpoints excluded",
            "tvl1": {
                "implementation": "skimage.registration.optical_flow_tvl1, native 3-D",
                "downsample_zyx": list(args.downsample),
                "physical_spacing_after_downsample_um": (
                    np.asarray((1.625, 0.40625, 0.40625)) * downsample
                ).tolist(),
                "attachment": args.attachment,
                "tightness": args.tightness,
                "num_warp": args.num_warp,
                "num_iter": args.num_iter,
                "skipped": args.skip_flow,
            },
        },
        "events": combined,
        "selection": {
            "cohort": "prefix_52 breaks classified as final_pair_geometry_above_gate_radius",
            "unique_frame_pairs": len(event_by_pair),
            "unique_videos": len({str(row["identity"]) for row in events}),
        },
        "provenance": {
            "fragmentation": {"path": str(args.fragmentation), "sha256": _sha256(args.fragmentation)},
            "scores": {"path": str(args.scores), "sha256": _sha256(args.scores)},
            "archive": {"path": str(args.archive), "bytes_rehashed": False},
            "script": {"path": str(Path(__file__).resolve())},
        },
        "execution": {"elapsed_seconds": time.perf_counter() - started},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--fragmentation", type=Path, default=DEFAULT_FRAGMENTATION)
    value.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--downsample", type=int, nargs=3, default=DEFAULT_DOWNSAMPLE)
    value.add_argument("--attachment", type=float, default=15.0)
    value.add_argument("--tightness", type=float, default=0.3)
    value.add_argument("--num-warp", type=int, default=3)
    value.add_argument("--num-iter", type=int, default=5)
    value.add_argument("--skip-flow", action="store_true")
    value.add_argument("--force", action="store_true")
    return value


if __name__ == "__main__":
    payload = run(parser().parse_args())
    print(json.dumps({"summary": payload["summary"], "decision": payload["decision"]}, indent=2))
