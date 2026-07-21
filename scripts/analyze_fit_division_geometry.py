#!/usr/bin/env python3
"""Measure inherited division-geometry coverage on fold-fit GT only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from zarr.storage import ZipStore


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "src", ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from celltrack_alpha.bh_graph import V45_GRAPH_CONFIG  # noqa: E402
from celltrack_baseline.rule_based import parse_scale, read_geff_arrays  # noqa: E402


class DivisionGeometryError(RuntimeError):
    """Fit-only division geometry inputs are incomplete or malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DivisionGeometryError(f"JSON root is not an object: {path}")
    return value


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            key: None
            for key in ("min", "p05", "p25", "p50", "p75", "p90", "p95", "max")
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def analyze_video(store: Any, dataset: str) -> dict[str, Any]:
    gt = read_geff_arrays(store, "train", dataset)
    group = zarr.open_group(store=store, path=f"train/{dataset}.zarr", mode="r")
    scale = np.asarray(parse_scale(dict(group.attrs)), dtype=np.float64)
    nodes = gt["nodes"]
    by_id = {
        int(node_id): (
            int(t),
            np.asarray((float(z), float(y), float(x)), dtype=np.float64),
        )
        for node_id, t, z, y, x in zip(
            nodes["node_id"], nodes["t"], nodes["z"], nodes["y"], nodes["x"], strict=True
        )
    }
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for source, target in gt["edges"]:
        outgoing.setdefault(int(source), []).append(int(target))
        incoming.setdefault(int(target), []).append(int(source))
    divisions: list[dict[str, Any]] = []
    for parent_id, children_raw in sorted(outgoing.items()):
        children = sorted(set(children_raw))
        if len(children) < 2:
            continue
        parent_t, parent_coord = by_id[parent_id]
        child_rows = [by_id[child] for child in children]
        child_distances = [
            float(np.linalg.norm((coord - parent_coord) * scale)) for _t, coord in child_rows
        ]
        # Official GT is binary here; retain an explicit failure if that changes.
        if len(children) != 2:
            raise DivisionGeometryError(f"GT divider {dataset}:{parent_id} has {len(children)} children")
        sister_distance = float(np.linalg.norm((child_rows[0][1] - child_rows[1][1]) * scale))
        symmetry_delta = abs(child_distances[0] - child_distances[1])
        predecessors = sorted(set(incoming.get(parent_id, ())))
        predecessor_time_delta: int | None = None
        midpoint_residual: float | None = None
        if len(predecessors) == 1:
            previous_t, previous_coord = by_id[predecessors[0]]
            predecessor_time_delta = int(parent_t - previous_t)
            predicted_coord = 2.0 * parent_coord - previous_coord
            midpoint_coord = 0.5 * (child_rows[0][1] + child_rows[1][1])
            midpoint_residual = float(np.linalg.norm((midpoint_coord - predicted_coord) * scale))
        deltas = [int(t - parent_t) for t, _coord in child_rows]
        if any(delta <= 0 for delta in deltas):
            raise DivisionGeometryError("GT division edge is not forward in time")
        current_envelope = bool(
            deltas == [1, 1]
            and sister_distance <= V45_GRAPH_CONFIG.division_sister_um + 1.0e-12
            and (
                (
                    child_distances[0] <= V45_GRAPH_CONFIG.division_existing_child_um + 1.0e-12
                    and child_distances[1] <= V45_GRAPH_CONFIG.division_candidate_child_um + 1.0e-12
                )
                or (
                    child_distances[1] <= V45_GRAPH_CONFIG.division_existing_child_um + 1.0e-12
                    and child_distances[0] <= V45_GRAPH_CONFIG.division_candidate_child_um + 1.0e-12
                )
            )
        )
        divisions.append(
            {
                "parent_id": parent_id,
                "parent_t": parent_t,
                "child_ids": children,
                "child_time_deltas": deltas,
                "parent_child_um": child_distances,
                "sister_um": sister_distance,
                "symmetry_delta_um": symmetry_delta,
                "predecessor_ids": predecessors,
                "predecessor_time_delta": predecessor_time_delta,
                "midpoint_residual_um": midpoint_residual,
                "current_geometry_envelope": current_envelope,
            }
        )
    return {
        "identity": f"train:{dataset}",
        "dataset": dataset,
        "scale_zyx": scale.tolist(),
        "division_count": len(divisions),
        "divisions": divisions,
    }


def analyze(zip_path: Path, split_path: Path, *, fold: int) -> dict[str, Any]:
    split = _load(split_path)
    fold_row = split.get("folds", {}).get(str(fold))
    fit_ids = fold_row.get("fit_video_ids") if isinstance(fold_row, Mapping) else None
    if not isinstance(fit_ids, list) or len(fit_ids) != len(set(fit_ids)) or not fit_ids:
        raise DivisionGeometryError("fold fit video IDs are invalid")
    samples: list[dict[str, Any]] = []
    with ZipStore(zip_path, mode="r") as store:
        for identity in sorted(str(value) for value in fit_ids):
            if not identity.startswith("train:"):
                raise DivisionGeometryError("fit selection contains a non-train identity")
            samples.append(analyze_video(store, identity.split(":", 1)[1]))
    divisions = [division for sample in samples for division in sample["divisions"]]
    parent_child = [distance for division in divisions for distance in division["parent_child_um"]]
    sister = [float(division["sister_um"]) for division in divisions]
    maximum_parent_child = [max(division["parent_child_um"]) for division in divisions]
    symmetry = [float(division["symmetry_delta_um"]) for division in divisions]
    midpoint = [
        float(division["midpoint_residual_um"])
        for division in divisions
        if division["midpoint_residual_um"] is not None
    ]
    predecessor_deltas = [
        float(division["predecessor_time_delta"])
        for division in divisions
        if division["predecessor_time_delta"] is not None
    ]
    divisions_by_video_time: list[int] = []
    for sample in samples:
        counts: dict[int, int] = {}
        for division in sample["divisions"]:
            parent_t = int(division["parent_t"])
            counts[parent_t] = counts.get(parent_t, 0) + 1
        divisions_by_video_time.extend(counts.values())
    deltas = [int(delta) for division in divisions for delta in division["child_time_deltas"]]
    dt1 = sum(division["child_time_deltas"] == [1, 1] for division in divisions)
    covered = sum(bool(division["current_geometry_envelope"]) for division in divisions)
    archive_sha = "538ed8a4d7b6669fa948865f947ed43c23aba7ee006fea8826d6ddd1277dfd19"
    return {
        "schema": "celltrack-alpha.fit-division-geometry.v2",
        "scope": {
            "fold": fold,
            "fit_video_count": len(fit_ids),
            "fit_video_ids": sorted(str(value) for value in fit_ids),
            "held_out_used": False,
            "audit_used": False,
            "test_used": False,
        },
        "current_geometry": {
            "existing_child_um": V45_GRAPH_CONFIG.division_existing_child_um,
            "candidate_child_um": V45_GRAPH_CONFIG.division_candidate_child_um,
            "sister_um": V45_GRAPH_CONFIG.division_sister_um,
            "requires_next_frame_children": True,
        },
        "summary": {
            "gt_divisions": len(divisions),
            "both_children_next_frame": dt1,
            "both_children_next_frame_fraction": dt1 / max(1, len(divisions)),
            "current_geometry_envelope_count": covered,
            "current_geometry_envelope_fraction": covered / max(1, len(divisions)),
            "parent_child_um": _quantiles(parent_child),
            "maximum_parent_child_um": _quantiles(maximum_parent_child),
            "sister_um": _quantiles(sister),
            "symmetry_delta_um": _quantiles(symmetry),
            "midpoint_residual_um": _quantiles(midpoint),
            "predecessor_time_delta": _quantiles(predecessor_deltas),
            "divisions_per_occupied_video_time": _quantiles(
                [float(value) for value in divisions_by_video_time]
            ),
            "divisions_with_single_predecessor": len(midpoint),
            "child_time_delta": _quantiles([float(value) for value in deltas]),
        },
        "inputs": {
            "zip": {
                "path": str(zip_path),
                "bytes": zip_path.stat().st_size,
                "sha256": archive_sha,
            },
            "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        },
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, default=ROOT / "data/kaggle/raw/biohub-cell-tracking-during-development.zip"
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "reports/g02/grouped_3fold_audit20_v1.json"
    )
    parser.add_argument("--fold", choices=(0, 1, 2), type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/g02/fit_division_geometry_fold0_v2.json"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise DivisionGeometryError(f"refusing to overwrite output: {args.output}")
    report = analyze(args.zip, args.split, fold=args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, allow_nan=False, sort_keys=True).encode("ascii") + b"\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
